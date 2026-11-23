"""Audit the question dataset against a recrawled corpus.

For every answer-row, resolves each golden URL in the old and new corpus,
proposes the next academic year's URL when a year-stamped course guide moved
(sara.uma.es/ht/2025/... -> /ht/2026/...), and checks whether each key fact
that appeared in the old reference text still appears in the new one.

Verdicts:
  ok       - every URL resolves unchanged-or-changed and no key fact was lost
  remap    - a URL only resolves under the next year's path, no fact lost;
             the proposed dataset applies the new URL
  review   - a URL is missing, or a key fact present before is gone
  refuse   - refuse-row, listed for a manual skim (cannot be checked here)

Writes <out>/audit.json, <out>/audit.md and <out>/dataset.proposed.jsonl.
Usage:
  uv run python -m scripts.audit_dataset OLD_DB NEW_DB OUT_DIR [--dataset dataset.jsonl]
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path

from etsi_eval import dataset, db

YEAR_PATH = re.compile(r"/ht/(20\d\d)/")


def _fold(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace: how a reader matches a fact."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def _next_year(url: str) -> str | None:
    m = YEAR_PATH.search(url)
    if not m:
        return None
    return url[: m.start(1)] + str(int(m.group(1)) + 1) + url[m.end(1) :]


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return round(difflib.SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False).ratio(), 3)


def audit_row(q: dataset.Question, facts: list[str], old_db: Path, new_db: Path) -> dict:
    row = {"id": q.id, "category": q.category, "question": q.question,
           "expected_behavior": q.expected_behavior, "urls": [], "lost_facts": [], "verdict": "ok"}
    if q.expected_behavior != "answer":
        row["verdict"] = "refuse"
        return row
    old_text, new_text = [], []
    for url in q.golden_urls:
        old = db.fetch_by_url(old_db, url)
        new = db.fetch_by_url(new_db, url)
        entry = {"url": url, "in_old": old is not None, "in_new": new is not None, "proposed_url": None}
        if new is None and (nxt := _next_year(url)) and (moved := db.fetch_by_url(new_db, nxt)):
            new, entry["proposed_url"] = moved, nxt
        entry["similarity"] = _similarity(old.content, new.content) if old and new else None
        row["urls"].append(entry)
        old_text.append(old.content if old else "")
        new_text.append(new.content if new else "")
    old_folded, new_folded = _fold("\n".join(old_text)), _fold("\n".join(new_text))
    row["lost_facts"] = [f for f in facts if _fold(f) in old_folded and _fold(f) not in new_folded]
    if any(not u["in_new"] and not u["proposed_url"] for u in row["urls"]) or row["lost_facts"]:
        row["verdict"] = "review"
    elif any(u["proposed_url"] for u in row["urls"]):
        row["verdict"] = "remap"
    return row


def main() -> int:
    p = argparse.ArgumentParser(prog="audit_dataset")
    p.add_argument("old_db", type=Path)
    p.add_argument("new_db", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent.parent / "dataset.jsonl")
    args = p.parse_args()

    raw = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    facts = {r["id"]: list(r.get("key_facts") or []) for r in raw}
    rows = [audit_row(q, facts[q.id], args.old_db, args.new_db) for q in dataset.load(args.dataset)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audit.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    remap = {r["id"]: {u["url"]: u["proposed_url"] for u in r["urls"] if u["proposed_url"]}
             for r in rows if r["verdict"] in ("remap", "review")}
    with (args.out_dir / "dataset.proposed.jsonl").open("w", encoding="utf-8") as out:
        for r in raw:
            moves = remap.get(r["id"], {})
            r = dict(r, golden_urls=[moves.get(u, u) for u in r["golden_urls"]])
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    lines = ["# Dataset audit", "", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())), "",
             "| id | verdict | question | URLs (old/new, similarity, proposed) | lost key facts |",
             "|---:|---|---|---|---|"]
    for r in rows:
        if r["verdict"] == "ok":
            continue
        urls = "<br>".join(
            f"{'✓' if u['in_old'] else '✗'}/{'✓' if u['in_new'] else '✗'} {u['similarity']} "
            f"{('→ ' + u['proposed_url']) if u['proposed_url'] else ''}" for u in r["urls"]) or "-"
        lines.append(f"| {r['id']} | {r['verdict']} | {r['question'][:90]} | {urls} | {'; '.join(r['lost_facts']) or '-'} |")
    (args.out_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
