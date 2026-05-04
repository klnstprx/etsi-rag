"""D1 diagnostic: re-probe retrieval at higher top_k for the misses.

Reads a results JSON, extracts answer-rows where Recall@5 == 0, and
re-probes /v1/rag/query at top_k=30. For each miss, reports the rank at
which the first golden URL surfaces (or MISS if it never does).

This answers: is the bug RANKING (gold is findable, just not at the top)
or COVERAGE (gold isn't being found at all)?

Usage:
  uv run python -m scripts.d1_topk_probe
  uv run python -m scripts.d1_topk_probe path/to/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

from etsi_eval import config
from etsi_eval.url_norm import canonicalize


def _latest_results(results_dir: Path) -> Path:
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no results JSON found in {results_dir}")
    return files[-1]


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    rt = config.load(project_root)

    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1]).resolve()
    else:
        results_path = _latest_results(Path(__file__).resolve().parent.parent / "results")
    print(f"reading {results_path}")
    with results_path.open() as f:
        data = json.load(f)

    misses = [
        r for r in data["rows"]
        if r.get("expected_behavior") == "answer" and (r.get("recall_at_5") or 0) == 0
    ]
    print(f"re-probing {len(misses)} miss row(s) at top_k=30 against {rt.server_base_url}")
    print()

    # Counters: how many of the misses have ANY gold found at each cutoff.
    cuts = (5, 10, 20, 30)
    found_any: dict[int, int] = {k: 0 for k in cuts}
    # For multi-golden rows: how many of the N golds appear at each cutoff.
    multi_golden_any: dict[int, int] = {k: 0 for k in cuts}
    multi_golden_all: dict[int, int] = {k: 0 for k in cuts}
    n_single = 0
    n_multi = 0

    print(f"{'id':>4} {'cat':<11}{'diff':<8}{'#g':>3} {'rank(s)':<22}  question")
    print("-" * 110)
    with httpx.Client(timeout=30.0) as client:
        for row in misses:
            qid = row["id"]
            question = row["question"]
            gold = {canonicalize(u) for u in row.get("golden_urls", [])}
            n_gold = len(gold)
            if n_gold == 1:
                n_single += 1
            else:
                n_multi += 1

            r = client.post(
                f"{rt.server_base_url}/v1/rag/query",
                json={"query": question, "top_k": 30},
            )
            r.raise_for_status()
            hits = r.json().get("results", [])
            urls = [canonicalize(h.get("document_url") or "") for h in hits]

            # Per-cutoff: which golds appear?
            ranks_per_gold: list[int | None] = []
            for g in gold:
                rank = None
                for i, u in enumerate(urls, start=1):
                    if u == g:
                        rank = i
                        break
                ranks_per_gold.append(rank)

            for k in cuts:
                hit_count = sum(1 for rk in ranks_per_gold if rk is not None and rk <= k)
                if hit_count > 0:
                    found_any[k] += 1
                    if n_gold > 1:
                        multi_golden_any[k] += 1
                if n_gold > 1 and hit_count == n_gold:
                    multi_golden_all[k] += 1

            # Format ranks for display
            if all(rk is None for rk in ranks_per_gold):
                rank_repr = "MISS"
            else:
                parts = [str(rk) if rk is not None else "-" for rk in ranks_per_gold]
                rank_repr = "@" + ",".join(parts)
            print(f"{qid:>4} {row['category']:<11}{row['difficulty']:<8}{n_gold:>3} {rank_repr:<22}  {question[:60]}")

    n = len(misses)
    print()
    print(f"Of the {n} rows that missed at top_k=5:")
    for k in cuts:
        print(f"  found ≥1 gold in top_k={k:<3} : {found_any[k]:>3}/{n}  ({100*found_any[k]/n:.0f}%)")
    if n_multi > 0:
        print()
        print(f"Multi-golden subset ({n_multi} rows):")
        for k in cuts:
            print(
                f"  any gold in top_k={k:<3}: {multi_golden_any[k]:>2}/{n_multi}   "
                f"all golds in top_k={k:<3}: {multi_golden_all[k]:>2}/{n_multi}"
            )

    print()
    print("Interpretation:")
    if found_any[30] >= int(0.85 * n):
        print(f"  -> RANKING bug. {found_any[30]}/{n} misses surface in top-30; gold is in")
        print("     the corpus, the retriever just doesn't promote it. Hybrid retrieval")
        print("     (BM25+dense) or a reranker should close most of the gap.")
    elif found_any[30] >= int(0.5 * n):
        print(f"  -> MIXED. {found_any[30]}/{n} surface eventually — about half ranking,")
        print("     half coverage. Look at the MISS rows to find chunking gaps.")
    else:
        print(f"  -> COVERAGE bug. Only {found_any[30]}/{n} surface even at top-30.")
        print("     The gold chunks aren't being found at all — chunking strategy is")
        print("     producing chunks that don't match the question semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
