"""Orchestration: iterate dataset rows → agent → judge → collect verdicts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import agent_client, config, db, dataset, judge, metrics, report, retrieval_client


def _parse_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(x) for x in raw.split(",") if x.strip()}


def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    if not project_root.exists():
        print(f"project root not found: {project_root}", file=sys.stderr)
        return 2

    rt = config.load(project_root)
    questions_path = args.dataset.resolve()
    all_questions = dataset.load(questions_path)
    questions = dataset.filter_questions(
        all_questions,
        category=args.category,
        difficulty=args.difficulty,
        ids=_parse_ids(args.ids),
    )
    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        print("no questions selected after filtering", file=sys.stderr)
        return 2

    print(
        f"eval: {len(questions)} question(s)  "
        f"server={rt.server_base_url}  agent={rt.agent_backend.name}/{rt.agent_model}  "
        f"judge={rt.judge_backend.name}/{rt.judge_model}  db={rt.db_path}"
    )
    if rt.judge_backend.name == rt.agent_backend.name and rt.judge_model == rt.agent_model:
        print(
            "  WARNING: judge backend/model is identical to the agent — results will "
            "be biased by self-preference. Pick a different backend in [eval]."
        )

    rows: list[report.Row] = []
    for i, q in enumerate(questions, start=1):
        row = report.Row(
            id=q.id,
            category=q.category,
            difficulty=q.difficulty,
            question=q.question,
            golden_urls=list(q.golden_urls),
            expected_behavior=q.expected_behavior,
        )
        print(
            f"[{i}/{len(questions)}] #{q.id} ({q.category}/{q.difficulty}/{q.expected_behavior}) ... ",
            end="",
            flush=True,
        )

        # 1. Fetch reference(s) from DB. Refuse-mode rows skip this — there's
        # nothing to fetch.
        refs: list[db.ReferenceDocument] = []
        if q.expected_behavior == "answer":
            missing: list[str] = []
            for url in q.golden_urls:
                ref = db.fetch_by_url(rt.db_path, url)
                if ref is None:
                    missing.append(url)
                else:
                    refs.append(ref)
            if missing:
                row.error = f"references not found in DB: {', '.join(missing)}"
                print(f"SKIP ({row.error})")
                rows.append(row)
                continue
            row.ref_found = True
            row.ref_titles = [r.title for r in refs]
            row.ref_content_len = sum(len(r.content) for r in refs)

        # 2. Probe the retrieval layer directly. Skip for refuse rows: there
        # are no golden URLs to score against, so Recall@k / MRR are
        # undefined. The agent will still run for those — we want to know if
        # it correctly refuses.
        if q.expected_behavior == "answer":
            try:
                hits = retrieval_client.query(
                    rt.server_base_url, q.question, top_k=args.retrieval_top_k
                )
                retrieved_urls = [h.document_url for h in hits if h.document_url]
                row.retrieved_urls = retrieved_urls
                row.recall_at_5 = metrics.recall_at_k(retrieved_urls, q.golden_urls, k=5)
                row.recall_at_10 = metrics.recall_at_k(retrieved_urls, q.golden_urls, k=10)
                row.mrr = metrics.mrr(retrieved_urls, q.golden_urls)
            except Exception as e:  # noqa: BLE001
                # A retrieval failure shouldn't kill the run — we still want
                # the judge score for this row. Surface it in the error
                # field but keep going.
                row.error = f"retrieval probe failed: {e}"
                print(f"WARN retrieval ({e}) ", end="", flush=True)

        # 3. Ask the agent.
        try:
            reply = agent_client.ask(rt.server_base_url, rt.agent_model, q.question)
            row.agent_answer = reply.content
        except Exception as e:  # noqa: BLE001 — show the user what happened
            row.error = f"agent call failed: {e}"
            print(f"ERROR agent ({e})")
            rows.append(row)
            continue

        # 4. Score with the judge.
        try:
            verdict = judge.grade(
                rt.judge_backend,
                rt.judge_model,
                q.question,
                row.agent_answer,
                refs,
                expected_behavior=q.expected_behavior,
            )
            row.faithfulness = verdict.faithfulness.score
            row.correctness = verdict.correctness.score
            row.completeness = verdict.completeness.score
            row.overall = verdict.overall
            row.faithfulness_rationale = verdict.faithfulness.rationale
            row.correctness_rationale = verdict.correctness.rationale
            row.completeness_rationale = verdict.completeness.rationale
            print(
                f"F={verdict.faithfulness.score} "
                f"C={verdict.correctness.score} "
                f"Cm={verdict.completeness.score} "
                f"(O={verdict.overall})"
            )
        except Exception as e:  # noqa: BLE001
            row.error = f"judge failed: {e}"
            print(f"ERROR judge ({e})")

        rows.append(row)

    agg = report.aggregate(rows)
    report.print_summary(rows, agg)

    out_path = args.out
    if out_path is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        out_path = results_dir / f"{stamp}.json"

    out = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "server_base_url": rt.server_base_url,
        "agent_backend": rt.agent_backend.name,
        "agent_model": rt.agent_model,
        "judge_backend": rt.judge_backend.name,
        "judge_model": rt.judge_model,
        "aggregate": agg,
        "rows": [r.to_dict() for r in rows],
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="etsi-eval",
        description="Run the agent against the question dataset and grade answers with an LLM judge.",
    )
    here = Path(__file__).resolve().parent.parent
    p.add_argument(
        "--project-root",
        type=Path,
        default=here.parent,
        help="path to the etsi-rag project root (contains config/ and data/)",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=here / "dataset.jsonl",
        help="path to the JSONL question dataset",
    )
    p.add_argument("--category", help="filter by category (course, degree, procedure, general, cross-doc)")
    p.add_argument("--difficulty", help="filter by difficulty (easy, medium, hard)")
    p.add_argument("--ids", help="comma-separated question ids")
    p.add_argument("--limit", type=int, help="stop after N questions")
    p.add_argument(
        "--retrieval-top-k",
        type=int,
        default=10,
        help="top_k passed to /v1/rag/query when probing retrieval (default: 10)",
    )
    p.add_argument("--out", type=Path, help="write results JSON to this path (default: eval/results/TIMESTAMP.json)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
