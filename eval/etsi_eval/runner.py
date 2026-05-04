"""Orchestration: iterate dataset rows → agent → judge → collect verdicts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import agent_client, config, db, dataset, judge, metrics, report, retrieval_client
from .url_norm import canonicalize


def _parse_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(x) for x in raw.split(",") if x.strip()}


def _process_question(
    rt: config.Runtime,
    q: dataset.Question,
    run_id: str,
    retrieval_top_k: int,
) -> tuple[report.Row, str]:
    """Run the full per-row pipeline. Returns (row, status_line).

    Pure-ish: only mutates the local `row` it builds and the per-process DB
    canonical-index cache (which is concurrency-safe — dict reads/writes for
    distinct keys are atomic under the GIL, and the only writer is during
    initial population which is idempotent).
    """
    row = report.Row(
        id=q.id,
        category=q.category,
        difficulty=q.difficulty,
        question=q.question,
        golden_urls=list(q.golden_urls),
        expected_behavior=q.expected_behavior,
    )

    # 1. Fetch reference(s) from DB. Refuse-mode rows skip this.
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
            err = f"references not found in DB: {', '.join(missing)}"
            row.errors.append(f"db: {err}")
            return row, f"#{q.id} SKIP ({err})"
        row.ref_found = True
        row.ref_titles = [r.title for r in refs]
        row.ref_content_len = sum(len(r.content) for r in refs)

    # 2. Probe the retrieval layer. Skip for refuse rows: no goldens to score.
    retrieval_warn = ""
    if q.expected_behavior == "answer":
        try:
            hits = retrieval_client.query(
                rt.server_base_url, q.question, top_k=retrieval_top_k
            )
            retrieved_urls = [
                canonicalize(h.document_url) for h in hits if h.document_url
            ]
            row.retrieved_urls = retrieved_urls
            row.recall_at_5 = metrics.recall_at_k(retrieved_urls, q.golden_urls, k=5)
            row.recall_at_10 = metrics.recall_at_k(retrieved_urls, q.golden_urls, k=10)
            row.mrr = metrics.mrr(retrieved_urls, q.golden_urls)
        except Exception as e:  # noqa: BLE001
            row.errors.append(f"retrieval: {e}")
            retrieval_warn = f" WARN retrieval({e})"

    # 3. Ask the agent.
    try:
        reply = agent_client.ask(
            rt.server_base_url,
            rt.agent_model,
            q.question,
            session_id=f"eval-{run_id}-{q.id}",
        )
        row.agent_answer = reply.content
    except Exception as e:  # noqa: BLE001
        row.errors.append(f"agent: {e}")
        return row, f"#{q.id} ERROR agent ({e}){retrieval_warn}"

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
        row.judge_prompt_tokens = verdict.usage.prompt_tokens
        row.judge_completion_tokens = verdict.usage.completion_tokens
        row.judge_total_tokens = verdict.usage.total_tokens
        usage_part = (
            f" toks={verdict.usage.prompt_tokens}+{verdict.usage.completion_tokens}"
            if verdict.usage.total_tokens
            else ""
        )
        return (
            row,
            f"#{q.id} F={verdict.faithfulness.score} "
            f"C={verdict.correctness.score} "
            f"Cm={verdict.completeness.score} "
            f"(O={verdict.overall}){usage_part}{retrieval_warn}",
        )
    except Exception as e:  # noqa: BLE001
        row.errors.append(f"judge: {e}")
        return row, f"#{q.id} ERROR judge ({e}){retrieval_warn}"


class _UsageTracker:
    """Thread-safe running total of judge tokens + optional cost."""

    def __init__(self, pricing: config.JudgePricing):
        self._pricing = pricing
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, row: report.Row) -> str:
        """Add one row's tokens and return a formatted suffix for the log line."""
        with self._lock:
            self.prompt_tokens += row.judge_prompt_tokens
            self.completion_tokens += row.judge_completion_tokens
            cum_total = self.prompt_tokens + self.completion_tokens
            cum_prompt = self.prompt_tokens
            cum_completion = self.completion_tokens
        if cum_total == 0:
            return ""
        suffix = f" cum={cum_total:,}"
        if self._pricing.enabled:
            cost = self._pricing.cost_usd(cum_prompt, cum_completion)
            suffix += f" cost=${cost:.4f}"
        return suffix


def _run_serial(
    rt: config.Runtime,
    questions: list[dataset.Question],
    run_id: str,
    retrieval_top_k: int,
    tracker: _UsageTracker,
) -> list[report.Row]:
    rows: list[report.Row] = []
    total = len(questions)
    for i, q in enumerate(questions, start=1):
        prefix = f"[{i}/{total}] #{q.id} ({q.category}/{q.difficulty}/{q.expected_behavior}) ... "
        print(prefix, end="", flush=True)
        row, status = _process_question(rt, q, run_id, retrieval_top_k)
        # status repeats the id for the parallel path; in serial we already
        # printed the prefix, so strip the redundant "#id " head.
        tail = status.split(" ", 1)[1] if status.startswith(f"#{q.id} ") else status
        print(tail + tracker.add(row))
        rows.append(row)
    return rows


def _run_parallel(
    rt: config.Runtime,
    questions: list[dataset.Question],
    run_id: str,
    retrieval_top_k: int,
    concurrency: int,
    tracker: _UsageTracker,
) -> list[report.Row]:
    rows: list[report.Row] = []
    total = len(questions)
    print_lock = threading.Lock()
    completed = {"n": 0}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_process_question, rt, q, run_id, retrieval_top_k): q
            for q in questions
        }
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                row, status = fut.result()
            except Exception as e:  # noqa: BLE001 — defensive; _process_question normally swallows
                row = report.Row(
                    id=q.id,
                    category=q.category,
                    difficulty=q.difficulty,
                    question=q.question,
                    golden_urls=list(q.golden_urls),
                    expected_behavior=q.expected_behavior,
                )
                row.errors.append(f"runner: {e}")
                status = f"#{q.id} ERROR runner ({e})"
            cum = tracker.add(row)
            with print_lock:
                completed["n"] += 1
                print(
                    f"[{completed['n']}/{total}] {q.category}/{q.difficulty}/{q.expected_behavior} {status}{cum}"
                )
            rows.append(row)

    rows.sort(key=lambda r: r.id)
    return rows


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

    if args.concurrency < 1:
        print(
            f"--concurrency must be >= 1, got {args.concurrency}", file=sys.stderr
        )
        return 2

    if args.validate_urls:
        missing: list[tuple[int, str]] = []
        for q in questions:
            if q.expected_behavior != "answer":
                continue
            for url in q.golden_urls:
                if db.fetch_by_url(rt.db_path, url) is None:
                    missing.append((q.id, url))
        if missing:
            print(
                f"validate-urls: {len(missing)} golden URL(s) not found in {rt.db_path}:",
                file=sys.stderr,
            )
            for qid, url in missing:
                print(f"  #{qid}  {url}", file=sys.stderr)
            return 3
        print(f"validate-urls: all golden URLs resolved against {rt.db_path}")

    print(
        f"eval: {len(questions)} question(s)  concurrency={args.concurrency}  "
        f"server={rt.server_base_url}  agent={rt.agent_backend.name}/{rt.agent_model}  "
        f"judge={rt.judge_backend.name}/{rt.judge_model}  db={rt.db_path}"
    )
    if args.concurrency > 1:
        print(
            f"  NOTE: concurrency>1 only helps if the agent backend has enough slots. "
            f"For local llama-server, start it with --parallel >= {args.concurrency}."
        )
    if rt.judge_backend.name == rt.agent_backend.name and rt.judge_model == rt.agent_model:
        print(
            "  WARNING: judge backend/model is identical to the agent — results will "
            "be biased by self-preference. Pick a different backend in [eval]."
        )

    # Per-run identifier: every row gets a fresh session id derived from this,
    # so chat history never bleeds between questions even if the server is in
    # session mode.
    run_id = uuid.uuid4().hex[:8]

    tracker = _UsageTracker(rt.judge_pricing)
    if rt.judge_pricing.enabled:
        print(
            f"  pricing: ${rt.judge_pricing.input_per_mtok:.2f}/MTok in, "
            f"${rt.judge_pricing.output_per_mtok:.2f}/MTok out — running cost shown per row"
        )

    if args.concurrency == 1:
        rows = _run_serial(rt, questions, run_id, args.retrieval_top_k, tracker)
    else:
        rows = _run_parallel(
            rt, questions, run_id, args.retrieval_top_k, args.concurrency, tracker
        )

    agg = report.aggregate(rows)
    if rt.judge_pricing.enabled:
        agg["judge_usage"]["estimated_cost_usd"] = round(
            rt.judge_pricing.cost_usd(
                agg["judge_usage"]["prompt_tokens"],
                agg["judge_usage"]["completion_tokens"],
            ),
            4,
        )
    report.print_summary(rows, agg)

    out_path = args.out
    if out_path is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        out_path = results_dir / f"{stamp}.json"

    out = {
        "schema_version": 2,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "concurrency": args.concurrency,
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
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "number of rows to process in parallel (default: 1, serial). "
            "For local llama-server, requires --parallel N+ on the server side; "
            "the remote judge handles concurrency natively."
        ),
    )
    p.add_argument("--out", type=Path, help="write results JSON to this path (default: eval/results/TIMESTAMP.json)")
    p.add_argument(
        "--validate-urls",
        action="store_true",
        help="before running, fail fast if any golden_urls entry isn't in the documents table",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
