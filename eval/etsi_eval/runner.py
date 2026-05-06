"""Orchestration: iterate dataset rows → agent → record evidence.

This runner captures only objective evidence per row: the agent's answer,
the tool calls it made, and retrieval-stage metrics. Per-row LLM judging
was retired 2026-05-06 in favour of the pairwise runner — see
`docs/eval-methodology.md` and `pairwise_runner.py`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import agent_client, config, db, dataset, metrics, report, retrieval_client
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
    """Run the full per-row pipeline. Returns (row, status_line)."""
    row = report.Row(
        id=q.id,
        category=q.category,
        difficulty=q.difficulty,
        question=q.question,
        golden_urls=list(q.golden_urls),
        expected_behavior=q.expected_behavior,
    )

    # 1. Fetch reference(s) from DB. Only used as a sanity-check that the
    # eval is pointed at the indexed corpus we expect; not consumed by any
    # judge here. Refuse-mode rows skip.
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

    # 3. Ask the agent. The chat-completions response carries tool_calls as
    # a custom root-level extension; we record them on the row to attribute
    # downstream wins to specific tools (e.g. read_document).
    try:
        reply = agent_client.ask(
            rt.server_base_url,
            rt.agent_model,
            q.question,
            session_id=f"eval-{run_id}-{q.id}",
        )
        row.agent_answer = reply.content
        row.tool_calls = list(reply.tool_calls)
    except Exception as e:  # noqa: BLE001
        row.errors.append(f"agent: {e}")
        return row, f"#{q.id} ERROR agent ({e}){retrieval_warn}"

    tool_summary = ""
    if reply.tool_calls:
        names = [str(c.get("name", "?")) for c in reply.tool_calls]
        tool_summary = f" tools=[{','.join(names)}]"
    return row, f"#{q.id} OK{tool_summary}{retrieval_warn}"


def _run_serial(
    rt: config.Runtime,
    questions: list[dataset.Question],
    run_id: str,
    retrieval_top_k: int,
) -> list[report.Row]:
    rows: list[report.Row] = []
    total = len(questions)
    for i, q in enumerate(questions, start=1):
        prefix = f"[{i}/{total}] #{q.id} ({q.category}/{q.difficulty}/{q.expected_behavior}) ... "
        print(prefix, end="", flush=True)
        row, status = _process_question(rt, q, run_id, retrieval_top_k)
        tail = status.split(" ", 1)[1] if status.startswith(f"#{q.id} ") else status
        print(tail)
        rows.append(row)
    return rows


def _run_parallel(
    rt: config.Runtime,
    questions: list[dataset.Question],
    run_id: str,
    retrieval_top_k: int,
    concurrency: int,
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
            with print_lock:
                completed["n"] += 1
                print(
                    f"[{completed['n']}/{total}] {q.category}/{q.difficulty}/{q.expected_behavior} {status}"
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
        f"db={rt.db_path}"
    )
    if args.concurrency > 1:
        print(
            f"  NOTE: concurrency>1 only helps if the agent backend has enough slots. "
            f"For local llama-server, start it with --parallel >= {args.concurrency}."
        )

    # Per-run identifier: every row gets a fresh session id derived from this,
    # so chat history never bleeds between questions even if the server is in
    # session mode.
    run_id = uuid.uuid4().hex[:8]

    if args.concurrency == 1:
        rows = _run_serial(rt, questions, run_id, args.retrieval_top_k)
    else:
        rows = _run_parallel(
            rt, questions, run_id, args.retrieval_top_k, args.concurrency
        )

    agg = report.aggregate(rows)
    report.print_summary(rows, agg)

    out_path = args.out
    if out_path is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        out_path = results_dir / f"{stamp}.json"

    out = {
        "schema_version": 3,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "concurrency": args.concurrency,
        "server_base_url": rt.server_base_url,
        "agent_backend": rt.agent_backend.name,
        "agent_model": rt.agent_model,
        "aggregate": agg,
        "rows": [r.to_dict() for r in rows],
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="etsi-eval",
        description=(
            "Run the agent against the question dataset and capture per-row "
            "evidence (answer, tool calls, retrieval metrics). "
            "Pairwise comparison is via etsi_eval.pairwise_runner."
        ),
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
            "For local llama-server, requires --parallel N+ on the server side."
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
