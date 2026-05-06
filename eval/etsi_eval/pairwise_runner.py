"""Pairwise preference runner: compare two saved eval runs head-to-head.

Workflow:
  1. Run ``etsi_eval.runner`` against config X → ``results/X.json``.
  2. Change config (or code), run again → ``results/Y.json``.
  3. ``python -m etsi_eval.pairwise_runner --a results/X.json --b results/Y.json``
     pairs rows by id, asks the judge which answer better addresses each
     question, and reports a win-rate with a binomial p-value.

Each pair is judged TWICE with answer order swapped (see ``pairwise.py``).
A verdict only counts when both passes agree on the same underlying answer;
positional flips are tagged ``unstable`` and excluded from the win-rate.

The orchestration here is split into pure helpers (``load_run``,
``pair_rows``, ``aggregate_pairs``, ``binomial_p_two_sided``) plus the
network-driving ``run`` entry point. The pure helpers are unit-tested.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from math import comb
from pathlib import Path
from typing import Any

from . import config, db, pairwise


# ---- data types ----------------------------------------------------------


@dataclass(frozen=True)
class RunRow:
    id: int
    category: str
    difficulty: str
    question: str
    expected_behavior: str
    golden_urls: list[str]
    agent_answer: str
    has_error: bool


@dataclass(frozen=True)
class Run:
    """One saved result JSON, indexed by row id for fast pairing."""

    path: Path
    label: str
    rows_by_id: dict[int, RunRow]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PairedRow:
    id: int
    category: str
    difficulty: str
    question: str
    expected_behavior: str
    golden_urls: list[str]
    answer_a: str
    answer_b: str


@dataclass
class PairResult:
    """One pair's verdict, position-frame stripped (winner is in the
    original frame: A means run-a's answer won)."""

    id: int
    category: str
    difficulty: str
    expected_behavior: str
    winner: str  # "A" | "B" | "tie" | "unstable" | "" when errored
    forward_winner: str
    swapped_winner: str
    forward_rationale: str
    swapped_rationale: str
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_total_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- pure helpers --------------------------------------------------------


def load_run(path: Path, *, label: str | None = None) -> Run:
    """Parse a results JSON written by ``etsi_eval.runner``."""
    body = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[int, RunRow] = {}
    for raw in body.get("rows", []):
        rid = raw["id"]
        rows[rid] = RunRow(
            id=rid,
            category=raw.get("category", ""),
            difficulty=raw.get("difficulty", ""),
            question=raw.get("question", ""),
            expected_behavior=raw.get("expected_behavior", "answer"),
            golden_urls=list(raw.get("golden_urls") or []),
            agent_answer=raw.get("agent_answer", "") or "",
            has_error=bool(raw.get("errors")),
        )
    metadata = {k: v for k, v in body.items() if k != "rows"}
    return Run(
        path=path,
        label=label if label is not None else path.stem,
        rows_by_id=rows,
        metadata=metadata,
    )


def pair_rows(a: Run, b: Run) -> tuple[list[PairedRow], list[tuple[int, str]]]:
    """Pair rows by id. Returns (paired, skipped) where skipped is a list of
    (id, human-readable reason) tuples."""
    paired: list[PairedRow] = []
    skipped: list[tuple[int, str]] = []

    a_ids = set(a.rows_by_id)
    b_ids = set(b.rows_by_id)
    for rid in sorted(b_ids - a_ids):
        skipped.append((rid, "missing in A"))
    for rid in sorted(a_ids - b_ids):
        skipped.append((rid, "missing in B"))

    for rid in sorted(a_ids & b_ids):
        ra = a.rows_by_id[rid]
        rb = b.rows_by_id[rid]
        if ra.has_error or rb.has_error:
            skipped.append((rid, "errored on one side"))
            continue
        if not ra.agent_answer.strip() or not rb.agent_answer.strip():
            skipped.append((rid, "agent answer empty on one side"))
            continue
        if ra.expected_behavior != rb.expected_behavior:
            skipped.append((rid, "expected_behavior differs"))
            continue
        if list(ra.golden_urls) != list(rb.golden_urls):
            skipped.append((rid, "golden_urls differ"))
            continue
        paired.append(
            PairedRow(
                id=rid,
                category=ra.category,
                difficulty=ra.difficulty,
                question=ra.question,
                expected_behavior=ra.expected_behavior,
                golden_urls=list(ra.golden_urls),
                answer_a=ra.agent_answer,
                answer_b=rb.agent_answer,
            )
        )
    return paired, skipped


def binomial_p_two_sided(wins: int, total: int, p: float = 0.5) -> float:
    """Two-sided binomial test against H0=p.

    Returns the probability of observing |wins - total*p| or more extreme
    deviation from the expected count under H0. Pure stdlib so the eval
    harness stays scipy-free.
    """
    if total == 0:
        return 1.0
    expected = total * p
    deviation = abs(wins - expected)
    p_total = 0.0
    for k in range(total + 1):
        if abs(k - expected) >= deviation:
            p_total += comb(total, k) * (p ** k) * ((1 - p) ** (total - k))
    return min(1.0, p_total)


_BUCKETS = ("wins_a", "wins_b", "ties", "unstable")


def _bucket_for(winner: str) -> str | None:
    if winner == "A":
        return "wins_a"
    if winner == "B":
        return "wins_b"
    if winner == "tie":
        return "ties"
    if winner == "unstable":
        return "unstable"
    return None


def aggregate_pairs(results: list[PairResult]) -> dict[str, Any]:
    scored = [r for r in results if not r.errors]
    errored = [r for r in results if r.errors]

    counts = dict.fromkeys(_BUCKETS, 0)
    for r in scored:
        bucket = _bucket_for(r.winner)
        if bucket is not None:
            counts[bucket] += 1

    decisive = counts["wins_a"] + counts["wins_b"]
    win_rate_a: float | None = None
    win_rate_b: float | None = None
    binom_p: float | None = None
    if decisive > 0:
        win_rate_a = round(counts["wins_a"] / decisive, 3)
        win_rate_b = round(counts["wins_b"] / decisive, 3)
        binom_p = round(binomial_p_two_sided(counts["wins_a"], decisive), 4)

    by_category = _breakdown(scored, "category")
    by_difficulty = _breakdown(scored, "difficulty")

    judge_usage = {
        "prompt_tokens": sum(r.judge_prompt_tokens for r in results),
        "completion_tokens": sum(r.judge_completion_tokens for r in results),
        "total_tokens": sum(r.judge_total_tokens for r in results),
        "rows_with_usage": sum(1 for r in results if r.judge_total_tokens > 0),
    }

    return {
        "total_pairs": len(results),
        "scored": len(scored),
        "errored": len(errored),
        **counts,
        "win_rate_a": win_rate_a,
        "win_rate_b": win_rate_b,
        "binomial_p_two_sided": binom_p,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "judge_usage": judge_usage,
    }


def _breakdown(scored: list[PairResult], key: str) -> dict[str, dict[str, int]]:
    groups: dict[str, dict[str, int]] = {}
    for r in scored:
        bucket = _bucket_for(r.winner)
        if bucket is None:
            continue
        k = getattr(r, key)
        groups.setdefault(k, dict.fromkeys(_BUCKETS, 0))[bucket] += 1
    return dict(sorted(groups.items()))


# ---- network-driving orchestration --------------------------------------


def _process_pair(
    rt: config.Runtime,
    pair: PairedRow,
) -> PairResult:
    refs: list[db.ReferenceDocument] = []
    if pair.expected_behavior == "answer":
        for url in pair.golden_urls:
            ref = db.fetch_by_url(rt.db_path, url)
            if ref is not None:
                refs.append(ref)
        if not refs:
            return PairResult(
                id=pair.id,
                category=pair.category,
                difficulty=pair.difficulty,
                expected_behavior=pair.expected_behavior,
                winner="",
                forward_winner="",
                swapped_winner="",
                forward_rationale="",
                swapped_rationale="",
                errors=[f"db: no references resolved for {pair.golden_urls}"],
            )

    try:
        verdict = pairwise.grade_pair(
            rt.judge_backend,
            rt.judge_model,
            pair.question,
            pair.answer_a,
            pair.answer_b,
            refs,
            expected_behavior=pair.expected_behavior,
        )
    except pairwise.JudgeError as e:
        return PairResult(
            id=pair.id,
            category=pair.category,
            difficulty=pair.difficulty,
            expected_behavior=pair.expected_behavior,
            winner="",
            forward_winner="",
            swapped_winner="",
            forward_rationale="",
            swapped_rationale="",
            errors=[f"judge: {e}"],
        )

    return PairResult(
        id=pair.id,
        category=pair.category,
        difficulty=pair.difficulty,
        expected_behavior=pair.expected_behavior,
        winner=verdict.winner,
        forward_winner=verdict.forward.winner,
        swapped_winner=verdict.swapped.winner,
        forward_rationale=verdict.forward.rationale,
        swapped_rationale=verdict.swapped.rationale,
        judge_prompt_tokens=verdict.usage.prompt_tokens,
        judge_completion_tokens=verdict.usage.completion_tokens,
        judge_total_tokens=verdict.usage.total_tokens,
    )


class _UsageTracker:
    """Thread-safe running total of judge tokens + optional cost.

    Mirrors ``etsi_eval.runner._UsageTracker`` so the pairwise runner shows
    the same per-row ``cum=N cost=$X`` suffix as the absolute-score runner.
    """

    def __init__(self, pricing: config.JudgePricing):
        self._pricing = pricing
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, result: PairResult) -> str:
        with self._lock:
            self.prompt_tokens += result.judge_prompt_tokens
            self.completion_tokens += result.judge_completion_tokens
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


def _format_progress(label_a: str, label_b: str, result: PairResult) -> str:
    if result.errors:
        return f"#{result.id} ERROR ({result.errors[0]})"
    if result.winner == "A":
        verdict = f"{label_a} wins"
    elif result.winner == "B":
        verdict = f"{label_b} wins"
    elif result.winner == "tie":
        verdict = "tie"
    else:
        verdict = f"unstable (fwd={result.forward_winner}, swap={result.swapped_winner})"
    toks = (
        f" toks={result.judge_prompt_tokens}+{result.judge_completion_tokens}"
        if result.judge_total_tokens
        else ""
    )
    return f"#{result.id} {verdict}{toks}"


def _run_serial(
    rt: config.Runtime,
    paired: list[PairedRow],
    label_a: str,
    label_b: str,
    tracker: _UsageTracker,
) -> list[PairResult]:
    results: list[PairResult] = []
    total = len(paired)
    for i, pair in enumerate(paired, start=1):
        prefix = f"[{i}/{total}] {pair.category}/{pair.difficulty}/{pair.expected_behavior} ... "
        print(prefix, end="", flush=True)
        result = _process_pair(rt, pair)
        print(_format_progress(label_a, label_b, result) + tracker.add(result))
        results.append(result)
    return results


def _run_parallel(
    rt: config.Runtime,
    paired: list[PairedRow],
    label_a: str,
    label_b: str,
    concurrency: int,
    tracker: _UsageTracker,
) -> list[PairResult]:
    results: list[PairResult] = []
    total = len(paired)
    print_lock = threading.Lock()
    completed = {"n": 0}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_pair, rt, pair): pair for pair in paired}
        for fut in as_completed(futures):
            pair = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001 — defensive
                result = PairResult(
                    id=pair.id,
                    category=pair.category,
                    difficulty=pair.difficulty,
                    expected_behavior=pair.expected_behavior,
                    winner="",
                    forward_winner="",
                    swapped_winner="",
                    forward_rationale="",
                    swapped_rationale="",
                    errors=[f"runner: {e}"],
                )
            cum = tracker.add(result)
            with print_lock:
                completed["n"] += 1
                print(
                    f"[{completed['n']}/{total}] {pair.category}/{pair.difficulty}/{pair.expected_behavior} "
                    f"{_format_progress(label_a, label_b, result)}{cum}"
                )
            results.append(result)

    results.sort(key=lambda r: r.id)
    return results


def _print_summary(
    label_a: str,
    label_b: str,
    skipped: list[tuple[int, str]],
    agg: dict[str, Any],
) -> None:
    print()
    print("=" * 92)
    print(f"Pairs: {agg['total_pairs']}   scored: {agg['scored']}   errored: {agg['errored']}   skipped: {len(skipped)}")
    if skipped:
        print("Skipped:")
        for rid, reason in skipped[:10]:
            print(f"  #{rid}  {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    print()
    print(
        f"  {label_a:<24} wins: {agg['wins_a']}\n"
        f"  {label_b:<24} wins: {agg['wins_b']}\n"
        f"  {'ties':<24}     : {agg['ties']}\n"
        f"  {'unstable (positional)':<24}     : {agg['unstable']}"
    )
    if agg["win_rate_a"] is not None:
        print(
            f"\n  Win rate (decisive only, n={agg['wins_a'] + agg['wins_b']}): "
            f"{label_a}={agg['win_rate_a']:.3f}  {label_b}={agg['win_rate_b']:.3f}  "
            f"binomial p={agg['binomial_p_two_sided']:.4f}"
        )
    else:
        print("\n  Win rate: undefined (no decisive verdicts)")

    if agg["by_category"]:
        print("\nBy category:")
        for cat, c in agg["by_category"].items():
            print(
                f"  {cat:<12} A={c['wins_a']:>3}  B={c['wins_b']:>3}  "
                f"tie={c['ties']:>3}  unstable={c['unstable']:>3}"
            )
    if agg["by_difficulty"]:
        print("By difficulty:")
        for diff, c in agg["by_difficulty"].items():
            print(
                f"  {diff:<12} A={c['wins_a']:>3}  B={c['wins_b']:>3}  "
                f"tie={c['ties']:>3}  unstable={c['unstable']:>3}"
            )

    usage = agg["judge_usage"]
    if usage["total_tokens"]:
        n = usage["rows_with_usage"]
        avg_p = usage["prompt_tokens"] // n if n else 0
        avg_c = usage["completion_tokens"] // n if n else 0
        line = (
            f"\nJudge tokens: prompt={usage['prompt_tokens']:,}  "
            f"completion={usage['completion_tokens']:,}  "
            f"total={usage['total_tokens']:,}  "
            f"(avg per pair: {avg_p:,} in / {avg_c:,} out, n={n})"
        )
        cost = usage.get("estimated_cost_usd")
        if cost is not None:
            line += f"  estimated cost: ${cost:.4f}"
        print(line)
    print("=" * 92)


def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    if not project_root.exists():
        print(f"project root not found: {project_root}", file=sys.stderr)
        return 2

    if not args.a.exists():
        print(f"--a not found: {args.a}", file=sys.stderr)
        return 2
    if not args.b.exists():
        print(f"--b not found: {args.b}", file=sys.stderr)
        return 2

    rt = config.load(project_root)

    run_a = load_run(args.a, label=args.label_a)
    run_b = load_run(args.b, label=args.label_b)
    paired, skipped = pair_rows(run_a, run_b)

    if not paired:
        print("no rows could be paired — skipped:", file=sys.stderr)
        for rid, reason in skipped:
            print(f"  #{rid}  {reason}", file=sys.stderr)
        return 3

    if args.concurrency < 1:
        print(f"--concurrency must be >= 1, got {args.concurrency}", file=sys.stderr)
        return 2

    print(
        f"pairwise: {len(paired)} pair(s) (skipped {len(skipped)})  "
        f"concurrency={args.concurrency}\n"
        f"  A = {run_a.label} ({run_a.metadata.get('agent_backend')}/"
        f"{run_a.metadata.get('agent_model')})\n"
        f"  B = {run_b.label} ({run_b.metadata.get('agent_backend')}/"
        f"{run_b.metadata.get('agent_model')})\n"
        f"  judge = {rt.judge_backend.name}/{rt.judge_model}  "
        f"(2 passes per pair → {2 * len(paired)} judge calls)"
    )
    if rt.judge_pricing.enabled:
        print(
            f"  pricing: ${rt.judge_pricing.input_per_mtok:.2f}/MTok in, "
            f"${rt.judge_pricing.output_per_mtok:.2f}/MTok out — running cost shown per pair"
        )

    tracker = _UsageTracker(rt.judge_pricing)
    if args.concurrency == 1:
        results = _run_serial(rt, paired, run_a.label, run_b.label, tracker)
    else:
        results = _run_parallel(rt, paired, run_a.label, run_b.label, args.concurrency, tracker)

    agg = aggregate_pairs(results)
    if rt.judge_pricing.enabled:
        agg["judge_usage"]["estimated_cost_usd"] = round(
            rt.judge_pricing.cost_usd(
                agg["judge_usage"]["prompt_tokens"],
                agg["judge_usage"]["completion_tokens"],
            ),
            4,
        )
    _print_summary(run_a.label, run_b.label, skipped, agg)

    out_path = args.out
    if out_path is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).resolve().parent.parent / "pairwise_results"
        results_dir.mkdir(exist_ok=True)
        out_path = results_dir / f"{stamp}.json"

    out = {
        "schema_version": 1,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": uuid.uuid4().hex[:8],
        "concurrency": args.concurrency,
        "a_source": str(args.a),
        "b_source": str(args.b),
        "label_a": run_a.label,
        "label_b": run_b.label,
        "judge_backend": rt.judge_backend.name,
        "judge_model": rt.judge_model,
        "skipped": [{"id": rid, "reason": reason} for rid, reason in skipped],
        "aggregate": agg,
        "rows": [r.to_dict() for r in results],
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="etsi-pairwise",
        description=(
            "Compare two saved eval runs head-to-head. The judge picks which "
            "answer better addresses each question; each pair is judged twice "
            "with order swapped to detect positional bias."
        ),
    )
    here = Path(__file__).resolve().parent.parent
    p.add_argument("--a", type=Path, required=True, help="path to result JSON for side A")
    p.add_argument("--b", type=Path, required=True, help="path to result JSON for side B")
    p.add_argument("--label-a", help="display label for side A (default: filename stem)")
    p.add_argument("--label-b", help="display label for side B (default: filename stem)")
    p.add_argument(
        "--project-root",
        type=Path,
        default=here.parent,
        help="path to the etsi-rag project root (contains config/ and data/)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="number of pairs to judge in parallel (default: 1). The remote "
        "judge handles concurrent requests natively; bumping helps with "
        "latency-bound runs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        help="write results JSON to this path (default: eval/pairwise_results/TIMESTAMP.json)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
