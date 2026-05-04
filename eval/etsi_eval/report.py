"""Aggregate eval rows and print a compact summary."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Row:
    id: int
    category: str
    difficulty: str
    question: str
    golden_urls: list[str] = field(default_factory=list)
    expected_behavior: str = "answer"
    # populated as the run progresses
    agent_answer: str = ""
    ref_found: bool = False
    ref_titles: list[str] = field(default_factory=list)
    ref_content_len: int = 0
    # judge sub-scores; None when the judge errored or wasn't run
    faithfulness: int | None = None
    correctness: int | None = None
    completeness: int | None = None
    overall: int | None = None
    # rationales per dimension
    faithfulness_rationale: str = ""
    correctness_rationale: str = ""
    completeness_rationale: str = ""
    # retrieval metrics; None for refuse rows or when retrieval failed
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    mrr: float | None = None
    retrieved_urls: list[str] = field(default_factory=list)
    # token usage reported by the judge backend (OpenAI-compatible `usage` block)
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_total_tokens: int = 0
    # Per-stage failures accumulated during the run. Each entry is prefixed
    # with the stage name (e.g. "retrieval: ...", "agent: ...", "judge: ...")
    # so a row that hit two failures retains both.
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DIMS: tuple[str, ...] = ("faithfulness", "correctness", "completeness", "overall")


def _stats(scores: list[int]) -> dict[str, float | int]:
    return {
        "count": len(scores),
        "mean": round(statistics.mean(scores), 1),
        "min": min(scores),
        "max": max(scores),
    }


def _retrieval_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def aggregate(rows: list[Row]) -> dict[str, Any]:
    scored = [r for r in rows if r.overall is not None]
    errored = [r for r in rows if r.errors]

    by_dim: dict[str, dict[str, float | int] | None] = {}
    for dim in _DIMS:
        vals = [getattr(r, dim) for r in scored if getattr(r, dim) is not None]
        by_dim[dim] = _stats(vals) if vals else None  # type: ignore[arg-type]

    def breakdown(key: str) -> dict[str, dict[str, float | int]]:
        groups: dict[str, list[int]] = {}
        for r in scored:
            assert r.overall is not None
            groups.setdefault(getattr(r, key), []).append(r.overall)
        return {k: _stats(v) for k, v in sorted(groups.items())}

    retrieval: dict[str, dict[str, float | int] | None] = {}
    for metric in ("recall_at_5", "recall_at_10", "mrr"):
        vals = [getattr(r, metric) for r in rows if getattr(r, metric) is not None]
        retrieval[metric] = _retrieval_stats(vals) if vals else None

    judge_usage = {
        "prompt_tokens": sum(r.judge_prompt_tokens for r in rows),
        "completion_tokens": sum(r.judge_completion_tokens for r in rows),
        "total_tokens": sum(r.judge_total_tokens for r in rows),
        "rows_with_usage": sum(1 for r in rows if r.judge_total_tokens > 0),
    }

    return {
        "total": len(rows),
        "scored": len(scored),
        "errored": len(errored),
        "by_dimension": by_dim,
        "retrieval": retrieval,
        "by_category": breakdown("category"),
        "by_difficulty": breakdown("difficulty"),
        "judge_usage": judge_usage,
    }


def _fmt_score(v: int | None) -> str:
    return f"{v}" if v is not None else "-"


def print_summary(rows: list[Row], agg: dict[str, Any]) -> None:
    print()
    print("=" * 92)
    print(f"Scored: {agg['scored']}/{agg['total']}   errors: {agg['errored']}")
    overall_stats = agg["by_dimension"].get("overall")
    if overall_stats:
        means = {d: agg["by_dimension"][d]["mean"] for d in _DIMS if agg["by_dimension"][d]}
        print(
            "Means: "
            + "  ".join(f"{d}={means[d]}" for d in _DIMS if d in means)
        )
    print()
    print(f"{'id':<4}{'cat':<11}{'diff':<8}{'F':>4}{'C':>4}{'Cm':>4}{'O':>4}  rationale (faithfulness)")
    print("-" * 92)
    for r in rows:
        if r.errors and r.overall is None:
            joined = "; ".join(r.errors)[:60]
            print(f"{r.id:<4}{r.category:<11}{r.difficulty:<8}{'ERR':>16}  {joined}")
            continue
        rat = r.faithfulness_rationale.replace("\n", " ")[:60]
        print(
            f"{r.id:<4}{r.category:<11}{r.difficulty:<8}"
            f"{_fmt_score(r.faithfulness):>4}{_fmt_score(r.correctness):>4}"
            f"{_fmt_score(r.completeness):>4}{_fmt_score(r.overall):>4}  {rat}"
        )
    print()
    print("By category (overall):")
    for cat, stats in agg["by_category"].items():
        print(f"  {cat:<12} mean={stats['mean']:>5}  (n={stats['count']})")
    print("By difficulty (overall):")
    for diff, stats in agg["by_difficulty"].items():
        print(f"  {diff:<12} mean={stats['mean']:>5}  (n={stats['count']})")
    retrieval = agg.get("retrieval") or {}
    if any(v is not None for v in retrieval.values()):
        print("Retrieval (URL-level, excluding refuse rows):")
        for metric, stats in retrieval.items():
            if stats is None:
                continue
            print(f"  {metric:<14} mean={stats['mean']:>5}  (n={stats['count']})")
    usage = agg.get("judge_usage") or {}
    total = usage.get("total_tokens", 0)
    if total:
        n = usage["rows_with_usage"]
        avg_prompt = usage["prompt_tokens"] // n if n else 0
        avg_completion = usage["completion_tokens"] // n if n else 0
        line = (
            f"Judge tokens: prompt={usage['prompt_tokens']:,}  "
            f"completion={usage['completion_tokens']:,}  "
            f"total={total:,}  "
            f"(avg per row: {avg_prompt:,} in / {avg_completion:,} out, n={n})"
        )
        cost = usage.get("estimated_cost_usd")
        if cost is not None:
            line += f"  estimated cost: ${cost:.4f}"
        print(line)
    print("=" * 92)
