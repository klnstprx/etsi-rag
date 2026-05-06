"""Aggregate eval rows and print a compact summary.

Per-row LLM-as-judge grading was retired (see plan/decision 2026-05-06):
absolute scores at this dataset size sit inside positional-flip noise. The
runner now captures only objective evidence — agent answer, tool calls
fired during the turn, retrieval metrics — and downstream comparison goes
through the pairwise runner.
"""

from __future__ import annotations

import statistics
from collections import Counter
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
    # tool invocations recorded by the agent loop. Each entry mirrors
    # agent.ToolCallRecord on the Go side: {name, args, error, duration_ms}.
    tool_calls: list[dict] = field(default_factory=list)
    # retrieval metrics; None for refuse rows or when retrieval failed
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    mrr: float | None = None
    retrieved_urls: list[str] = field(default_factory=list)
    # Per-stage failures accumulated during the run. Each entry is prefixed
    # with the stage name (e.g. "retrieval: ...", "agent: ...") so a row
    # that hit two failures retains both.
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _retrieval_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def _tool_call_stats(rows: list[Row]) -> dict[str, Any]:
    """Per-tool aggregation. Counts both invocations and rows that used the
    tool at least once. The latter answers "how often does the agent reach
    for this tool" without being skewed by chains of calls in one row."""
    by_tool_invocations: Counter[str] = Counter()
    by_tool_rows: dict[str, set[int]] = {}
    rows_using_any_tool: set[int] = set()
    total = 0

    for r in rows:
        if not r.tool_calls:
            continue
        rows_using_any_tool.add(r.id)
        for call in r.tool_calls:
            name = str(call.get("name", "")) or "?"
            by_tool_invocations[name] += 1
            by_tool_rows.setdefault(name, set()).add(r.id)
            total += 1

    by_tool: dict[str, dict[str, int]] = {}
    for name in sorted(by_tool_invocations):
        by_tool[name] = {
            "invocations": by_tool_invocations[name],
            "rows": len(by_tool_rows[name]),
        }
    return {
        "total_invocations": total,
        "rows_using_tool": len(rows_using_any_tool),
        "by_tool": by_tool,
    }


def aggregate(rows: list[Row]) -> dict[str, Any]:
    answered = [r for r in rows if r.agent_answer.strip()]
    errored = [r for r in rows if r.errors]

    retrieval: dict[str, dict[str, float | int] | None] = {}
    for metric in ("recall_at_5", "recall_at_10", "mrr"):
        vals = [getattr(r, metric) for r in rows if getattr(r, metric) is not None]
        retrieval[metric] = _retrieval_stats(vals) if vals else None

    return {
        "total": len(rows),
        "answered": len(answered),
        "errored": len(errored),
        "retrieval": retrieval,
        "tool_calls": _tool_call_stats(rows),
    }


def print_summary(rows: list[Row], agg: dict[str, Any]) -> None:
    print()
    print("=" * 92)
    print(f"Answered: {agg['answered']}/{agg['total']}   errors: {agg['errored']}")

    retrieval = agg.get("retrieval") or {}
    if any(v is not None for v in retrieval.values()):
        print("\nRetrieval (URL-level, excluding refuse rows):")
        for metric, stats in retrieval.items():
            if stats is None:
                continue
            print(f"  {metric:<14} mean={stats['mean']:>5}  (n={stats['count']})")

    tc = agg.get("tool_calls") or {}
    if tc.get("total_invocations"):
        print(
            f"\nTool calls: {tc['total_invocations']} invocations across "
            f"{tc['rows_using_tool']} rows"
        )
        for name, stats in tc.get("by_tool", {}).items():
            print(f"  {name:<20} invocations={stats['invocations']:>4}  rows={stats['rows']:>4}")
    else:
        print("\nTool calls: none recorded")

    if errored_rows := [r for r in rows if r.errors and not r.agent_answer.strip()]:
        print(f"\nErrored rows ({len(errored_rows)}):")
        for r in errored_rows[:10]:
            joined = "; ".join(r.errors)[:80]
            print(f"  #{r.id:<4} {r.category:<11} {r.difficulty:<8} {joined}")
        if len(errored_rows) > 10:
            print(f"  ... and {len(errored_rows) - 10} more")

    print("=" * 92)
