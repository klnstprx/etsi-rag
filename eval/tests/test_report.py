"""Aggregation tests for the slim Row.

Per-row LLM-as-judge scoring was retired (2026-05-06); the runner now
captures only objective evidence — agent answer, tool calls, retrieval
metrics. Pairwise is the only judge-graded surface.
"""

from __future__ import annotations

from etsi_eval.report import Row, aggregate


def _answered_row(rid: int, **kw) -> Row:
    return Row(
        id=rid,
        category=kw.get("category", "course"),
        difficulty=kw.get("difficulty", "easy"),
        question="q",
        agent_answer=kw.get("agent_answer", "an answer"),
        tool_calls=kw.get("tool_calls", []),
        recall_at_5=kw.get("recall_at_5"),
        recall_at_10=kw.get("recall_at_10"),
        mrr=kw.get("mrr"),
    )


def test_aggregate_counts_answered_and_errored():
    rows = [
        _answered_row(1),
        _answered_row(2),
        Row(id=3, category="x", difficulty="easy", question="q",
            errors=["agent: timeout"]),
    ]
    agg = aggregate(rows)
    assert agg["total"] == 3
    assert agg["answered"] == 2
    assert agg["errored"] == 1


def test_aggregate_retrieval_metrics():
    rows = [
        _answered_row(1, recall_at_5=1.0, recall_at_10=1.0, mrr=1.0),
        _answered_row(2, recall_at_5=0.0, recall_at_10=0.5, mrr=0.5),
    ]
    agg = aggregate(rows)
    assert agg["retrieval"]["recall_at_5"]["mean"] == 0.5
    assert agg["retrieval"]["recall_at_10"]["mean"] == 0.75
    assert agg["retrieval"]["mrr"]["count"] == 2


def test_aggregate_skips_none_retrieval_values():
    """Refuse rows have no goldens; their metrics are None and should be
    excluded from aggregates rather than treated as zero."""
    rows = [
        _answered_row(1, recall_at_5=0.8, recall_at_10=1.0, mrr=0.5),
        _answered_row(2, recall_at_5=None, recall_at_10=None, mrr=None),
    ]
    agg = aggregate(rows)
    assert agg["retrieval"]["recall_at_5"]["count"] == 1
    assert agg["retrieval"]["recall_at_5"]["mean"] == 0.8


def test_tool_calls_aggregation_per_tool():
    rows = [
        _answered_row(1, tool_calls=[
            {"name": "semantic_search", "args": "{}", "duration_ms": 10},
        ]),
        _answered_row(2, tool_calls=[
            {"name": "semantic_search", "args": "{}", "duration_ms": 12},
            {"name": "read_document", "args": "{}", "duration_ms": 30},
        ]),
        _answered_row(3, tool_calls=[]),
    ]
    agg = aggregate(rows)
    tc = agg["tool_calls"]
    assert tc["total_invocations"] == 3
    assert tc["rows_using_tool"] == 2
    assert tc["by_tool"]["semantic_search"]["invocations"] == 2
    assert tc["by_tool"]["semantic_search"]["rows"] == 2
    assert tc["by_tool"]["read_document"]["invocations"] == 1
    assert tc["by_tool"]["read_document"]["rows"] == 1


def test_tool_calls_aggregation_no_calls():
    rows = [_answered_row(1, tool_calls=[])]
    agg = aggregate(rows)
    assert agg["tool_calls"]["total_invocations"] == 0
    assert agg["tool_calls"]["rows_using_tool"] == 0
    assert agg["tool_calls"]["by_tool"] == {}


def test_multiple_errors_on_same_row_are_preserved():
    """Regression: earlier code path used to clobber prior errors; both
    stages' failures should survive."""
    r = Row(id=1, category="x", difficulty="easy", question="q")
    r.errors.append("retrieval: HTTPError")
    r.errors.append("agent: 429 Too Many Requests")
    agg = aggregate([r])
    assert agg["answered"] == 0
    assert agg["errored"] == 1
    assert r.errors == ["retrieval: HTTPError", "agent: 429 Too Many Requests"]
