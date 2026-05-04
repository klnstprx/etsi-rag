"""Aggregation tests for report.Row, especially per-stage error accumulation."""

from __future__ import annotations

from etsi_eval.report import Row, aggregate


def _scored_row(rid: int, f: int, c: int, cm: int, **kw) -> Row:
    return Row(
        id=rid,
        category=kw.get("category", "course"),
        difficulty=kw.get("difficulty", "easy"),
        question="q",
        faithfulness=f,
        correctness=c,
        completeness=cm,
        overall=min(f, c, cm),
    )


def test_aggregate_basic_means():
    rows = [
        _scored_row(1, 100, 100, 100),
        _scored_row(2, 60, 80, 40),
    ]
    agg = aggregate(rows)
    assert agg["scored"] == 2
    assert agg["errored"] == 0
    assert agg["by_dimension"]["overall"]["mean"] == 70.0
    assert agg["by_dimension"]["overall"]["min"] == 40
    assert agg["by_dimension"]["overall"]["max"] == 100


def test_row_with_recovered_retrieval_error_still_scored():
    """A retrieval probe failure no longer blocks the judge stage; the row
    still ends up in `scored` but is also counted in `errored`."""
    r = _scored_row(1, 90, 90, 90)
    r.errors.append("retrieval: HTTPError boom")
    agg = aggregate([r])
    assert agg["scored"] == 1
    assert agg["errored"] == 1
    assert agg["by_dimension"]["overall"]["mean"] == 90.0


def test_multiple_errors_on_same_row_are_preserved():
    """Regression: previously errors clobbered each other (last-error-wins).
    Now both stages' failures should survive."""
    r = Row(id=1, category="x", difficulty="easy", question="q")
    r.errors.append("retrieval: HTTPError")
    r.errors.append("judge: 429 Too Many Requests")
    agg = aggregate([r])
    assert agg["scored"] == 0
    assert agg["errored"] == 1
    assert r.errors == ["retrieval: HTTPError", "judge: 429 Too Many Requests"]


def test_breakdown_groups_by_category_and_difficulty():
    rows = [
        _scored_row(1, 100, 100, 100, category="course", difficulty="easy"),
        _scored_row(2, 50, 50, 50, category="course", difficulty="hard"),
        _scored_row(3, 80, 80, 80, category="degree", difficulty="easy"),
    ]
    agg = aggregate(rows)
    assert agg["by_category"]["course"]["count"] == 2
    assert agg["by_category"]["degree"]["count"] == 1
    assert agg["by_difficulty"]["easy"]["mean"] == 90.0
    assert agg["by_difficulty"]["hard"]["mean"] == 50.0
