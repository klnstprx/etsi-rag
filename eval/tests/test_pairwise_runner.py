"""Tests for the pairwise runner orchestration.

The runner consumes two existing result JSONs and pairs rows by id. We test
the pure helpers (load, pair, aggregate, binomial p) in isolation; the
end-to-end network path is exercised by the manual eval run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from etsi_eval.pairwise_runner import (
    PairResult,
    aggregate_pairs,
    binomial_p_two_sided,
    load_run,
    pair_rows,
)


# ---- binomial p-value (pure stdlib) -------------------------------------


def test_binomial_p_zero_total_is_one():
    assert binomial_p_two_sided(0, 0) == 1.0


def test_binomial_p_exact_half_is_one():
    # Observed = expected → no deviation → p = 1 (we cover every k).
    assert binomial_p_two_sided(5, 10) == pytest.approx(1.0)


def test_binomial_p_extreme_is_small():
    # 0 wins out of 10 with p=0.5 → P(|k-5| >= 5) = P(k=0) + P(k=10) = 2/1024
    assert binomial_p_two_sided(0, 10) == pytest.approx(2 / 1024)
    # Symmetric: 10 wins out of 10 must give the same p.
    assert binomial_p_two_sided(10, 10) == pytest.approx(2 / 1024)


def test_binomial_p_seven_of_ten():
    # |k-5| >= 2 → k in {0,1,2,3,7,8,9,10}
    # = 2 * (C(10,0)+C(10,1)+C(10,2)+C(10,3)) / 1024
    # = 2 * (1+10+45+120) / 1024 = 352/1024 = 0.34375
    assert binomial_p_two_sided(7, 10) == pytest.approx(352 / 1024)


def test_binomial_p_capped_at_one():
    # Numerical accumulation should never exceed 1.0.
    p = binomial_p_two_sided(50, 100)
    assert p <= 1.0


# ---- run JSON loader ----------------------------------------------------


def _make_result_json(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    body = {
        "schema_version": 2,
        "timestamp": "2026-05-06T00:00:00",
        "run_id": "stub",
        "concurrency": 1,
        "server_base_url": "http://stub",
        "agent_backend": "stub-backend",
        "agent_model": "stub-model",
        "judge_backend": "stub-judge",
        "judge_model": "stub-judge-model",
        "aggregate": {},
        "rows": rows,
    }
    p = tmp_path / name
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


_ROW_TEMPLATE = {
    "id": 1,
    "category": "course",
    "difficulty": "easy",
    "question": "q?",
    "expected_behavior": "answer",
    "golden_urls": ["https://example.org/doc"],
    "agent_answer": "answer",
    "errors": [],
}


def _row(**overrides) -> dict:
    return {**_ROW_TEMPLATE, **overrides}


def test_load_run_indexes_rows_by_id(tmp_path: Path):
    path = _make_result_json(
        tmp_path,
        "a.json",
        [_row(id=1), _row(id=2, agent_answer="b"), _row(id=5, agent_answer="c")],
    )
    run = load_run(path, label="A")
    assert run.label == "A"
    assert set(run.rows_by_id.keys()) == {1, 2, 5}
    assert run.rows_by_id[2].agent_answer == "b"


def test_load_run_default_label_is_filename_stem(tmp_path: Path):
    path = _make_result_json(tmp_path, "hybrid_topk10.json", [_row(id=1)])
    run = load_run(path)
    assert run.label == "hybrid_topk10"


# ---- pairing ------------------------------------------------------------


def test_pair_rows_pairs_by_id(tmp_path: Path):
    a = load_run(
        _make_result_json(
            tmp_path, "a.json", [_row(id=1, agent_answer="a1"), _row(id=2, agent_answer="a2")]
        )
    )
    b = load_run(
        _make_result_json(
            tmp_path, "b.json", [_row(id=1, agent_answer="b1"), _row(id=2, agent_answer="b2")]
        )
    )
    paired, skipped = pair_rows(a, b)
    assert {p.id for p in paired} == {1, 2}
    by_id = {p.id: p for p in paired}
    assert by_id[1].answer_a == "a1"
    assert by_id[1].answer_b == "b1"
    assert skipped == []


def test_pair_rows_skips_rows_missing_on_one_side(tmp_path: Path):
    a = load_run(_make_result_json(tmp_path, "a.json", [_row(id=1), _row(id=2)]))
    b = load_run(_make_result_json(tmp_path, "b.json", [_row(id=1)]))
    paired, skipped = pair_rows(a, b)
    assert {p.id for p in paired} == {1}
    assert skipped == [(2, "missing in B")]


def test_pair_rows_skips_when_golden_urls_differ(tmp_path: Path):
    """Different gold URLs means the runs scored against different ground
    truth — comparing them is meaningless."""
    a = load_run(
        _make_result_json(
            tmp_path, "a.json", [_row(id=1, golden_urls=["https://x"])]
        )
    )
    b = load_run(
        _make_result_json(
            tmp_path, "b.json", [_row(id=1, golden_urls=["https://y"])]
        )
    )
    paired, skipped = pair_rows(a, b)
    assert paired == []
    assert len(skipped) == 1
    assert skipped[0][0] == 1
    assert "golden" in skipped[0][1].lower()


def test_pair_rows_skips_when_expected_behavior_differs(tmp_path: Path):
    a = load_run(_make_result_json(tmp_path, "a.json", [_row(id=1, expected_behavior="answer")]))
    b = load_run(
        _make_result_json(
            tmp_path, "b.json", [_row(id=1, expected_behavior="refuse", golden_urls=[])]
        )
    )
    paired, skipped = pair_rows(a, b)
    assert paired == []
    assert "expected_behavior" in skipped[0][1]


def test_pair_rows_skips_when_either_answer_is_empty(tmp_path: Path):
    a = load_run(_make_result_json(tmp_path, "a.json", [_row(id=1, agent_answer="")]))
    b = load_run(_make_result_json(tmp_path, "b.json", [_row(id=1, agent_answer="ok")]))
    paired, skipped = pair_rows(a, b)
    assert paired == []
    assert "empty" in skipped[0][1].lower()


def test_pair_rows_skips_when_either_side_errored(tmp_path: Path):
    """If a run errored on a row (no agent answer to compare) we cannot
    fairly judge it — skip it and report it."""
    a = load_run(
        _make_result_json(
            tmp_path, "a.json", [_row(id=1, errors=["agent: timeout"], agent_answer="")]
        )
    )
    b = load_run(_make_result_json(tmp_path, "b.json", [_row(id=1)]))
    paired, skipped = pair_rows(a, b)
    assert paired == []
    assert len(skipped) == 1


def test_pair_rows_preserves_pair_metadata(tmp_path: Path):
    a = load_run(
        _make_result_json(
            tmp_path,
            "a.json",
            [_row(id=7, category="general", difficulty="hard", question="big Q?")],
        )
    )
    b = load_run(_make_result_json(tmp_path, "b.json", [_row(id=7, category="general", difficulty="hard", question="big Q?")]))
    paired, _ = pair_rows(a, b)
    p = paired[0]
    assert p.category == "general"
    assert p.difficulty == "hard"
    assert p.question == "big Q?"
    assert p.expected_behavior == "answer"
    assert p.golden_urls == ["https://example.org/doc"]


# ---- aggregation --------------------------------------------------------


def _result(
    id: int = 1,
    *,
    winner: str = "A",
    category: str = "course",
    difficulty: str = "easy",
    expected_behavior: str = "answer",
    prompt_tokens: int = 100,
    completion_tokens: int = 10,
    errors: list[str] | None = None,
) -> PairResult:
    return PairResult(
        id=id,
        category=category,
        difficulty=difficulty,
        expected_behavior=expected_behavior,
        winner=winner,
        forward_winner="A",
        swapped_winner="B",
        forward_rationale="",
        swapped_rationale="",
        judge_prompt_tokens=prompt_tokens,
        judge_completion_tokens=completion_tokens,
        judge_total_tokens=prompt_tokens + completion_tokens,
        errors=errors or [],
    )


def test_aggregate_counts_winners():
    results = [
        _result(id=1, winner="A"),
        _result(id=2, winner="A"),
        _result(id=3, winner="B"),
        _result(id=4, winner="tie"),
        _result(id=5, winner="unstable"),
    ]
    agg = aggregate_pairs(results)
    assert agg["wins_a"] == 2
    assert agg["wins_b"] == 1
    assert agg["ties"] == 1
    assert agg["unstable"] == 1
    assert agg["scored"] == 5
    assert agg["errored"] == 0


def test_aggregate_excludes_errored_from_counts():
    results = [
        _result(id=1, winner="A"),
        _result(id=2, winner="A", errors=["judge: boom"]),
    ]
    agg = aggregate_pairs(results)
    assert agg["wins_a"] == 1  # only the non-errored row counts
    assert agg["scored"] == 1
    assert agg["errored"] == 1


def test_aggregate_win_rate_excludes_ties_and_unstable():
    """Win-rate denominator is decisive verdicts only (A + B), so ties and
    unstable drop out — both reflect "no signal" not "evidence for either"."""
    results = [
        _result(id=1, winner="A"),
        _result(id=2, winner="A"),
        _result(id=3, winner="A"),
        _result(id=4, winner="B"),
        _result(id=5, winner="tie"),
        _result(id=6, winner="unstable"),
    ]
    agg = aggregate_pairs(results)
    # Decisive: 3 A vs 1 B = 4 total → win_rate_a = 0.75
    assert agg["win_rate_a"] == pytest.approx(0.75)
    assert agg["win_rate_b"] == pytest.approx(0.25)


def test_aggregate_win_rate_none_when_no_decisive_verdicts():
    results = [_result(id=1, winner="tie"), _result(id=2, winner="unstable")]
    agg = aggregate_pairs(results)
    assert agg["win_rate_a"] is None
    assert agg["win_rate_b"] is None
    assert agg["binomial_p_two_sided"] is None


def test_aggregate_binomial_p_three_a_one_b():
    # 3 wins for A, 1 win for B, no ties/unstable → 3/4 → p ≈ 0.625
    # (sum P(|k-2|>=1) for n=4, p=0.5: k in {0,1,3,4} → (1+4+4+1)/16 = 10/16)
    results = [
        _result(id=1, winner="A"),
        _result(id=2, winner="A"),
        _result(id=3, winner="A"),
        _result(id=4, winner="B"),
    ]
    agg = aggregate_pairs(results)
    assert agg["binomial_p_two_sided"] == pytest.approx(10 / 16)


def test_aggregate_breakdown_by_category():
    results = [
        _result(id=1, winner="A", category="course"),
        _result(id=2, winner="A", category="course"),
        _result(id=3, winner="B", category="general"),
        _result(id=4, winner="tie", category="general"),
    ]
    agg = aggregate_pairs(results)
    assert agg["by_category"]["course"] == {
        "wins_a": 2,
        "wins_b": 0,
        "ties": 0,
        "unstable": 0,
    }
    assert agg["by_category"]["general"] == {
        "wins_a": 0,
        "wins_b": 1,
        "ties": 1,
        "unstable": 0,
    }


def test_aggregate_judge_usage_sums_tokens():
    results = [
        _result(id=1, winner="A", prompt_tokens=200, completion_tokens=30),
        _result(id=2, winner="B", prompt_tokens=100, completion_tokens=10),
    ]
    agg = aggregate_pairs(results)
    usage = agg["judge_usage"]
    assert usage["prompt_tokens"] == 300
    assert usage["completion_tokens"] == 40
    assert usage["total_tokens"] == 340
    assert usage["rows_with_usage"] == 2
