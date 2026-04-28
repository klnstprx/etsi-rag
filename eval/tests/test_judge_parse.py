"""Smoke tests for the judge-response parser. No network."""

from __future__ import annotations

import pytest

from etsi_eval.judge import JudgeError, _parse_verdict


_GOOD = """\
{
  "faithfulness":  {"score": 95, "rationale": "All claims supported."},
  "correctness":   {"score": 90, "rationale": "Coordinator name correct."},
  "completeness":  {"score": 80, "rationale": "Email present, room missing."}
}
"""


def test_plain_json():
    v = _parse_verdict(_GOOD)
    assert v.faithfulness.score == 95
    assert v.correctness.score == 90
    assert v.completeness.score == 80
    # overall is min across the three dims
    assert v.overall == 80
    assert "supported" in v.faithfulness.rationale


def test_fenced_json():
    raw = "```json\n" + _GOOD + "\n```"
    v = _parse_verdict(raw)
    assert v.overall == 80


def test_prose_with_json_tail():
    raw = "Here is my verdict:\n" + _GOOD
    v = _parse_verdict(raw)
    assert v.faithfulness.score == 95


def test_overall_is_min_not_mean():
    raw = """{
        "faithfulness": {"score": 100, "rationale": "ok"},
        "correctness":  {"score": 100, "rationale": "ok"},
        "completeness": {"score": 10, "rationale": "missed half"}
    }"""
    v = _parse_verdict(raw)
    assert v.overall == 10  # not 70 (mean) — min is the policy


def test_score_out_of_range():
    raw = """{
        "faithfulness": {"score": 150, "rationale": "x"},
        "correctness":  {"score": 90, "rationale": "x"},
        "completeness": {"score": 80, "rationale": "x"}
    }"""
    with pytest.raises(JudgeError, match="faithfulness"):
        _parse_verdict(raw)


def test_missing_dimension():
    raw = """{
        "faithfulness": {"score": 90, "rationale": "x"},
        "correctness":  {"score": 90, "rationale": "x"}
    }"""
    with pytest.raises(JudgeError, match="completeness"):
        _parse_verdict(raw)


def test_missing_score_field():
    raw = """{
        "faithfulness": {"rationale": "no score"},
        "correctness":  {"score": 90, "rationale": "x"},
        "completeness": {"score": 80, "rationale": "x"}
    }"""
    with pytest.raises(JudgeError, match="faithfulness"):
        _parse_verdict(raw)


def test_no_json():
    with pytest.raises(JudgeError):
        _parse_verdict("The answer looks mostly correct.")


def test_python_style_single_quote_escape():
    """Regression: gemma4-e4b emits \\' inside JSON strings (valid Python, invalid JSON)."""
    raw = (
        '{"faithfulness": {"score": 100, "rationale": "Cites the \\\'Jornada\\\' correctly."},'
        ' "correctness":  {"score": 100, "rationale": "ok"},'
        ' "completeness": {"score": 100, "rationale": "ok"}}'
    )
    v = _parse_verdict(raw)
    assert v.faithfulness.score == 100
    assert "Jornada" in v.faithfulness.rationale
