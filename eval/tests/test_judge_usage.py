"""Tests for judge token-usage extraction and pricing."""

from __future__ import annotations

import json

import httpx
import pytest

from etsi_eval.config import JudgePricing
from etsi_eval.db import ReferenceDocument
from etsi_eval.judge import Usage, _parse_usage, grade
from etsi_eval.report import Row, aggregate


_OK_VERDICT = json.dumps(
    {
        "faithfulness": {"score": 90, "rationale": "ok"},
        "correctness": {"score": 85, "rationale": "ok"},
        "completeness": {"score": 80, "rationale": "ok"},
    }
)


def _ok_body(prompt: int, completion: int) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": _OK_VERDICT}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


# ---- _parse_usage --------------------------------------------------------


def test_parse_usage_happy_path():
    body = _ok_body(123, 45)
    u = _parse_usage(body)
    assert u == Usage(prompt_tokens=123, completion_tokens=45, total_tokens=168)


def test_parse_usage_missing_block_returns_zeros():
    """Some proxies strip the usage block — never crash, just zero-fill."""
    u = _parse_usage({"choices": []})
    assert u == Usage()


def test_parse_usage_handles_string_numbers():
    """Some non-conformant proxies wrap numeric fields as strings."""
    body = {"usage": {"prompt_tokens": "10", "completion_tokens": "5", "total_tokens": "15"}}
    u = _parse_usage(body)
    assert u.prompt_tokens == 10
    assert u.total_tokens == 15


def test_parse_usage_handles_garbage_gracefully():
    body = {"usage": {"prompt_tokens": "not-a-number"}}
    u = _parse_usage(body)
    assert u == Usage()


# ---- end-to-end via grade() ---------------------------------------------


def test_grade_returns_usage_from_response():
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, json=_ok_body(200, 30)))
    from etsi_eval.config import BackendConfig

    backend = BackendConfig(name="b", base_url="http://x", api_key="")
    with httpx.Client(transport=transport) as client:
        v = grade(
            backend,
            "judge-model",
            "q",
            "a",
            [ReferenceDocument(url="u", title="t", content="c")],
            client=client,
        )
    assert v.usage.prompt_tokens == 200
    assert v.usage.completion_tokens == 30
    assert v.usage.total_tokens == 230


def test_grade_works_when_usage_missing():
    body = {"choices": [{"message": {"role": "assistant", "content": _OK_VERDICT}}]}
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, json=body))
    from etsi_eval.config import BackendConfig

    backend = BackendConfig(name="b", base_url="http://x", api_key="")
    with httpx.Client(transport=transport) as client:
        v = grade(
            backend,
            "judge-model",
            "q",
            "a",
            [ReferenceDocument(url="u", title="t", content="c")],
            client=client,
        )
    assert v.usage == Usage()
    assert v.faithfulness.score == 90  # parsing still works


# ---- aggregate sums tokens ----------------------------------------------


def test_aggregate_sums_judge_tokens():
    rows = [
        Row(id=1, category="x", difficulty="easy", question="q",
            judge_prompt_tokens=100, judge_completion_tokens=20, judge_total_tokens=120,
            faithfulness=90, correctness=90, completeness=90, overall=90),
        Row(id=2, category="x", difficulty="easy", question="q",
            judge_prompt_tokens=150, judge_completion_tokens=30, judge_total_tokens=180,
            faithfulness=80, correctness=80, completeness=80, overall=80),
        Row(id=3, category="x", difficulty="easy", question="q"),  # no usage
    ]
    agg = aggregate(rows)
    usage = agg["judge_usage"]
    assert usage["prompt_tokens"] == 250
    assert usage["completion_tokens"] == 50
    assert usage["total_tokens"] == 300
    assert usage["rows_with_usage"] == 2


# ---- pricing -------------------------------------------------------------


def test_pricing_cost_calculation():
    p = JudgePricing(input_per_mtok=3.0, output_per_mtok=15.0)
    # 1M in @ $3 + 1M out @ $15 = $18.00
    assert p.cost_usd(1_000_000, 1_000_000) == pytest.approx(18.0)
    # 100k in @ $3 = $0.30
    assert p.cost_usd(100_000, 0) == pytest.approx(0.30)


def test_pricing_disabled_by_default():
    assert not JudgePricing().enabled
    assert JudgePricing(input_per_mtok=0.5).enabled
    assert JudgePricing(output_per_mtok=0.5).enabled
