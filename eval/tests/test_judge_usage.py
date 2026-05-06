"""Tests for judge token-usage extraction and pricing."""

from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI

from etsi_eval.config import BackendConfig, JudgePricing
from etsi_eval.db import ReferenceDocument
from etsi_eval.judge import Usage, grade


_GOOD_CONTENT = {
    "faithfulness": {"score": 90, "rationale": "ok"},
    "correctness": {"score": 85, "rationale": "ok"},
    "completeness": {"score": 80, "rationale": "ok"},
}


def _completion_body(content: dict, *, usage: dict | None) -> dict:
    body = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "judge",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(content)},
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _make_client(handler) -> OpenAI:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAI(base_url="http://judge/v1", api_key="dummy", http_client=http)


def test_grade_returns_usage_from_response():
    body = _completion_body(
        _GOOD_CONTENT,
        usage={"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230},
    )

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    v = grade(
        backend,
        "judge-model",
        "q",
        "a",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    assert v.usage.prompt_tokens == 200
    assert v.usage.completion_tokens == 30
    assert v.usage.total_tokens == 230


def test_grade_works_when_usage_missing():
    """Some proxies strip the usage block — never crash, just zero-fill."""
    body = _completion_body(_GOOD_CONTENT, usage=None)

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    v = grade(
        backend,
        "judge-model",
        "q",
        "a",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    assert v.usage == Usage()
    assert v.faithfulness.score == 90  # parsing still works


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
