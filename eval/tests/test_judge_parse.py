"""Tests for the structured-output judge.

The judge sends a JSON-schema response_format and the OpenAI client returns
a parsed Pydantic model. There is no manual JSON-string parsing path here:
the schema enforces shape on the server, and Pydantic enforces it on the
client. These tests cover the wire shape and the result mapping.
"""

from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI

from etsi_eval.config import BackendConfig
from etsi_eval.db import ReferenceDocument
from etsi_eval.judge import JudgeError, grade


def _completion_body(content: dict, *, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    return {
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
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_client(handler) -> OpenAI:
    """Build an OpenAI client wired to a mock transport.

    The base_url and api_key are stub values: nothing leaves the process.
    """
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAI(base_url="http://judge/v1", api_key="dummy", http_client=http)


_GOOD_VERDICT = {
    "faithfulness": {"score": 95, "rationale": "All claims supported."},
    "correctness": {"score": 90, "rationale": "Names correct."},
    "completeness": {"score": 80, "rationale": "Email present, room missing."},
}


def test_grade_returns_verdict_from_parsed_model():
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_completion_body(_GOOD_VERDICT))

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    v = grade(
        backend,
        "judge-model",
        "q",
        "a",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    assert v.faithfulness.score == 95
    assert v.correctness.score == 90
    assert v.completeness.score == 80
    assert v.overall == 80
    assert "supported" in v.faithfulness.rationale


def test_grade_sends_json_schema_response_format():
    """The whole point: response_format must be a json_schema constraining
    the three-dimension shape. Without this the server can return free text."""
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_completion_body(_GOOD_VERDICT))

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    grade(
        backend,
        "judge-model",
        "q",
        "a",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    rf = captured["body"]["response_format"]
    assert rf["type"] == "json_schema"
    schema = rf["json_schema"]["schema"]
    # Top-level must require the three dimensions as siblings — the exact
    # shape the malformed gemma4-e4b output violated.
    assert set(schema["required"]) == {"faithfulness", "correctness", "completeness"}
    # Every dim must require score + rationale.
    dim_schema = schema["$defs"][next(iter(schema["$defs"]))]
    assert set(dim_schema["required"]) == {"score", "rationale"}


def test_grade_overall_is_min_not_mean():
    body = _completion_body(
        {
            "faithfulness": {"score": 100, "rationale": "ok"},
            "correctness": {"score": 100, "rationale": "ok"},
            "completeness": {"score": 10, "rationale": "missed half"},
        }
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
    assert v.overall == 10  # not 70 (mean)


def test_grade_rejects_score_out_of_range():
    """Defense in depth: even if a server ignores the schema and returns
    a 150, Pydantic validation must catch it."""
    body = _completion_body(
        {
            "faithfulness": {"score": 150, "rationale": "x"},
            "correctness": {"score": 90, "rationale": "x"},
            "completeness": {"score": 80, "rationale": "x"},
        }
    )

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    with pytest.raises(JudgeError, match="score"):
        grade(
            backend,
            "judge-model",
            "q",
            "a",
            [ReferenceDocument(url="u", title="t", content="c")],
            client=_make_client(handler),
        )


def test_grade_raises_on_refusal():
    """If the judge model refuses (safety classifier triggered), surface
    that as a JudgeError rather than silently producing zero scores."""
    body = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "judge",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "refusal": "I can't help with that.",
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    with pytest.raises(JudgeError, match="refus"):
        grade(
            backend,
            "judge-model",
            "q",
            "a",
            [ReferenceDocument(url="u", title="t", content="c")],
            client=_make_client(handler),
        )


def test_grade_refuse_mode_uses_refuse_system_prompt():
    """expected_behavior='refuse' selects the no-reference-doc rubric."""
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_completion_body(_GOOD_VERDICT))

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    grade(
        backend,
        "judge-model",
        "q",
        "(empty)",
        [],
        expected_behavior="refuse",
        client=_make_client(handler),
    )
    system = captured["body"]["messages"][0]
    assert system["role"] == "system"
    # Refuse rubric mentions refusal/lack-of-information explicitly.
    assert "refuse" in system["content"].lower() or "doesn't have" in system["content"]


def test_grade_unknown_expected_behavior_raises():
    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    with pytest.raises(ValueError, match="expected_behavior"):
        grade(
            backend,
            "judge-model",
            "q",
            "a",
            [ReferenceDocument(url="u", title="t", content="c")],
            expected_behavior="bogus",
        )
