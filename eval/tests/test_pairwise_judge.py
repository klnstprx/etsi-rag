"""Tests for the pairwise preference judge.

The pairwise judge picks a winner between two candidate answers to the same
question. To control for positional bias the runner does TWO passes — one
with (A=answer1, B=answer2) and one swapped — and only counts a verdict
when both passes agree on the same underlying answer. These tests cover:

- Wire shape: response_format=json_schema constraining {winner, rationale}
- Winner enum: only "A" / "B" / "tie"
- Refuse vs answer rubric routing
- PairVerdict winner classification across all (forward, swapped) combos
- Usage extraction
"""

from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI

from etsi_eval.config import BackendConfig
from etsi_eval.db import ReferenceDocument
from etsi_eval.pairwise import (
    JudgeError,
    PairVerdict,
    SingleVerdict,
    Usage,
    grade_pair,
    grade_single,
)


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
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAI(base_url="http://judge/v1", api_key="dummy", http_client=http)


# ---- single-pass grade_single -------------------------------------------


def test_grade_single_returns_winner_a():
    body = _completion_body({"winner": "A", "rationale": "A is more grounded."})

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    v = grade_single(
        backend,
        "judge-model",
        "q",
        "answer-A",
        "answer-B",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    assert v.winner == "A"
    assert "grounded" in v.rationale


def test_grade_single_returns_winner_b():
    body = _completion_body({"winner": "B", "rationale": "B has the email."})

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    v = grade_single(
        backend,
        "judge-model",
        "q",
        "answer-A",
        "answer-B",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    assert v.winner == "B"


def test_grade_single_returns_tie():
    body = _completion_body({"winner": "tie", "rationale": "Both equivalent."})

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    v = grade_single(
        backend,
        "judge-model",
        "q",
        "answer-A",
        "answer-B",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    assert v.winner == "tie"


def test_grade_single_sends_json_schema_with_winner_enum():
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200, json=_completion_body({"winner": "A", "rationale": "ok"})
        )

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    grade_single(
        backend,
        "judge-model",
        "q",
        "answer-A",
        "answer-B",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    rf = captured["body"]["response_format"]
    assert rf["type"] == "json_schema"
    schema = rf["json_schema"]["schema"]
    assert set(schema["required"]) == {"winner", "rationale"}
    # winner must be enum-constrained at the schema level.
    winner_field = schema["properties"]["winner"]
    assert set(winner_field["enum"]) == {"A", "B", "tie"}


def test_grade_single_rejects_invalid_winner():
    """Defense in depth: even if the server ignores the enum, Pydantic catches it."""
    body = _completion_body({"winner": "C", "rationale": "x"})

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    with pytest.raises(JudgeError, match="winner"):
        grade_single(
            backend,
            "judge-model",
            "q",
            "answer-A",
            "answer-B",
            [ReferenceDocument(url="u", title="t", content="c")],
            client=_make_client(handler),
        )


def test_grade_single_refuse_mode_uses_refuse_system_prompt():
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200, json=_completion_body({"winner": "tie", "rationale": "ok"})
        )

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    grade_single(
        backend,
        "judge-model",
        "q",
        "answer-A",
        "answer-B",
        [],
        expected_behavior="refuse",
        client=_make_client(handler),
    )
    system = captured["body"]["messages"][0]
    assert system["role"] == "system"
    assert "refuse" in system["content"].lower() or "doesn't have" in system["content"]


def test_grade_single_unknown_expected_behavior_raises():
    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    with pytest.raises(ValueError, match="expected_behavior"):
        grade_single(
            backend,
            "judge-model",
            "q",
            "answer-A",
            "answer-B",
            [ReferenceDocument(url="u", title="t", content="c")],
            expected_behavior="bogus",
        )


def test_grade_single_includes_both_answers_in_user_message():
    """The judge needs both answers in the user message, labelled A and B."""
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200, json=_completion_body({"winner": "A", "rationale": "ok"})
        )

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    grade_single(
        backend,
        "judge-model",
        "what is X?",
        "FOO_ANSWER",
        "BAR_ANSWER",
        [ReferenceDocument(url="u", title="t", content="REF_DOC_CONTENT")],
        client=_make_client(handler),
    )
    user = captured["body"]["messages"][1]["content"]
    assert "FOO_ANSWER" in user
    assert "BAR_ANSWER" in user
    assert "REF_DOC_CONTENT" in user
    assert "what is X?" in user
    # The judge must see clear A/B labels.
    assert "Answer A" in user or "answer A" in user
    assert "Answer B" in user or "answer B" in user


def test_grade_single_returns_usage():
    body = _completion_body(
        {"winner": "A", "rationale": "ok"},
        prompt_tokens=300,
        completion_tokens=20,
    )

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    v = grade_single(
        backend,
        "judge-model",
        "q",
        "a",
        "b",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    assert v.usage.prompt_tokens == 300
    assert v.usage.completion_tokens == 20
    assert v.usage.total_tokens == 320


def test_grade_single_works_when_usage_missing():
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
                    "content": json.dumps({"winner": "A", "rationale": "ok"}),
                },
            }
        ],
    }

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    v = grade_single(
        backend,
        "judge-model",
        "q",
        "a",
        "b",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    assert v.usage == Usage()
    assert v.winner == "A"


# ---- two-pass grade_pair (the headline API) -----------------------------


def test_grade_pair_does_two_calls_with_swapped_order():
    """grade_pair must issue TWO judge calls; the second swaps A and B in the
    user prompt so positional bias is detectable."""
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        user = body["messages"][1]["content"]
        # Capture the order in which FOO/BAR appear so we can verify the swap.
        foo_idx = user.find("FOO_ANSWER")
        bar_idx = user.find("BAR_ANSWER")
        assert foo_idx >= 0 and bar_idx >= 0
        seen.append("FOO_first" if foo_idx < bar_idx else "BAR_first")
        return httpx.Response(
            200, json=_completion_body({"winner": "A", "rationale": "ok"})
        )

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    grade_pair(
        backend,
        "judge-model",
        "q",
        "FOO_ANSWER",
        "BAR_ANSWER",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    assert len(seen) == 2
    assert set(seen) == {"FOO_first", "BAR_first"}


def _stub_handler_factory(forward_winner: str, swapped_winner: str):
    """Return a handler that yields different verdicts on consecutive calls.

    The first call sees FOO_ANSWER before BAR_ANSWER (forward); the second
    sees them swapped. Each receives the corresponding stubbed verdict.
    """
    state = {"call": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        state["call"] += 1
        winner = forward_winner if state["call"] == 1 else swapped_winner
        return httpx.Response(
            200,
            json=_completion_body({"winner": winner, "rationale": f"call{state['call']}"}),
        )

    return handler


def _grade(handler) -> PairVerdict:
    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    return grade_pair(
        backend,
        "judge-model",
        "q",
        "FOO_ANSWER",
        "BAR_ANSWER",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )


def test_pair_winner_a_when_both_passes_pick_first_answer():
    # Forward: A=foo,B=bar → judge says A (= foo)
    # Swapped: A=bar,B=foo → judge says B (= foo)
    # Both passes agree the FIRST answer (foo) is best → winner="A".
    v = _grade(_stub_handler_factory("A", "B"))
    assert v.winner == "A"
    assert v.consistent is True


def test_pair_winner_b_when_both_passes_pick_second_answer():
    # Forward: A=foo,B=bar → judge says B (= bar)
    # Swapped: A=bar,B=foo → judge says A (= bar)
    # Both passes agree the SECOND answer (bar) is best → winner="B".
    v = _grade(_stub_handler_factory("B", "A"))
    assert v.winner == "B"
    assert v.consistent is True


def test_pair_tie_when_both_passes_say_tie():
    v = _grade(_stub_handler_factory("tie", "tie"))
    assert v.winner == "tie"
    assert v.consistent is True


def test_pair_unstable_on_positional_flip():
    # Both passes claim "position A wins" — i.e. judge prefers whatever it
    # sees first regardless of content. This is the canonical positional
    # bias signature and must be flagged, not silently counted.
    v = _grade(_stub_handler_factory("A", "A"))
    assert v.winner == "unstable"
    assert v.consistent is False


def test_pair_unstable_when_one_pass_ties_and_other_picks_a_side():
    # A real win should reproduce under swap; a tie + pick is ambiguous.
    v = _grade(_stub_handler_factory("A", "tie"))
    assert v.winner == "unstable"
    assert v.consistent is False


def test_pair_verdict_records_both_passes_separately():
    """Per-pass winners and rationales must be retrievable for debugging."""
    v = _grade(_stub_handler_factory("A", "B"))
    assert isinstance(v.forward, SingleVerdict)
    assert isinstance(v.swapped, SingleVerdict)
    assert v.forward.winner == "A"
    assert v.swapped.winner == "B"
    assert v.forward.rationale == "call1"
    assert v.swapped.rationale == "call2"


def test_pair_verdict_aggregates_usage_across_both_passes():
    state = {"call": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        state["call"] += 1
        return httpx.Response(
            200,
            json=_completion_body(
                {"winner": "A", "rationale": "ok"},
                prompt_tokens=100 * state["call"],
                completion_tokens=10 * state["call"],
            ),
        )

    backend = BackendConfig(name="b", base_url="http://judge/v1", api_key="")
    v = grade_pair(
        backend,
        "judge-model",
        "q",
        "FOO_ANSWER",
        "BAR_ANSWER",
        [ReferenceDocument(url="u", title="t", content="c")],
        client=_make_client(handler),
    )
    # Call 1: 100 prompt + 10 completion. Call 2: 200 + 20.
    assert v.usage.prompt_tokens == 300
    assert v.usage.completion_tokens == 30
    assert v.usage.total_tokens == 330
