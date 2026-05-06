"""Agent client tests — header propagation and session isolation.

Uses httpx.MockTransport (stdlib of the dependency) so no extra mocking
package is needed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from etsi_eval.agent_client import ask


def _stub_response(captured: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": json.loads(request.content.decode("utf-8")),
            }
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}},
                ]
            },
        )

    return httpx.MockTransport(handler)


def test_session_id_is_required():
    with pytest.raises(ValueError, match="session_id"):
        ask("http://x", "m", "q", session_id="")


def test_session_id_is_sent_as_header():
    captured: list[dict] = []
    transport = _stub_response(captured)
    with httpx.Client(transport=transport) as client:
        ask(
            "http://x",
            "m",
            "q",
            session_id="eval-abc12345-7",
            client=client,
        )
    assert captured[0]["headers"]["x-session-id"] == "eval-abc12345-7"


def test_distinct_session_ids_across_calls():
    """Regression: previously every call used a hardcoded "eval-transient"
    session id, so questions in session-mode bled into each other."""
    captured: list[dict] = []
    transport = _stub_response(captured)
    with httpx.Client(transport=transport) as client:
        ask("http://x", "m", "q1", session_id="eval-run-1", client=client)
        ask("http://x", "m", "q2", session_id="eval-run-2", client=client)
    seen = {c["headers"]["x-session-id"] for c in captured}
    assert seen == {"eval-run-1", "eval-run-2"}


def test_payload_shape():
    captured: list[dict] = []
    transport = _stub_response(captured)
    with httpx.Client(transport=transport) as client:
        ask("http://x/", "agent-model", "Hola?", session_id="s1", client=client)
    body = captured[0]["body"]
    assert body["model"] == "agent-model"
    assert body["messages"] == [{"role": "user", "content": "Hola?"}]
    assert body["stream"] is False
    # base url should have its trailing slash stripped before /v1/...
    assert captured[0]["url"] == "http://x/v1/chat/completions"


def test_tool_calls_propagate_from_response():
    """The custom root-level tool_calls extension must round-trip from the
    chat completion response into the AgentReply, in order, untouched."""
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "answer"}}],
                "tool_calls": [
                    {"name": "semantic_search", "args": '{"query":"x"}', "duration_ms": 12},
                    {"name": "read_document", "args": '{"url":"u"}', "error": False, "duration_ms": 30},
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reply = ask("http://x", "m", "q", session_id="s", client=client)
    assert reply.content == "answer"
    assert len(reply.tool_calls) == 2
    assert reply.tool_calls[0]["name"] == "semantic_search"
    assert reply.tool_calls[1]["name"] == "read_document"


def test_tool_calls_empty_when_field_absent():
    """Backwards-compatibility: a response without the tool_calls field
    must not blow up — the agent simply made no tool calls (or the server
    is too old to emit them)."""
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        reply = ask("http://x", "m", "q", session_id="s", client=client)
    assert reply.tool_calls == []
