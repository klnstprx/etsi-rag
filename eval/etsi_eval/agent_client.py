"""Thin client for etsi-rag's proprietary agent endpoint.

We POST to /v1/agent/chat so the agent runs its tool-calling loop and
semantic_search pipeline — this is the "system under test". The proprietary
endpoint (unlike the OpenAI-compatible /v1/chat/completions) returns the full
RAG context, including the root-level tool_calls the eval harness records.
The caller supplies an explicit session_id so each eval row is isolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class AgentReply:
    content: str
    tool_calls: list[dict]  # ToolCallRecord shape: {name, args, error, duration_ms}
    raw: dict  # full response body for later introspection


def ask(
    server_base_url: str,
    question: str,
    *,
    session_id: str,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> AgentReply:
    """Send a single user turn to the agent and return its text reply.

    session_id is sent as X-Session-ID. Callers MUST pass a value that is
    unique per row (and ideally per run) so chat history from one question
    doesn't bleed into the next.
    """
    if not session_id:
        raise ValueError("session_id is required")

    headers = {
        "Content-Type": "application/json",
        "X-Session-ID": session_id,
    }
    payload = {
        "message": question,
        "stream": False,
    }
    url = f"{server_base_url.rstrip('/')}/v1/agent/chat"

    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=timeout)
    try:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        body = r.json()
    finally:
        if owns_client:
            client.close()

    content = body.get("content", "") or ""
    tool_calls = list(body.get("tool_calls") or [])
    return AgentReply(content=content, tool_calls=tool_calls, raw=body)
