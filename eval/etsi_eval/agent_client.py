"""Thin OpenAI-compatible client that calls the etsi-rag agent.

We post to etsi-rag's own /v1/chat/completions so the agent runs its
tool-calling loop and semantic_search pipeline — this is the "system under
test". The caller supplies an explicit session_id so each eval row is
isolated when the server is in session mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class AgentReply:
    content: str
    raw: dict  # full response body for later introspection (sources, tool calls, etc.)


def ask(
    server_base_url: str,
    model: str,
    question: str,
    *,
    session_id: str,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> AgentReply:
    """Send a single user turn to the agent and return its text reply.

    session_id is sent as X-Session-ID. Callers MUST pass a value that is
    unique per row (and ideally per run) so chat history from one question
    doesn't bleed into the next when the server is in session mode.
    """
    if not session_id:
        raise ValueError("session_id is required")

    headers = {
        "Content-Type": "application/json",
        "X-Session-ID": session_id,
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    url = f"{server_base_url.rstrip('/')}/v1/chat/completions"

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

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"agent returned no choices: {body}")
    content = choices[0].get("message", {}).get("content", "") or ""
    return AgentReply(content=content, raw=body)
