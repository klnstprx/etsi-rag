"""Thin OpenAI-compatible client that calls the etsi-rag agent.

We post to etsi-rag's own /v1/chat/completions (passthrough mode) so the agent
runs its tool-calling loop and semantic_search pipeline — this is the "system
under test".
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
    timeout: float = 120.0,
    passthrough: bool = True,
) -> AgentReply:
    """Send a single user turn to the agent and return its text reply.

    passthrough=True stops the agent from attaching any session state, so eval
    rows don't contaminate each other.
    """
    headers = {"Content-Type": "application/json"}
    if passthrough:
        # Passthrough mode is controlled by config (API.Passthrough). When the
        # server is in session mode, we still want each question standalone;
        # we set a unique session header so history never bleeds.
        headers["X-Session-ID"] = "eval-transient"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    url = f"{server_base_url.rstrip('/')}/v1/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        body = r.json()

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"agent returned no choices: {body}")
    content = choices[0].get("message", {}).get("content", "") or ""
    return AgentReply(content=content, raw=body)
