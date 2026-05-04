"""Retry / backoff tests for the judge HTTP client."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from etsi_eval.config import BackendConfig
from etsi_eval.db import ReferenceDocument
from etsi_eval.judge import _post_with_retry, grade


_OK_BODY = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "faithfulness": {"score": 90, "rationale": "ok"},
                        "correctness": {"score": 90, "rationale": "ok"},
                        "completeness": {"score": 90, "rationale": "ok"},
                    }
                ),
            }
        }
    ]
}


def _scripted(responses: Iterator[httpx.Response]) -> httpx.MockTransport:
    def handler(_req: httpx.Request) -> httpx.Response:
        return next(responses)

    return httpx.MockTransport(handler)


def test_post_with_retry_recovers_from_429():
    sleeps: list[float] = []
    seq = iter(
        [
            httpx.Response(429, text="rate limited"),
            httpx.Response(429, text="rate limited"),
            httpx.Response(200, json=_OK_BODY),
        ]
    )
    transport = _scripted(seq)
    with httpx.Client(transport=transport) as client:
        r = _post_with_retry(
            client,
            "http://x/chat/completions",
            json_body={},
            headers={},
            backoff=(0.0, 0.0, 0.0),
            sleep=lambda s: sleeps.append(s),
        )
    assert r.status_code == 200
    # Two retries → two sleeps. Each gets jitter 0.5..1.5 of the base, but
    # base is 0 here, so sleeps stay 0.
    assert len(sleeps) == 2


def test_post_with_retry_recovers_from_5xx():
    seq = iter(
        [
            httpx.Response(503, text="upstream"),
            httpx.Response(200, json=_OK_BODY),
        ]
    )
    transport = _scripted(seq)
    with httpx.Client(transport=transport) as client:
        r = _post_with_retry(
            client,
            "http://x",
            json_body={},
            headers={},
            backoff=(0.0, 0.0, 0.0),
            sleep=lambda _s: None,
        )
    assert r.status_code == 200


def test_post_with_retry_does_not_retry_on_4xx_other_than_429():
    """401/403/400 are user errors — a retry won't fix them."""
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="unauthorized")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            _post_with_retry(
                client,
                "http://x",
                json_body={},
                headers={},
                backoff=(0.0, 0.0, 0.0),
                sleep=lambda _s: None,
            )
    assert calls["n"] == 1


def test_post_with_retry_gives_up_after_exhausting_attempts():
    seq = iter([httpx.Response(429) for _ in range(10)])
    transport = _scripted(seq)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            _post_with_retry(
                client,
                "http://x",
                json_body={},
                headers={},
                backoff=(0.0, 0.0, 0.0),
                sleep=lambda _s: None,
            )


def test_grade_uses_retry_path():
    """End-to-end: grade() should recover from a transient 429."""
    seq = iter(
        [
            httpx.Response(429),
            httpx.Response(200, json=_OK_BODY),
        ]
    )
    transport = _scripted(seq)
    backend = BackendConfig(name="b", base_url="http://x", api_key="")
    with httpx.Client(transport=transport) as client:
        # Force backoff to 0 by monkey-patching the module constant via the
        # client knob: pass our own client, and pre-empt sleeps by replacing
        # time.sleep at the source.
        import etsi_eval.judge as judge_mod

        original = judge_mod._RETRY_BACKOFF_SECONDS
        judge_mod._RETRY_BACKOFF_SECONDS = (0.0, 0.0, 0.0)
        try:
            verdict = grade(
                backend,
                "judge-model",
                "q",
                "a",
                [ReferenceDocument(url="u", title="t", content="c")],
                client=client,
            )
        finally:
            judge_mod._RETRY_BACKOFF_SECONDS = original
    assert verdict.faithfulness.score == 90
