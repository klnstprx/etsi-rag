"""Thin client for the etsi-rag retrieval-introspection endpoint.

Calls POST /v1/rag/query directly (no agent loop), returns the ranked chunks
in the order the retrieval layer emitted them so we can compute Recall@k /
MRR against ground-truth golden URLs.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: str
    document_title: str
    document_url: str  # may be empty if the source doc had no URL
    section_title: str
    chunk_index: int
    score: float
    snippet: str


def query(
    server_base_url: str,
    question: str,
    *,
    top_k: int | None = None,
    language: str | None = None,
    timeout: float = 30.0,
) -> list[RetrievedChunk]:
    """POST /v1/rag/query and return the ranked chunk list (highest score first)."""
    payload: dict[str, object] = {"query": question}
    if top_k is not None:
        payload["top_k"] = top_k
    if language is not None:
        payload["language"] = language

    url = f"{server_base_url.rstrip('/')}/v1/rag/query"
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        body = r.json()

    return [
        RetrievedChunk(
            chunk_id=row.get("chunk_id", 0),
            document_id=row.get("document_id", ""),
            document_title=row.get("document_title", ""),
            document_url=row.get("document_url") or "",
            section_title=row.get("section_title", ""),
            chunk_index=row.get("chunk_index", 0),
            score=row.get("score", 0.0),
            snippet=row.get("snippet", ""),
        )
        for row in (body.get("results") or [])
    ]
