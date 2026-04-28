"""URL-level retrieval metrics for the eval harness.

Operating at URL granularity (not chunk-id) keeps the metrics stable across
re-crawls — chunk IDs change every time content is re-ingested, but the
canonical URL is what matters for "did the retriever find the right document".

For multi-URL goldens (cross-doc questions), recall = fraction of golden URLs
that appear anywhere in retrieved[:k]; MRR = 1/rank of the EARLIEST hit on
ANY golden URL (not the rank where every golden has been seen).

For refusal-mode rows golden_urls is empty and the metrics are undefined; the
functions return None so the runner can report them as `null` rather than
falsifying with a 0.
"""

from __future__ import annotations


def recall_at_k(
    retrieved_urls: list[str],
    golden_urls: list[str],
    k: int,
) -> float | None:
    """Fraction of golden URLs that appear in the first k retrieved URLs.

    Returns None when golden_urls is empty (metric is undefined).
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    if not golden_urls:
        return None
    head = set(retrieved_urls[:k])
    hits = sum(1 for u in golden_urls if u in head)
    return hits / len(golden_urls)


def mrr(
    retrieved_urls: list[str],
    golden_urls: list[str],
) -> float | None:
    """Reciprocal rank of the first retrieved URL that is a golden URL.

    Multi-golden case: uses rank of the earliest hit on any golden URL,
    matching the standard IR definition where MRR is "first relevant doc"
    even when multiple are relevant.

    Returns None when golden_urls is empty.
    """
    if not golden_urls:
        return None
    golden_set = set(golden_urls)
    for i, url in enumerate(retrieved_urls, start=1):
        if url in golden_set:
            return 1.0 / i
    return 0.0
