"""Pure-function tests for retrieval metrics — no network."""

from __future__ import annotations

import math

import pytest

from etsi_eval.metrics import mrr, recall_at_k


# ---- recall_at_k ---------------------------------------------------------


def test_recall_at_k_full_hit():
    """Single golden URL, present at rank 1 → recall = 1.0."""
    retrieved = ["https://a", "https://b", "https://c"]
    golden = ["https://a"]
    assert recall_at_k(retrieved, golden, k=5) == 1.0


def test_recall_at_k_miss():
    retrieved = ["https://x", "https://y"]
    golden = ["https://a"]
    assert recall_at_k(retrieved, golden, k=5) == 0.0


def test_recall_at_k_truncates_to_k():
    """A hit beyond rank k must NOT count."""
    retrieved = ["https://x", "https://y", "https://a"]
    golden = ["https://a"]
    assert recall_at_k(retrieved, golden, k=2) == 0.0
    assert recall_at_k(retrieved, golden, k=3) == 1.0


def test_recall_at_k_multi_golden_partial():
    """Two golden URLs, one retrieved → 0.5."""
    retrieved = ["https://a", "https://x", "https://y"]
    golden = ["https://a", "https://b"]
    assert recall_at_k(retrieved, golden, k=3) == 0.5


def test_recall_at_k_dedupes_retrieved():
    """Same URL appearing twice in top-k shouldn't double-count."""
    retrieved = ["https://a", "https://a", "https://x"]
    golden = ["https://a", "https://b"]
    # Only one of two goldens hit, regardless of duplicates → 0.5
    assert recall_at_k(retrieved, golden, k=5) == 0.5


def test_recall_at_k_undefined_for_empty_golden():
    """Refusal questions have golden=[]; the metric is undefined → None."""
    assert recall_at_k(["https://a"], [], k=5) is None


def test_recall_at_k_zero_k_raises():
    with pytest.raises(ValueError):
        recall_at_k(["https://a"], ["https://a"], k=0)


# ---- mrr -----------------------------------------------------------------


def test_mrr_first_position():
    assert mrr(["https://a", "https://b"], ["https://a"]) == 1.0


def test_mrr_third_position():
    rr = mrr(["https://x", "https://y", "https://a"], ["https://a"])
    assert math.isclose(rr, 1 / 3)


def test_mrr_no_hit():
    assert mrr(["https://x", "https://y"], ["https://a"]) == 0.0


def test_mrr_uses_first_golden_hit():
    """When multiple goldens exist, MRR uses rank of the EARLIEST hit."""
    retrieved = ["https://x", "https://b", "https://a"]
    golden = ["https://a", "https://b"]
    # b appears at rank 2 first; a is at rank 3. MRR uses rank 2 → 0.5.
    assert mrr(retrieved, golden) == 0.5


def test_mrr_undefined_for_empty_golden():
    assert mrr(["https://a"], []) is None
