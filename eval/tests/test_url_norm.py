"""Pure-function tests for the URL canonicaliser."""

from __future__ import annotations

import pytest

from etsi_eval.url_norm import canonicalize


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Identity for already-canonical URLs.
        ("https://www.uma.es/foo", "https://www.uma.es/foo"),
        # Trailing slash collapsed.
        ("https://www.uma.es/foo/", "https://www.uma.es/foo"),
        # Root path keeps its slash (or empty).
        ("https://www.uma.es/", "https://www.uma.es"),
        # Scheme + host case-folded.
        ("HTTPS://WWW.UMA.ES/Foo", "https://www.uma.es/Foo"),
        # Fragment dropped.
        ("https://www.uma.es/foo#section", "https://www.uma.es/foo"),
        ("https://www.uma.es/foo/#section", "https://www.uma.es/foo"),
        # Default ports stripped.
        ("https://www.uma.es:443/foo", "https://www.uma.es/foo"),
        ("http://www.uma.es:80/foo", "http://www.uma.es/foo"),
        # Non-default port preserved.
        ("http://www.uma.es:8080/foo", "http://www.uma.es:8080/foo"),
        # Query preserved unchanged.
        ("https://www.uma.es/foo?a=1&b=2", "https://www.uma.es/foo?a=1&b=2"),
        # Whitespace trimmed.
        ("  https://www.uma.es/foo  ", "https://www.uma.es/foo"),
    ],
)
def test_canonicalize(raw: str, expected: str):
    assert canonicalize(raw) == expected


def test_canonicalize_is_idempotent():
    once = canonicalize("HTTPS://WWW.UMA.ES/Foo/#bar")
    twice = canonicalize(once)
    assert once == twice


def test_canonicalize_empty():
    assert canonicalize("") == ""
