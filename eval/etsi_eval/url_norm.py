"""URL canonicalisation for ground-truth matching.

The crawler and the eval dataset are curated independently, and small cosmetic
differences (trailing slash, scheme case, default ports, fragments) can cause
exact string matching to fail silently — a missed match shows up as Recall=0
or "reference not found in DB", with no obvious cause.

We normalise on a small, conservative set of axes that are known to be
semantically irrelevant for UMA URLs:

- lowercase scheme and host (RFC 3986: scheme/host are case-insensitive)
- strip default ports (:80 for http, :443 for https)
- drop fragment (#... is client-side only)
- collapse a single trailing slash on the path

Query strings are left alone — UMA pages don't use them meaningfully and
sorting them could change semantics on services that we haven't audited.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

_DEFAULT_PORTS = {"http": "80", "https": "443"}


def canonicalize(url: str) -> str:
    """Return a canonical form of url. Pure function, idempotent."""
    if not url:
        return ""
    parts = urlsplit(url.strip())

    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    host = host.lower()

    # Drop default port for the scheme; keep non-default ports.
    netloc = host
    if parts.port is not None and str(parts.port) != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"

    # Preserve userinfo if (rarely) present.
    if parts.username:
        userinfo = parts.username
        if parts.password:
            userinfo = f"{userinfo}:{parts.password}"
        netloc = f"{userinfo}@{netloc}"

    path = parts.path or ""
    if path.endswith("/"):
        path = path[:-1]

    # Drop fragment; keep query as-is.
    return urlunsplit((scheme, netloc, path, parts.query, ""))
