"""Read-only access to the etsi-rag SQLite DB for reference content lookup.

Lookups are tolerant of cosmetic URL differences (trailing slash, scheme
case, default ports, fragments). We try exact match first; on miss we scan
the documents table once and match by canonical form. The canonical->row
map is cached per-process so repeated lookups are cheap.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .url_norm import canonicalize


@dataclass(frozen=True)
class ReferenceDocument:
    url: str
    title: str
    content: str


def _connect(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


# Cache: db_path -> {canonical_url: (stored_url, title, content)}
_CANONICAL_INDEX: dict[Path, dict[str, tuple[str, str, str]]] = {}


def _canonical_index(db_path: Path) -> dict[str, tuple[str, str, str]]:
    cached = _CANONICAL_INDEX.get(db_path)
    if cached is not None:
        return cached
    index: dict[str, tuple[str, str, str]] = {}
    with _connect(db_path) as conn:
        for url, title, content in conn.execute("SELECT url, title, content FROM documents"):
            if not url:
                continue
            index[canonicalize(url)] = (url, title or "", content or "")
    _CANONICAL_INDEX[db_path] = index
    return index


def reset_cache() -> None:
    """Drop the canonical-index cache. Test-only hook."""
    _CANONICAL_INDEX.clear()


def fetch_by_url(db_path: Path, url: str) -> ReferenceDocument | None:
    """Look up the indexed document by URL, tolerant to canonicalisation.

    Returns None if no match — callers should surface that as an eval error
    rather than silently pass an empty context to the judge.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT url, title, content FROM documents WHERE url = ? LIMIT 1",
            (url,),
        ).fetchone()
    if row is not None:
        return ReferenceDocument(url=row[0] or url, title=row[1] or "", content=row[2] or "")

    canon = canonicalize(url)
    hit = _canonical_index(db_path).get(canon)
    if hit is None:
        return None
    stored_url, title, content = hit
    return ReferenceDocument(url=stored_url or url, title=title, content=content)
