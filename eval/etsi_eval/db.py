"""Read-only access to the etsi-rag SQLite DB for reference content lookup."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceDocument:
    url: str
    title: str
    content: str


def _connect(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def fetch_by_url(db_path: Path, url: str) -> ReferenceDocument | None:
    """Look up the indexed document whose canonical URL matches.

    Returns None if no match — callers should surface that as an eval error
    rather than silently pass an empty context to the judge.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT url, title, content FROM documents WHERE url = ? LIMIT 1",
            (url,),
        ).fetchone()
    if row is None:
        return None
    return ReferenceDocument(url=row[0] or url, title=row[1] or "", content=row[2] or "")
