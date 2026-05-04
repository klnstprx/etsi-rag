"""DB lookup tests — exact match plus canonical-form fallback."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from etsi_eval import db


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    p = tmp_path / "etsi.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE documents (url TEXT PRIMARY KEY, title TEXT, content TEXT);
        INSERT INTO documents VALUES
            ('https://www.uma.es/foo', 'Foo', 'foo body'),
            ('https://www.uma.es/bar/', 'Bar (trailing slash in DB)', 'bar body');
        """
    )
    conn.commit()
    conn.close()
    db.reset_cache()
    yield p
    db.reset_cache()


def test_exact_match(populated_db: Path):
    ref = db.fetch_by_url(populated_db, "https://www.uma.es/foo")
    assert ref is not None
    assert ref.title == "Foo"
    assert ref.content == "foo body"


def test_canonical_fallback_trailing_slash(populated_db: Path):
    """Dataset URL has no trailing slash; DB stored one. Should still resolve."""
    ref = db.fetch_by_url(populated_db, "https://www.uma.es/bar")
    assert ref is not None
    assert ref.title.startswith("Bar")
    # The returned url echoes what the DB actually stored, not the query.
    assert ref.url == "https://www.uma.es/bar/"


def test_canonical_fallback_case_and_fragment(populated_db: Path):
    ref = db.fetch_by_url(populated_db, "HTTPS://WWW.UMA.ES/foo#section")
    assert ref is not None
    assert ref.title == "Foo"


def test_miss_returns_none(populated_db: Path):
    assert db.fetch_by_url(populated_db, "https://www.uma.es/nope") is None
