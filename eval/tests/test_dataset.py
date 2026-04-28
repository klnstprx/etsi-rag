"""Verify the shipped dataset.jsonl loads cleanly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from etsi_eval.dataset import Question, load


def test_dataset_loads():
    path = Path(__file__).resolve().parent.parent / "dataset.jsonl"
    rows = load(path)
    assert rows, "dataset is empty"
    ids = [r.id for r in rows]
    assert ids == sorted(set(ids)), "ids should be unique and monotonic"
    allowed_difficulty = {"easy", "medium", "hard"}
    allowed_behavior = {"answer", "refuse"}
    for r in rows:
        assert r.difficulty in allowed_difficulty
        assert r.expected_behavior in allowed_behavior
        if r.expected_behavior == "answer":
            assert r.golden_urls, f"row {r.id} expects answer but has no golden_urls"
            for u in r.golden_urls:
                assert u.startswith("http"), f"row {r.id} bad URL: {u}"
        else:
            assert r.golden_urls == [], (
                f"row {r.id} expects refusal but has golden_urls"
            )
        assert r.key_facts, f"row {r.id} has no key_facts"


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_default_expected_behavior_is_answer(tmp_path: Path):
    """Rows without expected_behavior default to 'answer' for back-compat."""
    p = _write_jsonl(
        tmp_path,
        [
            {
                "id": 1,
                "category": "course",
                "difficulty": "easy",
                "question": "q",
                "key_facts": ["f"],
                "golden_urls": ["https://example.com/a"],
            }
        ],
    )
    rows = load(p)
    assert rows[0].expected_behavior == "answer"
    assert rows[0].notes == ""


def test_refuse_row_with_empty_golden_urls(tmp_path: Path):
    p = _write_jsonl(
        tmp_path,
        [
            {
                "id": 1,
                "category": "general",
                "difficulty": "easy",
                "question": "Out of corpus",
                "key_facts": ["should refuse"],
                "expected_behavior": "refuse",
                "golden_urls": [],
                "notes": "info not in DB",
            }
        ],
    )
    rows = load(p)
    assert rows[0].expected_behavior == "refuse"
    assert rows[0].golden_urls == []
    assert rows[0].notes == "info not in DB"


def test_invalid_expected_behavior_rejected(tmp_path: Path):
    p = _write_jsonl(
        tmp_path,
        [
            {
                "id": 1,
                "category": "x",
                "difficulty": "easy",
                "question": "q",
                "key_facts": ["f"],
                "golden_urls": ["https://example.com/a"],
                "expected_behavior": "ignore",
            }
        ],
    )
    with pytest.raises(ValueError, match="expected_behavior"):
        load(p)


def test_legacy_source_url_field_migrated(tmp_path: Path):
    """Old-format rows with source_url get auto-promoted to golden_urls=[that]."""
    p = _write_jsonl(
        tmp_path,
        [
            {
                "id": 1,
                "category": "course",
                "difficulty": "easy",
                "question": "q",
                "key_facts": ["f"],
                "source_url": "https://example.com/a",
            }
        ],
    )
    rows = load(p)
    assert rows[0].golden_urls == ["https://example.com/a"]


def test_question_helper_primary_url():
    q = Question(
        id=1,
        category="x",
        difficulty="easy",
        question="q",
        key_facts=["f"],
        golden_urls=["https://example.com/a", "https://example.com/b"],
        expected_behavior="answer",
        notes="",
    )
    assert q.primary_url == "https://example.com/a"

    refuse_q = Question(
        id=2,
        category="x",
        difficulty="easy",
        question="q",
        key_facts=["f"],
        golden_urls=[],
        expected_behavior="refuse",
        notes="",
    )
    assert refuse_q.primary_url is None
