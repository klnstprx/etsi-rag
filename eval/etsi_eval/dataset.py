"""Load and filter the eval dataset from JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .url_norm import canonicalize

_ALLOWED_BEHAVIOR = {"answer", "refuse"}


@dataclass(frozen=True)
class Question:
    id: int
    category: str
    difficulty: str
    question: str
    key_facts: list[str]
    golden_urls: list[str]
    expected_behavior: str = "answer"
    notes: str = ""

    @property
    def primary_url(self) -> str | None:
        """First golden URL, or None for refusal questions."""
        return self.golden_urls[0] if self.golden_urls else None


def _coerce_row(raw: dict, lineno: int) -> Question:
    """Build a Question from one JSON object, handling legacy source_url."""
    # Legacy: pre-Step-3 rows used source_url (string) instead of golden_urls.
    if "golden_urls" not in raw and "source_url" in raw:
        raw = {**raw, "golden_urls": [raw.pop("source_url")]}

    behavior = raw.get("expected_behavior", "answer")
    if behavior not in _ALLOWED_BEHAVIOR:
        raise ValueError(
            f"line {lineno}: expected_behavior={behavior!r} is not one of "
            f"{sorted(_ALLOWED_BEHAVIOR)}"
        )

    golden_urls = [canonicalize(u) for u in (raw.get("golden_urls") or [])]
    return Question(
        id=raw["id"],
        category=raw["category"],
        difficulty=raw["difficulty"],
        question=raw["question"],
        key_facts=list(raw.get("key_facts") or []),
        golden_urls=golden_urls,
        expected_behavior=behavior,
        notes=str(raw.get("notes", "")),
    )


def load(path: Path) -> list[Question]:
    rows: list[Question] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            rows.append(_coerce_row(raw, lineno))
    return rows


def filter_questions(
    questions: list[Question],
    *,
    category: str | None = None,
    difficulty: str | None = None,
    ids: set[int] | None = None,
) -> list[Question]:
    def keep(q: Question) -> bool:
        if category and q.category != category:
            return False
        if difficulty and q.difficulty != difficulty:
            return False
        if ids and q.id not in ids:
            return False
        return True

    return [q for q in questions if keep(q)]
