"""Concurrency tests for the parallel runner.

We don't go through real HTTP — instead we monkeypatch the agent and judge
clients with synchronous stubs and verify:
  - parallel execution actually runs more than one row at a time
  - the per-row session id is unique across rows
  - row order in the final list is sorted by id (regardless of completion order)
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from etsi_eval import agent_client, db, judge, runner
from etsi_eval.config import BackendConfig, JudgePricing, Runtime
from etsi_eval.dataset import Question
from etsi_eval.judge import DimScore, Verdict


@pytest.fixture
def runtime(tmp_path: Path) -> Runtime:
    db_path = tmp_path / "etsi.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE documents (url TEXT PRIMARY KEY, title TEXT, content TEXT);
        INSERT INTO documents VALUES
            ('https://www.uma.es/q1', 't1', 'c1'),
            ('https://www.uma.es/q2', 't2', 'c2'),
            ('https://www.uma.es/q3', 't3', 'c3'),
            ('https://www.uma.es/q4', 't4', 'c4');
        """
    )
    conn.commit()
    conn.close()
    db.reset_cache()
    return Runtime(
        agent_backend=BackendConfig(name="agent", base_url="http://x", api_key=""),
        agent_model="agent-model",
        judge_backend=BackendConfig(name="judge", base_url="http://y", api_key=""),
        judge_model="judge-model",
        judge_pricing=JudgePricing(),
        db_path=db_path,
        server_base_url="http://localhost:0",
    )


def _make_questions(n: int) -> list[Question]:
    return [
        Question(
            id=i,
            category="course",
            difficulty="easy",
            question=f"q{i}",
            key_facts=[f"f{i}"],
            golden_urls=[f"https://www.uma.es/q{i}"],
            expected_behavior="answer",
        )
        for i in range(1, n + 1)
    ]


def _stub_agent_and_judge(monkeypatch, sleep_ms: int = 0, sessions: list | None = None):
    """Patch agent_client.ask, judge.grade, and the retrieval probe so the
    runner doesn't hit any network. Capture session ids if requested."""

    def fake_ask(server_base_url, model, question, *, session_id, timeout=120.0, client=None):
        if sessions is not None:
            sessions.append(session_id)
        if sleep_ms:
            time.sleep(sleep_ms / 1000)
        return agent_client.AgentReply(content=f"answer for {question}", raw={})

    def fake_grade(backend, model, question, answer, refs, *, expected_behavior="answer", **kw):
        return Verdict(
            faithfulness=DimScore(score=90, rationale="ok"),
            correctness=DimScore(score=90, rationale="ok"),
            completeness=DimScore(score=90, rationale="ok"),
            raw="{}",
        )

    def fake_query(server_base_url, question, *, top_k=None, timeout=30.0):
        return []

    monkeypatch.setattr(agent_client, "ask", fake_ask)
    monkeypatch.setattr(judge, "grade", fake_grade)
    monkeypatch.setattr("etsi_eval.retrieval_client.query", fake_query)


def test_parallel_runs_faster_than_serial(monkeypatch, runtime: Runtime):
    """Each row simulates ~50ms of work. With 4 rows and concurrency=4, total
    wall time should be closer to 50ms than 200ms."""
    _stub_agent_and_judge(monkeypatch, sleep_ms=50)
    questions = _make_questions(4)

    t0 = time.perf_counter()
    rows = runner._run_parallel(runtime, questions, run_id="r1", retrieval_top_k=10, concurrency=4, tracker=runner._UsageTracker(JudgePricing()))
    elapsed = time.perf_counter() - t0

    assert len(rows) == 4
    # 4 rows * 50ms each * 2 calls (agent + judge stub instant) = 200ms serial.
    # With concurrency=4 we expect well under 150ms; pad for CI noise.
    assert elapsed < 0.15, f"parallel took {elapsed:.3f}s, expected <0.15s"


def test_parallel_preserves_id_order_in_output(monkeypatch, runtime: Runtime):
    """Completion order is non-deterministic under threads; the runner must
    sort rows by id before returning them."""
    _stub_agent_and_judge(monkeypatch, sleep_ms=10)
    questions = _make_questions(4)

    rows = runner._run_parallel(runtime, questions, run_id="r1", retrieval_top_k=10, concurrency=4, tracker=runner._UsageTracker(JudgePricing()))
    assert [r.id for r in rows] == [1, 2, 3, 4]


def test_parallel_session_ids_are_unique_per_row(monkeypatch, runtime: Runtime):
    sessions: list[str] = []
    _stub_agent_and_judge(monkeypatch, sleep_ms=5, sessions=sessions)
    questions = _make_questions(4)

    runner._run_parallel(runtime, questions, run_id="abc12345", retrieval_top_k=10, concurrency=4, tracker=runner._UsageTracker(JudgePricing()))

    assert len(sessions) == 4
    assert len(set(sessions)) == 4, "every row must get a distinct session id"
    assert all(s.startswith("eval-abc12345-") for s in sessions)


def test_parallel_actually_overlaps_calls(monkeypatch, runtime: Runtime):
    """Two threads should be in-flight simultaneously — verify by counting
    the high-water mark of concurrent agent calls."""
    in_flight = 0
    high_water = 0
    lock = threading.Lock()

    def fake_ask(server_base_url, model, question, *, session_id, timeout=120.0, client=None):
        nonlocal in_flight, high_water
        with lock:
            in_flight += 1
            high_water = max(high_water, in_flight)
        time.sleep(0.03)
        with lock:
            in_flight -= 1
        return agent_client.AgentReply(content="ok", raw={})

    def fake_grade(*a, **kw):
        return Verdict(
            faithfulness=DimScore(score=90, rationale="ok"),
            correctness=DimScore(score=90, rationale="ok"),
            completeness=DimScore(score=90, rationale="ok"),
            raw="{}",
        )

    monkeypatch.setattr(agent_client, "ask", fake_ask)
    monkeypatch.setattr(judge, "grade", fake_grade)
    monkeypatch.setattr("etsi_eval.retrieval_client.query", lambda *a, **kw: [])

    questions = _make_questions(4)
    runner._run_parallel(runtime, questions, run_id="r1", retrieval_top_k=10, concurrency=4, tracker=runner._UsageTracker(JudgePricing()))

    assert high_water >= 2, f"expected >=2 concurrent agent calls, saw {high_water}"


def test_serial_path_still_works(monkeypatch, runtime: Runtime, capsys):
    _stub_agent_and_judge(monkeypatch, sleep_ms=0)
    questions = _make_questions(2)

    rows = runner._run_serial(runtime, questions, run_id="r1", retrieval_top_k=10, tracker=runner._UsageTracker(JudgePricing()))
    assert [r.id for r in rows] == [1, 2]
    assert all(r.overall == 90 for r in rows)
