"""#2736 stays open: a busy session-end flush whose rollback also fails loses the session's final rows.

At a foreground session end, a busy error inside the end flush's write transaction normally retries once, then
freezes the tail and queues it, so the final rows are stored, queued or counted. When the rollback after the busy
error also raises, ``rollback_ok`` is false, so the retry, the frozen re-run and the cold-tail queue are all skipped:
the end only logs that final messages may be absent, and the final rows (C, D) live only in the host's state. No later
path in the same process (tail retry, next session, shutdown) stores them; only a resume that re-sends the full
history does. These strict xfails pin that loss; a fix flips them. The working-rollback end and the resume replay are
controls that pass today.
"""

import copy
import sqlite3

import pytest

from tests.invariants.test_p1e_decided_tails import contents, history
from tests.test_issue_2055_ingest_ledger_writes import engines as engines

_PIN = pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#2736: a busy session-end flush whose rollback also fails neither stores nor queues the final rows",
)


def _require(condition, what):
    # Setup checks raise pytest's Failed, never AssertionError, so the strict xfails cannot absorb them.
    if not condition:
        pytest.fail(f"setup: {what}")


class RollbackProxy:
    """Runs the real ROLLBACK, then raises: the transaction is gone, but Python saw a failure."""

    def __init__(self, conn):
        self.conn = conn

    def __getattr__(self, name):
        return getattr(self.conn, name)

    def rollback(self):
        self.conn.rollback()
        raise RuntimeError("rollback sentinel")


def busy_append(engine, monkeypatch):
    """A busy error after the attempt's rows are written inside the transaction."""
    real = engine._append_ingest_batch

    def append(batch):
        real(batch)  # A frozen (read-only) re-run raises here, before any write.
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine, "_append_ingest_batch", append)


def tail_contents(engine):
    return [m.get("content") for tail in engine._pending_session_tails
            for m in (tail.get("view") or tail.get("messages") or [])]


def disk_contents(engine, path):
    with sqlite3.connect(path) as conn:
        return [r[0] for r in conn.execute("SELECT content FROM messages WHERE session_id='S' ORDER BY store_id")]


def end_with_busy(engine, monkeypatch, *, rollback_fails):
    view = history()
    engine.ingest(copy.deepcopy(view[:2]))
    _require(contents(engine) == ["A", "B"], "the first ingest stored A and B")
    conn = engine._store._conn
    with monkeypatch.context() as patch:
        busy_append(engine, patch)
        if rollback_fails:
            engine._store._conn = RollbackProxy(conn)
        try:
            engine.on_session_end("S", copy.deepcopy(view))
        finally:
            engine._store._conn = conn
    if rollback_fails:
        _require(engine._store._transaction_owner.rollback_failures >= 1, "the end's rollback failed")
    return view


def stored_or_queued(engine, rows):
    stored, queued = contents(engine), tail_contents(engine)
    return all(r in stored or r in queued for r in rows)


def test_control_busy_end_with_working_rollback_queues_the_tail(engines, monkeypatch):
    engine = engines()
    end_with_busy(engine, monkeypatch, rollback_fails=False)
    assert engine._store._transaction_owner.rollback_failures == 0
    assert stored_or_queued(engine, ["C", "D"]), (contents(engine), engine._pending_session_tails)


@_PIN
def test_busy_end_with_failed_rollback_stores_queues_or_counts_final_rows(engines, monkeypatch):
    """Correct engine: C and D are stored, or queued for retry."""
    engine = engines()
    end_with_busy(engine, monkeypatch, rollback_fails=True)
    assert stored_or_queued(engine, ["C", "D"]), (
        f"final rows neither stored nor queued: stored={contents(engine)} queued={tail_contents(engine)}")


@pytest.mark.parametrize("later", [
    pytest.param("retry_tails", marks=_PIN),
    pytest.param("next_session", marks=_PIN),
    pytest.param("shutdown", marks=_PIN),
    "resume_replay",  # control: a resume that re-sends the full history stores C and D
])
def test_later_paths_after_failed_rollback_end(engines, monkeypatch, later):
    """Does any later path store C and D? Asserts the correct outcome."""
    engine = engines()
    view = end_with_busy(engine, monkeypatch, rollback_fails=True)
    path = engine._store.db_path
    if later == "retry_tails":
        engine._retry_pending_session_tails()
    elif later == "next_session":
        engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv")
        engine.ingest([{"role": "user", "content": "E", "timestamp": 10.0}])
    elif later == "shutdown":
        engine.shutdown()
        stored = disk_contents(engine, path)
        # After shutdown only the rows on disk survive; a tail still held in memory dies with the process.
        assert all(r in stored for r in ["C", "D"]), stored
        return
    elif later == "resume_replay":
        engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
        engine.ingest(copy.deepcopy(view) + [{"role": "user", "content": "E", "timestamp": 10.0}])
    assert stored_or_queued(engine, ["C", "D"]), (later, contents(engine), tail_contents(engine))
