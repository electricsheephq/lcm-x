"""I1: a busy end preserves occurrences until replay or the same append commits."""

import logging
import sqlite3
import threading
import time

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "pending.db"), embeddings_enabled=False,
        temporal_rollups_enabled=False,
    ), hermes_home=str(tmp_path / "home"))
    yield instance
    instance.shutdown()


def _rows(engine, session="ended"):
    return engine._store.get_session_messages(session)


def _history():
    return [
        {"role": "user", "content": "repeat", "timestamp": 1760000001.0},
        {"role": "assistant", "content": "answer", "timestamp": 1760000002.0},
        {"role": "user", "content": "repeat", "timestamp": 1760000003.0},
        {"role": "assistant", "content": "answer", "timestamp": 1760000004.0},
    ]


def _skip(engine, history):
    engine.on_session_start("ended", conversation_id="ended-conversation", platform="cli")
    engine.ingest(history[:2])
    locker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("ended", history)
    finally:
        locker.rollback()
        locker.close()
    assert len(_rows(engine)) == 2


def _assert_history(engine, history):
    assert [(r["role"], r["content"], r["observed_at"]) for r in _rows(engine)] == [
        (m["role"], m["content"], m["timestamp"]) for m in history
    ]
    assert {r["conversation_id"] for r in _rows(engine)} == {"ended-conversation"}
    assert {r["source"] for r in _rows(engine)} == {"cli"}


@pytest.mark.parametrize("replayed", [0, 1])
def test_replay_then_retry_is_ordered_and_exactly_once(engine, replayed):
    history = _history()
    _skip(engine, history)
    # Ordinary ingest (no lifecycle boundary) may replay part or all of the tail.
    engine.ingest(history[:2 + replayed])
    engine.on_session_start("other", conversation_id="other-conversation", platform="telegram")
    _assert_history(engine, history)
    assert _rows(engine, "other") == []


def test_full_replay_is_an_exactly_once_control(engine):
    """CONTROL: full history replay recovers the rows even without pending tails."""
    history = _history()
    _skip(engine, history)
    engine.ingest(history)
    engine.on_session_start("other", conversation_id="other-conversation", platform="telegram")
    _assert_history(engine, history)
    assert _rows(engine, "other") == []


def test_retry_then_replay_preserves_repeated_occurrences(engine):
    history = _history()
    _skip(engine, history)
    engine.on_session_start("other", platform="telegram")
    _assert_history(engine, history)
    engine.on_session_start("ended", conversation_id="ended-conversation", platform="cli")
    engine.ingest(history)
    engine.ingest(history)
    _assert_history(engine, history)


def test_retry_with_another_bound_session_keeps_identity(engine, monkeypatch):
    history = _history()
    _skip(engine, history)
    # Defer only the ended batch at bind, without locking the unrelated bind's
    # metadata writes. HEAD has no retry and must still reach the identity oracle.
    append = engine._store._append_protected_batch

    def busy_ended(session_id, *args, **kwargs):
        if session_id == "ended":
            raise sqlite3.OperationalError("database is locked")
        return append(session_id, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "_append_protected_batch", busy_ended)
        engine.on_session_start("other", conversation_id="other-conversation", platform="telegram")
    assert engine._session_id == "other"
    assert len(_rows(engine)) == 2
    other = [{"role": "user", "content": "other question"}]
    engine.ingest(other)
    engine.on_session_end("other", other)
    _assert_history(engine, history)
    assert [r["content"] for r in _rows(engine, "other")] == ["other question"]
    assert engine._session_id == "other"


def test_shutdown_bound_and_remaining_count(engine, monkeypatch, caplog):
    history = _history()
    _skip(engine, history)
    # A replay covers one row before the bounded shutdown attempt.
    engine.ingest(history[:3])
    monkeypatch.setattr(engine_module, "_SESSION_END_SHUTDOWN_TIMEOUT_MS", 25, raising=False)
    caplog.clear()  # HEAD warns at end; assert only the shutdown diagnostic here.
    locker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING):
            engine.shutdown()
        elapsed = time.monotonic() - started
        assert elapsed < 0.3
        assert locker.in_transaction  # The bound expires with the writer still held.
        warnings = [r.getMessage() for r in caplog.records if "raw-message ingest skipped" in r.getMessage()]
        assert len(warnings) == 1
        assert "unflushed=1" in warnings[0]
        assert "answer" not in warnings[0]
    finally:
        locker.rollback()
        locker.close()


def test_pending_rows_are_frozen_against_host_mutation(engine):
    history = _history()
    _skip(engine, history)
    history[-1]["content"] = "host rewrote this after end"
    engine.on_session_start("other", platform="cli")
    _assert_history(engine, _history())


@pytest.mark.parametrize("foreign_session", ["ended", "another-session"])
@pytest.mark.parametrize("replayed", [0, 1, 2])
def test_foreign_conversation_replay_does_not_cover_pending_tail(engine, foreign_session, replayed):
    """Identical payloads in B cannot discharge A's pending occurrences (I1/I2)."""
    history = _history()
    _skip(engine, history)
    # Model rows committed by a replay in another conversation before retry.
    # Use the existing store path so this also runs on pre-pending-tail HEAD.
    for message in history[2:]:
        engine._store.append(foreign_session, message, conversation_id="foreign-conversation")
    for message in history[2:2 + replayed]:
        engine._store.append("ended", message, source="cli", conversation_id="ended-conversation")
    engine.on_session_start("other", conversation_id="other-conversation", platform="cli")
    owned = [r for r in _rows(engine) if r["conversation_id"] == "ended-conversation"]
    assert [(r["role"], r["content"], r["observed_at"]) for r in owned] == [
        (m["role"], m["content"], m["timestamp"]) for m in history
    ]
    foreign = [r for r in _rows(engine, foreign_session) if r["conversation_id"] == "foreign-conversation"]
    assert len(foreign) == 2
    assert _rows(engine, "other") == []


def test_exit_lifecycle_lock_is_within_pending_retry_bound(engine, monkeypatch, caplog):
    """The exit hook never waits on a held lifecycle lock; shutdown() keeps its
    existing contract of waiting for active use (test_lcm_engine.py)."""
    history = _history()
    _skip(engine, history)
    monkeypatch.setattr(engine_module, "_SESSION_END_SHUTDOWN_TIMEOUT_MS", 25, raising=False)
    monkeypatch.setattr(engine_module, "_PENDING_TAIL_ENGINES", {engine}, raising=False)
    acquired = threading.Event()
    release = threading.Event()
    released = threading.Event()

    def hold_lifecycle():
        with engine._stable_use_lock:
            acquired.set()
            release.wait(1.0)  # A failing pre-fix run must still terminate.
        released.set()

    holder = threading.Thread(target=hold_lifecycle)
    holder.start()
    assert acquired.wait(1.0)
    embedding_waits = []
    monkeypatch.setattr(engine_module._EMBEDDING_MAINTENANCE_SCHEDULER, "quiesce",
                        lambda timeout: embedding_waits.append(timeout) or True)
    caplog.clear()
    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING):
            engine_module._quiesce_embedding_maintenance()
        elapsed = time.monotonic() - started
        assert elapsed < 0.3
        assert not released.is_set(), "exit must return while the lifecycle lock remains held"
        assert len(_rows(engine)) == 2
        warnings = [r.getMessage() for r in caplog.records if "raw-message ingest skipped" in r.getMessage()]
        assert len(warnings) == 1
        assert "unflushed=2" in warnings[0]
        assert "lifecycle lock" in warnings[0]
        assert "repeat" not in warnings[0] and "answer" not in warnings[0]
        assert embedding_waits == [engine_module._EMBEDDING_EXIT_TIMEOUT_S]
    finally:
        release.set()
        holder.join(timeout=2.0)
    # Timeout must leave the still-live engine usable for a later retry.
    engine.on_session_start("other", platform="cli")
    _assert_history(engine, history)


def test_padded_conversation_id_replay_covers_pending_tail(engine):
    """Stored conversation ids are normalized; a replay under a padded id still covers the tail once."""
    history = _history()
    engine.on_session_start("ended", conversation_id=" ended-conversation ", platform="cli")
    engine.ingest(history[:2])
    locker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("ended", history)
    finally:
        locker.rollback()
        locker.close()
    assert len(_rows(engine)) == 2
    # Ordinary ingest (no lifecycle boundary) replays the tail before the retry runs.
    engine.ingest(history)
    engine.on_session_start("other", conversation_id="other-conversation", platform="telegram")
    _assert_history(engine, history)
