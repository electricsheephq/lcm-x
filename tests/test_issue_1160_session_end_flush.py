"""#1160: bounded session-end retry and shutdown diagnostics."""

from contextlib import contextmanager
import logging
import sqlite3

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engines(tmp_path):
    instances = []

    def create():
        instance = LCMEngine(config=LCMConfig(
            database_path=str(tmp_path / "flush.db"), temporal_rollups_enabled=False,
        ))
        instances.append(instance)
        return instance

    yield create
    for instance in reversed(instances):
        instance.shutdown()


def _contents(engine, session_id="S"):
    return [row["content"] for row in engine._store.get_session_messages(session_id)]


def _locked_flush(engine, caplog):
    engine.on_session_start("S", platform="cli")
    messages = [
        {"role": "user", "content": "stored question"},
        {"role": "assistant", "content": "stored answer"},
        {"role": "user", "content": "private pending tail"},
    ]
    engine.ingest(messages[:2])
    assert engine._ingest_cursor == 2
    locker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with caplog.at_level(logging.WARNING):
            engine.on_session_end("S", messages)
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    assert "retry pending at the next session start or shutdown" in caplog.text
    assert "may be absent" not in caplog.text
    assert _contents(engine) == [message["content"] for message in messages[:2]]
    return messages


def test_locked_flush_reports_pending_count_without_content(engines, monkeypatch, caplog):
    """Warn only after shutdown's bounded retry still cannot store the tail."""
    engine = engines()
    messages = _locked_flush(engine, caplog)
    monkeypatch.setattr(engine_module, "_SESSION_END_SHUTDOWN_TIMEOUT_MS", 20)
    caplog.clear()  # Assert the shutdown diagnostic, not the end notice.
    locker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with caplog.at_level(logging.WARNING):
            engine.shutdown()
    finally:
        locker.rollback()
        locker.close()
    assert "raw-message ingest skipped due to SQLite lock" in caplog.text
    assert "unflushed=1" in caplog.text
    assert all(message["content"] not in caplog.text for message in messages)


def test_resume_recovers_skipped_tail_exactly_once(engines, caplog):
    """CONTROL: same-session resume reconciles the missing tail without duplicates."""
    original = engines()
    messages = _locked_flush(original, caplog)
    original.shutdown()
    resumed = engines()
    resumed.on_session_start("S", platform="cli")
    resumed.ingest(messages)
    resumed.ingest(messages)
    assert _contents(resumed) == [message["content"] for message in messages]


def test_never_resumed_session_is_flushed_at_shutdown(engines, caplog):
    """A graceful shutdown drains the tail even when the session never resumes."""
    original = engines()
    messages = _locked_flush(original, caplog)
    original.shutdown()
    next_engine = engines()
    next_engine.on_session_start("new-session", platform="cli")
    next_engine.ingest([{"role": "user", "content": "new session question"}])
    assert _contents(next_engine) == [message["content"] for message in messages]
    assert _contents(next_engine, "new-session") == ["new session question"]


def test_no_pending_tail_does_not_warn(engines, caplog):
    """CONTROL: a successful end with no pending rows has no unflushed=0 warning."""
    engine = engines()
    engine.on_session_start("S", platform="cli")
    messages = [{"role": "user", "content": "already stored"}]
    engine.ingest(messages)
    with caplog.at_level(logging.WARNING):
        engine.on_session_end("S", messages)
    assert "session-end" not in caplog.text
    assert "unflushed=0" not in caplog.text


def test_interrupted_flush_uses_pre_attempt_count(engines, monkeypatch, caplog):
    """REGRESSION: interruption reports the count captured before ingest started."""
    engine = engines()
    engine.on_session_start("S", platform="cli")

    def interrupted(messages, **kwargs):
        engine._ingest_cursor = len(messages)
        raise KeyboardInterrupt()

    monkeypatch.setattr(engine, "_ingest_messages", interrupted)
    with caplog.at_level(logging.WARNING):
        engine.on_session_end("S", [{"role": "user", "content": "private pending tail"}])
    assert "raw-message ingest interrupted" in caplog.text
    assert "unflushed=1" in caplog.text
    assert "private pending tail" not in caplog.text


@pytest.mark.parametrize("error", [KeyboardInterrupt, sqlite3.OperationalError])
@pytest.mark.parametrize("after_ingest", [False, True])
def test_outer_warning_count_only_before_ingest(engines, monkeypatch, caplog, error, after_ingest):
    """REGRESSION before ingest; CONTROL after ingest: no stale at-risk count."""
    engine = engines()
    engine.on_session_start("S", platform="cli")

    @contextmanager
    def failing_timeout(*args):
        if after_ingest:
            yield
        raise error("database is locked")

    monkeypatch.setattr(engine_module, "_temporary_sqlite_busy_timeout", failing_timeout)
    with caplog.at_level(logging.WARNING):
        engine.on_session_end("S", [{"role": "user", "content": "private pending tail"}])
    assert "session-end ingest/finalize" in caplog.text
    if after_ingest:
        assert "unflushed=" not in caplog.text
        assert _contents(engine) == ["private pending tail"]
    else:
        assert "unflushed=1" in caplog.text
        assert _contents(engine) == []
    assert "private pending tail" not in caplog.text
