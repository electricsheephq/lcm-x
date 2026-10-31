"""#1127: a pre-leaf phase that runs past the hard bound names where its time went.

Engine-level: a host list in, a second raw connection on the same store holding the write lock for a
real second or more, the log lines out. Summaries are stubbed; the hard bound is configured small, so
real time crosses it."""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from contextlib import nullcontext

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
LINE = "LCM pre-leaf phase ran past the hard bound"


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


@pytest.fixture
def summaries(monkeypatch):
    calls: list[str] = []

    def summarize(**kwargs):
        calls.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return calls


def _engine(tmp_path, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": True, "max_assembly_tokens": 100_000,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _view(turns: int = 6) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[row for i in range(turns) for row in _turn(f"T{i}", 10.0 * (i + 1))]]


class _Writer:
    """Another connection on the same store: holds a write transaction for ``hold`` seconds."""

    def __init__(self, db_path, hold: float):
        self.held, self.done = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._run, args=(str(db_path), hold), daemon=True)

    def _run(self, path, hold):
        conn = sqlite3.connect(path, timeout=30, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            self.held.set()
            self.done.wait(hold)
            conn.execute("COMMIT")
        finally:
            conn.close()

    def __enter__(self):
        self._thread.start()
        assert self.held.wait(10)
        return self

    def __exit__(self, *exc):
        self.done.set()
        self._thread.join(10)


def _lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if LINE in record.getMessage()]


def _seconds(line: str, name: str) -> float:
    match = re.search(rf"\b{name}=(\d+\.\d)s", line)
    assert match, line
    return float(match.group(1))


@pytest.mark.parametrize("sweep", [True, False])
def test_a_held_write_lock_is_attributed_to_the_slow_statement(tmp_path, summaries, caplog, sweep):
    """The ingest INSERT waits ~1.2 s on the other connection's write lock; the hard bound is 0.5 s."""
    engine = _engine(tmp_path, foreground_hard_seconds=0.5, threshold_full_sweep_enabled=sweep)
    view = _view()
    try:
        engine.ingest(view)
        grown = [*view, *_turn("N1", 100.0), *_turn("N2", 110.0)]  # new rows: compress() writes them
        with _Writer(tmp_path / "lcm.db", hold=1.2), caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            engine.compress(grown, current_tokens=engine.threshold_tokens + 1)
        lines = _lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        # The sweep stops at its deadline before the first leaf; with the sweep off the first leaf call ends it.
        assert "(budget 0.5s) reached=" + ("stopped" if sweep else "first_leaf") + ";" in line
        assert _seconds(line, "total") >= 1.0
        assert _seconds(line, "pre_loop") >= 1.0 and _seconds(line, "ingest") >= 1.0
        assert "unaccounted=" in line and "lock_wait=" in line
        match = re.search(r"slow_sql=(\d+)/(\d+\.\d)s slowest=(\d+\.\d)s \"([^\"]*)\"", line)
        assert match, line
        assert int(match.group(1)) == 1 and float(match.group(2)) >= 1.0
        assert match.group(4).startswith("INSERT INTO messages") and len(match.group(4)) <= 40
        assert "user turn" not in line and "alpha" not in line  # a template and figures, never content
        # The timing wrapper is gone once the phase ends.
        for part in (engine._store, engine._dag, engine._lifecycle):
            assert type(part._conn) is sqlite3.Connection
    finally:
        engine.shutdown()


def test_a_pre_leaf_phase_inside_the_bound_logs_nothing(tmp_path, summaries, caplog):
    engine = _engine(tmp_path)
    view = _view()
    try:
        engine.ingest(view)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            engine.compress([*view, *_turn("N1", 100.0)], current_tokens=engine.threshold_tokens + 1)
        assert summaries and _lines(caplog) == []
        assert type(engine._store._conn) is sqlite3.Connection
    finally:
        engine.shutdown()


def test_statements_of_other_threads_are_not_counted(tmp_path, summaries, caplog):
    """A background thread's write on the same connection waits out the held lock: the phase is long, but
    that slow statement is not this compaction's, so none is counted."""
    engine = _engine(tmp_path, foreground_hard_seconds=0.5)
    view = _view()
    original = engine._ingest_messages
    background = []

    def background_write_then_ingest(messages, **kwargs):
        if background:  # the first ingest of compress() only
            return original(messages, **kwargs)

        def other():
            with engine._store._write_lock, engine._store._conn as conn:
                conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('test_1127', 'x')")
                background.append(1)
        thread = threading.Thread(target=other)
        thread.start()
        thread.join(10)
        return original(messages, **kwargs)

    try:
        engine.ingest(view)
        engine._ingest_messages = background_write_then_ingest
        grown = [*view, *_turn("N1", 100.0)]
        with _Writer(tmp_path / "lcm.db", hold=1.2), caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            engine.compress(grown, current_tokens=engine.threshold_tokens + 1)
        lines = _lines(caplog)
        assert background == [1] and len(lines) == 1, lines
        assert _seconds(lines[0], "total") >= 1.0 and "slow_sql=0/0.0s" in lines[0], lines[0]
    finally:
        engine.shutdown()


def test_an_ingest_that_raises_on_the_lock_still_records_its_span(tmp_path, summaries, caplog):
    """Review r1: the ingest waits ~1.2 s on the held lock and then raises "database is locked"; the line
    still puts that wait in the ingest span, not only in the slow-statement count."""
    engine = _engine(tmp_path, foreground_hard_seconds=0.5)
    view = _view()
    try:
        engine.ingest(view)
        engine._store._conn.execute("PRAGMA busy_timeout=1200")  # the store's statements give up after 1.2 s
        grown = [*view, *_turn("N1", 100.0)]
        with _Writer(tmp_path / "lcm.db", hold=3.0), caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                engine.compress(grown, current_tokens=engine.threshold_tokens + 1)
        lines = _lines(caplog)
        assert len(lines) == 1, lines
        line = lines[0]
        assert re.search(r"slow_sql=1/1\.\ds", line), line
        assert _seconds(line, "ingest") >= 1.0 and _seconds(line, "pre_loop") >= _seconds(line, "ingest"), line
        assert _seconds(line, "unaccounted") >= 0.0
        assert type(engine._store._conn) is sqlite3.Connection
    finally:
        engine.shutdown()


class _SlowExit:
    """A connection stand-in whose ``with`` exit (the implicit COMMIT or ROLLBACK) takes 1.5 s on a fake clock."""

    def __init__(self, clock: list[float]):
        self.clock = clock

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.clock[0] += 1.5
        return False


@pytest.mark.parametrize("raised,sql", [(False, "COMMIT"), (True, "ROLLBACK")])
def test_a_slow_with_block_exit_is_a_slow_statement(monkeypatch, raised, sql):
    import hermes_lcm.sqlite_util as sqlite_util

    clock = [100.0]
    monkeypatch.setattr(sqlite_util, "_monotonic", lambda: clock[0])
    trace = sqlite_util.PhaseSqlTrace()
    proxy = sqlite_util._TimedConnection(_SlowExit(clock), trace)
    with pytest.raises(RuntimeError) if raised else nullcontext():
        with proxy:
            if raised:
                raise RuntimeError("the block failed")
    assert (trace.slow_count, trace.slow_seconds, trace.slowest_sql) == (1, 1.5, sql)
