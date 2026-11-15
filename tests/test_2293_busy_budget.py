"""#2681: implicit admission spends one busy budget; failed cleanup aborts its owner."""
import sqlite3
import time

import pytest

from hermes_lcm.engine import LCMEngine
from hermes_lcm.sqlite_util import _SqliteSavepointError, _sqlite_savepoint
from hermes_lcm.store import MessageStore
from tests.test_2293_prelock import _engine, _pinned_salt, _rows, _shape  # noqa: F401

TIMEOUT_MS = 600


def _run(root, prelock, monkeypatch, *, swallow=False, release_lock=False):
    root.mkdir(parents=True)
    setups, view = _shape("warm-append")
    engine = _engine(root, prelock=prelock)
    waits, begin_timeouts = [], []
    holder = None
    try:
        for setup in setups:
            engine.ingest(setup)
        conn = engine._store._conn
        conn.execute(f"PRAGMA busy_timeout = {TIMEOUT_MS}")
        with monkeypatch.context() as patch:
            original = LCMEngine._identity_anchor_prematch

            def prematch(self, *args, **kwargs):
                if self is engine:
                    started = time.perf_counter()
                    try:
                        if swallow:
                            # Exercise the real fail-open writer, including its failure recorder.
                            self._write_generated_ignored_dependent_reply_records(
                                [{"store": "a" * 16}], keys=["t-2681"])
                        else:
                            conn.execute("INSERT INTO metadata VALUES('t-2681', '1') "
                                         "ON CONFLICT(key) DO UPDATE SET value = value || '1'")
                    finally:
                        waits.append(("implicit-write", (time.perf_counter() - started) * 1000))
                        if release_lock and holder is not None and holder.in_transaction:
                            holder.rollback()
                return original(self, *args, **kwargs)

            patch.setattr(LCMEngine, "_identity_anchor_prematch", prematch)
            original_begin = MessageStore._begin_ingest_prelock

            def begin(store, state):
                started = time.perf_counter()
                try:
                    return original_begin(store, state)
                finally:
                    waits.append(("explicit-BEGIN", (time.perf_counter() - started) * 1000))

            patch.setattr(MessageStore, "_begin_ingest_prelock", begin)

            def trace(sql):
                if sql == "BEGIN IMMEDIATE":
                    begin_timeouts.append(conn.execute("PRAGMA busy_timeout").fetchone()[0])

            conn.set_trace_callback(trace)
            if prelock or not release_lock:
                holder = sqlite3.connect(root / "lcm.db", isolation_level=None)
                holder.execute("BEGIN IMMEDIATE")
            started = time.perf_counter()
            engine.ingest(view)
            total = (time.perf_counter() - started) * 1000
            conn.set_trace_callback(None)
        restored = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        rows = _rows(engine)
        witness = conn.execute("SELECT value FROM metadata WHERE key='t-2681'").fetchall()
        counts = dict(engine._store.ingest_prelock_counts)
    finally:
        if holder is not None:
            holder.rollback()
            holder.close()
        engine.shutdown()
    return dict(total=total, waits=waits, timeouts=begin_timeouts, restored=restored,
                rows=rows, witness=witness, counts=counts)


@pytest.mark.parametrize("swallow", [False, True], ids=["raising", "swallowing"])
def test_busy_budget_is_spent_once(tmp_path, monkeypatch, swallow):
    eager = _run(tmp_path / "eager", False, monkeypatch, swallow=swallow)
    lazy = _run(tmp_path / "lazy", True, monkeypatch, swallow=swallow)
    assert sum(ms > 300 for _, ms in lazy["waits"]) <= 1, lazy
    assert lazy["total"] < eager["total"] + 300, (eager, lazy)
    assert lazy["timeouts"] == [TIMEOUT_MS, 0, 0]
    assert lazy["restored"] == eager["restored"] == TIMEOUT_MS


@pytest.mark.parametrize("swallow", [False, True], ids=["raising", "swallowing"])
def test_free_lock_reacquires_without_waiting(tmp_path, monkeypatch, swallow):
    eager = _run(tmp_path / "eager", False, monkeypatch, swallow=swallow, release_lock=True)
    lazy = _run(tmp_path / "lazy", True, monkeypatch, swallow=swallow, release_lock=True)
    assert lazy["timeouts"] == [TIMEOUT_MS, 0], "reacquisition must use the zero busy wait budget"
    assert sum(ms > 300 for _, ms in lazy["waits"]) == 1, lazy
    assert lazy["restored"] == eager["restored"] == TIMEOUT_MS
    assert lazy["counts"].get("reclassified", 0) == eager["counts"].get("reclassified", 0) + 1
    assert lazy["rows"] == eager["rows"] and all(n == 1 for n in lazy["rows"].values())
    assert lazy["witness"] == eager["witness"] and len(lazy["witness"]) == 1


def test_uncovered_rollback_failure_aborts_ingest(tmp_path):
    store = MessageStore(tmp_path / "rollback.db")
    conn, armed = store._conn, [True]

    def authorize(action, arg1, *_):
        if action == sqlite3.SQLITE_TRANSACTION and arg1 == "ROLLBACK" and armed:
            armed.pop()
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    try:
        conn.set_authorizer(authorize)
        with pytest.raises(_SqliteSavepointError, match="Could not roll back uncovered ingest scope"):
            with store.ingest_atomic(prelock=True):
                with _sqlite_savepoint(conn, abort_on_cleanup_failure=True, owner_active=True,
                                       foreground_ingest=True):
                    conn.execute("INSERT INTO metadata VALUES('partial-evidence', 'x')")
                    raise RuntimeError("evidence writer failed after its first write")
        assert not armed and not conn.in_transaction
        assert conn.execute("SELECT value FROM metadata WHERE key='partial-evidence'").fetchone() is None
    finally:
        conn.set_authorizer(None)
        store.close()
