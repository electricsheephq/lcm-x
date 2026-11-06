"""#2345: doctor apply paths on the store's shared connection must not roll back
or half-commit a concurrent foreground ingest.

Each test runs two real threads. The ingest thread pauses after the first row
of a two-row batch and starts the doctor thread. It resumes once the doctor
thread either asks for the store write lock (fixed code: it then waits for the
ingest to commit) or has finished without it (the unfixed race). Nothing of the
batch may be lost, and no other connection may see half of it.
"""

import sqlite3
import threading
from types import SimpleNamespace

import pytest

from hermes_lcm import command, maintenance
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.db_bootstrap import join_background_stem_backfills
from hermes_lcm.store import MessageStore

_WAIT_S = 30.0
_ROWS = [{"role": "user", "content": "row A: user turn"}, {"role": "assistant", "content": "row B: reply"}]


class _SignalLock:
    """The store's own RLock; flags the doctor thread's first request for it."""

    def __init__(self, inner, race):
        self._inner, self._race = inner, race

    def acquire(self, *args, **kwargs):
        if threading.current_thread() is self._race.doctor_thread:
            self._race.doctor_reached.set()
        return self._inner.acquire(*args, **kwargs)

    def release(self):
        self._inner.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


@pytest.fixture
def race(tmp_path):
    db = tmp_path / "lcm.db"
    store = MessageStore(db)
    dag = SummaryDAG(db)
    for text in ("old one", "old two"):
        store.append("ignored-cron", {"role": "user", "content": text}, token_estimate=2)
    state = SimpleNamespace(doctor_thread=None, doctor_reached=threading.Event(), doctor={}, ingest={})
    store._write_lock = _SignalLock(store._write_lock, state)
    state.store, state.db = store, db
    state.engine = SimpleNamespace(
        _store=store, _dag=dag, _session_id="live", _config=SimpleNamespace(embeddings_enabled=False),
        backup_dir=lambda: tmp_path / "backups",
    )
    yield state
    join_background_stem_backfills(_WAIT_S)
    dag.close()
    store.close()


def _run_race(state, doctor):
    def run_doctor():
        try:
            state.doctor["result"] = doctor(state.engine)
        except Exception as exc:
            state.doctor["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            state.doctor_reached.set()

    def after_first_row(message, store_id):
        if state.doctor_thread is None:
            state.doctor_thread = threading.Thread(target=run_doctor)
            state.doctor_thread.start()
            assert state.doctor_reached.wait(_WAIT_S)
            # Another connection must not see half the batch before the ingest commits.
            state.ingest["visible_mid_batch"] = len(_rows(state.db, "live"))
        return []

    def run_ingest():
        try:
            with state.store.ingest_atomic():
                state.ingest["ids"] = state.store._append_protected_batch(
                    "live", _ROWS, [5, 5], source="t", conversation_id="",
                    metadata_factory=after_first_row, metadata_messages=_ROWS)
        except BaseException as exc:
            state.ingest["error"] = f"{type(exc).__name__}: {exc}"

    ingest = threading.Thread(target=run_ingest)
    ingest.start()
    ingest.join(_WAIT_S)
    assert not ingest.is_alive()
    assert state.doctor_thread is not None
    state.doctor_thread.join(_WAIT_S)
    assert not state.doctor_thread.is_alive()


def _rows(db, session_id):
    with sqlite3.connect(db) as check:
        return check.execute(
            "SELECT store_id, content FROM messages WHERE session_id = ? ORDER BY store_id", (session_id,)).fetchall()


def _assert_ingest_kept(state):
    assert "error" not in state.ingest, state.ingest
    assert state.ingest["visible_mid_batch"] == 0
    ids = state.ingest["ids"]
    assert len(set(ids)) == 2
    assert _rows(state.db, "live") == [(ids[0], _ROWS[0]["content"]), (ids[1], _ROWS[1]["content"])]


def test_doctor_clean_apply_waits_for_a_concurrent_ingest(race):
    _run_race(race, lambda engine: command._delete_clean_candidates_atomically(engine, {"ignored-cron"}))

    _assert_ingest_kept(race)
    assert "error" not in race.doctor, race.doctor
    assert race.doctor["result"]["messages_deleted"] == 2
    assert _rows(race.db, "ignored-cron") == []


def test_doctor_repair_apply_waits_for_a_concurrent_ingest(race):
    _run_race(race, command._doctor_repair_apply_text)

    _assert_ingest_kept(race)
    text = race.doctor.get("result", "")
    assert "status: ok" in text, race.doctor
    assert "backup_path:" in text


def test_doctor_repair_apply_body_waits_without_the_backup_flush(race, monkeypatch):
    # The backup flush already serializes the command above; this isolates the repair body itself.
    monkeypatch.setattr(command, "backup_database", lambda engine: {
        "ok": True, "db_path": race.db, "backup_path": race.db.with_suffix(".bak"), "backup_size": 0})
    _run_race(race, command._doctor_repair_apply_text)

    _assert_ingest_kept(race)
    assert "status: ok" in race.doctor.get("result", ""), race.doctor


def test_backup_flush_does_not_commit_half_an_ingest(race):
    _run_race(race, maintenance.backup_database)

    _assert_ingest_kept(race)
    assert race.doctor["result"]["ok"], race.doctor
    with sqlite3.connect(race.doctor["result"]["backup_path"]) as backup:
        # The backup is taken after the ingest committed: both rows or none, never half the batch.
        assert backup.execute("SELECT COUNT(*) FROM messages WHERE session_id = 'live'").fetchone()[0] == 2


@pytest.mark.parametrize("foreign", ["rollback", "commit"])
def test_ingest_detects_a_foreign_rollback_or_commit(race, foreign):
    # A writer that skips the store write lock (the pre-#2345 doctor shape) ends the ingest's transaction
    # mid-batch. Python opens a new one for the next row, so only the attempt savepoint shows the loss.
    def foreign_end(engine):
        getattr(engine._store._conn, foreign)()

    _run_race(race, foreign_end)

    assert race.ingest.get("error", "").startswith("_SqliteSavepointError: "), race.ingest
    assert "lost before COMMIT" in race.ingest["error"]
    # The owner rolls back what it can; the engine reconciles from a cursor that never moved.
    assert [content for _id, content in _rows(race.db, "live")] == (
        [] if foreign == "rollback" else [_ROWS[0]["content"]])
