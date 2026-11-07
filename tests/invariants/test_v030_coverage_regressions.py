"""v0.30.0 gate coverage: row-level regressions for fixes that shipped after v0.29.0.

Each case uses only APIs present in both v0.29.0 and v0.30.0, so it runs on both
trees: RED on v0.29.0 for the stored-row behaviour, GREEN on v0.30.0 defaults.
"""

import copy
import sqlite3
from contextlib import contextmanager

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore
from tests import test_issue_534_recovery_descriptors as r534


@pytest.fixture
def make(tmp_path):
    made = []

    def create(name="case"):
        engine = LCMEngine(config=LCMConfig(
            database_path=str(tmp_path / name / "lcm.db"), embeddings_enabled=False,
            temporal_rollups_enabled=False,
        ), hermes_home=str(tmp_path / name / "home"))
        made.append(engine)
        return engine

    yield create
    for engine in reversed(made):
        engine.shutdown()


def _history():
    return [{"role": role, "content": text, "timestamp": float(i + 1)}
            for i, (role, text) in enumerate((("user", "A"), ("assistant", "B"),
                                              ("user", "C"), ("assistant", "D")))]


def _contents(engine, session="S"):
    return [row["content"] for row in engine._store.get_session_messages(session)]


@contextmanager
def _writer_held(engine):
    """Another connection holds the write lock across the call."""
    locker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        yield
    finally:
        locker.rollback()
        locker.close()


def _start(engine):
    engine.on_session_start("S", conversation_id="conv", platform="cli")
    engine.ingest(copy.deepcopy(_history()[:2]))
    assert _contents(engine) == ["A", "B"]


# -- #1440 item 1 ---------------------------------------------------------------------------------------

def test_1440_replay_committed_after_pending_match_is_stored_once(make, monkeypatch):
    """Another writer commits C and D after the retry's first match read and before its append.

    v0.29.0 trimmed the tail in one read and appended in a separate transaction, so C and D
    were stored twice. The fix re-matches inside the append's transaction.
    """
    engine = make()
    _start(engine)
    with _writer_held(engine):
        engine.on_session_end("S", copy.deepcopy(_history()))
    assert _contents(engine) == ["A", "B"]
    assert len(engine._pending_session_tails) == 1
    store, read, fired = engine._store, engine._store.get_range, []

    def racing(*args, **kwargs):
        rows = read(*args, **kwargs)
        if not fired:  # The first store read of the retry; commit the replay right after it.
            fired.append(True)
            other = MessageStore(store.db_path)
            try:
                for message in _history()[2:]:
                    other.append("S", message, source="cli", conversation_id="conv")
            finally:
                other.close()
        return rows

    with monkeypatch.context() as patch:
        patch.setattr(store, "get_range", racing)
        engine._retry_pending_session_tails()
    assert fired == [True]
    assert _contents(engine) == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails


# -- #1805 item 1 ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("read", ["get_session_count", "get_session_tail"])
def test_1805_locked_pre_append_read_keeps_final_messages(make, monkeypatch, read):
    """A locked count or tail read before the session-end append used to drop C and D."""
    engine = make()
    _start(engine)
    engine._ingest_cursor_needs_reconcile = True
    calls = []

    def locked(*args, **kwargs):
        calls.append(read)
        raise sqlite3.OperationalError("database is locked")

    with _writer_held(engine), monkeypatch.context() as patch:
        patch.setattr(engine._store, read, locked)
        engine.on_session_end("S", copy.deepcopy(_history()))
    assert calls, "the session end must reach the failing read"
    engine._retry_pending_session_tails()
    rows = engine._store.get_session_messages("S")
    assert [row["content"] for row in rows] == ["A", "B", "C", "D"]
    assert [row["observed_at"] for row in rows] == [1.0, 2.0, 3.0, 4.0]
    assert not engine._pending_session_tails


# -- #534 residual: the recovery exit write fails, then a cold restart ----------------------------------

def _relations(conn, store_id):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_relations'").fetchone():
        return 0  # v0.29.0 has no relation table on this path
    return conn.execute("SELECT count(*) FROM message_relations WHERE store_id=?", (store_id,)).fetchone()[0]


@pytest.mark.parametrize("merged", [False, True], ids=["separate", "host-merged"])
def test_534_recovery_rows_after_failed_exit_write_are_not_restored_silently(tmp_path, monkeypatch, merged):
    """Forced-overflow recovery whose exit writes fail once (the proof publication at the boundary and,
    where it exists, the ledger exit write); a cold restart re-sends the recovered view. A recovery row
    stored again must carry a relation, and no real row is lost."""
    engine = r534._engine(tmp_path)
    failed = []
    publish = engine._write_compress_commit_proof_payload

    def publish_fails_once(proof, payload):
        if "proof" not in failed:
            failed.append("proof")
            raise sqlite3.OperationalError("database is locked")
        return publish(proof, payload)

    def exit_write_lost(self, returned, capture):
        failed.append("exit")  # the write did not happen; the restart below drops any in-process retry

    reply = {"role": "assistant", "content": "recovery reply"}
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_write_compress_commit_proof_payload", publish_fails_once)
        # v0.29.0 has no ledger exit write; there this patch adds an attribute nothing calls.
        patch.setattr(LCMEngine, "_compress_exit_write", exit_write_lost, raising=False)
        pre, out = r534._recover(engine, patch)
        note = out[-1]["content"]
        r534._commit(engine, pre)
        assert "proof" in failed
        host = r534._merged(out) if merged else copy.deepcopy(out)
        engine.ingest(host + [reply])
        engine.shutdown()
    replay = r534._engine(tmp_path)
    try:
        replay.ingest(host + [reply, {"role": "user", "content": "next request"}])
        conn = replay._store._conn
        rows = conn.execute("SELECT store_id, content FROM messages").fetchall()
        silent = [(sid, content[:80]) for sid, content in rows if note in content and not _relations(conn, sid)]
        assert silent == []
        contents = [content for _sid, content in rows]
        # Nothing is lost. (The real rows' duplicate direction after a failed publication is a separate,
        # accepted #534 residual, so this case does not bound their count.)
        assert all(content in contents for content in ("older request", "recovery reply", "next request"))
    finally:
        replay.shutdown()
