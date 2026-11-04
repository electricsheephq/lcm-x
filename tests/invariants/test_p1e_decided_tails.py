"""P1e PR-A: the decided store-space window, owner boundary and binding continuity."""

import copy
import sqlite3
from contextlib import contextmanager

import pytest

from hermes_lcm import ingest_ledger as ledger
from hermes_lcm.store import MessageStore
from tests.test_issue_2055_ingest_ledger_writes import (
    count, engines as engines, ledger_state, sql_rows,
)
from tests.test_host_uid_shadow import _m


def history(stamped=True):
    return [_m(role, text, float(i + 1) if stamped else None, text)
            for i, (role, text) in enumerate((("user", "A"), ("assistant", "B"),
                                            ("user", "C"), ("assistant", "D")))]


def queue(engine, view=None):
    view = history() if view is None else view
    engine.ingest(copy.deepcopy(view[:2]))
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("S", copy.deepcopy(view))
    finally:
        locker.rollback()
        locker.close()
    assert len(engine._pending_session_tails) == 1
    return engine._pending_session_tails[0]


def replay(engine, rows, *, bound=True, session="S", conversation="conv", ordinals=None, with_view=True):
    """Another writer's committed rows, optionally with ordinary ledger provenance."""
    store = engine._store
    salt = store.ingest_ledger_salt()
    entries, stamps = engine._ledger_view_entries(salt, rows)
    lineage = sql_rows(engine, "occurrence_bindings")[0]["lineage"]
    with store.ingest_atomic():
        ids = [store.append(session, row, source="cli", conversation_id=conversation) for row in rows]
        if bound:
            seq = ledger.next_view_seq(store._conn)
            maximum = store._conn.execute("SELECT max(ordinal) FROM occurrence_bindings").fetchone()[0] or 0
            positions = ordinals or [maximum + (i + 1) * 1024 for i in range(len(rows))]
            occs = ledger.insert_bindings(store._conn, lineage, [dict(
                ordinal=pos, store_id=sid, identity_digest=e.identity_digest, stamp=stamp,
                disposition="stored", key_rank=None, view_seq=seq)
                for pos, sid, e, stamp in zip(positions, ids, entries, stamps)])
            if with_view:
                ledger.write_view(store._conn, salt, lineage, "ingest", None,
                                  [e._replace(ref=occ) for e, occ in zip(entries, occs)], view_seq=seq)
    return ids


def contents(engine):
    return [r["content"] for r in engine._store.get_session_messages("S")]


def test_replay_between_match_and_append_is_excluded(engines, monkeypatch):
    engine = engines()
    queue(engine)
    append, witnessed = engine._append_ingest_batch, []
    writer = sqlite3.connect(engine._store.db_path, isolation_level=None, timeout=0)

    def competing(batch):
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            writer.execute("BEGIN IMMEDIATE")
        witnessed.append(True)
        return append(batch)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_append_ingest_batch", competing)
            engine._retry_pending_session_tails()
        assert witnessed == [True]
        assert contents(engine) == ["A", "B", "C", "D"]
    finally:
        writer.rollback()
        writer.close()


def test_fully_covered_tail_takes_no_write_lock(engines):
    engine = engines()
    queue(engine)
    replay(engine, history()[2:])
    statements = []
    engine._store._conn.set_trace_callback(statements.append)
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine._retry_pending_session_tails()
        assert not engine._pending_session_tails
        assert not any(s.upper().startswith("BEGIN") for s in statements)
    finally:
        engine._store._conn.set_trace_callback(None)
        locker.rollback()
        locker.close()


@pytest.mark.parametrize("disagreement", ["uid", "restamp", "observation"])
def test_uid_disagreement_is_not_replay(engines, disagreement):
    engine = engines()
    queue(engine)
    rows = [{**row, **({"message_uid": "foreign-" + row["message_uid"]} if disagreement == "uid"
                      else {"timestamp": row["timestamp"] + 100.})} for row in history()[2:]]
    ids = replay(engine, rows)
    if disagreement == "observation":
        conn = engine._store._conn
        with engine._store.ingest_atomic():
            for binding, row in zip([b for b in sql_rows(engine, "occurrence_bindings") if b["store_id"] in ids], history()[2:]):
                ledger.add_stamp_observation(conn, binding["lineage"], binding["occ_id"], str(row["timestamp"]),
                                             "alt_stamp", binding["view_seq"])
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"] + ([] if disagreement == "observation" else ["C", "D"])
    assert count(engine, "tail.retry.candidates_unmarked") == (0 if disagreement == "observation" else 2)


@pytest.mark.parametrize("head", ["head_known", "head_none"])
def test_unstamped_twins_before_head_never_cover(engines, head):
    engine = engines()
    twins = [_m("user", "C"), _m("assistant", "D")]
    view = copy.deepcopy(twins + twins)
    queue(engine, view)
    if head == "head_none":
        engine._store._conn.execute("DELETE FROM occurrence_bindings")
        engine._store._conn.commit()
    else:
        # Post-queue store ids still cannot replay bindings placed before the frozen head's ordinal.
        replay(engine, twins, ordinals=[100, 200])
    engine._retry_pending_session_tails()
    assert contents(engine) == ["C", "D", "C", "D"] + (["C", "D"] if head == "head_known" else [])
    assert len(sql_rows(engine, "occurrence_bindings")) == (6 if head == "head_known" else 2)


def test_unbound_replay_covers_positionally(engines):
    engine = engines()
    queue(engine)
    replay(engine, history()[2:], bound=False)
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]


def test_salt_change_falls_back(engines):
    engine = engines()
    queue(engine)
    engine._store._conn.execute("UPDATE ledger_meta SET value=? WHERE key='salt'", ((b"z" * 32).hex(),))
    engine._store._conn.commit()
    # Public reader is the contract even when a test fixture pins the installed salt.
    engine._store.ingest_ledger_salt = lambda: b"z" * 32
    replay(engine, history()[2:3], bound=False)
    before = ledger_state(engine)
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]
    assert ledger_state(engine) == before
    assert count(engine, "tail.retry.fallback.salt_changed") == 1


@pytest.mark.parametrize("failure", ["corrupt_view", "busy"])
def test_unreadable_ledger_keeps_positional_rule(engines, monkeypatch, failure):
    engine = engines()
    queue(engine)
    ids = replay(engine, history()[2:])
    deferred = count(engine, "tail.retry.deferred")
    if failure == "corrupt_view":
        engine._store._conn.execute("UPDATE ingest_views SET entries=X'00' WHERE view_seq=("
                                    "SELECT view_seq FROM occurrence_bindings WHERE store_id=?)", (ids[0],))
        engine._store._conn.commit()
    else:
        def busy(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")
        monkeypatch.setattr(ledger, "persisted_lineage", busy)
    engine._retry_pending_session_tails()
    if failure == "busy":
        # A busy read still defers the retry; it never falls back to the positional rule.
        assert len(engine._pending_session_tails) == 1
        assert count(engine, "tail.retry.deferred") == deferred + 1
        monkeypatch.undo()
        engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails
    assert count(engine, "tail.retry.fallback.read_failed") == (1 if failure == "corrupt_view" else 0)


def test_store_rebound_reads_ledger_from_tail_store(engines, tmp_path, monkeypatch):
    engine = engines()
    tail = queue(engine)
    original = engine._store
    rebound = MessageStore(tmp_path / "rebound.db")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_store", rebound)
            patch.setattr(original, "close", lambda: None)
            engine._retry_pending_session_tails()
        assert contents(engine) == ["A", "B", "C", "D"]
        assert len(sql_rows(engine, "occurrence_bindings")) == 4
        assert not rebound.get_session_messages("S")
        assert tail["store"] is original
    finally:
        rebound.close()


def test_busy_message_without_locked_still_queues(engines, monkeypatch):
    engine = engines()
    engine.ingest(history()[:2])
    append = engine._append_ingest_batch

    def busy(batch):
        if getattr(engine, "_ingest_read_only", None) is None:
            error = sqlite3.OperationalError("injected busy snapshot")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
            raise error
        return append(batch)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_append_ingest_batch", busy)
        engine.on_session_end("S", history())
    assert len(engine._pending_session_tails) == 1
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]


def test_identical_ends_queue_one_tail(engines):
    engine = engines()
    queue(engine)
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("S", copy.deepcopy(history()))
        assert len(engine._pending_session_tails) == 1
        assert engine.__dict__.get("_frozen_tail_reached") is True
    finally:
        locker.rollback()
        locker.close()


def test_retry_bindings_contiguous_for_cold_cursor(engines):
    engine = engines()
    view = [*history(), _m("user", "E", 5., "E")]
    queue(engine, view)
    engine.ingest(view[:3])
    replay(engine, [_m("user", "later", 99.)], session="other", conversation="other", with_view=False)
    engine._retry_pending_session_tails()
    bindings = {r["store_id"]: r for r in sql_rows(engine, "occurrence_bindings")}
    ordered = [bindings[r["store_id"]]["ordinal"] for r in engine._store.get_session_messages("S")]
    assert ordered == sorted(ordered) and len(set(ordered)) == 5
    assert ordered[-1] < max(b["ordinal"] for b in bindings.values())
    # A later ordinary host checkpoint supplies the vector: the retry itself must not write one.
    engine.ingest(view)
    assert contents(engine) == ["A", "B", "C", "D", "E"]
    attempt = engine._prepare_ingest_ledger(view, {"ledger_digest_ms": 0.})
    result, _counts = ledger.cold_cursor_candidate(engine._store._conn, session_id="S", attempt=attempt,
                                                  ignored=[False] * len(view))
    assert result["k"] == len(view)


@pytest.mark.parametrize("bound", [False, True])
def test_replay_committed_between_pretrim_and_begin(engines, monkeypatch, bound):
    """#1440 item 1: another writer commits the tail's replay after the pre-trim and before BEGIN IMMEDIATE.

    The retry must re-match inside its transaction, so the replayed rows are covered and stored once.
    """
    engine = engines()
    queue(engine)
    store = engine._store
    atomic, fired = store.ingest_atomic, []

    @contextmanager
    def racing(**kwargs):
        if not fired:
            fired.append(True)
            if bound:
                with monkeypatch.context() as inner:
                    inner.setattr(store, "ingest_atomic", atomic)
                    replay(engine, history()[2:])
            else:
                other = MessageStore(store.db_path)
                try:
                    for row in history()[2:]:
                        other.append("S", row, source="cli", conversation_id="conv")
                finally:
                    other.close()
        with atomic(**kwargs):
            yield

    monkeypatch.setattr(store, "ingest_atomic", racing)
    engine._retry_pending_session_tails()
    assert fired == [True]
    assert contents(engine) == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails
