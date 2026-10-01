"""#574: a fault in the pre-match write sequence leaves no partial bookkeeping."""

import sqlite3

import pytest

from hermes_lcm.store import MessageStore
from tests.test_issue_436_identity_anchor import (
    PAD,
    SYSTEM,
    T13,
    _a,
    _crash_child,
    _engine,
    _relations,
    _state_db,
    _t13,
    _turns,
    _u,
)


def _fail_nth(store, name, n):
    """Raise a SQLite fault before the named helper's n-th write."""
    original = getattr(store, name)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == n:
            raise sqlite3.OperationalError("disk I/O error (injected)")
        return original(*args, **kwargs)

    setattr(store, name, fail)


def _meta(engine, prefix):
    return [key for (key,) in engine._store._conn.execute(
        "SELECT key FROM metadata WHERE key LIKE ?", (prefix + "%",)
    )]


def _composite_child(tmp_path):
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None)])
    engine = _engine(tmp_path, "P")
    r, u = _u("R prompt" + PAD, 500.0), _u("U prompt" + PAD, 510.0)
    head = [SYSTEM, *_turns(1, 3, 0.0)]
    engine.ingest([*head, r, u])
    engine.on_session_start("C", platform="cli", context_length=200_000, conversation_id="conv")
    return engine, head, _u(r["content"] + "\n\n" + u["content"], 500.0)


def test_574_composite_relations_roll_back_with_a_failed_carry_write(tmp_path):
    engine, head, composite = _composite_child(tmp_path)
    try:
        _fail_nth(engine._store, "write_metadata_json", 1)
        engine.ingest([*head, composite, _a("reply to U", 511.0)])
        assert not _relations(engine)
        assert not _meta(engine, "identity_anchor_carry")
    finally:
        engine.shutdown()


def test_574_composite_with_schema_not_yet_created(tmp_path):
    """Ensure the schema before entering the block: executescript commits implicitly."""
    engine, head, composite = _composite_child(tmp_path)
    try:
        engine._store._identity_anchor_schema_ready = False
        engine._store._conn.execute("DROP TABLE IF EXISTS message_relations")
        engine._store._conn.commit()
        _fail_nth(engine._store, "write_metadata_json", 1)
        engine.ingest([*head, composite, _a("reply to U", 511.0)])
        assert not _relations(engine)
    finally:
        engine.shutdown()


def test_574_r1ws_relation_then_override_then_failure_leaves_nothing(tmp_path):
    """The override and its in-memory copy roll back when the later carry write fails."""
    engine = _crash_child(tmp_path, [_u(T13 + "\n", 130.0)])
    head, t14 = [SYSTEM, *_turns(1, 3, 0.0)], _u("[T14] user turn 14:" + PAD, 140.0)
    try:
        _fail_nth(engine._store, "write_metadata_json", 2)
        engine.ingest([*head, _u(T13, 130.0), t14, _a("reply to T14", 141.0)])
        assert not _meta(engine, "host_rewrite")
        [parent] = _t13(engine, "P")
        assert engine._host_rewrite_override_content(parent) is None
    finally:
        engine.shutdown()


def test_574_r1ws_metadata_failure_after_relation_write(tmp_path, monkeypatch):
    """A real composite relation precedes the R1-ws override fault; neither stays durable."""
    _state_db(tmp_path, [("P", None, "compression"), ("C", "P", None)])
    engine = _engine(tmp_path, "P")
    head = [SYSTEM, *_turns(1, 3, 0.0)]
    r, u = _u("R prompt" + PAD, 500.0), _u("U prompt" + PAD, 510.0)
    try:
        engine.ingest([*head, _u(T13 + "\n", 130.0), r, u])
        engine.on_session_start("C", platform="cli", context_length=200_000, conversation_id="conv")
        plans = []
        real_commit = engine._identity_anchor_commit

        def spy(plan, remainder_ids=None):
            if remainder_ids is None:
                plans.append((len(plan["relations"]), len(plan.get("ws", ()))))
            return real_commit(plan, remainder_ids)

        monkeypatch.setattr(engine, "_identity_anchor_commit", spy)
        _fail_nth(engine._store, "write_metadata_json", 1)
        composite = _u(r["content"] + "\n\n" + u["content"], 500.0)
        engine.ingest([*head, _u(T13, 130.0), composite, _a("reply", 511.0)])
        assert plans == [(1, 1)]
        assert not _relations(engine)
        assert not _meta(engine, "host_rewrite")
    finally:
        engine.shutdown()


def test_574_no_fault_paths_are_unchanged(tmp_path):
    engine = _crash_child(tmp_path, [_u(T13 + "\n", 130.0)])
    head, t14 = [SYSTEM, *_turns(1, 3, 0.0)], _u("[T14] user turn 14:" + PAD, 140.0)
    try:
        engine.ingest([*head, _u(T13, 130.0), t14, _a("reply to T14", 141.0)])
        assert _meta(engine, "host_rewrite") and _meta(engine, "identity_anchor_carry")
        assert not _t13(engine, "C")
        assert not engine._store._conn.in_transaction
    finally:
        engine.shutdown()


def test_574_store_atomic_unit(tmp_path):
    store = MessageStore(tmp_path / "lcm.db")
    try:
        sid = store.append("S", {"role": "user", "content": "x"})
        with pytest.raises(RuntimeError):
            with store.atomic():
                store.add_message_relations([[(sid, "alt_stamp", None, None, 1.0)]])
                store.backfill_observed_at(sid, 5.0)
                store.write_metadata_json(["k"], "1")
                raise RuntimeError("boom")
        assert store.read_metadata_json("k") is None and store.get(sid).get("observed_at") is None
        assert not store.get_message_relations([sid], "alt_stamp") and not store._commit_deferred
        with store.atomic():
            store.write_metadata_json(["k"], "2")
        assert store.read_metadata_json("k") == 2 and not store._conn.in_transaction
        with sqlite3.connect(tmp_path / "lcm.db") as other:
            assert other.execute("SELECT value FROM metadata WHERE key='k'").fetchone() == ("2",)
        other.close()
    finally:
        store.close()


def test_574_atomic_block_that_writes_nothing_takes_no_write_lock(tmp_path):
    """Deduplicated relations and unchanged metadata do not wait for a sibling's write lock."""
    store = MessageStore(tmp_path / "lcm.db")
    other = sqlite3.connect(tmp_path / "lcm.db", timeout=0.05)
    try:
        sid = store.append("S", {"role": "user", "content": "x", "timestamp": 7.0})
        store.add_message_relations([[(sid, "alt_stamp", None, None, 1.0)]])
        store.write_metadata_json(["k"], "1")
        store._conn.execute("PRAGMA busy_timeout=50")
        other.execute("BEGIN IMMEDIATE")
        with store.atomic():
            assert store.add_message_relations([[(sid, "alt_stamp", None, None, 1.0)]]) == 0
            assert not store.write_metadata_json(["k"], "1", skip_unchanged=True)
    finally:
        other.rollback()
        other.close()
        store.close()
