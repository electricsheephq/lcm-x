"""Deterministic cross-connection lifecycle races from #634."""

import pytest

from hermes_lcm import lifecycle_state as lifecycle_state_module
from hermes_lcm.lifecycle_state import LifecycleStateStore

CONV = "agent:main:telegram:dm:42"


def _other_writes_after_next_read(store, fn):
    read, fired = store.get_by_conversation, []

    def racing(conversation_id):
        state = read(conversation_id)
        if not fired:
            fired.append(1)
            fn()
        return state

    store.get_by_conversation = racing
    return fired


def _other_writes_before_next_write(store, fn, max_fires=1):
    connection, fired = store._conn, []

    class RacingConnection:
        def execute(self, statement, *args, **kwargs):
            sql = statement.lstrip().upper()
            if (
                sql.startswith(("UPDATE", "INSERT"))
                and "LCM_LIFECYCLE_STATE" in sql
                and len(fired) < max_fires
            ):
                fired.append(1)
                fn()
            return connection.execute(statement, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(connection, name)

    store._conn = RacingConnection()
    return connection, fired


def _start_a(a):
    a.bind_session("A", conversation_id=CONV)
    a.advance_frontier(CONV, "A", 100)


def _start_b_after_a_finalize(a, b):
    _start_a(a)
    a.finalize_session(CONV, "A", frontier_store_id=100)
    b.bind_session("B", conversation_id=CONV)
    b.advance_frontier(CONV, "B", 900)


def _row_tuple(row):
    return (
        row.current_session_id,
        row.current_frontier_store_id,
        row.last_finalized_session_id,
        row.last_finalized_frontier_store_id,
        row.debt_kind,
    )


def test_stale_finalize_does_not_unbind_a_session_that_bound_after_its_read(tmp_path):
    db = tmp_path / "lcm.db"
    a, b = LifecycleStateStore(db), LifecycleStateStore(db)
    try:
        _start_a(a)

        def b_binds_and_compacts():
            b.bind_session("B", conversation_id=CONV)
            b.advance_frontier(CONV, "B", 900)

        fired = _other_writes_after_next_read(a, b_binds_and_compacts)
        a.finalize_session(CONV, "A", frontier_store_id=100)
        assert fired
        row = b.get_by_conversation(CONV)
        assert (row.current_session_id, row.current_frontier_store_id) == ("B", 900)
    finally:
        a.close()
        b.close()


def test_stale_bind_does_not_overwrite_a_later_finalized_checkpoint(tmp_path):
    db = tmp_path / "lcm.db"
    a, b = LifecycleStateStore(db), LifecycleStateStore(db)
    try:
        _start_b_after_a_finalize(a, b)
        fired = _other_writes_after_next_read(
            a, lambda: b.finalize_session(CONV, "B", frontier_store_id=900)
        )
        a.bind_session("A", conversation_id=CONV)
        assert fired
        row = b.get_by_conversation(CONV)
        assert (row.last_finalized_session_id, row.last_finalized_frontier_store_id) == ("B", 900)
    finally:
        a.close()
        b.close()


def test_finalize_racing_a_bind_before_its_write_keeps_the_new_binding(tmp_path):
    db = tmp_path / "lcm.db"
    a, b = LifecycleStateStore(db), LifecycleStateStore(db)
    real = a._conn
    try:
        _start_a(a)

        def b_binds_and_compacts():
            b.bind_session("B", conversation_id=CONV)
            b.advance_frontier(CONV, "B", 900)

        _, fired = _other_writes_before_next_write(a, b_binds_and_compacts)
        a.finalize_session(CONV, "A", frontier_store_id=100)
        assert fired == [1]
        assert _row_tuple(b.get_by_conversation(CONV)) == ("B", 900, "A", 100, None)
    finally:
        a._conn = real
        a.close()
        b.close()


def test_bind_racing_a_finalize_before_its_write_rereads_and_keeps_the_checkpoint(tmp_path):
    db = tmp_path / "lcm.db"
    a, b = LifecycleStateStore(db), LifecycleStateStore(db)
    real = a._conn
    try:
        _start_b_after_a_finalize(a, b)
        _, fired = _other_writes_before_next_write(
            a, lambda: b.finalize_session(CONV, "B", frontier_store_id=900)
        )
        returned = a.bind_session("A", conversation_id=CONV)
        row = b.get_by_conversation(CONV)
        assert fired == [1]
        assert _row_tuple(row) == ("A", 0, "B", 900, None)
        assert returned == row
    finally:
        a._conn = real
        a.close()
        b.close()


def test_bind_does_not_clobber_debt_recorded_in_its_window(tmp_path):
    db = tmp_path / "lcm.db"
    a, b = LifecycleStateStore(db), LifecycleStateStore(db)
    real = a._conn
    try:
        _start_b_after_a_finalize(a, b)
        _, fired = _other_writes_before_next_write(
            a, lambda: b.record_debt(CONV, kind="raw_backlog", size_estimate=7)
        )
        a.bind_session("A", conversation_id=CONV)
        row = b.get_by_conversation(CONV)
        assert fired == [1]
        assert row.current_session_id == "A"
        assert (row.debt_kind, row.debt_size_estimate) == ("raw_backlog", 7)
    finally:
        a._conn = real
        a.close()
        b.close()


def test_bind_falls_back_to_a_write_transaction_after_repeated_lost_races(tmp_path, monkeypatch):
    # raising=False lets the same regression run on the pre-CAS base.
    monkeypatch.setattr(lifecycle_state_module, "_BIND_CAS_ATTEMPTS", 2, raising=False)
    db = tmp_path / "lcm.db"
    a, b = LifecycleStateStore(db), LifecycleStateStore(db)
    real = a._conn
    statements, resets = [], []
    try:
        _start_b_after_a_finalize(a, b)
        real.set_trace_callback(statements.append)

        def reset():
            resets.append(b.record_reset(CONV).last_reset_at)

        _, fired = _other_writes_before_next_write(a, reset, max_fires=2)
        returned = a.bind_session("A", conversation_id=CONV)
        row = b.get_by_conversation(CONV)
        assert len(fired) == 2
        assert _row_tuple(row) == ("A", 0, "A", 100, None)
        assert row.last_reset_at == resets[-1]
        assert returned == row
        assert "BEGIN IMMEDIATE" in statements
        assert "COMMIT" in statements
        assert not real.in_transaction
    finally:
        real.set_trace_callback(None)
        a._conn = real
        a.close()
        b.close()


@pytest.mark.parametrize("sequence", ["finalize", "bind"])
def test_sequential_controls_match_todays_semantics(tmp_path, sequence):
    db = tmp_path / "lcm.db"
    a, b = LifecycleStateStore(db), LifecycleStateStore(db)
    try:
        if sequence == "finalize":
            _start_a(a)
            b.bind_session("B", conversation_id=CONV)
            b.advance_frontier(CONV, "B", 900)
            a.finalize_session(CONV, "A", frontier_store_id=100)
            expected = ("B", 900, "A", 100, None)
        else:
            _start_b_after_a_finalize(a, b)
            b.finalize_session(CONV, "B", frontier_store_id=900)
            a.bind_session("A", conversation_id=CONV)
            expected = ("A", 0, "B", 900, None)
        assert _row_tuple(b.get_by_conversation(CONV)) == expected
    finally:
        a.close()
        b.close()


def test_finalize_of_a_missing_row_returns_none(tmp_path):
    db = tmp_path / "lcm.db"
    a, b = LifecycleStateStore(db), LifecycleStateStore(db)
    try:
        before = a.row_count()
        assert a.finalize_session("nope", "A", 5) is None
        assert a.row_count() == before
    finally:
        a.close()
        b.close()
