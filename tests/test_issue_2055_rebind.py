"""#2055 P1g-1: exact own-session rebinds and the examined-through mark."""

from collections import Counter
from copy import deepcopy
import sqlite3
import subprocess
import sys

import pytest

from hermes_lcm import ingest_ledger as ledger
from tests.test_host_uid_shadow import _m
from tests.test_issue_2055_ingest_ledger_writes import (
    SALT, count, engines as engines, sql_rows, restart,
)


def tool_history():
    return [_m("user", "question", 1.),
            _m("assistant", "", 2., tool_calls=[
                {"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]),
            _m("tool", "result", 3., tool_call_id="call-1", name="read", tool_name="read"),
            _m("assistant", "answer", 4.)]


def test_preledger_session_binds(engines, monkeypatch):
    engine = engines(mode="off")
    view = tool_history()
    engine.ingest(deepcopy(view))
    stored = engine._store.get_session_messages("S")
    assert len(stored) == len(view)
    # Stop premise: the lookup's actual stored bytes, including both tool rows,
    # must yield the incoming pre-protection salted identity.
    hits = engine._store.find_rows_by_observed_at("conv", ["S"], [1., 2., 3., 4.])
    assert [ledger.identity_digest(SALT, r) for r in hits] == [ledger.identity_digest(SALT, r) for r in view]
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    engine.ingest(deepcopy([*view, _m("user", "next", 5.)]))
    bindings = sql_rows(engine, "occurrence_bindings")
    ordered = sorted(bindings, key=lambda b: b["ordinal"])
    assert [b["disposition"] for b in ordered] == ["migrated"] * 4 + ["stored"]
    # N2: migrated and newly stored refs must survive in the persisted view.
    newest = ledger.newest_view(engine._store._conn, ordered[0]["lineage"], "ingest")
    entries = ledger.decode_entries(newest["entries"])
    assert [entry.ref for entry in entries[:4]] == [b["occ_id"] for b in ordered[:4]]
    assert entries[4].ref == ordered[4]["occ_id"]
    assert {b["store_id"] for b in bindings} == {r["store_id"] for r in engine._store.get_session_messages("S")}
    assert sql_rows(engine, "lineage_marks")[0]["max_store_id"] == max(b["store_id"] for b in bindings)


def history(n=4):
    return [_m("user" if i % 2 == 0 else "assistant", f"row-{i}", float(i + 1)) for i in range(n)]


def unbound(engine, view, *, session="S"):
    with engine._store.ingest_atomic():
        return [engine._store.append(session, r, conversation_id="conv") for r in view]


def write(engine, view, *, matched=None, stored=None, no_view=False):
    attempt = engine._prepare_ingest_ledger(view, Counter())
    attempt.update(stored=stored or {}, matched=matched or {}, remainders={}, observations=[], no_view=no_view)
    callbacks = []
    with engine._store.ingest_atomic():
        engine._write_ingest_ledger(attempt, callbacks, Counter())
    for callback in callbacks:
        callback()
    return attempt


def mark(engine):
    rows = sql_rows(engine, "lineage_marks")
    return rows[0]["max_store_id"] if rows else None


def migrated(engine):
    return [b for b in sql_rows(engine, "occurrence_bindings") if b["disposition"] == "migrated"]


def bind(engine, ids, view, ordinals, lineage=None):
    attempt = engine._prepare_ingest_ledger(view, Counter())
    with engine._store.ingest_atomic():
        return ledger.insert_bindings(engine._store._conn, lineage or attempt["lineage"], [
            dict(store_id=sid, ordinal=o, identity_digest=e.identity_digest, stamp=stamp,
                 disposition="stored", view_seq=0)
            for sid, o, e, stamp in zip(ids, ordinals, attempt["entries"], attempt["stamps"])])


def test_name_only_tool_row_stays_unbound_and_counted(engines, monkeypatch):
    engine = engines(mode="off")
    view = tool_history()
    del view[2]["tool_name"]
    engine.ingest(deepcopy(view))
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    engine.ingest(deepcopy(view))
    rows = engine._store.get_session_messages("S")
    assert len(rows) == 4
    assert {b["store_id"] for b in migrated(engine)} == {r["store_id"] for i, r in enumerate(rows) if i != 2}
    assert not engine._store._conn.execute("SELECT 1 FROM occurrence_bindings WHERE store_id=?", (rows[2]["store_id"],)).fetchone()
    assert count(engine, "rebind.residual") == 1 and mark(engine) == rows[-1]["store_id"]


def test_cold_cursor_uses_migrated_prefix(engines, monkeypatch):
    from tests.test_issue_2055_cold_cursor import force_today
    engine = engines(mode="off")
    view = history()
    engine.ingest(deepcopy(view))
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    engine.ingest(deepcopy([*view, _m("user", "new", 5.)]))
    restart(engine)
    force_today(engine, monkeypatch)
    engine.ingest(deepcopy([*view, _m("user", "new", 5.), _m("assistant", "reply", 6.)]))
    assert len(migrated(engine)) == 4
    assert engine._last_ingest_reconciliation["k"] == 5
    assert len(engine._store.get_session_messages("S")) == 6


def test_protected_rows_stay_unbound_counted(engines, monkeypatch):
    engine = engines(mode="off")
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["api_key"]
    view = [_m("user", "api_key=sk-user1234567890abcdef", 1.), history()[1]]
    engine.ingest(deepcopy(view))
    assert engine._store.get_session_messages("S")[0]["content"] != view[0]["content"]
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    write(engine, view)
    assert len(migrated(engine)) == 1 and count(engine, "rebind.residual") == 1
    assert mark(engine) == 2 and len(engine._store.get_session_messages("S")) == 2


@pytest.mark.parametrize("side", ["incoming", "stored"])
def test_twins_never_bind(engines, side):
    engine = engines()
    row = history()[0]
    unbound(engine, [row, row] if side == "stored" else [row])
    write(engine, [row, row] if side == "incoming" else [row])
    assert not migrated(engine) and count(engine, "rebind.skipped.ambiguous")
    assert count(engine, "rebind.residual") == (2 if side == "stored" else 1)
    assert mark(engine) == (2 if side == "stored" else 1)


@pytest.mark.parametrize("other_index", [False, True])
def test_anchor_disagreement_wins(engines, other_index):
    engine = engines()
    view = history(2)
    ids = unbound(engine, view)
    write(engine, view, matched={1: ids[0]} if other_index else {0: ids[1], 1: ids[1]})
    assert count(engine, "rebind.skipped.anchor_disagree") >= 1
    assert ids[0] not in {b["store_id"] for b in migrated(engine)}
    assert mark(engine) == ids[-1]


def test_anchor_match_on_own_row_binds(engines):
    engine = engines()
    view = history(2)
    ids = unbound(engine, view)
    write(engine, view, matched={0: ids[0]})
    assert not count(engine, "rebind.skipped.anchor_disagree")
    assert {b["store_id"] for b in migrated(engine)} == set(ids)


def test_unstamped_left_unbound(engines):
    engine = engines()
    view = [_m("user", "untimed", None), history()[1]]
    unbound(engine, view)
    write(engine, view)
    assert len(migrated(engine)) == 1 and count(engine, "rebind.skipped.unstamped") == 1
    assert count(engine, "rebind.residual") == 1 and mark(engine) == 2


def test_crossed_pair_dropped_rest_bound(engines):
    engine = engines()
    view = history(5)
    ids = unbound(engine, view)
    bind(engine, [ids[0], ids[2], ids[4]], [view[0], view[2], view[4]], [20, 10, 40])
    write(engine, view)
    assert [b["store_id"] for b in migrated(engine)] == [ids[3]]
    assert count(engine, "rebind.skipped.crossed") == 1 and count(engine, "rebind.residual") == 1
    assert mark(engine) == ids[-1]


def test_rebind_respace_counted(engines):
    engine = engines()
    view = history(3)
    ids = unbound(engine, view)
    bind(engine, [ids[0], ids[2]], [view[0], view[2]], [1, 2])
    write(engine, view)
    assert len(migrated(engine)) == 1 and count(engine, "respaced") == 1
    assert sorted(b["store_id"] for b in sql_rows(engine, "occurrence_bindings")) == ids


def test_rebind_after_write_failed(engines, monkeypatch):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_bindings", lambda *a: (_ for _ in ()).throw(ValueError("writer failed")))
        engine.ingest(history())
    assert count(engine, "write_failed") == 1 and mark(engine) is None
    engine.ingest(history())
    assert len(migrated(engine)) == 4 and mark(engine) == 4


def test_rebind_after_crossed(engines, monkeypatch):
    engine = engines()
    original = ledger.plan_ordinals
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "plan_ordinals", lambda *a: (_ for _ in ()).throw(
            ledger.LedgerError("placement bounds must be non-decreasing in view order")))
        engine.ingest(history())
    assert ledger.plan_ordinals is original
    assert count(engine, "skipped_crossed") == 1 and mark(engine) is None
    engine.ingest(history())
    assert len(migrated(engine)) == 4 and mark(engine) == 4


def test_rebind_after_salt_changed(engines, monkeypatch):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "ingest_ledger_salt", lambda: b"x" * 32)
        engine.ingest(history())
    assert count(engine, "salt_changed") and mark(engine) is None
    engine.ingest(history())
    assert len(migrated(engine)) == 4 and mark(engine) == 4


def test_rebind_after_tail_fallback(engines, monkeypatch):
    from tests.invariants.test_p1e_decided_tails import queue
    engine = engines()
    view = history()
    queue(engine, view)
    engine._pending_session_tails[0].pop("ledger")
    engine._retry_pending_session_tails()
    assert len(engine._store.get_session_messages("S")) == 4
    engine.ingest(view)
    assert len(migrated(engine)) == 2 and mark(engine) == 4


@pytest.mark.parametrize("route", ["other_process", "s3"])
def test_other_process_and_s3_rows_rebound_next_ingest(engines, route):
    from hermes_lcm.store import MessageStore
    engine = engines()
    view = history()
    engine.ingest(view[:2])
    if route == "other_process":
        store = MessageStore(engine._store.db_path)
        try:
            store.append_batch("S", view[2:], conversation_id="conv")
        finally:
            store.close()
    else:
        engine._append_off_current_session_end_suffix("S", view[2:], source="cli", conversation_id="conv")
    engine.ingest(view)
    assert len(migrated(engine)) == 2 and mark(engine) == 4
    assert len(engine._store.get_session_messages("S")) == 4


def test_salt_reset_remigrates(engines, monkeypatch):
    from hermes_lcm import store as store_module
    engine = engines()
    view = history()
    engine.ingest(view)
    with engine._store.atomic(immediate=True):
        engine._store._conn.execute("DELETE FROM ledger_meta WHERE key='digest_salt'")
    monkeypatch.setattr(store_module.secrets, "token_bytes", lambda size: b"n" * size)
    engine._store.ensure_ingest_ledger()
    assert engine._store.ingest_ledger_salt() != SALT
    restart(engine)
    engine.ingest(view)
    assert len(migrated(engine)) == 4 and mark(engine) == 4
    assert all(b["identity_digest"] == ledger.identity_digest(b"n" * 32, r)
               for b, r in zip(migrated(engine), view))
    assert len(engine._store.get_session_messages("S")) == 4


def test_v029_rows_rebound(engines, monkeypatch):
    engine = engines(mode="off", initialized=False)
    view = history()
    engine.ingest(view)
    assert engine._store.ingest_ledger_state() == "absent"
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    restart(engine)
    engine.ingest(view)
    assert len(migrated(engine)) == 4 and mark(engine) == 4


@pytest.mark.parametrize("failure", ["plan", "insert", "too_large"])
def test_mark_not_advanced_on_failure_or_too_large(engines, monkeypatch, failure):
    engine = engines()
    view = history(4097 if failure == "too_large" else 4)
    unbound(engine, view)
    original_flag = getattr(engine._store, "_pending_relation_schema", False)
    name = "plan_rebinds" if failure == "plan" else "insert_bindings"
    with monkeypatch.context() as patch:
        if failure != "too_large":
            original = getattr(ledger, name)
            def fail(*args, **kwargs):
                result = original(*args, **kwargs)
                if failure == "insert" and not args[2]:
                    return result
                engine._store._pending_relation_schema = not original_flag
                raise ValueError("rebind failed after effects")
            patch.setattr(ledger, name, fail)
        write(engine, view)
        assert not migrated(engine) and mark(engine) is None
        assert getattr(engine._store, "_pending_relation_schema", False) == original_flag
    if failure == "too_large":
        assert count(engine, "rebind.skipped.too_large") == 1
        write(engine, view[:4])  # Memo vetoes another plan in this process.
        assert not migrated(engine) and mark(engine) is None
        assert count(engine, "rebind.runs") == 1
    else:
        assert count(engine, "rebind_failed") == 1
        write(engine, view)
        assert len(migrated(engine)) == 4 and mark(engine) == 4


def test_never_binds_other_session(engines):
    engine = engines()
    view = history()
    own = unbound(engine, view)
    foreign = unbound(engine, view, session="other")
    write(engine, view)
    assert {b["store_id"] for b in migrated(engine)} == set(own)
    assert not set(foreign) & {b["store_id"] for b in sql_rows(engine, "occurrence_bindings")}
    assert mark(engine) == own[-1]


def test_alias_and_other_lineage_and_mark_are_vetoes(engines):
    engine = engines()
    view = history(5)
    ids = unbound(engine, view)
    # T1: the foreign binding is above the mark, so only the lineage veto protects it.
    bind(engine, ids[3:4], view[3:4], [100], lineage="foreign-lineage")
    with engine._store.ingest_atomic():
        engine._store.add_message_relations([[(ids[1], "alt_stamp", None, None, 42.)]])
        ledger.update_store_mark(engine._store._conn, engine._prepare_ingest_ledger(view, Counter())["lineage"], ids[2])
    write(engine, [view[0], {**view[1], "timestamp": 42.}, *view[2:]])
    assert [b["store_id"] for b in migrated(engine)] == [ids[4]]
    assert count(engine, "rebind.skipped.already_bound") == 3  # foreign binding and two below-mark canonical hits
    assert count(engine, "rebind.residual") == 0 and mark(engine) == ids[-1]
    protected = [b for b in sql_rows(engine, "occurrence_bindings") if b["store_id"] == ids[3]]
    assert len(protected) == 1 and protected[0]["lineage"] == "foreign-lineage"


def test_no_view_retry_binds_and_leaves_the_mark(engines):
    engine = engines()
    view = history(6)
    ids = unbound(engine, view)
    # The cold-tail ingest route receives an older view than the session's history.
    engine._ingest_messages(view[:4], no_view=True)
    assert len(migrated(engine)) == 4 and mark(engine) is None
    assert not sql_rows(engine, "ingest_views")
    engine.ingest(view)
    assert len(migrated(engine)) == 6 and mark(engine) == ids[-1]
    assert len(engine._store.get_session_messages("S")) == 6


def test_frozen_writer_ingests_rebound_store_with_same_cursor(engines, monkeypatch):
    from tests.test_issue_2055_cold_cursor import force_today
    from tests.test_issue_2055_emission_restart import _main_ledger_write
    results = []
    for name, frozen in [("current", False), ("frozen", True)]:
        engine = engines(name)
        view = history()
        unbound(engine, view)
        write(engine, view)
        restart(engine)
        if frozen:
            monkeypatch.setattr(engine, "_write_ingest_ledger", _main_ledger_write.__get__(engine))
        force_today(engine, monkeypatch)
        engine.ingest([*view, _m("user", "next", 5.)])
        results.append((engine._last_ingest_reconciliation["k"],
                        [(b["store_id"], b["ordinal"]) for b in sql_rows(engine, "occurrence_bindings")]))
        assert not count(engine, "write_failed")
    assert results[0] == results[1] and results[0][0] == 4


CRASH_SCRIPT = r'''
import os, sys
from pathlib import Path
from contextlib import contextmanager
from collections import Counter
from unittest.mock import patch
import pytest
import tests.conftest
from hermes_lcm.engine import LCMEngine
from tests.harness import pin_char_counter
from tests.test_host_uid_shadow import _engine
from tests.test_issue_2055_rebind import history, write
root, point = Path(sys.argv[1]), sys.argv[2]
pin_char_counter(pytest.MonkeyPatch())
LCMEngine._schedule_rollup_maintenance = lambda *a: None
LCMEngine._schedule_embedding_maintenance = lambda *a: None
engine = _engine(root)
atomic = engine._store.ingest_atomic
@contextmanager
def interrupted(**kwargs):
    with atomic(**kwargs):
        yield
        if point == "before_commit": os._exit(91)
    os._exit(92)
engine._store.ingest_atomic = interrupted
write(engine, history())
'''


@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
def test_rebind_crash_cell(engines, point):
    engine = engines()
    unbound(engine, history())
    result = subprocess.run([sys.executable, "-c", CRASH_SCRIPT, str(engine._hermes_home), point],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == (91 if point == "before_commit" else 92), result.stderr
    assert len(migrated(engine)) == (0 if point == "before_commit" else 4)
    assert mark(engine) == (None if point == "before_commit" else 4)
    restart(engine)
    engine.ingest(history())
    bindings = sql_rows(engine, "occurrence_bindings")
    assert len(migrated(engine)) == 4 and mark(engine) == 4
    assert len({b["store_id"] for b in bindings}) == len(bindings) == 4
    assert len(engine._store.get_session_messages("S")) == 4


def test_first_ingest_2000_rows_ledger_ms(engines, record_property):
    engine = engines()
    view = history(2000)
    unbound(engine, view)
    engine.ingest(view)
    record_property("ledger_ms", engine._last_ingest_attribution["ledger_ms"])
    assert len(migrated(engine)) == 2000 and mark(engine) == 2000


def test_rebind_failure_preserves_new_stored_binding_and_view(engines, monkeypatch):
    engine = engines()
    view = history()
    ids = unbound(engine, view)
    original = ledger.insert_bindings
    def fail(conn, lineage, rows):
        result = original(conn, lineage, rows)
        if rows and rows[0]["disposition"] == "migrated":
            raise ValueError("after migrated inserts")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_bindings", fail)
        engine.ingest([*view, _m("user", "fresh", 5.)])
    bindings = sql_rows(engine, "occurrence_bindings")
    assert len(bindings) == 1 and bindings[0]["disposition"] == "stored"
    assert count(engine, "rebind_failed") == 1 and mark(engine) is None
    entries = ledger.decode_entries(sql_rows(engine, "ingest_views")[-1]["entries"])
    assert all(e.ref is None for e in entries[:4])
    assert entries[-1].ref == bindings[0]["occ_id"]
    engine.ingest([*view, _m("user", "fresh", 5.)])
    assert {b["store_id"] for b in migrated(engine)} == set(ids)
    assert mark(engine) == 5 and len(engine._store.get_session_messages("S")) == 5


@pytest.mark.parametrize("failure", ["busy", "interrupt", "savepoint", "owner_lost"])
def test_rebind_owner_failures_propagate(engines, monkeypatch, failure):
    from hermes_lcm.sqlite_util import _SqliteSavepointError
    engine = engines()
    unbound(engine, history())
    def fail(*args):
        if failure == "busy":
            exc = sqlite3.OperationalError("database is locked")
            exc.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
        elif failure == "interrupt":
            exc = KeyboardInterrupt()
        elif failure == "savepoint":
            exc = _SqliteSavepointError("owner cleanup failed")
        else:
            engine._store._conn.rollback()
            exc = ValueError("owner lost")
        raise exc
    monkeypatch.setattr(ledger, "plan_rebinds", fail)
    with pytest.raises(BaseException):
        write(engine, history())
    assert not migrated(engine) and mark(engine) is None
    assert not count(engine, "rebind_failed")


def test_read_only_ingest_leaves_rebinds_and_mark(engines, monkeypatch):
    engine = engines()
    unbound(engine, history())
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_ingest_read_only", object(), raising=False)
        write(engine, history())
    assert not migrated(engine) and mark(engine) is None
    write(engine, history())
    assert len(migrated(engine)) == 4 and mark(engine) == 4


def test_plan_rebinds_writes_nothing(engines):
    engine = engines()
    view = history()
    ids = unbound(engine, view)
    engine._store.find_rows_by_observed_at("conv", ["S"], [1.])  # Existing lookup's lazy schema, outside the planner proof.
    attempt = engine._prepare_ingest_ledger(view, Counter())
    attempt.update(stored={}, matched={})
    conn, statements = engine._store._conn, []
    conn.set_trace_callback(statements.append)
    try:
        plan = ledger.plan_rebinds(engine._store, "conv", "S", SALT, attempt, None)
    finally:
        conn.set_trace_callback(None)
    assert plan["pairs"] == list(enumerate(ids))
    assert all(s.lstrip().upper().startswith("SELECT") for s in statements)
