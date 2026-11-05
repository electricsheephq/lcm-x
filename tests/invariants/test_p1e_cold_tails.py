"""P1e PR-B: raw cold tails preserve final messages and session ownership (I1/I2)."""

import copy
import inspect
import logging
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_lcm import maintenance_scheduler
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import get_large_output_storage_dir, load_externalized_payload, extract_externalized_ref
from hermes_lcm.host_uid import HOST_UID_COUNTER_KEY
from hermes_lcm.session_patterns import compile_session_patterns
from hermes_lcm.store import MessageStore
from tests.invariants.test_p1e_decided_tails import contents, history, queue, replay
from tests.test_issue_2055_ingest_ledger_writes import count, engines as engines, restart, sql_rows
from tests.test_host_uid_shadow import _m, _state_db
from tests.test_host_uid_emit_carry import _ContainsPattern


def queue_cold(engine, monkeypatch, view=None, *, read="get_session_count", error=sqlite3.OperationalError,
               reconcile=True, stored=None):
    view = history() if view is None else view
    engine.ingest(copy.deepcopy(view[:2] if stored is None else stored))
    engine._ingest_cursor_needs_reconcile = reconcile

    def failed(*args, **kwargs):
        raise error("database is locked" if issubclass(error, sqlite3.Error) else "injected frozen read failure")

    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, read, failed)
            engine.on_session_end("S", view)
    finally:
        locker.rollback()
        locker.close()
    assert len(engine._pending_session_tails) == 1
    tail = engine._pending_session_tails[0]
    assert tail["kind"] == "cold" and tail["view"] == view
    assert engine in maintenance_scheduler._PENDING_TAIL_ENGINES
    return tail


def off_current_end(engine):
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    view = history()
    engine.ingest(copy.deepcopy(view[:2]))
    engine.on_session_start("S2", platform="api", context_length=200_000, conversation_id="conv2")
    assert engine._lcm_session_last_normal_conversation_id["S1"] == "conv1"
    return view


def fail_end_read(engine, *, error=sqlite3.OperationalError, recorded_probe=False):
    read = engine._store.get_range

    def failed(session_id, *args, **kwargs):
        if session_id == "S1" and (not recorded_probe or kwargs.get("limit") == 3):
            raise error("database is busy" if issubclass(error, sqlite3.Error) else "injected prefix failure")
        return read(session_id, *args, **kwargs)

    return failed


def assert_off_current_cold_retry(engine, view, monkeypatch, caplog):
    assert [row["content"] for row in engine._store.get_session_messages("S1")] == ["A", "B"]
    assert not engine._store.get_session_messages("S2")
    assert len(engine._pending_session_tails) == 1
    tail = engine._pending_session_tails[0]
    assert (tail["kind"], tail["session_id"], tail["conversation_id"], tail["source"]) == (
        "cold", "S1", "conv1", "cli")
    assert tail["view"] == view and tail["view"] is not view and tail["view"][0] is not view[0]
    assert engine in maintenance_scheduler._PENDING_TAIL_ENGINES
    assert count(engine, "tail.queued.cold.OperationalError") == count(engine, "tail.busy_at_end") == 1
    assert "retry pending" in caplog.text and "SQLite lock" in caplog.text
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_ingest_messages", lambda *a, **k: pytest.fail("cold current route"))
        patch.setattr(engine._lifecycle, "finalize_session", lambda *a, **k: pytest.fail("retry finalized"))
        engine._retry_pending_session_tails()
        engine._retry_pending_session_tails()
    rows = engine._store.get_session_messages("S1")
    assert [row["content"] for row in rows] == ["A", "B", "C", "D"]
    assert all(row["conversation_id"] == "conv1" and row["source"] == "cli" for row in rows)
    assert engine._session_id == "S2" and engine._conversation_id == "conv2"
    assert not engine._store.get_session_messages("S2") and not engine._pending_session_tails


@pytest.mark.parametrize("recorded_probe", [False, True])
def test_off_current_end_lock_queues_cold_tail(engines, monkeypatch, caplog, recorded_probe):
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    view = off_current_end(engine)
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_range", fail_end_read(engine, recorded_probe=recorded_probe))
        patch.setattr(engine._lifecycle, "finalize_session", lambda *a, **k: pytest.fail("queued end finalized"))
        engine.on_session_end("S1", view)
    assert_off_current_cold_retry(engine, view, monkeypatch, caplog)


def test_off_current_end_busy_append_queues_cold_tail(engines, monkeypatch, caplog):
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    view = off_current_end(engine)

    def busy(*args, **kwargs):
        raise sqlite3.OperationalError("database is busy")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_append_off_current_session_end_suffix", busy)
        patch.setattr(engine._lifecycle, "finalize_session", lambda *a, **k: pytest.fail("queued end finalized"))
        engine.on_session_end("S1", view)
    assert_off_current_cold_retry(engine, view, monkeypatch, caplog)


def committed_rows(path, session):
    with sqlite3.connect(path) as reader:
        reader.row_factory = sqlite3.Row
        return [dict(row) for row in reader.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY store_id", (session,))]


def home_rebound_cold_tail(engines, monkeypatch, *, shared=False, origin="current", large=False, reused_uid=False,
                           tool_outputs=None):
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    home = Path(engine._hermes_home)
    other_home = home.parent / "other-home"
    other_home.mkdir()
    _state_db(other_home, [("S1", "other-root", None), ("other-root", None, "compression"),
                           ("S2", None, None)])
    if not shared:
        engine._config.database_path = ""
    engine.on_session_start("S1", platform="cli", conversation_id="conv1")
    view = [_m("user", "A1", 1., "a1"), _m("assistant", "A2", 2., "a2"),
            _m("user", "A3", 3., "a3")]
    if reused_uid:
        view[-1]["message_uid"] = "a1"
    if large:
        view[1]["tool_calls"] = [{"id": "call", "type": "function",
                                  "function": {"name": "shell", "arguments": "{}"}}]
        view.insert(2, _m("tool", "large output " * 4000, 2.5, "tool-result",
                          tool_call_id="call", tool_name="shell"))
    if tool_outputs is not None:
        outputs = tool_outputs(home)
        view[1]["tool_calls"] = [{"id": row["tool_call_id"], "type": "function",
                                  "function": {"name": "shell", "arguments": "{}"}} for row in outputs]
        view[2:2] = outputs
    engine.ingest(copy.deepcopy(view[:1]))
    store = engine._store
    if origin == "off_current":
        engine.on_session_start("S2", platform="api", conversation_id="conv2")

    def failed(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    locker = sqlite3.connect(store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(store, "get_range" if origin == "off_current" else "get_session_count", failed)
            engine._ingest_cursor_needs_reconcile = True
            engine.on_session_end("S1", copy.deepcopy(view))
        assert len(engine._pending_session_tails) == 1
        tail = engine._pending_session_tails[0]
        assert tail["kind"] == "cold" and tail["view"] == view
        deferred = count(engine, "tail.retry.deferred")
        rebind = engine._rebind_storage_for_home

        def unlocked_rebind(new_home):
            # The pre-bind retry has already observed the lock. A shared DB's lifecycle bind needs it free.
            locker.rollback()
            return rebind(new_home)

        with monkeypatch.context() as patch:
            patch.setattr(engine, "_rebind_storage_for_home", unlocked_rebind)
            engine.on_session_start("S2", hermes_home=str(other_home), platform="api", conversation_id="conv2")
        assert count(engine, "tail.retry.deferred") == deferred + 1
        assert engine._pending_session_tails == [tail] and store._conn is not None
    finally:
        locker.rollback()
        locker.close()
    assert (engine._store is store) is shared
    return engine, tail, view, home, other_home


@pytest.mark.parametrize("origin", ["current", "off_current"])
@pytest.mark.parametrize("reused_uid", [False, True])
def test_cold_tail_after_home_rebind_is_stored_in_its_own_store(engines, monkeypatch, origin, reused_uid):
    engine, tail, view, home, _ = home_rebound_cold_tail(
        engines, monkeypatch, origin=origin, reused_uid=reused_uid)
    store = tail["store"]
    active_watch, active_overrides = engine._host_rewrite_state()
    with sqlite3.connect(store.db_path) as reader:
        before = reader.execute("SELECT count(*) FROM occurrence_bindings").fetchone()[0]
    foreign_reads = []
    read = engine._store.read_metadata_json_many

    def observed(*args, **kwargs):
        foreign_reads.append(args)
        return read(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "read_metadata_json_many", observed)
        engine._retry_pending_session_tails()
        engine._retry_pending_session_tails()
    assert not foreign_reads
    if reused_uid:
        assert engine._host_rewrite_store is engine._store
        assert engine._host_rewrite_watch is active_watch and engine._host_rewrite_overrides is active_overrides
    assert [r["content"] for r in committed_rows(store.db_path, "S1")] == [m["content"] for m in view]
    assert not committed_rows(engine._store.db_path, "S1")
    assert not committed_rows(engine._store.db_path, "S2")
    assert not engine._pending_session_tails and count(engine, "tail.cold_detached_stored") == 1
    assert count(engine, "tail.cold_store_unbound") == 0 and store._conn is None
    with sqlite3.connect(store.db_path) as reader:
        assert reader.execute("SELECT count(*) FROM occurrence_bindings").fetchone()[0] == before
        bindings = reader.execute("SELECT lineage_key, uid FROM host_uid_bindings ORDER BY rowid").fetchall()
        assert [uid for _, uid in bindings] == [m["message_uid"] for m in view]
        assert {key for key, _ in bindings} == {engine._host_uid_lineage_key("S1", hermes_home=str(home))[0]}
        assert reader.execute("SELECT value FROM metadata WHERE key=?", (HOST_UID_COUNTER_KEY,)).fetchone()
    assert engine._store.read_metadata_json(HOST_UID_COUNTER_KEY) is None


@pytest.mark.parametrize("origin", ["current", "off_current"])
def test_shared_store_home_rebind_retries_under_the_tail_home(engines, monkeypatch, origin):
    engine, tail, view, home, other_home = home_rebound_cold_tail(
        engines, monkeypatch, shared=True, origin=origin, large=True)
    engine._retry_pending_session_tails()
    engine._retry_pending_session_tails()
    rows = committed_rows(tail["store"].db_path, "S1")
    ref = extract_externalized_ref(rows[2]["content"])
    assert ref
    assert load_externalized_payload(ref, config=engine._config, hermes_home=str(home))["content"] == view[2]["content"]
    assert get_large_output_storage_dir(engine._config, str(home), create=False).exists()
    assert not get_large_output_storage_dir(engine._config, str(other_home), create=False).exists()
    assert [r["content"] for r in rows if r["role"] != "tool"] == ["A1", "A2", "A3"]
    assert not committed_rows(engine._store.db_path, "S2")
    assert not engine._pending_session_tails and count(engine, "tail.cold_detached_stored") == 1
    assert engine._store._conn is not None
    with sqlite3.connect(engine._store.db_path) as reader:
        keys = {row[0] for row in reader.execute("SELECT lineage_key FROM host_uid_bindings")}
    assert keys == {engine._host_uid_lineage_key("S1", hermes_home=str(home))[0]}


def test_shared_store_other_home_ingest_does_not_mark_cold_tail(engines, monkeypatch):
    engine, tail, view, home, other_home = home_rebound_cold_tail(engines, monkeypatch, shared=True)
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_retry_session_end_tails", lambda **kwargs: None)
        engine.on_session_start("S1", hermes_home=str(other_home), platform="api", conversation_id="conv2")
    engine.ingest([_m("user", "home B row", 10., "b1")])
    assert [r["content"] for r in committed_rows(engine._store.db_path, "S1")] == ["A1", "home B row"]
    assert not tail.get("went_on", False)
    # Return to A without draining at session start, so the retry takes the bound current-session route.
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_retry_session_end_tails", lambda **kwargs: None)
        engine.on_session_start("S1", hermes_home=str(home), platform="cli", conversation_id="conv1")
    engine._retry_pending_session_tails()
    engine._retry_pending_session_tails()
    rows = committed_rows(engine._store.db_path, "S1")
    assert [r["content"] for r in rows if r["conversation_id"] == "conv1"] == [m["content"] for m in view]
    assert [r["content"] for r in rows if r["conversation_id"] == "conv2"] == ["home B row"]
    assert not engine._pending_session_tails


def test_detached_cold_externalized_identity_preserves_active_state(engines, monkeypatch):
    from hermes_lcm import reconcile
    from tests.test_issue_10_persisted_output_spillover import _write_output
    from tests.test_reconcile_oob_message_block import OOB_BLOCK

    payload = "tail output " * 4000

    def outputs(home):
        marker = _write_output(home / "cache" / "spillover" / "call.txt", full=payload)
        return [_m("tool", content, 2.1 + i / 10, "a1", tool_call_id=f"call-{i}", tool_name="shell")
                for i, content in enumerate((payload, "result" + OOB_BLOCK,
                                              reconcile._HERMES_PRUNED_TOOL_PLACEHOLDER, marker))]

    engine, tail, view, home, _ = home_rebound_cold_tail(engines, monkeypatch, tool_outputs=outputs)
    watch, overrides = engine._host_rewrite_state()
    engine._pruned_tool_source_store = engine._store
    engine._pruned_tool_source_generation = engine._store.write_generation()
    pruned = engine._pruned_tool_source_identities = {"active": ()}
    oob = engine._out_of_band_durable_rows = {"active"}
    foreign_reads, recovered, durable = [], [], []
    recover = reconcile.recover_hermes_persisted_output_with_file_stat
    find = reconcile.find_externalized_tool_result_content_for_call

    def recover_at(content, **kwargs):
        recovered.append(kwargs["hermes_home"])
        return recover(content, **kwargs)

    def find_at(**kwargs):
        durable.append((kwargs["hermes_home"], kwargs["session_id"]))
        return find(**kwargs)

    with monkeypatch.context() as patch:
        for name in ("write_generation", "find_session_tool_rows", "get_session_tail", "get_batch"):
            original = getattr(engine._store, name)

            def observed(*args, _name=name, _original=original, **kwargs):
                foreign_reads.append(_name)
                return _original(*args, **kwargs)

            patch.setattr(engine._store, name, observed)
        patch.setattr(reconcile, "recover_hermes_persisted_output_with_file_stat", recover_at)
        patch.setattr(reconcile, "find_externalized_tool_result_content_for_call", find_at)
        engine._retry_pending_session_tails()
        engine._retry_pending_session_tails()
    assert not foreign_reads
    assert recovered and set(recovered) == {str(home)}
    assert durable and set(durable) == {(str(home), "S1")}
    assert engine._host_rewrite_store is engine._store
    assert engine._host_rewrite_watch is watch and engine._host_rewrite_overrides is overrides
    assert engine._pruned_tool_source_store is engine._store
    assert engine._pruned_tool_source_identities is pruned and pruned == {"active": ()}
    assert engine._out_of_band_durable_rows is oob and oob == {"active"}
    rows = committed_rows(tail["store"].db_path, "S1")
    assert len(rows) == len(view) and len({r["store_id"] for r in rows}) == len(view)
    ref = extract_externalized_ref(rows[2]["content"])
    assert ref and load_externalized_payload(ref, config=engine._config, hermes_home=str(home))["content"] == payload
    assert not committed_rows(engine._store.db_path, "S1")
    assert not committed_rows(engine._store.db_path, "S2")
    assert not engine._pending_session_tails and count(engine, "tail.cold_detached_stored") == 1


def test_detached_cold_tail_with_a_stamp_conflict_is_kept_and_counted(engines, monkeypatch):
    engine, tail, _, _, _ = home_rebound_cold_tail(engines, monkeypatch)
    store = tail["store"]
    with sqlite3.connect(store.db_path) as writer:
        writer.execute("UPDATE messages SET observed_at='99.0' WHERE session_id='S1'")
    engine._retry_pending_session_tails()
    assert [r["content"] for r in committed_rows(store.db_path, "S1")] == ["A1"]
    assert engine._pending_session_tails == [tail] and count(engine, "tail.cold_unmatched") == 1
    assert count(engine, "tail.cold_store_unbound") == count(engine, "tail.cold_detached_stored") == 0
    assert not committed_rows(engine._store.db_path, "S1") and store._conn is not None


def test_exit_drain_stores_a_detached_cold_tail(engines, monkeypatch):
    engine, tail, view, _, _ = home_rebound_cold_tail(engines, monkeypatch)
    engine._retry_pending_session_tails(exiting=True)
    assert [r["content"] for r in committed_rows(tail["store"].db_path, "S1")] == [m["content"] for m in view]
    assert not engine._pending_session_tails and count(engine, "tail.unflushed") == 0
    assert count(engine, "tail.cold_detached_stored") == 1 and tail["store"]._conn is None
    assert not committed_rows(engine._store.db_path, "S1")


_SUMMARY_SHAPED = "[Recent Summary (d0, node 1)]\nA3\n[Expand for details: A3]"


@pytest.mark.parametrize("uid, last", [
    pytest.param("a3", "A3", id="plain"),
    pytest.param("a1", "A3", id="reused_uid"),
    pytest.param("a3", _SUMMARY_SHAPED, id="summary_shaped"),
])
def test_shutdown_after_a_failed_rebind_stores_the_detached_tail_without_reopening(engines, monkeypatch, uid, last):
    """A detached retry reads only the tail's store, so shutdown after a failed rebind never retries the open (#2152).

    A summary-shaped user row has no bound DAG node to verify it, so it is kept as a real user row."""
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    engine._config.database_path = ""
    engine.on_session_start("S1", platform="cli", conversation_id="conv1")
    view = [_m("user", "A1", 1., "a1"), _m("assistant", "A2", 2., "a2"), _m("user", last, 3., uid)]
    engine.ingest(copy.deepcopy(view[:1]))
    store = engine._store

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(store, "get_session_count", locked)
        engine._ingest_cursor_needs_reconcile = True
        engine.on_session_end("S1", copy.deepcopy(view))
    assert [tail["kind"] for tail in engine._pending_session_tails] == ["cold"]
    attempts = []

    def failed_open(*args, **kwargs):
        attempts.append(args)
        raise sqlite3.OperationalError("injected rebind failure")

    monkeypatch.setattr(engine, "_bind_storage", failed_open)
    with pytest.raises(sqlite3.OperationalError):
        engine._rebind_storage_for_home(str(Path(engine._hermes_home).parent / "other-home"))
    engine.shutdown()
    rows = committed_rows(store.db_path, "S1")
    assert [r["content"] for r in rows] == ["A1", "A2", last]
    assert len(attempts) == 1 and not engine._pending_session_tails
    assert count(engine, "tail.cold_detached_stored") == 1
    with sqlite3.connect(store.db_path) as reader:
        provenance = reader.execute("SELECT key FROM metadata WHERE key LIKE 'real_user_scaffold_store_id:%'").fetchall()
    assert provenance == ([(f"real_user_scaffold_store_id:{rows[2]['store_id']}",)] if last != "A3" else [])


@pytest.mark.parametrize("closed", [True, False], ids=["closed_store", "failing_append"])
def test_a_detached_tail_its_store_cannot_take_stays_pending(engines, monkeypatch, closed):
    """A closed store counts cold_store_unbound; a failing append counts only retry_failed. Both keep the tail (I2)."""
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    engine._config.database_path = ""
    engine.on_session_start("S1", platform="cli", conversation_id="conv1")
    view = [_m("user", "A1", 1., "a1"), _m("assistant", "A2", 2., "a2")]
    engine.ingest(copy.deepcopy(view[:1]))
    store = engine._store

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    def malformed(*args, **kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")

    with monkeypatch.context() as patch:
        patch.setattr(store, "get_session_count", locked)
        engine._ingest_cursor_needs_reconcile = True
        engine.on_session_end("S1", copy.deepcopy(view))
    engine._rebind_storage_for_home(str(Path(engine._hermes_home).parent / "other-home"))
    assert [tail["kind"] for tail in engine._pending_session_tails] == ["cold"] and store._conn is not None
    if closed:
        store.close()
    else:
        monkeypatch.setattr(store, "_append_protected_batch", malformed)
    engine._retry_pending_session_tails()
    assert [tail["kind"] for tail in engine._pending_session_tails] == ["cold"]
    assert (count(engine, "tail.cold_store_unbound"), count(engine, "tail.retry_failed")) == ((1, 0) if closed else (0, 1))
    assert [r["content"] for r in committed_rows(store.db_path, "S1")] == ["A1"]


def test_off_current_partial_suffix_commit_then_lock(engines, monkeypatch):
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    view = off_current_end(engine)
    append = engine._append_off_current_session_end_suffix

    def partial(session, suffix, **kwargs):
        append(session, suffix[:1], **kwargs)
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_append_off_current_session_end_suffix", partial)
        engine.on_session_end("S1", copy.deepcopy(view))
    assert [r["content"] for r in committed_rows(engine._store.db_path, "S1")] == ["A", "B", "C"]
    assert len(engine._pending_session_tails) == 1 and engine._pending_session_tails[0]["kind"] == "cold"
    engine._retry_pending_session_tails()
    engine._retry_pending_session_tails()
    assert [r["content"] for r in committed_rows(engine._store.db_path, "S1")] == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails


def test_resumed_s1_after_an_off_current_tail(engines, monkeypatch):
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    view = off_current_end(engine)
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_range", fail_end_read(engine))
        engine.on_session_end("S1", copy.deepcopy(view))
    assert len(engine._pending_session_tails) == 1 and engine._pending_session_tails[0]["kind"] == "cold"
    engine.on_session_start("S1", platform="cli", conversation_id="conv1")
    engine.ingest(copy.deepcopy(view))
    assert [r["content"] for r in committed_rows(engine._store.db_path, "S1")] == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails


@pytest.mark.parametrize("binding", ["ignored", "stateless"])
@pytest.mark.parametrize("origin", ["off_current", "current"])
def test_cold_tail_retry_after_an_ignored_or_stateless_rebind_keeps_the_rows(
    engines, monkeypatch, binding, origin,
):
    engine = engines(sessions=[("S", None, None), ("S1", None, None), ("S2", None, None)])
    if origin == "off_current":
        view = off_current_end(engine)
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "get_range", fail_end_read(engine))
            engine.on_session_end("S1", copy.deepcopy(view))
    else:
        view = history()
        queue_cold(engine, monkeypatch, view)
    assert len(engine._pending_session_tails) == 1
    tail = engine._pending_session_tails[0]
    session, conversation = tail["session_id"], tail["conversation_id"]
    assert tail["kind"] == "cold" and tail["view"] == view
    assert [row["content"] for row in engine._store.get_session_messages(session)] == ["A", "B"]

    pattern_kind = "ignore" if binding == "ignored" else "stateless"
    monkeypatch.setattr(engine._config, f"{pattern_kind}_session_patterns", [session])
    monkeypatch.setattr(engine, f"_compiled_{pattern_kind}_session_patterns", compile_session_patterns([session]))
    deferred = count(engine, "tail.retry.deferred")
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        # Session start retries before binding. Keep the view pending until the filtered rebind finishes.
        engine.on_session_start(session, platform="cli", context_length=200_000, conversation_id=conversation)
        assert engine._pending_session_tails == [tail]
        assert count(engine, "tail.retry.deferred") == deferred + 1
    finally:
        locker.rollback()
        locker.close()
    assert engine._session_id == session and engine._conversation_id == conversation
    assert engine._session_ignored is (binding == "ignored")
    assert engine._session_stateless is (binding == "stateless")
    engine._retry_pending_session_tails()
    engine._retry_pending_session_tails()
    rows = engine._store.get_session_messages(session)
    assert [row["content"] for row in rows] == ["A", "B", "C", "D"]
    assert all(row["conversation_id"] == conversation for row in rows)
    assert not engine._pending_session_tails


def filtered_rebind(engine, monkeypatch, tail, binding):
    """Resume the tail's session ignored or stateless; the start's own retry is busy, so the tail stays pending."""
    session, conversation = tail["session_id"], tail["conversation_id"]
    kind = "ignore" if binding == "ignored" else "stateless"
    monkeypatch.setattr(engine._config, f"{kind}_session_patterns", [session])
    monkeypatch.setattr(engine, f"_compiled_{kind}_session_patterns", compile_session_patterns([session]))
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_start(session, platform="cli", context_length=200_000, conversation_id=conversation)
        assert engine._pending_session_tails == [tail]
    finally:
        locker.rollback()
        locker.close()
    assert engine._session_ignored is (binding == "ignored")
    assert engine._session_stateless is (binding == "stateless")


@pytest.mark.parametrize("binding", ["ignored", "stateless"])
def test_retyped_row_after_the_end_stays_pending_after_a_filtered_rebind(engines, monkeypatch, binding):
    engine = engines()
    tail = queue_cold(engine, monkeypatch, [*history(), _m("user", "Q", 5.)])
    # The session undid Q and the user sent the same text again before the retry.
    engine.ingest(copy.deepcopy([*history(), _m("user", "Q", 6.)]))
    assert tail.get("went_on") is True
    filtered_rebind(engine, monkeypatch, tail, binding)
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D", "Q"]
    # As under a storing binding, Q@5 is another occurrence: kept and counted, never taken as stored by Q@6.
    assert engine._pending_session_tails == [tail] and count(engine, "tail.cold_unmatched") >= 1


@pytest.mark.parametrize("binding", ["ignored", "stateless"])
def test_restamped_row_stays_pending_after_a_filtered_rebind(engines, monkeypatch, binding):
    engine = engines()
    # No host ids: the later stamp alone makes B a new occurrence (test_restamped_row_is_a_new_occurrence_not_covered).
    tail = queue_cold(engine, monkeypatch, [_m("user", "A", 995.), _m("user", "B", 1010.)],
                      stored=[_m("user", "A", 995.), _m("user", "B", 1000.)])
    assert not tail.get("went_on")
    filtered_rebind(engine, monkeypatch, tail, binding)
    engine._retry_pending_session_tails()
    # The binding stores nothing, so B@1010 stays pending and counted; B@1000 never stands for it.
    assert contents(engine) == ["A", "B"]
    assert engine._pending_session_tails == [tail] and count(engine, "tail.cold_unmatched") >= 1


def test_off_current_end_lock_bypass_lineage_unchanged(engines, monkeypatch):
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    view = off_current_end(engine)
    engine._mark_lcm_bypass_lineage_session("S1", platform="cron")
    fallback = []
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_range", fail_end_read(engine))
        patch.setattr(engine, "_end_host_fallback_compressor_for_session", lambda *a, **k: fallback.append((a, k)))
        engine.on_session_end("S1", view)
    assert fallback == [(("S1", view), {"current_session_bypasses": False})]
    assert not engine._pending_session_tails and count(engine, "tail.busy_at_end") == 0
    assert [row["content"] for row in engine._store.get_session_messages("S1")] == ["A", "B"]


def test_off_current_end_non_busy_prefix_error_unchanged(engines, monkeypatch):
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    view = off_current_end(engine)
    fallback = []
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_range", fail_end_read(engine, error=ValueError))
        patch.setattr(engine, "_end_host_fallback_compressor_for_session", lambda *a, **k: fallback.append((a, k)))
        engine.on_session_end("S1", view)
    assert fallback == [(("S1", view), {"current_session_bypasses": False})]
    assert not engine._pending_session_tails and count(engine, "tail.busy_at_end") == 0
    assert [row["content"] for row in engine._store.get_session_messages("S1")] == ["A", "B"]


@pytest.mark.parametrize("error", [sqlite3.OperationalError, ValueError])
def test_1805_locked_count_read_queues_cold_tail(engines, monkeypatch, caplog, error):
    engine = engines()
    view = history()
    tail = queue_cold(engine, monkeypatch, view, error=error)
    assert tail["view"] is not view and tail["view"][0] is not view[0]
    assert count(engine, "tail.queued.cold." + error.__name__) == 1
    assert "retry pending" in caplog.text and error.__name__ in caplog.text
    view[-1]["content"] = "host mutation"
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails


@pytest.mark.parametrize("reconcile", [False, True])
def test_1805_locked_tail_read_queues_cold_tail(engines, monkeypatch, reconcile):
    engine = engines()
    queue_cold(engine, monkeypatch, read="get_session_tail", reconcile=reconcile)
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails


def test_dedupe_hit_queues_no_cold_tail(engines):
    engine = engines()
    queue(engine)
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("S", history())
        assert len(engine._pending_session_tails) == 1
        assert engine._pending_session_tails[0]["kind"] == "warm"
        assert count(engine, "tail.queued.cold.OperationalError") == 0
    finally:
        locker.rollback()
        locker.close()


def test_second_end_after_rolled_back_append_queues_cold(engines, monkeypatch):
    engine = engines()
    engine.ingest(history()[:2])
    append, attempts = engine._append_ingest_batch, []

    def busy_once(batch):
        attempts.append(True)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database is locked")
        return append(batch)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_append_ingest_batch", busy_once)
        engine.on_session_end("S", history())
    assert attempts == [True, True] and not engine._pending_session_tails
    # The first attempt reached append, but its warm queue rolled back.
    assert engine.__dict__.get("_frozen_tail_reached") is True
    view = [*history(), _m("user", "E", 5., "E")]
    queue_cold(engine, monkeypatch, view)
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D", "E"]


def test_cold_tail_retried_after_other_ingests(engines, monkeypatch):
    engine = engines()
    queue_cold(engine, monkeypatch)
    engine.ingest(history()[:3])
    replay(engine, [_m("user", "other", 9.)], session="other", conversation="other")
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails


def longer(stamped):
    stamp = (lambda value: value) if stamped else (lambda value: None)
    return [*history(stamped), _m("user", "E", stamp(5.), "E"), _m("assistant", "F", stamp(6.), "F")], stamp


@pytest.mark.parametrize("stamped", [True, False])
@pytest.mark.parametrize("then", ["ingest", "end"])
def test_cold_view_stored_by_later_ingests_is_not_stored_again(engines, monkeypatch, stamped, then):
    engine = engines()
    queue_cold(engine, monkeypatch, history(stamped))
    view, stamp = longer(stamped)
    engine.ingest(copy.deepcopy(view))
    view.append(_m("user", "G", stamp(7.), "G"))
    if then == "ingest":
        engine._retry_pending_session_tails()
        engine.ingest(copy.deepcopy(view))
    else:
        engine.on_session_end("S", copy.deepcopy(view))
    assert contents(engine) == ["A", "B", "C", "D", "E", "F", "G"]
    assert not engine._pending_session_tails


@pytest.mark.parametrize("stamped", [True, False])
def test_resumed_session_stores_the_cold_view_once(engines, monkeypatch, stamped):
    old = engines()
    queue_cold(old, monkeypatch, history(stamped))
    resumed = LCMEngine(config=old._config, hermes_home=old._hermes_home)
    try:
        locker = sqlite3.connect(old._store.db_path, isolation_level=None)
        locker.execute("BEGIN IMMEDIATE")
        try:  # The resumed engine's sibling retry defers behind the lock.
            resumed.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
        finally:
            locker.rollback()
            locker.close()
        assert len(old._pending_session_tails) == 1
        resumed.ingest(longer(stamped)[0])
        old._retry_pending_session_tails(exiting=True)  # the ended engine's exit drain
    finally:
        resumed.shutdown()
    assert contents(old) == ["A", "B", "C", "D", "E", "F"]
    assert not old._pending_session_tails


@pytest.mark.parametrize("stamped", [True, False])
def test_ignored_row_view_after_live_ingest_is_kept_without_duplicates(engines, monkeypatch, stamped):
    engine = engines()
    engine._config.ignore_message_patterns = ["^ignore$"]
    engine._compiled_ignore_message_patterns = [_ContainsPattern("ignore")]
    stamp = (lambda value: value) if stamped else (lambda value: None)
    # An ignored row is never stored, so the stored prefix differs from the view position by position.
    view = [*history(stamped)[:2], _m("user", "ignore", stamp(2.5)), *history(stamped)[2:]]
    queue_cold(engine, monkeypatch, view)
    live = [*view, _m("user", "E", stamp(5.), "E"), _m("assistant", "F", stamp(6.), "F")]
    engine.ingest(copy.deepcopy(live))
    engine._retry_pending_session_tails()
    engine.ingest(copy.deepcopy([*live, _m("user", "G", stamp(7.), "G")]))
    assert contents(engine) == ["A", "B", "C", "D", "E", "F", "G"]
    # The session ingested after the end, so the differing view is kept and counted, never stored again.
    assert len(engine._pending_session_tails) == 1 and count(engine, "tail.cold_unmatched") >= 1


@pytest.mark.parametrize("stamped", [True, False])
@pytest.mark.parametrize("then", ["ingest", "end"])
def test_live_session_that_rewrote_the_cold_view_loses_and_repeats_nothing(engines, monkeypatch, stamped, then):
    engine = engines()
    stamp = (lambda value: value) if stamped else (lambda value: None)
    queue_cold(engine, monkeypatch, [*history(stamped), _m("user", "Q1", stamp(5.), "Q1"),
                                     _m("assistant", "Q2", stamp(6.), "Q2")])
    # The same session undid Q1 and Q2 and went on before the retry ran.
    live = [*history(stamped), _m("user", "E", stamp(7.), "E")]
    engine.ingest(copy.deepcopy(live))
    engine._retry_pending_session_tails()
    live.append(_m("assistant", "F", stamp(8.), "F"))
    if then == "ingest":
        engine.ingest(copy.deepcopy(live))
    else:
        engine.on_session_end("S", copy.deepcopy(live))
    assert contents(engine) == ["A", "B", "C", "D", "E", "F"]
    assert [tail["kind"] for tail in engine._pending_session_tails] == ["cold"]  # Q1 and Q2 stay queued


def test_retyped_row_after_the_end_keeps_the_cold_view(engines, monkeypatch):
    engine = engines()
    queue_cold(engine, monkeypatch, [*history(), _m("user", "Q", 5.)])
    # The session undid Q and the user sent the same text again before the retry.
    engine.ingest(copy.deepcopy([*history(), _m("user", "Q", 6.)]))
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D", "Q"]
    # Q@5 is another occurrence: kept and counted, never taken as stored by Q@6.
    assert len(engine._pending_session_tails) == 1 and count(engine, "tail.cold_unmatched") >= 1


@pytest.mark.parametrize("stamped", [True, False])
def test_session_that_rewrote_the_view_after_a_cold_start_loses_nothing(engines, monkeypatch, stamped):
    engine = engines()
    stamp = (lambda value: value) if stamped else (lambda value: None)
    queue_cold(engine, monkeypatch, [*history(stamped), _m("user", "Q1", stamp(5.), "Q1"),
                                     _m("assistant", "Q2", stamp(6.), "Q2")])
    engine._retry_pending_session_tails()  # nothing ingested since the end: the cold start stores the view
    # The host then undid Q1 and Q2 and went on with as many rows as the stored view.
    engine.ingest(copy.deepcopy([*history(stamped), _m("user", "E", stamp(7.), "E"),
                                 _m("assistant", "F", stamp(8.), "F")]))
    # Unstamped, the next ingest cannot place C and D, so it stores them again (#2203): duplication, never loss.
    assert contents(engine) == ["A", "B", "C", "D", "Q1", "Q2", *([] if stamped else ["C", "D"]), "E", "F"]
    assert not engine._pending_session_tails


def ignoring(engine):
    engine._config.ignore_message_patterns = ["^ignore$"]
    engine._compiled_ignore_message_patterns = [_ContainsPattern("ignore")]
    return engine


@pytest.mark.parametrize("stamped", [True, False])
def test_frozen_run_that_moved_its_own_cursor_is_not_taken_for_progress(engines, monkeypatch, stamped):
    engine = ignoring(engines())
    # The ignored row sits inside the stored prefix, so a prefix compare can never match this view.
    view = [history(stamped)[0], _m("user", "ignore", 1.5 if stamped else None), *history(stamped)[1:]]
    engine.ingest(copy.deepcopy(view[:3]))
    restart(engine)  # a resumed session: its cursor is reconciled at the next ingest
    tail_read = engine._store.get_session_tail

    def locked_before_append(*args, **kwargs):
        if any(frame.function == "_store_ingest_rows" for frame in inspect.stack()[1:4]):
            raise sqlite3.OperationalError("database is locked")
        return tail_read(*args, **kwargs)

    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with monkeypatch.context() as patch:
            # The frozen run reconciles its own cursor, then fails before its append.
            patch.setattr(engine._store, "get_session_tail", locked_before_append)
            engine.on_session_end("S", copy.deepcopy(view))
    finally:
        locker.rollback()
        locker.close()
    assert [tail["kind"] for tail in engine._pending_session_tails] == ["cold"]
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"] and not engine._pending_session_tails


def test_restart_whose_retry_waited_is_not_taken_for_progress(engines, monkeypatch):
    engine = ignoring(engines())
    view = [history()[0], _m("user", "ignore", 1.5), *history()[1:]]
    queue_cold(engine, monkeypatch, view, stored=view[:3])
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        restart(engine)  # the start's retry waits on the lock; the rebind restarts the cursor
    finally:
        locker.rollback()
        locker.close()
    assert [tail["kind"] for tail in engine._pending_session_tails] == ["cold"]
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"] and not engine._pending_session_tails


def test_reset_after_the_end_is_not_taken_for_progress(engines, monkeypatch):
    engine = ignoring(engines())
    view = [history()[0], _m("user", "ignore", 1.5), *history()[1:]]
    queue_cold(engine, monkeypatch, view, stored=view[:3])
    # Hermes /new: the reset drops the reconcile the failed end asked for, then the next start retries the tail.
    engine.on_session_reset()
    engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv")
    assert contents(engine) == ["A", "B", "C", "D"] and not engine._pending_session_tails


def test_cold_start_of_an_earlier_end_is_not_progress_for_a_later_one(engines, monkeypatch):
    engine = ignoring(engines())
    view = [history()[0], _m("user", "ignore", 1.5), *history()[1:]]
    engine.ingest(copy.deepcopy(view[:3]))
    restart(engine)

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "get_session_count", locked)
            for end in (view, [*view, _m("user", "E", 5.)]):
                engine.on_session_end("S", copy.deepcopy(end))
    finally:
        locker.rollback()
        locker.close()
    assert [tail["kind"] for tail in engine._pending_session_tails] == ["cold", "cold"]
    # The first tail's cold start stores an older view than the second's, which still needs its own cold start.
    # A wide deadline: both retries share the default 50 ms budget, which a slow runner can spend on the first.
    engine._retry_pending_session_tails(deadline=time.monotonic() + 5)
    assert contents(engine) == ["A", "B", "C", "D", "E"] and not engine._pending_session_tails


@pytest.mark.parametrize("stamped", [True, False])
def test_session_that_went_on_then_restarted_before_the_retry_repeats_nothing(engines, monkeypatch, stamped):
    engine = engines()
    stamp = (lambda value: value) if stamped else (lambda value: None)
    queue_cold(engine, monkeypatch, [*history(stamped), _m("user", "Q1", stamp(5.), "Q1"),
                                     _m("assistant", "Q2", stamp(6.), "Q2")])
    live = [*history(stamped), _m("user", "E", stamp(7.), "E")]  # Q1 and Q2 undone
    engine.ingest(copy.deepcopy(live))
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        restart(engine)  # the start's retry waits on the lock; the rebind asks for a reconcile again
    finally:
        locker.rollback()
        locker.close()
    engine._retry_pending_session_tails()
    engine.ingest(copy.deepcopy([*live, _m("assistant", "F", stamp(8.), "F")]))
    assert contents(engine) == ["A", "B", "C", "D", "E", "F"]
    assert [tail["kind"] for tail in engine._pending_session_tails] == ["cold"]  # Q1 and Q2 stay queued


def test_second_cold_tail_sees_an_ingest_committed_before_the_drain(engines, monkeypatch):
    engine = engines()
    engine.ingest(copy.deepcopy(history()[:2]))
    restart(engine)

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "get_session_count", locked)
            for view in (history(), [*history(), _m("user", "Q", 5.)]):
                engine.on_session_end("S", copy.deepcopy(view))
    finally:
        locker.rollback()
        locker.close()
    assert [tail["kind"] for tail in engine._pending_session_tails] == ["cold", "cold"]
    # The user sent Q again before any retry, and the session ingested it. Retrying the first tail asks for a
    # reconcile again, which must not hide that from the second.
    engine.ingest(copy.deepcopy([*history(), _m("user", "Q", 6.)]))
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D", "Q"]
    assert [tail["view"][-1]["timestamp"] for tail in engine._pending_session_tails] == [5.]


def test_restamped_row_is_a_new_occurrence_not_covered(engines, monkeypatch):
    engine = engines()
    # No host ids: the later stamp alone makes B a new user occurrence (reconcile.py's #1511 rewind).
    queue_cold(engine, monkeypatch, [_m("user", "A", 995.), _m("user", "B", 1010.)],
               stored=[_m("user", "A", 995.), _m("user", "B", 1000.)])
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "B"]
    assert not engine._pending_session_tails


def test_covered_view_reconciles_the_live_cursor(engines, monkeypatch):
    engine = engines()
    queue_cold(engine, monkeypatch, history(False))
    engine.ingest(copy.deepcopy(history(False)[:3]))
    replay(engine, history(False)[3:])  # another writer commits D
    engine._retry_pending_session_tails()
    assert not engine._pending_session_tails
    engine.ingest(copy.deepcopy([*history(False), _m("user", "E", None, "E")]))
    assert contents(engine) == ["A", "B", "C", "D", "E"]


@pytest.mark.parametrize("stamped", [True, pytest.param(False, marks=pytest.mark.xfail(
    strict=True, reason="#2203: an unstamped cold view that a resumed session rewrote is stored again"))])
def test_resumed_session_that_rewrote_the_cold_view(engines, monkeypatch, stamped):
    old = engines()
    stamp = (lambda value: value) if stamped else (lambda value: None)
    queue_cold(old, monkeypatch, [*history(stamped), _m("user", "Q", stamp(5.), "Q")])
    resumed = LCMEngine(config=old._config, hermes_home=old._hermes_home)
    try:
        locker = sqlite3.connect(old._store.db_path, isolation_level=None)
        locker.execute("BEGIN IMMEDIATE")
        try:
            resumed.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
        finally:
            locker.rollback()
            locker.close()
        # The resumed session undid Q and went on; the ended engine's exit drain runs afterwards.
        resumed.ingest([*history(stamped), _m("user", "E", stamp(6.), "E"), _m("assistant", "F", stamp(7.), "F")])
        old._retry_pending_session_tails(exiting=True)
    finally:
        resumed.shutdown()
    assert contents(old) == ["A", "B", "C", "D", "E", "F", "Q"]


@pytest.mark.parametrize("frozen", ["read", "recovery"])
def test_frozen_run_after_a_rolled_back_append_queues_cold(engines, monkeypatch, frozen):
    engine = engines()
    engine.ingest(copy.deepcopy(history()[:2]))
    store = engine._store
    atomic, append, read = store.ingest_atomic, engine._append_ingest_batch, store.get_session_count
    recover = engine._identity_anchor_observe_recovery
    calls = {"atomic": 0, "append": 0}

    def busy_second_begin(**kwargs):
        calls["atomic"] += 1
        if calls["atomic"] == 2:
            raise sqlite3.OperationalError("database is locked")
        return atomic(**kwargs)

    def busy_first_append(batch):
        calls["append"] += 1
        if calls["append"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return append(batch)

    def busy_frozen_read(*args, **kwargs):
        if engine.__dict__.get("_ingest_read_only") is not None:
            raise sqlite3.OperationalError("database is locked")
        return read(*args, **kwargs)

    def failed_frozen_recovery():
        if engine.__dict__.get("_ingest_read_only") is not None:
            raise ValueError("injected recovery failure")
        return recover()

    engine._ingest_cursor_needs_reconcile = True
    with monkeypatch.context() as patch:
        patch.setattr(store, "ingest_atomic", busy_second_begin)
        patch.setattr(engine, "_append_ingest_batch", busy_first_append)
        if frozen == "read":
            patch.setattr(store, "get_session_count", busy_frozen_read)
        else:
            patch.setattr(engine, "_identity_anchor_observe_recovery", failed_frozen_recovery)
        engine.on_session_end("S", copy.deepcopy(history()))
    # The first attempt reached append and its warm queue rolled back; the second failed at BEGIN and its
    # frozen run before any append: at the reconcile read, or before its transaction began.
    assert calls == {"atomic": 2, "append": 1}
    assert [tail["kind"] for tail in engine._pending_session_tails] == ["cold"]
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]


@pytest.mark.parametrize("case", ["unreplayed", "covered", "longer_store", "mismatch_kept"])
def test_cold_off_current_s3(engines, monkeypatch, caplog, case):
    engine = engines()
    tail = queue_cold(engine, monkeypatch)
    if case in {"covered", "longer_store"}:
        replay(engine, history()[2:])
    if case == "longer_store":
        replay(engine, [_m("user", "new", 8.)])
    if case == "mismatch_kept":
        engine._store._conn.execute("UPDATE messages SET content='mismatch' WHERE store_id=1")
        engine._store._conn.commit()
    engine._session_id = "next"
    before_views = sql_rows(engine, "ingest_views")
    append = engine._append_off_current_session_end_suffix

    def owned(*args, **kwargs):
        assert engine._store._transaction_owner.owns_connection(engine._store._conn)
        return append(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_append_off_current_session_end_suffix", owned)
        patch.setattr(engine._lifecycle, "finalize_session", lambda *a, **k: pytest.fail("retry finalized"))
        engine._retry_pending_session_tails()
        engine._retry_pending_session_tails()
    if case == "mismatch_kept":
        assert engine._pending_session_tails == [tail]
        assert contents(engine) == ["mismatch", "B"]
        assert count(engine, "tail.cold_unmatched") == 2
        assert sum("cold tail prefix mismatch" in r.getMessage() for r in caplog.records) == 1
    else:
        assert contents(engine) == ["A", "B", "C", "D"] + (["new"] if case == "longer_store" else [])
        assert not engine._pending_session_tails
    assert sql_rows(engine, "ingest_views") == before_views


@pytest.mark.parametrize("route", ["sibling", "unbound_store", "different_conversation"])
def test_sibling_cold_uses_s3(engines, monkeypatch, tmp_path, route):
    engine = engines()
    tail = queue_cold(engine, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_ingest_messages", lambda *a, **k: pytest.fail("cold current route"))
        if route == "sibling":
            sibling = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
            try:
                sibling.on_session_start("next", platform="cli")
            finally:
                sibling.shutdown()
        elif route == "different_conversation":
            engine._conversation_id = "other"
            engine._retry_pending_session_tails()
        else:
            rebound = MessageStore(tmp_path / "rebound.db")
            try:
                patch.setattr(engine, "_store", rebound)
                engine._retry_pending_session_tails()
                assert not engine._pending_session_tails
                assert count(engine, "tail.cold_store_unbound") == 0
                assert [r["content"] for r in committed_rows(tail["store"].db_path, "S")] == ["A", "B", "C", "D"]
                assert tail["store"]._conn is None
                assert not rebound.get_session_messages("S")
            finally:
                rebound.close()
    if route != "unbound_store":
        assert contents(engine) == ["A", "B", "C", "D"]
        assert not engine._pending_session_tails


def test_cold_retry_busy_never_nests(engines, monkeypatch):
    engine = engines()
    tail = queue_cold(engine, monkeypatch)
    engine._session_end_flush_active = True
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine._retry_pending_session_tails(exiting=True)
        assert engine._pending_session_tails == [tail]
        assert not engine._session_end_flush_active
        assert count(engine, "tail.retry.deferred") == 1
        assert count(engine, "tail.unflushed") == len(history())
    finally:
        locker.rollback()
        locker.close()
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]


def test_poisoned_tail_does_not_block(engines, monkeypatch, caplog):
    engine = engines()
    tail = queue(engine)
    poison = {**tail, "messages": [_m("user", "poison")]}
    engine._pending_session_tails.insert(0, poison)
    match = engine._match_pending_tail

    def failed(candidate):
        if candidate is poison:
            raise ValueError("poisoned tail")
        return match(candidate)

    with monkeypatch.context() as patch, caplog.at_level(logging.WARNING):
        patch.setattr(engine, "_match_pending_tail", failed)
        engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]
    assert engine._pending_session_tails == [poison]
    assert count(engine, "tail.retry_failed") == 1
    assert "ValueError" in caplog.text
    engine._pending_session_tails.clear()  # Synthetic malformed input is not a recoverable host view.


def test_cold_current_writes_no_view(engines, monkeypatch):
    engine = engines()
    engine._config.ignore_message_patterns = ["^ignore$"]
    engine._compiled_ignore_message_patterns = [_ContainsPattern("ignore")]
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["api_key"]
    view = [*history()[:2], _m("user", "ignore", 3.),
            _m("assistant", "api_key=sk-user1234567890abcdef", 4.)]
    queue_cold(engine, monkeypatch, view)
    before_views = sql_rows(engine, "ingest_views")
    with monkeypatch.context() as patch:
        patch.setattr(engine._lifecycle, "finalize_session", lambda *a, **k: pytest.fail("retry finalized"))
        engine._retry_pending_session_tails()
    assert sql_rows(engine, "ingest_views") == before_views
    assert len(sql_rows(engine, "occurrence_bindings")) == 3
    assert contents(engine)[:2] == ["A", "B"]
    assert len(contents(engine)) == 3 and contents(engine)[-1] != view[-1]["content"]
    assert count(engine, "views_written") == len(before_views)
    assert not engine._pending_session_tails


@pytest.mark.parametrize("error", [KeyboardInterrupt, ValueError])
def test_committed_uid_disagreement_tail_removed_after_counter_interrupt(engines, monkeypatch, error):
    engine = engines()
    queue(engine)
    replay(engine, [{**row, "message_uid": "foreign-" + row["message_uid"]} for row in history()[2:]])
    counter, interrupted = engine._ledger_count, []

    def after_commit(key, *args, **kwargs):
        if key == "ledger.tail.retry.replay_rows" and len(contents(engine)) == 6 and not interrupted:
            interrupted.append(True)
            raise error("after COMMIT")
        return counter(key, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_ledger_count", after_commit)
        engine._retry_pending_session_tails()
    assert interrupted == [True]
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D", "C", "D"]
    assert not engine._pending_session_tails
