"""P1c-2a: failed exit writes recover atomically and remain visible until then."""
import copy
from contextlib import contextmanager
from pathlib import Path
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_lcm import ingest_ledger as ledger
from hermes_lcm.sqlite_util import _SqliteSavepointError
from tests.test_host_uid_shadow import _engine, _state_db
from tests.test_issue_2055_emission_writes import adopt, compress_path, emission_views, emissions
from tests.test_issue_2055_ingest_ledger_writes import SALT, count, engines as _engines, sql_rows

engines = _engines


@pytest.fixture(autouse=True)
def isolated_marks(monkeypatch):
    monkeypatch.setattr(ledger, "_PENDING_EMISSIONS", {})


def pending(engine):
    identity = engine._store.db_identity
    with ledger._PENDING_EMISSIONS_LOCK:
        return [entry for (key, _lineage), entries in ledger._PENDING_EMISSIONS.items()
                if key[0] == identity for entry in entries]


def output(label="generated"):
    return [dict(role="user", content=label, timestamp=10.),
            dict(role="assistant", content="ordinary host row", timestamp=11.)]


def emit(engine, rows):
    engine._compress_exit_write(rows, (rows, [dict(kind="summary", output_occurrence=dict(index=0, of=len(rows)))], None))


def failing(*args, **kwargs):
    raise ValueError("emission write sentinel")


def fail_exit(engine, monkeypatch, rows=None, site="insert_emissions"):
    rows = output() if rows is None else rows
    with monkeypatch.context() as patch:
        patch.setattr(ledger, site, failing)
        emit(engine, rows)
    assert len(pending(engine)) == 1
    return rows


def stored_shape(engine):
    return [(r["role"], r["content"], r.get("observed_at")) for r in engine._store.get_session_messages("S")]


def unproven(engine):
    return [r for r in sql_rows(engine, "message_relations") if r["kind"] == "generated_unproven"]


@pytest.mark.parametrize("other_store_pending", [False, True])
def test_empty_store_registry_retry_does_not_touch_ledger(engines, monkeypatch, other_store_pending):
    engine = engines()
    if other_store_pending:
        fail_exit(engines("other"), monkeypatch)
    statements, attempt, callbacks = [], {}, []
    with monkeypatch.context() as patch:
        for name in ("ingest_ledger_state", "ingest_ledger_salt"):
            patch.setattr(engine._store, name, lambda: pytest.fail("unexpected store read"))
        engine._store._conn.set_trace_callback(statements.append)
        try:
            engine._retry_pending_emissions(attempt, callbacks, object())
        finally:
            engine._store._conn.set_trace_callback(None)
    assert attempt["retried_emissions"] == [] and callbacks == statements == []
    monkeypatch.setattr(ledger, "pending_emission_digests", lambda *args: pytest.fail("unexpected pending lookup"))
    monkeypatch.setattr(engine._store, "add_message_relations", lambda groups: pytest.fail("unexpected relation call"))
    engine._ingest_messages(output())
    assert count(engine, "views_written") == 1


@pytest.mark.parametrize("site", ["insert_emissions", "write_view"])
def test_t1_write_failure_retries_exact_payload(engines, monkeypatch, site):
    engine, control = engines("failure"), engines("control")
    rows = fail_exit(engine, monkeypatch, site=site)
    entry = pending(engine)[0]
    assert count(engine, "emission_unproven_set") == 1
    assert entry["origin"]() is engine and entry["created_at"] > 0
    assert all(e.ref is None and not e.is_descriptor for e in entry["entries"])
    emit(control, rows)
    engine._ingest_messages(copy.deepcopy(rows))
    fields = ("kind", "output_index", "descriptor_digest")
    assert [[r[k] for k in fields] for r in emissions(engine)] == [[r[k] for k in fields] for r in emissions(control)]
    assert [ledger.decode_entries(v["entries"]) for v in emission_views(engine)] == [
        ledger.decode_entries(v["entries"]) for v in emission_views(control)]
    assert pending(engine) == [] and count(engine, "emission_retry_ok") == 1
    assert unproven(engine) == []
    engine._ingest_messages(copy.deepcopy(rows))
    assert len(emissions(engine)) == 1 and count(engine, "emission_retry_ok") == 1


@pytest.mark.parametrize("site", ["insert_emissions", "write_view"])
def test_t2_retry_failure_commits_identical_rows_and_exact_relations(engines, monkeypatch, caplog, site):
    engine, control = engines("failure"), engines("control")
    rows = fail_exit(engine, monkeypatch)
    with monkeypatch.context() as patch:
        if site == "insert_emissions":
            patch.setattr(ledger, site, failing)
        else:
            write_view = ledger.write_view

            def fail_emission(conn, salt, lineage, kind, *args, **kwargs):
                if kind == "emission":
                    failing()
                return write_view(conn, salt, lineage, kind, *args, **kwargs)

            patch.setattr(ledger, site, fail_emission)
        engine._ingest_messages(copy.deepcopy(rows))
    control._ingest_messages(copy.deepcopy(rows))
    assert stored_shape(engine) == stored_shape(control)
    assert count(engine, "emission_retry_failed") == 1
    assert len(pending(engine)) == 1 and pending(engine)[0]["claim"] is None
    digest = pending(engine)[0]["rows"][0]["descriptor_digest"]
    expected = {r["store_id"] for r in engine._store.get_session_messages("S")
                if ledger.identity_digest(SALT, r) == digest}
    assert expected and {r["store_id"] for r in unproven(engine)} == expected
    assert len(unproven(engine)) == len(expected) == count(engine, "generated_unproven_rows")
    assert "ledger.emission_retry_failed (ValueError); ingest unaffected" in caplog.text
    assert emissions(engine) == emission_views(engine) == []


@pytest.mark.parametrize("case", ["busy", "in_transaction"])
def test_t3_other_attempted_write_failures_retry(engines, case):
    engine, rows = engines(), output()
    if case == "busy":
        with sqlite3.connect(engine._store.db_path) as locker:
            locker.execute("BEGIN IMMEDIATE")
            emit(engine, rows)
        assert count(engine, "emission_busy") == 1
    else:
        with engine._store.atomic(immediate=True):
            emit(engine, rows)
        assert count(engine, "emission_skipped_in_transaction") == 1
    assert len(pending(engine)) == 1 and count(engine, "emission_unproven_set") == 1
    engine._ingest_messages(rows)
    assert len(emissions(engine)) == 1 and not pending(engine)


@pytest.mark.parametrize("case", ["prepare_failed", "not_ready", "owner_not_ready", "salt_changed"])
def test_t4_unattempted_writes_do_not_mark(engines, monkeypatch, case):
    engine, rows = engines(), output()
    if case == "prepare_failed":
        monkeypatch.setattr(engine, "_ledger_view_entries", failing)
    elif case in {"not_ready", "owner_not_ready"}:
        reads = iter(["ready", "absent"] if case == "owner_not_ready" else ["absent"])
        monkeypatch.setattr(engine._store, "ingest_ledger_state", lambda: next(reads))
    else:
        reads = iter([SALT, b"changed" * 4])
        # Avoid the state's own salt read so the second read is the owner check.
        monkeypatch.setattr(engine._store, "ingest_ledger_state", lambda: "ready")
        monkeypatch.setattr(engine._store, "ingest_ledger_salt", lambda: next(reads))
    emit(engine, rows)
    assert pending(engine) == [] and count(engine, "emission_unproven_set") == 0
    key = "emission_prepare_failed" if case == "prepare_failed" else "salt_changed" if case == "salt_changed" else "emission_skipped_not_ready"
    assert count(engine, key) == 1


@pytest.mark.parametrize("reason", ["no_capture", "list_mismatch", "stale", "no_session", "non_list_output",
                                   "reconcile_pending", "cursor_mismatch", "identity_noop", "construction_error"])
def test_t4_no_descriptors_never_mark(engines, reason):
    engine, rows = engines(), output()
    capture = (rows, [], reason)
    if reason == "no_capture":
        capture = None
    elif reason == "list_mismatch":
        capture = (list(rows), [], None)
    elif reason == "stale":
        capture = (rows, [dict(kind="summary", output_occurrence=dict(index=99, of=2))], None)
    engine._compress_exit_write(rows, capture)
    assert pending(engine) == [] and count(engine, "emission_no_descriptors." + reason) == 1


def test_t5_bound_keeps_newest_two(engines, monkeypatch):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        for label in ("old", "middle", "new"):
            emit(engine, output(label))
    assert [e["rows"][0]["descriptor_digest"] for e in pending(engine)] == [
        ledger.identity_digest(SALT, output(label)[0]) for label in ("middle", "new")]
    assert count(engine, "emission_unproven_set") == 3 and count(engine, "emission_unproven_dropped") == 1


def test_pending_relation_is_limited_to_this_ingests_lineage(engines, monkeypatch):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    entry = pending(engine)[0]
    with ledger._PENDING_EMISSIONS_LOCK:
        key = next(iter(ledger._PENDING_EMISSIONS))
        ledger._PENDING_EMISSIONS[(key[0], "different-lineage")] = ledger._PENDING_EMISSIONS.pop(key)
        entry["lineage"] = "different-lineage"
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        engine._ingest_messages(rows)
    assert stored_shape(engine) and unproven(engine) == []
    assert len(pending(engine)) == 1 and count(engine, "generated_unproven_rows") == 0


def test_t6_copy_does_not_retry_and_changed_salt_drops(engines, monkeypatch, tmp_path):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    root = tmp_path / "copied"
    root.mkdir()
    engine._store._conn.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copyfile(engine._store.db_path, root / "lcm.db")
    _state_db(root, [("S", None, None)])
    peer = _engine(root)
    try:
        assert peer._store.ingest_ledger_salt() == SALT
        peer._ingest_messages(copy.deepcopy(rows))
        assert emissions(peer) == [] and count(peer, "emission_retry_ok") == 0
        assert len(pending(engine)) == 1
    finally:
        peer.shutdown()
    with engine._store.atomic(immediate=True):
        engine._store._conn.execute("UPDATE ledger_meta SET value=? WHERE key='digest_salt'", (b"n" * 32,))
    engine._ingest_messages(rows)
    assert pending(engine) == [] and emissions(engine) == []
    assert count(engine, "emission_unproven_salt_changed") == 1


def test_t6_pending_key_uses_identity_fixed_at_open(engines, monkeypatch):
    engine = engines()
    identity = engine._store.db_identity
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "db_path", engine._store.db_path.parent / "changed-after-open.db")
        rows = fail_exit(engine, patch)
        assert ledger.has_pending(identity)
        engine._ingest_messages(rows)
    assert pending(engine) == [] and count(engine, "emission_retry_ok") == 1


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt, "busy_snapshot"])
def test_t7_owner_rollback_releases_claim_and_retries(engines, monkeypatch, error):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    transaction = engine._ingest_messages_transaction
    calls = []

    def abort(*args, **kwargs):
        assert len(emissions(engine)) == 1 and pending(engine)[0]["claim"] is not None
        assert count(engine, "emission_retry_ok") == 0
        calls.append(True)
        if error == "busy_snapshot" and len(calls) > 1:
            return transaction(*args, **kwargs)
        if error == "busy_snapshot":
            exc = sqlite3.OperationalError("database is locked")
            exc.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
            raise exc
        raise error("ingest after retry sentinel")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_ingest_messages_transaction", abort)
        if error == "busy_snapshot":
            engine._ingest_messages(copy.deepcopy(rows))
            assert len(calls) == 2
        else:
            with pytest.raises(error, match="ingest after retry sentinel"):
                engine._ingest_messages(copy.deepcopy(rows))
            assert emissions(engine) == [] and count(engine, "emission_retry_ok") == 0
            assert len(pending(engine)) == 1 and pending(engine)[0]["claim"] is None
    if error != "busy_snapshot":
        engine._ingest_messages(rows)
    assert len(emissions(engine)) == 1 and pending(engine) == []
    assert count(engine, "emission_retry_ok") == 1


@pytest.mark.parametrize("state", ["adopted", "refused"])
def test_t8_origin_handle_supports_existing_signals(engines, monkeypatch, state):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        _, rows = compress_path(engine, patch)
    assert engine._emission_handle is None and len(pending(engine)) == 1
    engine._ingest_messages(copy.deepcopy(rows))
    assert engine._emission_handle is not None and count(engine, "emission_retry_unhandled") == 0
    if state == "adopted":
        assert adopt(engine)
    else:
        engine.record_rejected_compaction()
    assert {r["state"] for r in emissions(engine)} == {state}
    assert {v["state"] for v in emission_views(engine)} == {state}


@pytest.mark.parametrize("other", [True, False])
def test_t8_other_engine_or_newer_compress_leaves_proposed(engines, monkeypatch, other):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    peer = _engine(Path(engine._store.db_path).parent) if other else engine
    if not other:
        engine._compress_calls += 1
    try:
        peer._ingest_messages(rows)
        assert count(peer, "emission_retry_unhandled") == 1
        assert {r["state"] for r in emissions(peer)} == {"proposed"}
        assert peer._emission_handle is None and engine._emission_handle is None
        assert pending(engine) == []
    finally:
        if other:
            peer.shutdown()


def test_t9_ledger_off_does_not_mark_retry_or_relate(engines, monkeypatch):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    emit(engine, rows)
    engine._ingest_messages(rows)
    assert len(pending(engine)) == 1 and count(engine, "emission_unproven_set") == 1
    assert count(engine, "emission_retry_ok") == count(engine, "generated_unproven_rows") == 0
    assert emissions(engine) == [] and unproven(engine) == []
    fresh = engines("off", mode="off", initialized=False)
    emit(fresh, rows)
    fresh._ingest_messages(rows)
    assert pending(fresh) == [] and count(fresh, "emission_unproven_set") == 0


def test_t10_two_threads_retry_once_even_before_first_callback(engines, monkeypatch):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    peer = _engine(Path(engine._store.db_path).parent)
    committed, peer_done = threading.Event(), threading.Event()
    finish_memory = engine._finish_ingest_memory
    overlap = []

    def delayed_callback():
        # COMMIT is complete, but the pending-removal callback has not run yet.
        committed.set()
        overlap.append(peer_done.wait(5))
        assert overlap[-1]
        finish_memory()

    def second():
        try:
            assert committed.wait(5)
            peer._ingest_messages(copy.deepcopy(rows))
            assert unproven(peer) == []
        finally:
            peer_done.set()

    monkeypatch.setattr(engine, "_finish_ingest_memory", delayed_callback)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Leave the descriptor's row for the peer to store while the mark is claimed.
            first, other = pool.submit(engine._ingest_messages, copy.deepcopy(rows[1:])), pool.submit(second)
            first.result(timeout=15)
            other.result(timeout=15)
        assert len(emissions(engine)) == len(emission_views(engine)) == 1
        assert count(engine, "emission_retry_ok") + count(peer, "emission_retry_ok") == 1
        assert pending(engine) == [] and overlap == [True]
    finally:
        peer.shutdown()


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_ingest_ledger_savepoint_restores_relation_schema(engines, monkeypatch, error):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    store, callbacks = engine._store, []
    store_id = store.append_batch("S", rows[:1])[0]
    attempt = dict(engine._prepare_ingest_ledger(rows, {"ledger_digest_ms": 0.}),
                   stored={0: store_id}, matched={}, remainders={}, observations=[])
    store._conn.execute("DROP TABLE IF EXISTS message_relations")
    store._identity_anchor_schema_ready = store._pending_relation_schema = False

    def fail_after_schema(groups):
        assert groups == [[(attempt["stored"][0], "generated_unproven", None, None, None)]]
        store._ensure_identity_anchor_schema()
        assert store._pending_relation_schema
        raise error("relation savepoint sentinel")

    monkeypatch.setattr(store, "add_message_relations", fail_after_schema)
    with store.ingest_atomic():
        if error is KeyboardInterrupt:
            with pytest.raises(error, match="relation savepoint sentinel"):
                engine._write_ingest_ledger(attempt, callbacks, {"ledger_ms": 0.})
        else:
            engine._write_ingest_ledger(attempt, callbacks, {"ledger_ms": 0.})
        assert not store._pending_relation_schema and callbacks == []
        assert not store._conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone()
        # A later writer in this same transaction must recreate the rolled-back schema.
        store._ensure_identity_anchor_schema()
        assert store._conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone()


@pytest.mark.parametrize("error", ["busy", "savepoint", "lost_transaction"])
def test_retry_fatal_errors_reach_owner_and_release_claims(engines, monkeypatch, error):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)

    def fatal(conn, *args):
        if error == "lost_transaction":
            conn.rollback()
            raise ValueError("lost transaction")
        if error == "savepoint":
            raise _SqliteSavepointError("retry savepoint sentinel")
        exc = sqlite3.OperationalError("database is locked")
        exc.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
        raise exc

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", fatal)
        with pytest.raises((sqlite3.Error, ValueError, _SqliteSavepointError)):
            engine._ingest_messages(rows)
    assert len(pending(engine)) == 1 and pending(engine)[0]["claim"] is None
    assert count(engine, "emission_retry_ok") == 0 and emissions(engine) == []


def test_retry_interrupt_rolls_back_partial_write_and_releases_claim(engines, monkeypatch):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)

    def interrupt(*args, **kwargs):
        assert len(emissions(engine)) == 1
        raise KeyboardInterrupt("retry view interrupt")

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "write_view", interrupt)
        with pytest.raises(KeyboardInterrupt, match="retry view interrupt"):
            engine._ingest_messages(rows)
    assert emissions(engine) == emission_views(engine) == []
    assert len(pending(engine)) == 1 and pending(engine)[0]["claim"] is None
    assert all(e.ref is None and not e.is_descriptor for e in pending(engine)[0]["entries"])
    engine._ingest_messages(rows)
    assert pending(engine) == [] and count(engine, "emission_retry_ok") == 1


@pytest.mark.parametrize("signals,expected", [(("refused",), "refused"), (("adopted",), "adopted"),
                                             (("adopted", "refused", "adopted"), "refused")])
def test_host_signal_before_retry_is_applied_and_refusal_sticks(engines, monkeypatch, signals, expected):
    engine = engines()
    insert, calls = ledger.insert_emissions, []

    def fail_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise sqlite3.OperationalError("exit write sentinel")
        return insert(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", fail_once)
        _, rows = compress_path(engine, patch)
        assert engine._emission_handle is None and len(pending(engine)) == 1
        for signal in signals:
            if signal == "refused":
                engine.record_rejected_compaction()
            else:
                assert adopt(engine)
        assert count(engine, "emission_signal_deferred") == len(signals)
        assert count(engine, "emission_signal_unmatched") == 0
        engine._ingest_messages(copy.deepcopy(rows))
    assert len(calls) == 2 and pending(engine) == []
    recovered = emissions(engine)
    assert len(recovered) == len(emission_views(engine)) == 1
    assert recovered[0]["state"] == emission_views(engine)[0]["state"] == expected
    assert recovered[0]["view_seq"] == emission_views(engine)[0]["view_seq"]
    if expected == "refused":
        assert engine._emission_handle is None
        assert adopt(engine)
        assert emissions(engine) == recovered and emission_views(engine)[0]["state"] == "refused"
    else:
        assert engine._emission_handle.view_seq == recovered[0]["view_seq"]


@pytest.mark.parametrize("refused", [False, True])
def test_older_retries_keep_descriptors_without_evicting_current_view(engines, monkeypatch, refused):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        for label in ("oldest", "older"):
            # Exercise distinct exit writes without an intervening ingest retry.
            engine._compress_calls += 1
            emit(engine, output(label))
            if refused and label == "oldest":
                engine.record_rejected_compaction()
    assert len(pending(engine)) == 2
    engine._compress_calls += 1
    rows = output("current")
    emit(engine, rows)
    current_handle, current_view = engine._emission_handle, emission_views(engine)[0]
    assert all(entry["superseded"] for entry in pending(engine))
    engine._ingest_messages(rows)
    recovered = [r for r in emissions(engine) if r["view_seq"] != current_handle.view_seq]
    assert len(recovered) == 2 and pending(engine) == []
    assert [r["descriptor_digest"] for r in recovered] == [
        ledger.identity_digest(SALT, output(label)[0]) for label in ("oldest", "older")]
    assert [r["state"] for r in recovered] == ["refused" if refused else "proposed", "proposed"]
    assert emission_views(engine) == [current_view]
    assert engine._emission_handle is current_handle
    assert count(engine, "emission_retry_superseded") == 2


def test_exit_write_that_rolls_back_supersedes_nothing(engines, monkeypatch):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        engine._compress_calls += 1
        emit(engine, output("older"))
    assert len(pending(engine)) == 1
    real_atomic, armed = engine._store.atomic, [True]

    @contextmanager
    def commit_fails(*, immediate=False):
        with real_atomic(immediate=immediate):
            yield
            if armed[0] and immediate:
                armed[0] = False
                raise sqlite3.OperationalError("disk I/O error")  # raised inside the owner, so it rolls back

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "atomic", commit_fails)
        engine._compress_calls += 1
        emit(engine, output("newer"))
    assert not armed[0] and emission_views(engine) == [] and len(pending(engine)) == 2
    assert not any(entry["superseded"] for entry in pending(engine))
    engine._ingest_messages(output("newer"))
    assert [ledger.decode_entries(v["entries"])[0].identity_digest for v in emission_views(engine)] == [
        ledger.identity_digest(SALT, output(label)[0]) for label in ("older", "newer")]
    assert count(engine, "emission_retry_superseded") == 0


def test_supersession_is_marked_before_the_exit_write_commits(engines, monkeypatch):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        engine._compress_calls += 1
        emit(engine, output("older"))
    real_note, seen = ledger.note_emission_written, []

    def note(*args):
        # Another store's retry reads the mark only inside its own write transaction, so it must precede COMMIT.
        seen.append(engine._store._conn.in_transaction)
        return real_note(*args)

    monkeypatch.setattr(ledger, "note_emission_written", note)
    engine._compress_calls += 1
    emit(engine, output("current"))
    assert seen == [True] and len(pending(engine)) == 1 and pending(engine)[0]["superseded"]


def test_pending_retries_without_newer_exit_keep_registry_order(engines, monkeypatch):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        for label in ("older", "newer"):
            engine._compress_calls += 1
            emit(engine, output(label))
    engine._ingest_messages(output("newer"))
    views = emission_views(engine)
    assert len(views) == 2 and views[0]["view_seq"] < views[1]["view_seq"]
    assert [ledger.decode_entries(v["entries"])[0].identity_digest for v in views] == [
        ledger.identity_digest(SALT, output(label)[0]) for label in ("older", "newer")]
    assert engine._emission_handle.view_seq == views[-1]["view_seq"]
    assert count(engine, "emission_retry_superseded") == 0


@pytest.mark.parametrize("initial,late", [(None, "refused"), (None, "adopted"), ("adopted", "refused")])
def test_signal_after_retry_read_is_counted_and_late_refusal_restores_no_handle(
        engines, monkeypatch, initial, late):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    if initial:
        engine._signal_emission_state(initial, "S", require_conversation=True)
    read, observed = ledger.pending_emission_signal, []

    def signal_after_read(entry):
        signal = read(entry)
        if not observed:
            observed.append(signal)
            engine._signal_emission_state(late, "S", require_conversation=True)
        return signal

    monkeypatch.setattr(ledger, "pending_emission_signal", signal_after_read)
    engine._ingest_messages(rows)
    assert observed == [initial] and pending(engine) == []
    assert count(engine, "emission_signal_late") == 1
    assert {r["state"] for r in emissions(engine)} == {initial or "proposed"}
    assert {v["state"] for v in emission_views(engine)} == {initial or "proposed"}
    assert (engine._emission_handle is None) == (late == "refused")


@pytest.mark.parametrize("mismatch", ["store", "origin", "compress_call", "session", "conversation"])
def test_pending_signal_requires_matching_owner_and_scope(engines, monkeypatch, mismatch):
    engine = engines()
    fail_exit(engine, monkeypatch)
    args = [engine._store.db_identity, engine, engine._compress_calls, "S", engine._hold_conversation_key()]
    index = ["store", "origin", "compress_call", "session", "conversation"].index(mismatch)
    args[index] = object()
    assert not ledger.record_pending_signal(*args, "refused", True)
    assert ledger.pending_emission_signal(pending(engine)[0]) is None
    if mismatch == "conversation":
        assert ledger.record_pending_signal(*args, "adopted", False)
        assert ledger.pending_emission_signal(pending(engine)[0]) == "adopted"


def test_pending_signal_selects_newest_matching_entry_across_registry_keys(engines):
    engine = engines()
    key = (engine._store.db_identity, SALT)
    for created, lineage in enumerate(("first", "second"), 1):
        ledger.mark_pending_emission(key, lineage, origin=engine, compress_call=engine._compress_calls,
                                     session_id="S", conversation_key=engine._hold_conversation_key())
        pending(engine)[-1]["created_at"] = created
    assert ledger.record_pending_signal(key[0], engine, engine._compress_calls, "S",
                                        engine._hold_conversation_key(), "refused", True)
    assert [ledger.pending_emission_signal(e) for e in pending(engine)] == [None, "refused"]
