"""P1c-2a: failed exit writes recover atomically and remain visible until then."""
import asyncio
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
from tests.test_issue_2055_emission_writes import adopt, compress_path, emission_views, emissions, prepare_path
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


def candidate(engine, lineage=None):
    conn = engine._store._conn
    if lineage is None:
        persisted = conn.execute("SELECT value FROM ledger_meta WHERE key='session_lineage:S'").fetchone()
        lineage = persisted[0] if persisted else ledger.lineage_key(engine._host_uid_lineage_key()[0], "S")
    return ledger.emission_candidate(conn, lineage)


def assert_next_live_candidate(engine, monkeypatch):
    compress_path(engine, monkeypatch)
    view = emission_views(engine)[-1]
    assert candidate(engine, view["lineage"]) == (view, "usable")
    assert engine._emission_handle.view_seq == view["view_seq"]
    return view


def assert_recovered(engine, number=1):
    assert count(engine, "emission_retry_ok") == count(engine, "emission_recovered_provenance_only") == number
    assert all("signal" not in entry and "superseded" not in entry for entry in pending(engine))



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
    assert emission_views(engine) == [] and len(emission_views(control)) == 1
    assert engine._emission_handle is None and candidate(engine) == (None, "none")
    assert_recovered(engine)
    assert engine._store._conn.execute("SELECT value FROM ledger_meta WHERE key='emission_provenance_floor'").fetchone() == (0,)
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
    if site == "write_view":
        assert pending(engine) == [] and emission_views(engine) == []
        assert len(emissions(engine)) == 1 and unproven(engine) == []
        assert_recovered(engine)
        return
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
def test_t8_origin_recovery_never_receives_handle_or_signal(engines, monkeypatch, state):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        _, rows = compress_path(engine, patch)
    assert engine._emission_handle is None and len(pending(engine)) == 1
    engine._ingest_messages(copy.deepcopy(rows))
    assert engine._emission_handle is None
    assert_recovered(engine)
    if state == "adopted":
        assert adopt(engine)
    else:
        engine.record_rejected_compaction()
    assert count(engine, "emission_signal_unmatched") == 1
    assert {r["state"] for r in emissions(engine)} == {"proposed"}
    assert emission_views(engine) == [] and candidate(engine) == (None, "none")
    assert_next_live_candidate(engine, monkeypatch)


@pytest.mark.parametrize("other", [True, False])
def test_t8_other_engine_or_newer_compress_keeps_provenance_only(engines, monkeypatch, other):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    peer = _engine(Path(engine._store.db_path).parent) if other else engine
    if not other:
        engine._compress_calls += 1
    try:
        peer._ingest_messages(rows)
        assert_recovered(peer)
        assert {r["state"] for r in emissions(peer)} == {"proposed"}
        assert emission_views(peer) == [] and candidate(peer) == (None, "none")
        assert peer._emission_handle is None and engine._emission_handle is None
        assert pending(engine) == []
        assert_next_live_candidate(peer, monkeypatch)
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
        assert len(emissions(engine)) == 1 and emission_views(engine) == []
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

    insert = ledger.insert_emissions

    def interrupt(*args, **kwargs):
        insert(*args, **kwargs)
        assert len(emissions(engine)) == 1
        raise KeyboardInterrupt("retry descriptor interrupt")

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", interrupt)
        with pytest.raises(KeyboardInterrupt, match="retry descriptor interrupt"):
            engine._ingest_messages(rows)
    assert emissions(engine) == emission_views(engine) == []
    assert len(pending(engine)) == 1 and pending(engine)[0]["claim"] is None
    assert all(e.ref is None and not e.is_descriptor for e in pending(engine)[0]["entries"])
    engine._ingest_messages(rows)
    assert pending(engine) == [] and count(engine, "emission_retry_ok") == 1



@pytest.mark.parametrize("signals", [("refused",), ("adopted",), ("adopted", "refused", "adopted")])
def test_host_signal_before_retry_is_unmatched_and_cannot_affect_provenance(engines, monkeypatch, signals):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        _, rows = compress_path(engine, patch)
    for signal in signals:
        if signal == "refused":
            engine.record_rejected_compaction()
        else:
            assert adopt(engine)
    assert count(engine, "emission_signal_unmatched") == len(signals)
    assert "signal" not in pending(engine)[0]
    engine._ingest_messages(copy.deepcopy(rows))
    assert pending(engine) == [] and engine._emission_handle is None
    assert len(emissions(engine)) == 1 and emissions(engine)[0]["state"] == "proposed"
    assert emission_views(engine) == [] and candidate(engine) == (None, "none")
    assert_recovered(engine)
    assert_next_live_candidate(engine, monkeypatch)


@pytest.mark.parametrize("refused", [False, True])
def test_older_retries_keep_descriptors_without_evicting_current_view(engines, monkeypatch, refused):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        for label in ("oldest", "older"):
            engine._compress_calls += 1
            emit(engine, output(label))
            if refused and label == "oldest":
                engine.record_rejected_compaction()
    assert len(pending(engine)) == 2
    engine._compress_calls += 1
    rows = output("current")
    emit(engine, rows)
    current_handle, current_view = engine._emission_handle, emission_views(engine)[0]
    assert all("superseded" not in entry for entry in pending(engine))
    engine._ingest_messages(rows)
    recovered = [r for r in emissions(engine) if r["view_seq"] != current_handle.view_seq]
    assert len(recovered) == 2 and pending(engine) == []
    assert [r["descriptor_digest"] for r in recovered] == [
        ledger.identity_digest(SALT, output(label)[0]) for label in ("oldest", "older")]
    assert [r["state"] for r in recovered] == ["proposed", "proposed"]
    assert emission_views(engine) == [current_view] and engine._emission_handle is current_handle
    assert candidate(engine, current_view["lineage"]) == (None, "withheld_recovered")
    assert_recovered(engine, 2)
    assert_next_live_candidate(engine, monkeypatch)


def test_exit_write_that_rolls_back_leaves_both_payloads_for_provenance(engines, monkeypatch):
    engine = engines()
    fail_exit(engine, monkeypatch, output("older"))
    real_atomic, armed = engine._store.atomic, [True]

    @contextmanager
    def commit_fails(*, immediate=False):
        with real_atomic(immediate=immediate):
            yield
            if armed[0] and immediate:
                armed[0] = False
                raise sqlite3.OperationalError("disk I/O error")

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "atomic", commit_fails)
        engine._compress_calls += 1
        emit(engine, output("newer"))
    assert not armed[0] and emission_views(engine) == [] and len(pending(engine)) == 2
    assert not any("superseded" in entry for entry in pending(engine))
    engine._ingest_messages(output("newer"))
    assert [r["descriptor_digest"] for r in emissions(engine)] == [
        ledger.identity_digest(SALT, output(label)[0]) for label in ("older", "newer")]
    assert emission_views(engine) == [] and candidate(engine) == (None, "none")
    assert_recovered(engine, 2)
    assert_next_live_candidate(engine, monkeypatch)


def test_live_exit_commit_never_changes_pending_payload(engines, monkeypatch):
    engine = engines()
    fail_exit(engine, monkeypatch, output("older"))
    entry = pending(engine)[0]
    before = dict(entry)
    write, seen = ledger.write_view, []

    def observe(*args, **kwargs):
        result = write(*args, **kwargs)
        seen.append((engine._store._conn.in_transaction, entry == before))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "write_view", observe)
        engine._compress_calls += 1
        emit(engine, output("current"))
    assert seen == [(True, True)] and entry == before
    engine._ingest_messages(output("current"))
    assert candidate(engine) == (None, "withheld_recovered")
    assert_next_live_candidate(engine, monkeypatch)


def test_pending_retries_without_newer_exit_keep_descriptor_registry_order(engines, monkeypatch):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        for label in ("older", "newer"):
            engine._compress_calls += 1
            emit(engine, output(label))
    engine._ingest_messages(output("newer"))
    recovered = emissions(engine)
    assert len(recovered) == 2 and recovered[0]["view_seq"] < recovered[1]["view_seq"]
    assert [r["descriptor_digest"] for r in recovered] == [
        ledger.identity_digest(SALT, output(label)[0]) for label in ("older", "newer")]
    assert emission_views(engine) == [] and engine._emission_handle is None
    assert candidate(engine) == (None, "none")
    assert_recovered(engine, 2)
    assert_next_live_candidate(engine, monkeypatch)


@pytest.mark.parametrize("initial,late", [(None, "refused"), (None, "adopted"), ("adopted", "refused")])
def test_signal_during_retry_is_unmatched_and_recovery_restores_no_handle(engines, monkeypatch, initial, late):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    if initial:
        engine._signal_emission_state(initial, "S", require_conversation=True)
    insert, observed = ledger.insert_emissions, []

    def signal_after_insert(*args, **kwargs):
        result = insert(*args, **kwargs)
        observed.append(True)
        engine._signal_emission_state(late, "S", require_conversation=True)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", signal_after_insert)
        engine._ingest_messages(rows)
    assert observed == [True] and pending(engine) == []
    assert count(engine, "emission_signal_unmatched") == 1 + bool(initial)
    assert {r["state"] for r in emissions(engine)} == {"proposed"}
    assert emission_views(engine) == [] and engine._emission_handle is None
    assert candidate(engine) == (None, "none")
    assert_next_live_candidate(engine, monkeypatch)


@pytest.mark.parametrize("mismatch", ["store", "origin", "compress_call", "session", "conversation"])
def test_pending_signal_is_unmatched_regardless_of_owner_and_scope(engines, monkeypatch, mismatch):
    engine = engines()
    rows = fail_exit(engine, monkeypatch)
    target = engines("other") if mismatch == "store" else copy.copy(engine) if mismatch == "origin" else engine
    with monkeypatch.context() as patch:
        if mismatch == "compress_call":
            patch.setattr(target, "_compress_calls", engine._compress_calls + 1)
        elif mismatch == "conversation":
            patch.setattr(target, "_hold_conversation_key", lambda: "other")
        target._signal_emission_state("refused", "other" if mismatch == "session" else "S", require_conversation=True)
    assert count(target, "emission_signal_unmatched") == 1
    assert "signal" not in pending(engine)[0]
    engine._ingest_messages(rows)
    assert candidate(engine) == (None, "none") and emission_views(engine) == []
    assert_next_live_candidate(engine, monkeypatch)


def test_pending_signal_never_selects_an_entry_across_registry_keys(engines, monkeypatch):
    engine = engines()
    fail_exit(engine, monkeypatch)
    first = pending(engine)[0]
    key = (engine._store.db_identity, SALT)
    ledger.mark_pending_emission(key, "second", origin=engine, compress_call=engine._compress_calls,
                                 session_id="S", conversation_key=engine._hold_conversation_key(),
                                 rows=first["rows"], indices=first["indices"], entries=first["entries"])
    engine.record_rejected_compaction()
    assert count(engine, "emission_signal_unmatched") == 1
    assert all("signal" not in entry for entry in pending(engine))
    engine._ingest_messages(output())
    assert len(emissions(engine)) == 2 and emission_views(engine) == []
    assert candidate(engine) == (None, "none")
    assert_next_live_candidate(engine, monkeypatch)


def failed_compress(engine, monkeypatch, path="normal"):
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        _, rows = compress_path(engine, patch, path)
    assert pending(engine) and engine._emission_handle is None
    return rows


def seeded_pending(engines, monkeypatch):
    engine = engines()
    compress_path(engine, monkeypatch)
    live = emission_views(engine)[-1]
    assert candidate(engine) == (live, "usable")
    rows = failed_compress(engine, monkeypatch)
    return engine, rows, live


def test_window_1_refusal_after_retry_insert_never_becomes_a_candidate(engines, monkeypatch):
    engine, rows, live = seeded_pending(engines, monkeypatch)
    insert = ledger.insert_emissions

    def refuse_after_insert(*args, **kwargs):
        ids = insert(*args, **kwargs)
        engine.record_rejected_compaction()
        return ids

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", refuse_after_insert)
        engine._ingest_messages(rows)
    assert emission_views(engine) == [live] and engine._emission_handle is None
    assert count(engine, "emission_signal_unmatched") == 1
    assert candidate(engine) == (None, "withheld_recovered")
    assert_next_live_candidate(engine, monkeypatch)


def test_window_2_refusal_after_settle_before_publish_has_no_recovered_handle(engines, monkeypatch):
    engine, rows, live = seeded_pending(engines, monkeypatch)
    finish, seen = engine._finish_ingest_memory, []

    def refuse_before_publish():
        assert not pending(engine) and not engine._store._conn.in_transaction
        assert engine._emission_handle is None
        engine.record_rejected_compaction()
        seen.append(candidate(engine))
        finish()

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_finish_ingest_memory", refuse_before_publish)
        engine._ingest_messages(rows)
    assert seen == [(None, "withheld_recovered")] and emission_views(engine) == [live]
    assert engine._emission_handle is None and count(engine, "emission_signal_unmatched") == 1
    assert_next_live_candidate(engine, monkeypatch)


def test_window_3_other_engine_recovery_cannot_take_originators_refusal(engines, monkeypatch):
    engine, rows, live = seeded_pending(engines, monkeypatch)
    peer = _engine(Path(engine._store.db_path).parent)
    try:
        peer._ingest_messages(rows)
        engine.record_rejected_compaction()
        assert engine._emission_handle is peer._emission_handle is None
        assert count(engine, "emission_signal_unmatched") == 1
        assert emission_views(peer) == [live]
        assert candidate(engine) == candidate(peer) == (None, "withheld_recovered")
        assert_next_live_candidate(peer, monkeypatch)
    finally:
        peer.shutdown()


def test_window_4_retry_marks_persisted_lineage_after_provisional_key_drift(engines, monkeypatch):
    engine, _rows, live = seeded_pending(engines, monkeypatch)
    # Clear the first pending case so this scenario isolates the real provisional exit.
    ledger._PENDING_EMISSIONS.clear()
    rows = failed_compress(engine, monkeypatch, "provisional")
    entry = pending(engine)[0]
    assert entry["lineage"] == "provisional:S"
    engine.record_rejected_compaction()
    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", failing)
        engine._ingest_messages(rows)  # Resolves/persists S without consuming the pending payload.
    lineage = engine._store._conn.execute("SELECT value FROM ledger_meta WHERE key='session_lineage:S'").fetchone()[0]
    assert lineage == live["lineage"] and entry["lineage"] != lineage
    engine._ingest_messages(rows)
    recovered = emissions(engine)[-1]
    assert recovered["lineage"] == lineage and emission_views(engine) == [live]
    assert engine._store._conn.execute("SELECT value FROM ledger_meta WHERE key=?",
                                      ("emission_recovered:" + lineage,)).fetchone() == (str(recovered["view_seq"]),)
    assert candidate(engine, lineage) == (None, "withheld_recovered")
    assert_next_live_candidate(engine, monkeypatch)


def test_window_5_cross_connection_candidate_uses_durable_mark_without_registry(engines, monkeypatch):
    engine, rows, live = seeded_pending(engines, monkeypatch)
    peer = _engine(Path(engine._store.db_path).parent)
    try:
        assert peer._store._conn is not engine._store._conn
        peer._ingest_messages(rows)
        ledger._PENDING_EMISSIONS.clear()  # A new process has no registry knowledge of the retry.
        engine.record_rejected_compaction()
        assert emission_views(engine) == [live]
        assert candidate(engine) == candidate(peer) == (None, "withheld_recovered")
        assert_next_live_candidate(engine, monkeypatch)
        assert candidate(peer) == candidate(engine)
    finally:
        peer.shutdown()


def test_window_6_partial_retry_failure_cannot_reorder_live_views(engines, monkeypatch):
    engine, rows, live = seeded_pending(engines, monkeypatch)
    failed_compress(engine, monkeypatch)
    assert len(pending(engine)) == 2
    engine.record_rejected_compaction()
    insert, calls = ledger.insert_emissions, []

    def fail_older_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            failing()
        return insert(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "insert_emissions", fail_older_once)
        engine._ingest_messages(rows)
    assert len(pending(engine)) == 1 and len(emissions(engine)) == 2
    first_mark = engine._store._conn.execute("SELECT value FROM ledger_meta WHERE key=?",
                                            ("emission_recovered:" + live["lineage"],)).fetchone()[0]
    assert candidate(engine) == (None, "withheld_recovered")
    engine._ingest_messages(rows)
    final_mark = engine._store._conn.execute("SELECT value FROM ledger_meta WHERE key=?",
                                            ("emission_recovered:" + live["lineage"],)).fetchone()[0]
    assert int(final_mark) > int(first_mark) and not pending(engine)
    assert emission_views(engine) == [live] and candidate(engine) == (None, "withheld_recovered")
    assert_recovered(engine, 2)
    assert_next_live_candidate(engine, monkeypatch)


@pytest.mark.parametrize("site", ["before_seq", "before_commit", "readback_failure", "owner_cleanup",
                                 "store_publication", "timeout_restore", "after_with"])
@pytest.mark.parametrize("error", [KeyboardInterrupt, pytest.param(asyncio.CancelledError, id="cancelled")])
def test_window_7_exit_interrupt_registers_only_without_successful_commit(engines, monkeypatch, site, error):
    engine = engines()
    before = prepare_path(engine, monkeypatch)
    interrupt = error("window 7 sentinel")
    real_owner = engine._store._transaction_owner
    active_owners = real_owner._active_owners
    real_scope = engine._emission_owner
    real_count = engine._ledger_count
    publication = engine._store._publish_ingest_schemas
    armed, denied_readbacks, committed_rows = [], [], []

    def abort(*args, **kwargs):
        raise interrupt

    def deny_readback(action, table, *_):
        if action == sqlite3.SQLITE_READ and table == "ingest_views":
            denied_readbacks.append(True)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def abort_publication():
        if armed:
            raise interrupt
        return publication()

    @contextmanager
    def timeout_interrupt(*args, **kwargs):
        from hermes_lcm.sqlite_util import _temporary_sqlite_busy_timeout
        with _temporary_sqlite_busy_timeout(*args, **kwargs):
            yield
        raise interrupt

    class InterruptCleanup:
        def get(self):
            return active_owners.get()

        def set(self, value):
            return active_owners.set(value)

        def reset(self, token):
            active_owners.reset(token)
            if armed:
                raise interrupt

    @contextmanager
    def after_owner(*args, **kwargs):
        armed.append(True)
        try:
            with real_scope(*args, **kwargs) as value:
                yield value
        finally:
            armed.clear()

    def after_with(key, *args, **kwargs):
        if key == "ledger.emissions_written":
            if site == "readback_failure":
                assert not engine._store._conn.in_transaction
                committed_rows.extend(emission_views(engine))
                assert len(committed_rows) == 1  # The exact exit row committed before readback fails.
                engine._store._conn.set_authorizer(deny_readback)
            raise interrupt
        return real_count(key, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_emission_owner", after_owner)
        if site == "before_seq":
            patch.setattr(ledger, "next_view_seq", abort)
        elif site == "before_commit":
            patch.setattr(ledger, "write_view", abort)
        elif site == "owner_cleanup":
            patch.setattr(real_owner, "_active_owners", InterruptCleanup())
        elif site == "store_publication":
            patch.setattr(engine._store, "_publish_ingest_schemas", abort_publication)
        elif site == "timeout_restore":
            import hermes_lcm.engine as engine_module
            patch.setattr(engine_module, "_temporary_sqlite_busy_timeout", timeout_interrupt)
        else:
            patch.setattr(engine, "_emission_owner", after_owner)
            patch.setattr(engine, "_ledger_count", after_with)
        try:
            with pytest.raises(error) as caught:
                engine.compress(before, current_tokens=100, force=True)
        finally:
            engine._store._conn.set_authorizer(None)
        assert caught.value is interrupt
    engine.record_rejected_compaction()
    assert count(engine, "emission_signal_unmatched") == 1
    if site in ("before_seq", "before_commit"):
        assert len(pending(engine)) == 1 and emissions(engine) == emission_views(engine) == []
        engine._ingest_messages(output())
        assert_recovered(engine)
        assert candidate(engine) == (None, "none")
    elif site == "readback_failure":
        assert denied_readbacks and emission_views(engine) == committed_rows
        assert len(pending(engine)) == 1
        engine._ingest_messages(output())
        assert_recovered(engine)
        assert candidate(engine) == (None, "withheld_recovered")
    else:
        assert not pending(engine) and len(emissions(engine)) == len(emission_views(engine)) == 1
        assert candidate(engine) == (emission_views(engine)[0], "usable")
    assert not engine._store._conn.in_transaction
    assert_next_live_candidate(engine, monkeypatch)


@pytest.mark.parametrize("peer_shape", ["different_digest", "different_index", "extra", "missing", "exact"])
def test_window_7_reused_sequence_checks_exact_write(engines, monkeypatch, peer_shape):
    engine = engines()
    before = prepare_path(engine, monkeypatch)
    peer = _engine(Path(engine._store.db_path).parent)
    interrupt = KeyboardInterrupt("window 7 reused sequence sentinel")
    scope, insert = engine._emission_owner, ledger.insert_emissions
    attempted, peer_rows = {}, []

    def interrupt_after_insert(conn, lineage, seq, rows):
        insert(conn, lineage, seq, rows)
        attempted.update(lineage=lineage, seq=seq, rows=copy.deepcopy(rows))
        raise interrupt

    @contextmanager
    def rollback_then_peer_commit(*args, **kwargs):
        try:
            with scope(*args, **kwargs) as value:
                yield value
        except KeyboardInterrupt:
            assert not engine._store._conn.in_transaction
            assert emissions(engine) == emission_views(engine) == []
            peer_rows.extend(copy.deepcopy(attempted["rows"]))
            if peer_shape == "different_digest":
                peer_rows[0]["descriptor_digest"] = "peer digest"
            elif peer_shape == "different_index":
                peer_rows[0]["output_index"] += 1
            elif peer_shape == "extra":
                peer_rows.append(dict(peer_rows[0], output_index=peer_rows[-1]["output_index"] + 1))
            elif peer_shape == "missing":
                peer_rows.clear()
            with peer._emission_owner(SALT) as (conn, reason):
                assert reason is None
                seq = ledger.next_view_seq(conn)
                assert seq == attempted["seq"]
                insert(conn, attempted["lineage"], seq, peer_rows)
                ledger.write_view(conn, SALT, attempted["lineage"], "emission", "proposed", [], view_seq=seq)
            raise

    try:
        assert peer._store._conn is not engine._store._conn
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_emission_owner", rollback_then_peer_commit)
            patch.setattr(ledger, "insert_emissions", interrupt_after_insert)
            with pytest.raises(KeyboardInterrupt) as caught:
                engine.compress(before, current_tokens=100, force=True)
        assert caught.value is interrupt
        view = emission_views(engine)[0]
        assert (view["lineage"], view["view_seq"]) == (attempted["lineage"], attempted["seq"])
        assert len(emissions(engine)) == len(peer_rows)
        if peer_shape == "exact":
            assert not pending(engine)
        else:
            assert len(pending(engine)) == 1
            assert pending(engine)[0]["rows"] == attempted["rows"]
    finally:
        peer.shutdown()


@pytest.mark.parametrize("error", [KeyboardInterrupt, pytest.param(asyncio.CancelledError, id="cancelled")])
def test_window_7_open_transaction_skips_readback_and_registers(engines, monkeypatch, error):
    engine = engines()
    before = prepare_path(engine, monkeypatch)
    conn, trace = engine._store._conn, []
    interrupt = error("window 7 open transaction sentinel")
    real_scope = engine._emission_owner

    @contextmanager
    def unfinished_owner(*args, **kwargs):
        with real_scope(*args, **kwargs) as value:
            yield value
        conn.execute("BEGIN")  # Teardown leaves a transaction open with the exact row visible.
        raise interrupt

    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_emission_owner", unfinished_owner)
            conn.set_trace_callback(trace.append)
            try:
                with pytest.raises(error) as caught:
                    engine.compress(before, current_tokens=100, force=True)
            finally:
                conn.set_trace_callback(None)
            assert caught.value is interrupt
        assert conn.in_transaction and len(emission_views(engine)) == 1
        assert len(pending(engine)) == 1
        assert not any(sql.startswith("SELECT 1 FROM ingest_views WHERE lineage=") for sql in trace)
    finally:
        conn.rollback()
    assert len(emission_views(engine)) == 1  # The row committed, but readback was forbidden.
    assert_next_live_candidate(engine, monkeypatch)


def test_window_8_rollback_cannot_clear_another_stores_committed_recovery(engines, monkeypatch):
    engine, rows, live = seeded_pending(engines, monkeypatch)
    peer = _engine(Path(engine._store.db_path).parent)
    scope = engine._emission_owner

    @contextmanager
    def rollback_then_peer_commit(*args, **kwargs):
        try:
            with scope(*args, **kwargs) as value:
                yield value
                raise ValueError("rollback before peer commit")
        except ValueError:
            peer._ingest_messages(rows)
            assert candidate(peer) == (None, "withheld_recovered")
            raise

    try:
        # Prepare before injecting so retry cannot consume the older payload first.
        with monkeypatch.context() as patch:
            patch.setattr(ledger, "insert_emissions", failing)
            before = prepare_path(engine, patch)
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_emission_owner", rollback_then_peer_commit)
            engine.compress(before, current_tokens=100, force=True)
        assert candidate(engine) == (None, "withheld_recovered")
        assert emission_views(engine) == [live] and len(pending(engine)) == 1
        engine.record_rejected_compaction()
        assert all("superseded" not in entry for entry in pending(engine))
        engine._ingest_messages(rows)
        assert candidate(peer) == (None, "withheld_recovered")
        assert_next_live_candidate(peer, monkeypatch)
    finally:
        peer.shutdown()


def test_window_9_retry_publication_cannot_replace_a_new_live_handle(engines, monkeypatch):
    engine, rows, live = seeded_pending(engines, monkeypatch)
    finish, observed = engine._finish_ingest_memory, []

    def live_before_retry_publication():
        finish()
        assert candidate(engine) == (None, "withheld_recovered")
        assert engine._emission_handle is None and not pending(engine)
        engine.record_rejected_compaction()
        emit(engine, output("live between commit and publish"))
        observed.append(engine._emission_handle)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_finish_ingest_memory", live_before_retry_publication)
        engine._ingest_messages(rows)
    assert observed == [engine._emission_handle]
    assert engine._emission_handle.view_seq > live["view_seq"]
    assert candidate(engine) == (emission_views(engine)[-1], "usable")
    assert_recovered(engine)
    assert_next_live_candidate(engine, monkeypatch)


def test_window_10_no_deferred_signal_survives_real_session_reset(engines, monkeypatch):
    engine, rows, live = seeded_pending(engines, monkeypatch)
    engine.record_rejected_compaction()
    assert count(engine, "emission_signal_unmatched") == 1
    engine.on_session_reset()
    assert all("signal" not in entry for entry in pending(engine))
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    engine._ingest_messages(rows)
    assert engine._emission_handle is None
    assert emission_views(engine) == [live] and candidate(engine) == (None, "withheld_recovered")
    assert_next_live_candidate(engine, monkeypatch)


def test_candidate_snapshot_is_one_statement_and_sees_uncommitted_retry(engines, monkeypatch):
    engine, _rows, live = seeded_pending(engines, monkeypatch)
    conn, token, callbacks, attempt = engine._store._conn, object(), [], {}
    trace = []
    conn.set_trace_callback(trace.append)
    try:
        assert candidate(engine) == (live, "usable")
    finally:
        conn.set_trace_callback(None)
    # candidate() resolves the session lineage; the helper itself uses exactly one statement.
    assert len(trace) == 2 and trace[-1].startswith("SELECT v.*") and not conn.in_transaction
    try:
        with pytest.raises(ValueError, match="rollback candidate snapshot"):
            with engine._store.atomic(immediate=True):
                engine._retry_pending_emissions(attempt, callbacks, token)
                assert conn.in_transaction and candidate(engine) == (None, "withheld_recovered")
                raise ValueError("rollback candidate snapshot")
    finally:
        ledger.settle_pending_emissions(token)
    assert candidate(engine) == (live, "usable") and len(pending(engine)) == 1
    assert len(emissions(engine)) == 1 and count(engine, "emission_retry_ok") == 0


def test_candidate_uses_callers_read_snapshot_across_another_connection_commit(engines, monkeypatch):
    engine, rows, live = seeded_pending(engines, monkeypatch)
    peer, conn = _engine(Path(engine._store.db_path).parent), engine._store._conn
    try:
        conn.execute("BEGIN")
        assert candidate(engine) == (live, "usable")
        peer._ingest_messages(rows)
        assert candidate(peer) == (None, "withheld_recovered")
        assert candidate(engine) == (live, "usable")  # The existing read snapshot stays consistent.
        conn.rollback()
        assert candidate(engine) == (None, "withheld_recovered")
    finally:
        conn.rollback()
        peer.shutdown()


def test_historical_recovered_view_is_below_floor_until_next_live_exit(engines, monkeypatch):
    engine, rows = engines(), output()
    emit(engine, rows)  # Pre-change retries wrote exactly this ordinary proposed view.
    historical = emission_views(engine)[0]
    conn = engine._store._conn
    # Simulate an upgrade from a store whose recovered views predate the floor.
    conn.execute("DELETE FROM ledger_meta WHERE key='emission_provenance_floor'")
    conn.commit()
    assert engine._store.ingest_ledger_state() == "ready"
    counter = conn.execute("SELECT value FROM ledger_meta WHERE key='view_seq'").fetchone()
    assert candidate(engine, historical["lineage"]) == (None, "below_floor")
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    assert conn.execute("SELECT value FROM ledger_meta WHERE key='emission_provenance_floor'").fetchone() == counter
    assert not conn.in_transaction
    engine._ingest_messages(rows)
    assert candidate(engine) == (None, "below_floor")
    assert_next_live_candidate(engine, monkeypatch)
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    assert conn.execute("SELECT value FROM ledger_meta WHERE key='emission_provenance_floor'").fetchone() == counter
    assert candidate(engine)[1] == "usable"


def test_fresh_store_floor_zero_and_refused_newest_never_falls_back(engines, monkeypatch):
    engine = engines()
    compress_path(engine, monkeypatch)
    conn = engine._store._conn
    assert conn.execute("SELECT value FROM ledger_meta WHERE key='emission_provenance_floor'").fetchone() == (0,)
    first = emission_views(engine)[0]
    assert adopt(engine)
    assert candidate(engine)[0]["state"] == "adopted"
    compress_path(engine, monkeypatch)
    engine.record_rejected_compaction()
    assert len(emission_views(engine)) == 2 and emission_views(engine)[0]["view_seq"] == first["view_seq"]
    assert candidate(engine) == (None, "refused")
    assert_next_live_candidate(engine, monkeypatch)


def test_retry_high_water_mark_is_numeric_max_and_savepoint_atomic(engines, monkeypatch):
    engine, rows, live = seeded_pending(engines, monkeypatch)
    conn, key = engine._store._conn, "emission_recovered:" + live["lineage"]
    with engine._store.atomic():
        conn.execute("INSERT INTO ledger_meta(key,value) VALUES (?,?)", (key, "9"))
        conn.execute("UPDATE ledger_meta SET value='10' WHERE key='view_seq'")
    engine._ingest_messages(rows)
    assert conn.execute("SELECT value FROM ledger_meta WHERE key=?", (key,)).fetchone() == ("11",)
    failed_compress(engine, monkeypatch)
    with engine._store.atomic():
        conn.execute("UPDATE ledger_meta SET value='99' WHERE key=?", (key,))
    engine._ingest_messages(rows)
    assert conn.execute("SELECT value FROM ledger_meta WHERE key=?", (key,)).fetchone() == ("99",)
    assert candidate(engine) == (None, "withheld_recovered")
    with engine._store.atomic():
        conn.execute("UPDATE ledger_meta SET value='99' WHERE key='view_seq'")
    assert_next_live_candidate(engine, monkeypatch)


def fault_product_bytes(engines, monkeypatch, cell):
    """The same t1/t2 fault script used to record the pinned 97383e316 base.

    Compare every messages/relation column and every binding column except
    view_seq: a conservatively registered interrupt can consume another sequence.
    SQLite and Python clocks, lineage and token counting are deterministic.
    """
    import json
    from tests.behaviour.reconcile_trace_harness import install_clock

    install_clock(monkeypatch)
    connect = sqlite3.connect

    def frozen_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.create_function("strftime", -1, lambda *args: "1700001000")
        return conn

    monkeypatch.setattr(sqlite3, "connect", frozen_connect)
    engine = engines()
    monkeypatch.setattr(engine, "_host_uid_lineage_key", lambda *a, **kw: ("fixture-lineage", None))
    rows = fail_exit(engine, monkeypatch)
    with monkeypatch.context() as patch:
        if cell == "t2":
            patch.setattr(ledger, "insert_emissions", failing)
        returned = engine._ingest_messages(copy.deepcopy(rows))
    result = dict(returned=returned)
    for table in ("messages", "message_relations", "occurrence_bindings"):
        columns = engine._store._conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        if not columns:
            result[table] = []
            continue
        names = [row[1] for row in columns if table != "occurrence_bindings" or row[1] != "view_seq"]
        selected = ",".join('"' + name + '"' for name in names)
        result[table] = dict(columns=names, rows=engine._store._conn.execute(
            f'SELECT {selected} FROM "{table}" ORDER BY rowid').fetchall())
    return json.dumps(result, sort_keys=True, separators=(",", ":"),
                      default=lambda value: {"hex": value.hex()}).encode()


@pytest.mark.parametrize("cell,base_sha256", [
    ("t1", "9812b1710d81d24b33fe79ce35a9cf691169c177b31e847bf13592b7f4950bab"),
    ("t2", "e54ba9dffcbe49972ee5241f1b0d318a0a873e7fab31ed5829d6be847f374b06"),
])
def test_t1_t2_fault_product_bytes_match_pinned_base(engines, monkeypatch, cell, base_sha256):
    import hashlib

    # Captured from 97383e31631571274143c7d2e3f795542c87fda3 with this exact fault script.
    # Includes complete messages, generated_unproven relations and all binding
    # columns except view_seq, plus the returned host rows. No golden is regenerated.
    assert hashlib.sha256(fault_product_bytes(engines, monkeypatch, cell)).hexdigest() == base_sha256
