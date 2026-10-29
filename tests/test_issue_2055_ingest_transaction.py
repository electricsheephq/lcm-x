"""P1b-1: fault production evidence helpers without replacing row decisions."""

import copy
import gc
import json
import sqlite3
import threading
from contextlib import contextmanager, nullcontext

import pytest

from tests.test_host_uid_shadow import _engine, _m, _state_db


@pytest.fixture
def ingest_case(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    # A real two-row prefix permits replay proof; direct append leaves reconciliation off.
    prefix = [_m("user", "seed", uid="seed-uid"), _m("assistant", "seed reply", 11., "reply-uid")]
    engine.ingest(prefix)
    assert engine._ingest_failure_count == 0
    seed = engine._store.get_session_messages("S")[0]["store_id"]
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    host = [{**prefix[0], "timestamp": 10.}, prefix[1], _m("user", "fresh", 20., "fresh-uid")]
    yield engine, seed, host
    engine.shutdown()


def durable(engine, conn=None):
    """Exact SQL values, including timestamps, relations, evidence and uid tables."""
    conn = conn or engine._store._conn
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND (name IN ('messages', 'message_relations', 'metadata') OR name LIKE 'host_uid%') ORDER BY name")]
    return {table: [tuple(row) for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid')]
            for table in tables}


class Crash(BaseException):
    """Escapes each writer's Exception handler, like a process exit before COMMIT."""


@pytest.mark.parametrize("fault", ["error", "crash"])
@pytest.mark.parametrize("boundary", ["b", "c", "d", "e", "f", "INSERT", "h", "i", "j", "k"])
def test_fault_at_each_former_commit(ingest_case, monkeypatch, caplog, boundary, fault):
    caplog.set_level("DEBUG")
    engine, seed, host = ingest_case
    store = engine._store
    before = durable(engine)
    frontier = copy.deepcopy(engine._frontier)
    receipt = {"blocked": False, "used": False, "stamp": None, "row": host[0],
               "content": "seed", "keys": engine._identity_anchor_occurrence_keys(host[0]),
               "store": store, "home": str(engine._hermes_home), "conversation": "conv",
               "session": "S", "index": 0, "max_store_id": seed}
    engine._identity_anchor_recoveries = [receipt]
    engine._identity_anchor_recovery = receipt
    previous_receipt = dict(receipt)
    fired = []

    def fail(point):
        if point == boundary and not fired:
            store.write_metadata_json(["transaction-probe:" + point], "true")
            fired.append(point)
            raise (Crash if fault == "crash" else sqlite3.OperationalError)("injected former commit " + point)

    def wrap(target, name, point, after=None):
        original = getattr(target, name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if after:
                after(*args, **kwargs)
            fail(point(*args, **kwargs) if callable(point) else point)
            return result

        monkeypatch.setattr(target, name, wrapped)

    capture = engine._capture_host_rewrites

    def rewrites(messages):
        capture(messages)
        if boundary == "b":
            row = store.get(seed)
            changed = {**messages[0], "content": " seed"}
            engine._host_rewrite_state()[0][seed] = (engine._message_replay_identity(row), row["content"], [changed])
            capture([changed])

    monkeypatch.setattr(engine, "_capture_host_rewrites", rewrites)
    wrap(engine, "_capture_host_rewrite", "b")
    reconcile = engine._reconcile_ingest_cursor_from_store
    rewrite_carry = store.write_metadata_json
    in_c = []

    def metadata(keys, *args, **kwargs):
        result = rewrite_carry(keys, *args, **kwargs)
        if in_c and any(key.startswith("compaction_commit_proof:") for key in keys):
            fail("c")
        return result

    monkeypatch.setattr(store, "write_metadata_json", metadata)

    def reconciled(*args, **kwargs):
        result = reconcile(*args, **kwargs)
        key = engine._replay_snapshot_metadata_key("compaction_commit_proof")
        store.write_metadata_json([key], json.dumps({"carry_ranges": []}))
        in_c.append(True)
        try:
            engine._rewrite_own_carry_ranges([], "transaction probe", [])
        finally:
            in_c.clear()
        return result

    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", reconciled)
    backfill = store.backfill_observed_at_many

    def backfilled(pairs):
        result = backfill([*pairs, (seed, 10.)])
        fail("d")
        return result

    monkeypatch.setattr(store, "backfill_observed_at_many", backfilled)
    prematch = engine._identity_anchor_prematch

    def prematched(*args, **kwargs):
        if boundary == "e":
            engine._capture_merged_user_head({**host[0], "content": "seed\n\nextra", "_merged_turn_prefix": "seed"},
                                             [(store.get(seed), ())], set())
        return prematch(*args, **kwargs)

    monkeypatch.setattr(engine, "_identity_anchor_prematch", prematched)
    wrap(engine, "_record_ws_host_rewrite", "e")

    def anchor_evidence(plan, remainder_ids=None):
        store.add_message_relations([[(seed, "alt_stamp", None, None, 10.)]])
        store.write_metadata_json([engine._identity_anchor_carry_key()], "[]")

    wrap(engine, "_identity_anchor_commit", lambda plan, remainder_ids=None: "f" if remainder_ids is None else "i",
         anchor_evidence)
    wrap(store, "_append_protected_batch", "INSERT")
    for point, name in (("h", "_identity_anchor_mark_carried"), ("j", "_identity_anchor_record_versions")):
        def evidence(*args, _point=point, **kwargs):
            newest = store.get_session_tail("S", limit=1)[0]["store_id"]
            store.add_message_relations([[(newest, "carried_stamp" if _point == "h" else "supersedes",
                                          seed, None, 10.)]])
        wrap(engine, name, point, evidence)
    wrap(store, "_host_uid_write", "k")
    aborts = fault == "crash" or boundary == "INSERT"
    if fault == "crash":
        with pytest.raises(Crash):
            engine._ingest_messages(host)
    else:
        engine.ingest(host)
    assert fired == [boundary]
    assert not store._conn.in_transaction
    assert store.read_metadata_json("transaction-probe:" + boundary) is None
    if aborts:
        assert durable(engine) == before
        assert engine._frontier.cursor == frontier.cursor
        assert engine._frontier.commit_proof == frontier.commit_proof
        assert engine._frontier.needs_reconcile
        assert receipt == previous_receipt
        assert not getattr(engine, "_host_rewrite_overrides", {})
    else:
        assert [row["content"] for row in store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]
        assert engine._ingest_cursor == len(host)
        assert engine._ingest_failure_count == 0
        assert any("failed" in record.getMessage() for record in caplog.records)
        if boundary in {"h", "j", "k"}:
            newest = store.get_session_tail("S", limit=1)[0]["store_id"]
            assert bool(store.get_message_relations([newest], "carried_stamp")) == (boundary != "h")
            assert bool(store.get_message_relations([newest], "supersedes")) == (boundary != "j")
        if boundary == "k":
            assert engine._host_uid_counters["errors"] == 1
    engine.ingest(host)
    rows = store.get_session_messages("S")
    assert [row["content"] for row in rows] == ["seed", "seed reply", "fresh"]
    assert len(store._conn.execute("SELECT * FROM host_uid_bindings WHERE uid='fresh-uid'").fetchall()) == (
        0 if boundary == "k" and not aborts else 1)
    engine.ingest(host)
    assert [row["content"] for row in store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]


@pytest.mark.parametrize("load_fails", [False, True])
def test_encoder_warmup_precedes_begin_and_runs_once_across_retry(ingest_case, monkeypatch, load_fails):
    import hermes_lcm.engine as engine_module

    engine, _seed, host = ingest_case
    store, events, attempts = engine._store, [], []
    reserve = store.ingest_atomic

    def encoder_load():
        events.append(("encoder", store._conn.in_transaction))
        if load_fails:
            raise RuntimeError("encoder unavailable")

    @contextmanager
    def admission(**kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database is locked")
        with reserve(**kwargs):
            yield

    monkeypatch.setattr(engine_module, "_get_encoder", encoder_load)
    monkeypatch.setattr(store, "ingest_atomic", admission)
    store._conn.set_trace_callback(lambda sql: events.append(("begin", True)) if sql == "BEGIN IMMEDIATE" else None)
    try:
        engine._ingest_messages(host)
    finally:
        store._conn.set_trace_callback(None)
    assert events == [("encoder", False), ("begin", True)] and len(attempts) == 2
    assert [row["content"] for row in store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]


@pytest.mark.parametrize("boundary", ["SAVEPOINT", "INSERT"])
def test_lost_ingest_transaction_aborts_instead_of_reopening(ingest_case, monkeypatch, boundary):
    engine, seed, host = ingest_case
    store, lost = engine._store, []
    before, failures = durable(engine), engine._ingest_failure_count

    def read():
        assert store._transaction_owner.owns_connection(store._conn)
        store.write_metadata_json(["lost-owner-evidence"], "true")
        store._conn.execute("SELECT 1").fetchone()
        if not lost:
            lost.append(True)
            store._conn.execute("ROLLBACK")
            raise sqlite3.OperationalError("simulated transaction-aborting read")

    if boundary == "SAVEPOINT":
        original = engine._durable_commit_proof_payload

        def proof(*args, **kwargs):
            if store._transaction_owner.owns_connection(store._conn) and not engine._ingest_cursor_needs_reconcile:
                read()  # The classifier catches this read failure before its next evidence scope.
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, "_durable_commit_proof_payload", proof)
    else:
        original = store._append_protected_batch

        def append(*args, **kwargs):
            try:
                read()
            except sqlite3.Error:
                pass  # Model a swallowed read error immediately before the owned INSERT.
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "_append_protected_batch", append)
    engine.ingest(host)
    assert lost == [True] and engine._ingest_failure_count == failures + 1
    assert engine._last_ingest_error == "_SqliteSavepointError: Foreground ingest transaction lost before " + boundary
    assert durable(engine) == before and not store._conn.in_transaction
    engine.ingest(host)
    assert engine._ingest_failure_count == failures + 1
    assert store.read_metadata_json("lost-owner-evidence") is True
    assert store.get(seed)["observed_at"] == 10.
    assert len(store._conn.execute("SELECT 1 FROM host_uid_bindings WHERE uid='fresh-uid'").fetchall()) == 1
    engine.ingest(host)
    assert [row["content"] for row in store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]


def test_crash_after_savepoint_release_rolls_back_whole_ingest(ingest_case, monkeypatch):
    engine, seed, host = ingest_case
    before, released = durable(engine), []
    original = engine._ingest_evidence

    @contextmanager
    def evidence():
        with original():
            yield
        if not released:
            released.append(True)
            raise Crash("after RELEASE")

    monkeypatch.setattr(engine, "_ingest_evidence", evidence)
    with pytest.raises(Crash):
        engine._ingest_messages(host)
    assert released and durable(engine) == before
    engine.ingest(host)
    assert [row["content"] for row in engine._store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]


def test_begin_precedes_classification_and_reads_see_own_evidence(ingest_case, monkeypatch):
    engine, seed, host = ingest_case
    store, trace, observed = engine._store, [], []
    store._conn.set_trace_callback(trace.append)
    prematch = engine._identity_anchor_prematch

    def matched(*args, **kwargs):
        assert store._conn.in_transaction
        assert store.get(seed)["observed_at"] == 10.  # prefix backfill -> anchor
        result = prematch(*args, **kwargs)
        engine._record_ws_host_rewrite(store.get(seed), host[0])
        assert engine._host_rewrite_override_content(store.get(seed)) == "seed"
        assert store.read_metadata_json(f"host_rewrite_identity:{seed}")["content"] == "seed"
        observed.append(True)
        return result

    monkeypatch.setattr(engine, "_identity_anchor_prematch", matched)
    try:
        engine._ingest_messages(host)
    finally:
        store._conn.set_trace_callback(None)
    assert observed == [True]
    assert trace[0] == "BEGIN IMMEDIATE"
    assert [sql for sql in trace if sql in ("BEGIN IMMEDIATE", "COMMIT")] == ["BEGIN IMMEDIATE", "COMMIT"]
    assert [row["content"] for row in store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]
    assert engine._last_ingest_attribution["rows_written"] == 1
    assert engine._last_ingest_attribution["bytes_written"] > 0
    assert engine._last_ingest_attribution["lock_hold_ms"] >= 0
    assert engine._last_ingest_attribution["busy_snapshot_retry"] is False


def test_c2_reads_uncommitted_rows_relations_carry_and_uid_evidence(ingest_case, monkeypatch):
    engine, seed, host = ingest_case
    store, seen = engine._store, []
    marker = engine._identity_anchor_mark_carried

    def marked(messages, stored_at, plan):
        result = marker(messages, stored_at, plan)
        newest = stored_at[2]
        # C2 sees C1 and INSERT; another connection sees none.
        assert store.get(newest)["content"] == "fresh"
        store.add_message_relations([[(newest, "supersedes", seed, None, 20.)]])
        assert store.get_message_relations([newest], "supersedes")[0]["related_store_id"] == seed
        key = engine._identity_anchor_carry_key()
        store.write_metadata_json([key], json.dumps([["S", seed - 1, seed]]))
        assert store.read_metadata_json(key) == [["S", seed - 1, seed]]
        lineage = engine._host_uid_lineage_key()[0]
        store.add_host_uid_bindings(lineage, [(newest, "fresh-uid", "canonical", "stored_new")])
        assert (newest, "canonical") in store.host_uid_bindings_for(lineage, ["fresh-uid"])["fresh-uid"]
        with sqlite3.connect(store.db_path) as reader:
            assert reader.execute("SELECT content FROM messages WHERE store_id=?", (newest,)).fetchone() is None
        seen.append(True)
        return result

    monkeypatch.setattr(engine, "_identity_anchor_mark_carried", marked)
    engine.ingest(host)
    assert seen == [True]
    assert [row["content"] for row in store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]


def test_second_writer_waits_and_reader_observes_whole_commit(ingest_case, monkeypatch):
    engine, seed, host = ingest_case
    ready, done = threading.Event(), threading.Event()
    observations, errors, threads = [], [], []
    append = engine._store._append_protected_batch
    before = durable(engine)

    def writer():
        try:
            with sqlite3.connect(engine._store.db_path, timeout=5) as conn:
                ready.set()
                conn.execute("BEGIN IMMEDIATE")
                observations.append(durable(engine, conn))
                conn.execute("INSERT INTO metadata(key, value) VALUES ('second-writer', 'true')")
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    def appended(*args, **kwargs):
        result = append(*args, **kwargs)
        with sqlite3.connect(engine._store.db_path) as reader:
            assert durable(engine, reader) == before
        thread = threading.Thread(target=writer)
        threads.append(thread)
        thread.start()
        assert ready.wait(2)
        assert not done.wait(.05)
        return result

    monkeypatch.setattr(engine._store, "_append_protected_batch", appended)
    try:
        engine._ingest_messages(host)
    finally:
        for thread in threads:
            thread.join(6)
    assert done.is_set() and not errors
    assert len(observations) == 1
    after = durable(engine)
    after["metadata"] = [row for row in after["metadata"] if row[0] != "second-writer"]
    assert observations[0] == after
    assert engine._ingest_failure_count == 0
    assert [row["content"] for row in engine._store.get_session_messages("S")].count("fresh") == 1
    assert after != before


@pytest.mark.parametrize("cache", ["_watch_stored_user_rows", "_identity_anchor_remember",
                                  "_remember_active_replay_messages"])
def test_post_commit_cache_failure_never_restores_old_cursor(ingest_case, monkeypatch, cache):
    engine, seed, host = ingest_case

    def fail(*args, **kwargs):
        assert not engine._store._conn.in_transaction
        raise RuntimeError("injected cache failure")

    with monkeypatch.context() as patch:
        patch.setattr(engine, cache, fail)
        engine.ingest(host)
    assert engine._ingest_cursor == len(host)
    assert engine._ingest_failure_count == 1
    engine.ingest(host)
    assert [row["content"] for row in engine._store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]
    assert engine._last_ingest_attribution["rows_written"] == 0


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_busy_snapshot_restarts_classification_once(ingest_case, monkeypatch, failures):
    engine, seed, host = ingest_case
    append, classify = engine._store._append_protected_batch, engine._capture_host_rewrites
    classifications, inserts = [], []
    before = durable(engine)

    def classified(messages):
        classifications.append(True)
        return classify(messages)

    def appended(*args, **kwargs):
        inserts.append(True)
        result = append(*args, **kwargs)
        if len(inserts) <= failures:
            error = sqlite3.OperationalError("injected busy snapshot")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
            raise error
        return result

    monkeypatch.setattr(engine, "_capture_host_rewrites", classified)
    monkeypatch.setattr(engine._store, "_append_protected_batch", appended)
    engine.ingest(host)
    assert len(classifications) == len(inserts) == (2 if failures else 1)
    assert engine._last_ingest_attribution["busy_snapshot_retry"] is bool(failures)
    assert not engine._store._conn.in_transaction
    if failures == 2:
        assert durable(engine) == before
        assert engine._ingest_failure_count == 1
        assert engine._ingest_cursor_needs_reconcile
        engine.ingest(host)
    assert [row["content"] for row in engine._store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]
    engine.ingest(host)
    assert [row["content"] for row in engine._store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]


@pytest.mark.parametrize("code", [None, sqlite3.SQLITE_BUSY, sqlite3.SQLITE_BUSY_SNAPSHOT])
@pytest.mark.parametrize("writer", ["marker", "uid", "placeholder_count", "placeholder_ordinal"])
def test_fail_open_handlers_isolate_io_and_never_swallow_busy(ingest_case, monkeypatch, writer, code):
    engine, seed, host = ingest_case
    prefix = ("ignored_active_replay_placeholder_hash_counts:" if writer.endswith("count") else
              "ignored_active_replay_placeholder_hash_ordinals:")
    before = {key: engine._store.read_metadata_json(key) for key in (
        engine._ignored_placeholder_count_metadata_keys() if writer.endswith("count")
        else engine._ignored_placeholder_ordinal_metadata_keys())}
    target, name = ((engine, "_identity_anchor_mark_carried") if writer == "marker" else
                    (engine._store, "add_host_uid_bindings" if writer == "uid" else "write_metadata_json"))
    original, fired = getattr(target, name), []

    def written(*args, **kwargs):
        result = original(*args, **kwargs)
        selected = writer in {"marker", "uid"} or any(key.startswith(prefix) for key in args[0])
        if selected and not fired:
            fired.append(True)
            error = sqlite3.OperationalError("injected I/O error" if code is None else "injected contention")
            if code is not None:
                error.sqlite_errorcode = code
            raise error
        return result

    monkeypatch.setattr(target, name, written)
    engine.ingest(host)
    assert fired == [True] and engine._ingest_failure_count == 0
    assert engine._last_ingest_attribution["busy_snapshot_retry"] is (code is not None)
    assert [row["content"] for row in engine._store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]
    if code is None and writer.startswith("placeholder"):
        assert {key: engine._store.read_metadata_json(key) for key in before} == before
    engine.ingest(host)
    assert [row["content"] for row in engine._store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]


def replay_fields(rows):
    return [(row.get("role"), row.get("content"), row.get("message_uid")) for row in rows]


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("shape", ["snapshot", "write"])
def test_ingest_ends_only_unowned_transactions(ingest_case, owned, shape):
    engine, seed, host = ingest_case
    store, trace = engine._store, []
    with store.atomic() if owned else nullcontext():
        if shape == "snapshot":
            store._conn.execute("BEGIN")
            store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()
            with sqlite3.connect(store.db_path) as publisher:
                publisher.execute("INSERT INTO metadata VALUES ('publisher', 'true')")
        else:
            store._conn.execute("INSERT INTO metadata VALUES ('dangling', 'true')")
        store._conn.set_trace_callback(trace.append)
        try:
            if owned and shape == "snapshot":
                with pytest.raises(sqlite3.OperationalError):
                    engine._ingest_messages(host)  # Only the holder may end its stale snapshot.
            else:
                assert replay_fields(engine._ingest_messages(host)) == replay_fields(host)
        finally:
            store._conn.set_trace_callback(None)
        assert ("COMMIT" not in trace) if owned else (trace[0] == "COMMIT" and trace[1] == "BEGIN IMMEDIATE")
        with sqlite3.connect(store.db_path) as reader:
            assert reader.execute("SELECT COUNT(*) FROM messages WHERE content='fresh'").fetchone()[0] == int(not owned)
            if shape == "write":
                assert bool(reader.execute("SELECT 1 FROM metadata WHERE key='dangling'").fetchone()) == (not owned)
        assert store._conn.in_transaction == owned
        assert len(store.get_session_messages("S")) == (2 if owned and shape == "snapshot" else 3)
    if owned and shape == "snapshot":
        assert replay_fields(engine._ingest_messages(host)) == replay_fields(host)
    assert [row["content"] for row in store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]
    engine._ingest_messages(host)
    assert [row["content"] for row in store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]


def test_foreign_lineage_lock_counts_error_and_commits_rows(ingest_case, monkeypatch):
    engine, _seed, host = ingest_case
    engine._host_uid_lineage_cache = {}
    previous = engine._host_uid_counters.get("errors", 0)

    def locked(*args):
        error = sqlite3.OperationalError("database is locked")
        error.sqlite_errorcode = sqlite3.SQLITE_BUSY
        raise error

    monkeypatch.setattr(engine, "_host_uid_read_lineage", locked)
    engine._ingest_messages(host)
    assert [row["content"] for row in engine._store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]
    assert engine._host_uid_counters.get("errors", 0) == previous + 1
    assert engine._ingest_failure_count == 0 and not engine._last_ingest_attribution["busy_snapshot_retry"]


def test_oob_positive_after_insert_does_not_survive_a_crash(ingest_case, monkeypatch):
    engine, _seed, host = ingest_case
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "oob-call", "type": "function",
            "function": {"name": "read", "arguments": "{}"}}]}
    result = {"role": "tool", "content": "R", "tool_call_id": "oob-call"}
    view = [*host, call, result]
    engine._ingest_messages(view)
    enriched = {**result, "content": "R\n\n[OUT-OF-BAND USER MESSAGE]keep this[/OUT-OF-BAND USER MESSAGE]"}
    view = [*view[:-1], enriched]
    engine._ingest_cursor_needs_reconcile = True
    before, memo = durable(engine), set(getattr(engine, "_out_of_band_durable_rows", ()))
    original = engine._identity_anchor_mark_carried

    def crash(*args):
        original(*args)
        assert any(row["content"] == enriched["content"] for row in engine._store.get_session_messages("S"))
        engine._message_replay_identity(enriched, strip_carrier=False)
        assert engine._pending_ingest_frontier.oob_rows - memo
        assert set(getattr(engine, "_out_of_band_durable_rows", ())) == memo
        raise Crash()

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_identity_anchor_mark_carried", crash)
        with pytest.raises(Crash):
            engine._ingest_messages(view)
    assert durable(engine) == before and set(getattr(engine, "_out_of_band_durable_rows", ())) == memo
    engine._ingest_messages(view)
    engine._ingest_messages(view)
    assert [row["content"] for row in engine._store.get_session_messages("S")].count(enriched["content"]) == 1


def test_j_failure_consumes_an_unstamped_version_after_commit(ingest_case, monkeypatch):
    engine, _seed, host = ingest_case
    message = _m("user", "M")
    view = [*host, message]
    engine._ingest_messages(view)
    message["content"] = "M rewritten"
    original, failed = engine._store.add_message_relations, []

    def relations(groups):
        result = original(groups)
        if not failed and any(row[1] == "supersedes" for group in groups for row in group):
            failed.append(True)
            raise OSError("injected j failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "add_message_relations", relations)
        engine._ingest_messages(view)
    assert failed == [True]
    assert all(entry[0] is not message for entry in engine._identity_anchor_versions)
    engine._ingest_messages(view)
    assert [row["content"] for row in engine._store.get_session_messages("S")].count("M rewritten") == 1


@pytest.mark.parametrize("busy", [False, True])
def test_begin_retry_preserves_the_post_compaction_proof(ingest_case, monkeypatch, busy):
    engine, _seed, host = ingest_case
    prefix = host[:2]
    identities = [engine._proof_replay_identity(row, strip_carrier=False) for row in prefix]
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = 2, False
    engine._compress_commit_proof = {"session_id": "S", "conversation_id": "conv", "durable": False,
        "output": identities, "input": identities, "output_metadata": copy.deepcopy(prefix), "consulted": False}
    reserve, classify = engine._store.ingest_atomic, engine._ingest_messages_transaction
    attempts, states = [], []

    @contextmanager
    def admission(**kwargs):
        attempts.append(True)
        if busy and len(attempts) == 1:
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        with reserve(**kwargs):
            yield

    def classified(*args, **kwargs):
        states.append(engine._ingest_cursor_needs_reconcile)
        return classify(*args, **kwargs)

    monkeypatch.setattr(engine._store, "ingest_atomic", admission)
    monkeypatch.setattr(engine, "_ingest_messages_transaction", classified)
    engine._ingest_messages(host)
    assert states == [False] and len(attempts) == 1 + int(busy)
    assert engine._compress_commit_proof is None
    assert [row["content"] for row in engine._store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]


def test_committed_lazy_schema_invalidates_a_batched_negative(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    store = engine._store
    try:
        assert store.get_carried_stamp_ids([1], batched=True) == set()
        assert store._memory_relation_connections[store._conn] is False
        original = engine._identity_anchor_mark_carried

        def marked(messages, stored_at, plan):
            original(messages, stored_at, plan)
            store.add_message_relations([[(stored_at[0], "carried_stamp", None, None, 10.)]])

        monkeypatch.setattr(engine, "_identity_anchor_mark_carried", marked)
        engine._ingest_messages([_m("user", "carried", 10., "carried-uid")])
        assert store._identity_anchor_schema_ready and store._host_uid_schema_ready
        assert store._conn in store._carried_relation_connections
        assert store._conn not in store._memory_relation_connections
        assert store.get_carried_stamp_ids([1], batched=True) == {1}
    finally:
        engine.shutdown()


def test_a_rolled_back_session_end_attempt_leaves_no_queued_tail(ingest_case, monkeypatch):
    engine, _seed, host = ingest_case
    append, attempts = engine._append_ingest_batch, []

    def busy_once(batch):
        attempts.append(True)
        if len(attempts) == 1:
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        return append(batch)

    monkeypatch.setattr(engine, "_append_ingest_batch", busy_once)
    engine._session_end_flush_active = True
    try:
        engine._ingest_messages(host)
    finally:
        engine._session_end_flush_active = False
    assert len(attempts) == 2 and engine._pending_session_tails == []
    assert [row["content"] for row in engine._store.get_session_messages("S")].count("fresh") == 1


def test_transaction_ownership_is_not_inherited_by_another_thread(ingest_case):
    engine, _seed, _host = ingest_case
    store, seen = engine._store, []
    with store.atomic():
        assert store._transaction_owner.owns_connection(store._conn)
        thread = threading.Thread(target=lambda: seen.append(store._transaction_owner.owns_connection(store._conn)))
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
    assert seen == [False]
    assert not store._transaction_owner.owns_connection(store._conn)


def test_session_end_classifies_read_only_and_freezes_once(ingest_case, monkeypatch):
    engine, _seed, host = ingest_case
    classify, statements, fallback = engine._ingest_messages_transaction, [], []

    def classified(*args, **kwargs):
        if getattr(engine, "_ingest_read_only", None) is not None:
            fallback.append(True)
            assert not engine._store._conn.in_transaction
            assert engine._host_rewrite_store is engine._store
            engine._store._conn.set_trace_callback(statements.append)
        return classify(*args, **kwargs)

    monkeypatch.setattr(engine, "_ingest_messages_transaction", classified)
    locker = sqlite3.connect(engine._store.db_path)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("S", host)
    finally:
        engine._store._conn.set_trace_callback(None)
        locker.rollback()
        locker.close()
    gc.collect()
    assert engine._store._conn.execute("SELECT 1").fetchone()[0] == 1
    assert fallback == [True] and len(engine._pending_session_tails) == 1
    assert engine._pending_session_tails[0]["messages"][-1]["content"] == "fresh"
    assert all(sql.lstrip().upper().startswith(("SELECT", "WITH", "PRAGMA")) for sql in statements)
    engine.on_session_start("other", platform="cli", context_length=200_000, conversation_id="other")
    assert [row["content"] for row in engine._store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]


def test_batched_negative_relation_probe_is_cached_inside_a_transaction(tmp_path):
    engine = _engine(tmp_path)
    store, statements = engine._store, []
    try:
        with store.atomic(immediate=True):
            store._conn.set_trace_callback(statements.append)
            try:
                assert store.get_carried_stamp_ids([1], batched=True) == set()
                assert store.get_carried_stamp_ids([1], batched=True) == set()
                assert store._memory_relation_connections[store._conn] is False
            finally:
                store._conn.set_trace_callback(None)
        assert sum("sqlite_master" in sql for sql in statements) == 1
    finally:
        engine.shutdown()


def test_profiled_assembly_scenarios_persist_placeholder_counts_back_to_back(tmp_path, monkeypatch):
    from tests.behaviour import assembly_trace_harness as harness

    setup, databases = harness.setup, []

    def without_prior_counts(engine, *args):
        result = setup(engine, *args)
        keys = engine._ignored_placeholder_count_metadata_keys()
        databases.append((engine._store.db_path, keys))
        engine._store._conn.executemany("DELETE FROM metadata WHERE key=?", [(key,) for key in keys])
        engine._store._conn.commit()
        return result

    monkeypatch.setattr(harness, "setup", without_prior_counts)
    for name in ("stub-first", "compress-scaffold"):
        case = next(case for case in harness.SCENARIOS if case.name == name)
        events = harness.observe(case, tmp_path / name)
        assert not any("ignored placeholder count metadata write failed" in event for event in events)
        database, keys = databases[-1]
        assert keys
        with sqlite3.connect(database) as reader:
            for key in keys:
                row = reader.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
                assert row is not None and isinstance(json.loads(row[0]), dict)


def test_ingest_outcome_snapshot_excludes_diagnostic_attribution(ingest_case, tmp_path):
    from tests.behaviour import ingest_trace_harness as harness

    engine, _seed, host = ingest_case
    canonical = harness.IngestCanonical(tmp_path, host)
    before = harness.snapshot(engine, canonical)
    assert "_last_ingest_attribution" not in before
    engine._last_ingest_attribution = dict(lock_hold_ms=999, rows_written=999, bytes_written=999, busy_snapshot_retry=True)
    assert harness.snapshot(engine, canonical) == before


def test_begin_waits_for_a_writer_within_the_existing_budget(ingest_case):
    engine, seed, host = ingest_case
    store, ready, attempted, released, errors = engine._store, threading.Event(), threading.Event(), threading.Event(), []
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000

    def writer():
        try:
            with sqlite3.connect(store.db_path) as holder:
                holder.execute("BEGIN IMMEDIATE")
                ready.set()
                assert attempted.wait(2)
                released.wait(.05)  # Hold through BEGIN's admission, well below the 30s budget.
        except BaseException as exc:
            errors.append(exc)
        finally:
            released.set()

    thread = threading.Thread(target=writer)
    thread.start()
    assert ready.wait(2)
    store._conn.set_trace_callback(lambda sql: attempted.set() if sql == "BEGIN IMMEDIATE" else None)
    try:
        assert replay_fields(engine._ingest_messages(host)) == replay_fields(host)
    finally:
        store._conn.set_trace_callback(None)
        thread.join(3)
    assert released.is_set() and not errors and engine._ingest_failure_count == 0
    assert engine._last_ingest_attribution["busy_snapshot_retry"] is False
    assert [row["content"] for row in store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]
