"""#2624 review regressions: declaration scope, host reads, isolation and races."""
from __future__ import annotations

import copy
import hashlib
import logging
import multiprocessing
import queue
import sqlite3
import sys
import threading
import time
import traceback

import pytest

# Spawned interpreters need the same package bootstrap as the pytest parent.
if "hermes_lcm" not in sys.modules:
    from tests import conftest as _bootstrap  # noqa: F401

from hermes_lcm import host_row_recovery as recovery
from hermes_lcm import maintenance_scheduler as schedulers
from hermes_lcm.engine import LCMEngine
from tests.test_issue_2624_host_row_recovery import (
    C, P, Host, assert_suffix, case as _case_fixture, pair, rows, tools,
)

case = _case_fixture


def _run_in_recovery_thread(job):
    errors = []

    def worker():
        try:
            job()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker, name="lcm-host-row-recovery", daemon=True)
    thread.start()
    thread.join(30)
    assert not thread.is_alive(), "recovery worker did not finish"
    if errors:
        raise errors[0]


def _tool_group(field="id"):
    group = copy.deepcopy(tools(results=1, reply=False)[1:3])
    group[0]["tool_calls"] = group[0]["tool_calls"][:1]
    if field != "id":
        call = group[0]["tool_calls"][0]
        call[field] = call.pop("id")
    return group


@pytest.mark.parametrize("boundary", [True, False], ids=["proof-boundary", "inside-suffix"])
@pytest.mark.parametrize("field", ["id", "tool_call_id"])
def test_reused_call_ids_recover_exact_suffix(case, boundary, field):
    first, second = _tool_group(field), _tool_group(field)
    second[1]["content"] += " second declaration"
    prefix = case.prefix
    before = []
    if boundary:
        child = case.start(C, conversation=True)
        prefix = prefix + first
        child.ingest(prefix)
        before = rows(case)
        assert len(before) == len(first)
    suffix = second if boundary else first + second
    case.host.lists[C] = prefix + suffix
    assert not case.run(case.jobs[0])
    recovered = rows(case)
    assert recovered[:len(before)] == before
    appended = recovered[len(before):]
    fields = ("role", "content", "tool_calls", "tool_call_id")
    actual = [case.live._store.to_openai_msg(row) for row in appended]
    assert [tuple(m.get(k) for k in fields) for m in actual] == [tuple(m.get(k) for k in fields) for m in suffix]
    assert all(row["session_id"] == C and row["conversation_id"] == P for row in appended)
    assert all(case.live._store.read_metadata_json("host_row_recovered:" + str(row["store_id"]))
               == {"version": 1, "issue": 2624} for row in appended)
    assert not case.run(case.jobs[0]) and rows(case) == recovered


def test_second_result_for_same_declaration_still_rejected(case):
    group = _tool_group()
    case.host.lists[C] = case.prefix + group + [copy.deepcopy(group[1])]
    warnings = case.run()
    assert not rows(case) and len(warnings) == 1
    assert "reason=suffix_rejected" in warnings[0]


@pytest.mark.parametrize("calls", [[], {}, "broken", [None], [{"function": {}}],
                                    [{"id": "same"}, {"tool_call_id": "same"}], [{"id": "same"}, {"id": "same"}]],
                         ids=["empty", "mapping", "string", "non-dict", "missing-id", "duplicate", "duplicate-same"])
def test_damaged_declaration_rejects_entire_leaf(case, calls):
    case.host.lists[C] = case.prefix + pair() + [{"role": "assistant", "content": "", "tool_calls": calls}]
    warnings = case.run()
    assert not rows(case) and len(warnings) == 1
    assert "reason=suffix_rejected" in warnings[0]
    assert not case.run() and not rows(case)


@pytest.mark.parametrize("results", [0, 1], ids=["unanswered", "partial"])
def test_crash_partial_declarations_remain_recoverable(case, results):
    suffix = tools(results=results, reply=False)
    case.host.lists[C] = case.prefix + suffix
    assert not case.run()
    assert_suffix(case, suffix)


def test_real_full_scheduler_warning_hides_raw_session_id(case, monkeypatch, caplog):
    scheduler = schedulers._RollupMaintenanceScheduler(kind="host-row recovery", thread_name="lcm-host-row-recovery")
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(15)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(recovery, "_HOST_ROW_RECOVERY_SCHEDULER", scheduler)
            patch.setattr(recovery, "_SEEN", set())
            assert scheduler.schedule(("test-database", "blocked"), blocked)
            assert entered.wait(10)
            for i in range(scheduler._max_pending_jobs):
                assert scheduler.schedule(("test-database", str(i)), lambda: None)
            caplog.clear()
            with caplog.at_level(logging.DEBUG):
                case.live._enqueue_host_row_recovery(P, {})
            warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1 and "queue is full" in warnings[0]
            assert P not in warnings[0] and C not in warnings[0]
            assert str(case.live._store.db_path) not in warnings[0]
            assert str(case.live._store.db_path.resolve()) not in warnings[0]
            db_digest = hashlib.sha256(case.live._store.db_identity.encode()).hexdigest()[:16]
            assert "host-row-recovery-db:" + db_digest in warnings[0]
            scope = repr((case.live._hermes_home, case.live._conversation_id, P))
            assert "host-row-recovery:" + hashlib.sha256(scope.encode()).hexdigest()[:16] in warnings[0]
            assert not recovery._SEEN
    finally:
        release.set()
        assert scheduler.quiesce(10)
        if scheduler._worker is not None:
            scheduler._worker.join(10)
            assert not scheduler._worker.is_alive()


@pytest.mark.parametrize("resolved", [True, False], ids=["lineage", "no-lineage"])
def test_uid_lineage_host_read_does_not_hold_store_write_lock(case, monkeypatch, caplog, resolved):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    suffix = pair(uids=True)
    case.host.lists[C] = case.prefix + suffix
    entered, release = threading.Event(), threading.Event()
    original = LCMEngine._host_uid_read_lineage
    capture, shadow = LCMEngine._host_uid_capture, LCMEngine._host_uid_shadow
    reads, captures, shadows = [], [], []

    def capture_outside(det, *args, **kwargs):
        captures.append(det._store.connection.in_transaction)
        return capture(det, *args, **kwargs)

    def blocked(det, path, session_id, **kwargs):
        reads.append(det._store.connection.in_transaction)
        entered.set()
        assert release.wait(15)
        return original(det, path, session_id, **kwargs) if resolved else (None, "unresolved")

    def shadow_inside(det, *args, **kwargs):
        shadows.append(det._store.connection.in_transaction)
        return shadow(det, *args, **kwargs)

    monkeypatch.setattr(LCMEngine, "_host_uid_capture", capture_outside)
    monkeypatch.setattr(LCMEngine, "_host_uid_read_lineage", blocked)
    monkeypatch.setattr(LCMEngine, "_host_uid_shadow", shadow_inside)
    thread = threading.Thread(target=case.jobs[0], name="lcm-host-row-recovery", daemon=True)
    caplog.clear()
    try:
        thread.start()
        assert entered.wait(10)
        with sqlite3.connect(case.live._db_path, timeout=0.2) as other:
            other.execute("BEGIN IMMEDIATE")
            other.commit()
    finally:
        release.set()
        thread.join(20)
    assert not thread.is_alive()
    assert reads == captures == [False]
    assert shadows == ([True] if resolved else [])
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    recovered = assert_suffix(case, suffix)
    conn = case.live._store.connection
    bindings = conn.execute(
        "SELECT store_id,uid FROM host_uid_bindings WHERE store_id IN (?,?)",
        tuple(row["store_id"] for row in recovered),
    ).fetchall() if conn.execute("SELECT 1 FROM sqlite_master WHERE name='host_uid_bindings'").fetchone() else []
    assert {(row[0], row[1]) for row in bindings} == (
        {(row["store_id"], msg["message_uid"]) for row, msg in zip(recovered, suffix)} if resolved else set())


@pytest.mark.parametrize("phase", ["before-read", "after-read"])
def test_candidate_adopted_on_tip_path_is_skipped(case, monkeypatch, caplog, phase):
    tips = iter([P, C] if phase == "before-read" else [P, P, C])

    def moving_tip(session_id):
        assert session_id == P
        return next(tips)

    monkeypatch.setattr(case.host, "get_compression_tip", moving_tip)
    before = recovery.host_row_recovery_counts()["on_tip_path"]
    assert not case.run() and not rows(case)
    assert recovery.host_row_recovery_counts()["on_tip_path"] == before + 1
    assert sum(kind == "read" for kind, sid in case.host.calls) == int(phase == "after-read")
    assert any("skipped on tip path" in r.getMessage() and r.levelno == logging.DEBUG for r in caplog.records)


def _spawn_recovery(config, home, db_path, state_path, messages, barrier, results):
    """Each process owns its detached engine, SQLite connection and first proof."""
    original = recovery._strict_proof_cursor
    proofs = []

    def synchronized(det, host_messages):
        cursor = original(det, host_messages)
        if not det._store.connection.in_transaction:
            proofs.append(cursor)
            if cursor is None:
                raise AssertionError("first proof failed")
            barrier.wait(timeout=20)
        return cursor

    try:
        recovery._strict_proof_cursor = synchronized
        before = recovery.host_row_recovery_counts()
        recovery._recover(LCMEngine, config, home, db_path, state_path, P, P, "acp", 200_000, Host({C: messages}))
        after = recovery.host_row_recovery_counts()
        results.put({"proofs": proofs, "delta": {k: after[k] - before[k] for k in ("recovered_rows", "raced", "unproven")}})
    except BaseException:
        results.put({"error": traceback.format_exc()})
    finally:
        recovery._strict_proof_cursor = original


def test_spawned_processes_recover_suffix_exactly_once(case, tmp_path):
    try:
        ctx = multiprocessing.get_context("spawn")
        barrier, results = ctx.Barrier(2), ctx.Queue()
    except (OSError, RuntimeError, ValueError, NotImplementedError) as exc:
        pytest.skip(f"spawn unavailable: {type(exc).__name__}")
    db_path = tmp_path / "race-copy.db"
    with sqlite3.connect(db_path) as conn:
        case.live._store.connection.backup(conn)
    config = copy.deepcopy(case.live._config)
    config.database_path = str(db_path)
    args = (config, case.live._hermes_home, str(db_path), str(case.state), case.host.lists[C], barrier, results)
    processes = [ctx.Process(target=_spawn_recovery, args=args) for _ in range(2)]
    started, receipts = [], []
    try:
        for process in processes:
            try:
                process.start()
            except (OSError, RuntimeError, NotImplementedError) as exc:
                pytest.skip(f"spawn unavailable: {type(exc).__name__}")
            started.append(process)
        deadline = time.monotonic() + 45
        for process in started:
            process.join(max(0, deadline - time.monotonic()))
        assert all(not p.is_alive() and p.exitcode == 0 for p in started), "spawned recovery failed or timed out"
        for _ in started:
            try:
                receipts.append(results.get(timeout=2))
            except queue.Empty:
                pytest.fail("spawned recovery did not return a receipt")
        assert all("error" not in receipt for receipt in receipts), receipts
        assert all(receipt["proofs"] == [len(case.prefix)] for receipt in receipts), receipts
        assert sorted(receipt["delta"]["recovered_rows"] for receipt in receipts) == [0, 2]
        assert sorted(receipt["delta"]["raced"] for receipt in receipts) == [0, 1]
        assert sorted(receipt["delta"]["unproven"] for receipt in receipts) == [0, 1]
        before = recovery.host_row_recovery_counts()["recovered_rows"]
        recovery._recover(LCMEngine, config, case.live._hermes_home, str(db_path), str(case.state),
                          P, P, "acp", 200_000, case.host)
        assert recovery.host_row_recovery_counts()["recovered_rows"] == before
        with sqlite3.connect(db_path) as conn:
            stored = conn.execute("SELECT session_id,conversation_id,role,content FROM messages WHERE session_id=? ORDER BY store_id", (C,)).fetchall()
            assert stored == [(C, P, m["role"], m["content"]) for m in pair()]
            assert conn.execute("SELECT count(*) FROM metadata WHERE key LIKE 'host_row_recovered:%'").fetchone()[0] == 2
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            process.close()
        results.close()
        results.join_thread()


@pytest.mark.parametrize("fail_open", [False, True], ids=["success", "storage-open-failure"])
def test_detached_close_preserves_live_prepared_leaves(case, monkeypatch, fail_open):
    from hermes_lcm import engine as engine_module, leaf_prep
    discarded = []
    monkeypatch.setattr(leaf_prep, "discard", lambda path: discarded.append(str(path)))
    failed_paths = []

    def fail_dag(path):
        failed_paths.append(str(path))
        raise RuntimeError("injected detached storage open failure")

    if fail_open:
        monkeypatch.setattr(engine_module, "SummaryDAG", fail_dag)
    warnings = case.run()
    if fail_open:
        assert failed_paths == [str(case.live._store.db_path)]
        assert len(warnings) == 1 and "reason=worker_error" in warnings[0]
        assert not rows(case)
    else:
        assert not warnings
        assert_suffix(case, pair())
    assert not discarded
    # Positive control: the same live engine's explicit cleanup still discards.
    case.live._discard_prepared_leaves()
    assert discarded == [str(case.live._store.db_path)]


@pytest.mark.parametrize("pattern,source,stored", [("C5*", "acp", False), ("cli:*", "cli", False),
                                                   ("cli:*", "acp", True), ("Z*", "acp", True)],
                         ids=["session-id", "child-platform", "other-platform", "unmatched"])
@pytest.mark.parametrize("field", ["ignore_session_patterns", "stateless_session_patterns"])
def test_operator_filtered_child_is_never_stored(case, field, pattern, source, stored):
    setattr(case.live._config, field, [pattern])  # never matches the bound session (bound platform acp)
    with sqlite3.connect(case.state) as conn:
        conn.execute("UPDATE sessions SET source=? WHERE id=?", (source, C))
    case.host.lists[C] = case.prefix + pair()
    recovery._SEEN.clear()
    case.live._enqueue_host_row_recovery(P, {})
    before = recovery.host_row_recovery_counts().get("session_filtered", 0)
    assert not case.run()
    if stored:
        assert_suffix(case, pair())
    else:
        assert not rows(case) and ("read", C) not in case.host.calls
        assert recovery.host_row_recovery_counts()["session_filtered"] == before + 1


@pytest.mark.parametrize("field", ["ignore_session_patterns", "stateless_session_patterns"])
def test_filtered_intermediate_prunes_its_subtree(case, field):
    leaf = "D519"
    with sqlite3.connect(case.state) as conn:
        conn.execute("UPDATE sessions SET ended_at=2, end_reason='compression' WHERE id=?", (C,))
        conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)", (leaf, C, None, None, None, "acp"))
    case.host.lists[leaf] = case.prefix + pair()
    setattr(case.live._config, field, ["C5*"])
    recovery._SEEN.clear()
    case.live._enqueue_host_row_recovery(P, {})
    before = recovery.host_row_recovery_counts()
    assert not case.run()
    after = recovery.host_row_recovery_counts()
    assert not rows(case, leaf) and ("read", leaf) not in case.host.calls
    assert after["session_filtered"] == before["session_filtered"] + 1
    assert after["candidates"] == before["candidates"] and after["no_provenance"] == before["no_provenance"]


def test_bind_without_handle_keeps_host_handle(case):
    held = case.live._host_session_db
    case.live.bind_session_state(None, P)
    assert case.live._host_session_db is held


def test_payload_protection_runs_outside_store_write_lock(case, monkeypatch):
    original, probed = recovery.protect_messages_for_ingest, []

    def probe(*args, **kwargs):
        conn = sqlite3.connect(case.live._store.db_path, timeout=0)
        try:
            conn.execute("BEGIN IMMEDIATE")  # fails at once while recovery holds the write lock
            conn.rollback()
            probed.append(True)
        finally:
            conn.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery, "protect_messages_for_ingest", probe)
    case.host.lists[C] = case.prefix + pair()
    assert not case.run()
    assert probed == [True]
    assert_suffix(case, pair())


def test_scheduler_key_separates_bindings_of_one_session(case, monkeypatch):
    keys = []
    monkeypatch.setattr(recovery._HOST_ROW_RECOVERY_SCHEDULER, "schedule", lambda key, job: keys.append(key) or True)
    recovery._SEEN.clear()
    case.live._enqueue_host_row_recovery(P, {})
    monkeypatch.setattr(case.live, "_conversation_id", "another-conversation")
    case.live._enqueue_host_row_recovery(P, {})
    assert len(keys) == 2 and keys[0][0] == keys[1][0] and keys[0][1] != keys[1][1]
    assert len(recovery._SEEN) == 2


def test_every_skip_reason_is_counted(case, monkeypatch):
    before = recovery.host_row_recovery_counts()
    case.host.lists[C] = [{"role": "user", "content": "no proof covers this"}] + pair()
    assert len(case.run()) == 1  # the strict proof fails: one WARNING, and its own counter
    monkeypatch.setattr(recovery, "_candidates", lambda *args: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
    assert not case.run()
    after = recovery.host_row_recovery_counts()
    assert after["proof"] == before["proof"] + 1 and after["walk_error"] == before["walk_error"] + 1


def test_profile_switch_drops_the_host_reader(case, tmp_path):
    scheduled = len(case.jobs)
    (other := tmp_path / "other-profile").mkdir()
    case.live.on_session_start(P, platform="acp", context_length=200_000, hermes_home=str(other))
    assert case.live._hermes_home == str(other) and case.live._host_session_db is None
    assert len(case.jobs) == scheduled  # no job pairs this profile's store with the other profile's reader

