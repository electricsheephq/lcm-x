"""Hermes-shaped S0 lifecycle, occurrence, retry and fail-open acceptance."""
from copy import deepcopy
import importlib
import json
import sqlite3
import sys
import threading
import time

import pytest

from tests.harness import engine_context
from tests.invariants.test_issue_1508_pending_predecessor import register
from tests.test_sender_columns_schema import derived_snapshot, prepare_derived


def capture_module():
    return importlib.import_module("hermes_lcm.sender_capture")


@pytest.fixture(autouse=True)
def clean_capture_registry():
    def clear():
        for name, cap in list(sys.modules.items()):
            if name.endswith(".sender_capture"):
                with cap._LOCK:
                    cap._PENDING.clear()
                    cap._RETIRED.clear()
    clear()
    yield
    clear()


def message(index=0, **extra):
    return dict(role="user", content=f"synthetic request {index}", timestamp=100.0 + index, **extra)


def payload(view, turn="turn-a", **extra):
    return dict(session_id="S", turn_id=turn, platform="telegram", sender_id="synthetic-a",
                conversation_history=view, **extra)


def begin(engine, view, turn="turn-a", **extra):
    values = payload(view, turn)
    values.update(extra)
    return capture_module().create_pending(engine, values)


def senders(engine):
    return list(engine._store.connection.execute("SELECT sender_id, sender_source FROM messages ORDER BY store_id"))


def expected():
    return ("telegram:synthetic-a", "gateway_principal")


def assert_capture(engine, expected_rows):
    # Avoid sender values in failure diagnostics, even for synthetic fixtures.
    if senders(engine) != expected_rows:
        raise AssertionError("sender alignment differs")


@pytest.fixture
def hook_context(tmp_path, monkeypatch):
    ctx = register(tmp_path, monkeypatch)
    ctx.engine.on_session_start("S", platform="telegram", context_length=128000, conversation_id="conv")
    assert ctx.engine.drain_rollup_maintenance(timeout=5)
    monkeypatch.setattr(ctx.module, "_pre_llm_context", lambda *_: None)
    try:
        yield ctx
    finally:
        ctx.engine.shutdown()


def test_t1_preflight_before_hook_updates_only_watched_row(hook_context, record_property, caplog):
    ctx, view = hook_context, []
    for i in range(14):
        view.extend([message(i), dict(role="assistant", content="synthetic reply", timestamp=100.5 + i)])
    current = message(20)
    view.append(current)
    assert len(view) > ctx.engine._config.fresh_tail_count
    ctx.engine.should_compress_preflight(view)
    row = ctx.engine._store.get_session_messages("S")[-1]
    prepare_derived(ctx.engine._store)
    conn = ctx.engine._store.connection
    before, changes = derived_snapshot(conn), conn.total_changes
    admission_errors = []
    class AdmissionProbe:
        def __getattr__(self, name):
            return getattr(conn, name)
        def execute(self, sql, *args):
            try:
                return conn.execute(sql, *args)
            except Exception as exc:
                admission_errors.append({"operation": sql.split()[0], "error_type": type(exc).__name__,
                                         "sqlite_code": getattr(exc, "sqlite_errorcode", None)})
                raise
    ctx.engine._store._conn = AdmissionProbe()
    try:
        with caplog.at_level("DEBUG"):
            ctx.hooks["pre_llm_call"](**payload(view))
    finally:
        ctx.engine._store._conn = conn
    record_property("s0_t1_numeric", {"sender_write_changes": conn.total_changes - changes,
                    "derived_equal": derived_snapshot(conn) == before,
                    "admission_errors": admission_errors,
                    "sender_reason_codes": [r.getMessage() for r in caplog.records
                                            if r.getMessage().startswith("sender_capture:")]})
    assert conn.total_changes == changes + 1
    assert derived_snapshot(conn) == before
    assert ctx.engine._store.get_message_senders([row["store_id"]]) == {row["store_id"]: expected()}
    capture_module().try_update(ctx.engine)
    assert conn.total_changes == changes + 1


@pytest.mark.parametrize("skip", ["cheap-gate", "cooldown", "codex-native"])
def test_t2_skipped_preflight_first_insert_then_post_turn(hook_context, skip):
    ctx, view = hook_context, [message()]
    if skip == "cooldown":
        ctx.engine._last_compaction_at = time.time()
    # Host skip means the preflight method is not called at all.
    ctx.hooks["pre_llm_call"](**payload(view))
    view[0]["message_uid"] = "a" * 32
    ctx.engine.ingest(view)
    assert_capture(ctx.engine, [expected()])
    changes = ctx.engine._store.connection.total_changes
    ctx.hooks["post_llm_call"](session_id="S", conversation_id="conv", conversation_history=view,
                               platform="telegram")
    assert_capture(ctx.engine, [expected()])
    assert ctx.engine._store.connection.total_changes == changes


def test_t3_interrupted_tail_catchup_and_abandoned_worker(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        a, b = message(0), message(1)
        begin(e, [a])
        e.on_turn_complete([deepcopy(a)], interrupted=True, turn_id="turn-a")
        begin(e, [a, b], turn="turn-b", sender_id="synthetic-b")
        assert begin(e, [a], turn="turn-a") is None
        e.ingest([a, b])
        assert_capture(e, [(None, None), ("telegram:synthetic-b", "gateway_principal")])
        e.on_turn_complete([a, b], turn_id="turn-b")
        assert capture_module().snapshot(e) is None


def test_t5_session_end_retry_freeze_trim_and_alignment(tmp_path, monkeypatch):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        old, current = message(0), message(1)
        begin(e, [old, current])
        original = e._append_ingest_batch
        def busy(_batch):
            raise sqlite3.OperationalError("database is locked")
        monkeypatch.setattr(e, "_append_ingest_batch", busy)
        e._session_end_flush_active = True
        with pytest.raises(sqlite3.OperationalError):
            e._ingest_messages([old, current])
        tail = e._pending_session_tails[0]
        assert "senders" in tail
        current["content"] = "synthetic mutated after freeze"
        monkeypatch.setattr(e, "_append_ingest_batch", original)
        e._store.append_batch("S", [old], conversation_id="conv")
        e._trim_replayed_pending_tail(tail)
        assert len(tail["messages"]) == len(tail["senders"]) == 1
        e._retry_pending_session_tails()
        assert not e._pending_session_tails
        assert_capture(e, [(None, None), expected()])


MATRIX = [
    ("native", {}, {}, True),
    ("display-kind", {}, {"display_kind": "notice"}, False),
    ("bot-author", {"turn_author": {"id": "synthetic-a", "is_bot": True}}, {}, False),
    ("other-author", {"turn_author": {"id": "synthetic-other"}}, {}, False),
    ("empty", {"sender_id": ""}, {}, False),
    ("too-long", {"sender_id": "x" * 201}, {}, False),
    ("control", {"sender_id": "synthetic\ncontrol"}, {}, False),
    ("cli", {"platform": "cli", "sender_id": ""}, {}, False),
    ("cron", {"platform": "cron", "sender_id": ""}, {}, False),
    ("subagent", {"platform": "subagent", "sender_id": ""}, {}, False),
    ("invalid-platform", {"platform": "not a platform"}, {}, False),
    ("missing-turn", {"turn_id": ""}, {}, False),
    ("wrong-session", {"session_id": "other"}, {}, False),
    ("preanswer-off", {}, {}, True),
]


@pytest.mark.parametrize("scenario,overrides,row_extra,stamped", MATRIX, ids=[r[0] for r in MATRIX])
def test_t6_transport_matrix(hook_context, monkeypatch, scenario, overrides, row_extra, stamped):
    if scenario == "preanswer-off":
        monkeypatch.setenv("LCM_PREANSWER", "off")
    ctx, view = hook_context, [message(**row_extra)]
    values = payload(view)
    values.update(overrides)
    ctx.hooks["pre_llm_call"](**values)
    ctx.engine.ingest(view)
    assert_capture(ctx.engine, [expected() if stamped else (None, None)])
    assert ctx.engine._store.connection.execute("SELECT count(*) FROM messages WHERE sender_source='gateway'").fetchone()[0] == 0


@pytest.mark.parametrize("scenario", ["desktop-tui-telegram", "heartbeat-loop", "viewer-submit"])
def test_t6_record_admitted_shapes(hook_context, scenario, record_property):
    ctx, view = hook_context, [message()]
    ctx.hooks["pre_llm_call"](**payload(view))
    ctx.engine.ingest(view)
    rows = senders(ctx.engine)
    assert len(rows) == 1  # Observation only: admission semantics are outside S0.
    assert rows[0][1] != "gateway"
    record = {"scenario": scenario, "sender_label": "[redacted synthetic principal]",
              "sender_source": rows[0][1]}
    record_property("s0_admitted_shape", record)
    # The caller snapshots these safe outcomes into t6-recorded.json after GREEN.
    assert json.loads(json.dumps(record))["scenario"] == scenario


def test_t7_engine_busy_hook_has_no_pending(hook_context, monkeypatch):
    ctx, view = hook_context, [message()]
    from hermes_lcm.engine_registry import ActiveEngineUseResult, ActiveEngineUseStatus
    def busy(_operation, **options):
        assert options["timeout"] == 5.0
        return ActiveEngineUseResult(ActiveEngineUseStatus.BUSY)
    monkeypatch.setattr(ctx.engine, "_run_stably", busy)
    thread = threading.Thread(target=lambda: ctx.hooks["pre_llm_call"](**payload(view)))
    thread.start()
    thread.join(2)
    assert not thread.is_alive()
    ctx.engine.ingest(view)
    assert_capture(ctx.engine, [(None, None)])
    assert capture_module().snapshot(ctx.engine) is None


def test_t7_registry_bounded_supersession_ttl_and_late_creation(tmp_path, monkeypatch):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        cap = capture_module()
        start = time.monotonic()
        monkeypatch.setattr(cap.time, "monotonic", lambda: start)
        for i in range(1000):
            begin(e, [message()], turn=f"turn-{i}")
        assert len(cap._PENDING) <= 256 and len(cap._RETIRED) <= 1024
        assert begin(e, [message()], turn="turn-998") is None
        monkeypatch.setattr(cap.time, "monotonic", lambda: start + 901)
        begin(e, [message()], turn="turn-after-ttl")
        assert len(cap._PENDING) <= 1
        # Distinct bindings exercise the live-entry cap, not just supersession.
        original = e._session_id
        for i in range(300):
            e._session_id = f"binding-{i}"
            begin(e, [message()], turn=f"cap-{i}", session_id=e._session_id)
        e._session_id = original
        assert len(cap._PENDING) <= 256 and len(cap._RETIRED) <= 1024


@pytest.mark.parametrize("lock_kind", ["sqlite", "store-lock"])
def test_t7_update_contention_is_nonblocking_and_retried_after_ingest(tmp_path, lock_kind):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        current = message()
        e.ingest([current])
        # Only the test's holder may contend: the bind-time rollup pass would
        # otherwise race the retried UPDATE after release.
        assert e.drain_rollup_maintenance(timeout=10)
        conn = e._store.connection
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        ready, release = threading.Event(), threading.Event()
        def holder():
            if lock_kind == "sqlite":
                other = sqlite3.connect(e._store.db_path)
                other.execute("BEGIN IMMEDIATE")
                ready.set()
                release.wait(2)
                other.rollback()
                other.close()
            else:
                with e._store._write_lock:
                    ready.set()
                    release.wait(2)
        thread = threading.Thread(target=holder)
        thread.start()
        assert ready.wait(2)
        try:
            begin(e, [current])
            assert thread.is_alive() and not release.is_set()
            assert not conn.in_transaction
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == timeout
            assert_capture(e, [(None, None)])
        finally:
            release.set()
            thread.join(2)
        e.ingest([current])
        assert_capture(e, [expected()])


def test_t7_hook_thread_and_cold_prototype_share_registry(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        current = message(message_uid="b" * 32)
        ready = threading.Event()
        thread = threading.Thread(target=lambda: (begin(e, [current]), ready.set()))
        thread.start()
        assert ready.wait(2)
        thread.join(2)
        with engine_context(tmp_path, window=128000, conversation_id="conv") as cold:
            # Rebinding retires old state; the hook runs after both objects bind.
            begin(e, [current], turn="cold-turn")
            cold.ingest([deepcopy(current)])
            assert_capture(cold, [expected()])


def test_t7_ingest_while_preanswer_hook_thread_is_running(hook_context, monkeypatch):
    ctx, view = hook_context, [message(message_uid="e" * 32)]
    ready, release = threading.Event(), threading.Event()
    def preanswer(*_):
        ready.set()
        release.wait(2)
    monkeypatch.setattr(ctx.module, "_pre_llm_context", preanswer)
    thread = threading.Thread(target=lambda: ctx.hooks["pre_llm_call"](**payload(view)))
    thread.start()
    assert ready.wait(2)
    try:
        ctx.engine.handle_tool_call("lcm_status", {}, messages=view)
        assert_capture(ctx.engine, [expected()])
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()


def test_interrupted_clone_insert_precedes_retirement(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        prefix = [message(0), dict(role="assistant", content="synthetic reply", timestamp=100.5)]
        e.ingest(prefix)
        current = message(2, message_uid="f" * 32)
        begin(e, [*prefix, current])
        view = [*prefix, current, dict(role="assistant", content="synthetic stopped reply")]
        e.on_turn_complete(deepcopy(view), interrupted=True, turn_id="turn-a")
        assert_capture(e, [(None, None), (None, None), expected(), (None, None)])
        assert capture_module().snapshot(e) is None


@pytest.mark.parametrize("boundary", ["complete", "end", "reset", "rebind", "shutdown"])
def test_retirement_all_lifecycle_boundaries(tmp_path, boundary):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        current = message()
        begin(e, [current])
        if boundary == "complete":
            e.on_turn_complete([current], turn_id="turn-a")
        elif boundary == "end":
            e.on_session_end("S", [current])
        elif boundary == "reset":
            e.on_session_reset()
        elif boundary == "rebind":
            e.on_session_start("next", platform="cli", context_length=128000)
        else:
            e.shutdown()
        assert capture_module().snapshot(e) is None


def test_fail_open_matching_and_creation_errors(tmp_path, monkeypatch):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        current = message()
        cap = capture_module()
        begin(e, [current])
        monkeypatch.setattr(cap, "matches", lambda *_a: (_ for _ in ()).throw(ValueError("synthetic")))
        e.ingest([current])
        assert_capture(e, [(None, None)])
        monkeypatch.setattr(e, "_message_replay_identity", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("synthetic")))
        assert begin(e, [message(1)], turn="error-turn") is None


def test_t10_no_sender_batch_omits_key_and_store_keyword(tmp_path, monkeypatch):
    with engine_context(tmp_path, window=128000, conversation_id="conv") as e:
        batches, calls = [], []
        append, stored = e._append_ingest_batch, e._store._append_protected_batch
        def observe_batch(batch):
            batches.append(deepcopy({k: v for k, v in batch.items() if k != "store"}))
            return append(batch)
        def observe_call(*args, **kwargs):
            calls.append(kwargs)
            return stored(*args, **kwargs)
        monkeypatch.setattr(e, "_append_ingest_batch", observe_batch)
        monkeypatch.setattr(e._store, "_append_protected_batch", observe_call)
        e.ingest([message()])
        assert batches and calls
        assert all("senders" not in value for value in batches + calls)
        assert_capture(e, [(None, None)])
        assert not any(name.startswith("_sender") for name in vars(e))
