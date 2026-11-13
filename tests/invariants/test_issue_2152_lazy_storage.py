"""#2152: foreign-profile plugin preflight must not open or create storage."""

import importlib.util
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.retrieval_core import _acquire_vector_store
from hermes_lcm.vector_store import VectorStore


@pytest.fixture
def preflight(tmp_path, monkeypatch):
    import hermes_lcm

    spec = importlib.util.spec_from_file_location(
        "hermes_lcm", Path(__file__).resolve().parents[2] / "__init__.py",
    )
    spec.loader.exec_module(hermes_lcm)
    home = tmp_path / "profiles" / "foreign"
    host = ModuleType("hermes_cli")
    config = ModuleType("hermes_cli.config")
    config.get_hermes_home = lambda: home
    config.load_config = lambda: {}
    host.config = config
    monkeypatch.setitem(sys.modules, "hermes_cli", host)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launching-profile"))
    monkeypatch.setenv("LCM_ENABLE_SLASH_COMMAND", "1")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "0")
    connects, binds, engines = [], [], []
    connect, bind = sqlite3.connect, LCMEngine._bind_storage

    def connect_spy(path, *args, **kwargs):
        connects.append(str(path))
        return connect(path, *args, **kwargs)

    def bind_spy(engine, *args, **kwargs):
        binds.append(str(args[0]))
        return bind(engine, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect_spy)
    monkeypatch.setattr(LCMEngine, "_bind_storage", bind_spy)

    def register(*, inactive=False, abandoned=False, supports_unload=False, on_publish=None, on_unload=None):
        ctx = SimpleNamespace(
            _manager=SimpleNamespace(_context_engine=object() if inactive else None),
            _load_abandoned=abandoned,
            context_engine_tool_handlers_receive_messages=True,
            hooks={}, tools={}, commands={},
        )

        def take_slot(engine):
            engines.append(engine)
            if not inactive and not abandoned:
                ctx._manager._context_engine = engine
                if on_publish is not None:
                    on_publish(engine)

        ctx.register_context_engine = take_slot
        ctx.register_skill = lambda *args, **kwargs: None
        ctx.register_system_prompt_section = lambda *args: None
        ctx.register_hook = lambda name, handler: ctx.hooks.update({name: handler})
        ctx.register_tool = lambda **kwargs: ctx.tools.update({kwargs["name"]: kwargs["handler"]})
        ctx.register_command = lambda name, handler, **kwargs: ctx.commands.update({name: handler})
        if supports_unload:
            ctx.unload_callbacks = []
            ctx.on_unload = on_unload if on_unload is not None else ctx.unload_callbacks.append
        hermes_lcm.register(ctx)
        return engines[-1], ctx

    yield SimpleNamespace(home=home, connects=connects, binds=binds, register=register)
    for engine in engines:
        engine.shutdown()


@pytest.mark.parametrize("slot", ["active", "inactive", "abandoned"])
def test_register_foreign_home_opens_nothing(preflight, slot):
    engine, ctx = preflight.register(inactive=slot == "inactive", abandoned=slot == "abandoned")
    assert ctx.hooks and ctx.tools and ctx.commands
    assert (preflight.home.exists(), preflight.connects) == (False, [])
    clone = engine.clone_for_agent()
    clone.shutdown()
    assert (preflight.home.exists(), preflight.connects) == (False, [])


def test_first_ingest_after_register_opens_and_stores(preflight):
    engine, _ = preflight.register()
    assert preflight.connects == []
    engine.on_session_start("first", platform="cli")
    engine.ingest([{"role": "user", "content": "first real profile turn"}])
    rows = engine._store.get_session_messages("first")
    assert [row["content"] for row in rows] == ["first real profile turn"]
    assert (preflight.home / "lcm.db").exists()
    assert preflight.binds == [str(preflight.home / "lcm.db")]
    assert preflight.connects


def test_shutdown_before_use_opens_nothing(preflight):
    engine, _ = preflight.register()
    engine.shutdown()
    engine.shutdown()
    assert (preflight.home.exists(), preflight.connects) == (False, [])


def test_rejected_unload_callback_keeps_legacy_registration(preflight, caplog):
    from hermes_lcm import logger

    clones, order = [], []

    def reject(_callback):
        order.append("unload")
        raise TypeError("unsupported callback")

    def publish(engine):
        order.append("publish")
        clones.append(engine.clone_for_agent())

    with caplog.at_level(logging.INFO, logger=logger.name):
        prototype, ctx = preflight.register(supports_unload=True, on_unload=reject, on_publish=publish)
    clone, = clones
    try:
        assert order == ["unload", "publish"]
        assert ctx.hooks and ctx.tools and ctx.commands
        assert prototype._unload_group is None and clone._unload_group is None
        assert ctx.unload_callbacks == []
        assert "unload callback registration failed" in caplog.text
        clone.on_session_start("legacy", platform="cli")
        clone.ingest([{"role": "user", "content": "store normally"}])
        assert clone._store.get_session_count("legacy") == 1
    finally:
        clone.shutdown()


def test_deadline_worker_ignores_mock_unload_group():
    from hermes_lcm.engine_registry import EngineUnloadGroup
    from hermes_lcm.tools import _run_within_deadline

    engine = MagicMock(wraps=SimpleNamespace(_unload_group=None))
    engine._unload_group = MagicMock(spec=EngineUnloadGroup)
    slots = threading.BoundedSemaphore(1)
    for expected in (1, 2):
        assert _run_within_deadline(
            lambda: expected, remaining_s=2.0, name="test-unowned", worker_slots=slots, engine=engine,
        ) == expected


def test_vector_store_ignores_mock_unload_group(preflight):
    from hermes_lcm.engine_registry import EngineUnloadGroup
    from hermes_lcm.retrieval_core import _reset_vector_store_pool

    engine, _ = preflight.register()
    engine.on_session_start("legacy", platform="cli")
    wrapped = SimpleNamespace(
        _store=engine._store, _config=engine._config, _unload_group=MagicMock(spec=EngineUnloadGroup),
    )
    try:
        store, lease, transient = _acquire_vector_store(wrapped, vector_store_cls=VectorStore, scan_rows=None)
        with lease:
            assert not transient
            assert store.connection.execute("SELECT 1").fetchone()[0] == 1
        same_store, lease, _ = _acquire_vector_store(engine, vector_store_cls=VectorStore, scan_rows=None)
        with lease:
            assert same_store is store
    finally:
        _reset_vector_store_pool()


def test_plugin_unload_closes_all_owned_engines(preflight):
    eager_clones = []
    prototype, ctx = preflight.register(supports_unload=True, on_publish=lambda engine: eager_clones.append(engine.clone_for_agent()))
    other, other_ctx = preflight.register(supports_unload=True)
    owned = [prototype, *eager_clones, prototype.clone_for_agent()]
    connections = []
    try:
        # Only the last engine bound to an id is reachable by the session registry.
        for engine in owned:
            engine.on_session_start("shared", platform="cli")
            connections.append(engine._store.connection)
        owned[-1].ingest([{"role": "user", "content": "keep this message"}])
        other.on_session_start("other", platform="cli")
        pooled_connections = []
        for engine in (owned[-1], other):
            store, lease, _ = _acquire_vector_store(engine, vector_store_cls=VectorStore, scan_rows=None)
            with lease:
                pooled_connections.append(store.connection)
        assert pooled_connections[0] is not pooled_connections[1]
        connections.append(pooled_connections[0])
        assert len(ctx.unload_callbacks) == 1
        ctx.unload_callbacks[0]()

        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
        assert other._store.connection.execute("SELECT 1").fetchone()[0] == 1
        assert pooled_connections[1].execute("SELECT 1").fetchone()[0] == 1
        assert other._store.get_session_messages("shared")[0]["content"] == "keep this message"
        with pytest.raises(RuntimeError, match="unload"):
            prototype.clone_for_agent()
        with pytest.raises(RuntimeError, match="closed|unload"):
            owned[1].on_session_start("late", hermes_home=str(preflight.home / "late"))
        assert not (preflight.home / "late" / "lcm.db").exists()
    finally:
        for callback in other_ctx.unload_callbacks:
            callback()
        for engine in owned:
            engine.shutdown()


def _durable_messages(path, session):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        return conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY store_id", (session,),
        ).fetchall()


@pytest.mark.parametrize("already_stored", [True, False])
def test_late_session_end_after_unload_stores_suffix_once(preflight, already_stored):
    prototype, old_ctx = preflight.register(supports_unload=True)
    clone = prototype.clone_for_agent()
    clone.on_session_start("late-end", platform="cli")
    history = [{"role": "user", "content": "already stored"}]
    if already_stored:
        clone.ingest(history)
    old_ctx.unload_callbacks[0]()
    history.append({"role": "assistant", "content": "late answer"})
    clone.on_session_end("late-end", history)
    clone.on_session_end("late-end", history)
    path = preflight.home / "lcm.db"
    assert _durable_messages(path, "late-end") == [
        ("user", "already stored"), ("assistant", "late answer"),
    ]
    path.unlink()
    clone.on_session_end("late-end", history)
    assert not path.exists()


def test_retired_end_reconciles_the_current_writer(preflight, monkeypatch):
    prototype, old_ctx = preflight.register(supports_unload=True)
    retired = prototype.clone_for_agent()
    retired.on_session_start("same-lane", platform="cli", conversation_id="original")
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    retired.ingest(history)
    old_ctx.unload_callbacks[0]()
    current, ctx = preflight.register(supports_unload=True)
    plugin_api = ModuleType("hermes_cli.plugins")
    plugin_api.get_plugin_manager = lambda: SimpleNamespace(_context_engine=current)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugin_api)
    active = current.clone_for_agent()
    active.on_session_start("same-lane", platform="cli", conversation_id="original")
    history += [
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    active.ingest(history)
    history.append({"role": "user", "content": "late tail"})
    retired.on_session_end("same-lane", history)
    retired.on_session_end("same-lane", history)
    active.ingest(history)
    db = preflight.home / "lcm.db"
    try:
        assert _durable_messages(db, "same-lane") == [(row["role"], row["content"]) for row in history]
    finally:
        ctx.unload_callbacks[0]()
    db.unlink()


def test_retired_tail_replay_after_current_writer_saw_only_prefix(preflight):
    original, old_ctx = preflight.register(supports_unload=True)
    old = original.clone_for_agent()
    old.on_session_start("retired-prefix", platform="cli", conversation_id="prefix-lane")
    prefix = [
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "original answer"},
    ]
    old.ingest(prefix)
    old_ctx.unload_callbacks[0]()
    current, ctx = preflight.register(supports_unload=True)
    new = current.clone_for_agent()
    new.on_session_start("retired-prefix", platform="cli", conversation_id="prefix-lane")
    tail = prefix + [{"role": "user", "content": "late row from retired writer"}]
    try:
        new.ingest(prefix)
        old.on_session_end("retired-prefix", tail)
        new.ingest(prefix)
        new.ingest(tail)
        assert _durable_messages(preflight.home / "lcm.db", "retired-prefix") == [
            (row["role"], row["content"]) for row in tail
        ]
    finally:
        ctx.unload_callbacks[0]()


@pytest.mark.parametrize("passes_compressor", [True, False])
def test_reloaded_hook_keeps_ingesting_for_two_retained_clones(preflight, passes_compressor):
    prototype, old_ctx = preflight.register(supports_unload=True)
    clones = [prototype.clone_for_agent(), prototype.clone_for_agent()]
    views = {}
    for i, clone in enumerate(clones):
        sid = f"cached-{i}"
        clone.on_session_start(sid, platform="cli", conversation_id=f"conv-{i}")
        views[sid] = [
            {"role": "user", "content": sid},
            {"role": "assistant", "content": "original answer"},
        ]
        clone.ingest(views[sid])
    old_ctx.unload_callbacks[0]()
    _, current_ctx = preflight.register(supports_unload=True)
    db = preflight.home / "lcm.db"
    try:
        for turn in range(2):
            for i, clone in enumerate(clones):
                sid = f"cached-{i}"
                views[sid].extend([
                    {"role": "user", "content": f"question {turn}"},
                    {"role": "assistant", "content": f"answer {turn}"},
                ])
                extra = {"context_compressor": clone, "conversation_id": f"conv-{i}"} if passes_compressor else {}
                current_ctx.hooks["post_llm_call"](
                    session_id=sid, platform="cli", conversation_history=views[sid], **extra,
                )
                assert _durable_messages(db, sid) == [
                    (row["role"], row["content"]) for row in views[sid]
                ]
    finally:
        current_ctx.unload_callbacks[0]()
    db.unlink()


def test_reloaded_hook_replays_compacted_second_lane(preflight, monkeypatch):
    from hermes_lcm import escalation, tokens

    monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "2")
    monkeypatch.setenv("LCM_LEAF_CHUNK_TOKENS", "16")
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *args, **kw: "Earlier turns retained.")
    prototype, old_ctx = preflight.register(supports_unload=True)
    first, second = prototype.clone_for_agent(), prototype.clone_for_agent()
    first.on_session_start("first-cache", platform="cli", conversation_id="first-lane")
    second.on_session_start("second-cache", platform="cli", conversation_id="second-lane", context_length=200_000)
    first_view = [{"role": "user", "content": "First lane question"}, {"role": "assistant", "content": "First answer"}]
    first.ingest(first_view)
    second_view = [
        {"role": role, "content": f"{label} {sentence} " * 40}
        for label in ("one", "two") for role, sentence in (("user", "question"), ("assistant", "answer"))
    ]
    second.ingest(second_view)
    projected = second.compress(second_view, current_tokens=second.threshold_tokens + 1, force=True)
    assert second._last_compression_status == "compacted"
    assert projected[0]["content"].startswith(("[Recent Summary (", "[Session Arc Summary ("))
    old_ctx.unload_callbacks[0]()
    _, ctx = preflight.register(supports_unload=True)
    hook = ctx.hooks["post_llm_call"]
    fresh = [{"role": "user", "content": "new question"}, {"role": "assistant", "content": "new answer"}]
    try:
        hook(session_id="first-cache", platform="cli", conversation_history=first_view + fresh)
        hook(session_id="second-cache", platform="cli", conversation_history=projected + fresh)
        db = preflight.home / "lcm.db"
        with closing(sqlite3.connect(db)) as conn:
            rows = conn.execute(
                "SELECT role, content, conversation_id FROM messages WHERE session_id=? ORDER BY store_id",
                ("second-cache",),
            ).fetchall()
        assert rows == [(row["role"], row["content"], "second-lane") for row in second_view + fresh]
    finally:
        ctx.unload_callbacks[0]()


def test_reloaded_hook_does_not_move_a_lane_back_after_rollover(preflight, monkeypatch):
    old, old_ctx = preflight.register(supports_unload=True)
    live = old.clone_for_agent()
    retiring = old.clone_for_agent()
    live.on_session_start("active", platform="cli")
    retiring.on_session_start("before-rollover", platform="cli", conversation_id="moving-lane")
    prompt = [{"role": "user", "content": "old row"}]
    live.ingest(prompt)
    retiring.ingest(prompt)
    old_ctx.unload_callbacks[0]()
    current, ctx = preflight.register(supports_unload=True)
    callback = ctx.hooks["post_llm_call"]
    callback(session_id="active", platform="cli", conversation_history=prompt)
    spawn = current.clone_for_agent
    successor = spawn()

    def roll_before_scoped_writer():
        successor.on_session_start("after-rollover", platform="cli", conversation_id="moving-lane")
        successor.ingest([{"role": "user", "content": "new row"}])
        return spawn()

    monkeypatch.setattr(current, "clone_for_agent", roll_before_scoped_writer)
    db = preflight.home / "lcm.db"
    try:
        callback(session_id="before-rollover", platform="cli", conversation_history=prompt + [
            {"role": "assistant", "content": "stale answer"},
        ])
        state = successor._lifecycle.get_by_conversation("moving-lane")
        assert state.current_session_id == "after-rollover"
        assert _durable_messages(db, "before-rollover") == [("user", "old row")]
        assert _durable_messages(db, "after-rollover") == [("user", "new row")]
    finally:
        ctx.unload_callbacks[0]()


def test_unload_refuses_clone_that_finishes_construction_after_unload(preflight, monkeypatch):
    prototype, ctx = preflight.register(supports_unload=True)
    constructing, release, unloading = threading.Event(), threading.Event(), threading.Event()
    create = prototype._clone_for_agent

    def paused_clone():
        constructing.set()
        assert release.wait(5)
        return create()

    def unload():
        unloading.set()
        ctx.unload_callbacks[0]()

    monkeypatch.setattr(prototype, "_clone_for_agent", paused_clone)
    with ThreadPoolExecutor(max_workers=2) as executor:
        clone_future = executor.submit(prototype.clone_for_agent)
        try:
            assert constructing.wait(5)
            unload_future = executor.submit(unload)
            assert unloading.wait(5)
            unload_future.result(timeout=1)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="plugin unload"):
            clone_future.result(timeout=5)
    assert preflight.connects == []


def test_unload_drains_timed_out_sqlite_worker(preflight, monkeypatch, caplog):
    from hermes_lcm.tools import _run_within_deadline

    prototype, ctx = preflight.register(supports_unload=True)
    prototype.on_session_start("worker", platform="cli")
    prototype.on_session_end("worker", [])
    conversation = prototype._conversation_id
    ended = prototype._lifecycle.get_by_conversation(conversation)
    assert ended.last_finalized_session_id == "worker"
    active = prototype.clone_for_agent()
    active.on_session_start("active", platform="cli")
    first = {"role": "user", "content": "stored turn"}
    active.ingest([first])
    monkeypatch.setattr(
        active._dag, "reassign_session_nodes",
        lambda *_: pytest.fail("late carry-over reached SQLite"),
    )
    opened, release, closed = threading.Event(), threading.Event(), threading.Event()

    def query():
        conn = sqlite3.connect(preflight.home / "lcm.db")
        try:
            opened.set()
            assert release.wait(5)
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
            closed.set()

    with pytest.raises(TimeoutError):
        _run_within_deadline(query, remaining_s=0.05, name="test-unload-read", engine=prototype)
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            assert opened.wait(5)
            finish = executor.submit(ctx.unload_callbacks[0])
            # Wait for the closed gate without using sleep to guess that unload started.
            assert prototype._unload_group.closing.wait(5)
            assert not finish.done()
            prototype.on_session_reset()
            prototype.bind_session_state(session_id="late-reset")
            assert prototype._session_id == "worker"
            assert prototype._lifecycle.get_by_conversation(conversation) == ended
            assert prototype._lifecycle.get_by_session("late-reset") is None
            active.on_session_end("active", [first, {"role": "assistant", "content": "late tail"}])
            assert "could not store a proven tail" not in caplog.text
            assert active.should_compress_preflight([first, {"role": "assistant", "content": "late turn"}]) is False
            assert active._store.get_session_count("active") == 2
            assert active.carry_over_new_session_context("active", "late") == 0
            assert active._lifecycle.get_by_session("active").current_session_id == "active"
            assert active._store.get_session_count("active") == 2
        finally:
            release.set()
        finish.result(timeout=5)
    assert closed.is_set()


def test_unload_waits_for_direct_ingest(preflight, monkeypatch):
    engine, ctx = preflight.register(supports_unload=True)
    engine.on_session_start("direct", platform="cli")
    enter, release = threading.Event(), threading.Event()
    append = engine._store._append_protected_batch

    def blocked_append(*args, **kwargs):
        enter.set()
        assert release.wait(5)
        return append(*args, **kwargs)

    monkeypatch.setattr(engine._store, "_append_protected_batch", blocked_append)
    with ThreadPoolExecutor(max_workers=2) as executor:
        write = executor.submit(engine.ingest, [{"role": "user", "content": "direct row"}])
        try:
            assert enter.wait(5)
            finish = executor.submit(ctx.unload_callbacks[0])
            assert engine._unload_group.closing.wait(5)
            assert not finish.done()
        finally:
            release.set()
        write.result(timeout=5)
        finish.result(timeout=5)
    db = preflight.home / "lcm.db"
    assert _durable_messages(db, "direct") == [("user", "direct row")]
    db.unlink()


def test_unload_drains_owned_host_recovery(preflight):
    from hermes_lcm.maintenance_scheduler import _HOST_ROW_RECOVERY_SCHEDULER as scheduler

    engine, ctx = preflight.register(supports_unload=True)
    engine.on_session_start("recovery", platform="cli")
    db = preflight.home / "lcm.db"
    opened, release = threading.Event(), threading.Event()

    def recover():
        with closing(sqlite3.connect(db)) as conn:
            opened.set()
            assert release.wait(5)
            conn.execute("SELECT 1")

    owner = engine._unload_group
    assert owner.schedule_host_recovery(scheduler, ("database", "recovery"), recover)
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            assert opened.wait(5)
            finish = executor.submit(ctx.unload_callbacks[0])
            assert owner.closing.wait(5)
            assert not finish.done()
            assert not owner.schedule_host_recovery(scheduler, ("database", "late"), recover)
        finally:
            release.set()
        finish.result(timeout=5)
    db.unlink()


def test_unload_waits_for_admitted_compaction_preflight(preflight, monkeypatch):
    engine, ctx = preflight.register(supports_unload=True)
    engine.on_session_start("preflight", platform="cli")
    entered, release, shutdown_checked = threading.Event(), threading.Event(), threading.Event()
    waiting_for_preflight = threading.Event()
    original = engine._should_compress_preflight_impl
    wait_for = engine._compaction_gate.wait_for

    def held_preflight(messages):
        entered.set()
        assert release.wait(5)
        return original(messages)

    def check_drain(predicate, timeout=None):
        if not predicate():
            waiting_for_preflight.set()
        shutdown_checked.set()
        return wait_for(predicate, timeout)

    monkeypatch.setattr(engine, "_should_compress_preflight_impl", held_preflight)
    monkeypatch.setattr(engine._compaction_gate, "wait_for", check_drain)
    with ThreadPoolExecutor(max_workers=2) as executor:
        running = executor.submit(engine.should_compress_preflight, [{"role": "user", "content": "admitted"}])
        try:
            assert entered.wait(5)
            unload = executor.submit(ctx.unload_callbacks[0])
            assert shutdown_checked.wait(5)
            assert waiting_for_preflight.is_set()
            assert engine._store.connection.execute("SELECT 1").fetchone()[0] == 1
        finally:
            release.set()
        running.result(timeout=5)
        unload.result(timeout=5)
    assert engine._stable_use_closed


def test_unload_capable_host_does_not_serialize_native_tool_and_session_end(preflight, monkeypatch):
    engine, _ = preflight.register(supports_unload=True)
    engine.on_session_start("parallel", platform="cli")
    entered, release = threading.Event(), threading.Event()
    original = engine._handle_tool_call

    def held_tool(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_handle_tool_call", held_tool)
    with ThreadPoolExecutor(max_workers=2) as executor:
        tool = executor.submit(engine.handle_tool_call, "lcm_status", {})
        try:
            assert entered.wait(5)
            end = executor.submit(engine.on_session_end, "parallel", [])
            end.result(timeout=2)
            assert not tool.done()
        finally:
            release.set()
        assert "error" not in json.loads(tool.result(timeout=5))


def test_timed_out_unload_closes_storage_after_native_tool_finishes(preflight, monkeypatch):
    engine, ctx = preflight.register(supports_unload=True)
    engine.on_session_start("tool", platform="cli")
    connection = engine._store.connection
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = engine._handle_tool_call
    shutdown_storage = engine._shutdown_unlocked

    def held_tool(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    def close_storage(**kwargs):
        shutdown_storage(**kwargs)
        closed.set()

    monkeypatch.setenv("LCM_SHUTDOWN_COMPACTION_WAIT_S", "0.05")
    monkeypatch.setattr(engine, "_handle_tool_call", held_tool)
    monkeypatch.setattr(engine, "_shutdown_unlocked", close_storage)
    with ThreadPoolExecutor(max_workers=1) as executor:
        tool = executor.submit(engine.handle_tool_call, "lcm_status", {})
        try:
            assert entered.wait(5)
            ctx.unload_callbacks[0]()
            assert not closed.is_set()
            assert connection.execute("SELECT 1").fetchone()[0] == 1
        finally:
            release.set()
        assert "error" not in json.loads(tool.result(timeout=5))
    assert closed.wait(5)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_unload_retries_transient_store_close_failures(preflight, monkeypatch):
    engine, ctx = preflight.register(supports_unload=True)
    engine.on_session_start("store", platform="cli")
    connection = engine._store.connection
    closed = threading.Event()
    original_close = engine._store.close
    attempts = 0

    def close():
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise sqlite3.OperationalError("transient close failure")
        original_close()
        closed.set()

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "close", close)
        ctx.unload_callbacks[0]()
        assert closed.wait(5)
    assert attempts >= 3
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


@pytest.mark.parametrize("close_fails", [False, True])
def test_unload_retries_a_busy_tail_in_the_previous_profile(preflight, monkeypatch, close_fails):
    from hermes_lcm import maintenance_scheduler

    engine, ctx = preflight.register(supports_unload=True)
    engine.on_session_start("ended", conversation_id="ended-conversation", platform="cli")
    history = [
        {"role": "user", "content": "first", "timestamp": 1760000001.0},
        {"role": "assistant", "content": "tail", "timestamp": 1760000002.0},
    ]
    engine.ingest(history[:1])
    old_store = engine._store
    connection = old_store.connection
    retry_exhausted = threading.Event()
    close = old_store.close
    close_attempts = 0

    def close_old_store():
        nonlocal close_attempts
        close_attempts += 1
        if close_fails and close_attempts == 1:
            raise sqlite3.OperationalError("transient detached close failure")
        close()

    with closing(sqlite3.connect(old_store.db_path, isolation_level=None)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        try:
            engine.on_session_end("ended", history)
            engine.on_session_start("new", platform="cli", hermes_home=str(preflight.home / "new"))
            assert engine._store is not old_store
            assert len(engine._pending_session_tails) == 1
            monkeypatch.setattr(old_store, "close", close_old_store)
            monkeypatch.setenv("LCM_SHUTDOWN_COMPACTION_WAIT_S", "0.05")
            monkeypatch.setattr(maintenance_scheduler, "_SESSION_END_SHUTDOWN_TIMEOUT_MS", 100)
            retry = engine._retry_pending_session_tails

            def observe_retry(**kwargs):
                result = retry(**kwargs)
                if threading.current_thread().name == "lcm-deferred-unload":
                    retry_exhausted.set()
                return result

            monkeypatch.setattr(engine, "_retry_pending_session_tails", observe_retry)
            ctx.unload_callbacks[0]()
            assert retry_exhausted.wait(5), "deferred retry did not run while the old store was busy"
            assert not engine._stable_use_closed
        finally:
            writer.rollback()
    deadline = time.monotonic() + 5
    while not engine._stable_use_closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert engine._stable_use_closed
    assert close_attempts == 1 + close_fails
    assert engine not in maintenance_scheduler._PENDING_TAIL_ENGINES
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with closing(sqlite3.connect(old_store.db_path)) as reader:
        assert reader.execute("SELECT content FROM messages ORDER BY store_id").fetchall() == [("first",), ("tail",)]


def test_deferred_unload_progresses_past_a_terminal_tail(preflight, monkeypatch):
    import hermes_lcm.engine as engine_module

    engine, _ = preflight.register()
    engine.on_session_start("old", platform="cli")
    store = engine._store
    terminal = {"kind": "cold", "store": store, "view": [], "session_id": "terminal"}
    ready = {"kind": "cold", "store": store, "view": [], "session_id": "ready"}
    engine._pending_session_tails[:] = [terminal, ready]
    now = [0.0]
    visited = []

    def retry(tail, **kwargs):
        visited.append(tail["session_id"])
        if tail is terminal:
            now[0] += 6.0
            return False
        return True

    with monkeypatch.context() as patch:
        patch.setattr(engine_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
        patch.setattr(engine, "_retry_cold_tail", retry)
        terminal_tails = set()
        assert engine._retry_pending_session_tails(deadline=5.0, terminal_tails=terminal_tails)
        assert not engine._retry_pending_session_tails(deadline=11.0, terminal_tails=terminal_tails)
    assert visited == ["terminal", "ready"]
    assert engine._pending_session_tails == [terminal]
    engine._pending_session_tails.clear()


def test_concurrent_first_use_binds_once(preflight):
    engine, _ = preflight.register()
    assert preflight.binds == []
    start = threading.Barrier(3)

    def first_use(attribute):
        start.wait(timeout=5)
        return getattr(engine, attribute)

    with ThreadPoolExecutor(max_workers=2) as pool:
        store = pool.submit(first_use, "_store")
        dag = pool.submit(first_use, "_dag")
        start.wait(timeout=5)
        assert store.result(timeout=10) is engine._store
        assert dag.result(timeout=10) is engine._dag
    assert preflight.binds == [str(preflight.home / "lcm.db")]


@pytest.mark.parametrize("explicit", [False, True])
def test_rebind_before_use_opens_only_new_home(tmp_path, monkeypatch, preflight, explicit):
    db = tmp_path / "explicit.db"
    if explicit:
        monkeypatch.setenv("LCM_DATABASE_PATH", str(db))
    engine, _ = preflight.register()
    new_home = tmp_path / "served"
    assert engine._rebind_storage_for_home(str(new_home))
    assert not engine._rebind_storage_for_home(str(new_home))
    assert (preflight.home.exists(), new_home.exists(), preflight.connects) == (False, False, [])
    engine.on_session_start("new", hermes_home=str(new_home), platform="cli")
    engine.ingest([{"role": "user", "content": "new profile"}])
    target = db if explicit else new_home / "lcm.db"
    assert engine._store.db_path == target
    assert str(engine._store._hermes_home) == str(new_home)
    assert not preflight.home.exists()
    assert preflight.binds == [str(target)]


@pytest.mark.parametrize("entry", ["status", "doctor", "tool", "doctor_tool"])
def test_diagnostics_before_session_open_on_demand(preflight, entry):
    engine, ctx = preflight.register(inactive=True)
    assert preflight.connects == []
    if entry in {"tool", "doctor_tool"}:
        result = ctx.tools["lcm_status" if entry == "tool" else "lcm_doctor"]({})
    else:
        result = ctx.commands["lcm"](entry)
    assert result
    if entry != "status":  # Slash status has never rendered the inactive-process notice.
        assert "not active" in result
    assert preflight.binds == [str(preflight.home / "lcm.db")]
    from hermes_lcm.inactive_record import inactive_record_notice
    assert inactive_record_notice(str(preflight.home))


@pytest.mark.parametrize("attribute", ["_assertions", "_entities", "_query_views", "_adaptive_retrieval", "_assertion_extractor"])
def test_optional_helper_first_access_opens_once(preflight, attribute):
    engine, _ = preflight.register()
    assert preflight.binds == []
    getattr(engine, attribute)
    assert preflight.binds == [str(preflight.home / "lcm.db")]


def test_constructor_with_optional_stores_is_lazy(tmp_path, preflight):
    engine = LCMEngine(config=LCMConfig(
        assertions_enabled=True, entities_enabled=True, query_views_enabled=True,
        adaptive_retrieval_enabled=True, assertion_extraction_enabled=True,
    ), hermes_home=str(tmp_path / "optional"))
    try:
        assert preflight.connects == []
        assert engine._assertion_extractor is not None
        assert engine._entities is not None
        assert engine._adaptive_retrieval is not None
        assert len(preflight.binds) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("slot", ["inactive", "abandoned"])
def test_foreign_home_inactive_record_deferred_until_use(preflight, slot):
    from hermes_lcm.inactive_record import RECORD_DIR, RECORD_NAME

    engine, _ = preflight.register(inactive=slot == "inactive", abandoned=slot == "abandoned")
    record = preflight.home.joinpath(*RECORD_DIR) / RECORD_NAME.format(pid=os.getpid())
    assert not record.exists()
    assert engine._inactive_load_record["reason"]
    assert not preflight.home.exists()
    engine._ensure_storage()
    assert json.loads(record.read_text())["reason"]


@pytest.mark.parametrize("tool", ["lcm_status", "lcm_doctor"])
def test_first_diagnostic_folds_in_memory_record_without_publishing(preflight, monkeypatch, tool):
    from hermes_lcm import tools

    engine, ctx = preflight.register(inactive=True)
    read_notice = tools.inactive_record_notice
    notices = []

    def read_before_bind(*args, **kwargs):
        assert not engine._storage_ready
        assert preflight.connects == []
        notice = read_notice(*args, **kwargs)
        notices.append(notice)
        return notice

    monkeypatch.setattr(tools, "inactive_record_notice", read_before_bind)
    result = json.loads(ctx.tools[tool]({}))
    expected = f"LCM-X was not active in process {os.getpid()}"
    if tool == "lcm_status":
        assert expected in result["inactive_process"]
    else:
        check = next(row for row in result["checks"] if row["check"] == "inactive_process")
        assert check["status"] == "warn"
        assert expected in check["detail"]
    assert len(notices) == 1
    assert expected in notices[0]


def test_bind_failure_leaves_storage_retryable(preflight, monkeypatch):
    engine, _ = preflight.register()
    bind = engine._bind_storage
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(args)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("injected first-bind failure")
        return bind(*args, **kwargs)

    monkeypatch.setattr(engine, "_bind_storage", fail_once)
    with pytest.raises(sqlite3.OperationalError, match="injected first-bind failure"):
        _ = engine._store
    assert engine._storage_ready is False
    assert engine._storage_binding is False
    assert preflight.connects == []
    assert engine._store.db_path == preflight.home / "lcm.db"
    assert engine._storage_ready is True
    assert len(attempts) == 2
    assert len(preflight.binds) == 1


def test_rebind_open_failure_clears_old_profile_before_retry(preflight, tmp_path, monkeypatch):
    engine, _ = preflight.register()
    engine.on_session_start("old-session", platform="cli", conversation_id="old-conversation")
    engine.ingest([{"role": "user", "content": "old profile row"}])
    old_db = engine._store.db_path
    engine._identity_anchor_recent = ["old-profile-cache"]
    new_home = tmp_path / "rebound"
    bind = engine._bind_storage

    def fail_open(*args, **kwargs):
        raise sqlite3.OperationalError("injected rebind SQLITE_BUSY")

    monkeypatch.setattr(engine, "_bind_storage", fail_open)
    with pytest.raises(sqlite3.OperationalError, match="injected rebind SQLITE_BUSY"):
        engine._rebind_storage_for_home(str(new_home))
    assert engine._hermes_home == str(new_home)
    assert engine._db_path == new_home / "lcm.db"
    assert not engine._storage_ready
    assert not engine._storage_binding
    assert engine._session_id == engine._foreground_session_id == ""
    assert engine._conversation_id == engine._foreground_conversation_id == ""
    assert engine._identity_anchor_recent == []
    assert not new_home.exists()

    monkeypatch.setattr(engine, "_bind_storage", bind)
    assert engine._store.db_path == new_home / "lcm.db"
    assert engine._store.get_context_window_tokens("old-session") == 0
    assert engine._store.get_session_messages("old-session") == []
    engine.on_session_start("new-session", platform="cli", hermes_home=str(new_home))
    engine.ingest([{"role": "user", "content": "new profile row"}])
    assert [r["content"] for r in engine._store.get_session_messages("new-session")] == ["new profile row"]
    with sqlite3.connect(old_db) as conn:
        assert conn.execute("SELECT content FROM messages WHERE session_id = ?", ("old-session",)).fetchall() == [
            ("old profile row",),
        ]


def test_first_use_failure_warns_once_per_binding_through_post_llm_hook(preflight, monkeypatch, caplog, tmp_path):
    engine, ctx = preflight.register()
    assert preflight.connects == []

    def fail_open(*args, **kwargs):
        raise sqlite3.OperationalError("injected first-use SQLITE_BUSY")

    monkeypatch.setattr(engine, "_bind_storage", fail_open)
    hook = ctx.hooks["post_llm_call"]

    def turn():
        hook(context_compressor=engine, session_id="first-use", platform="cli",
             conversation_history=[{"role": "user", "content": "real turn"}])

    def warnings():
        return [r for r in caplog.records if r.name == "hermes_lcm.engine"
                and "LCM storage open failed on first use" in r.getMessage()]

    with caplog.at_level(logging.WARNING):
        turn()
        turn()
        assert len(warnings()) == 1
        assert warnings()[0].levelno == logging.WARNING
        assert "injected first-use SQLITE_BUSY" in warnings()[0].getMessage()
        assert not engine._storage_ready
        assert engine._rebind_storage_for_home(str(tmp_path / "next-binding"))
        turn()
        turn()
        assert len(warnings()) == 2
    assert preflight.connects == []


def test_state_db_path_before_use_does_not_open_storage(preflight, monkeypatch):
    engine, _ = preflight.register()
    engine._hermes_home = ""
    monkeypatch.setenv("LCM_HERMES_BASE_DIR", str(preflight.home.parent))
    assert engine._state_db_path() == preflight.home / "state.db"
    assert preflight.connects == []
    assert not preflight.home.exists()


@pytest.mark.parametrize("append_fails", [False, True], ids=["tail-drains", "tail-stays-locked"])
def test_shutdown_after_failed_rebind_with_pending_tail_never_reopens(preflight, tmp_path, monkeypatch, caplog, append_fails):
    """A pending tail of the old profile plus a failed rebind: shutdown must finish its cleanup without opening the new home."""
    engine, _ = preflight.register()
    engine.on_session_start("old-session", platform="cli", conversation_id="old-conversation")
    engine.ingest([{"role": "user", "content": "old profile row"}])
    old_store = engine._store
    engine._pending_session_tails.append({"store": old_store, "session_id": "old-session",
                                          "messages": [{"role": "user", "content": "tail row"}]})
    attempts = []

    def fail_open(*args, **kwargs):
        attempts.append(args)
        raise sqlite3.OperationalError("injected rebind failure")

    monkeypatch.setattr(engine, "_bind_storage", fail_open)
    with pytest.raises(sqlite3.OperationalError):
        engine._rebind_storage_for_home(str(tmp_path / "rebound"))
    assert len(attempts) == 1 and not engine._storage_ready

    def append(tail):
        if append_fails:
            raise sqlite3.OperationalError("database is locked")
        return []

    monkeypatch.setattr(engine, "_match_pending_tail", lambda tail: (0, 0, None, "no_ledger", "end", 0))
    monkeypatch.setattr(engine, "_append_ingest_batch", append)
    caplog.set_level("WARNING")
    engine.shutdown()
    assert len(attempts) == 1  # shutdown never retried the failed open
    assert engine._stable_use_closed is True
    assert old_store._conn is None  # the old profile's store was still closed
    unflushed = [r for r in caplog.records if "unflushed=1" in r.getMessage()]
    assert bool(unflushed) is append_fails

