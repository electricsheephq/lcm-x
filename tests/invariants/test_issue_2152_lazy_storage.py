"""#2152: foreign-profile plugin preflight must not open or create storage."""

import importlib.util
import json
import logging
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


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

    def register(*, inactive=False, abandoned=False):
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

        ctx.register_context_engine = take_slot
        ctx.register_skill = lambda *args, **kwargs: None
        ctx.register_system_prompt_section = lambda *args: None
        ctx.register_hook = lambda name, handler: ctx.hooks.update({name: handler})
        ctx.register_tool = lambda **kwargs: ctx.tools.update({kwargs["name"]: kwargs["handler"]})
        ctx.register_command = lambda name, handler, **kwargs: ctx.commands.update({name: handler})
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

    monkeypatch.setattr(engine, "_trim_replayed_pending_tail", lambda tail: None)
    monkeypatch.setattr(engine, "_append_ingest_batch", append)
    caplog.set_level("WARNING")
    engine.shutdown()
    assert len(attempts) == 1  # shutdown never retried the failed open
    assert engine._stable_use_closed is True
    assert old_store._conn is None  # the old profile's store was still closed
    unflushed = [r for r in caplog.records if "unflushed=1" in r.getMessage()]
    assert bool(unflushed) is append_fails

