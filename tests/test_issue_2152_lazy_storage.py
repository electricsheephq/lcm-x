"""#2152: register() must not open or create storage in a foreign process.

Storage opens lazily on first real use (session start, ingest, tool call, command).
A process that never serves the profile never touches its SQLite store or
inactive-slot record.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
from pathlib import Path
import sys
import types


from hermes_lcm.command_doctor import _doctor_text
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_plugin(name: str = "hermes_lcm_2152"):
    spec = importlib.util.spec_from_file_location(
        name, str(REPO_ROOT / "__init__.py"), submodule_search_locations=[str(REPO_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Manager:
    def __init__(self, existing=None):
        self._context_engine = existing


class _Ctx:
    def __init__(self, manager=None):
        self._manager = manager or _Manager()
        self._load_abandoned = False
        self.offered = None
        self.hooks = {}

    def register_context_engine(self, engine):
        self.offered = engine
        if self._manager._context_engine is None:
            self._manager._context_engine = engine


def _install_hermes_config(monkeypatch, home: Path):
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []
    config_module = types.ModuleType("hermes_cli.config")
    config_module.get_hermes_home = lambda: home
    config_module.load_config = lambda: {}
    hermes_cli.config = config_module
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_module)


def test_register_under_foreign_home_creates_and_opens_nothing(tmp_path, monkeypatch):
    """register() under a foreign profile home creates no database and no inactive record."""
    process_home = tmp_path / "default"
    process_home.mkdir()
    foreign_home = tmp_path / "foreign"
    foreign_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(process_home))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    _install_hermes_config(monkeypatch, foreign_home)

    module = _load_plugin("hermes_lcm_2152_foreign")
    ctx = _Ctx(_Manager(existing=object()))  # Foreign process slot held by another engine
    module.register(ctx)

    assert ctx.offered is not None
    assert getattr(ctx.offered, "_storage_bound", False) is False

    # Foreign home must be completely untouched
    assert not (foreign_home / "lcm.db").exists()
    assert not (foreign_home / "plugin-data").exists()
    assert list(foreign_home.iterdir()) == []

    # Clean shutdown of unused engine also touches nothing
    ctx.offered.shutdown()
    assert list(foreign_home.iterdir()) == []


def test_first_use_opens_once(tmp_path, monkeypatch):
    """Storage binds lazily on first use, not in __init__."""
    home = tmp_path / "profile_home"
    home.mkdir()
    db_file = home / "lcm.db"

    config = LCMConfig(database_path=str(db_file), embeddings_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(home))

    try:
        assert engine._storage_bound is False
        assert not db_file.exists()

        # First use: on_session_start binds storage
        engine.on_session_start("session_1")
        assert engine._storage_bound is True
        assert db_file.exists()

        # Subsequent store and DAG access reuses the bound storage
        store_conn = engine._store.connection
        dag_conn = engine._dag.connection
        assert store_conn is not None
        assert dag_conn is not None
    finally:
        engine.shutdown()


def test_concurrent_first_use_opens_once(tmp_path):
    """Concurrent first uses bind storage exactly once under the lock."""
    home = tmp_path / "concurrent_home"
    home.mkdir()
    db_file = home / "lcm.db"

    config = LCMConfig(database_path=str(db_file), embeddings_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(home))

    bind_calls = 0
    orig_bind = engine._bind_storage

    def tracking_bind(*args, **kwargs):
        nonlocal bind_calls
        bind_calls += 1
        return orig_bind(*args, **kwargs)

    engine._bind_storage = tracking_bind

    try:
        assert engine._storage_bound is False

        def worker(idx):
            if idx % 2 == 0:
                return engine._store.connection
            else:
                return engine._dag.connection

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(worker, i) for i in range(32)]
            results = [f.result() for f in futures]

        assert len(results) == 32
        assert engine._storage_bound is True
        assert bind_calls == 1
    finally:
        engine.shutdown()


def test_doctor_and_status_on_never_used_engine(tmp_path):
    """/lcm doctor and get_status() on a never-used engine still work."""
    home1 = tmp_path / "engine1"
    home1.mkdir()
    config1 = LCMConfig(database_path=str(home1 / "lcm.db"), embeddings_enabled=False)
    engine1 = LCMEngine(config=config1, hermes_home=str(home1))

    try:
        assert engine1._storage_bound is False
        status = engine1.get_status()
        assert isinstance(status, dict)
        assert status.get("engine") == "lcm-x"
        assert "total_compactions" in status
        assert engine1._storage_bound is True
    finally:
        engine1.shutdown()

    home2 = tmp_path / "engine2"
    home2.mkdir()
    config2 = LCMConfig(database_path=str(home2 / "lcm.db"), embeddings_enabled=False)
    engine2 = LCMEngine(config=config2, hermes_home=str(home2))

    try:
        assert engine2._storage_bound is False
        doctor_output = _doctor_text(engine2)
        assert "LCM doctor" in doctor_output
        assert engine2._storage_bound is True
    finally:
        engine2.shutdown()


def test_clone_for_agent_inherits_lazy_storage(tmp_path):
    """clone_for_agent() produces an engine that is also lazily bound."""
    home = tmp_path / "clone_home"
    home.mkdir()
    db_file = home / "lcm.db"

    config = LCMConfig(database_path=str(db_file), embeddings_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(home))

    try:
        assert engine._storage_bound is False
        clone = engine.clone_for_agent()
        assert clone._storage_bound is False
        assert not db_file.exists()

        clone.on_session_start("clone_session")
        assert clone._storage_bound is True
        assert db_file.exists()
        clone.shutdown()
    finally:
        engine.shutdown()


def test_rebind_before_first_use_touches_only_rebound_home(tmp_path):
    """Rebinding hermes_home before first use opens only the rebound profile's store."""
    home_initial = tmp_path / "initial"
    home_initial.mkdir()
    home_rebound = tmp_path / "rebound"
    home_rebound.mkdir()

    config = LCMConfig(embeddings_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(home_initial))

    try:
        assert engine._storage_bound is False
        engine.on_session_start("s1", hermes_home=str(home_rebound))
        assert engine._storage_bound is True

        assert not (home_initial / "lcm.db").exists()
        assert (home_rebound / "lcm.db").exists()
    finally:
        engine.shutdown()


def test_unused_engine_shutdown_creates_no_database(tmp_path):
    """Shutting down an engine that was never used does not create or open a database."""
    home = tmp_path / "unused"
    home.mkdir()
    db_file = home / "lcm.db"

    config = LCMConfig(database_path=str(db_file), embeddings_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(home))
    assert not db_file.exists()
    engine.shutdown()
    assert not db_file.exists()


def test_sender_capture_on_unbound_engine(tmp_path):
    """#2152/#2161: sender_capture._key and retire work on an unbound engine without raising NameError."""
    from hermes_lcm import sender_capture

    home = tmp_path / "sender_unbound"
    home.mkdir()
    config = LCMConfig(database_path=str(home / "lcm.db"), embeddings_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(home))

    try:
        assert engine._storage_bound is False
        # _key must resolve path cleanly using Path without NameError
        db_key, sess_key = sender_capture._key(engine)
        assert db_key == str((home / "lcm.db").resolve())
        assert sess_key == ""

        # retire and snapshot must succeed without error or force-binding storage
        sender_capture.retire(engine)
        assert sender_capture.snapshot(engine) is None
        assert engine._storage_bound is False
        assert not (home / "lcm.db").exists()
    finally:
        engine.shutdown()
