"""#622: a slow plugin load must not leave the gateway silently without LCM-X.

(b) When Hermes ignores ``register_context_engine()`` (0.21.5 load deadline) or
another engine holds the slot, ``register()`` says so loudly, never prints the
"active" line, and leaves a cross-process record that ``lcm_status`` and
``/lcm doctor`` report while that process lives.

(a-core) A steady-state store open takes no write lock and does no O(size)
read, so a due FTS deep check or another writer cannot stall ``register()``.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.inactive_record import RECORD_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKFILL_STEP = "messages_ingested_at_backfill_v1"
ACTIVE_LINE = "LCM plugin loaded — lossless context management active"


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    for name in (
        "LCM_DATABASE_PATH",
        "LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS",
        "LCM_FTS_INTEGRITY_BACKGROUND",
        "LCM_TEMPORAL_ROLLUPS_ENABLED",
        "LCM_ENABLE_SLASH_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    db_bootstrap.join_background_integrity_scans(timeout=10)


# --- (b) fake host with Hermes 0.21.5 semantics -------------------------------


class _Manager:
    def __init__(self, existing=None):
        self._context_engine = existing


class _OtherEngine:
    name = "compressor-plus"


class _Ctx:
    """PluginContext shape: registrars become no-ops once the load is abandoned."""

    def __init__(self, manager):
        self._manager = manager
        self._load_abandoned = False
        self.offered = None
        self.hooks = {}

    def _abandon_load(self):
        self._load_abandoned = True

    def _ignored(self, name):
        if self._load_abandoned:
            logging.getLogger("hermes_cli.plugins").warning(
                "Plugin 'hermes-lcm-x' called %s() after its load timed out; ignored", name
            )
            return True
        return False

    def register_context_engine(self, engine):
        self.offered = engine
        if self._ignored("register_context_engine"):
            return None
        if self._manager._context_engine is not None:
            return None  # "Only one context engine plugin is allowed."
        self._manager._context_engine = engine
        return object()

    def register_hook(self, name, callback):
        if not self._ignored("register_hook"):
            self.hooks.setdefault(name, []).append(callback)

    def register_tool(self, **kwargs):
        self._ignored("register_tool")

    def register_skill(self, *args, **kwargs):
        self._ignored("register_skill")

    def register_command(self, *args, **kwargs):
        self._ignored("register_command")


class _BareCtx:
    """Unknown host shape: no ``_manager`` to inspect."""

    def __init__(self):
        self.offered = None

    def register_context_engine(self, engine):
        self.offered = engine


def _install_host(monkeypatch, home: Path, *, delay: float = 0.0):
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []
    config_module = types.ModuleType("hermes_cli.config")

    def _get_hermes_home():
        if delay:
            time.sleep(delay)  # the slow part of the plugin load
        return home

    config_module.get_hermes_home = _get_hermes_home
    config_module.load_config = lambda: {}
    hermes_cli.config = config_module
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_module)


def _load_plugin(module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, str(REPO_ROOT / "__init__.py"), submodule_search_locations=[str(REPO_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _register_under_deadline(module, ctx, deadline: float) -> None:
    """``run_with_load_deadline``: join the worker, abandon the ctx on overrun."""
    failure: list[BaseException] = []

    def _worker():
        try:
            module.register(ctx)
        except BaseException as exc:  # pragma: no cover - surfaced below
            failure.append(exc)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(deadline)
    if worker.is_alive():
        ctx._abandon_load()
        worker.join(60)
    assert not worker.is_alive()
    if failure:
        raise failure[0]


def _shutdown(ctx):
    if getattr(ctx, "offered", None) is not None:
        ctx.offered.shutdown()


def test_t1_abandoned_load_logs_one_error_and_writes_record(tmp_path, monkeypatch, caplog):
    home = tmp_path / "hermes-home"
    home.mkdir()
    _install_host(monkeypatch, home, delay=0.8)
    module = _load_plugin("hermes_lcm_622_t1")
    ctx = _Ctx(_Manager())
    caplog.set_level(logging.INFO)
    try:
        _register_under_deadline(module, ctx, deadline=0.3)
    finally:
        _shutdown(ctx)

    assert ctx._manager._context_engine is None
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1, [r.getMessage() for r in errors]
    message = errors[0].getMessage()
    assert "LCM-X is NOT active in this process" in message
    assert "plugins.load_timeout_seconds" in message
    assert "plugins.load_timeout_seconds: 60" in message
    elapsed = float(re.search(r"took (\d+(?:\.\d+)?) s", message).group(1))
    assert elapsed >= 0.8
    assert "LCM plugin loaded" not in caplog.text
    assert "Path B" not in caplog.text

    record = json.loads((home / RECORD_NAME.format(pid=os.getpid())).read_text(encoding="utf-8"))
    assert set(record) == {"pid", "process_start", "elapsed_s", "reason", "written_at"}
    assert record["pid"] == os.getpid()
    assert record["elapsed_s"] >= 0.8
    assert "plugins.load_timeout_seconds" in record["reason"]


def test_t2_fast_load_logs_active_line_with_load_time(tmp_path, monkeypatch, caplog):
    home = tmp_path / "hermes-home"
    home.mkdir()
    _install_host(monkeypatch, home)
    module = _load_plugin("hermes_lcm_622_t2")
    ctx = _Ctx(_Manager())
    caplog.set_level(logging.INFO)
    try:
        _register_under_deadline(module, ctx, deadline=30)
    finally:
        _shutdown(ctx)

    assert ctx._manager._context_engine is ctx.offered
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "NOT active" not in caplog.text
    assert not (home / RECORD_NAME.format(pid=os.getpid())).exists()
    active = [r.getMessage() for r in caplog.records if ACTIVE_LINE in r.getMessage()]
    assert len(active) == 1
    assert re.search(r"\(load \d+\.\d\d s\)$", active[0]), active[0]


def test_t3_other_engine_in_slot_logs_warning_not_error(tmp_path, monkeypatch, caplog):
    home = tmp_path / "hermes-home"
    home.mkdir()
    _install_host(monkeypatch, home)
    module = _load_plugin("hermes_lcm_622_t3")
    ctx = _Ctx(_Manager(existing=_OtherEngine()))
    caplog.set_level(logging.INFO)
    try:
        _register_under_deadline(module, ctx, deadline=30)
    finally:
        _shutdown(ctx)

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and "LCM-X is NOT" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "_OtherEngine" in warnings[0]
    assert "load_timeout_seconds" not in warnings[0]
    assert "LCM plugin loaded" not in caplog.text
    record = json.loads((home / RECORD_NAME.format(pid=os.getpid())).read_text(encoding="utf-8"))
    assert "_OtherEngine" in record["reason"]


_CHILD = textwrap.dedent(
    """
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("inactive_record_child", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_inactive_record(sys.argv[2], elapsed_s=12.5, reason="test reason")
    print("ready", flush=True)
    sys.stdin.read()
    """
)


def test_t4_record_reported_while_pid_alive_and_dropped_after(tmp_path):
    from hermes_lcm.command import handle_lcm_command
    from hermes_lcm import tools as lcm_tools

    home = tmp_path / "hermes-home"
    home.mkdir()
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    engine = LCMEngine(config=config, hermes_home=str(home))
    engine.on_session_start("s-622", platform="cli", conversation_id="c-622")
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(REPO_ROOT / "inactive_record.py"), str(home)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        expected = f"LCM-X was not active in process {child.pid} (test reason)"
        status = json.loads(lcm_tools.lcm_status({}, engine=engine))
        assert status["inactive_process"] == expected
        assert expected in handle_lcm_command("doctor", engine)
    finally:
        child.stdin.close()
        child.wait(timeout=30)

    try:
        status = json.loads(lcm_tools.lcm_status({}, engine=engine))
        assert "inactive_process" not in status
        assert "was not active in process" not in handle_lcm_command("doctor", engine)
        assert not (home / RECORD_NAME.format(pid=child.pid)).exists()
    finally:
        engine.shutdown()


def test_t5_unknown_host_shape_is_treated_as_active(tmp_path, monkeypatch, caplog):
    home = tmp_path / "hermes-home"
    home.mkdir()
    _install_host(monkeypatch, home)
    module = _load_plugin("hermes_lcm_622_t5")
    ctx = _BareCtx()
    caplog.set_level(logging.INFO)
    try:
        module.register(ctx)
    finally:
        _shutdown(ctx)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "LCM-X is NOT" in r.getMessage()]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not (home / RECORD_NAME.format(pid=os.getpid())).exists()
    active = [r.getMessage() for r in caplog.records if ACTIVE_LINE in r.getMessage()]
    assert len(active) == 1
    assert re.search(r"\(load \d+\.\d\d s\)$", active[0]), active[0]


# --- (a-core) the store open ----------------------------------------------------


def _engine(db: Path, **overrides) -> LCMEngine:
    config = LCMConfig(database_path=str(db), **overrides)
    return LCMEngine(config=config, hermes_home=str(db.parent / "home"))


def _seed(db: Path, **overrides) -> None:
    engine = _engine(db, **overrides)
    try:
        engine.on_session_start("s-622", platform="cli", conversation_id="c-622")
        engine.ingest([
            {"role": "user", "content": f"alpha bravo charlie message {i}"} for i in range(20)
        ])
    finally:
        engine.shutdown()
    db_bootstrap.join_background_integrity_scans(timeout=10)


def _markers(db: Path) -> dict[str, float]:
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT key, value FROM metadata WHERE key LIKE 'fts_integrity_checked_at:%'"
        ).fetchall()
    finally:
        conn.close()
    return {key.split(":", 1)[1]: float(value) for key, value in rows}


def _age_markers(db: Path, hours: float) -> float:
    aged = time.time() - hours * 3600.0
    conn = sqlite3.connect(str(db))
    try:
        for table in ("messages_fts", "nodes_fts"):
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (f"fts_integrity_checked_at:{table}", str(aged)),
            )
        conn.execute("DELETE FROM metadata WHERE key LIKE 'fts_integrity_scan_started_at:%'")
        conn.commit()
    finally:
        conn.close()
    return aged


def _hold_write_lock(db: Path, seconds: float) -> threading.Timer:
    holder = sqlite3.connect(str(db), timeout=10, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")

    def _release():
        holder.rollback()
        holder.close()

    timer = threading.Timer(seconds, _release)
    timer.start()
    return timer


def _timed_open(db: Path, **overrides) -> float:
    started = time.monotonic()
    engine = _engine(db, **overrides)
    elapsed = time.monotonic() - started
    engine.shutdown()
    return elapsed


def test_t6_due_deep_check_behind_a_writer_does_not_stall_the_open(tmp_path):
    db = tmp_path / "lcm.db"
    _seed(db)
    aged = _age_markers(db, hours=25)

    timer = _hold_write_lock(db, seconds=5)
    try:
        elapsed = _timed_open(db)
    finally:
        timer.join()
    db_bootstrap.join_background_integrity_scans(timeout=30)
    assert elapsed < 1.0, f"store open took {elapsed:.2f} s behind a held write lock"
    # Skipped, not stamped: the markers stay due for a later open.
    assert _markers(db) == {"messages_fts": aged, "nodes_fts": aged}

    for _ in range(3):
        _timed_open(db)
        db_bootstrap.join_background_integrity_scans(timeout=30)
        if all(value > aged for value in _markers(db).values()):
            break
    stamped = _markers(db)
    assert stamped["messages_fts"] > aged and stamped["nodes_fts"] > aged, stamped


def _traced_open(db: Path, monkeypatch) -> list[str]:
    statements: list[str] = []
    original_connect = sqlite3.connect

    def _connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _connect)
    try:
        _engine(db).shutdown()
    finally:
        monkeypatch.setattr(sqlite3, "connect", original_connect)
    return statements


def _full_scans_of_messages(db: Path, statements: list[str]) -> list[str]:
    conn = sqlite3.connect(str(db))
    scans = []
    try:
        for statement in statements:
            if not statement.lstrip().upper().startswith(("SELECT", "WITH")):
                continue
            try:
                plan = conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()
            except sqlite3.Error:
                continue
            if any(str(row[-1]).strip() == "SCAN messages" for row in plan):
                scans.append(statement)
    finally:
        conn.close()
    return scans


def test_t7_steady_state_open_does_not_scan_messages(tmp_path, monkeypatch):
    db = tmp_path / "lcm.db"
    _seed(db)
    _engine(db).shutdown()  # records the one-time backfill marker
    db_bootstrap.join_background_integrity_scans(timeout=10)

    statements = _traced_open(db, monkeypatch)
    assert statements
    assert _full_scans_of_messages(db, statements) == []


def test_t7_legacy_null_ingested_at_backfills_once(tmp_path, monkeypatch):
    db = tmp_path / "lcm.db"
    _seed(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("UPDATE messages SET ingested_at = NULL WHERE store_id % 2 = 0")
        conn.execute("DELETE FROM lcm_migration_state WHERE step_name = ?", (BACKFILL_STEP,))
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE ingested_at IS NULL").fetchone()[0] > 0
    finally:
        conn.close()

    _engine(db).shutdown()
    db_bootstrap.join_background_integrity_scans(timeout=10)
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE ingested_at IS NULL").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE ingested_at != timestamp AND store_id % 2 = 0"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = ?", (BACKFILL_STEP,)
        ).fetchone() is not None
    finally:
        conn.close()

    statements = _traced_open(db, monkeypatch)
    assert _full_scans_of_messages(db, statements) == []


def test_t8_rollups_on_open_does_not_wait_for_another_writer(tmp_path):
    db = tmp_path / "lcm.db"
    _seed(db, temporal_rollups_enabled=True)

    timer = _hold_write_lock(db, seconds=5)
    try:
        elapsed = _timed_open(db, temporal_rollups_enabled=True)
    finally:
        timer.join()
    assert elapsed < 1.0, f"rollups-on store open took {elapsed:.2f} s behind a held write lock"


# --- review round 2 (#691) -----------------------------------------------------


class _AbandonOnActiveLine(logging.Handler):
    """A log sink slow enough that Hermes' deadline expires while the active line is logged."""

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx

    def emit(self, record):
        if ACTIVE_LINE in record.getMessage():
            self.ctx._abandon_load()


def test_b1_abandonment_during_the_active_line_is_still_reported(tmp_path, monkeypatch, caplog):
    home = tmp_path / "hermes-home"
    home.mkdir()
    _install_host(monkeypatch, home)
    module_name = "hermes_lcm_622_b1"
    module = _load_plugin(module_name)
    ctx = _Ctx(_Manager())
    sink = _AbandonOnActiveLine(ctx)
    logging.getLogger(module_name).addHandler(sink)
    caplog.set_level(logging.INFO)
    try:
        module.register(ctx)
    finally:
        logging.getLogger(module_name).removeHandler(sink)
        _shutdown(ctx)

    assert ctx._load_abandoned is True
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1, errors
    assert "LCM-X is NOT active in this process" in errors[0]
    record = json.loads((home / RECORD_NAME.format(pid=os.getpid())).read_text(encoding="utf-8"))
    assert "plugins.load_timeout_seconds" in record["reason"]


def test_n3_null_ingested_at_falls_back_to_timestamp_in_reasoning(tmp_path):
    from hermes_lcm.reasoning import _ground_one
    from hermes_lcm.store import MessageStore

    store = MessageStore(tmp_path / "lcm.db")
    try:
        quote = "The deck has 12 slides."
        store_id = store.append("s-622", {"role": "user", "content": quote})
        store._conn.execute(
            "UPDATE messages SET ingested_at = NULL, observed_at = NULL WHERE store_id = ?",
            (store_id,),
        )
        store._conn.commit()
        row = store.get(store_id)
        assert row["ingested_at"] is None and row["observed_at"] is None
        assert row["timestamp"] is not None

        grounded, error = _ground_one(
            {"store_id": store_id, "span_start": 0, "span_end": len(quote), "quote": quote},
            messages=store,
            assertions=None,
            as_of=time.time() + 3600,
            session_dates=None,
        )
    finally:
        store.close()

    assert error is None
    assert grounded is not None and grounded.store_id == store_id


def test_n3_non_null_ingested_at_zero_is_kept_in_reasoning(tmp_path):
    from hermes_lcm.reasoning import _ground_one
    from hermes_lcm.store import MessageStore

    store = MessageStore(tmp_path / "lcm.db")
    try:
        quote = "The deck has 12 slides."
        store_id = store.append("s-622", {"role": "user", "content": quote})
        store._conn.execute(
            "UPDATE messages SET ingested_at = 0.0, observed_at = NULL, timestamp = 100.0 WHERE store_id = ?",
            (store_id,),
        )
        store._conn.commit()

        grounded, error = _ground_one(
            {"store_id": store_id, "span_start": 0, "span_end": len(quote), "quote": quote},
            messages=store,
            assertions=None,
            as_of=75.0,
            session_dates=None,
        )
    finally:
        store.close()

    # ingested_at 0.0 is a stored value, not NULL: it is the observation time, so the row is before as_of.
    assert error is None
    assert grounded is not None and grounded.store_id == store_id
