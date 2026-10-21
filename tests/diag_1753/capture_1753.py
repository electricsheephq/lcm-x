"""Opt-in structural tracing; no SQL values, messages or exception text."""

from contextlib import closing
from functools import wraps
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import threading
import time

import pytest


STORES = (
    ("store", "MessageStore"),
    ("dag", "SummaryDAG"),
    ("lifecycle_state", "LifecycleStateStore"),
    ("assertion_store", "AssertionStore"),
    ("query_view_store", "QueryViewStore"),
    ("rollup_store", "RollupStore"),
    ("vector_store", "VectorStore"),
)
SQL_KINDS = {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "SELECT",
             "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "PRAGMA", "WITH"}


class Capture:
    def __init__(self):
        self.events = []
        self.connections = {}
        self.lock = threading.RLock()
        self.reports = []

    def states(self):
        result = []
        for connection, owners in list(self.connections.items()):
            try:
                transaction = connection.in_transaction
            except sqlite3.Error:
                transaction = None
            result.append({"conn": id(connection), "owners": sorted(owners),
                           "in_transaction": transaction, "closed": transaction is None})
        return result

    def record(self, kind, **fields):
        with self.lock:
            self.events.append({"kind": kind, "time": time.monotonic_ns(),
                                "thread": threading.current_thread().name,
                                "thread_id": threading.get_ident(),
                                "connections": self.states(), **fields})

    def register(self, owner):
        connection = getattr(owner, "_conn", None)
        if not isinstance(connection, sqlite3.Connection):
            return
        label = f"{type(owner).__name__}:{id(owner)}"
        with self.lock:
            fresh = connection not in self.connections
            self.connections.setdefault(connection, set()).add(label)
        if fresh:
            def trace(sql):
                # sqlite supplies expanded values: retain only a whitelisted verb.
                words = sql.lstrip().split(None, 1)
                verb = words[0].upper() if words else ""
                self.record("sql", conn=id(connection),
                            statement=verb if verb in SQL_KINDS else "OTHER")
            connection.set_trace_callback(trace)
        self.record("connection_registered", owner=label, conn=id(connection))

    def rows(self, store):
        try:
            return {"ids": [row[0] for row in store.connection.execute(
                "SELECT store_id FROM messages ORDER BY store_id").fetchall()]}
        except sqlite3.Error as exc:
            return {"error_type": type(exc).__name__,
                    "sqlite_code": getattr(exc, "sqlite_errorcode", None)}

    def install(self, patch):
        def wrap(owner, name, label, append=False, constructor=False):
            original = getattr(owner, name)

            @wraps(original)
            def traced(instance, *args, **kwargs):
                if not constructor:
                    self.register(instance)
                self.record(label + "_enter", owner_id=id(instance))
                try:
                    result = original(instance, *args, **kwargs)
                except BaseException as exc:
                    self.record(label + "_error", owner_id=id(instance),
                                error_type=type(exc).__name__,
                                sqlite_code=getattr(exc, "sqlite_errorcode", None))
                    raise
                if constructor:
                    self.register(instance)
                fields = {}
                if append:
                    fields = {"returned_ids": list(result), "rows": self.rows(instance),
                              "conn": id(instance.connection)}
                self.record(label + "_return", owner_id=id(instance), **fields)
                return result

            patch.setattr(owner, name, traced)

        for module_name, class_name in STORES:
            module = sys.modules.get("hermes_lcm." + module_name)
            owner = getattr(module, class_name)
            wrap(owner, "__init__", class_name + "_init", constructor=True)
        engine = sys.modules["hermes_lcm.engine"].LCMEngine
        for name in ("ingest", "compress", "_schedule_rollup_maintenance",
                     "on_session_start", "on_session_end", "shutdown"):
            wrap(engine, name, name)
        wrap(sys.modules["hermes_lcm.store"].MessageStore,
             "_append_protected_batch", "append", append=True)
        rollups = sys.modules["hermes_lcm.rollup_store"].RollupStore
        for name in ("upsert_building", "mark_ready", "mark_failed", "close"):
            wrap(rollups, name, "rollup_" + name)
        # Both the engine's imported maintenance alias and the builder entry point.
        for module_name in ("engine", "rollup_builder"):
            module = sys.modules["hermes_lcm." + module_name]
            wrap(module, "run_rollup_maintenance", "rollup_maintenance")

    def close(self):
        self.record("protocol_exit", threads=sorted(t.name for t in threading.enumerate()))
        for connection in self.connections:
            try:
                connection.set_trace_callback(None)
            except sqlite3.Error:
                pass


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if item.path.parent.name != "diag_1753":
        return (yield)
    capture = Capture()
    item._diag_1753_capture = capture
    with closing(sqlite3.connect(":memory:")) as connection:
        compile_options = [row[0] for row in connection.execute("PRAGMA compile_options")]
    capture.record("runtime", python=sys.version, sqlite=sqlite3.sqlite_version,
                   sqlite_compile_options=compile_options,
                   platform=platform.platform(), runner_os=os.environ.get("RUNNER_OS"),
                   runner_image=os.environ.get("ImageOS"),
                   runner_image_version=os.environ.get("ImageVersion"),
                   threads=sorted(t.name for t in threading.enumerate()))
    with pytest.MonkeyPatch.context() as patch:
        capture.install(patch)
        try:
            return (yield)
        finally:
            capture.close()
            output = Path(os.environ["RCA_TRACE_DIR"])
            output.mkdir(parents=True, exist_ok=True)
            name = item.nodeid.rsplit("::", 1)[-1] + ".json"
            (output / name).write_text(json.dumps({"reports": capture.reports,
                                                   "events": capture.events}, indent=2) + "\n")


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    report = yield
    capture = getattr(item, "_diag_1753_capture", None)
    if capture is not None:
        engine = item.funcargs.get("engine")
        if engine is not None and report.when == "call":
            store = engine._store
            capture.record("call_exit_rows", engine=id(engine), store=id(store),
                           conn=id(store.connection), cursor=engine._ingest_cursor,
                           rows=capture.rows(store))
            with closing(sqlite3.connect(store.db_path)) as reader:
                capture.record("fresh_reader", conn=id(reader), ids=[row[0] for row in reader.execute(
                    "SELECT store_id FROM messages ORDER BY store_id").fetchall()])
        error_type = type(call.excinfo.value).__name__ if call.excinfo else None
        capture.reports.append({"phase": report.when, "outcome": report.outcome,
                                "error_type": error_type})
        capture.record("report", phase=report.when, outcome=report.outcome,
                       error_type=error_type)
        # The original assertion and caplog may contain synthetic message text.
        # Preserve its outcome while excluding that text from console/XML artifacts.
        if report.failed:
            report.longrepr = f"diagnostic failure: {report.when}; {error_type}; see structural capture"
        report.sections = []
    return report


def pytest_collectreport(report):
    if report.failed:
        report.longrepr = "diagnostic collection failed; no message or exception text retained"
        report.sections = []
