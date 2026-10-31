"""Failure-only structural tracing adapted from #1764; never retain payloads."""
from collections import deque
from contextlib import closing
from functools import wraps
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

TARGET = "tests/test_issue_1306_cli_new.py::test_new_stores_all_turns_under_new_session"
SQL_KINDS = {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "INSERT", "UPDATE", "DELETE"}


def revision(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


class Capture:
    def __init__(self):
        self.events, self.connections = deque(maxlen=512), set()
        self.lock, self.failed = threading.RLock(), False

    def record(self, kind, **fields):
        with self.lock:
            self.events.append(dict(kind=kind, time=time.monotonic_ns(), thread=threading.get_ident(), **fields))

    def state(self, instance):
        conn = getattr(instance, "_conn", None)
        if isinstance(conn, sqlite3.Connection):
            return dict(conn=id(conn), in_transaction=conn.in_transaction)
        proof = getattr(instance, "_compress_commit_proof", None)
        return dict(cursor=getattr(instance, "_ingest_cursor", None), proof_present=proof is not None,
                    end_consumed=bool((proof or {}).get("end_consumed")),
                    frontier=getattr(instance, "_last_compacted_store_id", None),
                    pending_tails=len(getattr(instance, "_pending_session_tails", [])))

    def register(self, instance):
        conn = getattr(instance, "_conn", None)
        if not isinstance(conn, sqlite3.Connection) or conn in self.connections:
            return
        self.connections.add(conn)
        def trace(statement):
            # SQLite supplies expanded text: discard everything except an allowlisted verb.
            words = statement.lstrip().split(None, 1)
            verb = words[0].upper() if words else ""
            if verb in SQL_KINDS:
                self.record("statement", conn=id(conn), statement=verb)
        conn.set_trace_callback(trace)

    def install(self, patch):
        def wrap(owner, name, label):
            original = getattr(owner, name)
            @wraps(original)
            def traced(instance, *args, **kwargs):
                fields = self.state(instance) if name != "__init__" else {}
                if label == "append":
                    fields.update(input_count=len(kwargs["messages_to_store_with_index"]),
                                  protected_count=len(kwargs["protected_messages"]))
                if label == "protected_append":
                    estimates = args[2] if len(args) > 2 else kwargs.get("token_estimates")
                    fields.update(estimate_count=len(estimates) if estimates is not None else None)
                self.record(label + "_enter", owner=id(instance), **fields)
                try:
                    result = original(instance, *args, **kwargs)
                except BaseException:
                    self.record(label + "_error", owner=id(instance))
                    raise
                self.register(instance)
                fields = self.state(instance)
                if label in {"append", "protected_append"}:
                    fields["returned_ids"] = list(result)
                self.record(label + "_return", owner=id(instance), **fields)
                return result
            patch.setattr(owner, name, traced)
        for module, cls in (("store", "MessageStore"), ("dag", "SummaryDAG"), ("rollup_store", "RollupStore")):
            owner = getattr(sys.modules["hermes_lcm." + module], cls)
            wrap(owner, "__init__", cls)
        engine = sys.modules["hermes_lcm.engine"].LCMEngine
        for name in ("ingest", "compress", "on_session_start", "on_session_end", "_schedule_rollup_maintenance"):
            wrap(engine, name, name)
        wrap(engine, "_store_ingest_rows", "append")
        wrap(sys.modules["hermes_lcm.store"].MessageStore, "_append_protected_batch", "protected_append")
        for name in ("upsert_building", "mark_ready", "mark_failed"):
            wrap(sys.modules["hermes_lcm.rollup_store"].RollupStore, name, "rollup_" + name)
        for module in ("engine", "rollup_builder"):
            wrap(sys.modules["hermes_lcm." + module], "run_rollup_maintenance", "rollup_maintenance")

    def rows(self, conn, kind):
        try:
            rows = conn.execute("SELECT store_id, session_id, role FROM messages ORDER BY store_id").fetchall()
            self.record(kind, conn=id(conn), rows=[
                [sid, session if session in {"old", "new"} else "other",
                 role if role in {"user", "assistant"} else "other", ordinal]
                for ordinal, (sid, session, role) in enumerate(rows, 1)])
        except sqlite3.Error:
            self.record(kind + "_unavailable", conn=id(conn))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if item.nodeid not in {TARGET + "[cli]", TARGET + "[gateway]"}:
        return (yield)
    capture = item._diag_1753_capture = Capture()
    pins = dict(source=revision(Path(__file__).parents[1]), python=platform.python_version(),
                host=revision(os.environ["LCM_REAL_HERMES_SRC"]) if os.environ.get("LCM_REAL_HERMES_SRC") else None,
                sqlite=sqlite3.sqlite_version)
    with pytest.MonkeyPatch.context() as patch:
        capture.install(patch)
        try:
            return (yield)
        finally:
            for conn in capture.connections:
                try:
                    conn.set_trace_callback(None)
                except sqlite3.Error:
                    pass
            if capture.failed:
                output = Path(os.environ.get("LCM_DIAG_1753_DIR", "diag-1753-captures"))
                output.mkdir(parents=True, exist_ok=True)
                (output / (item.nodeid.rsplit("::", 1)[-1] + ".json")).write_text(
                    json.dumps(dict(schema=1, pins=pins, events=list(capture.events)), indent=2) + "\n")


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    report = yield
    capture = getattr(item, "_diag_1753_capture", None)
    if capture is not None:
        capture.failed |= report.failed
        capture.record("report", phase=report.when, outcome=report.outcome)
        engine = item.funcargs.get("engine")
        if report.when == "call" and engine is not None:
            capture.rows(engine._store.connection, "same_reader")
            try:
                with closing(sqlite3.connect(f"file:{engine._store.db_path}?mode=ro", uri=True)) as reader:
                    capture.rows(reader, "fresh_reader")
            except sqlite3.Error:
                capture.record("fresh_reader_unavailable")
        if report.failed:
            report.longrepr = "Target failed; see failure-only structural capture (#1753)."
        report.sections = []
    return report
