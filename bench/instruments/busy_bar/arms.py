"""Opt-in, process-local storage probes. Never installed for a default cell."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from unittest.mock import patch


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive(value: str) -> int:
    number = nonnegative(value)
    if not number:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def checkpointer(value: str) -> str:
    if value == "off":
        return value
    if not value.startswith("passive:"):
        raise argparse.ArgumentTypeError("expected off or passive:<positive ms>")
    try:
        positive(value.removeprefix("passive:"))
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("expected off or passive:<positive ms>") from exc
    return value


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wal-autocheckpoint", type=nonnegative)
    parser.add_argument("--checkpointer", type=checkpointer, default="off")
    parser.add_argument("--background-writers", choices=("none", "stem", "integrity", "embedding", "all"),
                        default="none")
    parser.add_argument("--stem-backlog", type=positive, default=20000)
    parser.add_argument("--record-holds", action="store_true")


def arguments(args: argparse.Namespace) -> list[str]:
    """Forward only opt-ins, preserving the original default child command."""
    result = []
    for name, default in (("wal_autocheckpoint", None), ("checkpointer", "off"),
                          ("background_writers", "none"), ("stem_backlog", 20000)):
        value = getattr(args, name)
        if value != default:
            result.extend(["--" + name.replace("_", "-"), str(value)])
    if args.record_holds:
        result.append("--record-holds")
    return result


class HoldRecorder:
    """Wrap SQLite calls, including implicit DML and cursor/context-manager paths.

    Explicit IMMEDIATE holds start after acquisition; implicit first writes and
    integrity INSERTs start at call entry (and may include their acquisition wait).
    Deferred BEGIN/read savepoints alone are not write holds. Busy attempts have
    their own outcome. Release-call ends are sampled before the call; autocommit
    write ends are sampled after the call. All timestamps use the process
    monotonic clock.
    """

    def __init__(self, database=None) -> None:
        self.database = database
        self.records: list[dict] = []
        self.local = threading.local()
        self.lock = threading.Lock()
        self.next_id = 0

    def writer(self) -> str:
        name = threading.current_thread().name
        return getattr(self.local, "writer", None) or {
            "MainThread": "foreground-A", "busy-bar-B": "foreground-B"}.get(name, name)

    @contextmanager
    def named(self, name):
        previous = getattr(self.local, "writer", None)
        self.local.writer = name
        try:
            yield
        finally:
            self.local.writer = previous

    def connection_class(self):
        recorder = self

        class Connection(sqlite3.Connection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.database = os.fsdecode(args[0] if args else kwargs["database"])
                self.record_holds = bool(
                    recorder.database and self.database and self.database != ":memory:"
                    and Path(self.database).resolve() == Path(recorder.database).resolve())
                with recorder.lock:
                    recorder.next_id += 1
                    self.connection_id = recorder.next_id
                self.hold = None
                self.scan_hold = None

            def begin_hold(self, start, writer=None, basis="call_entry"):
                if self.record_holds and self.hold is None:
                    self.hold = (start, writer or recorder.writer(), basis)

            def end_hold(self, outcome, end=None, *, scan=False):
                hold = self.scan_hold if scan else self.hold
                if hold is not None:
                    start, writer, basis = hold
                    end = time.monotonic() if end is None else end
                    row = {"writer": writer, "connection_id": self.connection_id,
                           "database": self.database, "start_basis": basis,
                           "start": start, "end": end,
                           "hold_ms": (end - start) * 1000, "outcome": outcome}
                    if basis == "acquired":
                        row["acquired_at"] = start
                    with recorder.lock:
                        recorder.records.append(row)
                    if scan:
                        self.scan_hold = None
                    else:
                        self.hold = None

            def statement(self, sql, call):
                if not self.record_holds:
                    return call()
                text = sql.strip().upper()
                write = text.startswith(("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER", "REINDEX"))
                integrity = "'INTEGRITY-CHECK'" in text
                before = time.monotonic()
                try:
                    result = call()
                except sqlite3.Error as exc:
                    if (getattr(exc, "sqlite_errorcode", 0) & 0xff in (5, 6)
                            and (write or text.startswith("BEGIN") or self.hold is not None)):
                        # A failed BEGIN acquired no lock; do not consume an existing hold.
                        if integrity:
                            self.scan_hold = (before, "integrity-scan", "call_entry")
                            self.end_hold("busy", scan=True)
                        elif self.hold is None:
                            self.begin_hold(before)
                            self.end_hold("busy")
                    elif write and self.in_transaction:
                        self.begin_hold(before)
                    raise
                after = time.monotonic()
                if text.startswith(("BEGIN IMMEDIATE", "BEGIN EXCLUSIVE")):
                    self.begin_hold(after, basis="acquired")
                elif integrity:
                    self.scan_hold = (before, "integrity-scan", "call_entry")
                elif write:
                    self.begin_hold(before)
                    if "INSERT INTO MESSAGES_FTS_STEM(ROWID" in text and self.hold:
                        self.hold = (self.hold[0], "stem-batch", self.hold[2])
                if text.startswith("ROLLBACK") and self.scan_hold is not None:
                    self.end_hold("rollback", before, scan=True)
                if not self.in_transaction:
                    release = text.startswith(("COMMIT", "END", "ROLLBACK", "RELEASE"))
                    self.end_hold("rollback" if text.startswith("ROLLBACK") else "commit",
                                  before if release else after)
                return result

            def execute(self, sql, parameters=()):
                return self.statement(sql, lambda: super(Connection, self).execute(sql, parameters))

            def executemany(self, sql, parameters):
                # In autocommit each row is a separate transaction.
                cursor = self.cursor()
                return cursor.executemany(sql, parameters)

            def cursor(self, factory=None):
                return super().cursor(factory or Cursor)

            def executescript(self, script):
                # Match sqlite3's legacy implicit pre-script commit, then preserve
                # complete SQL statements (including trigger bodies) individually.
                self.commit()
                isolation = self.isolation_level
                statement = ""
                cursor = self.cursor()
                try:
                    # Native executescript never adds an implicit DML transaction.
                    self.isolation_level = None
                    for char in script:
                        statement += char
                        if char == ";" and sqlite3.complete_statement(statement):
                            cursor.execute(statement)
                            statement = ""
                    if statement.strip():
                        cursor.execute(statement)
                finally:
                    # Assigning None commits even an explicit script transaction.
                    if isolation is not None:
                        self.isolation_level = isolation
                return cursor

            def commit(self):
                end = time.monotonic()
                result = super().commit()
                self.end_hold("commit", end)
                return result

            def rollback(self):
                end = time.monotonic()
                result = super().rollback()
                self.end_hold("rollback", end)
                return result

            def close(self):
                end = time.monotonic()
                result = super().close()
                self.end_hold("rollback", end)
                return result

            def __exit__(self, kind, value, traceback):
                # sqlite3's C context manager bypasses overridden commit/rollback.
                end = time.monotonic()
                try:
                    result = super().__exit__(kind, value, traceback)
                except BaseException:
                    self.end_hold("rollback", end)
                    raise
                self.end_hold("rollback" if kind else "commit", end)
                return result

        class Cursor(sqlite3.Cursor):
            def execute(self, sql, parameters=()):
                self.__dict__.pop("_many_rowcount", None)
                return self.connection.statement(sql, lambda: super(Cursor, self).execute(sql, parameters))

            def executemany(self, sql, parameters):
                if self.connection.isolation_level is not None or self.connection.in_transaction:
                    return self.connection.statement(
                        sql, lambda: super(Cursor, self).executemany(sql, parameters))
                count = 0
                for row in parameters:
                    self.execute(sql, row)
                    count += self.rowcount
                self._many_rowcount = count
                return self

            @property
            def rowcount(self):
                return getattr(self, "_many_rowcount", super().rowcount)

        return Connection


class FakeEmbedder:
    """Deterministic local vectors; retains the product's dispatch/batch protocol."""

    provider_id = "ollama"
    model_id = "busy-bar-fake"
    dim = 2
    last_skipped_documents = ()
    last_usage_tokens = 0

    def embed_documents(self, texts):
        return [[1.0, (int.from_bytes(hashlib.sha256(text.encode()).digest()[:2]) + 1) / 65536]
                for text in texts]

    def embed_document_batches(self, texts, *, before_dispatch):
        from hermes_lcm.embedding_provider import EmbeddedDocumentBatch

        indexes = tuple(range(len(texts)))
        before_dispatch(indexes)
        yield EmbeddedDocumentBatch(indexes, tuple(tuple(vector) for vector in self.embed_documents(texts)))

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class StorageArms:
    def __init__(self, args):
        self.args = args
        self.enabled = bool(arguments(args))
        self.writers = set() if args.background_writers == "none" else (
            {"stem", "integrity", "embedding"} if args.background_writers == "all" else {args.background_writers})
        self.stack = ExitStack()
        self.recorder = HoldRecorder()
        self.gate = threading.Event()
        self.stop = threading.Event()
        self.checkpoints = []
        self.configured = []
        self.seeding = False
        self.thread = None
        self.background_errors = []
        self.embedding_errors = []
        self.embedding_drains = []
        self.embedding_owners = []
        self.engines = []

    def __enter__(self):
        if not self.enabled:
            return self
        bootstrap = importlib.import_module("hermes_lcm.db_bootstrap")
        self.bootstrap = bootstrap
        if self.args.record_holds:
            original_connect = sqlite3.connect
            factory = self.recorder.connection_class()

            def connect(*args, **kwargs):
                if len(args) < 6 and "factory" not in kwargs:
                    kwargs["factory"] = factory
                return original_connect(*args, **kwargs)

            self.stack.enter_context(patch.object(sqlite3, "connect", connect))
        if self.args.wal_autocheckpoint is not None:
            original = bootstrap.configure_connection

            def configure(conn):
                original(conn)
                conn.execute(f"PRAGMA wal_autocheckpoint={self.args.wal_autocheckpoint}")
                self.configured.append({"wal_autocheckpoint": conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0],
                                        "synchronous": conn.execute("PRAGMA synchronous").fetchone()[0]})

            # Stores import this function by name. Replace every already-loaded
            # alias too; later imports see the wrapped bootstrap binding.
            for name, module in list(sys.modules.items()):
                if name.startswith("hermes_lcm") and getattr(module, "configure_connection", None) is original:
                    self.stack.enter_context(patch.object(module, "configure_connection", configure))
        if self.writers:
            for name, dispatch, body in (
                ("stem", "dispatch_background_stem_backfill", "_run_stem_backfill"),
                ("integrity", "_dispatch_background_integrity_scan", "_run_background_integrity_scan"),
            ):
                original_dispatch = getattr(bootstrap, dispatch)

                def dispatch_when_ready(*args, _name=name, _original=original_dispatch, **kwargs):
                    if self.seeding or _name not in self.writers:
                        return True
                    with self.recorder.named(f"{_name}-claim"):
                        return _original(*args, **kwargs)

                self.stack.enter_context(patch.object(bootstrap, dispatch, dispatch_when_ready))
                original_body = getattr(bootstrap, body)

                def run_when_ready(*args, _name=name, _original=original_body, **kwargs):
                    self.gate.wait()
                    with self.recorder.named(_name):
                        return _original(*args, **kwargs)

                self.stack.enter_context(patch.object(bootstrap, body, run_when_ready))
        if "embedding" in self.writers:
            command = importlib.import_module("hermes_lcm.command")
            provider = importlib.import_module("hermes_lcm.embedding_provider")
            maintenance = importlib.import_module("hermes_lcm.embedding_maintenance")
            vectors = importlib.import_module("hermes_lcm.vector_store")
            self.stack.enter_context(patch.object(command, "resolve_provider", lambda *a, **kw: FakeEmbedder()))
            self.stack.enter_context(patch.object(provider, "resolve_provider", lambda *a, **kw: FakeEmbedder()))
            self.stack.enter_context(patch.object(provider, "probe_provider_availability",
                                                 lambda config: {"available": True, "detail": "busy-bar fake"}))
            original_pass = maintenance.run_incremental_embedding_pass

            def embedding_pass(*args, **kwargs):
                self.gate.wait()
                with self.recorder.named("embedding"):
                    return original_pass(*args, **kwargs)

            # engine imported the pass by name.
            engine = importlib.import_module("hermes_lcm.engine")
            self.stack.enter_context(patch.object(engine, "run_incremental_embedding_pass", embedding_pass))
            original_retire = engine.LCMEngine._retire_embedding_maintenance

            def retire(instance, wait):
                self.gate.set()  # Also release a pass gated before a failed cell's first operation.
                if wait:
                    self.drain_embedding(instance)
                else:
                    self.embedding_owners.append(instance._embedding_maintenance_owner)
                try:
                    return original_retire(instance, wait)
                except Exception as exc:
                    self.embedding_errors.append(f"embedding retire:{type(exc).__name__}:{exc}")
                    raise

            self.stack.enter_context(patch.object(engine.LCMEngine, "_retire_embedding_maintenance", retire))
            scheduler_module = importlib.import_module("hermes_lcm.maintenance_scheduler")
            scheduler = scheduler_module._EMBEDDING_MAINTENANCE_SCHEDULER
            self.scheduler = scheduler
            original_retire_owner = scheduler.retire_owner
            original_warning = scheduler_module.logger.warning

            def warning(message, *args, **kwargs):
                if message == "LCM could not retire %s maintenance owner" and args == (scheduler._kind,) and kwargs.get("exc_info"):
                    self.embedding_errors.append(f"embedding retire:{sys.exc_info()[1]}")
                return original_warning(message, *args, **kwargs)

            self.stack.enter_context(patch.object(scheduler_module.logger, "warning", warning))

            def retire_owner(*args, **kwargs):
                try:
                    retired = original_retire_owner(*args, **kwargs)
                except Exception as exc:
                    self.embedding_errors.append(f"embedding retire:{type(exc).__name__}:{exc}")
                    raise
                if not retired and kwargs.get("timeout", 0) != 0:
                    self.embedding_errors.append("embedding retire:timeout")
                return retired

            # The engine deliberately swallows retirement exceptions and False.
            self.stack.enter_context(patch.object(scheduler, "retire_owner", retire_owner))
            original_shutdown = engine.LCMEngine.shutdown

            def shutdown(instance):
                try:
                    return original_shutdown(instance)
                except Exception as exc:
                    self.embedding_errors.append(f"embedding shutdown:{type(exc).__name__}:{exc}")
                    raise

            self.stack.enter_context(patch.object(engine.LCMEngine, "shutdown", shutdown))
            original_transaction = vectors.VectorStore._write_transaction

            @contextmanager
            def vector_transaction(store):
                with self.recorder.named("embedding-vector-batch"), original_transaction(store):
                    yield

            self.stack.enter_context(patch.object(vectors.VectorStore, "_write_transaction", vector_transaction))
        return self

    def drain_embedding(self, engine, owner=None):
        if "embedding" not in self.writers:
            return
        record = {"deferred_owner": owner is not None, "skipped": bool(self.embedding_errors), "duration_ms": 0.0}
        self.embedding_drains.append(record)
        if record["skipped"]:
            return
        started = time.monotonic()
        try:
            drained = (self.scheduler.retire_owner(owner, timeout=35) if owner is not None
                       else engine.drain_embedding_maintenance(timeout=35))
            if not drained:
                self.embedding_errors.append("embedding drain:timeout")
        except Exception as exc:
            self.embedding_errors.append(f"embedding drain:{type(exc).__name__}:{exc}")
        finally:
            record["duration_ms"] = (time.monotonic() - started) * 1000

    def prepare(self, config, directory):
        if self.embedding_errors:
            raise RuntimeError("cannot start another run after failed embedding retirement")
        self.engines.clear()
        self.embedding_drains.clear()
        self.embedding_owners.clear()
        self.directory = directory
        self.recorder.database = config.database_path
        self.schedule_start = None
        self.gate.clear()
        self.stop.clear()
        self.recorder.records.clear()
        self.configured.clear()
        self.checkpoints = []
        self.background_errors.clear()
        if not self.writers:
            return
        from hermes_lcm.store import MessageStore, build_message_stem_fts_spec

        self.seeding = True
        try:
            store = MessageStore(config.database_path)
            conn = store._conn
            count = self.args.stem_backlog if "stem" in self.writers else 2000
            if "stem" in self.writers:
                for sql in build_message_stem_fts_spec().trigger_sqls:
                    # Drop only the instrument's freshly created stem triggers.
                    name = sql.split("TRIGGER IF NOT EXISTS ")[1].split()[0]
                    conn.execute(f"DROP TRIGGER {name}")
            conn.executemany("INSERT INTO messages(session_id,role,content,timestamp) VALUES(?,?,?,?)",
                             (("busy-bar-seed", "user", f"seed {i} alpha beta gamma " * 8, float(i))
                              for i in range(count)))
            if "stem" in self.writers:
                for sql in build_message_stem_fts_spec().trigger_sqls:
                    conn.execute(sql)
                self.bootstrap._stem_metadata(conn, "fts_stem_state", "backfilling")
            if "integrity" in self.writers:
                conn.execute("DELETE FROM metadata WHERE key LIKE 'fts_integrity_%'")
            conn.commit()
            store.close()
            if "embedding" in self.writers:
                from hermes_lcm.dag import SummaryDAG, SummaryNode
                from hermes_lcm.vector_store import VectorStore

                dag = SummaryDAG(config.database_path)
                for i in range(64):
                    dag.add_node(SummaryNode(session_id="busy-bar-seed", depth=0, summary=f"seed summary {i}",
                                             created_at=float(i), latest_at=float(i)))
                dag.close()
                vectors = VectorStore(config.database_path, config=config)
                vectors.register_profile(FakeEmbedder.model_id, FakeEmbedder.provider_id, FakeEmbedder.dim)
                vectors.close()
        finally:
            self.seeding = False
        self.recorder.records.clear()

    def start(self, config):
        self.schedule_start = time.monotonic()
        self.gate.set()
        self.thread = None
        if self.args.checkpointer != "off":
            interval = int(self.args.checkpointer.split(":")[1]) / 1000

            def checkpoint():
                conn = None
                try:
                    conn = sqlite3.connect(config.database_path)
                    self.bootstrap.configure_connection(conn)
                    while not self.stop.is_set():
                        start = time.monotonic()
                        row, error = None, None
                        try:
                            row = tuple(conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone())
                        except sqlite3.Error as exc:
                            error = str(exc)
                        end = time.monotonic()
                        self.checkpoints.append({"start": start, "duration_ms": (end - start) * 1000,
                                                 "result": row, "error": error})
                        self.stop.wait(interval)
                except Exception as exc:
                    self.background_errors.append(f"checkpointer:{type(exc).__name__}:{exc}")
                finally:
                    if conn is not None:
                        conn.close()

            self.thread = threading.Thread(target=checkpoint, name="busy-bar-checkpointer", daemon=True)
            self.thread.start()

    def finish(self, output):
        self.stop.set()
        self.gate.set()
        for owner in self.embedding_owners:
            self.drain_embedding(None, owner)
        # The failed-before-first-operation path reaches finish before shutdown.
        for engine in self.engines:
            self.drain_embedding(engine)
        self.schedule_end = time.monotonic()
        if self.thread:
            self.thread.join(35)
            if self.thread.is_alive():
                raise RuntimeError("checkpointer failed to stop")
        self.bootstrap.join_background_integrity_scans(timeout=35)
        self.bootstrap.join_background_stem_backfills(timeout=35)
        if self.background_errors:
            raise RuntimeError("; ".join(self.background_errors))
        workers = [*self.bootstrap._stem_threads.values(), *self.bootstrap._integrity_scan_threads.values()]
        if any(worker.is_alive() for worker in workers):
            raise RuntimeError("background writer failed to stop")
        output.mkdir(parents=True, exist_ok=False)
        if self.args.record_holds:
            with (output / "holds.jsonl").open("x") as stream:
                for row in self.recorder.records:
                    stream.write(json.dumps(row) + "\n")
        if self.args.checkpointer != "off":
            with (output / "checkpoints.jsonl").open("x") as stream:
                for row in self.checkpoints:
                    stream.write(json.dumps(row) + "\n")
        details = {"wal_autocheckpoint": self.args.wal_autocheckpoint, "checkpointer": self.args.checkpointer,
                "background_writers": self.args.background_writers, "stem_backlog": self.args.stem_backlog,
                "record_holds": self.args.record_holds, "configured_connections": self.configured,
                "schedule_start": self.schedule_start, "schedule_end": self.schedule_end}
        if self.embedding_errors:
            details["errors"] = self.embedding_errors.copy()
        return details

    def __exit__(self, *exc):
        self.gate.set()
        self.stop.set()
        if self.thread is not None:
            self.thread.join(35)
        if self.enabled:
            self.bootstrap.join_background_integrity_scans(timeout=35)
            self.bootstrap.join_background_stem_backfills(timeout=35)
        return self.stack.__exit__(*exc)
