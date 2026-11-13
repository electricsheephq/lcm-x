"""Recording-only #2293 oracle. Never install a SQLite callback beside another thread."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class PrelockRecorder:
    store: object
    begins: list = field(default_factory=list)
    forbidden: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)
    explicit: bool = False
    denials: int = 0

    def authorize(self, action, arg1, arg2, database, source):
        state = self.store._ingest_prelock_state
        if state is None or state.phase != "prelock" or self.explicit:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_TRANSACTION and arg1 == "BEGIN":
            self.begins.append(dict(state=state, T_begin=self.store._conn.total_changes,
                                    T_clean=state.T_clean, failed=False, checked=False))
        forbidden = {getattr(sqlite3, name) for name in dir(sqlite3)
                     if name.startswith(("SQLITE_CREATE_", "SQLITE_DROP_"))}
        forbidden.update(getattr(sqlite3, "SQLITE_" + name) for name in
                         ("SAVEPOINT", "ALTER_TABLE", "REINDEX", "ANALYZE", "ATTACH", "DETACH"))
        if (action in forbidden or action == sqlite3.SQLITE_PRAGMA and arg2 is not None
                or action == sqlite3.SQLITE_TRANSACTION and arg1 in ("COMMIT", "ROLLBACK")):
            self.forbidden.append((action, arg1, arg2))
        return sqlite3.SQLITE_OK

    def failed(self, exc):
        if (isinstance(exc, sqlite3.Error) and not self.store._conn.in_transaction
                and self.begins and not self.begins[-1]["checked"]):
            entry = self.begins[-1]
            entry["failed"] = True
            self.failures.append(entry)

    def check(self):
        assert not self.forbidden, self.forbidden
        assert self.denials == 0
        for entry in self.begins:
            assert entry["T_begin"] == entry["T_clean"], entry
        for entry in self.failures:
            assert entry["state"].poisoned, entry
        assert all(production == oracle for production, oracle in self.verdicts), self.verdicts


@contextmanager
def recording_prelock(store, *, unsafe_threaded_positive_control=False):
    if threading.active_count() > 1 and not unsafe_threaded_positive_control:
        raise RuntimeError("#2293 recording authorizer requires a single-threaded process")
    conn = store._conn
    recorder = PrelockRecorder(store)
    setter = getattr(conn, "set_harness_authorizer", conn.set_authorizer)
    previous = getattr(conn, "_harness_authorizer", None)
    original_instance_setter = conn.__dict__.get("set_harness_authorizer") if hasattr(conn, "__dict__") else None

    def composed(authorizer):
        def record_then_fault(*args):
            recorder.authorize(*args)  # recording always allows; fault policy stays independent
            return sqlite3.SQLITE_OK if authorizer is None else authorizer(*args)
        setter(record_then_fault)

    store._prelock_test_recorder = recorder
    if hasattr(conn, "set_harness_authorizer"):
        conn.set_harness_authorizer = composed
    composed(previous)
    try:
        yield recorder
    finally:
        if hasattr(conn, "set_harness_authorizer"):
            if original_instance_setter is None:
                del conn.set_harness_authorizer
            else:
                conn.set_harness_authorizer = original_instance_setter
        setter(previous)
        del store._prelock_test_recorder
        if not unsafe_threaded_positive_control:
            recorder.check()


def install_oracle_hooks(monkeypatch):
    from hermes_lcm.store import MessageStore
    sync, begin, failed = (MessageStore._sync_ingest_prelock, MessageStore._begin_ingest_prelock,
                           MessageStore._ingest_prelock_write_failed)

    def checked_sync(store):
        rec = getattr(store, "_prelock_test_recorder", None)
        state, conn = store._ingest_prelock_state, store._conn
        if rec and state and state.phase == "prelock" and conn.in_transaction:
            entries = [e for e in rec.begins if e["state"] is state and not e["checked"] and not e["failed"]]
            if entries:
                entry = entries[-1]
                versions = store._ingest_write_generation()[1:]
                oracle = entry["T_begin"] == state.G0[0] and versions == state.G0[1:] and not state.poisoned
                production = state.T_clean == state.G0[0] and versions == state.G0[1:] and not state.poisoned
                rec.verdicts.append((production, oracle))
                entry["checked"] = True
        return sync(store)

    def explicit_begin(store, state):
        rec = getattr(store, "_prelock_test_recorder", None)
        if rec:
            rec.explicit = True
        try:
            return begin(store, state)
        finally:
            if rec:
                rec.explicit = False

    def write_failed(store, exc):
        rec = getattr(store, "_prelock_test_recorder", None)
        if rec:
            rec.failed(exc)
        return failed(store, exc)

    monkeypatch.setattr(MessageStore, "_sync_ingest_prelock", checked_sync)
    monkeypatch.setattr(MessageStore, "_begin_ingest_prelock", explicit_begin)
    monkeypatch.setattr(MessageStore, "_ingest_prelock_write_failed", write_failed)
