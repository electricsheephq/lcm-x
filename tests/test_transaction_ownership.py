"""Transaction-control traces captured before the ownership extraction.

Regenerate the complete file with LCM_REGENERATE_TRANSACTION_TRACES=1.
Only SQLite transaction-control statements enter the golden; row state and
exceptions are checked alongside them so an empty trace cannot hide a failure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from hermes_lcm import db_bootstrap  # type: ignore[import-not-found]
from hermes_lcm.store import MessageStore, build_message_fts_spec  # type: ignore[import-not-found]


GOLDEN = Path(__file__).with_name("fixtures") / "transaction_ownership.json"
CASES = (
    "append", "reassign", "delete", "gc", "pin", "unpin", "relations",
    "backfill", "metadata", "commit", "normalize", "host_uid",
)


class InjectedFailure(RuntimeError):
    pass


class TraceConnection(sqlite3.Connection):
    fail_write = False
    fail_commit = False

    def _after_write(self, sql: str) -> None:
        if self.fail_write and sql.lstrip().split()[0].upper() in (
            "INSERT", "UPDATE", "DELETE",
        ):
            self.fail_write = False
            raise InjectedFailure("after first write")

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        cursor = super().execute(sql, parameters)
        self._after_write(sql)
        return cursor

    def executemany(self, sql: str, parameters: Any, /) -> sqlite3.Cursor:
        cursor = super().executemany(sql, parameters)
        self._after_write(sql)
        return cursor

    def commit(self) -> None:
        if self.fail_commit:
            self.fail_commit = False
            raise InjectedFailure("before commit")
        super().commit()


@pytest.fixture
def message_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    original = sqlite3.connect

    def connect(*args: Any, **kwargs: Any) -> TraceConnection:
        return original(*args, **kwargs, factory=TraceConnection)

    monkeypatch.setattr(sqlite3, "connect", connect)
    owner = MessageStore(tmp_path / "transactions.db")
    owner.append("session", {"role": "user", "content": "source"})
    owner.append("session", {"role": "tool", "content": "result", "tool_call_id": "call"})
    owner._ensure_identity_anchor_schema()
    try:
        yield owner
    finally:
        owner.close()


@pytest.fixture(scope="module")
def golden() -> Iterator[dict[str, Any]]:
    regenerate = os.environ.get("LCM_REGENERATE_TRANSACTION_TRACES") == "1"
    records: dict[str, Any] = {} if regenerate else json.loads(GOLDEN.read_text())
    yield records
    if regenerate:
        expected = {f"{case}_{mode}" for case in CASES for mode in ("success", "failure")}
        expected |= {f"schema_{count}" for count in (1, 2)}
        expected |= {f"{case}_commit_failure" for case in ("host_uid", "normalize")}
        assert set(records) == expected, "Regenerate the entire file"
        GOLDEN.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def control_trace(conn: sqlite3.Connection) -> list[str]:
    statements: list[str] = []

    def record(sql: str) -> None:
        if sql.split()[0].upper() in ("BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE"):
            statements.append(sql)

    conn.set_trace_callback(record)
    return statements


def compare(golden: dict[str, Any], name: str, record: dict[str, Any]) -> None:
    if os.environ.get("LCM_REGENERATE_TRANSACTION_TRACES") == "1":
        golden[name] = record
    else:
        assert record == golden[name]


def operation(owner: Any, case: str) -> Callable[[], Any]:
    operations: dict[str, Callable[[], Any]] = {
        "append": lambda: owner.append("session", {"role": "user", "content": "new"}),
        "reassign": lambda: owner.reassign_session_messages("session", "next"),
        "delete": lambda: owner.delete_session_messages("session"),
        "gc": lambda: owner.gc_externalized_tool_result(2, "archived"),
        "pin": lambda: owner.pin(1),
        "unpin": lambda: owner.unpin(1),
        "relations": lambda: owner.add_message_relations([[(1, "alt_stamp", 2, 0, 10.0)]]),
        "backfill": lambda: owner.backfill_observed_at(1, 10.0),
        "metadata": lambda: owner.write_metadata_json(["trace"], "{}"),
        "commit": owner.commit,
        "normalize": owner.normalize_legacy_blank_sources,
        "host_uid": lambda: owner._host_uid_write(
            lambda conn: conn.execute("INSERT INTO metadata VALUES ('trace', '{}')")
        ),
    }
    return operations[case]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_transaction_trace(message_store: Any, golden: dict[str, Any], case: str, fail: bool) -> None:
    owner = message_store
    conn: TraceConnection = owner.connection
    if case == "unpin":
        owner.pin(1)
    if case == "normalize":
        conn.execute("UPDATE messages SET source = ''")
        conn.commit()
    trace = control_trace(conn)
    if case == "commit":
        conn.execute("INSERT INTO metadata VALUES ('trace', '{}')")
        conn.fail_commit = fail
    else:
        conn.fail_write = fail
    error = None
    try:
        operation(owner, case)()
    except InjectedFailure as exc:
        error = str(exc)
    assert (error is not None) == fail
    compare(golden, f"{case}_{'failure' if fail else 'success'}", {
        "trace": trace,
        "error": error,
        "in_transaction": conn.in_transaction,
        "messages": [list(row) for row in conn.execute(
            "SELECT store_id, session_id, role, content, pinned, source, observed_at FROM messages ORDER BY store_id"
        )],
        "metadata": [list(row) for row in conn.execute("SELECT * FROM metadata WHERE key='trace'")],
        "relations": [list(row) for row in conn.execute(
            "SELECT store_id, kind, related_store_id, ordinal, observed_at FROM message_relations"
        )],
    })


@pytest.mark.parametrize("failures", [1, 2])
def test_schema_recheck_trace(message_store: Any, golden: dict[str, Any], failures: int) -> None:
    conn: TraceConnection = message_store.connection
    attempts = 0

    class SchemaConnection:
        def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
            nonlocal attempts
            if sql.startswith("PRAGMA table_info"):
                attempts += 1
                if attempts <= failures:
                    exc = sqlite3.DatabaseError("database schema has changed")
                    exc.sqlite_errorcode = sqlite3.SQLITE_SCHEMA
                    raise exc
            return conn.execute(sql, parameters)

    trace = control_trace(conn)
    rebuild = db_bootstrap._fts_needs_rebuild_structural(SchemaConnection(), build_message_fts_spec())
    assert attempts == 2
    assert rebuild is (failures == 2)
    compare(golden, f"schema_{failures}", {
        "trace": trace, "attempts": attempts, "rebuild": rebuild,
        "in_transaction": conn.in_transaction,
    })


@pytest.mark.parametrize("case", ["host_uid", "normalize"])
def test_owner_commit_failure_trace(message_store: Any, golden: dict[str, Any], case: str) -> None:
    conn: TraceConnection = message_store.connection
    trace = control_trace(conn)
    conn.fail_commit = True
    if case == "normalize":
        # The native connection context commits internally in C; override a
        # deferred foreign key instead of its Python commit method for this case.
        conn.fail_commit = False
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("CREATE TABLE trace_parent(id PRIMARY KEY)")
        conn.execute("CREATE TABLE trace_child(id REFERENCES trace_parent(id) DEFERRABLE INITIALLY DEFERRED)")
        conn.execute("UPDATE messages SET source = ''")
        conn.commit()
        conn.execute("CREATE TRIGGER trace_invalid AFTER UPDATE ON messages BEGIN INSERT INTO trace_child VALUES (1); END")
        trace.clear()
    caught = None
    try:
        operation(message_store, case)()
    except (InjectedFailure, sqlite3.IntegrityError) as exc:
        caught = type(exc).__name__
    assert caught is not None
    compare(golden, f"{case}_commit_failure", {
        "trace": trace, "error": caught, "in_transaction": conn.in_transaction,
        "metadata": [list(row) for row in conn.execute("SELECT * FROM metadata WHERE key='trace'")],
    })


@pytest.mark.parametrize("immediate", [False, True], ids=["implicit", "immediate"])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_nested_atomic_trace(
    message_store: Any, golden: dict[str, Any], immediate: bool, fail: bool,
) -> None:
    owner = message_store
    conn: TraceConnection = owner.connection
    trace = control_trace(conn)
    error = None
    try:
        with owner.atomic(immediate=immediate):
            with owner.atomic(immediate=immediate):
                owner.write_metadata_json(["nested"], "{}")
                owner.backfill_observed_at(1, 10.0)
                owner.pin(1)
                if fail:
                    raise InjectedFailure("nested write")
    except InjectedFailure as exc:
        error = str(exc)
    assert (error is not None) == fail
    # The nested primitive must produce the same single owner trace captured
    # from the original explicit transaction, including on propagated failure.
    expected = golden[f"host_uid_{'failure' if fail else 'success'}"]["trace"]
    assert trace == expected
    assert not conn.in_transaction
    row = conn.execute("SELECT pinned, observed_at FROM messages WHERE store_id=1").fetchone()
    assert row == ((0, None) if fail else (1, 10.0))
    assert bool(conn.execute("SELECT 1 FROM metadata WHERE key='nested'").fetchone()) is not fail


def test_empty_implicit_atomic_has_no_transaction(message_store: Any, golden: dict[str, Any]) -> None:
    conn: TraceConnection = message_store.connection
    trace = control_trace(conn)
    with message_store.atomic():
        with message_store.atomic():
            pass
    assert trace == golden["schema_1"]["trace"] == []


def test_nested_immediate_reserves_inside_an_empty_scope(message_store: Any, tmp_path: Path) -> None:
    """An empty implicit scope has not reserved the database; a nested immediate entry still must."""
    owner = message_store
    conn: TraceConnection = owner.connection
    other = sqlite3.connect(tmp_path / "transactions.db", timeout=0, isolation_level=None)
    seen: list[bool] = []

    def write(raw: sqlite3.Connection) -> None:
        seen.append(raw.in_transaction)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            other.execute("BEGIN IMMEDIATE")
        raw.execute("DELETE FROM metadata WHERE key='absent'")

    try:
        with owner.atomic():
            owner._host_uid_write(write)
            assert conn.in_transaction
    finally:
        other.close()
    assert seen == [True]
    assert not conn.in_transaction


def test_host_uid_still_refuses_an_enclosing_transaction(message_store: Any) -> None:
    owner = message_store
    conn: TraceConnection = owner.connection
    with owner.atomic(immediate=True):
        owner.write_metadata_json(["pending"], "{}")
        with pytest.raises(sqlite3.OperationalError, match="host-uid shadow write skipped"):
            owner._host_uid_write(lambda raw: raw.execute("DELETE FROM metadata WHERE key='pending'"))
        assert conn.in_transaction
    assert conn.execute("SELECT value FROM metadata WHERE key='pending'").fetchone() == ("{}",)
