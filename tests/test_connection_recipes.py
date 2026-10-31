"""Pin connection recipes, not the schema SQL that follows them.

Regenerate with LCM_REGENERATE_CONNECTION_RECIPES=1 python -m pytest
tests/test_connection_recipes.py -q (writes the golden and skips trace comparisons).
Run the whole file so every success and error case is captured.
"""

import json
import os
from pathlib import Path
import sqlite3
import time

import pytest

from hermes_lcm import dag, db_bootstrap, lifecycle_state, store, vector_store
from hermes_lcm.assertion_store import AssertionStore
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.query_view_store import QueryViewStore
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.store import MessageStore
from hermes_lcm.trajectory_store import CorpusIdentity, TrajectoryStore
from hermes_lcm.vector_store import VectorStore


CORE_CASES = ("store", "dag")
WRITABLE_CASES = ("trajectory", "assertion", "rollup", "query_view", "lifecycle", "vector", *CORE_CASES)
CASES = ("trajectory", "trajectory_ro", "assertion", "assertion_ro", "rollup", "query_view", "lifecycle", "vector", *CORE_CASES)
GOLDEN = Path(__file__).with_name("fixtures") / "connection_recipes.json"


def recipe_trace(events):
    # The core stores retain SQLite's tuple rows, with no row_factory assignment.
    if not any(event[:2] == ["set", "row_factory"] for event in events):
        end = next(i for i, event in enumerate(events) if event[0] == "executescript")
        return events[:end] + [["executescript", events[end][1].split()[0].upper()]]
    end = next(i for i, event in enumerate(events) if event[:2] == ["set", "row_factory"])
    boundary = events[end + 1]
    assert boundary[0] in ("execute", "executescript")
    return events[:end + 1] + [[boundary[0], boundary[1].split()[0].upper()]]


@pytest.fixture(scope="session")
def golden_traces():
    regenerate = os.environ.get("LCM_REGENERATE_CONNECTION_RECIPES") == "1"
    traces = {} if regenerate else json.loads(GOLDEN.read_text())
    yield traces
    if regenerate:
        expected = set(CASES) | {
            f"{prefix}_{kind}"
            for prefix in ("future", "failure")
            for kind in WRITABLE_CASES
        } | {
            f"lifetime_{kind}_{state}"
            for kind in CORE_CASES
            for state in ("fresh", "existing")
        }
        assert set(traces) == expected, "Regenerate by running the entire test file"
        lines = ["{"]
        for name, events in traces.items():
            lines.append(f"  {json.dumps(name)}: [")
            lines.extend(f"    {json.dumps(event)}," for event in events)
            lines[-1] = lines[-1].removesuffix(",")
            lines.append("  ],")
        lines[-1] = lines[-1].removesuffix(",")
        GOLDEN.write_text("\n".join(lines + ["}", ""]))


@pytest.fixture
def check_trace(golden_traces):
    def check(name, events, error=None):
        trace = events if error is None else events + [
            ["exception", type(error).__name__, str(error)],
        ]
        if os.environ.get("LCM_REGENERATE_CONNECTION_RECIPES") == "1":
            golden_traces[name] = trace
            pytest.skip("Regenerating connection recipe golden")
        assert trace == golden_traces[name]
    return check


def prepare(kind, path):
    MessageStore(path).close()
    SummaryDAG(path).close()
    if kind.endswith("_ro"):
        construct(kind.removesuffix("_ro"), path).close()


def construct(kind, path):
    if kind.startswith("trajectory"):
        return TrajectoryStore(
            path,
            CorpusIdentity(
                dataset_name="example/recipes", dataset_revision="revision-1",
                harness_commit="harness-1", tier="small", domain="example",
                ingest_config_digest="ingest-1",
            ),
            asset_root=path.parent, read_only=kind.endswith("_ro"),
        )
    if kind.startswith("assertion"):
        return AssertionStore(path, read_only=kind.endswith("_ro"))
    return {
        "rollup": RollupStore, "query_view": QueryViewStore,
        "lifecycle": LifecycleStateStore, "vector": VectorStore,
        "store": MessageStore, "dag": SummaryDAG,
    }[kind](path)


def trace_connections(monkeypatch, path, *, fail_statement=None):
    """Record calls without changing the kwargs presented by production code."""
    events = []
    connections = []
    original = sqlite3.connect

    class Connection(sqlite3.Connection):
        def __setattr__(self, name, value):
            if name in ("row_factory", "isolation_level"):
                events.append(["set", name, "Row" if value is sqlite3.Row else value])
            super().__setattr__(name, value)

        def execute(self, sql, parameters=()):
            events.append(["execute", sql, list(parameters)])
            if sql == fail_statement:
                raise sqlite3.OperationalError("recipe failure")
            return super().execute(sql, parameters)

        def executescript(self, sql):
            events.append(["executescript", sql])
            return super().executescript(sql)

        def commit(self):
            events.append(["commit"])
            return super().commit()

        def close(self):
            events.append(["close"])
            return super().close()

    def connect(*args, **kwargs):
        # Bootstrap verifiers also open scratch databases; those are not the
        # store connection whose recipe is being characterized.
        if str(args[0]) == ":memory:":
            return original(*args, **kwargs)
        from urllib.parse import quote

        database = str(args[0]).replace(quote(str(path), safe="/"), "<database>")
        database = database.replace(str(path), "<database>")
        events.append(["connect", [database, *args[1:]], kwargs])
        conn = original(*args, **kwargs, factory=Connection)
        connections.append(conn)
        events.append(["initial", conn.isolation_level, conn.row_factory])
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    return events, connections


@pytest.fixture(autouse=True)
def deterministic_recipe_environment(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    monkeypatch.setattr(db_bootstrap, "sqlite_mmap_size", lambda: 268435456)


@pytest.mark.parametrize("kind", CASES)
def test_constructor_trace(kind, tmp_path, monkeypatch, check_trace):
    path = tmp_path / "recipe #?.db"
    prepare(kind, path)
    events, connections = trace_connections(monkeypatch, path)
    store = construct(kind, path)
    try:
        assert len(connections) == 1
        check_trace(kind, recipe_trace(events))
    finally:
        store.close()


@pytest.mark.parametrize("kind", WRITABLE_CASES)
def test_future_core_version_refused_before_writable_setup(kind, tmp_path, monkeypatch, check_trace):
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO metadata VALUES ('schema_version', '999')")
    conn.commit()
    conn.close()
    before = path.read_bytes()
    events, connections = trace_connections(monkeypatch, path)
    try:
        with pytest.raises(db_bootstrap.SchemaVersionTooNewError) as caught:
            construct(kind, path)
        # Keep the failed constructor alive while inspecting its open path;
        # RollupStore's destructor separately checkpoints and closes it.
        assert "999" in str(caught.value)
        statements = [event[1] for event in events if event[0] == "execute"]
        assert statements
        assert all(sql.lstrip().upper().startswith(("SELECT", "PRAGMA TABLE_INFO"))
                   for sql in statements)
        assert not any(event[0] == "executescript" for event in events)
        assert path.read_bytes() == before
        assert events.count(["close"]) == (1 if kind == "query_view" else 0)
        check_trace(f"future_{kind}", events, caught.value)
    finally:
        for conn in connections:
            conn.close()


@pytest.mark.parametrize("kind", ("trajectory_ro", "assertion_ro"))
def test_read_only_open_preserves_database_bytes(kind, tmp_path):
    path = tmp_path / "read only #?.db"
    prepare(kind, path)
    before = path.read_bytes()
    store = construct(kind, path)
    try:
        assert store.connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert path.read_bytes() == before
    finally:
        store.close()
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", WRITABLE_CASES)
def test_setup_failure_cleanup_and_exception(kind, tmp_path, monkeypatch, check_trace):
    path = tmp_path / "failure.db"
    events, connections = trace_connections(
        monkeypatch, path, fail_statement="PRAGMA busy_timeout=30000",
    )
    try:
        expected = RuntimeError if kind == "query_view" else sqlite3.OperationalError
        with pytest.raises(expected) as caught:
            construct(kind, path)
        prefix = "query-view schema ensure failed: " if kind == "query_view" else ""
        assert str(caught.value) == prefix + "recipe failure"
        assert events.count(["close"]) == (1 if kind == "query_view" else 0)
        check_trace(f"failure_{kind}", events, caught.value)
    finally:
        for conn in connections:
            conn.close()


@pytest.mark.parametrize("kind", ("lifecycle", "vector"))
def test_live_bootstrap_bindings_order_and_connection_lifetime(kind, tmp_path, monkeypatch):
    """Pin I6 refusal ordering, migration/commit timing and caller-owned handles."""
    module, cls = {
        "lifecycle": (lifecycle_state, LifecycleStateStore),
        "vector": (vector_store, VectorStore),
    }[kind]
    path = tmp_path / "lifetime.db"
    store = cls.__new__(cls)
    callbacks = []

    def observe(name, original):
        def call(conn):
            assert store.connection is conn
            assert conn.row_factory is (sqlite3.Row if name == "migrate" else None)
            callbacks.append(name)
            return original(conn)
        return call

    for name, attribute in (
        ("refuse", "refuse_schema_version_too_new"),
        ("configure", "configure_connection"),
        ("migrate", "run_versioned_migrations"),
    ):
        monkeypatch.setattr(module, attribute, observe(name, getattr(module, attribute)))
    if kind == "vector":
        original_schema = cls._ensure_embedding_schema

        def ensure_schema(self):
            assert callbacks == ["refuse", "configure", "migrate"]
            callbacks.append("embedding_schema")
            return original_schema(self)

        monkeypatch.setattr(cls, "_ensure_embedding_schema", ensure_schema)

    events, connections = trace_connections(monkeypatch, path)
    try:
        cls.__init__(store, path)
        assert callbacks == ["refuse", "configure", "migrate"] + (
            ["embedding_schema"] if kind == "vector" else []
        )
        assert events[-1] == ["commit"]
        assert store.connection is connections[0]
        assert store.connection.isolation_level is None
        assert store.connection.row_factory is sqlite3.Row
        store.close()
        assert store.connection is None
        assert events[-2:] == [["execute", "PRAGMA wal_checkpoint(PASSIVE)", []], ["close"]]
        store.close()
        assert events.count(["close"]) == 1
    finally:
        store.close()


@pytest.mark.parametrize("kind", CORE_CASES)
@pytest.mark.parametrize("state", ("fresh", "existing"))
def test_core_schema_write_and_close_trace(kind, state, tmp_path, monkeypatch, check_trace):
    """Pin schema ownership, tuple rows, IMMEDIATE writes and close ordering."""
    module, cls = {"store": (store, MessageStore), "dag": (dag, SummaryDAG)}[kind]
    path = tmp_path / "core.db"
    if state == "existing":
        construct(kind, path).close()
    owner = cls.__new__(cls)
    events, connections = trace_connections(monkeypatch, path)

    def observe(name, original):
        def call(conn, *args):
            assert owner.connection is conn
            assert conn.row_factory is None
            events.append(["helper", name])
            return original(conn, *args)
        return call

    for attribute in (
        "refuse_schema_version_too_new", "configure_connection",
        "ensure_external_content_fts", "run_versioned_migrations",
    ):
        monkeypatch.setattr(module, attribute, observe(attribute, getattr(module, attribute)))
    if kind == "store":
        monkeypatch.setattr(module, "ensure_message_stem_fts", observe(
            "ensure_message_stem_fts", module.ensure_message_stem_fts,
        ))

    def observe_method(name, original):
        def call(self):
            assert self is owner
            events.append(["helper", name])
            return original(self)
        return call

    helpers = ("_ensure_source_column", "_ensure_time_contract_columns") if kind == "store" else (
        "_ensure_source_window_columns",
    )
    for attribute in helpers:
        monkeypatch.setattr(cls, attribute, observe_method(attribute, getattr(cls, attribute)))

    try:
        cls.__init__(owner, path)
        assert len(connections) == 1
        conn = connections[0]
        assert owner.connection is conn
        assert conn.isolation_level == "IMMEDIATE"
        assert conn.row_factory is None
        assert not conn.in_transaction

        def transaction(sql):
            if sql.startswith(("BEGIN", "COMMIT", "ROLLBACK")):
                events.append(["transaction", sql])

        conn.set_trace_callback(transaction)
        events.append(["write"])
        if kind == "store":
            row_id = owner._append_protected_batch(
                "session", [{"role": "user", "content": "recipe write"}],
            )[0]
            assert owner.get(row_id)["content"] == "recipe write"
        else:
            row_id = owner.add_node(SummaryNode(session_id="session", summary="recipe write"))
            assert owner.get_node(row_id).summary == "recipe write"
        assert not conn.in_transaction
        assert [event for event in events if event[0] == "transaction"] == [
            ["transaction", "BEGIN IMMEDIATE"], ["transaction", "COMMIT"],
        ]
        owner.close()
        owner.close()
        assert owner.connection is None
        assert events.count(["close"]) == 1
        assert events[-2:] == [["execute", "PRAGMA wal_checkpoint(PASSIVE)", []], ["close"]]
        trace = [
            event for event in events
            if event[0] in ("helper", "commit", "transaction", "write", "close")
            or event[:2] == ["execute", "PRAGMA wal_checkpoint(PASSIVE)"]
        ]
        check_trace(f"lifetime_{kind}_{state}", trace)
    finally:
        owner.close()
