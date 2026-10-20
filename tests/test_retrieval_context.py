"""Pin read-clone ownership and retrieval results before grouping the bindings."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import importlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

tools = importlib.import_module("hermes_lcm.tools")
MessageStore = importlib.import_module("hermes_lcm.store").MessageStore
dag_module = importlib.import_module("hermes_lcm.dag")
LCMConfig = importlib.import_module("hermes_lcm.config").LCMConfig


@pytest.fixture
def retrieval_engine(tmp_path: Path) -> Iterator[SimpleNamespace]:
    config = LCMConfig(database_path=str(tmp_path / "retrieval.db"), embeddings_enabled=False)
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = dag_module.SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config, _store=store, _dag=dag, current_session_id="session-current",
    )
    try:
        ids = [
            store.append(session, {"role": "user", "content": content, "timestamp": stamp})
            for session, content, stamp in (
                ("session-current", "needle first message", 10.0),
                ("session-current", "needle second message", 20.0),
                ("session-archive", "needle archived message", 30.0),
            )
        ]
        node = dag_module.SummaryNode(
            session_id="session-current", summary="needle summary", source_ids=ids[:2],
            source_type="messages", created_at=20.0, token_count=3,
        )
        node.node_id = dag.add_node(node)
        engine.node = node
        engine.ids = ids
        yield engine
    finally:
        dag.close()
        store.close()


@pytest.fixture
def inline_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the worker body without scheduling or wall-clock assertions.
    monkeypatch.setattr(tools.time, "monotonic", lambda: 100.0)

    def inline(invoke: Callable[[], Any], **kwargs: Any) -> Any:
        return invoke()

    monkeypatch.setattr(tools, "_run_within_deadline", inline)


def _retrieve(
    engine: SimpleNamespace, path: str, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, Any]:
    stores: list[Any] = []
    dags: list[Any] = []
    if path == "grep":
        original = tools._lcm_grep_full_text

        def capture(args: dict[str, Any], **kwargs: Any) -> str:
            reader = kwargs["engine"]
            assert reader is not engine
            assert reader._config is engine._config
            stores.append(reader._store)
            dags.append(reader._dag)
            return str(original(args, **kwargs))

        monkeypatch.setattr(tools, "_lcm_grep_full_text", capture)
        result = tools._lcm_grep_full_text_with_deadline(
            {"query": "needle"}, engine=engine, deadline=101.0,
        )
        assert result["results"]
    else:
        original_batch = MessageStore.get_batch
        original_sources = dag_module.SummaryDAG.source_message_ids

        def batch(store: Any, ids: list[int]) -> Any:
            stores.append(store)
            return original_batch(store, ids)

        def sources(dag: Any, node_id: int, **kwargs: Any) -> Any:
            dags.append(dag)
            return original_sources(dag, node_id, **kwargs)

        monkeypatch.setattr(MessageStore, "get_batch", batch)
        monkeypatch.setattr(dag_module.SummaryDAG, "source_message_ids", sources)
        result = tools._lcm_recall_summary_source_hits(
            engine, [(engine.node, 1.0)], current=engine.current_session_id,
            candidate_limit=5, lead_limit=1, deadline=101.0,
        )
        hits, leads = result
        assert [hit["store_id"] for hit in hits] == engine.ids[:2]
        assert [hit["snippet"] for hit in hits] == ["needle first message", "needle second message"]
        assert leads[0]["node_id"] == engine.node.node_id
    assert len(stores) == len(dags) == 1
    return stores[0], dags[0], result


@pytest.mark.parametrize("path", ["grep", "summary"])
def test_read_clones_own_the_actual_store_and_dag_locks(
    retrieval_engine: SimpleNamespace, inline_deadline: None,
    monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    engine = retrieval_engine
    store_lock, dag_lock = engine._store._write_lock, engine._dag._db_lock
    store, dag, _ = _retrieve(engine, path, monkeypatch)
    assert store is not engine._store and dag is not engine._dag
    assert dag._db_lock is not dag_lock
    assert store._write_lock is not dag._db_lock
    assert engine._store._write_lock is store_lock
    assert engine._dag._db_lock is dag_lock
    assert store._write_lock is not store_lock


@pytest.mark.parametrize("path", ["grep", "summary"])
def test_read_connections_close_without_rebinding_live_connections(
    retrieval_engine: SimpleNamespace, inline_deadline: None,
    monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    engine = retrieval_engine
    live_store, live_dag = engine._store._conn, engine._dag._conn
    store, dag, _ = _retrieve(engine, path, monkeypatch)
    assert store._conn is not live_store and dag._conn is not live_dag
    assert (store._conn is dag._conn) == (path == "summary")
    for conn in (store._conn, dag._conn):
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")
    assert engine._store._conn is live_store and engine._dag._conn is live_dag
    assert live_store.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
    assert live_dag.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 1


@pytest.mark.parametrize("scope", ["current", "all", "session"])
@pytest.mark.parametrize("query", ["needle", "missing"])
def test_deadline_grep_preserves_full_text_results(
    retrieval_engine: SimpleNamespace, inline_deadline: None, scope: str, query: str,
) -> None:
    args = {"query": query, "session_scope": scope}
    if scope == "session":
        args["session_id"] = "session-archive"
    expected = json.loads(tools._lcm_grep_full_text(args, engine=retrieval_engine))
    actual = tools._lcm_grep_full_text_with_deadline(
        args, engine=retrieval_engine, deadline=101.0,
    )
    assert actual == expected
    assert bool(actual["results"]) == (query == "needle")


@pytest.mark.parametrize("path", ["grep", "summary"])
def test_read_bindings_remain_query_only_on_search_failure(
    retrieval_engine: SimpleNamespace, inline_deadline: None,
    monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    connections: list[sqlite3.Connection] = []

    def fail(reader: Any) -> None:
        for conn in (reader._store._conn, reader._dag._conn):
            connections.append(conn)
            assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("DELETE FROM messages")
        raise RuntimeError("synthetic retrieval failure")

    if path == "grep":
        def fail_grep(args: dict[str, Any], **kwargs: Any) -> str:
            fail(kwargs["engine"])
            raise AssertionError("unreachable")

        monkeypatch.setattr(tools, "_lcm_grep_full_text", fail_grep)
        result = tools._lcm_grep_full_text_with_deadline(
            {"query": "needle"}, engine=retrieval_engine, deadline=101.0,
        )
        assert result["error"] == "full-text fallback failed: synthetic retrieval failure"
    else:
        def fail_sources(dag: Any, node_id: int, **kwargs: Any) -> list[int]:
            fail(SimpleNamespace(_store=dag, _dag=dag))
            raise AssertionError("unreachable")

        monkeypatch.setattr(dag_module.SummaryDAG, "source_message_ids", fail_sources)
        with pytest.raises(RuntimeError, match="synthetic retrieval failure"):
            tools._lcm_recall_summary_source_hits(
                retrieval_engine, [(retrieval_engine.node, 1.0)],
                current=retrieval_engine.current_session_id,
                candidate_limit=5, lead_limit=1, deadline=101.0,
            )
    assert connections
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")
    assert retrieval_engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
