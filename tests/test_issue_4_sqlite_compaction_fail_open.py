"""SQLite publication fail-open regressions for LCM-X issue #4."""

from __future__ import annotations

import sqlite3

import hermes_lcm.engine as lcm_engine
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.sqlite_util import _run_sqlite_write_with_snapshot_retry
from hermes_lcm.store import MessageStore


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "stable system prompt"},
        {"role": "user", "content": "old user turn"},
        {"role": "assistant", "content": "old assistant turn"},
        {"role": "user", "content": "fresh user turn"},
    ]


def _summary(**_kwargs) -> tuple[str, int]:
    return "Earlier conversation. Expand for details: old turns", 1


def test_locked_dag_publication_returns_precompression_context(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    database_path = tmp_path / "issue-4-publication-lock.db"
    config = LCMConfig(
        database_path=str(database_path),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-4-publication-lock",
        platform="cli",
        conversation_id="issue-4-publication-lock",
        context_length=200_000,
    )
    messages = _messages()
    locker = sqlite3.connect(
        str(database_path),
        timeout=1.0,
        isolation_level=None,
    )
    locker.execute("PRAGMA journal_mode=WAL")
    engine._dag.connection.execute("PRAGMA busy_timeout=1")

    def summarize_and_lock(**_kwargs) -> tuple[str, int]:
        locker.execute("BEGIN IMMEDIATE")
        return "Earlier conversation. Expand for details: old turns", 1

    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        summarize_and_lock,
    )

    try:
        state_before = engine._lifecycle.get_by_conversation(
            engine._conversation_id
        )
        nodes_before = engine._dag.get_session_nodes(engine._session_id)
        with caplog.at_level("WARNING"):
            result = engine.compress(messages)
        state_after = engine._lifecycle.get_by_conversation(
            engine._conversation_id
        )
        nodes_after = engine._dag.get_session_nodes(engine._session_id)
        dag_in_transaction = engine._dag.connection.in_transaction
    finally:
        if locker.in_transaction:
            locker.execute("ROLLBACK")
        locker.close()
        engine.shutdown()

    assert result == messages
    assert nodes_after == nodes_before
    assert state_before is not None
    assert state_after is not None
    assert (
        state_after.current_frontier_store_id
        == state_before.current_frontier_store_id
    )
    assert engine.last_compression_status == "error"
    assert dag_in_transaction is False
    assert "preserving pre-compression context" in caplog.text


def test_locked_frontier_preserves_context_and_restores_runtime_frontier(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        database_path=str(tmp_path / "issue-4-frontier-lock.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-4-frontier-lock",
        platform="cli",
        conversation_id="issue-4-frontier-lock",
        context_length=200_000,
    )
    messages = _messages()
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)

    def locked_frontier() -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine, "_persist_frontier_marker", locked_frontier)

    try:
        result = engine.compress(messages)
        state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        nodes = engine._dag.get_session_nodes(engine._session_id)
        runtime_frontier = engine._last_compacted_store_id
    finally:
        engine.shutdown()

    assert result == messages
    assert len(nodes) == 1
    assert state is not None
    assert state.current_frontier_store_id == 0
    assert runtime_frontier == 0
    assert engine.last_compression_status == "error"


def test_non_lock_publication_failure_still_raises(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        database_path=str(tmp_path / "issue-4-non-lock.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-4-non-lock",
        platform="cli",
        conversation_id="issue-4-non-lock",
        context_length=200_000,
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    def fail_publication(_node) -> None:
        raise RuntimeError("schema failure")

    monkeypatch.setattr(engine._dag, "add_node", fail_publication)

    try:
        with pytest.raises(RuntimeError, match="schema failure"):
            engine.compress(_messages())
    finally:
        engine.shutdown()

    assert engine.last_compression_status == "error"


def test_locked_condensation_publication_returns_precompression_context(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        database_path=str(tmp_path / "issue-4-condensation-lock.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-4-condensation-lock",
        platform="cli",
        conversation_id="issue-4-condensation-lock",
        context_length=200_000,
    )
    messages = _messages()
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)

    def locked_condensation(**_kwargs) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine, "_maybe_condense", locked_condensation)

    try:
        result = engine.compress(messages)
        nodes = engine._dag.get_session_nodes(engine._session_id)
    finally:
        engine.shutdown()

    assert result == messages
    assert len(nodes) == 1
    assert engine.last_compression_status == "error"


def test_dag_write_retries_one_stale_snapshot(tmp_path) -> None:
    database_path = tmp_path / "issue-4-dag-snapshot.db"
    dag = SummaryDAG(database_path)
    writer = sqlite3.connect(database_path)

    try:
        dag.connection.execute("BEGIN")
        dag.connection.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()
        writer.execute(
            "INSERT OR REPLACE INTO metadata(key, value) "
            "VALUES('issue-4-writer', 'advanced')"
        )
        writer.commit()

        node_id = dag.add_node(
            SummaryNode(
                session_id="issue-4-dag-snapshot",
                depth=0,
                summary="retry from a fresh transaction",
                token_count=5,
                source_ids=[1],
                source_type="messages",
            )
        )
        stored = dag.get_node(node_id)
        in_transaction = dag.connection.in_transaction
    finally:
        writer.close()
        dag.close()

    assert node_id > 0
    assert stored is not None
    assert stored.summary == "retry from a fresh transaction"
    assert in_transaction is False


def test_message_batch_retries_stale_snapshot_with_metadata(tmp_path) -> None:
    database_path = tmp_path / "issue-4-store-snapshot.db"
    store = MessageStore(database_path)
    writer = sqlite3.connect(database_path)

    try:
        store.connection.execute("BEGIN")
        store.connection.execute("SELECT COUNT(*) FROM messages").fetchone()
        writer.execute(
            "INSERT OR REPLACE INTO metadata(key, value) "
            "VALUES('issue-4-writer', 'advanced')"
        )
        writer.commit()

        ids = store._append_protected_batch(
            "issue-4-store-snapshot",
            [{"role": "user", "content": "retry with provenance"}],
            metadata_factory=lambda _message, store_id: [
                (f"issue-4-provenance:{store_id}", '"real-user"')
            ],
        )
        metadata = store.read_metadata_json(
            f"issue-4-provenance:{ids[0]}"
        )
        stored = store.get(ids[0])
        in_transaction = store.connection.in_transaction
    finally:
        writer.close()
        store.close()

    assert stored is not None
    assert stored["content"] == "retry with provenance"
    assert metadata == "real-user"
    assert in_transaction is False


def test_ordinary_busy_uses_one_bounded_attempt(tmp_path) -> None:
    database_path = tmp_path / "issue-4-ordinary-busy.db"
    victim = sqlite3.connect(database_path)
    holder = sqlite3.connect(database_path)
    victim.execute("PRAGMA journal_mode=WAL")
    victim.execute("PRAGMA busy_timeout=1")
    holder.execute("PRAGMA busy_timeout=1")
    victim.execute(
        "CREATE TABLE IF NOT EXISTS issue_4_busy(key TEXT PRIMARY KEY)"
    )
    victim.commit()
    holder.execute("BEGIN IMMEDIATE")
    attempts = 0

    def blocked_write() -> None:
        nonlocal attempts
        attempts += 1
        victim.execute(
            "INSERT INTO issue_4_busy(key) VALUES('blocked')"
        )

    try:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            _run_sqlite_write_with_snapshot_retry(
                victim,
                blocked_write,
                operation_name="issue-4.ordinary-busy",
            )
        in_transaction = victim.in_transaction
    finally:
        if holder.in_transaction:
            holder.execute("ROLLBACK")
        holder.close()
        victim.close()

    assert attempts == 1
    assert in_transaction is False
