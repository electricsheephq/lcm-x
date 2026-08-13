"""Restart-safe SQLite publication regressions for LCM-X issue #4."""

from __future__ import annotations

import sqlite3

import hermes_lcm.engine as lcm_engine
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "stable system prompt"},
        {"role": "user", "content": "old user turn"},
        {"role": "assistant", "content": "old assistant turn"},
        {"role": "user", "content": "fresh user turn"},
    ]


def _summary(**_kwargs) -> tuple[str, int]:
    return "Earlier conversation. Expand for details: old turns", 1


def _engine(database_path, identity: str) -> LCMEngine:
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(database_path),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
        )
    )
    engine.on_session_start(
        identity,
        platform="cli",
        conversation_id=identity,
        context_length=200_000,
    )
    return engine


def test_locked_atomic_publication_returns_exact_context_and_restarts_cleanly(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "issue-4-atomic-lock.db"
    identity = "issue-4-atomic-lock"
    messages = _messages()
    engine = _engine(database_path, identity)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)

    def locked_frontier(*_args) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(
        engine._lifecycle,
        "stage_frontier_advance",
        locked_frontier,
        raising=False,
    )

    try:
        result = engine.compress(messages)
        state = engine._lifecycle.get_by_conversation(identity)
        node_count = len(engine._dag.get_session_nodes(identity))
        stored_before_restart = engine._store.get_session_count(identity)
        transaction_state = (
            engine._dag.connection.in_transaction,
            engine._store.connection.in_transaction,
            engine._lifecycle.connection.in_transaction,
        )
    finally:
        engine.shutdown()

    assert result == messages
    assert node_count == 0
    assert state is not None
    assert state.current_frontier_store_id == 0
    assert stored_before_restart == len(messages)
    assert transaction_state == (False, False, False)
    assert engine.last_compression_status == "error"

    restarted = _engine(database_path, identity)
    try:
        restarted._ingest_messages(result)
        stored_after_restart = restarted._store.get_session_count(identity)
    finally:
        restarted.shutdown()

    assert stored_after_restart == stored_before_restart


def test_frontier_failure_rolls_back_insert_and_frontier_together(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "issue-4-atomic-rollback.db"
    identity = "issue-4-atomic-rollback"
    engine = _engine(database_path, identity)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    original_stage = getattr(
        engine._lifecycle,
        "stage_frontier_advance",
        lambda *_args: None,
    )

    def stage_then_fail(*args) -> None:
        original_stage(*args)
        raise RuntimeError("injected post-frontier failure")

    monkeypatch.setattr(
        engine._lifecycle,
        "stage_frontier_advance",
        stage_then_fail,
        raising=False,
    )

    try:
        with pytest.raises(RuntimeError, match="post-frontier failure"):
            engine.compress(_messages())
        state = engine._lifecycle.get_by_conversation(identity)
        node_count = len(engine._dag.get_session_nodes(identity))
    finally:
        engine.shutdown()

    assert node_count == 0
    assert state is not None
    assert state.current_frontier_store_id == 0


def test_post_commit_gc_lock_returns_raw_context_without_restart_reingest(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "issue-4-post-commit-lock.db"
    identity = "issue-4-post-commit-lock"
    messages = _messages()
    engine = _engine(database_path, identity)
    engine._config.threshold_full_sweep_enabled = True
    engine.threshold_tokens = 1
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    monkeypatch.setattr(
        engine,
        "_maybe_gc_compacted_tool_results",
        lambda *_args: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
    )

    try:
        result = engine.compress(messages, current_tokens=100)
        state = engine._lifecycle.get_by_conversation(identity)
        nodes = engine._dag.get_session_nodes(identity)
        stored_before_restart = engine._store.get_session_count(identity)
        telemetry = engine.get_status()["threshold_full_sweep"]
    finally:
        engine.shutdown()

    assert result == messages
    assert len(nodes) == 1
    assert state is not None
    assert state.current_frontier_store_id == max(nodes[0].source_ids)
    assert telemetry["leaf_passes"] == 1
    assert telemetry["total_passes"] == 1

    restarted = _engine(database_path, identity)
    try:
        restarted._ingest_messages(result)
        stored_after_restart = restarted._store.get_session_count(identity)
    finally:
        restarted.shutdown()

    assert stored_after_restart == stored_before_restart


def test_condensation_lock_returns_raw_context_with_committed_leaf(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "issue-4-condensation-lock.db"
    identity = "issue-4-condensation-lock"
    messages = _messages()
    engine = _engine(database_path, identity)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    monkeypatch.setattr(
        engine,
        "_maybe_condense",
        lambda **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
    )

    try:
        result = engine.compress(messages)
        state = engine._lifecycle.get_by_conversation(identity)
        nodes = engine._dag.get_session_nodes(identity)
    finally:
        engine.shutdown()

    assert result == messages
    assert len(nodes) == 1
    assert state is not None
    assert state.current_frontier_store_id == max(nodes[0].source_ids)
    assert engine.last_compression_status == "error"
