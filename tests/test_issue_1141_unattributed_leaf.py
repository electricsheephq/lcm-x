"""#1141: a leaf whose host rows carry no durable source above the frontier must not reach publication.

Red on v0.26.2 and v0.27.0-rc4 (09e32580): each case pays a summariser call, then publication raises
"Compaction publication has no durable source coverage" and the next pass repeats it. Green: the pass is a
no-op ("selected leaf chunk lacks raw store lineage") before any summariser call.
"""
import logging
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

CONFLICT = "Compaction publication has no durable source coverage"
NOOP = "selected leaf chunk lacks raw store lineage"
WARNING = "LCM leaf chunk has no durable source above the frontier:"


@pytest.fixture
def summarize(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: len(text) // 4 if text else 0)
    fake = Mock(return_value=("Earlier turns.\nExpand for details about: turns", 1))
    monkeypatch.setattr(engine_module, "summarize_with_escalation", fake)
    return fake


def _config(tmp_path, **extra):
    return LCMConfig(**{
        "database_path": str(tmp_path / "lcm.db"), "context_threshold": 0.001,
        "fresh_tail_count": 8, "fresh_tail_max_tokens": 0, "leaf_chunk_tokens": 1000,
        "dynamic_leaf_chunk_max": 1000, "dynamic_leaf_chunk_enabled": False,
        "threshold_full_sweep_enabled": False, "incremental_max_depth": 0, "l3_truncate_tokens": 50,
        "fresh_tail_pressure_yield_enabled": False, "embeddings_enabled": False,
        "temporal_rollups_enabled": False, "empty_lifecycle_gc_enabled": False, **extra})


def _engine(tmp_path, **extra):
    engine = LCMEngine(config=_config(tmp_path, **extra), hermes_home=str(tmp_path))
    engine.on_session_start("issue-1141", conversation_id="conv-1141", context_length=128_000)
    return engine


def _history(turns=30, tag="T"):
    return [row for n in range(turns) for row in (
        {"role": "user", "content": f"{tag}-USER-{n}: " + "u" * 1200},
        {"role": "assistant", "content": f"{tag}-ANSWER-{n}: " + "a" * 1200})]


def _assert_refused_before_call(engine, summarize, calls_before, caplog):
    assert CONFLICT not in caplog.text
    assert summarize.call_count == calls_before  # no paid call for a leaf that cannot publish
    assert engine._last_compression_status in {"noop", "sanitized"}  # never "error"
    assert engine._last_compression_noop_reason == NOOP
    assert engine._no_progress_hold_active()
    assert engine._no_progress_hold[1] == "no_progress"


def _two_passes(engine, view, summarize, caplog):
    frontier = engine._store_complete_frontier()
    rows = engine._store.get_session_messages(engine._session_id)
    calls = summarize.call_count
    extraction = Mock()
    engine._run_pre_compaction_extraction = extraction
    engine._schedule_pre_compaction_assertions = extraction
    engine._config.extraction_enabled = engine._config.assertion_extraction_enabled = True
    for _ in range(2):
        engine._no_progress_hold = None  # exercise another pass after the hold expires
        with caplog.at_level(logging.INFO):
            result = engine.compress(list(view), current_tokens=count_messages_tokens(view))
        _assert_refused_before_call(engine, summarize, calls, caplog)
        # Existing committed summaries may be reassembled; no raw host row is removed or changed.
        assert [row for row in result if not engine._is_replayed_context_scaffold_message(row)] == [
            row for row in view if not engine._is_replayed_context_scaffold_message(row)
        ]
        assert engine._store_complete_frontier() == frontier
        assert engine._store.get_session_messages(engine._session_id) == rows
    extraction.assert_not_called()
    warnings = [record.getMessage() for record in caplog.records if record.getMessage().startswith(WARNING)]
    assert len(warnings) == 1
    assert f"frontier={frontier} rows=" in warnings[0]
    full_map = engine._current_compress_store_ids_by_message_id or {}
    first = min((sid for sid in full_map.values() if sid > frontier), default="none")
    assert warnings[0].endswith(f"first_mapped_store_id={first}")


def test_rotate_apply_then_host_resumed_transcript(tmp_path, summarize, caplog):
    """In-product path: `/lcm rotate apply`, a restart, and a host that resumes its own transcript."""
    engine = _engine(tmp_path)
    history = _history()
    engine._ingest_messages(history)
    assert engine.rotate_active_session(apply=True)["ok"]
    engine.shutdown()
    engine = _engine(tmp_path)
    try:
        _two_passes(engine, history, summarize, caplog)
    finally:
        engine.shutdown()


def test_published_leaf_not_adopted_with_hidden_copies_after_frontier(tmp_path, summarize, caplog):
    """Natural path: the host keeps rows a published leaf consumed, and hidden re-ingest copies right above
    the frontier defeat the #457 replay drop (its F+1 check). rc4: pass 0 publishes the copies, pass 1 raises;
    with #1124 (copies excluded) pass 0 raises."""
    engine = _engine(tmp_path, fresh_tail_count=2, leaf_chunk_tokens=400, dynamic_leaf_chunk_max=400)
    real = engine._lifecycle.stage_compaction_publication

    def publish(rows, expected):
        ids = [engine._store.append(engine._session_id, row, conversation_id="conv-1141") for row in rows]
        engine._dag.add_node(SummaryNode(session_id=engine._session_id, summary="Earlier", token_count=2,
                                         source_token_count=20, source_ids=ids, source_type="messages"),
                             before_commit=lambda conn, node_id: real(conn, "conv-1141", engine._session_id,
                                                                      node_id, expected, ids))
        return ids

    try:
        call = {"id": "call_c", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        covered = [{"role": "user", "content": "C " + "c" * 600},
                   {"role": "assistant", "content": "", "tool_calls": [call]},
                   {"role": "tool", "tool_call_id": "call_c", "content": "result " + "r" * 600}]
        c_ids = publish(covered, 0)
        stale = _history(2, "A")
        a_ids = publish(stale, c_ids[-1])
        engine._last_compacted_store_id = a_ids[-1]
        for row in covered[1:]:  # hidden byte-equal re-ingest copies of covered rows
            engine._store.append(engine._session_id, dict(row), conversation_id="conv-1141")
        later = _history(6, "B")
        for row in later:
            engine._store.append(engine._session_id, row, conversation_id="conv-1141")
        view = [*stale, *later]
        engine._ingest_cursor = len(view)
        # The first pass can publish the hidden copies; the next selected leaf has no source.
        view = engine.compress(view, current_tokens=count_messages_tokens(view))
        assert summarize.call_count == 1
        caplog.clear()
        _two_passes(engine, view, summarize, caplog)
    finally:
        engine.shutdown()


def test_oldest_prefix_rescue_loses_all_durable_source(tmp_path, summarize, caplog):
    """A mixed stale/durable leaf passes admission, but the real rescue keeps only stale rows."""
    engine = _engine(tmp_path, leaf_chunk_tokens=20_000, dynamic_leaf_chunk_max=20_000)
    history = _history()
    engine._ingest_messages(history)
    assert engine.rotate_active_session(apply=True)["ok"]
    engine.shutdown()
    engine = _engine(tmp_path, leaf_chunk_tokens=20_000, dynamic_leaf_chunk_max=20_000)
    try:
        view = [*history, *_history(6, "B")]
        engine.ingest(view)
        frontier = engine._store_complete_frontier()
        nodes = engine._dag.get_session_nodes(engine._session_id)
        rows = engine._store.get_session_messages(engine._session_id)
        summarize.side_effect = [RuntimeError("context length exceeded"), summarize.return_value]
        with caplog.at_level(logging.INFO):
            result = engine.compress(view, current_tokens=count_messages_tokens(view))
        assert summarize.call_count == 1
        assert "retrying with smaller oldest chunk" not in caplog.text
        assert sum(record.levelno == logging.WARNING and record.getMessage().startswith(WARNING)
                   for record in caplog.records) == 1
        assert CONFLICT not in caplog.text
        assert engine._last_compression_status in {"noop", "sanitized"}
        assert engine._last_compression_noop_reason == NOOP
        assert result == view
        assert engine._store_complete_frontier() == frontier
        assert engine._dag.get_session_nodes(engine._session_id) == nodes
        assert engine._store.get_session_messages(engine._session_id) == rows
        engine._no_progress_hold = None  # exercise another pass after the hold expires
        summarize.side_effect = [RuntimeError("context length exceeded"), summarize.return_value]
        with caplog.at_level(logging.INFO):
            result = engine.compress(view, current_tokens=count_messages_tokens(view))
        assert summarize.call_count == 2
        assert "retrying with smaller oldest chunk" not in caplog.text
        assert sum(record.levelno == logging.WARNING and record.getMessage().startswith(WARNING)
                   for record in caplog.records) == 1
        assert result == view
        assert engine._store_complete_frontier() == frontier
        assert engine._dag.get_session_nodes(engine._session_id) == nodes
        assert engine._store.get_session_messages(engine._session_id) == rows
    finally:
        engine.shutdown()


def test_oldest_prefix_rescue_keeps_durable_source_and_publishes(tmp_path, summarize, caplog):
    """#1147 r1: a mixed leaf still retries when its oldest prefix retains durable rows."""
    engine = _engine(tmp_path, leaf_chunk_tokens=20_000, dynamic_leaf_chunk_max=20_000)
    history = _history()
    engine._ingest_messages(history)
    assert engine.rotate_active_session(apply=True)["ok"]
    engine.shutdown()
    engine = _engine(tmp_path, leaf_chunk_tokens=20_000, dynamic_leaf_chunk_max=20_000)
    try:
        view = [*_history(6, "B"), *history]
        engine.ingest(view)
        frontier = engine._store_complete_frontier()
        rows = engine._store.get_session_messages(engine._session_id)
        summarize.side_effect = [RuntimeError("context length exceeded"), summarize.return_value]
        with caplog.at_level(logging.INFO):
            result = engine.compress(view, current_tokens=count_messages_tokens(view))
        assert summarize.call_count == 2
        assert "retrying with smaller oldest chunk" in caplog.text
        assert WARNING not in caplog.text
        assert CONFLICT not in caplog.text
        assert result != view
        assert engine._store_complete_frontier() > frontier
        assert engine._dag.get_session_nodes(engine._session_id)
        assert engine._store.get_session_messages(engine._session_id) == rows
    finally:
        engine.shutdown()
