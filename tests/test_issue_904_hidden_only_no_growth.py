"""#904: publish hidden leaves without growing the host view or spending its turn budget."""

from copy import deepcopy
import hashlib
import json
import logging
import re
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: len(text) // 4 if text else 0)
    summarize = Mock(return_value=("Earlier turns.\nExpand for details about: turns", 1))
    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), context_threshold=0.001,
        fresh_tail_count=8, fresh_tail_max_tokens=0,
        leaf_chunk_tokens=1000, dynamic_leaf_chunk_max=1000,
        dynamic_leaf_chunk_enabled=False, threshold_full_sweep_enabled=False,
        incremental_max_depth=0, l3_truncate_tokens=50,
        fresh_tail_pressure_yield_enabled=False, embeddings_enabled=False,
        temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
    ), hermes_home=str(tmp_path))
    instance.on_session_start("issue-904", conversation_id="hidden-only", context_length=128_000)
    try:
        yield instance, summarize
    finally:
        instance.shutdown()


def _history(turns=30):
    return [row for number in range(turns) for row in (
        {"role": "user", "content": f"USER-{number}: " + "u" * 1200},
        {"role": "assistant", "content": f"ANSWER-{number}: " + "a" * 1200},
    )]


def _hidden_view(instance):
    history = _history()
    instance._ingest_messages(history)
    host = history[-8:]
    assert instance._fresh_tail_start(host) == 0
    return host


def _compress(instance, host):
    return instance.compress(host, current_tokens=count_messages_tokens(host))


def test_hidden_only_returns_identical_list_and_holds_attempts(engine, caplog):
    instance, summarize = engine
    host = _hidden_view(instance)
    original = deepcopy(host)
    frontier = instance._store_complete_frontier()
    with caplog.at_level(logging.INFO):
        result = _compress(instance, host)

    assert result is host
    assert result == original
    nodes = instance._dag.get_session_nodes(instance._session_id)
    assert len(nodes) == 1 and nodes[0].depth == 0
    assert summarize.call_count == 1
    assert instance._store_complete_frontier() > frontier
    state = instance._lifecycle.get_by_conversation(instance._conversation_id)
    assert state.current_frontier_store_id == max(nodes[0].source_ids)
    assert instance._store_complete_backlog(host, 0).rows > len(nodes[0].source_ids)
    assert instance.get_status()["no_progress_hold"]["reason"] == "hidden_only"
    unchanged_logs = [record for record in caplog.records if "hidden-only" in record.getMessage()]
    assert len(unchanged_logs) == 1
    assert unchanged_logs[0].levelno == logging.INFO
    assert "leaves=1" in unchanged_logs[0].getMessage()
    assert "hidden_rows=" in unchanged_logs[0].getMessage()
    assert "USER-" not in caplog.text and "ANSWER-" not in caplog.text
    for observed in (count_messages_tokens(host), instance.threshold_tokens, instance._survival_ceiling() - 1):
        assert not instance.should_compress(observed)


def test_hidden_only_repairs_missing_tool_result_without_changing_stored_rows(engine):
    instance, summarize = engine
    history = _history()
    history[-3]["tool_calls"] = [{
        "id": "hidden-only-missing-result", "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }]
    instance._ingest_messages(history)
    host = history[-8:]
    original = deepcopy(host)
    assert instance._fresh_tail_start(host) == 0
    assert instance._store_complete_backlog(host, 0)
    stored_before = instance._store.get_session_messages(instance._session_id)

    result = _compress(instance, host)

    nodes = instance._dag.get_session_nodes(instance._session_id)
    assert len(nodes) == 1 and nodes[0].depth == 0
    assert summarize.call_count == 1
    assert instance._compress_host_rows_consumed == 0
    assert instance._compress_hidden_rows_consumed == len(nodes[0].source_ids) > 0
    assert instance.get_status()["no_progress_hold"]["reason"] == "hidden_only"
    stub = instance._missing_tool_result_stub("hidden-only-missing-result")
    assert result == original[:6] + [stub] + original[6:]
    assert result[5]["tool_calls"] == original[5]["tool_calls"]
    assert instance._ingest_cursor == len(result)
    assert host == original
    assert instance._store.get_session_messages(instance._session_id) == stored_before


@pytest.mark.parametrize("options", [{"force": True}, {"bypass_cooldown": True}], ids=["forced", "bypass"])
def test_hidden_only_nonautomatic_pass_returns_input_without_hold(engine, options):
    instance, summarize = engine
    host = _hidden_view(instance)
    original = deepcopy(host)
    frontier = instance._store_complete_frontier()

    # Isolate F1's non-shrinking path: recovery survival fitting would otherwise
    # shrink this fixture before the hidden-only guard can run.
    instance._config.survival_fit = False
    result = instance.compress(host, current_tokens=count_messages_tokens(host), **options)

    assert result is host
    assert result == original
    assert summarize.call_count >= 1
    assert instance._store_complete_frontier() > frontier
    assert instance._no_progress_hold is None
    assert not instance._no_progress_hold_active()


def test_turn_end_allows_drain_then_host_consumption(engine):
    instance, _ = engine
    host = _hidden_view(instance)
    original_count = instance._store.get_session_count(instance._session_id)
    previous_frontier = 0
    for _ in range(60):  # finite synthetic backlog; exactly one compaction invocation per turn
        result = _compress(instance, host)
        assert result is host
        assert instance._store_complete_frontier() > previous_frontier
        previous_frontier = instance._store_complete_frontier()
        assert instance._no_progress_hold[1] == "hidden_only"
        instance._mark_thread_context_stateless("aux-904")
        try:
            instance.note_turn_complete()
            assert instance._no_progress_hold[1] == "hidden_only"
        finally:
            instance._clear_thread_context_stateless()
        instance.note_turn_complete()
        assert instance._no_progress_hold is None
        assert instance.should_compress(count_messages_tokens(host))
        if not instance._store_complete_backlog(host, 0):
            break
    else:
        pytest.fail("hidden backlog did not drain")
    assert instance._store.get_session_count(instance._session_id) == original_count
    # A later ordinary turn moves the oldest host rows outside the fresh tail.
    following = host + _history(34)[60:]
    result = _compress(instance, following)
    assert len(result) < len(following)
    assert count_messages_tokens(result) < count_messages_tokens(following)
    assert instance._store_complete_frontier() > previous_frontier
    assert instance._no_progress_hold is None


@pytest.mark.parametrize("host_start", [24, 52], ids=["raw-prefix", "all-fresh-tail"])
def test_two_pass_carrier_has_no_duplicate_node_in_adoptable_output(engine, host_start):
    """Q1: stored hidden rows precede a host carrier containing the previous summary."""
    instance, _ = engine
    history = _history()
    first = _compress(instance, history)
    assert count_messages_tokens(first) < count_messages_tokens(history)
    # Model the host's narrower view after a native fit. Assembly supplies the
    # existing carrier; the omitted stored rows above the frontier remain hidden.
    host = instance._assemble_context(None, history[host_start:])
    assert instance._generated_context_carrier_remainder(host[0]) is not None
    assert instance._store_complete_backlog(host, 0)
    previous_nodes = len(instance._dag.get_session_nodes(instance._session_id))
    for _ in range(2):
        before = deepcopy(host)
        result = _compress(instance, host)
        assert count_messages_tokens(result) <= count_messages_tokens(before)
        node_ids = re.findall(r"Summary \(d\d+, node (\d+)\)", json.dumps(result))
        assert node_ids and len(node_ids) == len(set(node_ids))
        assert len(instance._dag.get_session_nodes(instance._session_id)) > previous_nodes
        previous_nodes = len(instance._dag.get_session_nodes(instance._session_id))
        instance.note_turn_complete()
        host = result


def test_host_consuming_pass_is_byte_identical_to_922_base(engine):
    instance, _ = engine
    host = _history(8)
    original = deepcopy(host)
    result = _compress(instance, host)
    assert host == original
    assert len(result) < len(host)
    assert count_messages_tokens(result) < count_messages_tokens(host)
    assert instance._store_complete_frontier() > 0
    assert instance._no_progress_hold is None
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    # Captured on #922's exact base, 72044e2c, before implementing D2.
    assert hashlib.sha256(encoded).hexdigest() == "712fda3dfa33c9fb71bda594b131ecf93adb4e5d635c6a16647bc5f526f17b51"
