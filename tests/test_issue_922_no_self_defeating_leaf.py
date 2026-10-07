"""#922: do not summarize the sole prompt that assembly would repeat verbatim."""

from copy import deepcopy
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: len(text) // 4 if text else 0)
    summarize = Mock(return_value=("Earlier tool work.\nExpand for details about: tool results", 1))
    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), context_threshold=0.12,
        fresh_tail_count=4, fresh_tail_max_tokens=2000,
        threshold_full_sweep_enabled=True, fresh_tail_pressure_yield_enabled=False,
        embeddings_enabled=False, temporal_rollups_enabled=False,
        empty_lifecycle_gc_enabled=False,
    ), hermes_home=str(tmp_path))
    instance.on_session_start("issue-922", conversation_id="sole-user", context_length=128_000)
    try:
        yield instance, summarize
    finally:
        instance.shutdown()


def _round(number):
    call_id = f"call-{number}"
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "read_file", "arguments": '{"file":"synthetic.txt"}'},
        }]},
        {"role": "tool", "tool_call_id": call_id,
         "content": f"TOOL-ROUND-{number}: " + "x" * 80_000},
    ]


def _cell():
    return [{"role": "user", "content": "SOLE-OBJECTIVE: " + "p" * 18_400}, *_round(1)]


def _host_anti_growth(engine, messages):
    """Unit analogue of the host guard; never run a Hermes host."""
    result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
    if count_messages_tokens(result) > count_messages_tokens(messages):
        engine.record_rejected_compaction()
    return result


def test_cell_no_leaf_no_growth_no_rejection(engine, monkeypatch):
    instance, summarize = engine
    messages = _cell()
    original = deepcopy(messages)
    assert 4600 <= count_messages_tokens(messages[:1]) < 4700
    assert 20_000 <= count_messages_tokens(messages[-1:]) < 20_100
    assert instance.threshold_tokens == 15_360
    assert instance._fresh_tail_start(messages) == 1
    assert instance._effective_assembly_token_cap() is None
    rejected = Mock(wraps=instance.record_rejected_compaction)
    monkeypatch.setattr(instance, "record_rejected_compaction", rejected)

    result = _host_anti_growth(instance, messages)

    assert instance._dag.get_session_nodes(instance._session_id) == []
    assert instance._objective_only_noop
    summarize.assert_not_called()
    assert result == original
    assert count_messages_tokens(result) <= count_messages_tokens(original)
    rejected.assert_not_called()
    assert instance.get_status()["no_progress_hold"]["reason"] == "objective_only"
    assert instance._compression_block_reason() == "cooldown:lcm_objective_only"
    assert not instance.should_compress(count_messages_tokens(result))
    assert instance._store.get_session_count(instance._session_id) == 3


def test_objective_only_writes_leaf_when_anchor_exceeds_assembly_cap(engine):
    instance, summarize = engine
    messages = _cell()
    summary = "SOLE-OBJECTIVE: continue the requested work.\nExpand for details about: user's request"
    summarize.return_value = (summary, 1)
    fresh_tail_start = instance._fresh_tail_start(messages)
    assert fresh_tail_start == 1
    tail_tokens = count_messages_tokens(messages[fresh_tail_start:])
    instance._config.max_assembly_tokens = tail_tokens + 1000
    cap = instance._effective_assembly_token_cap()
    anchor = instance._build_preserved_objective_summary_part(messages[0])
    assert tail_tokens + count_message_tokens({"role": "user", "content": summary}) <= cap
    assert tail_tokens + count_message_tokens({"role": "user", "content": anchor}) > cap

    result = instance.compress(messages, current_tokens=count_messages_tokens(messages))

    assert not instance._objective_only_noop
    nodes = instance._dag.get_session_nodes(instance._session_id)
    assert nodes and summarize.called
    assert 1 in {store_id for node in nodes for store_id in node.source_ids}
    assert any(summary in (message.get("content") or "") for message in result)
    assert result[-2:] == messages[-2:]


def test_objective_only_writes_leaf_when_anchor_exceeds_recovery_cap(engine):
    instance, summarize = engine
    messages = _cell()
    summary = "SOLE-OBJECTIVE: continue the requested work.\nExpand for details about: user's request"
    summarize.return_value = (summary, 1)
    fresh_tail_start = instance._fresh_tail_start(messages)
    tail_tokens = count_messages_tokens(messages[fresh_tail_start:])
    anchor = instance._build_preserved_objective_summary_part(messages[0])
    anchor_tokens = count_message_tokens({"role": "user", "content": anchor})
    instance._config.max_assembly_tokens = tail_tokens + anchor_tokens + 500
    observed = count_messages_tokens(messages) + 2000  # host overhead outside the messages
    assert instance._should_force_overflow_recovery(observed_tokens=observed, messages=messages)
    recovery_cap = instance._overflow_recovery_assembly_cap(observed_tokens=observed, messages=messages)
    assert tail_tokens + anchor_tokens <= instance._effective_assembly_token_cap()
    assert tail_tokens + anchor_tokens > recovery_cap
    assert tail_tokens + count_message_tokens({"role": "user", "content": summary}) <= recovery_cap

    result = instance.compress(messages, current_tokens=observed)

    assert not instance._objective_only_noop
    nodes = instance._dag.get_session_nodes(instance._session_id)
    assert nodes and summarize.called
    assert any(summary in (message.get("content") or "") for message in result)
    assert result[-2:] == messages[-2:]


@pytest.mark.parametrize("embeddings_enabled", [True, False])
def test_objective_only_noop_reserves_no_proactive_recall(engine, monkeypatch, embeddings_enabled):
    # The no-op returns without assembling, and its tool-only tail has no recall query, so recall needs no room.
    instance, summarize = engine
    messages = _cell()
    fresh_tail_start = instance._fresh_tail_start(messages)
    leading = instance._leading_anchor_count(messages)
    anchor = instance._build_preserved_objective_summary_part(messages[0])
    required = (
        count_messages_tokens(messages[:leading])
        + count_message_tokens({"role": "user", "content": anchor})
        + count_messages_tokens(messages[fresh_tail_start:])
    )
    instance._config.proactive_recall_enabled = True
    instance._config.embeddings_enabled = embeddings_enabled
    instance._config.proactive_recall_budget_tokens = 500
    instance._config.max_assembly_tokens = required + 250
    cap = instance._effective_assembly_token_cap()
    assert leading == 0 and fresh_tail_start == 1
    assert required <= cap < required + instance._config.proactive_recall_budget_tokens
    calls = []
    monkeypatch.setattr(type(instance), "_build_proactive_recall_message", lambda self, *a, **k: calls.append(a))

    result = instance.compress(messages, current_tokens=count_messages_tokens(messages))

    assert instance._objective_only_noop
    assert not summarize.called
    assert not instance._dag.get_session_nodes(instance._session_id)
    assert not calls
    assert count_messages_tokens(result) <= cap


def test_next_round_compacts_earlier_tool_pair_and_preserves_objective(engine, caplog):
    instance, summarize = engine
    messages = _cell()
    with caplog.at_level(logging.INFO):
        first = _host_anti_growth(instance, messages)
    assert "held until turn end (cap 600s): objective_only" in caplog.text
    assert instance.get_status()["no_progress_hold"]["reason"] == "objective_only"
    ceiling = instance._survival_ceiling()
    assert ceiling == 108_800
    for current_tokens in (instance.threshold_tokens, 25_000, 45_000, 80_000, ceiling - 1):
        assert not instance.should_compress(current_tokens)
    assert instance.should_compress(ceiling)
    assert instance.should_compress(ceiling + 1)
    instance._mark_thread_context_stateless("aux-922")
    try:
        instance.note_turn_complete()
        assert instance.get_status()["no_progress_hold"]["reason"] == "objective_only"
    finally:
        instance._clear_thread_context_stateless()
    instance.note_turn_complete()
    assert instance._no_progress_hold is None
    following = first + _round(2)
    assert instance.should_compress(count_messages_tokens(following))

    result = _host_anti_growth(instance, following)

    nodes = instance._dag.get_session_nodes(instance._session_id)
    assert nodes and summarize.called
    covered = {store_id for node in nodes for store_id in node.source_ids}
    assert {2, 3} <= covered  # The earlier assistant/tool pair is summarized together.
    assert result[-2:] == following[-2:]
    assert messages[0]["content"] in result[0]["content"]
    assert result[0]["content"].startswith("[Current user objective preserved from compacted history]")
    assert count_messages_tokens(result) < count_messages_tokens(following)
    assert instance._no_progress_hold is None
    assert not instance._objective_only_noop


def test_user_plus_other_backlog_still_forms_leaf(engine):
    instance, summarize = engine
    messages = _cell() + _round(2)
    assert instance._fresh_tail_start(messages) == 3

    result = instance.compress(messages, current_tokens=count_messages_tokens(messages))

    nodes = instance._dag.get_session_nodes(instance._session_id)
    assert nodes and summarize.called
    assert {1, 2, 3} <= {store_id for node in nodes for store_id in node.source_ids}
    assert messages[0]["content"] in result[0]["content"]
    assert result[-2:] == messages[-2:]


def test_all_rows_in_tail_keeps_no_progress_hold(engine):
    instance, summarize = engine
    messages = _cell()
    instance._config.fresh_tail_max_tokens = 0
    assert instance._fresh_tail_start(messages) == 0

    result = _host_anti_growth(instance, messages)

    assert result == messages
    summarize.assert_not_called()
    assert instance._dag.get_session_nodes(instance._session_id) == []
    assert instance.get_status()["no_progress_hold"]["reason"] == "no_progress"
    instance.note_turn_complete()
    assert instance.get_status()["no_progress_hold"]["reason"] == "no_progress"
    assert not instance.should_compress(count_messages_tokens(messages))


def test_objective_only_hold_has_600_second_cap_and_flag_resets(engine, monkeypatch):
    instance, _ = engine
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(engine_module, "time", SimpleNamespace(
        monotonic=lambda: clock.now, time=lambda: clock.now))
    _host_anti_growth(instance, _cell())
    assert instance._no_progress_hold == (610.0, "objective_only")
    clock.now = 609.999
    assert not instance.should_compress(instance.threshold_tokens)
    clock.now = 610.0
    assert instance.should_compress(instance.threshold_tokens)
    assert instance._no_progress_hold is None
    instance._compress_impl([])
    assert not instance._objective_only_noop
    instance._objective_only_noop = True
    instance._reset_session_scoped_runtime_state()
    assert not instance._objective_only_noop
