"""#922: do not summarize the sole prompt that assembly would repeat verbatim."""

from copy import deepcopy
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
    rejected = Mock(wraps=instance.record_rejected_compaction)
    monkeypatch.setattr(instance, "record_rejected_compaction", rejected)

    result = _host_anti_growth(instance, messages)

    assert instance._dag.get_session_nodes(instance._session_id) == []
    summarize.assert_not_called()
    assert result == original
    assert count_messages_tokens(result) <= count_messages_tokens(original)
    rejected.assert_not_called()
    assert instance._no_progress_hold is None
    assert instance.should_compress(count_messages_tokens(result))
    assert instance._store.get_session_count(instance._session_id) == 3


def test_next_round_compacts_earlier_tool_pair_and_preserves_objective(engine):
    instance, summarize = engine
    messages = _cell()
    first = _host_anti_growth(instance, messages)
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
