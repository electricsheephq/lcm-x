"""#930: bound default summary assembly without removing DAG coverage."""

import json
import re
import sys

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tools import lcm_expand


@pytest.fixture(autouse=True)
def deterministic_offline_counts(monkeypatch):
    """Pin the existing character estimate; these regressions make no model calls."""
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(engine_module, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)

    def no_model_calls(**kwargs):
        pytest.fail("assembly and group selection must not call a model")

    monkeypatch.setattr(engine_module, "summarize_with_escalation", no_model_calls)


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        max_assembly_tokens=0,
        reserve_tokens_floor=0,
        condensation_fanin=4,
    ))
    instance.on_session_start("issue-930", context_length=128_000)
    try:
        yield instance
    finally:
        instance.shutdown()


def add_nodes(engine, depth, count, token_count=2_000):
    ids = []
    for index in range(count):
        summary = f"depth {depth} entry {index}: " + "abcd" * (token_count - 10)
        ids.append(engine._dag.add_node(SummaryNode(
            session_id=engine.current_session_id,
            depth=depth,
            summary=summary,
            token_count=token_count,
            source_token_count=token_count * 2,
            source_ids=[],
            source_type="messages" if depth == 0 else "nodes",
            created_at=float(index + 1),
            expand_hint="synthetic history",
        )))
    return ids


def assemble(engine, tail, **kwargs):
    return engine._assemble_context(
        {"role": "system", "content": "Synthetic system prompt."}, tail,
        include_lcm_note=False, persist=False, **kwargs,
    )


def visible_node_ids(messages):
    return [int(node_id) for message in messages
            for node_id in re.findall(r"Summary \(d\d+, node (\d+)\)", message["content"])]


@pytest.mark.parametrize("survival_enabled", [True, False])
def test_default_prefix_bound_keeps_newest_and_preserves_dag(engine, survival_enabled):
    engine._config.survival_fit = survival_enabled
    ids = add_nodes(engine, 1, 100)
    tail = [{"role": "user", "content": "Latest question. " + "tail " * 400}]
    result = assemble(engine, tail)
    assert tokens.count_messages_tokens(result) <= int(128_000 * 0.85)
    kept = visible_node_ids(result)
    assert 0 < len(kept) < len(ids)
    assert kept == ids[-len(kept):]
    assert result[-1] == tail[-1]
    assert [node.node_id for node in engine._dag.get_session_nodes(
        engine.current_session_id, limit=100_000)] == ids
    for node_id in ids:
        assert engine._dag.get_node(node_id) is not None
    expanded = json.loads(lcm_expand({"node_id": ids[0]}, engine=engine))
    assert "error" not in expanded
    assert expanded["node_id"] == ids[0]


def test_tail_heavy_assembly_keeps_prefix_for_survival_fit(engine):
    ids = add_nodes(engine, 1, 5, token_count=2_000)
    ceiling = engine._survival_ceiling()
    tail = [message for index in range(4) for message in (
        {"role": "user", "content": f"Turn {index}"},
        {"role": "assistant", "content": "abcd" * (ceiling // 5)},
    )]
    system = {"role": "system", "content": "Synthetic system prompt."}
    source = [system, *tail]
    engine.ingest(source)
    engine.last_prompt_tokens = tokens.count_messages_tokens(source) + 20_000
    remaining = ceiling - 20_000 - tokens.count_messages_tokens(source)
    assert remaining < 10_000 < ceiling // 4
    assembled = assemble(engine, tail)
    assert visible_node_ids(assembled) == ids
    assert assembled[-len(tail):] == tail
    assert tokens.count_messages_tokens(assembled[1:-len(tail)]) < ceiling // 4
    budget = engine._survival_fit_budget(source, engine.last_prompt_tokens)
    assert engine._survival_measure(assembled) > budget
    engine._ingest_cursor = len(assembled)  # The host list contains only stored turns and DAG summaries.
    fitted = engine._survival_fit(source, assembled, engine.last_prompt_tokens, "issue-930")
    assert visible_node_ids(fitted) == ids
    assert len(fitted) < len(assembled)
    assert engine._survival_measure(fitted) <= budget
    assert fitted[-1] == tail[-1]


def test_tail_heavy_assembly_without_survival_fit_stays_within_ceiling(engine):
    engine._config.survival_fit = False
    ids = add_nodes(engine, 1, 5, token_count=2_000)
    ceiling = engine._survival_ceiling()
    tail = [message for index in range(4) for message in (
        {"role": "user", "content": f"Turn {index}"},
        {"role": "assistant", "content": "abcd" * (ceiling // 5)},
    )]
    source = [{"role": "system", "content": "Synthetic system prompt."}, *tail]
    engine.ingest(source)
    engine.last_prompt_tokens = tokens.count_messages_tokens(source) + 20_000
    remaining = ceiling - 20_000 - tokens.count_messages_tokens(source)
    assert 0 < remaining < 10_000 < ceiling // 4
    assembled = assemble(engine, tail)
    assert assembled[-len(tail):] == tail
    assert tokens.count_messages_tokens(assembled[1:-len(tail)]) <= remaining
    kept = visible_node_ids(assembled)
    assert kept == ids[-len(kept):] if kept else True


def test_below_bound_is_byte_identical_to_uncapped_path(engine):
    add_nodes(engine, 0, 4, token_count=200)
    add_nodes(engine, 1, 4, token_count=300)
    tail = [{"role": "user", "content": "Latest question."},
            {"role": "assistant", "content": "Latest reply."}]
    bounded = assemble(engine, tail)
    engine.context_length = 0  # The unchanged, unknown-window uncapped path.
    uncapped = assemble(engine, tail)
    assert json.dumps(bounded, ensure_ascii=False).encode() == json.dumps(uncapped, ensure_ascii=False).encode()


def test_explicit_max_assembly_tokens_preserves_existing_behavior(engine):
    ids = add_nodes(engine, 1, 100)
    tail = [{"role": "user", "content": "Latest question."}]
    engine._config.max_assembly_tokens = 120_000  # Above the default survival ceiling.
    known_window = assemble(engine, tail)
    engine.context_length = 0
    assert known_window == assemble(engine, tail)
    assert int(128_000 * 0.85) < tokens.count_messages_tokens(known_window) <= 120_000
    kept = visible_node_ids(known_window)
    assert kept == ids[-len(kept):]


@pytest.mark.parametrize("cap_kind", ["override", "reserve"])
def test_other_explicit_caps_preserve_existing_behavior(engine, cap_kind):
    add_nodes(engine, 1, 100)
    tail = [{"role": "user", "content": "Latest question."}]
    kwargs = {"assembly_cap_override": 120_000}
    if cap_kind == "reserve":
        engine._config.reserve_tokens_floor = 8_000
        kwargs = {}
    known_window = assemble(engine, tail, **kwargs)
    engine._config.reserve_tokens_floor = 0
    engine.context_length = 0
    assert known_window == assemble(engine, tail, assembly_cap_override=120_000)


def test_default_prefix_budget_subtracts_existing_host_overhead(engine):
    add_nodes(engine, 1, 100)
    tail = [{"role": "user", "content": "Latest question."}]
    system = {"role": "system", "content": "Synthetic system prompt."}
    engine.last_prompt_tokens = tokens.count_messages_tokens([system, *tail]) + 20_000
    assert tokens.count_messages_tokens(assemble(engine, tail)) + 20_000 <= engine._survival_ceiling()


def test_sweep_selects_heaviest_eligible_depth(engine):
    add_nodes(engine, 0, 6)
    depth_one = add_nodes(engine, 1, 38, token_count=3_000)
    group = engine._select_threshold_sweep_condensation_group()
    assert [node.depth for node in group] == [1] * 4
    assert [node.node_id for node in group] == depth_one[:4]


def test_equal_candidate_group_tokens_prefer_shallowest_depth(engine):
    shallow = add_nodes(engine, 0, 6, token_count=3_000)
    add_nodes(engine, 1, 4, token_count=3_000)
    assert [node.node_id for node in engine._select_threshold_sweep_condensation_group()] == shallow[:4]


@pytest.mark.parametrize("depth,count", [(1, 3), (3, 4)])
def test_heavier_ineligible_depth_does_not_change_routine_group(engine, depth, count):
    shallow = add_nodes(engine, 0, 4, token_count=200)
    add_nodes(engine, depth, count, token_count=10_000)
    assert [node.node_id for node in engine._select_threshold_sweep_condensation_group()] == shallow


def test_rebind_clears_gate_observation_and_matches_fresh_assembly(engine):
    engine.should_compress(prompt_tokens=200_000)
    assert engine._last_gate_tokens == 200_000
    engine.on_session_start("issue-954-B", context_length=128_000)
    ids = add_nodes(engine, 1, 60)
    tail = [{"role": "user", "content": "Latest question."}]
    fresh = LCMEngine(config=LCMConfig(database_path=engine._config.database_path,
                                     max_assembly_tokens=0, reserve_tokens_floor=0))
    try:
        fresh.on_session_start("issue-954-B", context_length=128_000)
        assert visible_node_ids(assemble(engine, tail)) == visible_node_ids(assemble(fresh, tail))
        assert engine._last_gate_tokens == 0
        assert all(engine._dag.get_node(node_id) is not None for node_id in ids)
    finally:
        fresh.shutdown()


def test_prefix_bound_reserves_missing_tool_result_stub(engine):
    engine._config.survival_fit = False
    engine.context_length = 2_000
    ids = add_nodes(engine, 1, 1, token_count=200)
    tail = [{"role": "user", "content": "Latest question."},
            {"role": "assistant", "content": "Running tool.", "tool_calls": [
                {"id": "missing-result", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]}]
    stub_cost = (tokens.count_messages_tokens(engine._sanitize_active_context_messages(tail))
                 - tokens.count_messages_tokens(tail))
    assert stub_cost > 0
    unbounded = assemble(engine, tail, assembly_cap_override=sys.maxsize)
    padding = engine._survival_ceiling() - tokens.count_messages_tokens(unbounded) + stub_cost // 2
    tail[0]["content"] += "abcd" * padding
    assert tokens.count_messages_tokens(assemble(engine, tail, assembly_cap_override=sys.maxsize)) > engine._survival_ceiling()
    result = assemble(engine, tail)
    assert tokens.count_messages_tokens(result) <= engine._survival_ceiling()
    assert any(row.get("tool_call_id") == "missing-result" for row in result)
    assert engine._dag.get_node(ids[0]) is not None


def test_prefix_bound_reserves_proactive_recall(engine, monkeypatch):
    engine._config.survival_fit = False
    engine._config.proactive_recall_enabled = True
    engine._config.embeddings_enabled = True
    engine._config.proactive_recall_budget_tokens = 500
    engine.context_length = 2_000
    add_nodes(engine, 1, 7, token_count=200)
    monkeypatch.setattr(engine_module.lcm_tools, "lcm_recall", lambda *args, **kwargs: json.dumps({
        "hits": [{"score": 1.0, "snippet": "remembered fact " * 80}],
    }))
    tail = [{"role": "user", "content": "Latest question."}]
    result = engine._assemble_context(
        {"role": "system", "content": "Synthetic system prompt."}, tail, include_lcm_note=False,
    )
    assert any("<relevant-memories>" in row["content"] for row in result)
    assert tokens.count_messages_tokens(result) <= engine._survival_ceiling()


def test_prefix_bound_applies_when_reserve_cap_is_inactive(engine):
    engine._config.reserve_tokens_floor = engine.context_length
    assert engine._effective_assembly_token_cap() is None
    ids = add_nodes(engine, 1, 100)
    result = assemble(engine, [{"role": "user", "content": "Latest question."}])
    assert tokens.count_messages_tokens(result) <= engine._survival_ceiling()
    assert 0 < len(visible_node_ids(result)) < len(ids)
    assert all(engine._dag.get_node(node_id) is not None for node_id in ids)


def test_conversation_only_rebind_clears_gate_observation(tmp_path):
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
                                          max_assembly_tokens=0, reserve_tokens_floor=0))
    try:
        instance.on_session_start("issue-954-conv", conversation_id="conv-A", context_length=128_000)
        instance.should_compress(prompt_tokens=200_000)
        assert instance._last_gate_tokens == 200_000
        instance.on_session_start("issue-954-conv", conversation_id="conv-B", context_length=128_000)
        assert instance.last_prompt_tokens == 0
        assert instance._last_gate_tokens == 0
    finally:
        instance.shutdown()


@pytest.mark.parametrize("embeddings,budget", [(False, 1_500), (True, 0), (True, -1_500)])
def test_prefix_bound_reserves_recall_only_when_it_can_run(engine, embeddings, budget):
    engine._config.survival_fit = False
    engine.context_length = 4_000
    add_nodes(engine, 1, 20, token_count=200)
    tail = [{"role": "user", "content": "Latest question."}]
    baseline = visible_node_ids(assemble(engine, tail))
    engine._config.proactive_recall_enabled = True
    engine._config.embeddings_enabled = embeddings
    engine._config.proactive_recall_budget_tokens = budget
    result = assemble(engine, tail)
    assert visible_node_ids(result) == baseline
    assert tokens.count_messages_tokens(result) <= engine._survival_ceiling()
