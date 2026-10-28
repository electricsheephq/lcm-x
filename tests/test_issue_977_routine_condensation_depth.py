"""#977: a routine sweep condenses the shallowest eligible depth; the heaviest depth (#930) only under prefix pressure."""

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


@pytest.fixture(autouse=True)
def deterministic_offline_counts(monkeypatch):
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(engine_module, "count_tokens", tokens._fallback_token_estimate)

    def no_model_calls(**kwargs):
        pytest.fail("group selection must not call a model")

    monkeypatch.setattr(engine_module, "summarize_with_escalation", no_model_calls)


def make_engine(tmp_path, context_length, **config):
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), condensation_fanin=4, **config))
    instance.on_session_start("issue-977", context_length=context_length)
    return instance


@pytest.fixture
def engine(tmp_path):
    instance = make_engine(tmp_path, 272_000)
    try:
        yield instance
    finally:
        instance.shutdown()


def add_nodes(engine, depth, count, token_count, start=0):
    ids = []
    for index in range(count):
        ids.append(engine._dag.add_node(SummaryNode(
            session_id=engine.current_session_id,
            depth=depth,
            summary=f"depth {depth} entry {index}: " + "abcd" * max(1, token_count - 10),
            token_count=token_count,
            source_token_count=token_count * 4,
            source_ids=[],
            source_type="messages" if depth == 0 else "nodes",
            created_at=float(start + index + 1),
            expand_hint="synthetic history",
        )))
    return ids


def selected(engine):
    return [(node.depth, node.node_id) for node in engine._select_threshold_sweep_condensation_group()]


def test_routine_sweep_keeps_the_shallowest_depth_when_a_deeper_group_is_heavier(engine):
    # The Track S cp-304 shape: a frontier just over the fleet's 8,000-token sweep target, depth 1 slightly heavier.
    shallow = add_nodes(engine, 0, 4, token_count=700)
    add_nodes(engine, 1, 8, token_count=900)
    assert selected(engine) == [(0, node_id) for node_id in shallow]


def test_prefix_pressure_condenses_the_heaviest_depth(engine):
    add_nodes(engine, 0, 4, token_count=700)
    deep = add_nodes(engine, 1, 80, token_count=900)
    assert selected(engine) == [(1, node_id) for node_id in deep[:4]]


def test_pressure_is_measured_on_the_frontier_as_assembly_renders_it(engine):
    assert engine._survival_ceiling() == 231_200
    quarter = 231_200 // 4
    shallow = add_nodes(engine, 0, 4, token_count=500)
    deep = add_nodes(engine, 1, 55, token_count=900)
    assert engine._rendered_summary_frontier_tokens() <= quarter
    assert selected(engine) == [(0, node_id) for node_id in shallow]
    deep += add_nodes(engine, 1, 7, token_count=900, start=55)
    # Stored token counts now total exactly a quarter, but headers, expand hints and separators put the
    # rendered prefix over it, where the default prefix bound could start to omit summaries.
    assert sum(node.token_count for node in engine._summary_frontier_nodes()) == quarter
    assert engine._rendered_summary_frontier_tokens() > quarter
    assert selected(engine) == [(1, node_id) for node_id in deep[:4]]


def test_unknown_window_keeps_the_shallowest_depth(tmp_path):
    instance = make_engine(tmp_path, 0)
    try:
        assert instance._survival_ceiling() is None
        shallow = add_nodes(instance, 0, 4, token_count=700)
        add_nodes(instance, 1, 80, token_count=900)
        assert selected(instance) == [(0, node_id) for node_id in shallow]
    finally:
        instance.shutdown()


def test_without_survival_fit_the_heaviest_depth_still_applies(tmp_path):
    # The #930 default prefix budget has no quarter-ceiling floor without survival fit, so a small frontier can be cut.
    instance = make_engine(tmp_path, 272_000, survival_fit=False)
    try:
        add_nodes(instance, 0, 4, token_count=700)
        deep = add_nodes(instance, 1, 8, token_count=900)
        assert selected(instance) == [(1, node_id) for node_id in deep[:4]]
    finally:
        instance.shutdown()


def test_a_light_shallowest_group_is_skipped_for_the_next_shrinkable_depth(engine):
    # #909: a group of at most 512 tokens is about as long as its summary, so condensing it cannot shrink the frontier.
    add_nodes(engine, 0, 4, token_count=100)
    middle = add_nodes(engine, 1, 4, token_count=900)
    add_nodes(engine, 2, 4, token_count=2_000)
    assert selected(engine) == [(1, node_id) for node_id in middle]


def test_only_light_groups_fall_back_to_the_heaviest_depth(engine):
    add_nodes(engine, 0, 4, token_count=50)
    deep = add_nodes(engine, 1, 4, token_count=100)
    assert selected(engine) == [(1, node_id) for node_id in deep]


def test_a_group_the_l3_bound_would_store_whole_is_light(tmp_path):
    # verbatim_small_source stores a source within the configured L3 bound whole, so it cannot shrink the frontier.
    instance = make_engine(tmp_path, 272_000, l3_truncate_tokens=2_000)
    try:
        add_nodes(instance, 0, 4, token_count=300)
        middle = add_nodes(instance, 1, 4, token_count=900)
        add_nodes(instance, 2, 4, token_count=2_000)
        assert selected(instance) == [(1, node_id) for node_id in middle]
    finally:
        instance.shutdown()


def test_unknown_window_keeps_the_routine_rule_without_survival_fit(tmp_path):
    # With no known window there is no default prefix bound to cut a frontier, with or without survival fit.
    instance = make_engine(tmp_path, 0, survival_fit=False)
    try:
        assert instance._survival_ceiling() is None
        shallow = add_nodes(instance, 0, 4, token_count=700)
        add_nodes(instance, 1, 8, token_count=900)
        assert selected(instance) == [(0, node_id) for node_id in shallow]
    finally:
        instance.shutdown()


def test_a_small_group_that_still_shrinks_is_condensed_at_the_shallowest_depth(engine):
    # Track S, seed 2: an 818-token depth-0 group condensed to 548 tokens. Skipping it would have sent the sweep to
    # depth 1 and written the depth-2 node this change removes.
    shallow = add_nodes(engine, 0, 4, token_count=205)
    add_nodes(engine, 1, 6, token_count=900)
    assert selected(engine) == [(0, node_id) for node_id in shallow]
