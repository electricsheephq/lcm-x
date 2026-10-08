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


def add_nodes(engine, depth, count, token_count):
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
            created_at=float(index + 1),
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


def test_pressure_starts_above_a_quarter_of_the_survival_ceiling(engine):
    assert engine._survival_ceiling() == 231_200
    shallow = add_nodes(engine, 0, 4, token_count=500)
    deep = add_nodes(engine, 1, 62, token_count=900)  # frontier 2,000 + 55,800 = 57,800, exactly a quarter
    assert selected(engine) == [(0, node_id) for node_id in shallow]
    add_nodes(engine, 2, 1, token_count=1)
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
