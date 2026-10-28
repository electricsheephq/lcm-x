"""v0.27.0 phase 0: larger default leaves with a fixed summary target."""

from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def test_p0_leaf_chunk_and_summary_defaults():
    config = LCMConfig()
    assert config.leaf_chunk_tokens == 24_000
    assert config.leaf_target_max_tokens == 2_000
    engine = SimpleNamespace(_config=config)
    assert [LCMEngine._leaf_target_tokens(engine, n) for n in (8_000, 20_000, 100_000)] == [2_000] * 3


@pytest.fixture
def threshold_sweep(tmp_path, monkeypatch, request):
    home = tmp_path / "hermes"
    config = LCMConfig(database_path=str(home / "lcm.db"))
    engine = LCMEngine(config=config, hermes_home=str(home))
    engine.on_session_start("phase-0", platform="cli", context_length=getattr(request, "param", 16_000))
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"FACT-{i} " + "alpha beta gamma delta " * 44}
        for i in range(48)
    ] + [{"role": "user" if i % 2 == 0 else "assistant", "content": f"tail {i}"} for i in range(24)]

    def fake_leaf(chunk, focus_topic=None, deadline=None):
        return chunk, count_messages_tokens(chunk), "Earlier facts.\nExpand for details about: facts", 1, 0

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_leaf)
    try:
        yield engine, messages
    finally:
        engine.shutdown()


def test_p0_threshold_sweep_compacts_less_than_one_default_chunk(threshold_sweep):
    engine, messages = threshold_sweep
    boundary = engine._fresh_tail_boundary(messages)
    raw_tokens = count_messages_tokens(messages[:boundary.start])
    assert 11_000 < raw_tokens < 13_000
    assert raw_tokens < engine._config.leaf_chunk_tokens
    assert engine._config.threshold_full_sweep_enabled is True
    current_tokens = count_messages_tokens(messages)
    assert current_tokens > engine.threshold_tokens
    engine.compress(messages, current_tokens=current_tokens)
    assert any(n.depth == 0 for n in engine._dag.get_session_nodes("phase-0"))
    assert engine.get_status()["threshold_full_sweep"]["leaf_passes"] >= 1


@pytest.mark.parametrize("threshold_sweep", [16_000, 272_000], indirect=True)
def test_p0_sweep_prefix_target_capped_by_threshold(threshold_sweep):
    engine, messages = threshold_sweep
    assert engine._config.summary_prefix_target_tokens == 0
    engine.compress(messages, current_tokens=max(engine.threshold_tokens, count_messages_tokens(messages)))
    telemetry = engine.get_status()["threshold_full_sweep"]
    assert engine._config.leaf_chunk_tokens == 24_000
    assert telemetry["summary_prefix_target_tokens"] == min(24_000, int(0.25 * engine.threshold_tokens))


@pytest.mark.parametrize("threshold_sweep", [32_000], indirect=True)
def test_p0_small_window_sweep_condenses_to_a_quarter_of_threshold(threshold_sweep, monkeypatch):
    import hermes_lcm.engine as engine_module

    engine, messages = threshold_sweep
    assert engine.threshold_tokens == 24_000
    for index in range(4):
        engine._dag.add_node(SummaryNode(
            session_id="phase-0", depth=0, summary=f"Earlier fact group {index}",
            token_count=2_000, source_token_count=24_000, source_ids=[],
            source_type="messages", created_at=index,
        ))
    assert 6_000 < engine._summary_frontier_tokens() < 24_000
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda **kwargs: ("Earlier fact groups 0 through 3.", 1))
    messages = [{**message, "content": message["content"] * 2} for message in messages]
    current_tokens = count_messages_tokens(messages)
    assert engine.threshold_tokens <= current_tokens < 32_000
    engine.compress(messages, current_tokens=current_tokens)
    telemetry = engine.get_status()["threshold_full_sweep"]
    assert telemetry["condensation_passes"] >= 1
    assert telemetry["summary_prefix_tokens_after"] <= 6_000
    assert telemetry["stop_reason"] == "summary_prefix_target_reached"


def test_p0_default_leaf_rescue_shrinks_below_the_chunk_for_a_small_context_summariser(tmp_path, monkeypatch, caplog):
    """A route that accepts at most half of the first transcript (a small-context summariser) still gets a summary
    within the three rescue attempts at the 24,000-token default: the last attempt shrinks to 8,000 tokens
    (24k -> 18k -> 8k; 24k -> 18k -> 13.5k never fits)."""
    import logging
    import re
    import sys
    from types import ModuleType

    calls: list[int] = []
    reply_text = "Older turns: alpha through epsilon.\nExpand for details about: older turns"

    def call_llm(**kwargs):
        policy, transcript = kwargs["messages"][0]["content"], kwargs["messages"][1]["content"]
        calls.append(len(transcript))
        if len(transcript) > calls[0] * 0.5:
            raise RuntimeError("Error code: 400 - This model's maximum context length is 16384 tokens. "
                               "(context_length_exceeded)")
        nonce = re.search(r'<lcm-summary nonce="([0-9a-f]{32})">', policy).group(1)
        reply = f'<lcm-summary nonce="{nonce}">\n{reply_text}\n</lcm-summary>'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    module = ModuleType("agent.auxiliary_client")
    module.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), fresh_tail_count=2, context_threshold=0.001,
        large_output_externalization_enabled=False, large_output_active_replay_stubbing_enabled=False,
        temporal_rollups_enabled=False))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    pad = " alpha beta gamma delta epsilon" * 40
    view = [{"role": "system", "content": "system prompt"}]
    for i in range(45):
        view += [{"role": "user", "content": f"[T{i}] user turn{pad}", "timestamp": 10.0 * (i + 1)},
                 {"role": "assistant", "content": f"reply to T{i}{pad}"}]
    try:
        engine.ingest(view)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert engine._config.leaf_chunk_tokens == 24_000
        assert min(calls) <= calls[0] * 0.5, calls  # the third attempt fits (24k -> 18k -> 8k sources)
        nodes = [n for n in engine._dag.get_session_nodes("S", limit=1000) if n.depth == 0]
        assert nodes and "deterministic truncation" not in nodes[0].summary, calls
        assert nodes[0].summary.startswith("Older turns"), calls
    finally:
        engine.shutdown()
