"""#659 r7b: carry uses at most half the remaining compaction-trigger headroom."""

import time

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens

HOST_OVERHEAD = 6800
CARRY = "[Earlier user messages in this session, verbatim, for reference only"


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    tokens._count_tokens_cached.cache_clear()
    real_monotonic, offset, calls = time.monotonic, [0.0], []
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + offset[0])

    def summarize(**kwargs):
        calls.append(kwargs)
        offset[0] += 40.0
        return "Synthetic summary of earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    built = []

    def build(window):
        home = tmp_path / str(len(built))
        config = LCMConfig(database_path=str(home / "lcm.db"), fresh_tail_count=24,
                           fresh_tail_max_tokens=24_000, leaf_chunk_tokens=8000, context_threshold=0.75,
                           embeddings_enabled=False, temporal_rollups_enabled=False,
                           large_output_externalization_enabled=False,
                           large_output_active_replay_stubbing_enabled=False)
        engine = LCMEngine(config=config, hermes_home=str(home / "home"))
        engine.on_session_start("S", platform="cli", context_length=window, conversation_id="S")
        built.append(engine)
        return engine, calls

    yield build
    for engine in built:
        engine.shutdown()
    tokens._count_tokens_cached.cache_clear()


@pytest.mark.parametrize("window", [65_536, 131_072])
def test_one_leaf_soft_stop_delivers_carry_below_trigger(make, window):
    engine, calls = make(window)
    messages = []
    while count_messages_tokens(messages) + HOST_OVERHEAD < engine.threshold_tokens:
        turn = len(messages) // 2
        messages.extend([{"role": "user", "content": f"Synthetic request {turn}: preserve this earlier wording."},
                         {"role": "assistant", "content": f"Synthetic response {turn}. " + "detail " * 560}])
    before = count_messages_tokens(messages)
    engine.last_prompt_tokens = before + HOST_OVERHEAD
    returned = engine.compress(messages, current_tokens=before + HOST_OVERHEAD)

    assert len(calls) == 1
    sweep = engine._last_threshold_full_sweep
    assert sweep["stop_reason"] == "soft_target_reached"
    assert sweep["leaf_passes"] == 1
    assert engine._last_compacted_store_id > 0
    status = engine._last_user_carry
    assert status["budget_tokens"] > 0
    assert status["empty_no_room"] is False
    assert any(CARRY in row.get("content", "") for row in returned)
    overhead = engine._survival_host_overhead(messages, max(engine.last_prompt_tokens, engine._last_gate_tokens))
    assert overhead == HOST_OVERHEAD
    assert count_messages_tokens(returned) + overhead <= engine.threshold_tokens
    # est is not exposed by diagnostics; bound delivered size using the returned request without the packet.
    request_without_packet = count_messages_tokens(returned) - status["delivered_tokens"] + overhead
    assert 0 < status["delivered_tokens"] <= status["budget_tokens"]
    assert status["delivered_tokens"] <= (engine.threshold_tokens - request_without_packet) // 2 + 1
    # The retained backlog is why a half-window target cannot deliver this packet.
    assert count_messages_tokens(returned) - status["delivered_tokens"] + overhead > window * 0.5


@pytest.mark.parametrize("window", [65_536, 131_072])
def test_unknown_threshold_keeps_half_window_budget(make, window):
    engine, _ = make(window)
    engine.threshold_tokens = 0
    tail = [{"role": "assistant", "content": "x" * int(window * 0.45 * 4)}]
    engine._user_carry_parts([], "user", [], [], [], tail, tail, None, None)
    # This is the pre-r7 formula with threshold unknown, including the empty-parts message overhead.
    estimate = count_message_tokens({"role": "user", "content": ""}) + count_messages_tokens(tail)
    expected = max(0, min(64_000, int(window * 0.15), int(window * 0.5) - estimate))
    assert 0 < expected < int(window * 0.15)
    assert engine._last_user_carry["budget_tokens"] == expected


@pytest.mark.parametrize("threshold", [0.75, 0.2])
@pytest.mark.parametrize("share", [0.5, 0.0])
def test_known_threshold_budget_uses_half_headroom(make, monkeypatch, threshold, share):
    engine, _ = make(65_536)
    engine.threshold_tokens = int(engine.context_length * threshold)
    monkeypatch.setattr(engine_module, "_USER_CARRY_HEADROOM_SHARE", share, raising=False)
    tail = [{"role": "assistant", "content": "x" * int(engine.threshold_tokens * 0.6 * 4)}]
    engine._user_carry_parts([], "user", [], [], [], tail, tail, None, None)
    estimate = count_message_tokens({"role": "user", "content": ""}) + count_messages_tokens(tail)
    expected = max(0, min(64_000, int(engine.context_length * 0.15),
                          int((engine.threshold_tokens - estimate) * share)))
    assert engine._last_user_carry["budget_tokens"] == expected
    assert expected == 0 if share == 0.0 else expected > 0
