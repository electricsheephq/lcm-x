"""#611 / #440: the summariser input clip arm (``summary_input_clip``: legacy | whole12k | budget).

``legacy`` (the default) keeps today's rule; the other arms keep a fact in the middle of a long message. Every
``_serialize_messages`` consumer goes through the arm: the leaf call, extraction, the route-stop
``verbatim_source`` check and level-3 repair."""

from __future__ import annotations

import logging

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.level3_repair as level3_repair
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import _deterministic_truncate
from hermes_lcm.externalize import is_externalized_placeholder
from hermes_lcm.summary_input_clip import CLIP_MARKER
from hermes_lcm.tokens import count_tokens

ARMS = ("legacy", "whole12k", "budget")
FACT = "FACT-611-MIDDLE-ZEBRA-4417"


def _long(chars: int, fact: str = FACT) -> str:
    """``chars`` characters with ``fact`` at the middle; head and tail tagged so a clip is visible."""
    filler = "lorem ipsum dolor sit amet "
    half = (chars - len(fact)) // 2
    body = (filler * (chars // len(filler) + 2))
    return "HEAD-TAG " + body[: half - 9] + fact + body[: chars - half - len(fact) - 9] + " TAIL-TAG"


def _engine(tmp_path, arm: str, **config) -> LCMEngine:
    settings = {"database_path": str(tmp_path / "lcm.db"), "summary_input_clip": arm, **config}
    return LCMEngine(config=LCMConfig(**settings), hermes_home=str(tmp_path / "hermes-home"))


def _kept(arm: str) -> bool:
    return arm != "legacy"


# -- configuration -------------------------------------------------------------------------------------------

def test_default_is_legacy_and_env_selects_an_arm(monkeypatch):
    monkeypatch.delenv("LCM_SUMMARY_INPUT_CLIP", raising=False)
    assert LCMConfig().summary_input_clip == "legacy"
    assert LCMConfig.from_env().summary_input_clip == "legacy"
    for arm in ARMS:
        monkeypatch.setenv("LCM_SUMMARY_INPUT_CLIP", f" {arm.upper()} ")
        cfg = LCMConfig.from_env()
        assert cfg.summary_input_clip == arm
        assert cfg.config_sources["summary_input_clip"] == "env:LCM_SUMMARY_INPUT_CLIP"


def test_unknown_env_value_falls_back_to_legacy_with_a_warning(monkeypatch):
    monkeypatch.setenv("LCM_SUMMARY_INPUT_CLIP", "none")
    cfg = LCMConfig.from_env()
    assert cfg.summary_input_clip == "legacy"
    assert any("LCM_SUMMARY_INPUT_CLIP" in w for w in cfg.config_source_warnings)


# -- the rule per arm ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_middle_fact_of_a_10000_char_message(tmp_path, arm, role):
    engine = _engine(tmp_path, arm, large_output_externalization_enabled=False)
    try:
        msg = {"role": role, "content": _long(10_000)}
        if role == "tool":
            msg["tool_call_id"] = "call_1"
        serialized = engine._serialize_messages([msg])
        assert (FACT in serialized) is _kept(arm)
        assert "HEAD-TAG" in serialized and "TAIL-TAG" in serialized
        assert (CLIP_MARKER in serialized) is (arm == "legacy")
    finally:
        engine.shutdown()


@pytest.mark.parametrize("chars,clipped", [(12_000, False), (12_001, True)])
def test_whole12k_boundary(tmp_path, chars, clipped):
    engine = _engine(tmp_path, "whole12k")
    try:
        content = _long(chars)
        assert len(content) == chars
        serialized = engine._serialize_messages([{"role": "user", "content": content}])
        if clipped:
            assert serialized == "[USER]: " + content[:8_000] + CLIP_MARKER + content[-2_000:]
        else:
            assert serialized == "[USER]: " + content
    finally:
        engine.shutdown()


@pytest.mark.parametrize("arm", ["whole12k", "budget"])
def test_a_1mb_single_message_is_bounded(tmp_path, arm):
    engine = _engine(tmp_path, arm, leaf_chunk_tokens=8_000, large_output_externalization_enabled=False)
    try:
        for role in ("user", "assistant", "tool"):
            msg = {"role": role, "content": _long(1_000_000)}
            if role == "tool":
                msg["tool_call_id"] = "call_big"
            serialized = engine._serialize_messages([msg])
            if arm == "whole12k":
                assert len(serialized) <= 10_000 + len(CLIP_MARKER) + 40
            else:
                assert count_tokens(serialized) <= 8_000 + 40
            assert "HEAD-TAG" in serialized and "TAIL-TAG" in serialized
    finally:
        engine.shutdown()


def test_budget_shares_the_chunk_budget_proportionally(tmp_path):
    engine = _engine(tmp_path, "budget", leaf_chunk_tokens=2_000)
    try:
        small, big = _long(4_000, "FACT-SMALL"), _long(40_000, "FACT-BIG")
        serialized = engine._serialize_messages([
            {"role": "user", "content": small}, {"role": "assistant", "content": big},
        ])
        assert count_tokens(serialized) <= 2_000 + 40
        user_part, assistant_part = serialized.split("\n\n[ASSISTANT]: ")
        # the same share of each text: the bigger message keeps about ten times the characters
        assert 8 <= len(assistant_part) / len(user_part) <= 12
    finally:
        engine.shutdown()


def test_budget_keeps_a_chunk_within_the_budget_whole(tmp_path):
    engine = _engine(tmp_path, "budget", leaf_chunk_tokens=8_000)
    try:
        content = _long(20_000)
        assert count_tokens(content) < 8_000
        assert engine._serialize_messages([{"role": "user", "content": content}]) == "[USER]: " + content
    finally:
        engine.shutdown()


@pytest.mark.parametrize("arm", ARMS)
def test_tool_arguments_per_arm(tmp_path, arm):
    engine = _engine(tmp_path, arm)
    try:
        args = '{"text": "' + _long(2_000) + '"}'
        serialized = engine._serialize_messages([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "write", "arguments": args}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ])
        assert (FACT in serialized) is (arm == "budget")  # whole12k keeps the legacy 400-character argument rule
    finally:
        engine.shutdown()


@pytest.mark.parametrize("arm", ARMS)
def test_externalized_result_preview_is_outside_the_placeholder(tmp_path, arm):
    engine = _engine(tmp_path, arm, large_output_externalization_enabled=True,
                     large_output_externalization_threshold_chars=12_000)
    try:
        content = _long(30_000)
        serialized = engine._serialize_messages([{"role": "tool", "tool_call_id": "call_ext", "content": content}])
        label = "[TOOL RESULT call_ext]: "
        assert serialized.startswith(label)
        body = serialized[len(label):]
        if arm == "whole12k":
            placeholder, preview = body.split("\n[preview: ", 1)
            assert is_externalized_placeholder(placeholder) and len(placeholder.strip()) <= 512
            assert preview == content[:600] + " … " + content[-300:] + "]"
        else:
            assert is_externalized_placeholder(body) and "[preview:" not in body
        assert FACT not in serialized
    finally:
        engine.shutdown()


# -- every consumer goes through the arm ----------------------------------------------------------------------

def _fact_message() -> dict:
    return {"role": "user", "content": _long(10_000)}


@pytest.mark.parametrize("arm", ARMS)
def test_consumer_leaf_call_and_recorder(tmp_path, monkeypatch, caplog, arm):
    seen = []

    def fake(**kwargs):
        seen.append(kwargs["text"])
        return "summary", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", fake)
    engine = _engine(tmp_path, arm)
    try:
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine._summarize_leaf_chunk_with_rescue([_fact_message()])
        assert len(seen) == 1 and (FACT in seen[0]) is _kept(arm)
        lines = [r.getMessage() for r in caplog.records if "LCM leaf summary input:" in r.getMessage()]
        assert lines == [f"LCM leaf summary input: input_tokens={count_tokens(seen[0])} "
                         f"source_tokens={lcm_engine.count_messages_tokens([_fact_message()])} messages=1 clip={arm}"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("arm", ARMS)
def test_consumer_extraction(tmp_path, monkeypatch, arm):
    seen = []
    monkeypatch.setattr(lcm_engine, "extract_before_compaction",
                        lambda **kwargs: seen.append(kwargs["serialized_messages"]) or True)
    engine = _engine(tmp_path, arm)
    try:
        engine._run_pre_compaction_extraction([_fact_message()])
        assert len(seen) == 1 and (FACT in seen[0]) is _kept(arm)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("arm", ARMS)
def test_consumer_route_stop_verbatim_source(tmp_path, monkeypatch, caplog, arm):
    seen = []
    real = lcm_engine.verbatim_source
    monkeypatch.setattr(lcm_engine, "verbatim_source", lambda text, bound: seen.append(text) or real(text, bound))
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), summary_input_clip=arm, fresh_tail_count=2, leaf_chunk_tokens=8_000,
        context_threshold=0.001, threshold_full_sweep_enabled=True, max_assembly_tokens=100_000))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    try:
        breaker = engine._summary_circuit_breaker
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(engine._config.summary_model)
        view = [{"role": "system", "content": "system prompt"}]
        for i in range(4):
            view += [{"role": "user", "content": _long(10_000, f"{FACT}-{i}"), "timestamp": 10.0 * (i + 1)},
                     {"role": "assistant", "content": f"reply {i}"}]
        engine.ingest(view)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert seen, "the route-stop check never saw a serialized source"
        assert ("FACT-611-MIDDLE-ZEBRA-4417-0" in seen[0]) is _kept(arm)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("arm", ARMS)
def test_consumer_level3_repair(tmp_path, monkeypatch, arm):
    seen = []

    def fake(**kwargs):
        seen.append(kwargs["text"])
        return "repaired summary", 1

    monkeypatch.setattr(level3_repair, "summarize_with_escalation", fake)
    engine = _engine(tmp_path, arm)
    try:
        row = engine._store.append("s1", _fact_message(), token_estimate=2_600)
        long_source = "\n".join(f"[user]: step {i}: checked host {i}." for i in range(400))
        fragment = _deterministic_truncate(long_source, 512)
        engine._dag.add_node(SummaryNode(
            session_id="s1", depth=0, summary=fragment, token_count=count_tokens(fragment),
            source_token_count=2_600, source_ids=[row], source_type="messages", created_at=1.0))
        rows, groups = level3_repair._groups(engine._dag.connection, level3_repair.scan_level3_fragments(engine))
        assert len(groups) == 1
        new, refusal = level3_repair._summarise_group(engine, rows, groups[0], {"calls": 0}, None)
        assert refusal == "" and len(seen) == 1 and (FACT in seen[0]) is _kept(arm)
    finally:
        engine.shutdown()
