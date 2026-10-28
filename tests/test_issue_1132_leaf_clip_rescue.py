"""#1132: indivisible leaves retry with the existing 8k serialization floor."""

import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.tokens import count_messages_tokens, count_tokens
from tests.test_issue_671_stub_first_exit import payload_of, tool_pair
from tests.test_issue_751_context_length_leaf_rescue import CONTEXT_LENGTH_ERROR, SUMMARY, _engine


@pytest.mark.parametrize("shape", ["single", "tool-group", "last-shrink", "final-shrink"])
@pytest.mark.parametrize("failure", ["context-length", "exception"])
def test_floor_retry_preserves_chunk_and_source_coverage(tmp_path, monkeypatch, shape, failure):
    calls = []
    # Keep the real escalation and circuit for size refusals; the host accepts <=16k tokens.
    def provider(**kwargs):
        policy = kwargs["messages"][0]["content"]
        text = kwargs["messages"][1]["content"]
        calls.append(count_tokens(text))
        if count_tokens(policy + text) > 16_000:
            raise RuntimeError(CONTEXT_LENGTH_ERROR)
        nonce = re.search(r'<lcm-summary nonce="([0-9a-f]{32})">', policy).group(1)
        reply = f'<lcm-summary nonce="{nonce}">\n{SUMMARY}\n</lcm-summary>'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    module = ModuleType("agent.auxiliary_client")
    module.call_llm = provider
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)
    engine = _engine(tmp_path, leaf_chunk_tokens=24_000, temporal_rollups_enabled=False,
                     large_output_externalization_threshold_chars=1_000_000)
    if failure == "exception":
        real = lcm_engine.summarize_with_escalation

        def summarize(**kwargs):
            if count_tokens(kwargs["text"]) > 16_000:
                calls.append(len(kwargs["text"]))
                raise TimeoutError("request timed out")
            return real(**kwargs)

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    chunk = ([{"role": "user", "content": payload_of(23_900)}] if shape != "tool-group"
             else tool_pair("large", payload_of(23_900)))
    if shape == "final-shrink":
        chunk = [{"role": "user", "content": payload_of(12_000)} for _ in range(3)]
    if shape == "last-shrink":
        chunk += [{"role": "assistant", "content": payload_of(1000)},
                  {"role": "user", "content": payload_of(1000)}]
    try:
        # A source under the floor still takes exactly one attempt, with unchanged serialization.
        small = [{"role": "user", "content": payload_of(1000)}]
        assert engine._summarize_leaf_chunk_with_rescue(small)[3:] == (1, 1)
        calls.clear()
        default_input = engine._serialize_messages(chunk)
        assert 16_000 < count_tokens(default_input) < 25_000
        rescued, source_tokens, summary, level, attempts = engine._summarize_leaf_chunk_with_rescue(chunk)
        assert level == 1, (level, attempts, calls)
        assert attempts == (3 if shape in ("last-shrink", "final-shrink") else 2)
        assert rescued == (chunk[:1] if shape in ("last-shrink", "final-shrink") else chunk)
        assert calls[-1] <= 8200  # Existing #611 clip allows 2% plus labels/markers.
        assert source_tokens == count_messages_tokens(rescued)
        assert "deterministic truncation" not in summary
        assert not engine._last_leaf_level_3_verbatim
        assert engine._summary_circuit_breaker._failures.get("", 0) == 0
        assert engine._serialize_messages(chunk) == default_input
    finally:
        engine.shutdown()


@pytest.mark.parametrize("leaf_chunk_tokens", [8000, 24_000])
def test_control_under_floor_keeps_default_serialization(tmp_path, monkeypatch, leaf_chunk_tokens):
    engine = _engine(tmp_path, leaf_chunk_tokens=leaf_chunk_tokens, temporal_rollups_enabled=False)
    chunk = [{"role": "user", "content": payload_of(1000)}]
    calls = []

    def summarize(**kwargs):
        calls.append(kwargs)
        return SUMMARY, 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    try:
        expected = engine._serialize_messages(chunk)
        assert engine._summarize_leaf_chunk_with_rescue(chunk) == (
            chunk, count_messages_tokens(chunk), SUMMARY, 1, 1)
        assert len(calls) == 1 and calls[0]["text"] == expected
        assert calls[0]["context_length_rescue"] is False
        assert calls[0]["verbatim_small_source"] is True
    finally:
        engine.shutdown()
