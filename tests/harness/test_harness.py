"""Contract tests for the shared behavioral harness."""

from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from tests.harness import (
    HostCall, ScriptedSummarizer, assistant, call, engine_context, host_call,
    make_engine, parallel_batch, pin_char_counter, prose, sequential_pairs,
    stored_rows, tool, user,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    pin_char_counter(monkeypatch)
    ScriptedSummarizer().install(monkeypatch)


def test_make_engine_preserves_every_product_default(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "999")
    engine = make_engine(tmp_path)
    try:
        defaults = LCMConfig()
        for field in fields(defaults):
            expected = str(tmp_path / "lcm.db") if field.name == "database_path" else getattr(defaults, field.name)
            assert getattr(engine._config, field.name) == expected, field.name
        assert engine._hermes_home == str(tmp_path / "home")
        assert engine.current_session_id == ""
    finally:
        engine.shutdown()


def test_make_engine_binding_and_overrides(tmp_path):
    with engine_context(tmp_path, window=32_768, session_id="session", conversation_id="conversation",
                        platform="telegram", fresh_tail_count=7) as engine:
        assert engine.context_length == 32_768
        assert engine.current_session_id == "session"
        assert engine.current_session_platform == "telegram"
        assert engine._conversation_id == "conversation"
        assert engine._config.fresh_tail_count == 7
    with pytest.raises(AttributeError, match="fresh_tial_count"):
        make_engine(tmp_path, fresh_tial_count=7)


@pytest.mark.parametrize("raises", [False, True])
def test_engine_context_always_shuts_down(tmp_path, monkeypatch, raises):
    with pytest.raises(ValueError) if raises else nullcontext():
        with engine_context(tmp_path) as engine:
            shutdown = Mock(wraps=engine.shutdown)
            monkeypatch.setattr(engine, "shutdown", shutdown)
            if raises:
                raise ValueError("test body failed")
    shutdown.assert_called_once_with()
    assert engine._stable_use_closed


def test_pin_char_counter_is_offline_alias_safe_and_reversible(monkeypatch):
    class Encoder:
        def encode(self, text):
            return [1] * 99

    # Seed a disagreeing encoder cache, then prove the pin neither uses nor poisons it. The seam is fixed too,
    # so neither test order nor a slow real encoder load decides what the unpinned counter uses.
    encoder = Encoder()
    monkeypatch.setattr(tokens, "_encoder", encoder)
    monkeypatch.setattr(tokens, "_encoder_ready", True)
    monkeypatch.setattr(tokens, "_get_encoder", lambda: encoder)
    assert tokens.count_tokens("cache probe") == 99
    cache = tokens._count_tokens_cached
    with monkeypatch.context() as pinned:
        pin_char_counter(pinned)
        for text in ("", "cache probe", "界" * 12):
            expected = tokens._fallback_token_estimate(text)
            assert tokens.count_tokens(text) == expected
            assert engine_module.count_tokens(text) == expected
            assert escalation.count_tokens(text) == expected
        assert survival_fit._host_estimate([user("text")]) is None
        assert tokens._get_encoder() is None
    assert tokens._count_tokens_cached is cache
    assert tokens.count_tokens("cache probe") == 99


def test_healthy_summarizer_uses_real_escalation(monkeypatch):
    provider = ScriptedSummarizer().install(monkeypatch)
    source = prose(4_000)
    summary, level = escalation.summarize_with_escalation(
        source, source_tokens=tokens.count_tokens(source), token_budget=100, model="test-model")
    assert (summary, level) == (provider.healthy_reply, 1)
    assert len(provider.calls) == 1
    seen = provider.calls[0]
    assert seen.model == "test-model" and seen.max_tokens >= 100
    assert seen.prompt_length > len(source)


def test_outage_opens_real_circuit_at_production_failure_threshold(tmp_path, monkeypatch):
    provider = ScriptedSummarizer("outage").install(monkeypatch)
    with engine_context(tmp_path) as engine:
        breaker = engine._summary_circuit_breaker
        source = prose(4_000)
        for _ in range(breaker.failure_threshold):
            escalation.summarize_with_escalation(
                source, source_tokens=tokens.count_tokens(source), token_budget=100,
                circuit_breaker=breaker)
        assert len(provider.calls) == breaker.failure_threshold
        assert not breaker.allows("")


def test_scripted_summarizer_repeats_last_reply_and_supports_callback(monkeypatch):
    provider = ScriptedSummarizer("scripted", replies=[None, "accepted"]).install(monkeypatch)
    assert [escalation._call_llm_for_summary("prompt", 123) for _ in range(3)] == [None, "accepted", "accepted"]
    fn = Mock(return_value="custom")
    provider = ScriptedSummarizer("scripted", fn=fn).install(monkeypatch)
    assert escalation._call_llm_for_summary("source", 42, model="m", timeout=5, reasoning_effort="low") == "custom"
    fn.assert_called_once_with("source", 42, model="m", timeout=5, reasoning_effort="low")
    assert provider.calls == [("m", 42, 6)]
    with pytest.raises(ValueError):
        ScriptedSummarizer("scripted")
    with pytest.raises(ValueError):
        ScriptedSummarizer("typo")


def test_user_shape():
    assert user("hello") == {"role": "user", "content": "hello"}


def test_assistant_shape():
    assert assistant("hello") == {"role": "assistant", "content": "hello"}
    calls = [call("c")]
    row = assistant("", calls=calls)
    assert row == {"role": "assistant", "content": "", "tool_calls": calls}
    assert row["tool_calls"] is not calls


def test_call_shape():
    assert call("c") == {"id": "c", "type": "function", "function": {"name": "probe", "arguments": "{}"}}
    assert call("c", "read", '{"key": 1}')["function"] == {"name": "read", "arguments": '{"key": 1}'}


def test_tool_shape():
    assert tool("c", "result") == {"role": "tool", "tool_call_id": "c", "content": "result",
                                    "name": "probe", "tool_name": "probe"}
    assert tool("c", "result", "read")["tool_name"] == "read"


def test_prose_is_sized_seeded_and_can_preserve_a_sentinel():
    text = prose(1100, seed=12, sentinel="⟦objective⟧")
    assert len(text) == 1100 and text.startswith("⟦objective⟧")
    assert text == prose(1100, seed=12, sentinel="⟦objective⟧")
    assert prose(1100, seed=12) != prose(1100, seed=13)
    assert prose(0) == ""
    with pytest.raises(ValueError):
        prose(1, sentinel="too long")


def test_sequential_pairs_preserve_call_result_order():
    rows = sequential_pairs(3, 100)
    assert rows[0]["role"] == "user" and len(rows) == 7
    for i in range(3):
        request, result = rows[1 + 2 * i:3 + 2 * i]
        assert request["tool_calls"][0]["id"] == result["tool_call_id"] == f"c{i}"
        assert len(request["content"]) == len(result["content"]) == 100
    assert rows == sequential_pairs(3, 100)


def test_parallel_batch_preserves_one_call_group():
    rows = parallel_batch(3, 100)
    assert [row["role"] for row in rows] == ["user", "assistant", "tool", "tool", "tool"]
    assert [c["id"] for c in rows[1]["tool_calls"]] == [r["tool_call_id"] for r in rows[2:]]
    assert all(len(row["content"]) == 100 for row in rows[2:])


def test_host_call_fits_default_engine_in_32768_window(tmp_path, monkeypatch):
    provider = ScriptedSummarizer().install(monkeypatch)
    view = sequential_pairs(60, 1100)
    assert tokens.count_messages_tokens(view) > 32_768
    with engine_context(tmp_path, window=32_768) as engine:
        result = host_call(engine, view)
        assert isinstance(result, HostCall)
        assert result.compressed and provider.calls
        assert result.sent_tokens == tokens.count_messages_tokens(result.view) <= engine.context_length
        assert not result.overflow_failed
        # Every stored row keeps its role, content and tool identity: call ids, the calls and the tool name.
        def shape(row):
            return (row["role"], row["content"], row.get("tool_call_id"), row.get("tool_calls") or None,
                    row.get("tool_name"))
        assert [shape(row) for row in stored_rows(engine)] == [shape(row) for row in view]


def test_make_engine_rejects_a_database_path_override(tmp_path):
    with pytest.raises(TypeError, match="database_path"):
        make_engine(tmp_path, database_path=str(tmp_path / "elsewhere.db"))
    assert not (tmp_path / "elsewhere.db").exists()


@pytest.mark.parametrize("pressure,retry,window,expected_calls", [
    (True, True, 10, 2), (True, False, 10, 1), (False, True, 10, 1),
    (False, True, 1000, 0), (False, True, 0, 0),
])
def test_host_call_counts_and_retries_at_most_once(pressure, retry, window, expected_calls):
    view = [user(prose(100))]
    n = tokens.count_messages_tokens(view)
    engine = SimpleNamespace(context_length=window, ingest=Mock(), should_compress=Mock(return_value=pressure),
                             compress=Mock(return_value=view), _last_overflow_recovery_failed=True)
    result = host_call(engine, view, retry_on_overflow=retry)
    engine.ingest.assert_called_once_with(view)
    engine.should_compress.assert_called_once_with(n)
    assert engine.compress.call_count == expected_calls
    assert result.sent_tokens == n and result.overflow_failed is True
    assert result.compressed is bool(expected_calls)
    assert result.retried is bool(retry and window == 10)
    if result.retried:
        engine.compress.assert_called_with(view, current_tokens=n, bypass_cooldown=True)


@pytest.mark.parametrize("pressure,blocked,wants,expected_calls", [
    (False, None, True, 1), (False, None, False, 0), (False, "cooldown", True, 0), (True, None, False, 1),
])
def test_host_call_asks_the_preflight_gate_below_threshold(pressure, blocked, wants, expected_calls):
    view = [user(prose(100))]
    n = tokens.count_messages_tokens(view)
    engine = SimpleNamespace(context_length=1000, ingest=Mock(), should_compress=Mock(return_value=pressure),
                             should_compress_info=Mock(return_value=(pressure, blocked)),
                             should_compress_preflight=Mock(return_value=wants), compress=Mock(return_value=view))
    result = host_call(engine, view)
    # As the host: the gate is asked only below the threshold and only when nothing blocks compression.
    assert engine.should_compress_preflight.call_count == (not pressure and not blocked)
    assert engine.compress.call_count == expected_calls and result.compressed is bool(expected_calls)
    if expected_calls:
        engine.compress.assert_called_once_with(view, current_tokens=n)


def test_pinned_counter_ignores_an_encoder_that_finishes_loading_later(monkeypatch):
    pin_char_counter(monkeypatch)
    expected = tokens.count_tokens("an uncached sentence for the fallback estimate")

    class Late:
        def encode(self, text, **kwargs):
            raise AssertionError("the pinned counter used a late encoder")

    monkeypatch.setattr(tokens, "_encoder", Late())  # an earlier test's loader thread completing now
    assert tokens.count_tokens("another uncached sentence for the fallback estimate") == \
        tokens._fallback_token_estimate("another uncached sentence for the fallback estimate")
    assert expected == tokens._fallback_token_estimate("an uncached sentence for the fallback estimate")


def test_stored_rows_reads_current_or_explicit_session(tmp_path):
    with engine_context(tmp_path, window=32_768) as engine:
        engine.ingest([user("first session")])
        original = stored_rows(engine)
        assert original == engine._store.get_session_messages("S")
        engine.on_session_start("other", platform="cli", context_length=32_768)
        engine.ingest([user("second session")])
        assert stored_rows(engine)[0]["content"] == "second session"
        assert stored_rows(engine, "S") == original
