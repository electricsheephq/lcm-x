"""#947: clipped leaf input cannot claim the whole-source verbatim exemption."""

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.summary_input_clip import CLIP_MARKER
from hermes_lcm.tokens import count_tokens

# counter -> (source, leaf_chunk_tokens, l3_truncate_tokens)
CASES = {
    # The issue's shape: a token-dense middle between whitespace ends, which are cheap only with tiktoken.
    "tiktoken": (" " * 4_000 + "\u6f22" * 20_000 + " " * 4_000, 1_500, 512),
    # CI has no tiktoken. With the character estimate, the legacy 2,800-character floor (about 700 tokens) still
    # fits a 1,024-token verbatim window, so the same routing defect is reachable.
    "estimate": ("x" * 40_000, 600, 1_024),
}


@pytest.fixture(params=sorted(CASES))
def case(request, tmp_path, monkeypatch):
    source, leaf_chunk_tokens, l3_truncate_tokens = CASES[request.param]
    encoder = (pytest.importorskip("tiktoken").get_encoding("cl100k_base")
               if request.param == "tiktoken" else None)
    monkeypatch.setattr(tokens, "_encoder", encoder)
    monkeypatch.setattr(tokens, "_encoder_ready", True)
    tokens._count_tokens_cached.cache_clear()
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), leaf_chunk_tokens=leaf_chunk_tokens,
        l3_truncate_tokens=l3_truncate_tokens, fresh_tail_count=2, context_threshold=0.001,
        threshold_full_sweep_enabled=False, max_assembly_tokens=100_000,
        extraction_enabled=False,
    ), hermes_home=str(tmp_path / "hermes-home"))
    try:
        yield instance, source
    finally:
        instance.shutdown()
        tokens._count_tokens_cached.cache_clear()


def test_t1_serializer_reports_clipping_and_preserves_wrapper(case):
    engine, source = case
    for content, expected in [(source, True), ("small whole source", False)]:
        messages = [{"role": "user", "content": content}]
        serialized, clipped = engine._serialize_messages_with_clip(messages)
        assert clipped is expected
        assert engine._serialize_messages(messages) == serialized


def test_t2_dense_middle_leaf_does_not_request_verbatim(case, monkeypatch):
    engine, source = case
    chunk = [{"role": "user", "content": source}]
    serialized = engine._serialize_messages(chunk)
    assert count_tokens(source) > 10 * engine._config.leaf_chunk_tokens
    assert CLIP_MARKER in serialized
    assert count_tokens(serialized) <= engine._config.l3_truncate_tokens
    calls = []

    def summarize(**kwargs):
        calls.append(kwargs)
        return "Earlier user turn.", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    result = engine._summarize_leaf_chunk_with_rescue(chunk)
    assert result[2:4] == ("Earlier user turn.", 1)
    assert len(calls) == 1
    assert calls[0]["verbatim_small_source"] is False


@pytest.mark.parametrize("clipped_source,expected_stop", [(True, True), (False, False)], ids=["clipped", "whole"])
def test_t3_compaction_route_stop_requires_a_whole_source(case, monkeypatch, clipped_source, expected_stop):
    engine, source = case
    content = source if clipped_source else "small whole source"
    engine.on_session_start("S", context_length=200_000, conversation_id="conv")
    monkeypatch.setattr(engine, "_summary_route_available", lambda: False)
    monkeypatch.setattr(engine, "_fit_can_rescue", lambda force_overflow: True)
    decisions = []
    original = engine._summary_route_stop_applies

    def route_stop(force_overflow, source_text=None):
        stopped = original(force_overflow, source_text)
        decisions.append((source_text, stopped))
        return stopped

    monkeypatch.setattr(engine, "_summary_route_stop_applies", route_stop)
    view = [{"role": "system", "content": "system prompt"},
            {"role": "user", "content": content},
            {"role": "user", "content": "fresh user turn"},
            {"role": "assistant", "content": "fresh reply"}]
    engine.ingest(view)
    engine.compress(view, current_tokens=engine.threshold_tokens + 1)
    assert decisions[0] == (None, True)  # entry: every route is refused
    assert decisions[1] == (None if expected_stop else "[USER]: " + content, expected_stop)
    leaves = engine._dag.get_session_nodes("S")
    assert bool(leaves) is not expected_stop
