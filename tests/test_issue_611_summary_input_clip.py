"""#611: one shared summarizer input budget, with a v0.26.0 retention floor."""
from __future__ import annotations

import logging
import random

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.level3_repair as level3_repair
import hermes_lcm.summary_input_clip as summary_input_clip
import hermes_lcm.tokens as tokens_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import _deterministic_truncate
from hermes_lcm.externalize import is_externalized_placeholder
from hermes_lcm.summary_input_clip import CLIP_MARKER, clip_to_budget
from hermes_lcm.tokens import count_tokens

FACT = "FACT-611-MIDDLE-ZEBRA-4417"


def _long(chars: int, fact: str = FACT) -> str:
    filler = "lorem ipsum dolor sit amet "
    half = (chars - len(fact)) // 2
    body = filler * (chars // len(filler) + 2)
    return "HEAD-TAG " + body[:half - 9] + fact + body[:chars - half - len(fact) - 9] + " TAIL-TAG"


def _engine(tmp_path, **config) -> LCMEngine:
    settings = {"database_path": str(tmp_path / "lcm.db"), **config}
    return LCMEngine(config=LCMConfig(**settings), hermes_home=str(tmp_path / "hermes-home"))


def _legacy(text, kind="message"):
    if kind == "preview" or len(text) <= (500 if kind == "arguments" else 3_000):
        return text
    return text[:400] + "..." if kind == "arguments" else text[:2_000] + CLIP_MARKER + text[-800:]


@pytest.fixture(params=["fallback", "tiktoken"])
def clip_counter(request, monkeypatch):
    if request.param == "fallback":
        def loader():
            raise ImportError("tiktoken unavailable")
    else:
        tiktoken = pytest.importorskip("tiktoken")
        from tiktoken import load as tiktoken_load

        def offline_only(*args, **kwargs):
            raise OSError("encoder is not cached offline")

        monkeypatch.setattr(tiktoken_load, "read_file", offline_only)
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            pytest.skip(f"tiktoken encoder unavailable offline: {exc}")

        def loader():
            return encoder

    monkeypatch.setattr(tokens_mod, "_load_encoder", loader)
    monkeypatch.setattr(tokens_mod, "_encoder", None)
    monkeypatch.setattr(tokens_mod, "_encoder_ready", False)
    monkeypatch.setattr(tokens_mod, "_encoder_thread", None)
    monkeypatch.setattr(tokens_mod, "_encoder_generation", 0)
    tokens_mod._count_tokens_cached.cache_clear()
    tokens_mod._encoder_loader()
    yield
    tokens_mod._count_tokens_cached.cache_clear()


def test_uneven_density_clip_honors_token_share(clip_counter):
    text = "漢" * 9_500 + "a" * 81_000 + "漢" * 9_500
    budget = 20_000
    assert count_tokens(text) > budget
    kept, = clip_to_budget([text], budget)
    retained = kept.replace(CLIP_MARKER, "")
    assert count_tokens(retained) <= budget * 1.02
    assert len(retained) > 2_800


def test_uniform_clip_preserves_first_clip_with_one_recount(clip_counter, monkeypatch):
    text, budget = "abc " * 25_000, 20_000
    keep = len(text) * budget // count_tokens(text)
    assert 2_800 < keep < len(text)
    head = keep * 5 // 7
    retained = text[:head] + text[-(keep - head):]
    expected = text[:head] + CLIP_MARKER + text[-(keep - head):]
    calls = []

    def recount(value):
        calls.append(value)
        return count_tokens(value)

    monkeypatch.setattr(summary_input_clip, "count_tokens", recount)
    assert clip_to_budget([text], budget) == [expected]
    assert calls == [text, retained]


def test_overshooting_proportional_clip_returns_legacy_floor(clip_counter):
    text, budget = "漢" * 9_500 + "a" * 81_000 + "漢" * 9_500, 1_500
    assert len(text) * budget // count_tokens(text) > 2_800
    floor = _legacy(text)
    assert count_tokens(floor.replace(CLIP_MARKER, "")) > budget * 1.02
    assert clip_to_budget([text], budget) == [floor]


@pytest.mark.parametrize("budget", [1, 20, 100, 8_000, 20_000])
def test_random_leaves_keep_legacy_characters_and_bound_total_tokens(budget):
    rng = random.Random(611)
    sizes = [0, 400, 500, 501, 3_000, 3_001, 6_000, 10_000, 40_000, 120_000]
    for _ in range(16):
        kinds = ["message", "arguments", "preview"] + [rng.choice(["message", "arguments"])
                                                       for _ in range(rng.randint(0, 5))]
        lengths = [min(900, rng.choice(sizes)) if kind == "preview" else rng.choice(sizes) for kind in kinds]
        texts = [("word " * (length // 5 + 1))[:length] for length in lengths]
        clipped = clip_to_budget(texts, budget, kinds)
        legacy = [_legacy(text, kind) for text, kind in zip(texts, kinds)]
        for text, kind, kept in zip(texts, kinds, clipped):
            floor_chars = len(text) if kind == "preview" or len(text) <= (
                500 if kind == "arguments" else 3_000) else (400 if kind == "arguments" else 2_800)
            kept_chars = len(kept.replace(CLIP_MARKER, ""))
            if kind == "arguments" and len(text) > 500 and kept == text[:400] + "...":
                kept_chars -= 3
            assert kept_chars >= floor_chars
            assert kept != CLIP_MARKER
        # Fixed clip markers are allowed outside the share; this stricter check includes them.
        assert sum(map(count_tokens, clipped)) <= sum(map(count_tokens, legacy)) + budget * 1.02
        assert clipped == clip_to_budget(texts, budget, kinds)
        if sum(map(count_tokens, texts)) <= budget:
            assert clipped == texts


@pytest.mark.parametrize("kind,chars", [("message", 3_000), ("message", 3_001),
                                        ("arguments", 500), ("arguments", 501), ("preview", 900)])
def test_tiny_budget_keeps_the_legacy_floor_at_boundaries(kind, chars):
    text = ("word " * (chars // 5 + 1))[:chars]
    assert clip_to_budget([text], 1, [kind]) == [_legacy(text, kind)]


def test_fitting_texts_are_all_whole_at_the_budget_boundary():
    texts = [_long(10_000), _long(2_000), "preview " * 100]
    budget = sum(map(count_tokens, texts))
    assert clip_to_budget(texts, budget, ["message", "arguments", "preview"]) == texts


def test_single_40000_character_message_uses_about_8000_tokens():
    text = "abc " * 10_000
    assert len(text) == 40_000 and count_tokens(text) > 8_000
    kept, = clip_to_budget([text], 8_000)
    assert CLIP_MARKER in kept and len(kept.replace(CLIP_MARKER, "")) > 2_800
    assert abs(count_tokens(kept) - 8_000) <= 8_000 * 0.02


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_middle_fact_of_a_10000_char_message(tmp_path, role):
    engine = _engine(tmp_path, large_output_externalization_enabled=False)
    try:
        msg = {"role": role, "content": _long(10_000), "tool_call_id": "call_1"}
        serialized = engine._serialize_messages([msg])
        assert FACT in serialized and "HEAD-TAG" in serialized and "TAIL-TAG" in serialized
        assert CLIP_MARKER not in serialized
    finally:
        engine.shutdown()


def test_a_1mb_single_message_is_bounded(tmp_path):
    engine = _engine(tmp_path, leaf_chunk_tokens=8_000, large_output_externalization_enabled=False)
    try:
        for role in ("user", "assistant", "tool"):
            msg = {"role": role, "content": _long(1_000_000), "tool_call_id": "call_big"}
            serialized = engine._serialize_messages([msg])
            assert count_tokens(serialized) <= 8_000 * 1.02 + 40
            assert "HEAD-TAG" in serialized and "TAIL-TAG" in serialized
    finally:
        engine.shutdown()


def test_budget_uses_the_floor_for_small_text_and_share_for_big_text(tmp_path):
    engine = _engine(tmp_path, leaf_chunk_tokens=2_000)
    try:
        small, big = _long(4_000, "FACT-SMALL"), _long(40_000, "FACT-BIG")
        serialized = engine._serialize_messages([
            {"role": "user", "content": small}, {"role": "assistant", "content": big}])
        user_part, assistant_part = serialized.split("\n\n[ASSISTANT]: ")
        assert user_part == "[USER]: " + _legacy(small)
        assert len(assistant_part.replace(CLIP_MARKER, "")) > 2_800
        assert count_tokens(serialized) <= sum(count_tokens(_legacy(x)) for x in (small, big)) + 2_000 * 1.02 + 40
    finally:
        engine.shutdown()


def test_budget_keeps_a_chunk_within_the_budget_whole(tmp_path):
    engine = _engine(tmp_path, leaf_chunk_tokens=8_000)
    try:
        content = _long(20_000)
        assert count_tokens(content) < 8_000
        assert engine._serialize_messages([{"role": "user", "content": content}]) == "[USER]: " + content
    finally:
        engine.shutdown()


@pytest.mark.parametrize("budget", [1, 20_000])
def test_tool_arguments_share_budget_with_their_legacy_floor(tmp_path, budget):
    engine = _engine(tmp_path, leaf_chunk_tokens=budget)
    try:
        args = '{"text": "' + _long(2_000) + '"}'
        serialized = engine._serialize_messages([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "write", "arguments": args}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}])
        expected = _legacy(args, "arguments") if budget == 1 else args
        assert serialized == "[ASSISTANT]: \n[Tool calls:\n  write(" + expected + ")\n]\n\n[TOOL RESULT c1]: ok"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("budget", [1, 8_000])
def test_externalized_result_preview_is_outside_the_placeholder(tmp_path, budget):
    engine = _engine(tmp_path, leaf_chunk_tokens=budget, large_output_externalization_enabled=True,
                     large_output_externalization_threshold_chars=12_000)
    try:
        content = _long(30_000)
        serialized = engine._serialize_messages([{"role": "tool", "tool_call_id": "call_ext", "content": content}])
        label = "[TOOL RESULT call_ext]: "
        assert serialized.startswith(label)
        placeholder, preview = serialized[len(label):].split("\n[preview: ", 1)
        assert is_externalized_placeholder(placeholder) and len(placeholder.strip()) <= 512
        assert preview == content[:600] + " … " + content[-300:] + "]"
        assert FACT not in serialized
    finally:
        engine.shutdown()


def _fact_message() -> dict:
    return {"role": "user", "content": _long(10_000)}


def test_consumer_leaf_call_and_recorder(tmp_path, monkeypatch, caplog):
    seen = []

    def fake(**kwargs):
        seen.append(kwargs["text"])
        return "summary", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", fake)
    engine = _engine(tmp_path)
    try:
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine._summarize_leaf_chunk_with_rescue([_fact_message()])
        assert len(seen) == 1 and FACT in seen[0]
        lines = [r.getMessage() for r in caplog.records if "LCM leaf summary input:" in r.getMessage()]
        assert lines == [f"LCM leaf summary input: input_tokens={count_tokens(seen[0])} "
                         f"source_tokens={lcm_engine.count_messages_tokens([_fact_message()])} messages=1"]
    finally:
        engine.shutdown()


def test_consumer_extraction(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(lcm_engine, "extract_before_compaction",
                        lambda **kwargs: seen.append(kwargs["serialized_messages"]) or True)
    engine = _engine(tmp_path)
    try:
        engine._run_pre_compaction_extraction([_fact_message()])
        assert len(seen) == 1 and FACT in seen[0]
    finally:
        engine.shutdown()


def test_consumer_route_stop_verbatim_source(tmp_path, monkeypatch, caplog):
    seen = []
    real = lcm_engine.verbatim_source
    monkeypatch.setattr(lcm_engine, "verbatim_source", lambda text, bound: seen.append(text) or real(text, bound))
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), fresh_tail_count=2, leaf_chunk_tokens=8_000,
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
        assert "FACT-611-MIDDLE-ZEBRA-4417-0" in seen[0]
    finally:
        engine.shutdown()


def test_consumer_level3_repair(tmp_path, monkeypatch):
    seen = []

    def fake(**kwargs):
        seen.append(kwargs["text"])
        return "repaired summary", 1

    monkeypatch.setattr(level3_repair, "summarize_with_escalation", fake)
    engine = _engine(tmp_path)
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
        assert refusal == "" and len(seen) == 1 and FACT in seen[0]
    finally:
        engine.shutdown()
