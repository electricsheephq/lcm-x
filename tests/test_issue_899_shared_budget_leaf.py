"""#899: clipped input must not become a publication-eligible verbatim leaf."""

import random

import pytest

import hermes_lcm.escalation as escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_tokens

CLIP_MARKER = "\n...[truncated]...\n"  # Literal also runs on the pre-#611 base.


@pytest.mark.parametrize("leaf_budget", [8_000, 20_000])
@pytest.mark.parametrize("externalize", [False, True])
@pytest.mark.parametrize("chars", [3_001, 6_000, 10_000, 40_000])
def test_clipped_input_is_never_a_verbatim_level3_leaf(tmp_path, monkeypatch, leaf_budget, externalize, chars):
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: "oversized reply " * 40_000)
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), leaf_chunk_tokens=leaf_budget, l3_truncate_tokens=512,
        summary_model="m1", fresh_tail_count=2, context_threshold=0.001, max_assembly_tokens=100_000,
        threshold_full_sweep_enabled=False, large_output_externalization_enabled=externalize,
        large_output_externalization_threshold_chars=12_000), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("s", platform="test", context_length=200_000, conversation_id="conv")
    try:
        message = {"role": "tool", "tool_call_id": "c", "content": ("word " * (chars // 5 + 1))[:chars]}
        serialized = engine._serialize_messages([message])
        _, _, summary, level, _ = engine._summarize_leaf_chunk_with_rescue([message])
        if level == 3:
            assert CLIP_MARKER not in summary
            if CLIP_MARKER in serialized:
                assert summary != serialized and not engine._last_leaf_level_3_verbatim
        if CLIP_MARKER in serialized:
            assert leaf_budget > 2 * engine._config.l3_truncate_tokens
            assert count_tokens(serialized) > 2 * engine._config.l3_truncate_tokens
        view = [message, {"role": "user", "content": "fresh user"},
                {"role": "assistant", "content": "fresh reply"}]
        engine.ingest(view)
        forced, original = [], engine._should_force_overflow_recovery
        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            forced.append(result)
            return result
        monkeypatch.setattr(engine, "_should_force_overflow_recovery", observe)
        engine.compress(view, current_tokens=100_001, force=True)
        assert any(forced)  # The invocation-scoped flag is reset by compress's finally.
        for leaf in engine._dag.get_session_nodes("s"):
            if engine._dag.get_node_provenance(leaf.node_id).get("escalation_level") == 3:
                assert CLIP_MARKER not in leaf.summary
                assert CLIP_MARKER not in serialized or leaf.summary != serialized
    finally:
        engine.shutdown()


def test_shared_budget_random_middle_determinism_and_forced_input(tmp_path, monkeypatch):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "bounds.db"), leaf_chunk_tokens=8_000, fresh_tail_count=2,
        context_threshold=0.001, max_assembly_tokens=100_000, threshold_full_sweep_enabled=False,
        large_output_externalization_enabled=False), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("bounds", platform="test", context_length=200_000, conversation_id="bounds")
    try:
        rng = random.Random(611)
        for _ in range(16):
            messages = [{"role": rng.choice(["user", "assistant"]), "content": "word " * rng.randint(100, 30_000)}
                        for _ in range(rng.randint(1, 8))]
            serialized = engine._serialize_messages(messages)
            fixed = "\n\n".join(f"[{m['role'].upper()}]: " + CLIP_MARKER for m in messages)
            legacy = [m["content"] if len(m["content"]) <= 3_000 else
                      m["content"][:2_000] + CLIP_MARKER + m["content"][-800:] for m in messages]
            assert count_tokens(serialized) <= sum(map(count_tokens, legacy)) + 8_000 * 1.02 + count_tokens(fixed)
            assert serialized == engine._serialize_messages(messages)
            for role in ("user", "assistant"):
                assert serialized.count(f"[{role.upper()}]: ") == sum(m["role"] == role for m in messages)
        content = "head " + "lorem ipsum " * 400 + " FACT-611-MIDDLE " + "dolor sit " * 400 + " tail"
        assert engine._serialize_messages([{"role": "user", "content": content}]) == "[USER]: " + content
        arguments = '{"text": "' + content + '"}'
        pair = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "function": {"name": "write", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "c", "content": "ok"}]
        assert "FACT-611-MIDDLE" in engine._serialize_messages(pair)
        seen = []
        monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation",
                            lambda **kwargs: (seen.append(kwargs["text"]) or "summary", 1))
        old = [{"role": "user" if i % 2 == 0 else "assistant",
                "content": f"chunk-{i} " + "abc " * 1_250, "timestamp": float(i + 1)} for i in range(30)]
        assert sum(count_tokens(m["content"]) for m in old) > 30_000
        view = old + [{"role": "user", "content": "fresh user"}, {"role": "assistant", "content": "fresh reply"}]
        engine.ingest(view)
        assert engine._should_force_overflow_recovery(200_000, view)
        engine.compress(view, current_tokens=200_000, force=True)
        assert seen
        originals = {m["content"].split(" ", 1)[0]: m["content"] for m in old}
        for text in seen:
            parts = [part.split(": ", 1)[1] for part in text.split("\n\n")]
            assert all(len(part.replace(CLIP_MARKER, "")) >= 2_800 for part in parts)
            sources = [originals[part.split(" ", 1)[0]] for part in parts]
            legacy_total = sum(count_tokens(src[:2_000] + CLIP_MARKER + src[-800:]) for src in sources)
            assert count_tokens(text) <= legacy_total + 8_000 * 1.02 + 100
    finally:
        engine.shutdown()
