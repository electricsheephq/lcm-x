from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def _stub_summarizer(chunk, focus_topic=None, **_kwargs):
    tokens = count_messages_tokens(chunk)
    return chunk, tokens, f"[test summary: {len(chunk)} messages]", 1, 1


def test_sustained_count_protected_tail_pressure_forces_progress(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "fresh-tail-pressure.db"),
        context_threshold=0.10,
        fresh_tail_count=128,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=8_000,
        max_assembly_tokens=0,
        reserve_tokens_floor=0,
    )
    config.fresh_tail_pressure_yield_enabled = True
    config.fresh_tail_pressure_yield_min_observations = 3
    engine = LCMEngine(
        config=config
    )
    engine.on_session_start(
        "fresh-tail-pressure",
        platform="telegram",
        conversation_id="fresh-tail-pressure",
        context_length=200_000,
    )
    engine.threshold_tokens = 5_000
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        _stub_summarizer,
    )
    messages = [
        {"role": "user", "content": f"turn {index}: " + ("data " * 800)}
        for index in range(40)
    ]

    try:
        observed_tokens = count_messages_tokens(messages)
        assert observed_tokens >= engine.threshold_tokens
        assert engine._effective_assembly_token_cap() is None

        verdicts = [
            engine.should_compress_preflight(list(messages))
            for _attempt in range(3)
        ]
        assert verdicts == [False, False, True]

        result = engine.compress(
            list(messages),
            current_tokens=observed_tokens,
        )
        assert engine._last_compression_status == "compacted"
        assert len(result) < len(messages)
        assert count_messages_tokens(result) < observed_tokens
    finally:
        engine.shutdown()
