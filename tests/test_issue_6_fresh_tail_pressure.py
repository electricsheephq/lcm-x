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


def _make_pressure_engine(tmp_path, monkeypatch, **overrides):
    config = LCMConfig(
        database_path=str(tmp_path / "pressure-review.db"),
        fresh_tail_count=128,
        leaf_chunk_tokens=200,
        **overrides,
    )
    engine = LCMEngine(config=config)
    engine._session_id = "pressure-review"
    engine.context_length = 200_000
    engine.threshold_tokens = 100
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        _stub_summarizer,
    )
    return engine


def _pressure_messages(count=12):
    return [
        {"role": "user", "content": f"turn {index}: " + ("data " * 80)}
        for index in range(count)
    ]


def test_preflight_yield_survives_compress_without_current_tokens(tmp_path, monkeypatch):
    engine = _make_pressure_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_pressure_yield_min_observations=1,
    )
    messages = _pressure_messages()
    try:
        assert engine.should_compress_preflight(list(messages)) is True

        compressed = engine.compress(list(messages))

        assert engine._last_compression_status == "compacted"
        assert compressed != messages
        assert count_messages_tokens(compressed) < count_messages_tokens(messages)
    finally:
        engine.shutdown()


def test_independent_preflight_work_clears_armed_blocked_verdict(tmp_path, monkeypatch):
    engine = _make_pressure_engine(tmp_path, monkeypatch)
    try:
        engine._pressure_yield_blocked_streak = 2
        with engine._fresh_tail_pressure_yield_invocation():
            engine._pressure_yield_tail_token_limit = 50
            engine._pressure_yield_invocation_verdict = "blocked"
            engine._mark_preflight_compression_requested()

        assert engine._pressure_yield_blocked_streak == 0
    finally:
        engine.shutdown()


def test_reset_crossing_scope_clears_post_reset_blocked_observation(tmp_path, monkeypatch):
    engine = _make_pressure_engine(tmp_path, monkeypatch)
    try:
        with engine._fresh_tail_pressure_yield_invocation():
            engine._clear_fresh_tail_pressure_yield_state()
            engine._pressure_yield_blocked_streak = 1
            engine._pressure_yield_invocation_verdict = "blocked"

        assert engine._pressure_yield_blocked_streak == 0
    finally:
        engine.shutdown()


def test_nested_pressure_relief_restores_outer_streak(tmp_path, monkeypatch):
    engine = _make_pressure_engine(tmp_path, monkeypatch)
    try:
        engine._pressure_yield_blocked_streak = 2
        with engine._fresh_tail_pressure_yield_invocation():
            with engine._fresh_tail_pressure_yield_invocation():
                engine._note_fresh_tail_pressure_relieved()
            assert engine._pressure_yield_blocked_streak == 2
    finally:
        engine.shutdown()


def test_yielded_placeholder_only_prefix_requests_cleanup(tmp_path, monkeypatch):
    engine = _make_pressure_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_pressure_yield_min_observations=1,
    )
    placeholder = engine._ignored_active_replay_placeholder("ignored content")
    digest = engine._active_replay_placeholder_digest(placeholder)
    assert digest is not None
    engine._generated_ignored_active_replay_placeholder_hashes = {digest}
    messages = [
        {"role": "assistant", "content": placeholder}
        for _index in range(12)
    ]
    try:
        assert count_messages_tokens(messages) >= engine.threshold_tokens
        assert engine.should_compress_preflight(list(messages)) is True
    finally:
        engine.shutdown()


def test_yielded_default_leaf_input_is_bounded(tmp_path, monkeypatch):
    captured_tokens = []

    def capture_summarizer(chunk, focus_topic=None, **_kwargs):
        captured_tokens.append(count_messages_tokens(chunk))
        return _stub_summarizer(chunk, focus_topic=focus_topic)

    engine = _make_pressure_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_pressure_yield_min_observations=1,
        dynamic_leaf_chunk_enabled=False,
    )
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        capture_summarizer,
    )
    messages = _pressure_messages(count=40)
    try:
        compressed = engine.compress(
            list(messages),
            current_tokens=count_messages_tokens(messages),
        )

        assert engine._last_compression_status == "compacted"
        assert compressed != messages
        assert count_messages_tokens(compressed) < count_messages_tokens(messages)
        assert captured_tokens
        assert captured_tokens[0] <= engine._config.leaf_chunk_tokens + 100
    finally:
        engine.shutdown()
