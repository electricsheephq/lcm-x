"""#420/#428: host-shaped preflights must preserve sustained tail pressure."""

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


OVER = 12_000


def _stub_summarizer(chunk, focus_topic=None, **_kwargs):
    tokens = count_messages_tokens(chunk)
    return chunk, tokens, f"[test summary: {len(chunk)} messages]", 1, 1


def _make_engine(tmp_path, monkeypatch):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_pressure_yield.db")
    config.fresh_tail_count = 128
    engine = LCMEngine(config=config)
    engine._session_id = "pressure-yield-session"
    engine.context_length = 200_000
    engine.threshold_tokens = 5_000
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", _stub_summarizer)
    assert config.fresh_tail_pressure_yield_min_observations == 3
    return engine


def _messages(repetitions):
    return [
        {"role": "user", "content": f"turn {index}: " + "data " * repetitions}
        for index in range(40)
    ]


def test_preflight_estimate_under_keeps_streak_under_host_pressure(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch)
    messages = _messages(80)
    try:
        assert count_messages_tokens(messages) < engine.threshold_tokens < OVER
        engine.compress(list(messages), current_tokens=OVER)
        assert engine._pressure_yield_blocked_streak == 1
        engine.update_from_response({"prompt_tokens": OVER})

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 1
    finally:
        engine.shutdown()


def test_preflight_during_no_progress_hold_keeps_streak(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch)
    messages = _messages(800)
    observed = count_messages_tokens(messages)
    try:
        assert observed > engine.threshold_tokens
        engine.compress(list(messages), current_tokens=observed)
        assert engine._pressure_yield_blocked_streak == 1
        assert engine._no_progress_hold_active()
        engine.update_from_response({"prompt_tokens": observed})

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 1
    finally:
        engine.shutdown()


def _assert_host_shaped_sequence(engine, messages, observed):
    original_length = len(messages)
    compress_streaks = []
    preflight_streaks = []
    for _round in range(3):
        # The host next attempts compression after the hold window expires.
        engine._no_progress_hold = None
        messages = engine.compress(list(messages), current_tokens=observed)
        compress_streaks.append(engine._pressure_yield_blocked_streak)
        status = engine._last_compression_status
        if _round == 2:
            # Check the successful compress before any new pressure observations.
            break
        engine.update_from_response({"prompt_tokens": observed})
        for _turn in range(2):
            assert engine.should_compress_preflight(list(messages)) is False
            preflight_streaks.append(engine._pressure_yield_blocked_streak)

    trace = f"compress={compress_streaks}, preflight={preflight_streaks}, status={status}"
    assert compress_streaks[:2] == [1, 2], trace
    assert compress_streaks[2] == 0, trace
    assert preflight_streaks == [1, 1, 2, 2], trace
    assert len(messages) < original_length, trace
    assert status == engine._last_compression_status == "compacted", trace


def test_host_shaped_sequence_yields_estimate_under(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch)
    messages = _messages(80)
    try:
        assert count_messages_tokens(messages) < engine.threshold_tokens < OVER
        _assert_host_shaped_sequence(engine, messages, OVER)
    finally:
        engine.shutdown()


def test_host_shaped_sequence_yields_estimate_over(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch)
    messages = _messages(800)
    observed = count_messages_tokens(messages)
    try:
        assert observed > engine.threshold_tokens
        _assert_host_shaped_sequence(engine, messages, observed)
    finally:
        engine.shutdown()


def test_preflight_relieves_when_host_and_estimate_under(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch)
    messages = _messages(80)
    try:
        engine.compress(list(messages), current_tokens=OVER)
        assert engine._pressure_yield_blocked_streak == 1
        engine.update_from_response({"prompt_tokens": 1_000})

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 0
    finally:
        engine.shutdown()


def test_replay_diff_preflight_counts_host_pressure(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch)
    messages = _messages(80)
    try:
        engine.compress(list(messages), current_tokens=OVER)
        assert engine._pressure_yield_blocked_streak == 1
        engine._no_progress_hold = None
        original_ingest = engine._ingest_messages

        def ingest_with_a_diff(incoming):
            replay = [dict(message) for message in original_ingest(incoming)]
            replay[1]["content"] += " (replayed)"
            assert count_messages_tokens(replay) < engine.threshold_tokens
            return replay

        monkeypatch.setattr(engine, "_ingest_messages", ingest_with_a_diff)
        engine.update_from_response({"prompt_tokens": OVER})

        # Observation uses host pressure; the request gate uses replay tokens.
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 2
    finally:
        engine.shutdown()


def test_preflight_returns_unchanged_under_host_pressure(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch)
    messages = _messages(80)
    try:
        engine.compress(list(messages), current_tokens=OVER)
        hold = engine._no_progress_hold
        assert hold is not None
        for held in (False, True):
            engine._no_progress_hold = hold if held else None
            engine.update_from_response({"prompt_tokens": 0})
            without_host_pressure = engine.should_compress_preflight(list(messages))
            engine.update_from_response({"prompt_tokens": OVER})
            with_host_pressure = engine.should_compress_preflight(list(messages))

            assert with_host_pressure is without_host_pressure is False
    finally:
        engine.shutdown()
