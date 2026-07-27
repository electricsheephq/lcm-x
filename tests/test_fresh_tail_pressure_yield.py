"""Fresh-tail pressure yield: compaction must not deadlock inside the count tail.

Regression tests for #441 (same class as #414): a count-protected fresh tail
that covers the session's whole token mass made every compaction attempt no-op
("no eligible raw backlog outside fresh tail" below the count,
"raw backlog outside fresh tail is below leaf chunk threshold" just above it)
while the host reported over-threshold pressure every turn, until the session
died at the provider hard limit.
"""

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def _fat_user(index, chars=4000):
    return {"role": "user", "content": f"turn {index}: " + ("data " * (chars // 5))}


def _tiny_user(index):
    return {"role": "user", "content": f"small turn {index}"}


def _stub_summarizer(chunk, focus_topic=None):
    tokens = count_messages_tokens(chunk)
    return chunk, tokens, f"[test summary: {len(chunk)} messages]", 1, 1


def _make_engine(tmp_path, monkeypatch, **config_overrides):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_pressure_yield.db")
    for key, value in config_overrides.items():
        setattr(config, key, value)
    engine = LCMEngine(config=config)
    engine._session_id = "pressure-yield-session"
    engine.context_length = 200_000
    engine.threshold_tokens = 5_000
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", _stub_summarizer)
    return engine


def test_count_tail_covering_everything_yields_under_pressure(tmp_path, monkeypatch):
    # 40 messages, all inside a 128-count tail: on an unfixed engine this is
    # a guaranteed "no eligible raw backlog outside fresh tail" no-op forever.
    engine = _make_engine(tmp_path, monkeypatch, fresh_tail_count=128)
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)
        assert observed > engine.threshold_tokens

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "compacted"
        assert len(compressed) < len(messages)
        assert count_messages_tokens(compressed) < observed
    finally:
        engine.shutdown()


def test_backlog_below_leaf_chunk_yields_under_pressure(tmp_path, monkeypatch):
    # A couple of tiny messages sit outside the count tail; everything heavy is
    # protected. On an unfixed engine this no-ops with "below leaf chunk
    # threshold" forever.
    engine = _make_engine(tmp_path, monkeypatch, fresh_tail_count=10)
    try:
        messages = [_tiny_user(0), _tiny_user(1)] + [_fat_user(i) for i in range(10)]
        observed = count_messages_tokens(messages)
        assert observed > engine.threshold_tokens

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "compacted"
        assert len(compressed) < len(messages)
    finally:
        engine.shutdown()


def test_no_pressure_preserves_count_tail_noop(tmp_path, monkeypatch):
    # Same deadlock topology, but the host reports no over-threshold pressure:
    # behavior must stay exactly the pre-fix no-op.
    engine = _make_engine(tmp_path, monkeypatch, fresh_tail_count=128)
    try:
        engine.threshold_tokens = 10_000_000
        messages = [_fat_user(i) for i in range(40)]

        compressed = engine.compress(list(messages))

        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == (
            "no eligible raw backlog outside fresh tail"
        )
        assert len(compressed) == len(messages)
    finally:
        engine.shutdown()


def test_kill_switch_preserves_noop_under_pressure(tmp_path, monkeypatch):
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_enabled=False,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == (
            "no eligible raw backlog outside fresh tail"
        )
        assert len(compressed) == len(messages)
    finally:
        engine.shutdown()


def test_preflight_advertises_compaction_under_deadlock_pressure(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch, fresh_tail_count=128)
    try:
        messages = [_fat_user(i) for i in range(40)]
        assert count_messages_tokens(messages) > engine.threshold_tokens

        assert engine.should_compress_preflight(list(messages)) is True
    finally:
        engine.shutdown()


def test_explicit_token_cap_still_wins_when_smaller(tmp_path, monkeypatch):
    # An operator-configured fresh_tail_max_tokens below the derived yield cap
    # keeps bounding the tail; the yield must not loosen it.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_max_tokens=500,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "compacted"
        assert len(compressed) < len(messages)
    finally:
        engine.shutdown()
