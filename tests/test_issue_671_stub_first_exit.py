"""#671: stub tiers (10k first sight, 2k aged at compaction) and the stub-first exit."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_tokens

STUB = "[Externalized tool output:"


def payload_of(tokens: int, word: str = "lorem") -> str:
    """About ``tokens`` tokens by the plugin's own counter (tiktoken or the character estimate)."""
    per_word = count_tokens((word + " ") * 1_000) / 1_000
    text = (word + " ") * max(1, round(tokens / per_word))
    assert abs(count_tokens(text) - tokens) <= max(10, tokens // 20)
    return text


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def build(**overrides):
        settings = dict(
            database_path=str(tmp_path / f"issue-671-{len(engines)}.db"),
            fresh_tail_count=2,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=1_000_000,
            large_output_active_replay_stubbing_enabled=True,
        )
        settings.update(overrides)
        engine = LCMEngine(config=LCMConfig(**settings), hermes_home=str(tmp_path / "hermes"))
        engine.on_session_start(
            f"issue-671-{len(engines)}",
            conversation_id=f"issue-671-conversation-{len(engines)}",
            context_length=200_000,
        )
        engines.append(engine)
        return engine

    yield build
    for engine in engines:
        engine.shutdown()


def tool_pair(call_id, payload, tool_name="read_file"):
    return [
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": payload},
    ]


def tool_content(result, call_id):
    return next(
        m["content"] for m in result if m.get("role") == "tool" and m.get("tool_call_id") == call_id
    )


# -- commit 1: stub tiers -------------------------------------------------------------------------


def test_defaults_are_10k_first_sight_and_2k_aged(monkeypatch):
    config = LCMConfig()
    assert config.large_output_active_replay_stub_threshold_tokens == 10_000
    assert config.large_output_active_replay_stub_aged_threshold_tokens == 2_000
    monkeypatch.setenv("LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_AGED_THRESHOLD_TOKENS", "3000")
    assert LCMConfig.from_env().large_output_active_replay_stub_aged_threshold_tokens == 3_000


def test_aged_tier_stubs_3k_outside_the_fresh_tail_and_keeps_it_inside(make_engine):
    engine = make_engine()
    aged = payload_of(3_000)
    tail = tool_pair("old-call", aged) + tool_pair("fresh-call", aged)

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert tool_content(result, "old-call").startswith(STUB)
    assert tool_content(result, "fresh-call") == aged  # the fresh tail never gets an aged stub


def test_aged_tier_zero_uses_the_first_sight_threshold(make_engine):
    engine = make_engine(large_output_active_replay_stub_aged_threshold_tokens=0)
    aged = payload_of(3_000)
    tail = tool_pair("old-call", aged) + tool_pair("fresh-call", "fresh")

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert tool_content(result, "old-call") == aged


def test_aged_tier_never_exceeds_the_first_sight_threshold(make_engine):
    engine = make_engine(
        large_output_active_replay_stub_threshold_tokens=50,
        large_output_active_replay_stub_aged_threshold_tokens=2_000,
    )
    tail = tool_pair("old-call", payload_of(300)) + tool_pair("fresh-call", "fresh")

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert tool_content(result, "old-call").startswith(STUB)


def test_first_sight_keeps_6k_whole_and_stubs_11k(make_engine):
    engine = make_engine(fresh_tail_count=32)
    small, large = payload_of(6_000), payload_of(11_000, word="ipsum")
    messages = [{"role": "system", "content": "system"}, *tool_pair("six", small), *tool_pair("eleven", large)]

    replay = engine._ingest_messages(messages)

    assert tool_content(replay, "six") == small
    assert tool_content(replay, "eleven").startswith(STUB)


def test_with_both_opt_in_flags_off_nothing_is_stubbed(make_engine):
    engine = make_engine(
        large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False,
    )
    aged = payload_of(3_000)
    tail = tool_pair("old-call", aged) + tool_pair("fresh-call", "fresh")

    result = engine._assemble_context({"role": "system", "content": "system"}, tail)

    assert tool_content(result, "old-call") == aged
