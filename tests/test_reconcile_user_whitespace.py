"""Replay identity of a user row ignores leading/trailing whitespace only.

Hermes' ACP adapter persists ``prompt.strip()`` and rewrites the live user dict to
it at turn end, after a same-turn compaction already stored the raw prompt. The
stored and live forms must share one identity; interior whitespace, and every
assistant/tool row, keep their exact identity.

All tests use synthetic messages.  No real session data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path: Path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine._session_id = "user-whitespace"
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.mark.parametrize("raw", ["fix the build\n", "  fix the build", "\n\tfix the build \r\n"])
def test_user_row_leading_or_trailing_whitespace_is_not_identity(engine, raw):
    trimmed = {"role": "user", "content": "fix the build"}
    live = {"role": "user", "content": raw}
    assert engine._message_replay_identity(live) == engine._message_replay_identity(trimmed)
    assert engine._message_replay_identity(live, stored_row=True) == engine._message_replay_identity(trimmed)


@pytest.mark.parametrize("other", ["fix  the build", "fix the\nbuild", "fix the build."])
def test_user_row_interior_difference_is_identity(engine, other):
    base = {"role": "user", "content": "fix the build"}
    assert engine._message_replay_identity({"role": "user", "content": other}) != engine._message_replay_identity(base)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "done\n"},
        {"role": "assistant", "content": " done"},
        {"role": "tool", "content": "ok\n", "tool_call_id": "call-1", "tool_name": "terminal"},
    ],
)
def test_assistant_and_tool_rows_keep_exact_whitespace(engine, message):
    trimmed = dict(message, content=message["content"].strip())
    identity = engine._message_replay_identity(message)
    assert identity[1] == message["content"]
    assert identity != engine._message_replay_identity(trimmed)
