"""Commit-proof identity of a user row ignores leading/trailing whitespace only.

Hermes' ACP adapter persists ``prompt.strip()`` and rewrites the adopted user dict
to it at turn end, after a same-turn compaction already stored and proved the raw
prompt. Positions a commit proof binds compare ``_proof_replay_identity``: the
stored and live forms of that row match. The replay identity itself stays exact,
so unbound store reconciliation never collapses a new exchange onto an old one
(#498). Interior whitespace and assistant/tool rows are never normalized.

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
def test_proof_identity_ignores_user_leading_or_trailing_whitespace(engine, raw):
    trimmed = {"role": "user", "content": "fix the build"}
    live = {"role": "user", "content": raw}
    assert engine._proof_replay_identity(live) == engine._proof_replay_identity(trimmed)
    # The replay identity used by unbound store reconciliation stays exact.
    assert engine._message_replay_identity(live) != engine._message_replay_identity(trimmed)


@pytest.mark.parametrize("other", ["fix  the build", "fix the\nbuild", "fix the build."])
def test_proof_identity_keeps_user_interior_differences(engine, other):
    base = {"role": "user", "content": "fix the build"}
    assert engine._proof_replay_identity({"role": "user", "content": other}) != engine._proof_replay_identity(base)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "done\n"},
        {"role": "assistant", "content": " done"},
        {"role": "tool", "content": "ok\n", "tool_call_id": "call-1", "tool_name": "terminal"},
    ],
)
def test_proof_identity_keeps_assistant_and_tool_whitespace(engine, message):
    trimmed = dict(message, content=message["content"].strip())
    identity = engine._proof_replay_identity(message)
    assert identity == engine._message_replay_identity(message)
    assert identity[1] == message["content"]
    assert identity != engine._proof_replay_identity(trimmed)
