"""#2583 stays open in v0.30.0: a turn whose tool-call arguments nest about twice the recursion limit deep.

Ingest protection walks the parsed arguments recursively, so arguments nested about 2,000 levels deep (at the default
recursion limit of 1,000) raise inside ingest and no row of the turn is stored. This strict xfail pins that loss on
the main-session ingest; a shallow control stores every row. A fix flips it. On Python 3.11 the JSON parser itself
stops at the recursion limit, the arguments stay unparsed text and the turn is stored, so the pin applies from 3.12.
"""

import sys

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _stored_rows(tmp_path, arguments):
    home = tmp_path / "hermes"
    engine = LCMEngine(config=LCMConfig(database_path=str(home / "lcm.db"),
                                        large_output_externalization_path=str(home / "externalized")),
                       hermes_home=str(home))
    try:
        engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
        engine.ingest([
            {"role": "user", "content": "go", "timestamp": 1.0},
            {"role": "assistant", "content": "", "timestamp": 2.0, "tool_calls": [
                {"id": "k1", "type": "function", "function": {"name": "terminal", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "k1", "content": "ok", "timestamp": 3.0},
            {"role": "assistant", "content": "done", "timestamp": 4.0},
        ])
        return len(engine._store.get_session_messages("S1"))
    finally:
        engine.shutdown()


def test_a_shallow_argument_turn_is_stored(tmp_path):
    assert _stored_rows(tmp_path, "[[0]]") == 4


@pytest.mark.xfail(sys.version_info >= (3, 12), strict=True, raises=AssertionError,
                   reason="#2583: arguments nested about 2,000 levels deep make ingest store no row of the turn")
def test_a_deeply_nested_argument_turn_is_stored(tmp_path):
    stored = _stored_rows(tmp_path, "[" * 2000 + "0" + "]" * 2000)
    if 0 < stored < 4:  # a partial store is a change of the pinned shape, not the expected failure
        pytest.fail(f"partial store: {stored} of 4 rows")
    assert stored == 4
