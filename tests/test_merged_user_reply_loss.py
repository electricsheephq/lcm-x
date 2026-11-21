"""A first reply with tool calls is lost when the host merges two user rows before the cursor
and the audit's presence check gives up (the reference agent's topic-9 loss, 2026-11-21).

The host (Hermes `_merge_consecutive_users`) folds B into A's dict in place and drops B, after the
start-of-turn ingest stored A and B. At turn end the pre-cursor audit finds the merged row unstored, skips
the reply M1 (it carries tool calls), and the call-id presence check gives up past its window. The
give-up fallback then marks every other index before the old cursor as stored, which is exactly M1.
"""

from __future__ import annotations

from collections import Counter

import pytest

from hermes_lcm import identity_anchor
from tests.test_anchor_unstored_before_cursor import (  # noqa: F401 (autouse fixture)
    _a, _engine, _fresh_presence_notices, _key, _result, _stored, _u,
)

U_A = "0123456789abcdef0123456789abcdef"
U_B = "fedcba9876543210fedcba9876543210"


def _reply_with_calls(ts) -> dict:
    return {"role": "assistant", "content": "Let me check both files.", "timestamp": float(ts), "tool_calls": [
        {"id": "call_m1_a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
        {"id": "call_m1_b", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b.txt"}'}},
    ]}


@pytest.mark.parametrize("window", [1, None], ids=["presence-window-exceeded", "control"])
def test_first_reply_with_calls_survives_a_merged_user_turn(tmp_path, monkeypatch, window):
    engine = _engine(tmp_path)
    try:
        history = []
        for index in range(3):
            history += [_u(f"T{index} user turn", 1000 + index), _a(f"reply to T{index}", 1100 + index)]
        engine.ingest(list(history))
        prompt_a = {**_u("A: first part of the new turn", 2000), "message_uid": U_A}
        prompt_b = _u("B: second part of the new turn", 2001)  # no uid yet: the host mints it at first DB insert
        engine.ingest([*history, prompt_a, prompt_b])  # the start-of-turn ingest stores A and B
        assert Counter(_key(row) for row in _stored(engine)) == Counter(_key(row) for row in [*history, prompt_a, prompt_b])

        # The host merges B into A's dict in place, records B's uid as absorbed, and drops B.
        original_a = prompt_a["content"]
        prompt_a["content"] = original_a + "\n\n" + prompt_b["content"]
        prompt_a["_absorbed_message_uids"] = [U_B]
        prompt_a["_merged_turn_prefix"] = original_a
        m1 = _reply_with_calls(2002)
        r1a = _result("call_m1_a", "contents of a.txt", 2003)
        r1b = _result("call_m1_b", "contents of b.txt", 2004)
        m2 = _a("Both files read; here is the answer", 2005)
        view = [*history, prompt_a, m1, r1a, r1b, m2]
        if window is not None:
            # Topic 9 holds about 52,000 rows, past the 10,000-row presence window.
            monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", window, raising=False)
        engine.ingest(view)

        stored = Counter(_key(row) for row in _stored(engine))
        assert stored[_key(m1)] == 1, "the first reply (text plus two tool calls) must be stored exactly once"
        for row in (r1a, r1b, m2):
            assert stored[_key(row)] == 1
        for row in history:  # nothing earlier is lost
            assert stored[_key(row)] >= 1
    finally:
        engine.shutdown()
