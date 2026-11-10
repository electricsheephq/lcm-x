"""RM probe (2026-11-08): the PR's class A shapes at DEFAULT externalization (on, 12,000 chars), with a large tool
output, so the stored result is an externalized placeholder while the host's view shows it inline. Asserts the
correct behaviour: the moved call, its result and the reply are each stored exactly once; nothing lost; a repeat
ingest stores nothing more."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s-rm-ext"
PAD = " alpha beta gamma delta" * 40
BIG = "line of terminal output 0123456789\n" * 600  # about 21,000 chars, over the 12,000 default


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))  # default externalization
    assert config.large_output_externalization_enabled and config.large_output_externalization_threshold_chars < len(BIG)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    return engine


def _u(text, ts):
    return {"role": "user", "content": text + PAD, "timestamp": float(ts)}


def _a(text, ts):
    return {"role": "assistant", "content": text + PAD, "timestamp": float(ts)}


def _call(call_id, ts):
    return {"role": "assistant", "content": "", "timestamp": float(ts), "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": '{"command": "ls -la"}'}}]}


def _result(call_id, text, ts):
    return {"role": "tool", "content": text, "tool_call_id": call_id, "tool_name": "terminal", "timestamp": float(ts)}


def _key(row):
    calls = row.get("tool_calls") or []
    calls = json.loads(calls) if isinstance(calls, str) else calls
    ids = ",".join(sorted(str(c.get("id")) for c in calls))
    content = str(row.get("content") or "").strip()
    return (row.get("role"), content[:50], row.get("tool_call_id") or "", ids)


def _stored(engine):
    return [r for r in engine._store.get_session_messages(SESSION) if r.get("role") != "system"]


def _history(big_old):
    return [_u("P1 first prompt", 100), _a("A1 first reply", 101),
            _u("P2 second prompt", 110), _a("A2 second reply", 111),
            _u("P14 in-flight prompt", 120), _a("A14 reply", 121),
            _call("call_old", 130), _result("call_old", BIG if big_old else "old file text", 131),
            _a("A15 after the old call", 132)]


@pytest.mark.parametrize("big_old", [False, True], ids=["small-old", "big-old"])
@pytest.mark.parametrize("big_new", [False, True], ids=["small-new", "big-new"])
@pytest.mark.parametrize("own_first", [True, False], ids=["own-first", "carried-first"])
def test_class_a_moved_call_with_default_externalization(tmp_path, own_first, big_new, big_old):
    engine = _engine(tmp_path)
    try:
        history = _history(big_old)
        slot = 4
        start_stamp, moved_stamp = (200, 120) if own_first else (120, 200)
        turn_start = [*history, _u("P60 new prompt", start_stamp)]
        turn_end = list(history)
        turn_end[slot] = _u("P60 new prompt", moved_stamp)
        turn_end += [_call("call_new", 201), _result("call_new", BIG if big_new else "new file text", 202),
                     _a("A60 after the new call", 203)]
        engine.ingest(history)
        engine.ingest(turn_start)
        before = Counter(_key(r) for r in _stored(engine))
        engine.ingest(turn_end)
        after = Counter(_key(r) for r in _stored(engine))
        added = after - before
        stored_calls = Counter(c for r in _stored(engine) for c in
                               [str(x.get("id")) for x in (json.loads(r["tool_calls"]) if isinstance(r.get("tool_calls"), str) else (r.get("tool_calls") or []))])
        stored_results = Counter(str(r.get("tool_call_id")) for r in _stored(engine) if r.get("role") == "tool")
        assert stored_calls["call_new"] == 1, ("moved call stored", stored_calls["call_new"], dict(added))
        assert stored_results["call_new"] == 1, ("its result stored", stored_results["call_new"], dict(added))
        assert stored_calls["call_old"] == 1 and stored_results["call_old"] == 1, (dict(stored_calls), dict(stored_results))
        assert not before - after
        engine.ingest(turn_end)
        assert Counter(_key(r) for r in _stored(engine)) == after, "a repeat ingest changed the store"
    finally:
        engine.shutdown()
