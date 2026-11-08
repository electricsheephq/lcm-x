"""#2466 review, round 5 (the release manager's rule-pairs lens), named limit: the presence check runs only on a turn
where the audit already missed a row. A tool row the host moved below the cursor while every text row there is
accounted for keeps main's result (a positional replay). Main-equal, tracked under #2467: the new-call tests assert
the correct behaviour and are strict xfails until that follow-up; the old pair and repeat-ingest checks run unmarked.
Default config, unique ids."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s-r51"
PAD = " alpha beta gamma delta" * 40
SLOT = 4
LIMIT = pytest.mark.xfail(strict=True, raises=AssertionError,
                          reason="#2467: the presence check runs only when the audit misses a row; a moved tool row "
                                 "with every text row accounted for keeps main's result")


def _engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    return engine


def _u(text, ts):
    return {"role": "user", "content": text + PAD, "timestamp": float(ts)}


def _a(text, ts):
    return {"role": "assistant", "content": text + PAD, "timestamp": float(ts)}


def _call(call_id, ts=None):
    row = {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": call_id})}}]}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _res(call_id, text, ts=None):
    row = {"role": "tool", "content": text, "tool_call_id": call_id, "tool_name": "read_file"}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _history():
    return [_u("P1 first prompt", 100), _a("A1 first reply", 101),
            _u("P2 second prompt", 110), _a("A2 second reply", 111),
            _u("P14 in-flight prompt", 120), _a("A14 reply", 121),
            _call("call_old", 130), _res("call_old", "old file text", 131), _a("A15 after the old call", 132)]


def _pairs(engine, ids):
    calls, results = Counter(), Counter()
    for row in engine._store.get_session_messages(SESSION):
        entries = row.get("tool_calls") or []
        for entry in json.loads(entries) if isinstance(entries, str) else entries:
            calls[str(entry.get("id"))] += 1
        if row.get("role") == "tool":
            results[str(row.get("tool_call_id"))] += 1
    return {i: (calls[i], results[i]) for i in ids}


def _dropped_interim(stamped_tools=True):
    history = _history()
    turn_start = [*history, _u("P60 new prompt", 200), _a("interim status note", 200.5)]
    turn_end = [*history, _u("P60 new prompt", 200), _call("call_new", 201), _res("call_new", "new", 202),
                _a("A60 after", 203)]
    return turn_start, turn_end


def _moved_prompt(stamped_tools):
    history = _history()
    turn_start = [*history, _u("P60 new prompt", 200)]
    turn_end = list(history)
    turn_end[SLOT] = _u("P60 new prompt", 200)
    turn_end += [_call("call_new", 201 if stamped_tools else None),
                 _res("call_new", "new", 202 if stamped_tools else None), _a("A60 after", 203)]
    return turn_start, turn_end


SCENARIOS = {"dropped-interim": (_dropped_interim, True), "moved-prompt-stamped": (_moved_prompt, True),
             "moved-prompt-unstamped": (_moved_prompt, False)}


def _go(tmp_path, scenario):
    """The store's pairs after the turn, and after the same view is ingested again."""
    build, stamped_tools = SCENARIOS[scenario]
    turn_start, turn_end = build(stamped_tools)
    engine = _engine(tmp_path)
    try:
        engine.ingest(_history())
        engine.ingest(turn_start)
        engine.ingest(turn_end)
        first = _pairs(engine, ["call_new", "call_old"])
        engine.ingest(turn_end)
        return first, _pairs(engine, ["call_new", "call_old"])
    finally:
        engine.shutdown()


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_the_gate_shapes_keep_the_old_pair_and_a_repeat_ingest_changes_nothing(tmp_path, scenario):
    """Runs without a marker, so an idempotence or old-pair regression cannot hide behind the strict xfail below."""
    first, again = _go(tmp_path, scenario)
    assert again == first, "repeat ingest changed the store"
    assert first["call_old"] == (1, 1), first


@LIMIT
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_the_new_call_is_stored_once_with_its_result(tmp_path, scenario):
    first, _ = _go(tmp_path, scenario)
    assert first["call_new"] == (1, 1), first
