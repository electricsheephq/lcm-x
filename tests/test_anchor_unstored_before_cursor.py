"""The prefix audit resumes at its anchor only when every row in front of it is accounted for.

When the host changes its list before the ingest cursor, the identity-anchor audit finds the rows there that no
stored occurrence explains and moves the cursor back to the first one. The other rows from there to the old cursor
were taken as replays by position. The audit never examines tool-call rows or tool results, so a tool-call row that
took the slot of a stored row was skipped as a replay and never stored.

Shape from the latest driven run: the host moved a new prompt into an in-flight prompt's slot, the list in front of
the new rows kept its length, and the turn's tool-call row took the prompt's old index. Each row in front of the
anchor must now be bound, matched to a stored row, or a compaction output; any other row is stored.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s-anchor-unstored"
PAD = " alpha beta gamma delta" * 40


def _engine(tmp_path) -> LCMEngine:
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), large_output_externalization_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    return engine


def _u(text, ts) -> dict:
    return {"role": "user", "content": text + PAD, "timestamp": float(ts)}


def _a(text, ts) -> dict:
    return {"role": "assistant", "content": text + PAD, "timestamp": float(ts)}


def _call(call_id, ts) -> dict:
    return {"role": "assistant", "content": "", "timestamp": float(ts), "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "notes.txt"}'}}]}


def _result(call_id, text, ts) -> dict:
    return {"role": "tool", "content": text, "tool_call_id": call_id, "tool_name": "read_file", "timestamp": float(ts)}


def _key(row) -> tuple:
    calls = row.get("tool_calls") or []
    calls = json.loads(calls) if isinstance(calls, str) else calls
    ids = ",".join(sorted(str(call.get("id")) for call in calls))
    return (row.get("role"), str(row.get("content") or "").strip(), row.get("tool_call_id") or "", ids)


def _stored(engine) -> list[dict]:
    return [row for row in engine._store.get_session_messages(SESSION) if row.get("role") != "system"]


def _history() -> list[dict]:
    """Two earlier turns, the in-flight prompt P14 at index 4, a stored tool turn, and the new prompt P60."""
    return [_u("P1 first prompt", 100), _a("A1 first reply", 101),
            _u("P2 second prompt", 110), _a("A2 second reply", 111),
            _u("P14 in-flight prompt", 120), _a("A14 reply", 121),
            _call("call_old", 130), _result("call_old", "old file text", 131), _a("A15 after the old call", 132)]


def _moved_turn(own_first: bool) -> tuple[list[dict], list[dict], list[dict]]:
    """(earlier view, turn-start view, turn-end view) for one tool turn whose prompt took P14's slot.

    own_first: the turn-start view holds the prompt on its own stamp at the end, and the turn-end view holds it in
    P14's slot on P14's stamp. Otherwise the reverse: the turn-start view holds it at the end on P14's stamp, and the
    turn-end view holds it in P14's slot on its own stamp. Either way the turn's tool-call row takes the prompt's old
    index, so the list in front of the new rows keeps its length."""
    history = _history()
    prompt = "P60 new prompt"
    start_stamp, moved_stamp = (200, 120) if own_first else (120, 200)
    turn_start = [*history, _u(prompt, start_stamp)]
    turn_end = list(history)
    turn_end[4] = _u(prompt, moved_stamp)  # the host moved the prompt into the in-flight prompt's slot
    turn_end += [_call("call_new", 201), _result("call_new", "new file text", 202), _a("A60 after the new call", 203)]
    assert len(turn_end) == len(turn_start) + 2  # the list in front of the new rows kept its length
    return history, turn_start, turn_end


@pytest.mark.parametrize("own_first", [True, False], ids=["own-stamp-first", "carried-stamp-first"])
def test_a_tool_call_row_in_a_moved_prompt_slot_is_stored_exactly_once(tmp_path, own_first):
    engine = _engine(tmp_path)
    try:
        history, turn_start, turn_end = _moved_turn(own_first)
        engine.ingest(history)
        engine.ingest(turn_start)
        before = Counter(_key(row) for row in _stored(engine))
        engine.ingest(turn_end)
        after = Counter(_key(row) for row in _stored(engine))
        added = after - before
        new_call, new_result, new_reply = (_key(row) for row in turn_end[-3:])
        assert added[new_call] == 1, f"the tool-call row was stored {added[new_call]} times"
        assert added[new_result] == 1 and added[new_reply] == 1
        assert not before - after
        # The only other row is the moved prompt at its other stamp (the #2421 borrowed-stamp copy, unchanged here).
        assert sum(added.values()) <= 4, added
        assert set(added) <= {new_call, new_result, new_reply, _key(turn_end[4])}, added
        # Nothing stored twice except that moved prompt.
        twice = {key for key, count in after.items() if count > 1}
        assert twice <= {_key(turn_end[4])}, twice
        # A later ingest of the same view stores nothing more.
        engine.ingest(turn_end)
        assert Counter(_key(row) for row in _stored(engine)) == after
    finally:
        engine.shutdown()


def test_pre_ledger_rows_in_front_of_the_cursor_store_nothing_extra(tmp_path, monkeypatch, caplog):
    """An upgraded store: the rows in front of the cursor were stored before the ledger existed, so none is bound.
    A genuinely new row before the cursor is stored; every stored row in front of the anchor (tool rows included)
    is counted against its stored copy, so nothing else is stored."""
    monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    engine = _engine(tmp_path)
    try:
        history = _history()
        engine.ingest(history)
        assert not engine._store._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='occurrence_bindings'").fetchone() or not engine._store._conn.execute(
            "SELECT 1 FROM occurrence_bindings").fetchone()
        before = Counter(_key(row) for row in _stored(engine))
        monkeypatch.setenv("LCM_INGEST_LEDGER", "on")
        changed = list(history)
        changed[4] = _u("P61 a prompt the host put in the in-flight slot", 125)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.ingest(changed)
        assert any("before the cursor" in r.getMessage() for r in caplog.records)  # the audit path ran
        added = Counter(_key(row) for row in _stored(engine)) - before
        assert added == Counter([_key(changed[4])]), added
    finally:
        engine.shutdown()


def test_unaccounted_rows_in_front_of_the_anchor_log_a_reasoned_warning(tmp_path, caplog):
    engine = _engine(tmp_path)
    try:
        history, turn_start, turn_end = _moved_turn(True)
        engine.ingest(history)
        engine.ingest(turn_start)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.ingest(turn_end)
        lines = [r for r in caplog.records if "before the cursor" in r.getMessage()]
        assert len(lines) == 1, [r.getMessage() for r in lines]
        assert lines[0].levelno == logging.WARNING
        message = lines[0].getMessage()
        # The moved prompt and the tool-call row: neither is stored, bound or a compaction output.
        assert "2 row(s)" in message and "neither stored nor compaction outputs" in message, message
    finally:
        engine.shutdown()
