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


def _engine(tmp_path, **overrides) -> LCMEngine:
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), large_output_externalization_enabled=False, **overrides)
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


def _history(earlier: int = 0) -> list[dict]:
    """``earlier`` long-ago turns, two earlier turns, the in-flight prompt P14, and a stored tool turn."""
    return [*(row for i in range(earlier) for row in (_u(f"E{i} earlier prompt", 10 + i / 10), _a(f"E{i} reply", 10 + i / 10 + 0.01))),
            _u("P1 first prompt", 100), _a("A1 first reply", 101),
            _u("P2 second prompt", 110), _a("A2 second reply", 111),
            _u("P14 in-flight prompt", 120), _a("A14 reply", 121),
            _call("call_old", 130), _result("call_old", "old file text", 131), _a("A15 after the old call", 132)]


def _moved_turn(own_first: bool, earlier: int = 0) -> tuple[list[dict], list[dict], list[dict]]:
    """(earlier view, turn-start view, turn-end view) for one tool turn whose prompt took P14's slot.

    own_first: the turn-start view holds the prompt on its own stamp at the end, and the turn-end view holds it in
    P14's slot on P14's stamp. Otherwise the reverse: the turn-start view holds it at the end on P14's stamp, and the
    turn-end view holds it in P14's slot on its own stamp. Either way the turn's tool-call row takes the prompt's old
    index, so the list in front of the new rows keeps its length."""
    history = _history(earlier)
    slot = 2 * earlier + 4  # P14's index
    prompt = "P60 new prompt"
    start_stamp, moved_stamp = (200, 120) if own_first else (120, 200)
    turn_start = [*history, _u(prompt, start_stamp)]
    turn_end = list(history)
    turn_end[slot] = _u(prompt, moved_stamp)  # the host moved the prompt into the in-flight prompt's slot
    turn_end += [_call("call_new", 201), _result("call_new", "new file text", 202), _a("A60 after the new call", 203)]
    assert len(turn_end) == len(turn_start) + 2  # the list in front of the new rows kept its length
    return history, turn_start, turn_end


@pytest.mark.parametrize("earlier", [0, 40], ids=["short", "past-the-audited-tail"])
@pytest.mark.parametrize("own_first", [True, False], ids=["own-stamp-first", "carried-stamp-first"])
def test_a_tool_call_row_in_a_moved_prompt_slot_is_stored_exactly_once(tmp_path, own_first, earlier):
    engine = _engine(tmp_path)
    try:
        history, turn_start, turn_end = _moved_turn(own_first, earlier)
        slot = 2 * earlier + 4
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
        assert set(added) <= {new_call, new_result, new_reply, _key(turn_end[slot])}, added
        # Nothing stored twice except that moved prompt.
        twice = {key for key, count in after.items() if count > 1}
        assert twice <= {_key(turn_end[slot])}, twice
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


STEER = {"role": "user", "content": "steer: keep the answer short"}  # unstamped, as the host inserts it


def _steered(engine, turns: int, reply, steer_at: int) -> tuple[list[dict], list[dict]]:
    """``turns`` stamped user turns with unstamped replies, then a /steer inserted ``steer_at`` rows from the end,
    then one more turn. Returns (the view before the steer, the view after it)."""
    view = []
    for index in range(turns):
        view += [_u(f"T{index} user turn", 1000 + index), {"role": "assistant", "content": reply(index)}]
        engine.ingest(view)
    before = list(view)
    view.insert(len(view) - steer_at, dict(STEER))
    view += [_u("the next user turn", 5000), {"role": "assistant", "content": "the next reply" + PAD}]
    engine.ingest(view)
    return before, view


def _count(engine, text) -> int:
    return sum(1 for row in _stored(engine) if row.get("content") == text)


def test_a_reply_repeated_past_the_audited_tail_is_not_stored_again(tmp_path):
    """Review: copies of a repeated reply outside the audited tail must not make a stored copy look unaccounted.
    With 40 equal replies the tail holds fewer copies than the prefix reserves; unsure counts as accounted."""
    engine = _engine(tmp_path)
    try:
        same = "the same reply every time" + PAD
        _before, view = _steered(engine, 40, lambda _i: same, 3)
        # As on main: the 40 replies and the displaced reply after the steer once more (#633's known duplicate).
        assert _count(engine, same) == 41
        assert _count(engine, STEER["content"]) == 1
    finally:
        engine.shutdown()


def test_a_redacted_reply_in_front_of_the_anchor_is_not_stored_again(tmp_path):
    """Review: a lossy (redacted) identity has no comparable key; such a row counts as accounted, as on main."""
    engine = _engine(tmp_path, sensitive_patterns_enabled=True)
    try:
        secret = "the reply quotes password=hunter2hunter2 from the file" + PAD
        _before, _view = _steered(engine, 6, lambda i: secret if i == 4 else f"reply {i}" + PAD, 5)
        # Main: 12 rows, the steer, the next turn's two rows, and the displaced reply after the steer (#633): 16.
        assert len(_stored(engine)) == 16
    finally:
        engine.shutdown()


def _pre_ledger(tmp_path, monkeypatch, view) -> LCMEngine:
    monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    engine = _engine(tmp_path)
    engine.ingest(view)
    monkeypatch.setenv("LCM_INGEST_LEDGER", "on")
    return engine


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_an_upgrade_past_the_audited_tail_stores_only_the_replaced_prompt(tmp_path, monkeypatch, cold):
    """Review: 50 pre-ledger turns with the same unstamped reply; the host replaces the last prompt. Only the new
    prompt is stored, as on main, on the warm and the cold upgrade path."""
    view = []
    for index in range(50):
        view += [_u(f"T{index} user turn", 1000 + index), {"role": "assistant", "content": "OK"}]
    engine = _pre_ledger(tmp_path, monkeypatch, view)
    try:
        before = Counter(_key(row) for row in _stored(engine))
        if cold:
            engine.shutdown()
            engine = _engine(tmp_path)
        changed = list(view)
        changed[-2] = _u("a replaced last prompt", 2000)
        engine.ingest(changed)
        added = Counter(_key(row) for row in _stored(engine)) - before
        # Cold: with no proof and no stamps on the replies, the restart's first ingest stores the replies again
        # as one persisted batch (#259), the same 50 rows on main; the audit adds nothing to it.
        assert added == Counter({_key(changed[-2]): 1, **({_key(view[1]): 50} if cold else {})}), added
    finally:
        engine.shutdown()


def test_an_upgrade_with_a_tool_result_the_store_holds_without_its_name_stores_only_the_prompt(tmp_path, monkeypatch):
    """Review: a pre-ledger tool result whose host form names its tool as ``name`` (the stored row has no tool
    name) has no equal stored identity. It is stored, so it counts as accounted for; only the new prompt is
    stored, and no tool result is ever stored without its call."""
    history = _history()
    history[7] = {"role": "tool", "content": "old file text", "tool_call_id": "call_old", "name": "read_file",
                  "timestamp": 131.0}
    engine = _pre_ledger(tmp_path, monkeypatch, history)
    try:
        before = Counter(_key(row) for row in _stored(engine))
        changed = list(history)
        changed[4] = _u("P61 a prompt the host put in the in-flight slot", 125)
        engine.ingest(changed)
        added = Counter(_key(row) for row in _stored(engine)) - before
        assert added == Counter([_key(changed[4])]), added
    finally:
        engine.shutdown()


def test_a_reply_that_left_the_audited_tail_is_not_stored_again(tmp_path):
    """Review (NeonDiff): a warm two-row view, a stamped user and an unstamped reply, whose user the host replaces
    more often than the audited tail holds rows. The reply's stored copy leaves the capped tail; absence from a
    capped tail is no proof that a row was never stored, so the reply is stored once, and each user once."""
    engine = _engine(tmp_path)
    try:
        reply = {"role": "assistant", "content": "the one reply" + PAD}
        users = [_u(f"replacement user {index}", 1000 + index) for index in range(70)]
        for user in users:
            engine.ingest([user, reply])
        assert _count(engine, reply["content"]) == 1
        assert all(_count(engine, user["content"]) == 1 for user in users)
    finally:
        engine.shutdown()

