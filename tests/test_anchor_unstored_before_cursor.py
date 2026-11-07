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
import sqlite3
from collections import Counter

import pytest

from hermes_lcm import identity_anchor
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
    if len(turn_end) != len(turn_start) + 2:
        pytest.fail("setup: the list in front of the new rows must keep its length")
    return history, turn_start, turn_end


@pytest.mark.parametrize("earlier", [0], ids=["short"])
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


def _long_session_class_a_shape(tmp_path, own_first, variant, call_id="call_new"):
    """Count every addition and loss, allowing only the existing moved-prompt copy beside the new turn."""
    engine = _engine(tmp_path)
    try:
        history, turn_start, turn_end = _moved_turn(own_first, earlier=40)
        slot = 2 * 40 + 4
        call, result, reply = turn_end[-3:]
        call["tool_calls"][0]["id"] = result["tool_call_id"] = call_id
        if variant == "V2":
            # One more stored row puts both the call and result before the cursor.
            turn_start.append({"role": "assistant", "content": "an interim note" + PAD})
        elif variant == "V3":
            turn_end.remove(result)
        engine.ingest(history)
        engine.ingest(turn_start)
        before = Counter(_key(row) for row in _stored(engine))
        if sum(before.values()) <= 64:
            pytest.fail("setup: the session must hold more stored rows than the audited tail")
        cursor = len(turn_start)
        if turn_end.index(call) >= cursor or (variant == "V2" and turn_end.index(result) >= cursor):
            pytest.fail("setup: the call (and V2 result) must be before the cursor")
        if variant == "V1" and turn_end.index(result) != cursor:
            pytest.fail("setup: V1's result must be at the cursor")
        if variant == "V3" and any(row.get("tool_call_id") == call_id for row in turn_end):
            pytest.fail("setup: V3's result must be missing from the view")
        if any(before[_key(row)] for row in (call, result, reply)):
            pytest.fail("setup: the new call, result and reply must not already be stored")
        engine.ingest(turn_end)
        after = Counter(_key(row) for row in _stored(engine))
        added = after - before
        engine.ingest(turn_end)
        assert Counter(_key(row) for row in _stored(engine)) == after
        allowed = {_key(row) for row in (call, result, reply, turn_end[slot])}
        return {"call": added[_key(call)], "result": added[_key(result)], "reply": added[_key(reply)],
                "moved_prompt": added[_key(turn_end[slot])],
                "other_added": {key: count for key, count in added.items() if key not in allowed},
                "lost": dict(before - after)}
    finally:
        engine.shutdown()


@pytest.mark.parametrize("own_first,call_id", [(True, "call_new"), (False, "call_new"), (True, "call_é新")],
                         ids=["own-stamp-first", "carried-stamp-first", "unicode"])
def test_long_session_class_a_past_the_audited_tail_is_stored_once(tmp_path, own_first, call_id):
    assert _long_session_class_a_shape(tmp_path, own_first, "V1", call_id) == {
        "call": 1, "result": 1, "reply": 1, "moved_prompt": 1, "other_added": {}, "lost": {}}


def test_long_session_class_a_v2_call_and_result_before_the_cursor_are_stored_once(tmp_path):
    assert _long_session_class_a_shape(tmp_path, True, "V2") == {
        "call": 1, "result": 1, "reply": 1, "moved_prompt": 1, "other_added": {}, "lost": {}}


def test_long_session_class_a_v3_result_missing_from_the_view_is_stored_once(tmp_path):
    assert _long_session_class_a_shape(tmp_path, True, "V3") == {
        "call": 1, "result": 0, "reply": 1, "moved_prompt": 1, "other_added": {}, "lost": {}}


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



def _long_resend(engine) -> dict[str, str]:
    """40 warm turns (80 stored rows, more than the 64-row audited tail), stamped users with unstamped replies;
    then the host replaces the last turn with a resent user and a new reply, and one more turn follows."""
    view = []
    for index in range(40):
        view += [_u(f"T{index} user turn", 1000 + index), {"role": "assistant", "content": f"reply to T{index}" + PAD}]
        engine.ingest(view)
    if len(_stored(engine)) <= 64:
        pytest.fail("setup: the session must hold more stored rows than the audited tail")
    rows = {"user": _u("the resent user turn", 3000), "reply": {"role": "assistant", "content": "a new reply" + PAD},
            "next user": _u("the next user turn", 3001), "next reply": {"role": "assistant", "content": "next" + PAD}}
    view[-2:] = [rows["user"], rows["reply"]]
    view += [rows["next user"], rows["next reply"]]
    before = Counter(_key(row) for row in _stored(engine))
    engine.ingest(view)
    after = Counter(_key(row) for row in _stored(engine))
    assert after - before == Counter(_key(row) for row in rows.values())
    assert not before - after
    engine.ingest(view)
    assert Counter(_key(row) for row in _stored(engine)) == after
    return {name: row["content"] for name, row in rows.items()}


def test_2116_a_replaced_reply_past_the_audited_tail_is_stored_warm(tmp_path):
    engine = _engine(tmp_path)
    try:
        texts = _long_resend(engine)
        assert {name: _count(engine, text) for name, text in texts.items()} == dict.fromkeys(texts, 1)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("call_id", ["call_kept", "call_é新"])
def test_a_stored_call_and_result_outside_the_audited_tail_are_not_stored_again(tmp_path, call_id):
    """A warm view of a stamped user, an unstamped call and its unstamped result; the host replaces the user 70 times,
    so the stored call and result leave the 64-row audited tail. Neither is stored again."""
    engine = _engine(tmp_path)
    try:
        # Unstamped, as the host's in-memory rows are, so no stamp match accounts for them: only the count does.
        call, result = _call(call_id, 2), _result(call_id, "kept file text", 3)
        del call["timestamp"], result["timestamp"]
        if not result.get("tool_name"):
            pytest.fail("setup: the stored result must carry a tool name")
        for index in range(70):
            engine.ingest([_u(f"replacement user {index}", 1000 + index), call, result])
        rows = _stored(engine)
        if len(rows) <= 64:
            pytest.fail("setup: the session must hold more stored rows than the audited tail")
        assert sum(1 for row in rows if row.get("tool_call_id") == call_id) == 1
        assert sum(1 for row in rows if call_id in _key(row)[3]) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("ledger", ["off", "on"])
def test_a_stored_call_whose_delayed_result_arrives_past_the_audited_tail_is_stored_once(tmp_path, monkeypatch, ledger):
    """Review (Codex, Astra): an unstamped call is stored, then leaves the 64-row audited tail while the host replaces
    the stamped user before it 70 times; its delayed result then arrives behind one more replacement. No stored
    result is no proof that the call was never stored: the call stays stored once, the result is stored once."""
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    engine = _engine(tmp_path)
    try:
        call, result = _call("call_held", 2), _result("call_held", "held file text", 3)
        del call["timestamp"], result["timestamp"]
        for index in range(70):
            engine.ingest([_u(f"replacement user {index}", 1000 + index), call])
        if len(_stored(engine)) <= 64:
            pytest.fail("setup: the session must hold more stored rows than the audited tail")
        engine.ingest([_u("one more replacement", 5000), call, result])
        rows = _stored(engine)
        assert sum(1 for row in rows if "call_held" in str(row.get("tool_calls") or "")) == 1
        assert sum(1 for row in rows if row.get("tool_call_id") == "call_held") == 1
    finally:
        engine.shutdown()


def test_past_the_audited_tail_a_moved_prompt_result_is_never_stored_without_its_call(tmp_path):
    """Review (Codex P1, NeonDiff P2): in a session past the audited tail, the moved-prompt turn's new call and its
    result are both before the cursor. The call is not proven new there (#2455 tracks storing it), so its result must
    not be stored alone either: a result is stored only beside its call, as on main."""
    engine = _engine(tmp_path)
    try:
        history = _history(40)
        slot = 2 * 40 + 4
        interim = {"role": "assistant", "content": "an interim note" + PAD}
        turn_start = [*history, _u("P60 new prompt", 200), interim]
        engine.ingest(history)
        engine.ingest(turn_start)
        if len(_stored(engine)) <= 64:
            pytest.fail("setup: the session must hold more stored rows than the audited tail")
        turn_end = list(history)
        turn_end[slot] = _u("P60 new prompt", 120)
        turn_end += [_call("call_new", 201), _result("call_new", "new file text", 202)]
        if len(turn_end) != len(turn_start):
            pytest.fail("setup: the list in front of the cursor must keep its length")
        turn_end += [_a("A60 after the new call", 204)]
        engine.ingest(turn_end)
        rows = _stored(engine)
        calls = sum(1 for row in rows if "call_new" in str(row.get("tool_calls") or ""))
        results = sum(1 for row in rows if row.get("tool_call_id") == "call_new")
        assert results == calls, (calls, results)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("earlier", [6, 40])
@pytest.mark.parametrize("statedb", ["none", "verified"])
def test_rotation_predecessor_scope_equals_main(tmp_path, monkeypatch, statedb, earlier):
    """#2463: predecessor presence prevents new tool copies; main's two text copies remain without state.db."""
    import hermes_lcm.engine as lcm_engine

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: (
        "Earlier turns.\nExpand for details about: turns", 1))
    if statedb == "verified":
        with sqlite3.connect(tmp_path / "state.db") as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
            conn.executemany("INSERT INTO sessions VALUES (?,?,?)", [("P", None, "compression"), ("C", "P", None)])
    config = LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=1, database_path=str(tmp_path / "lcm.db"),
                       large_output_externalization_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.on_session_start("P", platform="cli", context_length=200_000, conversation_id="conv")
    try:
        live = [{"role": "system", "content": "stable system prompt"}]
        for i in range(earlier):
            live += [_u(f"T{i} prompt", 10 * i), _a(f"R{i} reply", 10 * i + 1)]
        live += [_u("TQ tool prompt", 100), _call("call_p", 101), _result("call_p", "Pfile text", 102),
                 _a("RQ after call", 103)]
        engine.ingest(live)
        out = engine.compress(live)
        engine.on_session_end("P", list(live))
        engine.on_session_start("C", boundary_reason="compression", old_session_id="P", platform="cli",
                                context_length=200_000, conversation_id="conv")
        assert engine._identity_anchor_chain() == (["P"] if statedb == "verified" else [])
        engine.ingest([*out, _u("N1 prompt", 500), _a("N1 reply", 501)])
        before = Counter(_key(r) for r in engine._store.get_session_messages("C") if r["role"] != "system")
        ins = next((i for i, m in enumerate(out) if m["role"] == "user" and "timestamp" in m), 1)
        changed = [*out[:ins], _u("X inserted prompt", 450), *out[ins:], _u("N1 prompt", 500), _a("N1 reply", 501),
                   _u("N2 prompt", 600), _a("N2 reply", 601)]
        engine.ingest(changed)
        after = Counter(_key(r) for r in engine._store.get_session_messages("C") if r["role"] != "system")
        parent = Counter(_key(r) for r in engine._store.get_session_messages("P") if r["role"] != "system")
        added = after - before
        dup_of_p = sum(count for key, count in after.items() if parent[key])
        assert (sum(added.values()), dup_of_p) == ((5, 2) if statedb == "none" else (3, 0))
        assert not before - after
        added_roles = Counter((key[0], key[2], key[3]) for key in added.elements())
        assert added_roles[("assistant", "", "call_p")] == added_roles[("tool", "call_p", "")] == 0
        engine.ingest(changed)
        assert Counter(_key(r) for r in engine._store.get_session_messages("C") if r["role"] != "system") == after
    finally:
        engine.shutdown()


def test_a_capped_session_reusing_a_stored_call_id_stores_the_new_call_once(tmp_path):
    """#2172 host shape: the view holds the earlier call and a new call under the same id, the store holds one.

    The earlier call stays in the view's unchanged prefix, which reserves its stored copy, so the new call
    (other arguments) has no stored copy and is stored once. Main takes it as a replay and stores its result
    without it (the #2172 loss family); the candidate stores the call, each argument set exactly once.
    """
    engine = _engine(tmp_path)
    try:
        history, turn_start, turn_end = _moved_turn(True, earlier=40)
        call, result, reply = turn_end[-3:]
        call["tool_calls"][0]["id"] = result["tool_call_id"] = "call_old"
        call["tool_calls"][0]["function"]["arguments"] = '{"path":"different.txt"}'
        engine.ingest(history)
        engine.ingest(turn_start)
        before = Counter(_key(row) for row in _stored(engine))
        engine.ingest(turn_end)
        after = Counter(_key(row) for row in _stored(engine))
        assert after - before == Counter(_key(row) for row in (turn_end[84], call, result, reply))
        assert not before - after
        shown = Counter(c["function"]["arguments"] for row in turn_end for c in (row.get("tool_calls") or [])
                        if c["id"] == "call_old")
        stored = Counter()
        for row in _stored(engine):
            calls = row.get("tool_calls") or []
            calls = json.loads(calls) if isinstance(calls, str) else calls
            stored.update(c["function"]["arguments"] for c in calls if c.get("id") == "call_old")
        assert shown == stored and set(shown.values()) == {1}
        engine.ingest(turn_end)
        assert Counter(_key(row) for row in _stored(engine)) == after
    finally:
        engine.shutdown()


@pytest.mark.parametrize("failure", ["predecessor", "metadata", "tail"])
def test_a_failed_presence_lookup_keeps_only_the_audits_own_finds(tmp_path, monkeypatch, caplog, failure):
    engine = _engine(tmp_path)
    try:
        history, turn_start, turn_end = _moved_turn(True, earlier=40)
        # Leave an unstamped text candidate in the audited range.
        history[-1].pop("timestamp")
        engine.ingest(history)
        engine.ingest(turn_start)
        before = Counter(_key(row) for row in _stored(engine))
        store = engine._store

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic presence read failure")

        if failure == "predecessor":
            monkeypatch.setattr(engine, "_rotation_predecessor_session_ids", fail)
        elif failure == "metadata":
            read = store.read_metadata_json

            def fail_predecessor(key, *args, **kwargs):
                return fail() if key.startswith("rotation_predecessor_session:") else read(key, *args, **kwargs)

            monkeypatch.setattr(store, "read_metadata_json", fail_predecessor)
        else:
            read_tail = store.get_session_tail

            def fail_presence_tail(session, *, limit):
                if limit == min(256, getattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 10000) + 1):
                    return fail()
                return read_tail(session, limit=limit)

            monkeypatch.setattr(store, "get_session_tail", fail_presence_tail)
        with caplog.at_level(logging.WARNING):
            engine.ingest(turn_end)
        after = Counter(_key(row) for row in _stored(engine))
        # Main stores the moved prompt (audit's own find), plus the result and reply at/after the cursor.
        assert after - before == Counter(_key(row) for row in (turn_end[84], *turn_end[-2:]))
        assert not before - after
        assert any("presence" in record.message and record.levelno == logging.WARNING for record in caplog.records)
    finally:
        engine.shutdown()


def _presence(engine, view, *, held=None, capped=True, start=0):
    return engine._identity_anchor_unaccounted(
        view, view, len(view), start, lambda i: engine._message_replay_identity(view[i]), held or {}, capped,
        set(), {"replayed": set()}, {}, (), [])


@pytest.mark.parametrize("call_id", [True, 1, False, 0])
def test_non_string_view_call_ids_are_unsure(tmp_path, call_id):
    engine = _engine(tmp_path)
    try:
        assert _presence(engine, [_call(call_id, 1), _result(call_id, "result", 2)]) == []
    finally:
        engine.shutdown()


@pytest.mark.parametrize("flagged", [False, True], ids=["recorded", "earlier-hop-unrecorded"])
def test_an_unrecorded_earlier_rotation_ends_the_scope(tmp_path, caplog, flagged):
    """#2466 review: P -> C's predecessor write failed (C flagged), then C -> D was recorded. D's scope ends at C,
    so P's stored call would be invisible: the presence check must fall back to main, not report the call absent."""
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch("P", [_call("call_parent", 1), _result("call_parent", "held", 2)])
        engine._record_rotation_predecessor(SESSION, "C")
        if flagged:
            engine._rotation_lineage_unrecorded = {"C"}
        else:
            engine._record_rotation_predecessor("C", "P")
        with caplog.at_level(logging.WARNING):
            found = _presence(engine, [_call("call_parent", 1), _result("call_parent", "held", 2)])
        assert found == []
        assert any("presence unavailable" in r.getMessage() for r in caplog.records) is flagged
    finally:
        engine.shutdown()


def test_a_transient_predecessor_read_failure_cannot_hide_a_scope_session(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path)
    try:
        engine._record_rotation_predecessor(SESSION, "P")
        engine._store.append_batch("P", [_call("call_parent", 1), _result("call_parent", "held", 2)])
        read, attempts = engine._store.read_metadata_json, []

        def fail_once(key, *args, **kwargs):
            if key.startswith("rotation_predecessor_session:"):
                attempts.append(key)
                if len(attempts) == 1:
                    raise RuntimeError("transient predecessor read failure")
            return read(key, *args, **kwargs)

        monkeypatch.setattr(engine._store, "read_metadata_json", fail_once)
        with caplog.at_level(logging.WARNING):
            assert _presence(engine, [_call("call_new", 3), _result("call_new", "new", 4)]) == []
        assert any("presence" in record.message and record.levelno == logging.WARNING for record in caplog.records)
    finally:
        engine.shutdown()


_WINDOW_LIMIT = ("#2455/#2116: past the presence window (_PRESENCE_WINDOW_ROWS stored rows across the scope) "
                 "a moved or replaced row stays a positional replay, as on main")


def _window_ingest(tmp_path, monkeypatch, shape, window):
    """Literal main multisets: preserve every prior row and add only the audit's own finds and fresh tail."""
    engine = _engine(tmp_path)
    try:
        if shape == "V1":
            history, turn_start, changed = _moved_turn(True)
            engine.ingest(history)
            engine.ingest(turn_start)
            initial = [*history, turn_start[-1]]
            main_rows = [*initial, changed[4], *changed[-2:]]
            missing = changed[-3]
        else:
            initial = []
            for index in range(6):
                initial += [_u(f"T{index} user turn", 1000 + index),
                            {"role": "assistant", "content": f"reply to T{index}" + PAD}]
                engine.ingest(initial)
            new = [_u("the resent user turn", 3000), {"role": "assistant", "content": "a new reply" + PAD},
                   _u("the next user turn", 3001), {"role": "assistant", "content": "next" + PAD}]
            changed = [*initial[:-2], *new]
            main_rows = [*initial, new[0], *new[2:]]
            missing = new[1]
        before = Counter(_key(row) for row in _stored(engine))
        assert before == Counter(_key(row) for row in initial)
        # raising=False lets the same proof run on the base, which ignores this constant.
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", window, raising=False)
        engine.ingest(changed)
        after = Counter(_key(row) for row in _stored(engine))
        assert not before - after
        return after, Counter(_key(row) for row in main_rows), _key(missing)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shape", ["V1", "2116"])
def test_past_presence_window_stored_multiset_equals_main(tmp_path, monkeypatch, shape):
    after, main, _missing = _window_ingest(tmp_path, monkeypatch, shape, 8)
    assert after == main


@pytest.mark.xfail(strict=True, reason=_WINDOW_LIMIT)
def test_2455_past_presence_window_moved_call_is_stored_named_limit(tmp_path, monkeypatch):
    after, _main, missing = _window_ingest(tmp_path, monkeypatch, "V1", 8)
    assert after[missing] == 1


@pytest.mark.xfail(strict=True, reason=_WINDOW_LIMIT)
def test_2116_past_presence_window_replaced_reply_is_stored_named_limit(tmp_path, monkeypatch):
    after, _main, missing = _window_ingest(tmp_path, monkeypatch, "2116", 8)
    assert after[missing] == 1


@pytest.mark.parametrize("window,stored", [(10, 1), (9, 0)], ids=["exactly-N", "N-plus-one"])
def test_presence_window_ingest_boundary(tmp_path, monkeypatch, window, stored):
    after, main, missing = _window_ingest(tmp_path, monkeypatch, "V1", window)
    assert after == main + Counter({missing: stored})


@pytest.mark.parametrize("extra", [0, 1], ids=["exactly-N", "N-plus-one"])
@pytest.mark.parametrize("capped", [False, True])
def test_presence_window_counts_deduplicated_whole_scope(tmp_path, monkeypatch, caplog, extra, capped):
    engine = _engine(tmp_path)
    try:
        store = engine._store
        store.append_batch(SESSION, [_a("current", 1)])
        store.append_batch("A", [_a("ancestor", 2)])
        store.append_batch("P", [_a(f"predecessor {i}", 3 + i) for i in range(2 + extra)])
        engine._record_rotation_predecessor(SESSION, "P")
        monkeypatch.setattr(engine, "_identity_anchor_chain", lambda: ["A", "P"])
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 4, raising=False)
        read_tail, reads = store.get_session_tail, []

        def tail(source, *, limit):
            reads.append((source, limit))
            return read_tail(source, limit=limit)

        monkeypatch.setattr(store, "get_session_tail", tail)
        caplog.clear()  # inspect only the presence check, excluding engine setup warnings
        with caplog.at_level(logging.DEBUG, logger=identity_anchor.__name__):
            found = _presence(engine, [_call("new", 10), _result("new", "result", 11)], capped=capped)
        assert found == ([] if extra else [0, 1])  # over budget: main's positional behavior
        assert reads == [(SESSION, 5), ("A", 4), ("P", 3)]
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)
        if extra:
            assert any("presence window" in record.message and "scope=3" in record.message
                       and "budget=2" in record.message and SESSION in record.message for record in caplog.records)
    finally:
        engine.shutdown()


def test_tool_only_presence_uses_the_same_bounded_tail(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 2, raising=False)
        engine._store.append_batch(SESSION, [_call("kept", 1), _result("kept", "held", 2), _a("third", 3)])
        assert _presence(engine, [_call("new", 4), _result("new", "new result", 5)]) == []
    finally:
        engine.shutdown()


@pytest.mark.parametrize("failure", ["chain", "write"], ids=["state-db-read-fails", "predecessor-write-fails"])
def test_incomplete_rotation_lineage_keeps_main_multiset(tmp_path, monkeypatch, caplog, failure):
    """I1/I2: incomplete ancestry cannot prove a carried call/result absent from its parent."""
    earlier = 6
    import hermes_lcm.engine as lcm_engine

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: (
        "Earlier turns.\nExpand for details about: turns", 1))
    if failure == "chain":
        with sqlite3.connect(tmp_path / "state.db") as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
            conn.executemany("INSERT INTO sessions VALUES (?,?,?)", [("P", None, "compression"), ("C", "P", None)])
    config = LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=1, database_path=str(tmp_path / "lcm.db"),
                       large_output_externalization_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.on_session_start("P", platform="cli", context_length=200_000, conversation_id="conv")
    try:
        live = [{"role": "system", "content": "stable system prompt"}]
        for i in range(earlier):
            live += [_u(f"T{i} prompt", 10 * i), _a(f"R{i} reply", 10 * i + 1)]
        live += [_u("TQ tool prompt", 100), _call("call_p", 101), _result("call_p", "Pfile text", 102),
                 _a("RQ after call", 103)]
        engine.ingest(live)
        out = engine.compress(live)
        engine.on_session_end("P", list(live))
        if failure == "write":
            write = engine._store.write_metadata_json

            def write_metadata(keys, *args, **kwargs):
                if "rotation_predecessor_session:C" in keys:
                    raise OSError("synthetic predecessor write failure")
                return write(keys, *args, **kwargs)

            monkeypatch.setattr(engine._store, "write_metadata_json", write_metadata)
        elif failure == "chain":
            # Both lineage sources are unavailable: the prior process never wrote the record.
            monkeypatch.setattr(engine, "_record_rotation_predecessor", lambda *args: None)
            from hermes_lcm import identity_anchor as anchor_module
            monkeypatch.setattr(anchor_module.sqlite3, "connect", lambda *a, **kw: (_ for _ in ()).throw(
                sqlite3.OperationalError("synthetic ancestry read failure")))
        engine.on_session_start("C", boundary_reason="compression", old_session_id="P", platform="cli",
                                context_length=200_000, conversation_id="conv")

        engine.ingest([*out, _u("N1 prompt", 500), _a("N1 reply", 501)])
        before = Counter(_key(r) for r in engine._store.get_session_messages("C") if r["role"] != "system")
        ins = next((i for i, m in enumerate(out) if m["role"] == "user" and "timestamp" in m), 1)
        changed = [*out[:ins], _u("X inserted prompt", 450), *out[ins:], _u("N1 prompt", 500), _a("N1 reply", 501),
                   _u("N2 prompt", 600), _a("N2 reply", 601)]
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            engine.ingest(changed)
        after = Counter(_key(r) for r in engine._store.get_session_messages("C") if r["role"] != "system")
        parent = Counter(_key(r) for r in engine._store.get_session_messages("P") if r["role"] != "system")
        added = after - before
        dup_of_p = sum(count for key, count in after.items() if parent[key])
        # Literal main multiset: inserted prompt, the two carried text rows, and the fresh turn.
        expected = Counter(_key(r) for r in (changed[ins], *changed[-2:]))
        expected.update(_key(r) for r in out if _key(r) in parent and r["role"] in {"user", "assistant"}
                        and not r.get("tool_calls"))
        assert added == expected
        assert (sum(added.values()), dup_of_p) == (5, 2)
        assert any("presence unavailable" in r.message and r.levelno == logging.WARNING for r in caplog.records)
        assert not before - after
        added_roles = Counter((key[0], key[2], key[3]) for key in added.elements())
        assert added_roles[("assistant", "", "call_p")] == added_roles[("tool", "call_p", "")] == 0
        engine.ingest(changed)
        assert Counter(_key(r) for r in engine._store.get_session_messages("C") if r["role"] != "system") == after
    finally:
        engine.shutdown()



@pytest.mark.parametrize("over", [0, 1], ids=["exact-bytes", "one-byte-over"])
def test_presence_byte_budget_ingest_boundary(tmp_path, monkeypatch, caplog, over):
    # The V1 setup has ten rows; count the stored representation, including tool-call JSON.
    original = LCMEngine._identity_anchor_unaccounted
    budgets = []

    def bounded(self, *args, **kwargs):
        size = self._store._conn.execute(
            "SELECT SUM(length(content) + COALESCE(length(tool_calls), 0)) FROM messages WHERE session_id=?",
            (SESSION,)).fetchone()[0]
        budgets.append(size)
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_BYTES", size - over, raising=False)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LCMEngine, "_identity_anchor_unaccounted", bounded)
    with caplog.at_level(logging.INFO, logger=identity_anchor.__name__):
        after, main, missing = _window_ingest(tmp_path, monkeypatch, "V1", 10000)
    assert budgets
    assert after == main + Counter({missing: 0 if over else 1})
    limits = [r for r in caplog.records if "presence window incomplete" in r.message]
    assert len(limits) == over
    assert all(r.levelno == logging.INFO for r in limits)
    assert not any("presence window" in r.message and r.levelno >= logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("over", [0, 1], ids=["exact-text", "one-char-over"])
def test_presence_text_budget_counts_only_the_candidate_role(tmp_path, monkeypatch, caplog, over):
    # The normalization bound counts only stored rows of a candidate's text role: user text here is not normalized.
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [{"role": "user", "content": "u" * 500}, _a("stored reply one", 2),
                                             _a("stored reply two", 3)])
        size = sum(len(r["content"]) for r in engine._store.get_session_messages(SESSION) if r["role"] == "assistant")
        monkeypatch.setattr(identity_anchor, "_PRESENCE_TEXT_BYTES", size - over, raising=False)
        with caplog.at_level(logging.INFO, logger=identity_anchor.__name__):
            assert _presence(engine, [_a("a brand new reply", 4)]) == ([] if over else [0])
        limits = [r for r in caplog.records if "presence window incomplete" in r.message]
        assert len(limits) == over
    finally:
        engine.shutdown()


def test_tool_only_presence_computes_no_stored_text_identity(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [_a("irrelevant reply", 1), _call("kept", 2), _result("kept", "held", 3)])
        replay = engine._message_replay_identity

        def identity(msg, *, stored_row=False):
            assert not stored_row, "a tool-only presence audit needs no stored text identity"
            return replay(msg, stored_row=stored_row)

        monkeypatch.setattr(engine, "_message_replay_identity", identity)
        assert _presence(engine, [_call("new", 4), _result("new", "new result", 5)]) == [0, 1]
    finally:
        engine.shutdown()


def test_presence_window_info_is_once_per_session_without_warning(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path)
    try:
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_BYTES", 1, raising=False)
        engine._store.append_batch(SESSION, [_a("over budget", 1)])
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=identity_anchor.__name__):
            for _ in range(2):
                assert _presence(engine, [_call("new", 2)]) == []
            engine.on_session_start("another-session", platform="cli", context_length=1_000_000)
            engine._store.append_batch("another-session", [_a("over budget too", 3)])
            assert _presence(engine, [_call("new", 4)]) == []
            engine.shutdown()
            engine = _engine(tmp_path)
            assert _presence(engine, [_call("new", 5)]) == []
        limits = [r for r in caplog.records if "presence window incomplete" in r.message]
        assert len(limits) == 2
        assert all(r.levelno == logging.INFO for r in limits)
        assert not any("presence window" in r.message and r.levelno >= logging.WARNING for r in caplog.records)
    finally:
        engine.shutdown()


def test_presence_reads_pages_newest_first_and_finds_an_old_copy(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        store = engine._store
        store.append_batch(SESSION, [{"role": "assistant", "content": f"reply {i}"} for i in range(600)])
        tail, through, pages = store.get_session_tail, store.get_session_rows_through, []

        def read_tail(source, *, limit):
            rows = tail(source, limit=limit)
            pages.append([r["store_id"] for r in rows])
            return rows

        def read_through(source, end, limit):
            rows = through(source, end, limit)
            pages.append([r["store_id"] for r in rows])
            return rows

        monkeypatch.setattr(store, "get_session_tail", read_tail)
        monkeypatch.setattr(store, "get_session_rows_through", read_through)
        assert _presence(engine, [{"role": "assistant", "content": "reply 0"}]) == []
        assert list(map(len, pages)) == [256, 256, 88]
        assert pages[0][0] > pages[1][-1] > pages[2][-1]
        assert len({sid for page in pages for sid in page}) == 600
    finally:
        engine.shutdown()


def test_text_presence_only_normalizes_matching_role_text_rows(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [_u("other role", 1), _a("kept reply", 2),
                                             _call("kept", 3), _result("kept", "held", 4)])
        replay, seen = engine._message_replay_identity, []

        def identity(msg, *, stored_row=False):
            if stored_row:
                seen.append((msg["role"], bool(msg.get("tool_calls"))))
            return replay(msg, stored_row=stored_row)

        monkeypatch.setattr(engine, "_message_replay_identity", identity)
        assert _presence(engine, [{"role": "assistant", "content": "new reply"}]) == [0]
        assert seen == [("assistant", False)]
    finally:
        engine.shutdown()



@pytest.mark.parametrize("seeded", [False, True], ids=["exact-then-alias", "held-seeded-tail"])
def test_presence_takes_an_exact_copy_before_a_cleanup_alias(tmp_path, monkeypatch, seeded):
    # #2466 review: an exact stored copy is consumed before another row's cleanup alias, in store order, also when
    # the audited tail (``held``) seeds the newest row first. Both displayed replies are stored: nothing is new.
    engine = _engine(tmp_path)
    try:
        old = {"role": "assistant", "content": "old reasoning then reply"}
        reply = {"role": "assistant", "content": "reply"}
        first, second = (old, reply) if seeded else (reply, old)
        engine._store.append_batch(SESSION, [first, *[{"role": "user", "content": f"padding {i}"}
                                                     for i in range(598)], second])

        def cleanup(identity):
            return (identity[0], "reply", *identity[2:]) if identity[1] == old["content"] else None

        monkeypatch.setattr(engine, "_active_cleanup_replay_identity", cleanup)
        held = {}
        if seeded:
            newest = [r for r in engine._store.get_session_messages(SESSION) if r["content"] == "reply"]
            held = {("assistant", "reply"): newest}
        assert _presence(engine, [reply, old], held=held) == []
    finally:
        engine.shutdown()


def test_presence_gives_an_alias_only_to_demand_an_exact_copy_cannot_meet(tmp_path, monkeypatch):
    # #2466 review (adversarial): two reasoning rows clean to the same reply. The clean reply must take the older
    # row's alias, not the newer row that the displayed newer reasoning reply needs exactly. Nothing is new.
    engine = _engine(tmp_path)
    try:
        older = {"role": "assistant", "content": "older reasoning then reply"}
        newer = {"role": "assistant", "content": "newer reasoning then reply"}
        engine._store.append_batch(SESSION, [older, *[{"role": "user", "content": f"padding {i}"}
                                                     for i in range(598)], newer])

        def cleanup(identity):
            return (identity[0], "reply", *identity[2:]) if identity[1] in {older["content"], newer["content"]} else None

        monkeypatch.setattr(engine, "_active_cleanup_replay_identity", cleanup)
        held = {("assistant", newer["content"]): [r for r in engine._store.get_session_messages(SESSION)
                                                  if r["content"] == newer["content"]]}
        assert _presence(engine, [{"role": "assistant", "content": "reply"}, newer], held=held) == []
    finally:
        engine.shutdown()


def test_presence_keeps_main_result_when_cleanup_forms_chain(tmp_path, monkeypatch, caplog):
    # A cleanup form that is itself the form of a row with an alias breaks the exact-first allocation's optimality:
    # the audit then keeps main's result (only its own finds) with a WARNING.
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [{"role": "assistant", "content": "x one"},
                                             {"role": "assistant", "content": "x two"}])
        chain = {"x one": "x two", "x two": "x three"}

        def cleanup(identity):
            return (identity[0], chain[identity[1]], *identity[2:]) if identity[1] in chain else None

        monkeypatch.setattr(engine, "_active_cleanup_replay_identity", cleanup)
        with caplog.at_level(logging.WARNING, logger=identity_anchor.__name__):
            assert _presence(engine, [{"role": "assistant", "content": "a brand new reply"}]) == []
        assert any("presence unavailable" in r.message for r in caplog.records)
    finally:
        engine.shutdown()


def test_presence_never_stores_a_result_whose_call_the_ignore_patterns_drop(tmp_path, monkeypatch):
    # #2466 review: a call row the ignore patterns drop is never stored, so it cannot authorize storing its result.
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [_a("irrelevant reply", 1)])
        dropped = {**_call("call_new", 2), "content": "IGNORE this call"}
        monkeypatch.setattr(engine, "_matches_ignore_message_patterns",
                            lambda msg, **_k: str(msg.get("content") or "").startswith("IGNORE"))
        assert _presence(engine, [dropped, _result("call_new", "new result", 3)]) == []
        assert _presence(engine, [_call("call_kept", 4), _result("call_kept", "kept result", 5)]) == [0, 1]
    finally:
        engine.shutdown()


def test_presence_with_many_identical_copies_is_linear(tmp_path):
    # #2466 review: ~10k identical replies in one key bucket; every displayed copy is stored, nothing is new, and the
    # audit does not rescan the bucket per row (quadratic took seconds). The bound here is loose for CI; the measured
    # lock-time figure is in the PR body.
    import time

    engine = _engine(tmp_path)
    try:
        rows = [{"role": "assistant", "content": "OK"} for _ in range(9990)]
        for k in range(0, len(rows), 1000):
            engine._store.append_batch(SESSION, rows[k:k + 1000])
        began = time.perf_counter()
        assert _presence(engine, rows) == []
        assert time.perf_counter() - began < 1.5
        # Half the view is the unchanged prefix: its reserved copies are queued once per key, not per candidate.
        began = time.perf_counter()
        assert _presence(engine, rows, start=len(rows) // 2) == []
        assert time.perf_counter() - began < 1.5
    finally:
        engine.shutdown()


def test_paged_presence_preserves_cleanup_copy_consumption_order(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        old = {"role": "assistant", "content": "old reasoning then reply"}
        reply = {"role": "assistant", "content": "reply"}
        engine._store.append_batch(SESSION, [old, *[{"role": "user", "content": f"padding {i}"}
                                                   for i in range(598)], reply])

        def cleanup(identity):
            return (identity[0], "reply", *identity[2:]) if identity[1] == old["content"] else None

        monkeypatch.setattr(engine, "_active_cleanup_replay_identity", cleanup)
        # Consume the newest direct copy first, leaving the oldest copy for its original form.
        assert _presence(engine, [reply, old]) == []
    finally:
        engine.shutdown()
