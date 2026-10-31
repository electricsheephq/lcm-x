"""#773: Hermes's anti-growth salvage rewrites old tool outputs in its list: the rewritten rows are replays.

When a committed compaction would grow the list, Hermes (agent/conversation_compression.py
``_salvage_or_refuse_grown_transcript`` -> agent/context_compressor.py ``salvage_grown_transcript``)
replaces every string tool output over 200 characters, except the newest two, with exactly
``[Old tool output cleared to save context space]`` (the tool_call_id stays) and pops the reasoning
of every assistant row but the last. LCM keeps the full output in its store and treats such a host row
as the stored tool row with that tool_call_id: no new row, and the host keeps its placeholder.

The source is the one output under that tool_call_id in this session that the salvage could have
cleared (a string output over 200 characters, not a structured tool result), with the same tool name, else one where a side has no name (WebUI rows
carry none). A reused id with two different such outputs, or an id whose first row is the placeholder
text itself, has no source: the row stays new content, as before. A placeholder copy an older release
stored is read as its source too, so a store that already holds copies maps the host list as it did.
A row that took its source's identity is never a tool anchor or segment replay proof (#1385).

Every engine here has both externalization flags off, so the store and the host hold real outputs
(no LCM stubs, unlike #772).
"""

import copy
import json
import logging

import pytest

from hermes_lcm.tokens import count_messages_tokens
from tests.test_issue_671_stub_first_exit import (  # noqa: F401 (fixtures)
    exit_engine,
    make_engine,
    payload_of,
    run,
    summaries,
    tool_content,
    tool_pair,
)
from tests.test_issue_772_replay_keeps_host_stubs import CLEARED, _adopt

SYSTEM = {"role": "system", "content": "system prompt"}
WEDGE = "leaf would end before an unresolved retained occurrence"


def plain_engine(build, **overrides):
    return exit_engine(build, large_output_externalization_enabled=False,
                       large_output_active_replay_stubbing_enabled=False, **overrides)


def real_view(pairs=14, tokens=1_500, word="w{}x"):
    """Tool outputs under every stub threshold, with reasoning on each caller."""
    rows = []
    for i in range(pairs):
        call, result = tool_pair(f"call-{i}", payload_of(tokens, word.format(i)))
        rows += [{"role": "user", "content": f"turn {i}"}, {**call, "reasoning": f"why {i}"},
                 {**result, "tool_name": "read_file"}]
    return rows + [{"role": "user", "content": "latest ask"}, {"role": "assistant", "content": "latest answer"}]


def hermes_salvage(messages):
    """agent/context_compressor.py salvage_grown_transcript: reasoning off all but the last assistant,
    string tool outputs over 200 chars cleared except the newest two."""
    out = [dict(m) for m in messages]
    assistants = [i for i, m in enumerate(out) if m.get("role") == "assistant"]
    for i in assistants[:-1]:
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            out[i].pop(key, None)
    tools = [i for i, m in enumerate(out) if m.get("role") == "tool"]
    for i in tools[:-2]:
        if isinstance(out[i].get("content"), str) and len(out[i]["content"]) > 200:
            out[i]["content"] = CLEARED
    return out


def stored(engine):
    return engine._store.get_session_messages(engine._session_id)


def cleared_ids(messages):
    return [m["tool_call_id"] for m in messages if m.get("role") == "tool" and m.get("content") == CLEARED]


@pytest.mark.parametrize("word", ["w{}x", "출력{}결과"], ids=["ascii", "korean"])
@pytest.mark.parametrize("mode", ["same_process", "needs_reconcile", "fresh_engine"])
def test_a_salvaged_list_stores_only_the_new_row(make_engine, summaries, caplog, mode, word):  # noqa: F811
    """Hermes clears by character count (over 200) and the placeholder is one English constant, so a
    Korean output is cleared and recognised the same way."""
    engine = plain_engine(make_engine)
    view = real_view(word=word)
    payloads = {m["tool_call_id"]: m["content"] for m in view if m.get("role") == "tool"}
    out = run(engine, [SYSTEM, *view], caplog)
    salvaged = hermes_salvage(out)
    cleared = cleared_ids(salvaged)
    assert cleared  # the host cleared stored, un-stubbed outputs
    before = len(stored(engine))

    host = _adopt(engine, salvaged, new_rows=({"role": "user", "content": "third ask"},))
    if mode == "needs_reconcile":
        engine._ingest_cursor_needs_reconcile = True
    if mode == "fresh_engine":  # a restart: a new process binds the same store
        sid, conversation, db = engine._session_id, engine._conversation_id, engine._config.database_path
        engine = plain_engine(make_engine, database_path=db)
        engine.on_session_start(sid, conversation_id=conversation, context_length=200_000)
    replay = engine._ingest_messages(host)

    rows = stored(engine)
    assert len(rows) - before == 1  # only the new user row: no placeholder and no copy of a stored row
    assert rows[-1]["role"] == "user" and rows[-1]["content"] == "third ask"
    assert not any(row.get("content") == CLEARED for row in rows)
    call_ids = [row["tool_call_id"] for row in rows if row.get("role") == "tool"]
    assert len(call_ids) == len(set(call_ids))
    assert all(tool_content(rows, cid) == payloads[cid] for cid in cleared)  # the store keeps every full output
    assert all(tool_content(replay, cid) == CLEARED for cid in cleared)  # the host keeps its placeholder
    assert all(tool_content(engine._last_active_replay_messages, cid) == CLEARED for cid in cleared)
    assert engine._ingest_messages(copy.deepcopy(host)) == replay  # the no-new-rows path: the same view
    assert len(stored(engine)) == len(rows)


@pytest.mark.parametrize("new_rows", [(), ({"role": "user", "content": "next ask"},)])
def test_the_cached_view_never_hands_the_full_output_back(make_engine, new_rows):  # noqa: F811
    """The placeholder and its stored row share an identity, so the replay cache (#772) would put the
    full output back into the host's list; the host keeps its placeholder, the store its output."""
    engine = plain_engine(make_engine, fresh_tail_count=32)
    view = real_view(pairs=4)
    engine._ingest_messages(view)
    host = hermes_salvage(view) + list(new_rows)
    cleared = cleared_ids(host)
    assert cleared

    replay = engine._ingest_messages(host)

    assert all(tool_content(replay, cid) == CLEARED for cid in cleared)
    assert all(tool_content(engine._last_active_replay_messages, cid) == CLEARED for cid in cleared)
    assert len(stored(engine)) == len(view) + len(new_rows)


def test_compaction_keeps_making_progress_after_salvages(make_engine, summaries, caplog):  # noqa: F811
    """A list this small does not reach the production stop, so this pins the cause instead: after every
    salvage the store holds only the new rows, and every stored row above the frontier is a row of the
    returned list (no copies the list hides, which store-complete leaves chase)."""
    engine = plain_engine(make_engine)
    view = real_view()
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        out = engine.compress(view, current_tokens=count_messages_tokens(view))
        for turn in range(4):
            new_rows = ({"role": "user", "content": f"round {turn}"},
                        *tool_pair(f"round-{turn}", payload_of(1_500, f"r{turn}y")),
                        {"role": "assistant", "content": f"answer {turn}"})
            before, frontier = len(stored(engine)), engine._last_compacted_store_id
            host = _adopt(engine, hermes_salvage(out), new_rows=new_rows)
            engine._ingest_messages(host)
            assert len(stored(engine)) - before == len(new_rows)
            out = engine.compress(host, current_tokens=max(count_messages_tokens(host), 13_000))

            assert engine.last_compression_status == "compacted"
            assert engine._last_compacted_store_id > frontier
            mapped = engine._get_store_id_map_for_messages(out)
            above = {row["store_id"] for row in stored(engine) if row["store_id"] > engine._last_compacted_store_id}
            assert above <= {mapped[id(m)] for m in out if id(m) in mapped}  # nothing above it is hidden
    assert not any(row.get("content") == CLEARED for row in stored(engine))
    assert not any(WEDGE in record.getMessage() for record in caplog.records)


def test_a_host_placeholder_maps_onto_the_original_before_a_stored_copy(make_engine):  # noqa: F811
    """A store an older release wrote holds a placeholder copy after the original: the host row maps to the
    original, and both share the original's identity."""
    engine = plain_engine(make_engine, fresh_tail_count=32)
    view = real_view(pairs=4)
    engine._ingest_messages(view)
    original = next(row for row in stored(engine) if row.get("tool_call_id") == "call-1" and row["role"] == "tool")
    copy_id = engine._store.append(engine._session_id, {"role": "tool", "tool_call_id": "call-1", "tool_name": "read_file",
                                                        "content": CLEARED}, conversation_id=engine._conversation_id)
    host = [dict(m, content=CLEARED) if m.get("tool_call_id") == "call-1" and m["role"] == "tool" else m for m in view]
    placeholder = next(m for m in host if m.get("content") == CLEARED)

    assert engine._get_store_id_map_for_messages(host)[id(placeholder)] == original["store_id"] != copy_id
    assert engine._message_replay_identity(placeholder) == engine._message_replay_identity(original, stored_row=True)


def _restarted(build):
    """The view is stored; then a restart binds the same store, so the replay identity decides what is new."""
    first = plain_engine(build, fresh_tail_count=32)
    view = real_view(pairs=4)
    first._ingest_messages(view)
    engine = plain_engine(build, fresh_tail_count=32, database_path=first._config.database_path)
    engine.on_session_start(first._session_id, conversation_id=first._conversation_id, context_length=200_000)
    return engine, view


def test_a_placeholder_with_no_stored_twin_is_stored(make_engine):  # noqa: F811
    """An unknown tool_call_id: a genuine tool output that happens to be the placeholder text."""
    engine, view = _restarted(make_engine)
    before = len(stored(engine))
    call, _ = tool_pair("new-call", "unused")
    tool = {"role": "tool", "tool_call_id": "new-call", "tool_name": "read_file", "content": CLEARED}

    engine._ingest_messages(view + [{"role": "user", "content": "again"}, call, tool])

    rows = stored(engine)[before:]
    assert [row["role"] for row in rows] == ["user", "assistant", "tool"] and rows[-1]["content"] == CLEARED


def test_another_session_never_vouches_for_a_placeholder(make_engine):  # noqa: F811
    first, view = _restarted(make_engine)
    other = plain_engine(make_engine, fresh_tail_count=32, database_path=first._config.database_path)
    host = [dict(m, content=CLEARED) if m.get("tool_call_id") == "call-1" and m["role"] == "tool" else m for m in view]
    assert other._session_id != first._session_id and other._conversation_id != first._conversation_id

    other._ingest_messages(host)

    assert len(stored(other)) == len(host) and tool_content(stored(other), "call-1") == CLEARED


@pytest.mark.parametrize("rewrite", ["placeholder", "substring", "other_tool", "user_role"])
def test_only_the_exact_placeholder_on_its_own_tool_row_is_a_replay(make_engine, rewrite):  # noqa: F811
    engine, view = _restarted(make_engine)
    index = next(i for i, m in enumerate(view) if m.get("tool_call_id") == "call-1" and m["role"] == "tool")
    changed = {
        "placeholder": {**view[index], "content": CLEARED},
        "substring": {**view[index], "content": f"{CLEARED} (partial)"},
        "other_tool": {**view[index], "content": CLEARED, "tool_name": "write_file"},
        "user_role": {"role": "user", "content": CLEARED},
    }[rewrite]
    before = len(stored(engine))

    engine._ingest_messages(view[:index] + [changed] + view[index + 1:] + [{"role": "user", "content": "again"}])

    rows = stored(engine)[before:]
    if rewrite == "placeholder":
        assert [row["content"] for row in rows] == ["again"]  # the placeholder is its stored row: only the new turn
    else:  # a different row: stored as new content, as before (with the rows reconcile re-stores around it)
        assert any((row["role"], row["content"], row.get("tool_name")) == (
            changed["role"], changed["content"], changed.get("tool_name")) for row in rows)


def _older_release(monkeypatch, engine):
    """An older release: a host placeholder is new content (no source lookup at all)."""
    monkeypatch.setattr(type(engine), "_pruned_tool_output_identity", lambda self, *args, **kwargs: None)


def _rebind(build, engine, **overrides):
    """A restart: a new process binds the same store."""
    sid, conversation, db = engine._session_id, engine._conversation_id, engine._config.database_path
    engine = plain_engine(build, database_path=db, **overrides)
    engine.on_session_start(sid, conversation_id=conversation, context_length=200_000)
    return engine


@pytest.mark.parametrize("rounds", [1, 2])
def test_a_store_already_holding_placeholder_copies_maps_the_host_list_as_well(make_engine, summaries, caplog, monkeypatch, rounds):  # noqa: F811
    """A store an older release wrote holds placeholder copies and caller copies, and its leaves already
    summarised the originals the host list hid. A copy takes its original's identity like the host row, so
    the host list maps onto that store at least as well as the older release mapped it. If the original
    alone took the host row, the host placeholder would point at a summarised row and its caller at a later
    copy, and both would be unmapped."""
    engine = plain_engine(make_engine)
    out = run(engine, [SYSTEM, *real_view()], caplog)
    with monkeypatch.context() as patch:
        _older_release(patch, engine)
        for turn in range(rounds):
            new_rows = ({"role": "user", "content": f"round {turn}"},
                        *tool_pair(f"round-{turn}", payload_of(1_500, f"r{turn}y")),
                        {"role": "assistant", "content": f"answer {turn}"})
            host = _adopt(engine, hermes_salvage(out), new_rows=new_rows)
            engine._ingest_messages(host)
            out = engine.compress(host, current_tokens=max(count_messages_tokens(host), 13_000))
        host = _adopt(engine, hermes_salvage(out))
        engine = _rebind(make_engine, engine)
        before = engine._get_store_id_map_for_messages(host)
    rows = stored(engine)
    copies = [row for row in rows if row.get("content") == CLEARED]
    assert copies and any(row["store_id"] <= engine._last_compacted_store_id for row in rows
                          if row.get("tool_call_id") == copies[0]["tool_call_id"])  # the polluted shape

    engine = _rebind(make_engine, engine)
    after = engine._get_store_id_map_for_messages(host)

    assert len(after) >= len(before)
    assert all(id(m) in after for m in host if id(m) in before and m.get("content") == CLEARED)
    for row in copies:  # the copy, its original and the host placeholder are one occurrence
        original = next(r for r in rows if r.get("tool_call_id") == row["tool_call_id"] and r["role"] == "tool")
        assert (engine._message_replay_identity(row, stored_row=True)
                == engine._message_replay_identity(original, stored_row=True))
    assert len(stored(engine)) == len(rows)  # mapping never writes


def test_legacy_placeholder_copies_do_not_duplicate_on_follow_up_ingest(make_engine, monkeypatch):  # noqa: F811
    """A legacy writer persisted a complete salvaged view; the restart only adds the next turn."""
    first = plain_engine(make_engine, fresh_tail_count=32)
    view = real_view(pairs=4)
    first._ingest_messages(view)
    host = hermes_salvage(view)
    with monkeypatch.context() as patch:
        _older_release(patch, first)
        legacy = _rebind(make_engine, first, fresh_tail_count=32)
        legacy._ingest_messages(copy.deepcopy(host))
    engine = _rebind(make_engine, legacy, fresh_tail_count=32)
    before = stored(engine)
    assert any(row.get("content") == CLEARED for row in before)

    engine._ingest_messages(copy.deepcopy(host) + [{"role": "user", "content": "follow-up"}])

    assert stored(engine)[:-1] == before
    assert stored(engine)[-1]["content"] == "follow-up"
    assert _grouped(stored(engine))


def test_restart_replays_an_exact_tail_with_an_older_literal_placeholder(make_engine):  # noqa: F811
    """An old full X output must not mask the literal X occurrence in the durable tail."""
    first = plain_engine(make_engine, fresh_tail_count=32)
    original = [{"role": "user", "content": "original turn"}, _call("X"), _result("X", "X" * 300)]
    tail = [{"role": "user", "content": "later turn"}, _call("X"), _result("X", CLEARED),
            _call("Y"), _result("Y", "result Y"), _call("Z"), _result("Z", "result Z")]
    for row in original + tail:
        first._store.append(first._session_id, row, conversation_id=first._conversation_id)
    engine = _rebind(make_engine, first, fresh_tail_count=32)
    before = stored(engine)

    cursor = engine._reconcile_ingest_cursor_from_store(copy.deepcopy(tail))
    engine._ingest_messages(copy.deepcopy(tail))

    assert (cursor, stored(engine)) == (7, before)
    assert engine._last_ingest_reconciliation["cursor"] == 7
    assert _grouped(stored(engine))


def test_a_stored_placeholder_is_a_copy_only_of_an_earlier_row_of_this_session(make_engine):  # noqa: F811
    """A placeholder row stored before any same-id output (a genuine output that is the placeholder
    text), or stored by another session, keeps its own identity."""
    engine = plain_engine(make_engine, fresh_tail_count=32)
    sid, conversation = engine._session_id, engine._conversation_id
    first = engine._store.append(sid, {"role": "tool", "tool_call_id": "early", "tool_name": "read_file",
                                       "content": CLEARED}, conversation_id=conversation)
    engine._store.append(sid, {"role": "tool", "tool_call_id": "early", "tool_name": "read_file",
                               "content": "a later output"}, conversation_id=conversation)
    other = engine._store.append("another-session", {"role": "tool", "tool_call_id": "early",
                                                     "tool_name": "read_file", "content": CLEARED})
    rows = {row["store_id"]: row for row in engine._store.get_session_messages(sid)}
    foreign = engine._store.get_session_messages("another-session")[0]

    assert engine._message_replay_identity(rows[first], stored_row=True)[1] == CLEARED
    assert foreign["store_id"] == other and engine._message_replay_identity(foreign, stored_row=True)[1] == CLEARED


def test_other_rows_that_reuse_the_id_never_hide_the_original(make_engine):  # noqa: F811
    """Nine earlier rows reuse the call id under another tool: the original is still found, because the
    source lookup filters by tool name before it takes the earliest row."""
    reuse = []
    for i in range(9):
        call, result = tool_pair("call-1", f"other output {i}", tool_name="write_file")
        reuse += [{"role": "user", "content": f"pre {i}"}, call, {**result, "tool_name": "write_file"}]
    first = plain_engine(make_engine, fresh_tail_count=32)
    view = reuse + real_view(pairs=4)  # two newer outputs: call-1 really can be salvaged
    first._ingest_messages(view)
    engine = _rebind(make_engine, first, fresh_tail_count=32)
    before = len(stored(engine))
    host = [dict(m, content=CLEARED) if m.get("tool_call_id") == "call-1" and m.get("tool_name") == "read_file"
            else m for m in view]

    engine._ingest_messages(host + [{"role": "user", "content": "again"}])

    assert [row["content"] for row in stored(engine)[before:]] == ["again"]


@pytest.mark.parametrize("empty", ["host", "stored"])
def test_a_tool_name_only_one_side_carries_still_matches(make_engine, empty):  # noqa: F811
    """WebUI rows carry no tool name: a nameless placeholder is its named original, and a named
    placeholder is its nameless original. Two different names never match (see ``other_tool`` below)."""
    first = plain_engine(make_engine, fresh_tail_count=32)
    view = real_view(pairs=4)
    if empty == "stored":
        view = [dict(m, tool_name="") if m.get("role") == "tool" else m for m in view]
    first._ingest_messages(view)
    engine = _rebind(make_engine, first, fresh_tail_count=32)
    before = len(stored(engine))
    host = hermes_salvage(view)
    host = [dict(m, tool_name="read_file" if empty == "stored" else "") if m.get("content") == CLEARED else m
            for m in host]
    assert cleared_ids(host)

    replay = engine._ingest_messages(host + [{"role": "user", "content": "again"}])

    assert [row["content"] for row in stored(engine)[before:]] == ["again"]
    assert all(tool_content(replay, cid) == CLEARED for cid in cleared_ids(host))


def test_the_same_tool_name_wins_over_a_nameless_twin(make_engine):  # noqa: F811
    """A TUI original (named) and a WebUI full copy (nameless) of one call: each placeholder stands for
    the row with its own name, so a nameless placeholder maps onto the nameless copy it was cleared from."""
    engine = plain_engine(make_engine, fresh_tail_count=32)
    sid, conversation = engine._session_id, engine._conversation_id
    full = "full output " * 30  # over the salvage's 200 characters: only such a row can be cleared
    ids = [engine._store.append(sid, {"role": "tool", "tool_call_id": "c", "tool_name": name, "content": text},
                                conversation_id=conversation)
           for name, text in (("terminal", full), ("", full), ("", CLEARED))]
    named, nameless, copy_row = (next(r for r in stored(engine) if r["store_id"] == i) for i in ids)
    identity = engine._message_replay_identity

    assert identity({"role": "tool", "tool_call_id": "c", "content": CLEARED}) == identity(nameless, stored_row=True)
    assert (identity({"role": "tool", "tool_call_id": "c", "tool_name": "terminal", "content": CLEARED})
            == identity(named, stored_row=True))
    assert identity(copy_row, stored_row=True) == identity(nameless, stored_row=True)


@pytest.mark.parametrize("held", ["none", "unrelated"])
def test_the_cursor_search_hashes_prefixes_only_for_a_held_native_recovery_digest(make_engine, monkeypatch, held):  # noqa: F811
    """A prefix can match a native-recovery proof only when the session holds one, so with none the
    streaming prefix pass (#1090) skips native-recovery digests. Behaviour-neutral: a held digest no
    prefix matches makes the pass digest every prefix, and the ingest stores the same single row and
    hands back the same list."""
    engine, view = _restarted(make_engine)
    if held == "unrelated":
        engine._remember_native_recovery_replay_snapshot_digest("0" * 64)
    assert bool(engine._load_native_recovery_replay_snapshot_digests()) == (held == "unrelated")
    hashed = []
    prefix_digests = type(engine)._replay_snapshot_prefix_digests

    def recording(self, messages, **kwargs):
        hashed.append(kwargs.get("native_recovery", False))
        return prefix_digests(self, messages, **kwargs)

    monkeypatch.setattr(type(engine), "_replay_snapshot_prefix_digests", recording)
    host = hermes_salvage(view) + [{"role": "user", "content": "again"}]
    before = len(stored(engine))

    replay = engine._ingest_messages(host)

    assert any(hashed) == (held == "unrelated")
    assert [row["content"] for row in stored(engine)[before:]] == ["again"]
    assert [m.get("content") for m in replay] == [m.get("content") for m in host]


# -- review round 1: reused ids, lookups before the source, legacy conversation ids --------------------


def _call(call_id, args="{}", content=""):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": args}}]}


def _result(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "tool_name": "read_file", "content": text}


def _grouped(rows):
    """Tool grouping: every assistant call is answered right after it, and every tool row answers one."""
    open_ids = set()
    for row in rows:
        if row["role"] == "tool":
            if row.get("tool_call_id") not in open_ids:
                return False
            open_ids.discard(row["tool_call_id"])
            continue
        if open_ids:
            return False
        open_ids = {call["id"] for call in row.get("tool_calls") or ()} if row["role"] == "assistant" else set()
    return not open_ids


@pytest.mark.parametrize("caller", ["", "running tool"], ids=["contentless", "with_text"])
def test_a_genuine_second_call_whose_result_is_the_placeholder_text_is_stored(make_engine, caller):  # noqa: F811
    """A provider reuses an id for a second call of the same shape, and that call's result is literally
    the placeholder text. A restart hands over only the new rows: the result takes the stored output's
    identity, but that match is by construction, not by content, so it is no replay proof and the new
    call and its result are stored (#259), as they were before #773."""
    first = plain_engine(make_engine, fresh_tail_count=32)
    first._ingest_messages([SYSTEM, {"role": "user", "content": "u1"}, _call("X", content=caller),
                            _result("X", payload_of(300, "orig")), {"role": "user", "content": "u2"},
                            {"role": "assistant", "content": "a2"}])
    engine = _rebind(make_engine, first, fresh_tail_count=32)
    before = len(stored(engine))

    engine._ingest_messages([_call("X", content=caller), _result("X", CLEARED), {"role": "assistant", "content": "done"}])

    rows = stored(engine)
    assert [(row["role"], row["content"]) for row in rows[before:]] == [
        ("assistant", caller), ("tool", CLEARED), ("assistant", "done")]
    assert _grouped(rows)


def test_a_reused_id_never_aliases_one_occurrence_onto_another(make_engine):  # noqa: F811
    """Two calls of one tool reuse an id with different outputs, the salvage clears both, and a third call
    reusing it returned the placeholder text itself. Which occurrence a placeholder stands for is then
    unknown, so none takes another occurrence's identity or store row: each keeps its own identity, as
    before #773, and the store keeps every occurrence, grouped with its call."""
    engine = plain_engine(make_engine, fresh_tail_count=64)
    one, two = payload_of(300, "one"), payload_of(300, "two")
    view = [SYSTEM, {"role": "user", "content": "u1"}, _call("X", '{"n": 1}'), _result("X", one),
            {"role": "user", "content": "u2"}, _call("X", '{"n": 2}'), _result("X", two),
            {"role": "user", "content": "u3"}, _call("X", '{"n": 3}'), _result("X", CLEARED),
            {"role": "user", "content": "u4"}, _call("Y"), _result("Y", payload_of(300, "y")),
            _call("Z"), _result("Z", payload_of(300, "z")), {"role": "assistant", "content": "a4"}]
    engine._ingest_messages(view)
    originals = stored(engine)
    occurrence = {text: row["store_id"] for row in originals if row["role"] == "tool" for text in [row["content"]]}
    host = hermes_salvage(view)
    first, second, genuine = (m for m in host if m.get("tool_call_id") == "X")
    assert first["content"] == second["content"] == genuine["content"] == CLEARED  # the salvage cleared both
    by_id = {row["store_id"]: row for row in originals}
    identity = engine._message_replay_identity

    for row, own in ((first, occurrence[one]), (second, occurrence[two]), (genuine, occurrence[CLEARED])):
        others = {occurrence[one], occurrence[two]} - {own}  # the outputs it does not stand for
        assert all(identity(row) != identity(by_id[other], stored_row=True) for other in others)
        assert engine._get_store_id_map_for_messages(host).get(id(row)) in (None, own)

    engine = _rebind(make_engine, engine, fresh_tail_count=64)
    engine._ingest_messages(host + [{"role": "user", "content": "u5"}])

    rows = stored(engine)
    assert rows[:len(originals)] == originals  # stored rows are never changed
    assert {one, two} <= {row["content"] for row in rows}
    assert any(row["content"] == CLEARED and prev.get("tool_calls") and prev["tool_calls"][0]["function"]["arguments"]
               == '{"n": 3}' for prev, row in zip(rows, rows[1:]))  # the genuine third result, with its call
    assert _grouped(rows)


def test_a_placeholder_that_is_the_first_output_under_its_id_is_genuine(make_engine):  # noqa: F811
    """The first row under an id is the placeholder text: a genuine output. A later full output reusing the
    id is another occurrence, so neither a host placeholder nor a later stored copy is aliased onto it."""
    engine = plain_engine(make_engine, fresh_tail_count=32)
    sid, conversation = engine._session_id, engine._conversation_id
    ids = [engine._store.append(sid, _result("X", text), conversation_id=conversation)
           for text in (CLEARED, payload_of(300, "later"), CLEARED)]
    genuine, later, copy_row = (next(r for r in stored(engine) if r["store_id"] == i) for i in ids)
    identity = engine._message_replay_identity

    assert identity(_result("X", CLEARED)) == identity(genuine, stored_row=True) != identity(later, stored_row=True)
    assert identity(copy_row, stored_row=True) == identity(genuine, stored_row=True)


@pytest.mark.parametrize("writer", ["this_engine", "another_connection"])
def test_a_placeholder_looked_up_before_its_source_resolves_once_it_is_stored(make_engine, writer):  # noqa: F811
    """A placeholder with no stored row under its id is new content, but that answer is never kept past a
    write: once the source is stored, by this engine's ingest or by another connection to the same store,
    the placeholder is that row, and a restart decides the same."""
    from hermes_lcm.store import MessageStore

    engine = plain_engine(make_engine, fresh_tail_count=32)
    engine._ingest_messages([SYSTEM, {"role": "user", "content": "u1"}])
    placeholder = _result("X", CLEARED)
    assert engine._message_replay_identity(placeholder)[1] == CLEARED  # no source yet
    rows = [_call("X"), _result("X", payload_of(300, "late"))]
    if writer == "this_engine":
        engine._ingest_messages([SYSTEM, {"role": "user", "content": "u1"}, *rows])
    else:
        other = MessageStore(engine._config.database_path)
        try:
            for row in rows:
                other.append(engine._session_id, row, conversation_id=engine._conversation_id)
        finally:
            other.close()
    source = next(row for row in stored(engine) if row["role"] == "tool")

    warm = engine._message_replay_identity(placeholder)

    assert warm == engine._message_replay_identity(source, stored_row=True) != engine._message_replay_identity(
        dict(placeholder, content="other"))
    assert _rebind(make_engine, engine, fresh_tail_count=32)._message_replay_identity(placeholder) == warm


@pytest.mark.parametrize("padding", ["\t{}", "{}\n", " \t{}\r\n"], ids=["tab", "newline", "mixed"])
def test_a_source_under_a_padded_legacy_conversation_id_is_found(make_engine, padding):  # noqa: F811
    """An upgraded store can hold a legacy conversation_id padded with ASCII whitespace; the source lookup
    reads the conversation through the shared owned-conversation values (#581), like every
    conversation-range read, so a source under such an id is found and nothing is stored twice."""
    first = plain_engine(make_engine, fresh_tail_count=32)
    view = real_view(pairs=4)
    first._ingest_messages(view)
    conversation = first._conversation_id
    assert conversation
    first._store.connection.execute(
        "UPDATE messages SET conversation_id = ? WHERE session_id = ? AND role = 'tool'",
        (padding.format(conversation), first._session_id))
    first._store.connection.commit()
    engine = _rebind(make_engine, first, fresh_tail_count=32)
    before = len(stored(engine))
    host = hermes_salvage(view)
    assert cleared_ids(host)

    replay = engine._ingest_messages(host + [{"role": "user", "content": "again"}])

    assert [row["content"] for row in stored(engine)[before:]] == ["again"]
    assert all(tool_content(replay, cid) == CLEARED for cid in cleared_ids(host))


# -- review round 2: the whole-list replay, the salvage's 200-character floor --------------------------


@pytest.mark.parametrize("caller", ["", "running tool"], ids=["contentless", "with_text"])
def test_a_genuine_second_call_in_a_whole_list_after_a_restart_is_stored(make_engine, caller):  # noqa: F811
    """The same reused-id call as above, but the restart hands over the whole list. The stored prefix is a
    proven replay, and after it the new [call X, placeholder text] pair borrows the stored pair's identity.
    The list already holds that stored occurrence (its full row, in the proven prefix), so the borrowed
    identity proves no replayed tool segment, and the new call and its result are stored with the rest of
    the delta (#259), as before #773."""
    stored_view = [SYSTEM, {"role": "user", "content": "u1"}, _call("X", content=caller),
                   _result("X", payload_of(300, "orig")), {"role": "user", "content": "u2"},
                   {"role": "assistant", "content": "a2"}]
    first = plain_engine(make_engine, fresh_tail_count=64)
    first._ingest_messages(stored_view)
    engine = _rebind(make_engine, first, fresh_tail_count=64)
    before = len(stored(engine))

    engine._ingest_messages(stored_view + [{"role": "user", "content": "u3"}, _call("X", content=caller),
                                           _result("X", CLEARED), {"role": "assistant", "content": "done"}])

    rows = stored(engine)
    assert [(row["role"], row["content"]) for row in rows[before:]] == [
        ("user", "u3"), ("assistant", caller), ("tool", CLEARED), ("assistant", "done")]
    assert _grouped(rows)


def test_a_placeholder_never_stands_for_an_output_the_salvage_could_not_clear(make_engine):  # noqa: F811
    """The salvage clears only outputs over 200 characters, so an output of 200 or fewer is never the source
    of a placeholder: a host row with the placeholder text under that id is a different, genuine output and
    is stored as new content, as before #773. One character more and it is that row."""
    engine = plain_engine(make_engine, fresh_tail_count=32)
    sid, conversation = engine._session_id, engine._conversation_id
    identity = engine._message_replay_identity
    for call_id, size in (("at-floor", 200), ("over-floor", 201)):
        source_id = engine._store.append(sid, _result(call_id, "x" * size), conversation_id=conversation)
        source = next(row for row in stored(engine) if row["store_id"] == source_id)
        took = identity(_result(call_id, CLEARED)) == identity(source, stored_row=True)
        assert took == (size > 200), (call_id, size)

    view = [SYSTEM, {"role": "user", "content": "u1"}, _call("S"), _result("S", "short output " * 10),
            {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]
    first = plain_engine(make_engine, fresh_tail_count=32)
    first._ingest_messages(view)
    engine = _rebind(make_engine, first, fresh_tail_count=32)
    before = len(stored(engine))
    host = [dict(m, content=CLEARED) if m.get("tool_call_id") == "S" else m for m in view]

    engine._ingest_messages(host + [{"role": "user", "content": "u3"}])

    added = stored(engine)[before:]
    assert ("tool", CLEARED) in [(row["role"], row["content"]) for row in added]
    assert added[-1]["content"] == "u3"


def test_an_ambiguous_salvage_pair_after_a_new_row_is_stored(make_engine):  # noqa: F811
    """#1385: this shape is indistinguishable from a fresh same-id call returning the placeholder.
    Store the ambiguous pair; later pairs with their own content evidence remain proven replays."""
    first = plain_engine(make_engine, fresh_tail_count=64)
    a, b, c, d = (payload_of(300, word) for word in ("aaa", "bbb", "ccc", "ddd"))
    first._ingest_messages([SYSTEM, {"role": "user", "content": "u1"}, _call("A"), _result("A", a),
                            _call("B"), _result("B", b), _call("C"), _result("C", c), _call("D"), _result("D", d),
                            {"role": "assistant", "content": "a1"}])
    engine = _rebind(make_engine, first, fresh_tail_count=64)
    before = len(stored(engine))

    engine._ingest_messages([_call("A"), _result("A", a), {"role": "assistant", "content": "new"},
                             _call("B"), _result("B", CLEARED), _call("C"), _result("C", c), _call("D"), _result("D", d),
                             {"role": "user", "content": "u2"}])

    assert [row["content"] for row in stored(engine)[before:]] == ["new", "", CLEARED, "u2"]
    assert engine._last_ingest_reconciliation.get("replayed_tool_segment_rows") == 4


def test_issue_1385_fresh_placeholder_pair_outside_prefix_is_stored(make_engine):  # noqa: F811
    """All four conditions: proven prefix excludes X, exact reused call, literal placeholder,
    and two later tool outputs. Borrowed identity must not discard either fresh X row (I1/I8)."""
    first = plain_engine(make_engine, fresh_tail_count=64)
    x, a = payload_of(300, "xxx"), payload_of(300, "aaa")
    first._ingest_messages([SYSTEM, {"role": "user", "content": "u1"}, _call("X"), _result("X", x),
                            _call("A"), _result("A", a), {"role": "assistant", "content": "a1"}])
    engine = _rebind(make_engine, first, fresh_tail_count=64)
    before = stored(engine)
    prefix = [_call("A"), _result("A", a)]  # the older X occurrence is outside this proven prefix
    fresh = [{"role": "user", "content": "fresh turn"}, _call("X"), _result("X", CLEARED),
             _call("Y"), _result("Y", "new Y"), _call("Z"), _result("Z", "new Z")]
    host = prefix + fresh
    assert engine._reconcile_ingest_cursor_from_store(host) == len(prefix)
    original_x = next(row for row in before if row.get("tool_call_id") == "X")
    assert engine._message_replay_identity(fresh[2]) == engine._message_replay_identity(original_x, stored_row=True)

    engine._ingest_messages(host)

    rows = stored(engine)
    assert rows[:len(before)] == before
    assert [(row["role"], row["content"]) for row in rows[len(before):]] == [
        (row["role"], row["content"]) for row in fresh]
    assert rows[len(before) + 1]["tool_calls"] == fresh[1]["tool_calls"]
    assert _grouped(rows)


def test_a_placeholder_among_the_newest_two_outputs_is_a_genuine_call_after_a_restart(make_engine):  # noqa: F811
    """The salvage never clears the newest two tool outputs of its list, and the host only appends after it. After
    a restart the host list proves a stored pair, adds a new row, then a call that reuses a stored id with the same
    arguments and returns the placeholder text, as its newest tool output. Its stored same-id occurrence is outside
    the reconciled prefix, but the placeholder is no salvage product: the call and its result are stored."""
    first = plain_engine(make_engine, fresh_tail_count=64)
    x, a = payload_of(300, "xxx"), payload_of(300, "aaa")
    first._ingest_messages([SYSTEM, {"role": "user", "content": "u1"}, _call("X"), _result("X", x),
                            _call("A"), _result("A", a), {"role": "assistant", "content": "a1"}])
    engine = _rebind(make_engine, first, fresh_tail_count=64)
    before = len(stored(engine))

    engine._ingest_messages([_call("A"), _result("A", a), {"role": "assistant", "content": "new"},
                             _call("X"), _result("X", CLEARED)])

    assert [(row["role"], row["content"]) for row in stored(engine)[before:]][-1] == ("tool", CLEARED)
    assert [row["role"] for row in stored(engine)[before:]] == ["assistant", "assistant", "tool"]


@pytest.mark.parametrize("shape", ["content_parts", "multimodal"])
def test_a_structured_tool_result_is_never_the_source_of_a_placeholder(make_engine, shape):  # noqa: F811
    """The salvage clears only string outputs, so a structured tool result (typed content parts, or a
    ``_multimodal`` dict, stored in its canonical JSON form) never stands behind a placeholder, however long
    its text: a host row with the placeholder text under that id is a different, genuine output and is stored
    as new content, as before #773. A string output that merely holds JSON stays a source."""
    engine = plain_engine(make_engine, fresh_tail_count=32)
    sid, conversation = engine._session_id, engine._conversation_id
    identity = engine._message_replay_identity
    parts = [{"type": "text", "text": "structured " * 40}]
    content = parts if shape == "content_parts" else {"_multimodal": True, "content": parts, "text_summary": "s"}
    structured_id = engine._store.append(
        sid, {"role": "tool", "tool_call_id": "R", "content": content}, conversation_id=conversation)
    json_text_id = engine._store.append(sid, _result("J", '{"output": "' + "j" * 300 + '", "exit_code": 0}'),
                                        conversation_id=conversation)
    rows = {row["store_id"]: row for row in stored(engine)}
    assert len(str(rows[structured_id]["content"])) > 200
    assert identity(_result("R", CLEARED)) != identity(rows[structured_id], stored_row=True)
    assert identity(_result("J", CLEARED)) == identity(rows[json_text_id], stored_row=True)


def test_a_string_output_in_the_canonical_structured_form_fails_closed(make_engine):  # noqa: F811
    """Stored text cannot tell a structured tool result from a string output that is byte-identical to
    its canonical form, so such a string is read as structured: its cleared placeholder gets no source and
    stays new content, as before #773 (duplication, never aliasing). The usual JSON string, with keys in
    the producer's order, stays a source."""
    engine = plain_engine(make_engine, fresh_tail_count=32)
    sid, conversation = engine._session_id, engine._conversation_id
    identity = engine._message_replay_identity
    canonical = json.dumps([{"text": "c" * 300, "type": "text"}], ensure_ascii=False, sort_keys=True)
    producer = json.dumps([{"type": "text", "text": "p" * 300}])
    ids = {call_id: engine._store.append(sid, _result(call_id, text), conversation_id=conversation)
           for call_id, text in (("C", canonical), ("P", producer))}
    rows = {row["store_id"]: row for row in stored(engine)}
    assert identity(_result("C", CLEARED)) != identity(rows[ids["C"]], stored_row=True)
    assert identity(_result("P", CLEARED)) == identity(rows[ids["P"]], stored_row=True)


@pytest.mark.parametrize("protected", ["newest", "second_newest"])
def test_a_genuine_placeholder_after_a_protected_output_keeps_its_own_occurrence(make_engine, protected):  # noqa: F811
    """The salvage never clears the newest two tool outputs of its list, so an output in one of those
    positions never stands behind a placeholder. A later call that reuses its id and returns the placeholder
    text is a genuine result: it is stored with its call, maps onto its own row and never onto the
    protected output, and after a restart the same list replays without storing anything again."""
    engine = plain_engine(make_engine, fresh_tail_count=64)
    original = payload_of(300, "kept")
    tail = [] if protected == "newest" else [_call("Y"), _result("Y", payload_of(300, "y"))]
    view = ([SYSTEM, {"role": "user", "content": "u1"}, _call("X", '{"n": 1}'), _result("X", original)] + tail
            + [{"role": "user", "content": "u2"}, _call("X", '{"n": 2}'), _result("X", CLEARED),
               {"role": "assistant", "content": "done"}])
    engine._ingest_messages(view)
    rows = stored(engine)
    by_content = {row["content"]: row["store_id"] for row in rows if row["role"] == "tool"}
    assert {original, CLEARED} <= set(by_content) and _grouped(rows)  # both occurrences stored, grouped
    genuine = next(m for m in view if m.get("content") == CLEARED)
    kept = next(m for m in view if m.get("content") == original)
    mapping = engine._get_store_id_map_for_messages(view)
    assert mapping.get(id(kept)) == by_content[original]
    assert mapping.get(id(genuine)) in (None, by_content[CLEARED])  # never the protected output's row

    engine = _rebind(make_engine, engine, fresh_tail_count=64)
    engine._ingest_messages(view + [{"role": "user", "content": "u3"}])

    after = stored(engine)
    assert after[:len(rows)] == rows and [row["content"] for row in after[len(rows):]] == ["u3"]
    mapping = engine._get_store_id_map_for_messages(view)
    assert mapping.get(id(genuine)) in (None, by_content[CLEARED])
