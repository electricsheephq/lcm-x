"""#772: after a stub-first exit, the replay view keeps the tool-output stubs the host holds.

Hermes keeps the system prompt outside the list, so every view here drops the system row
(``tool_view()[1:]``): with it, LCM's note changes row 0 and the replay cache never matches.
"""

import copy
import hashlib
import json

from hermes_lcm.message_content import text_content_for_pattern_matching
from hermes_lcm.tokens import count_messages_tokens
from tests.test_issue_671_stub_first_exit import (  # noqa: F401 (fixtures)
    STUB,
    THRESHOLD,
    exit_engine,
    make_engine,
    run,
    summaries,
    tool_content,
    tool_view,
)

CLEARED = "[Old tool output cleared to save context space]"


def _adopt(engine, out, *, new_rows=({"role": "user", "content": "next ask"},)):
    """The host adopts compress()'s list in place and appends the next turn."""
    sid = engine._session_id
    engine.on_session_start(sid, boundary_reason="compression", old_session_id=sid,
                            conversation_id=engine._conversation_id, platform="cli")
    return [dict(m) for m in out] + [dict(m) for m in new_rows]


def _text(content):
    return text_content_for_pattern_matching(content) or ""


def _stubbed_ids(messages):
    return [m["tool_call_id"] for m in messages if m.get("role") == "tool" and _text(m["content"]).startswith(STUB)]


def _structured(view):
    """The same rows with every tool output as one text block (a supported list shape)."""
    return [dict(m, content=[{"type": "text", "text": m["content"]}]) if m.get("role") == "tool" else m for m in view]


def _exit_and_adopt(build, caplog, view=None, **adopt):
    engine = exit_engine(build)
    out = run(engine, tool_view()[1:] if view is None else view, caplog)
    assert engine._stub_first_exit_now is not None
    host = _adopt(engine, out, **adopt)
    stubbed = _stubbed_ids(host)
    assert stubbed and count_messages_tokens(host) < THRESHOLD
    return engine, host, stubbed


def _host_salvage(messages):
    """The host's anti-growth guard: clear every tool output except the newest two."""
    tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    return [dict(m, content=CLEARED) if i in tool_indexes[:-2] else dict(m) for i, m in enumerate(messages)]


def test_exit_stubs_survive_the_next_preflight(make_engine, summaries, caplog):  # noqa: F811
    engine, host, stubbed = _exit_and_adopt(make_engine, caplog)

    replay = engine._ingest_messages(host)

    assert all(tool_content(replay, cid).startswith(STUB) for cid in stubbed)  # rc2: the payload came back
    assert all(tool_content(engine._last_active_replay_messages, cid).startswith(STUB) for cid in stubbed)
    assert engine.should_compress_preflight(host) is False  # rc2: requested on the replay count


def test_cleanup_only_after_an_exit_never_grows_the_host_list(make_engine, summaries, caplog):  # noqa: F811
    engine, host, _ = _exit_and_adopt(make_engine, caplog)
    host_tokens = count_messages_tokens(host)
    engine._preflight_automatic_request = True  # what a preflight request hands compress() (#677)

    result = engine.compress(host, current_tokens=host_tokens)

    assert engine.last_compression_status == "sanitized" and summaries == []
    assert count_messages_tokens(result) <= host_tokens  # rc2: 15137 > 467


def test_a_turn_with_no_new_rows_keeps_every_stub(make_engine, summaries, caplog):  # noqa: F811
    engine, host, stubbed = _exit_and_adopt(make_engine, caplog)

    engine._ingest_messages(host)
    replay = engine._ingest_messages(copy.deepcopy(host))  # the no-new-rows path

    assert all(tool_content(replay, cid).startswith(STUB) for cid in stubbed)


def test_the_cached_view_keeps_every_stub_on_the_first_ingest_with_no_new_rows(make_engine, summaries, caplog):  # noqa: F811
    """Site B alone: the adopted list with nothing appended reaches the cached view directly."""
    engine, host, stubbed = _exit_and_adopt(make_engine, caplog, new_rows=())

    replay = engine._ingest_messages(host)
    fresh = engine._ingest_messages(copy.deepcopy(host))

    assert all(tool_content(replay, cid).startswith(STUB) for cid in stubbed)
    assert all(tool_content(engine._last_active_replay_messages, cid).startswith(STUB) for cid in stubbed)
    assert replay == fresh  # the stub row is what _ingest_messages produces for that host row


def test_rows_without_stubs_replay_as_before(make_engine, summaries, caplog):  # noqa: F811
    """Flags off: no externalization, so the cached view is today's, byte for byte (digest printed for base/head)."""
    engine = exit_engine(
        make_engine, large_output_externalization_enabled=False, large_output_active_replay_stubbing_enabled=False
    )
    turn1 = tool_view()[1:]
    replays = [engine._ingest_messages(turn1)]
    turn2 = turn1 + [{"role": "user", "content": "second ask"}, {"role": "assistant", "content": "second answer"}]
    replays.append(engine._ingest_messages(turn2))
    replays.append(engine._ingest_messages(copy.deepcopy(turn2)))  # no new rows
    out = run(engine, turn2, caplog)
    assert summaries  # a normal leaf pass, no exit
    host = _adopt(engine, out)
    replays.append(engine._ingest_messages(host))
    replays.append(engine._ingest_messages(copy.deepcopy(host)))  # no new rows

    assert replays[0] == turn1 and replays[1] == turn2 and replays[2] == turn2
    assert not any(str(m.get("content", "")).startswith(STUB) for replay in replays for m in replay)
    digest = hashlib.sha256(json.dumps(replays, sort_keys=True, default=str).encode()).hexdigest()
    print(f"T4 replay digest: {digest} rows={[len(r) for r in replays]}")


def test_the_store_never_holds_a_tool_output_twice(make_engine, summaries, caplog):  # noqa: F811
    engine, host, _ = _exit_and_adopt(make_engine, caplog)
    host_tokens = count_messages_tokens(host)
    engine._ingest_messages(host)
    engine._preflight_automatic_request = True  # what a preflight request hands compress() (#677)
    result = engine.compress(host, current_tokens=host_tokens)
    if count_messages_tokens(result) > host_tokens:  # the host refuses a grown list and salvages it
        result = _host_salvage(result)
    stored = engine._store.get_session_count(engine._session_id)

    nxt = _adopt(engine, result, new_rows=({"role": "user", "content": "third ask"},))
    engine._ingest_messages(nxt)

    rows = engine._store.get_session_messages(engine._session_id)
    call_ids = [r["tool_call_id"] for r in rows if r.get("role") == "tool"]
    assert len(call_ids) == len(set(call_ids))  # rc2: every cleared row stored again
    assert engine._store.get_session_count(engine._session_id) - stored == 1


def test_structured_tool_stubs_survive_the_next_preflight(make_engine, summaries, caplog):  # noqa: F811
    """Site A with list-shaped tool outputs: the stub keeps the text-block shape and must still count as a stub."""
    engine, host, stubbed = _exit_and_adopt(make_engine, caplog, view=_structured(tool_view()[1:]))
    assert all(isinstance(tool_content(host, cid), list) for cid in stubbed)

    replay = engine._ingest_messages(host)

    assert all(_text(tool_content(replay, cid)).startswith(STUB) for cid in stubbed)
    assert all(_text(tool_content(engine._last_active_replay_messages, cid)).startswith(STUB) for cid in stubbed)


def test_structured_tool_stubs_survive_a_turn_with_no_new_rows(make_engine, summaries, caplog):  # noqa: F811
    """Site B with list-shaped tool outputs."""
    engine, host, stubbed = _exit_and_adopt(make_engine, caplog, view=_structured(tool_view()[1:]), new_rows=())

    replay = engine._ingest_messages(host)
    again = engine._ingest_messages(copy.deepcopy(host))

    assert all(_text(tool_content(replay, cid)).startswith(STUB) for cid in stubbed)
    assert all(_text(tool_content(again, cid)).startswith(STUB) for cid in stubbed)
