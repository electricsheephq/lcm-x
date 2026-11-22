"""Kept rows mirror the host's refreshable keys in both directions (#2855)."""

import copy

import pytest

from hermes_lcm import engine as engine_module, host_uid_emit
from hermes_lcm.tokens import count_messages_tokens
from tests.harness.engine import engine_context

API_TEXT = "API-side reasoning text for the empty assistant row."
ABSENT = object()


@pytest.fixture(params=[None, "shadow", "on"], ids=["uid-unset", "uid-shadow", "uid-on"])
def uid_mode(request, monkeypatch):
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    if request.param is not None:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", request.param)


@pytest.fixture
def open_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: ("Synthetic leaf summary.", 1))
    return lambda folder="": engine_context(
        tmp_path / folder, window=200_000, fresh_tail_count=6, leaf_chunk_tokens=400)


def _history():
    rows = [row for i in range(12) for row in (
        {"role": "user", "content": f"Question {i}: " + "alpha beta gamma delta " * 40},
        {"role": "assistant", "content": f"Answer {i}.", "reasoning_content": f"Reasoning {i}."},
    )]
    for index, row in enumerate(rows, 1):
        row.update(message_uid=f"{index:032x}", timestamp=1_700_000_000.0 + index, host_extra=[index])
    return rows


def _call(engine, rows, *, leaves):
    given = copy.deepcopy(rows)
    if not leaves:
        # Below the host threshold: the cleanup path emits the replay view without a summary.
        engine._preflight_automatic_request = True
    result = engine.compress(given, current_tokens=count_messages_tokens(given), force=bool(leaves))
    assert given == rows
    assert (engine._last_compress_leaves or (None, 0))[1] == leaves
    return result


def _set_content(row, form):
    row.pop("content", None)
    if form is not ABSENT:
        row["content"] = form


def _empty_content_change(open_engine, before, after, *, position=-1, leaves=0):
    history = _history()
    api_only = {"role": "assistant", "api_content": API_TEXT,
                "message_uid": "a" * 32, "timestamp": 1_700_000_100.0}
    _set_content(api_only, before)
    # Both positions sit in the fresh tail, so a summary pass keeps the row too.
    history[position] = api_only
    with open_engine() as engine:
        engine.ingest(history)
        # The replay identity reads "", None and no content as the same empty content.
        _set_content(api_only, after)
        result = _call(engine, history, leaves=leaves)
    return [row for row in result if row.get("role") == "assistant" and row.get("content", ABSENT) in ("", None, ABSENT)]


FORMS = {"empty": "", "none": None, "absent": ABSENT}
EMPTY_CHANGES = pytest.mark.parametrize("before, after", [
    pytest.param(FORMS[a], FORMS[b], id=f"{a}-to-{b}") for a in FORMS for b in FORMS if a != b])
POSITIONS = pytest.mark.parametrize("position", [-1, -5], ids=["tail", "mid"])
LEAVES = pytest.mark.parametrize("leaves", [0, 1], ids=["no-leaf", "leaf"])


@LEAVES
@POSITIONS
@EMPTY_CHANGES
def test_api_only_row_survives_empty_content_change(open_engine, uid_mode, before, after, position, leaves):
    returned = _empty_content_change(open_engine, before, after, position=position, leaves=leaves)
    assert [row.get("api_content") for row in returned] == [API_TEXT]


@LEAVES
@POSITIONS
@EMPTY_CHANGES
def test_api_only_row_takes_host_empty_representation(open_engine, uid_mode, before, after, position, leaves):
    returned, = _empty_content_change(open_engine, before, after, position=position, leaves=leaves)
    assert returned.get("content", ABSENT) is after


@pytest.mark.parametrize("leaves", [0, 1], ids=["no-leaf", "leaf"])
def test_key_the_host_removed_leaves_the_kept_row(open_engine, uid_mode, leaves):
    history = _history()
    stamps = {row["content"]: row["timestamp"] for row in history}
    with open_engine() as engine:
        engine.ingest(history)
        # Both rows sit in the fresh tail; the assistant row before them keeps its key.
        stripped = [row for row in history if row["role"] == "assistant"][-2:]
        for row in stripped:
            del row["reasoning_content"]
            # The engine owns the stamp it recorded at ingest.
            del row["timestamp"]
        result = _call(engine, history, leaves=leaves)
        cached = {row["content"]: row for row in engine._last_active_replay_messages}

    by_content = {row["content"]: row for row in result if isinstance(row.get("content"), str)}
    for source in stripped:
        for output in (by_content[source["content"]], cached[source["content"]]):
            assert "reasoning_content" not in output
            assert output["host_extra"] == source["host_extra"]
        assert cached[source["content"]]["timestamp"] == stamps[source["content"]]
        if leaves:
            # The below-threshold cleanup returns this row without its stamp on main as well.
            assert by_content[source["content"]]["timestamp"] == stamps[source["content"]]
    kept = [row for row in history if row["role"] == "assistant" and row not in stripped
            and row["content"] in by_content]
    assert kept and all(by_content[row["content"]]["reasoning_content"] == row["reasoning_content"] for row in kept)


EXCLUDED = {"role", "content", "tool_calls", "tool_call_id", "api_content", "timestamp",
            *host_uid_emit.IDENTITY_KEYS, *host_uid_emit.ADDRESS_KEYS}


def _tool_turn(call_id, result_text, *, call_content=""):
    call = {"role": "assistant", "tool_calls": [{
        "id": call_id, "type": "function", "function": {"name": "test_tool", "arguments": "{}"},
    }]}
    if call_content is not ABSENT:
        call["content"] = call_content
    return call, {"role": "tool", "tool_call_id": call_id, "content": result_text}


def _with_turn(rows):
    history = _history()
    history.extend([{"role": "user", "content": "Probe ask."}, *rows])
    for index, row in enumerate(history[-len(rows) - 1:], 101):
        row.update(message_uid=f"{index:032x}", timestamp=1_700_000_000.0 + index)
    return history


@LEAVES
def test_kept_rows_carry_no_key_their_host_row_lacks(open_engine, uid_mode, leaves):
    """Guard: the engine writes only excluded keys on a kept row, so the refresh never strips one of its own."""
    call, result = _tool_turn("call_guard", "Synthetic guard result")
    api_only = {"role": "assistant", "content": "", "api_content": API_TEXT}
    history = _with_turn([call, api_only, result, {"role": "assistant", "content": "Final reply."}])
    with open_engine() as engine:
        engine.ingest(history)
        for when in ("after ingest", "after pass"):
            generated = engine._generated_ignored_active_replay_placeholder_message_ids
            cached = engine._last_active_replay_messages
            assert len(cached) == len(history), when
            extra = [(when, index, sorted(map(str, row.keys() - host.keys() - EXCLUDED)))
                     for index, (host, row) in enumerate(zip(history, cached)) if id(row) not in generated]
            assert not [item for item in extra if item[2]]
            if when == "after ingest":
                _call(engine, history, leaves=leaves)


def test_model_only_row_between_call_and_result_keeps_the_real_result(open_engine, uid_mode):
    """C1: the host turns the model-only row's "" into None on a below-threshold replay."""
    call, result = _tool_turn("x", "REAL RESULT for x")
    api_only = {"role": "assistant", "content": "", "api_content": API_TEXT}
    history = _with_turn([call, api_only, result, {"role": "assistant", "content": "Final reply."}])
    with open_engine() as engine:
        engine.ingest(history)
        api_only["content"] = None
        output = _call(engine, history, leaves=0)
        stub = engine._missing_tool_result_stub("x")

    kept = [row for row in output if row.get("api_content") == API_TEXT]
    assert len(kept) == 1 and kept[0]["content"] is None
    results = [row for row in output if row.get("role") == "tool" and row.get("tool_call_id") == "x"]
    assert [row["content"] for row in results] == ["REAL RESULT for x"] != [stub["content"]]
    calls = [index for index, row in enumerate(output) if row.get("tool_calls")]
    assert len(calls) == 1 and output.index(results[0]) == calls[0] + 1


def _comparable(rows):
    return [{key: value for key, value in row.items() if key != "timestamp"} for row in rows]


def _warm_and_cold(open_engine, history, change):
    with open_engine("warm") as engine:
        engine.ingest(history)
        change(history)
        warm = _call(engine, history, leaves=0)
    with open_engine("cold") as engine:
        cold = _call(engine, history, leaves=0)
    return _comparable(warm), _comparable(cold)


def test_call_row_gaining_empty_content_matches_cold_start(open_engine, uid_mode):
    """C2, WARM EQUALS COLD: the host adds content "" to a call row stored without a content key.

    Main's warm path kept the "Checking" row and the call row separate. A fresh engine given the
    host's current rows merges them, as it always has for two adjacent string-content assistants;
    the warm path now produces that same output.
    """
    call, result = _tool_turn("x", "Result for x", call_content=ABSENT)
    history = _with_turn([{"role": "assistant", "content": "Checking"}, call, result])
    warm, cold = _warm_and_cold(open_engine, history, lambda rows: rows[-2].__setitem__("content", ""))
    assert warm == cold


def test_reasoning_removed_from_first_assistant_matches_cold_start(open_engine, uid_mode):
    """C3, WARM EQUALS COLD: the host strips only the first assistant's reasoning_content."""
    call, result = _tool_turn("x", "Result for x")
    call["reasoning_content"] = "B"
    history = _with_turn([{"role": "assistant", "content": "Checking", "reasoning_content": "A"}, call, result])
    warm, cold = _warm_and_cold(open_engine, history, lambda rows: rows[-3].pop("reasoning_content"))
    assert warm == cold
