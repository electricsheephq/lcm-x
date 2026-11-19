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
    return lambda: engine_context(tmp_path, window=200_000, fresh_tail_count=6, leaf_chunk_tokens=400)


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


def _empty_content_change(open_engine, before, after):
    history = _history()
    api_only = {"role": "assistant", "content": before, "api_content": API_TEXT,
                "message_uid": "a" * 32, "timestamp": 1_700_000_100.0}
    history[-1:] = [api_only]
    with open_engine() as engine:
        engine.ingest(history)
        # The replay identity reads "", None and no content as the same empty content.
        if after is ABSENT:
            del api_only["content"]
        else:
            api_only["content"] = after
        result = _call(engine, history, leaves=0)
    return [row for row in result if row.get("role") == "assistant" and row.get("content") in ("", None)]


EMPTY_CHANGES = pytest.mark.parametrize(
    "before, after", [("", None), (None, ""), ("", ABSENT)],
    ids=["empty-to-none", "none-to-empty", "empty-to-absent"])


@EMPTY_CHANGES
def test_api_only_row_survives_empty_content_change(open_engine, uid_mode, before, after):
    returned = _empty_content_change(open_engine, before, after)
    assert [row.get("api_content") for row in returned] == [API_TEXT]


@EMPTY_CHANGES
def test_api_only_row_takes_host_empty_representation(open_engine, uid_mode, before, after):
    returned, = _empty_content_change(open_engine, before, after)
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
