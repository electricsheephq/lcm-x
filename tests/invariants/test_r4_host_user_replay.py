"""Synthetic live host replay shapes for the compaction return contract."""

from copy import deepcopy

import pytest

from hermes_lcm import tokens
from hermes_lcm.reconcile import _PRESERVED_TODO_CONTEXT_PREFIX, _strip_volatile_host_scaffolding
from tests.harness.engine import stored_rows
from tests.invariants.test_p0_host_user_rows import (
    _assistant, _compress, _history, _stamp, _user, open_engine as open_engine,
)
from tests.test_issue_671_stub_first_exit import tool_pair

SIDECAR = "s" * 1662


def _todo(text):
    return f"{_PRESERVED_TODO_CONTEXT_PREFIX}\n- [>] 1. {text} (in_progress)"


def _assert_replay(source, returned, cached):
    assert returned is not cached
    assert returned.get("content") == source.get("content")
    assert returned.get("api_content") == source.get("api_content")
    assert {key: value for key, value in returned.items() if key not in {"content", "api_content"}} == {
        key: value for key, value in cached.items() if key not in {"content", "api_content"}
    }


def test_flap_annotation_keeps_ingest_keys_and_store(open_engine):
    prefix = _history()
    plain = _user("x" * 46, timestamp=42.0, task_origin="ingest")
    host = [*_history()[-2:], plain]
    _stamp([*prefix, *host])
    annotation = _todo("x" * (1282 - len(_todo(""))))
    current = dict(plain, content=plain["content"] + "\n\n" + annotation, api_content=SIDECAR)
    assert len(annotation) == 1282 and len(current["content"]) == 1330
    assert _strip_volatile_host_scaffolding(current["content"]) == plain["content"]

    with open_engine() as engine:
        engine.ingest([*prefix, *host])
        stored = stored_rows(engine)
        host[-1] = current
        # Keep the store's earlier turns off the live view so the forced leaf
        # can make hidden progress while preserving the supplied row order.
        engine._ingest_cursor = len(host)
        engine._remember_active_replay_messages([*host[:-1], plain], [*host[:-1], plain],
                                                persist_placeholder_evidence=False)
        for force in (False, True, False, False):
            cache = engine._last_active_replay_messages[-1]
            if not force:
                engine._preflight_automatic_request = True
            returned = _compress(engine, host, force=force)
            assert (engine._last_compress_leaves or (None, 0))[1] == int(force)
            assert len(returned) == len(host)
            assert returned[:-1] == host[:-1]
            _assert_replay(current, returned[-1], cache)
            assert all(returned[-1] is not row for row in engine._last_active_replay_messages)
            assert stored_rows(engine) == stored
            host = returned


@pytest.mark.parametrize("route", ("cleanup", "leaf", "stub"))
def test_emptied_note_keeps_current_host_text(open_engine, route):
    tail = _user("\n\n" + _todo("synthetic note"),
                 timestamp=42.0, api_content=SIDECAR)
    if route == "stub":
        rows = [
            _user("Run the tool."), *tool_pair("aged", "synthetic output " * 900),
            tail, _assistant("Latest answer."),
        ]
    else:
        rows = [*_history(), tail]
    _stamp(rows)
    settings = dict(
        fresh_tail_count=2, large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1_000_000,
        large_output_active_replay_stubbing_enabled=True,
        large_output_active_replay_stub_threshold_tokens=10_000,
        large_output_active_replay_stub_aged_threshold_tokens=20,
    )
    with open_engine(**settings) as engine:
        engine.threshold_tokens = 2_000
        before = engine._message_replay_identity(tail)
        engine.ingest(rows)
        cached = next(row for row in engine._last_active_replay_messages if row.get("message_uid") == tail["message_uid"])
        assert "[LCM sensitive redaction:" not in cached["content"]
        stored = stored_rows(engine)
        tail.update(content="", api_content="")
        assert engine._message_replay_identity(tail) == before
        if route == "cleanup":
            engine._preflight_automatic_request = True
            engine.threshold_tokens = 999_999
        returned = _compress(engine, rows, force=route == "leaf")
        assert (engine._last_compress_leaves or (None, 0))[1] == int(route == "leaf")
        assert bool(engine._last_stub_first_exit) == (route == "stub")
        actual, = [row for row in returned if row.get("message_uid") == tail["message_uid"]]
        assert actual is not cached and actual["content"] == ""
        assert actual.get("api_content") in {"", None}
        assert stored_rows(engine) == stored


def test_reasoning_only_threshold_pass_arms_hold(open_engine):
    messages = [_user("Question"), _assistant("<think>" + "x" * 1000 + "</think>" + "Answer " * 180)]
    with open_engine(fresh_tail_count=64) as engine:
        rough = tokens.count_messages_tokens(messages)
        engine.threshold_tokens = 100
        result = _compress(engine, messages)
        launches = engine._compress_calls
        assert result is messages
        assert (engine._last_compress_leaves or (None, 0))[1] == 0
        assert engine.get_status()["no_progress_hold"]["reason"] == "no_progress"
        if engine.should_compress(rough):
            _compress(engine, messages)
        assert engine._compress_calls == launches


def test_reorder_survives_zero_leaf_pass(open_engine):
    call, a = tool_pair("A", "real A")
    other_call, b = tool_pair("B", "real B")
    call["tool_calls"].extend(other_call["tool_calls"])
    messages = [_user("Parallel calls"), call, b, a]
    with open_engine(fresh_tail_count=64) as engine:
        _stamp(messages)
        engine.ingest(deepcopy(messages))
        engine._preflight_automatic_request = True
        result = _compress(engine, messages)
        assert (engine._last_compress_leaves or (None, 0))[1] == 0
        assert result[-3:] == [call, a, b]


def test_generated_head_keeps_cache_occurrence_on_hit(open_engine):
    summary = _user("[Recent Summary (d0, node 1)]\nEarlier work.\n[Expand for details: earlier work]")
    messages = [summary, _assistant("<think>Hidden analysis</think>Follow-up")]
    _stamp(messages)
    with open_engine() as engine:
        engine.ingest(messages)
        summary["timestamp"] = 42.0
        # A separate cleanup change makes the cache-hit result observable.
        engine._last_active_replay_messages[1]["content"] = "Follow-up"
        engine._preflight_automatic_request = True
        result = _compress(engine, messages)
        head = result[0]
        assert head["content"] == summary["content"]
        assert head is not summary and "timestamp" not in head
