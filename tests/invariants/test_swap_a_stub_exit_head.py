"""A stub-only commit preserves the canonical head through replay (I1, I2)."""

import json
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy

import pytest

from hermes_lcm import host_uid_emit
from hermes_lcm.tokens import count_messages_tokens
from tests.behaviour.test_ingest_trace import reopen
from tests.harness import pin_char_counter
from tests.invariants.test_hotfix_host_shaped_passes import _stored_rows
from tests.invariants.test_issue_3006_post_refusal import _host_flush
from tests.test_compression_boundary import _host_merge_consecutive_users
from tests.test_issue_671_stub_first_exit import (
    STUB,
    exit_engine,
    make_engine,  # noqa: F401
    run,
    stop_line,
    summaries,  # noqa: F401
    text_view,
    tool_pair,
    tool_view,
)
from tests.test_refused_compaction_rollback import _drain_session_end_maintenance


class KnownStubPairCopy(AssertionError):
    """Cold bind stored exactly the five known stub pairs again."""


@pytest.fixture
def split_head_stub_exit(uid_mode, make_engine, summaries, monkeypatch, caplog):  # noqa: F811
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    pin_char_counter(monkeypatch)
    engine = exit_engine(make_engine, threshold_full_sweep_enabled=False, l3_truncate_tokens=1)
    assert engine._uid_recognition_enabled() == (uid_mode == "on")
    flush = _host_flush()

    def stamp(rows):
        stamped = flush(rows)
        if uid_mode == "on":
            stamped = [{**row, "message_uid": row.get("message_uid", f"{row['_row_id']:032x}")}
                       for row in stamped]
        return stamped

    first = run(engine, stamp(text_view()[1:]), caplog)
    assert stop_line(caplog).group(2) == "1"
    head, next_user = first[:2]
    assert head["role"] == next_user["role"] == "user"
    assert engine._is_verified_replay_scaffold_message(head)
    supplied = stamp([*first, *tool_view()[1:]])
    engine.ingest(supplied)
    engine.note_turn_complete()
    before = _stored_rows(engine)
    nodes = engine._dag.get_session_nodes(engine._session_id)
    supplied_before = deepcopy(supplied)  # the host's rows as sent, before any in-place change by the pass
    returned = run(engine, supplied, caplog)
    assert supplied == supplied_before, "the stub-only exit changed the host's rows in place"
    assert stop_line(caplog).group(1, 2) == ("stub_first_exit", "0")
    assert engine._dag.get_session_nodes(engine._session_id) == nodes
    assert _stored_rows(engine) == before, "stub-only pass copied or lost a durable row"
    assert engine._ingest_cursor == len(returned)
    assert engine._load_generated_ignored_placeholder_hash_counts() == (
        engine._generated_placeholder_digest_budget_for_active_replay(returned)
    )
    assert engine._load_generated_ignored_placeholder_hash_ordinals() == (
        engine._generated_placeholder_digest_ordinals_for_active_replay(returned)
    )
    status = engine._last_stub_first_exit
    assert status["tokens_after"] == count_messages_tokens(returned) <= status["target_tokens"]
    return engine, stamp, supplied, returned, before


@pytest.fixture
def replayed_stub_exit(bind, split_head_stub_exit, tmp_path):
    engine, stamp, supplied, returned, before = split_head_stub_exit

    new_rows = [{"role": "user", "content": "Check the final account."},
                *tool_pair("next-account", "final account result"),
                {"role": "assistant", "content": "The final account balances."}]
    with ExitStack() as engines:
        if bind == "cold":
            engine = reopen(engine, tmp_path, engines)
        live = stamp([*returned, *new_rows])
        engine.ingest(live)
        stored = _stored_rows(engine)
        assert stored[:len(before)] == before, "replay changed or lost a durable row"
        appended = stored[len(before):]
        assert [row[2:5] for row in appended[-len(new_rows):]] == [
            (row["role"], row["content"], row.get("tool_call_id")) for row in new_rows
        ], "replay lost, changed, or reordered the new rows"
        engine.ingest(deepcopy(live))
        assert _stored_rows(engine) == stored, "retry copied or lost a durable row"

    return supplied, returned, appended[:-len(new_rows)]


@pytest.mark.parametrize("uid_mode", ["off"])
def test_split_stub_exit_survives_rotation_with_host_merge(uid_mode, split_head_stub_exit):
    engine, stamp, supplied, returned, before = split_head_stub_exit
    parent, child = engine._session_id, "split-head-rotation-child"
    _drain_session_end_maintenance(engine)
    engine.on_session_end(parent, supplied)
    assert engine._compress_commit_proof["end_consumed"]
    engine.on_session_start(child, boundary_reason="compression", old_session_id=parent,
                            platform="cli", conversation_id=engine._conversation_id, context_length=200_000)
    assert engine._session_id == child
    assert _stored_rows(engine) == before, "rotation copied, changed, or lost a durable row"

    turn = [{"role": "user", "content": "New request after rotation."},
            {"role": "assistant", "content": "Reply after rotation."}]
    live = stamp(_host_merge_consecutive_users([*returned, *turn]))
    summary, retained = supplied[:2]
    assert live[0]["content"] == summary["content"] + "\n\n" + retained["content"]
    engine.ingest(live)

    after = _stored_rows(engine)
    assert after[:len(before)] == before, "replay changed or lost a durable row"
    assert not any(role == "user" and summary["content"] in content
                   for _, _, role, content, _, _ in after), "replay stored summary text as a user row"
    assert [row[1:5] for row in after[len(before):]] == [
        (child, row["role"], row["content"], None) for row in turn
    ], "replay copied a row or failed to store the new turn once in the child"
    engine.ingest(deepcopy(live))
    assert _stored_rows(engine) == after, "retry copied or lost a durable row"
    assert returned[:2] == supplied[:2], "stub-only exit rewrote the split head before rotation"


@pytest.mark.parametrize("uid_mode", ["off", "on"])
@pytest.mark.parametrize("bind", ["warm", "cold"])
def test_canonical_stub_exit_preserves_head_on_replay(bind, uid_mode, replayed_stub_exit):
    supplied, returned, extras = replayed_stub_exit
    if bind == "warm":
        assert not extras, "warm replay copied a row"
    else:
        summary = supplied[0]["content"]
        assert not any(
            role == "user" and summary in content for _, _, role, content, _, _ in extras
        ), "cold replay stored the summary or a folded carrier"

    assert len(returned) == len(supplied)
    expected = deepcopy(supplied)
    stubs = 0
    for index, (original, output) in enumerate(zip(supplied, returned)):
        if original.get("role") == "tool" and output.get("content", "").startswith(STUB):
            expected[index]["content"] = output["content"]
            stubs += 1
    assert stubs > 0
    assert returned == expected


@pytest.mark.parametrize("uid_mode", ["off", "on"])
@pytest.mark.parametrize("bind", ["cold"])
@pytest.mark.xfail(
    strict=True,
    raises=KnownStubPairCopy,
    reason="pre-existing cold-bind stub-pair copy (stub tier vs stored originals), tracked separately by the release manager",
)
def test_cold_stub_exit_copies_only_known_stub_pairs(bind, uid_mode, replayed_stub_exit):
    _, _, extras = replayed_stub_exit
    if not extras:
        return  # A repair must remove the strict xfail.

    observed = Counter()
    for _, _, role, content, tool_call_id, tool_calls in extras:
        if role == "assistant":
            call_ids = tuple(call["id"] for call in json.loads(tool_calls or "[]"))
        elif role == "tool" and content.startswith(STUB):
            call_ids = (tool_call_id,)
        else:
            pytest.fail(f"unexpected cold replay extra row with role {role} and tool call {tool_call_id}")
        observed[role, call_ids] += 1

    expected = Counter(
        (role, (f"call-{index}",)) for index in range(5) for role in ("assistant", "tool")
    )
    if observed != expected:
        pytest.fail(f"unexpected cold replay stub pairs: {observed}")
    raise KnownStubPairCopy("cold replay stored five assistant calls and their five tool stubs")
