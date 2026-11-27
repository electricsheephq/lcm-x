"""Host replay around compaction and cleanup preserves durable rows (I1, I2, I11)."""

import logging
from copy import deepcopy

import pytest

from hermes_lcm import host_uid_emit
from hermes_lcm.engine import _summary_part_text
from hermes_lcm.tokens import count_messages_tokens
from tests.invariants.test_head_copy_subset import head_copy_subset  # noqa: F401
from tests.invariants.test_hotfix_stored_summary_head import _append_bound_carrier
from tests.invariants.test_issue_3006_post_refusal import LINEAGE_STOP, _host_flush
from tests.test_issue_2854_no_summary_pass_unchanged import open_engine, published_head  # noqa: F401
from tests.test_issue_671_stub_first_exit import (
    STUB,
    THRESHOLD,
    TURN,
    exit_engine,
    make_engine,  # noqa: F401
    payload_of,
    run,
    stop_line,
    summaries,  # noqa: F401
    text_view,
    tool_content,
    tool_pair,
    tool_view,
)

pytestmark = pytest.mark.usefixtures("head_copy_subset")

RELATION = "generated_unproven"
NEW_TURN = [
    {"role": "user", "content": "Please compare the two accounts."},
    {"role": "assistant", "content": "Both accounts reconcile."},
]


@pytest.fixture(params=["unset", "on"])
def uid_mode(request, monkeypatch):
    if request.param == "on":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    else:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    # The test ContextEngine stub does not advertise the host metadata fields.
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    return request.param


def _build_host_twin(make_engine, caplog, *, with_copy):  # noqa: F811
    # Same seed and store writes as test_hotfix_stored_summary_head.twins.
    engine = exit_engine(make_engine, threshold_full_sweep_enabled=False, l3_truncate_tokens=1)
    engine.on_session_start("hf2", conversation_id="hf2", context_length=200_000)
    engine.threshold_tokens = THRESHOLD
    seed = text_view()[1:]
    seed[0]["content"] = "Earlier turns. " + seed[0]["content"]
    first = run(engine, seed, caplog)
    head_input = {**first[0], "content": first[0]["content"] + "\n\n---\n\n" + first[1]["content"]}
    supplied = tool_view(prefix=[head_input, *first[2:]])[1:]
    returned = run(engine, supplied, caplog)
    assert stop_line(caplog).group(1) == "stub_first_exit"
    assert len(returned) == len(supplied) and returned[0]["content"] != head_input["content"]
    head = returned[0]
    remainder = engine._generated_context_carrier_remainder(head)
    assert remainder and any(row["content"] == remainder for row in _session_rows(engine))

    flush = _host_flush()
    live = flush(returned + NEW_TURN)
    old_top = engine._store.max_store_id()
    engine.ingest(live)
    assert not any(
        row["content"] == head["content"]
        for row in engine._store.get_session_messages_after(engine._session_id, old_top)
    )
    copy_id = None
    if with_copy:
        from hermes_lcm.head_copy import is_known_head_copy

        copy_id = _append_bound_carrier(engine, head)
        assert is_known_head_copy(engine._store.connection, engine._store.get(copy_id))
    else:
        # Reserve the same global ID outside this session for the source comparison.
        engine._store.append_batch("unrelated", [NEW_TURN[0]], conversation_id="hf2")
    engine._config.large_output_active_replay_stubbing_enabled = False
    return engine, live, head["content"], copy_id, flush


def _session_rows(engine):
    return engine._store.get_session_messages(engine._session_id)


def _stored_rows(engine):
    return engine._store.connection.execute(
        "SELECT store_id, session_id, role, content, tool_call_id, tool_calls FROM messages ORDER BY store_id"
    ).fetchall()


def _host_pass(engine, live, original_head, copy_id, flush, caplog, pass_number):
    # Four ordinary turns keep the observed token count above threshold even after compaction.
    new_turns = [
        {"role": role, "content": f"Pass {pass_number}, turn {turn}, {role}." + TURN * 10}
        for turn in range(4)
        for role in ("user", "assistant")
    ]
    view = flush([
        {key: value for key, value in row.items() if key not in {"store_id", "session_id"}}
        for row in [*live, *new_turns]
    ])
    assert count_messages_tokens(view) >= engine.threshold_tokens
    replayed_head = view[0]["content"]
    assert view[0]["role"] == "user" and engine._verified_lcm_summary_prefix_end(replayed_head)
    if pass_number == 2:
        assert replayed_head != original_head
    before = _stored_rows(engine)
    if copy_id is not None:
        assert engine._store.get(copy_id)["content"] == original_head
    old_nodes = {node.node_id for node in engine._dag.get_session_nodes(engine._session_id)}
    engine.note_turn_complete()
    caplog.clear()
    result = run(engine, view, caplog)
    leaves = [
        node.source_ids
        for node in engine._dag.get_session_nodes(engine._session_id)
        if node.source_type == "messages" and node.node_id not in old_nodes
    ]
    context = f"pass={pass_number}, copy_id={copy_id}, rows={len(view)}, tokens={count_messages_tokens(view)}"
    assert LINEAGE_STOP not in engine.last_compression_noop_reason, context
    assert LINEAGE_STOP not in caplog.text, context
    assert "publication_invariant_conflict" not in caplog.text, context
    assert "Compaction publication" not in caplog.text, context
    assert leaves or engine.last_compression_noop_reason, (context, engine.last_compression_status)
    after = _stored_rows(engine)
    assert after[:len(before)] == before, context
    assert not any(role == "user" and content == replayed_head
                   for _, _, role, content, _, _ in after[len(before):]), context
    assert sum(row["content"] == replayed_head for row in result) <= 1, context
    return result, leaves


def test_host_shaped_compaction_matches_twin(uid_mode, make_engine, summaries, caplog):  # noqa: F811
    clean, clean_live, clean_head, _, clean_flush = _build_host_twin(make_engine, caplog, with_copy=False)
    copied, copied_live, copied_head, copy_id, copied_flush = _build_host_twin(make_engine, caplog, with_copy=True)
    assert clean_head == copied_head == clean_live[0]["content"] == copied_live[0]["content"]
    assert clean._uid_recognition_enabled() == copied._uid_recognition_enabled() == (uid_mode == "on")
    advanced = False
    for pass_number in (1, 2):
        clean_live, clean_leaves = _host_pass(clean, clean_live, clean_head, None, clean_flush, caplog, pass_number)
        copied_live, copied_leaves = _host_pass(
            copied, copied_live, copied_head, copy_id, copied_flush, caplog, pass_number,
        )
        assert copied_live == clean_live, f"pass={pass_number}, copy_id={copy_id}"
        copy_covered = any(copy_id in ids for ids in copied_leaves)
        logging.getLogger("hermes_lcm").info(
            "Host twin pass %d: leaves clean=%d copied=%d, copy %d covered=%s",
            pass_number, len(clean_leaves), len(copied_leaves), copy_id, copy_covered,
        )
        if copy_covered:
            # v0.30.2 can cover the surplus copy in a leaf. All other sources must match.
            assert [[source for source in ids if source != copy_id] for ids in copied_leaves] == clean_leaves
        else:
            assert copied_leaves == clean_leaves
        advanced |= bool(clean_leaves and copied_leaves)
    assert advanced, "neither threshold pass stored a leaf"


def test_ingest_after_same_length_cleanup(uid_mode, open_engine, summaries, caplog):  # noqa: F811
    engine = open_engine(survival_fit=False)
    flush = _host_flush()
    initial = published_head(engine)
    assert engine._uid_recognition_enabled() == (uid_mode == "on")
    nodes = engine._dag.get_session_nodes(engine._session_id)
    node, = nodes
    summary = _summary_part_text(node)
    assert summary in initial[1]["content"]
    summary_calls = list(summaries)
    call, result = tool_pair("aged-cleanup", payload_of(3_000, "aged"))
    call["content"] = ""
    view = flush([
        *initial,
        {"role": "user", "content": "Look up the earlier account."},
        call, result,
        {"role": "assistant", "content": "I found it."},
        *NEW_TURN,
    ])
    assert count_messages_tokens(view) < engine.threshold_tokens
    output = run(engine, view, caplog)
    assert engine.last_compression_status == "sanitized"
    assert not engine._last_stub_first_exit
    assert stop_line(caplog).group(2) == "0"
    assert engine._dag.get_session_nodes(engine._session_id) == nodes
    assert summaries == summary_calls
    assert len(output) == len(view) and output != view
    assert tool_content(output, "aged-cleanup").startswith(STUB)
    assert summary in output[1]["content"]
    assert "timestamp" in view[1] and "timestamp" not in output[1]

    before = _session_rows(engine)
    flushed = flush(output)
    next_turn = [
        {"role": "user", "content": "Please file the account comparison."},
        {"role": "assistant", "content": "The comparison is filed."},
    ]
    live = flush([*flushed, *next_turn])
    engine.ingest(live)
    after = _session_rows(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert not any(engine._verified_lcm_summary_prefix_end(row["content"]) for row in added)
    assert not engine._store.get_message_relations([row["store_id"] for row in added], RELATION)
    assert [(row["role"], row["content"]) for row in added] == [(row["role"], row["content"]) for row in next_turn]
    engine.ingest(deepcopy(live))
    assert _session_rows(engine) == after
