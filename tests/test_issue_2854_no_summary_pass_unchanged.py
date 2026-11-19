"""#2854: a no-work compression pass must leave the host replay unchanged."""

import json
import sqlite3
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_lcm.compaction as compaction
import hermes_lcm.engine as engine_module
import hermes_lcm.host_uid_emit as host_uid_emit
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig, host_message_uid_mode
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import _L3_TRUNCATION_MARKER
from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
from hermes_lcm.tokens import count_messages_tokens, count_tokens
from tests.test_ingest_protection import _broken_assistant_output
from tests.test_issue_667_level3_repair import _fake_route
from tests.test_issue_671_stub_first_exit import STUB, payload_of, tool_content, tool_pair, tool_view

PAD = " alpha beta gamma delta" * 30
SUMMARY = "Earlier turns.\nExpand for details about: turns"
pytestmark = pytest.mark.issue(2854)

_MODE_ON_NO_LEAF_GAPS = {
    "test_assembly_cap_trims_tail_behind_replayed_head[on]":
        "with uid recognition active a no-leaf pass never drops an uncovered row to meet the assembly cap",
}


@pytest.fixture(autouse=True)
def _mode_on_no_leaf_gap(request):
    gap = _MODE_ON_NO_LEAF_GAPS.get(request.node.name)
    if gap is not None:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=gap))


@pytest.fixture(params=["unset", "shadow", "on"])
def uid_mode(request, monkeypatch):
    mode = request.param
    if mode == "unset":
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    else:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    # The local ContextEngine stub has no message_metadata module.
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    assert host_message_uid_mode() == ("shadow" if mode == "unset" else mode)
    return mode


@pytest.fixture
def summaries(monkeypatch):
    calls = []

    def summarize(**kwargs):
        calls.append(kwargs["text"])
        return SUMMARY, 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    return calls


@pytest.fixture
def open_engine(tmp_path):
    engines = []

    def open_at(name="case", **overrides):
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        with sqlite3.connect(path / "state.db") as db:
            db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
                       "end_reason TEXT, source TEXT, model_config TEXT)")
            db.execute("INSERT OR IGNORE INTO sessions (id) VALUES ('S')")
        settings = dict(database_path=str(path / "lcm.db"), fresh_tail_count=2, leaf_chunk_tokens=400,
                        context_threshold=0.001, threshold_full_sweep_enabled=False,
                        embeddings_enabled=False, temporal_rollups_enabled=False)
        context_length = overrides.pop("context_length", 200_000)
        settings.update(overrides)
        engine = LCMEngine(config=LCMConfig(**settings), hermes_home=str(path))
        engine.on_session_start("S", platform="telegram", context_length=context_length, conversation_id="conv")
        engines.append(engine)
        return engine

    yield open_at
    for engine in reversed(engines):
        engine.shutdown()


def host_row(row, uid):
    row.update(message_uid=uid, display_metadata={"label": f"label-{uid}"})
    if row["role"] in ("user", "assistant"):
        row["api_content"] = row["content"] + f"\n[sent bytes for {uid}]"
    return row


def user_row(index):
    return host_row({"role": "user", "content": f"new user {index}", "timestamp": 5000.0 + index}, f"new-{index}")


def empty_row():
    return host_row({"role": "assistant", "content": ""}, "empty-assistant")


def host_view(with_api_only=False):
    rows = [{"role": "system", "content": "system prompt"}]
    for index in range(7):
        rows.append(host_row({"role": "user", "content": f"[T{index}] user{PAD}",
                              "timestamp": 1000.0 + index}, f"u{index}"))
        if index % 2 == 0:
            call, result = tool_pair(f"tool-{index}", f"tool result {index}{PAD}")
            call["content"] = ""
            rows.extend((host_row(call, f"c{index}"), host_row(result, f"t{index}")))
        if index == 2 and with_api_only:
            rows.append(empty_row())
        rows.append(host_row({"role": "assistant", "content": f"reply to T{index}{PAD}"}, f"a{index}"))
    return rows


def spend_budget(monkeypatch, engine):
    prepare = engine._prepare_retained_user_anchor
    monotonic = compaction.time.monotonic
    offset = 0

    def spend(*args):
        nonlocal offset
        result = prepare(*args)
        offset = 100_000
        return result

    monkeypatch.setattr(compaction.time, "monotonic", lambda: monotonic() + offset)
    monkeypatch.setattr(engine, "_prepare_retained_user_anchor", spend)


def close_route(engine):
    breaker = engine._summary_circuit_breaker
    for _ in range(breaker.failure_threshold):
        breaker.record_failure(engine._config.summary_model)
    assert not breaker.allows(engine._config.summary_model)


def assert_unchanged(engine, view, output, snapshot):
    assert output == snapshot
    assert output is view
    assert engine._ingest_cursor == len(view)


@pytest.mark.parametrize("exit_name", [
    "warm", "boundary-cleanup", "below-threshold-cleanup", "time-budget", "route-unavailable", "cold-bind", "hold",
])
@pytest.mark.parametrize("api_only", [False, True], ids=["plain", "api-only"])
def test_no_leaf_exit(open_engine, monkeypatch, uid_mode, summaries, exit_name, api_only):
    view = host_view(api_only)
    snapshot = deepcopy(view)
    options = {}
    if exit_name in {"warm", "cold-bind"}:
        options["fresh_tail_count"] = 64
    if exit_name in {"time-budget", "route-unavailable"}:
        options["threshold_full_sweep_enabled"] = True
    if exit_name == "route-unavailable":
        options.update(max_assembly_tokens=100_000, l3_truncate_tokens=2,
                       large_output_externalization_enabled=False, large_output_active_replay_stubbing_enabled=False)
    if exit_name == "below-threshold-cleanup":
        options.update(leaf_chunk_tokens=100, fresh_tail_pressure_yield_enabled=False,
                       deferred_maintenance_enabled=True, critical_budget_pressure_ratio=0.5)
    engine = open_engine(**options)
    observed = 24_000
    if exit_name == "below-threshold-cleanup":
        observed = count_messages_tokens(view)
        engine.threshold_tokens = 10 * observed
        engine.context_length = int(observed * 1.6)
        assert engine.should_compress_preflight(view)
        assert engine._compaction_request.below_threshold_cleanup_only
    else:
        engine.ingest(deepcopy(view))
    if exit_name == "cold-bind":
        engine.shutdown()
        engine = open_engine(**options)
        assert engine._compress_calls == 0
    elif exit_name == "boundary-cleanup":
        engine._compaction_request.cleanup_only_due_to_boundary_cooldown = True
    elif exit_name == "time-budget":
        spend_budget(monkeypatch, engine)
    elif exit_name == "route-unavailable":
        close_route(engine)
    elif exit_name == "hold":
        engine._no_progress_hold = (10**12, "no_progress")
        monkeypatch.setattr(engine, "_no_progress_hold_blocks", lambda *args, **kwargs: True)

    output = engine.compress(view, current_tokens=observed)

    assert view == snapshot
    assert not summaries
    assert engine._last_compress_leaves[1] == 0
    assert not engine._dag.get_session_nodes("S")
    if exit_name in {"time-budget", "route-unavailable"}:
        assert engine._last_threshold_full_sweep["stop_reason"] == {
            "time-budget": "time_budget_exhausted", "route-unavailable": "summary_route_unavailable",
        }[exit_name]
    assert_unchanged(engine, view, output, snapshot)


def published_head(engine):
    output = engine.compress(host_view(), current_tokens=24_000)
    assert engine._last_compress_leaves[1] > 0
    assert output[1]["content"].startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
    engine.note_turn_complete()
    engine._config.leaf_chunk_tokens = 50_000
    engine.threshold_tokens = 8_000
    return output


@pytest.mark.parametrize("above", [False, True], ids=["below", "above"])
@pytest.mark.parametrize("passes", [2, 3])
def test_head_stability(open_engine, uid_mode, summaries, above, passes):
    engine = open_engine(survival_fit=False)
    output = published_head(engine)
    head_bytes = json.dumps(output[:2], ensure_ascii=False).encode()
    nodes = engine._dag.get_session_node_count("S")
    calls = list(summaries)

    for index in range(passes):
        view = [*output, user_row(index)]
        snapshot = deepcopy(view)
        output = engine.compress(view, current_tokens=12_000 if above else 4_000)
        assert engine._last_compress_leaves[1] == 0
        assert json.dumps(output[:2], ensure_ascii=False).encode() == head_bytes
        assert_unchanged(engine, view, output, snapshot)
        engine.note_turn_complete()

    assert engine._dag.get_session_node_count("S") == nodes
    assert summaries == calls


def stored_rows(engine):
    return engine._store.get_session_messages("S", limit=100_000)


def stored_shape(rows):
    return [(row["role"], row["content"], row.get("tool_call_id"), row.get("tool_calls")) for row in rows]


def test_next_ingest_matches_control(open_engine, uid_mode, summaries):
    shapes = []
    for name in ("pass", "control"):
        engine = open_engine(name, survival_fit=False)
        view = [*published_head(engine), user_row(0)]
        if name == "pass":
            snapshot = deepcopy(view)
            output = engine.compress(view, current_tokens=4_000)
            assert_unchanged(engine, view, output, snapshot)
        else:
            engine.ingest(view)
        before = stored_rows(engine)
        call, result = tool_pair("next-tool", "next tool result")
        appended = [user_row(1), host_row(call, "next-call"), host_row(result, "next-result")]

        engine.ingest([*view, *appended])

        after = stored_rows(engine)
        assert after[:len(before)] == before
        assert [(row["role"], row["content"]) for row in after[len(before):]] == [
            (row["role"], row["content"]) for row in appended
        ]
        assert after[-1]["tool_call_id"] == "next-tool"
        shapes.append(stored_shape(after))
    assert shapes[0] == shapes[1]


def test_no_api_content_still_drops(open_engine, summaries):
    engine = open_engine(fresh_tail_count=64)
    view = host_view(True)
    empty = next(row for row in view if row.get("message_uid") == "empty-assistant")
    del empty["api_content"]
    output = engine.compress(view, current_tokens=24_000)
    assert output == [row for row in view if row is not empty]
    assert engine._ingest_cursor == len(output)
    assert not summaries


def test_publishing_pass_keeps_api_only_tail(open_engine, summaries):
    engine = open_engine(fresh_tail_count=3)
    row = empty_row()
    snapshot = deepcopy(row)
    view = [*host_view(), row, host_row({"role": "assistant", "content": "continued reply"}, "continuation")]
    output = engine.compress(view, current_tokens=24_000)
    assert engine._last_compress_leaves[1] > 0
    kept = next((kept for kept in output if kept.get("message_uid") == row["message_uid"]), None)
    assert kept == snapshot, [item.get("message_uid") for item in output]


def test_generated_list_keeps_api_only_row_and_direct_cleaners_keep_their_contract(open_engine):
    engine = open_engine()
    row = empty_row()
    generated = [row]
    output = engine._sanitize_active_context_messages([row, {"role": "assistant", "content": "next"}],
                                                      generated_rows=generated)
    assert engine_module._clean_active_assistant_message(row) is None
    assert engine._active_cleanup_replay_identity(("assistant", "", "", "", "")) is None
    assert output[0] is row
    assert generated[0] is row


def _compress_with_replay_observation(engine, view, uid_mode, tokens):
    recognition, decisions = [], []
    compress_impl = engine._compress_impl
    replay = engine._uid_trimmed_base_replay

    def observe_recognition(messages, *args, **kwargs):
        recognition.append(engine._uid_recognition_enabled() and any(row.get("message_uid") for row in messages))
        return compress_impl(messages, *args, **kwargs)

    def observe_replay(*args, **kwargs):
        decision = replay(*args, **kwargs)
        decisions.append(decision)
        return decision

    with patch.object(engine, "_compress_impl", observe_recognition), patch.object(
        engine, "_uid_trimmed_base_replay", observe_replay,
    ):
        output = engine.compress(view, current_tokens=tokens)

    assert recognition == [uid_mode == "on"]
    if uid_mode == "on":
        assert decisions and isinstance(decisions[0], tuple), "the pass did not record a trimmed base replay decision"
        assert engine._ingest_cursor == len(output)
    else:
        assert not decisions
    return output


def test_new_condensation_is_assembled(open_engine, uid_mode, summaries):
    engine = open_engine(survival_fit=False)
    previous = published_head(engine)
    leaves = engine._dag.get_session_nodes("S")
    marker = "Condensation added after the host received its head."
    engine._dag.add_node(SummaryNode(session_id="S", summary=marker, depth=1, source_type="nodes",
                                    source_ids=[node.node_id for node in leaves], token_count=15))
    view = [*previous, user_row(0)]
    output = _compress_with_replay_observation(engine, view, uid_mode, 4_000)
    assert engine._last_compress_leaves[1] == 0
    assert output != view and marker in output[1]["content"]
    assert output[2:] == view[2:]
    assert engine._ingest_cursor == len(output)


@pytest.mark.parametrize("repair", ["store-update", "doctor-command"])
def test_replayed_head_refreshes_after_in_place_repair(open_engine, monkeypatch, repair):
    engine = open_engine(survival_fit=False, dynamic_leaf_chunk_enabled=False)
    original = f"Truncated earlier turns.{_L3_TRUNCATION_MARKER}\nExpand for details about: old turns"
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: (original, 3))
    previous = published_head(engine)
    node, = engine._summary_frontier_nodes()
    old_part = engine_module._summary_part_text(node)
    assert old_part in previous[1]["content"]
    assert node.expand_hint == "old turns"

    if repair == "doctor-command":
        calls = _fake_route(monkeypatch)
        result = handle_lcm_command("doctor repair level3 apply", engine)
        assert "nodes_repaired: 1" in result
        assert len(calls) == 1
    else:
        text = "Recovered earlier turns.\nExpand for details about: recovered turns"
        with engine._dag.connection as conn:
            conn.execute("UPDATE summary_nodes SET summary = ?, token_count = ?, expand_hint = ? WHERE node_id = ?",
                         (text, count_tokens(text), "recovered turns", node.node_id))

    repaired, = engine._summary_frontier_nodes()
    assert (repaired.depth, repaired.node_id) == (node.depth, node.node_id)
    assert repaired.summary != node.summary and repaired.expand_hint != node.expand_hint
    view = [*previous, user_row(0)]
    output = engine.compress(view, current_tokens=4_000)
    assert engine._last_compress_leaves[1] == 0
    assert engine_module._summary_part_text(repaired) in output[1]["content"]
    assert old_part not in output[1]["content"]
    assert output[2:] == view[2:]


def test_replayed_head_does_not_count_a_quoted_header(open_engine, monkeypatch):
    engine = open_engine(survival_fit=False, dynamic_leaf_chunk_enabled=False)
    quoted_header = "[Recent Summary (d0, node 2)]"
    summary = f"A quoted header follows.\n\n---\n\n{quoted_header}\nQuotation ends here."
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: (summary, 1))
    previous = published_head(engine)
    first, = engine._summary_frontier_nodes()
    assert first.node_id == 1
    second_text = "Actual second frontier summary."
    uncondensed_sources = [row["store_id"] for row in stored_rows(engine)
                           if row["store_id"] not in first.source_ids]
    second = engine._dag.add_node(SummaryNode(session_id="S", summary=second_text, depth=0,
                                             source_ids=uncondensed_sources[:1], token_count=count_tokens(second_text)))
    second_part = engine_module._summary_part_text(engine._dag.get_node(second))
    assert second_part.startswith(quoted_header)
    assert quoted_header in previous[1]["content"] and second_part not in previous[1]["content"]

    view = [*previous, user_row(0)]
    output = engine.compress(view, current_tokens=4_000)
    assert engine._last_compress_leaves[1] == 0
    assert second_part in output[1]["content"]
    assert output[2:] == view[2:]


def test_replayed_head_keeps_a_summary_with_an_internal_separator(open_engine, monkeypatch):
    engine = open_engine(survival_fit=False, dynamic_leaf_chunk_enabled=False)
    summary = "Earlier turns.\n\n---\n\nFurther details."
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: (summary, 1))
    previous = published_head(engine)
    assert summary in previous[1]["content"]

    view = [*previous, user_row(0)]
    snapshot = deepcopy(view)
    output = engine.compress(view, current_tokens=4_000)
    assert engine._last_compress_leaves[1] == 0
    assert_unchanged(engine, view, output, snapshot)


def test_replayed_head_is_reassembled_for_proactive_recall(open_engine, monkeypatch, summaries):
    engine = open_engine(survival_fit=False, dynamic_leaf_chunk_enabled=False)
    previous = published_head(engine)
    engine._config.proactive_recall_enabled = True
    engine._config.embeddings_enabled = True
    engine._config.proactive_recall_provider = "fastembed"
    assert engine._proactive_recall_eligible()
    recall = "<relevant-memories>\nA synthetic memory.\n</relevant-memories>"
    monkeypatch.setattr(engine, "_build_proactive_recall_message",
                        lambda *args: {"role": "user", "content": recall})

    view = [*previous, user_row(0)]
    assert all(row["content"] != recall for row in view)
    output = engine.compress(view, current_tokens=4_000)
    assert engine._last_compress_leaves[1] == 0
    assert output is not view
    assert sum(row["content"] == recall for row in output) == 1


def test_redaction_is_still_returned(open_engine, summaries):
    engine = open_engine(fresh_tail_count=64, sensitive_patterns_enabled=True, sensitive_patterns=["api_key"])
    view = [{"role": "user", "content": "api_key=sk-synthetic-value"}, {"role": "assistant", "content": "ack"}]
    output = engine.compress(view)
    assert output != view and "sk-synthetic-value" not in output[0]["content"]
    assert output[1:] == view[1:] and not summaries


def test_quarantine_is_still_returned(open_engine, summaries):
    engine = open_engine(fresh_tail_count=64)
    view = [{"role": "user", "content": "question"}, {"role": "assistant", "content": _broken_assistant_output()}]
    output = engine.compress(view)
    assert output != view and "assistant output quarantined" in output[1]["content"]
    assert output[0] == view[0] and not summaries


def test_ignored_backlog_is_still_dropped(open_engine, summaries):
    engine = open_engine(fresh_tail_count=1, leaf_chunk_tokens=10_000, survival_fit=False)
    engine._compiled_ignore_message_patterns = [
        SimpleNamespace(search=lambda text, *, timeout: "IGNORE THIS" in text)
    ]
    view = [{"role": "user", "content": "IGNORE THIS"}, {"role": "assistant", "content": "hidden reply"},
            user_row(0)]
    output = engine.compress(view)
    assert output != view
    assert all("IGNORE THIS" not in row["content"] for row in output)
    assert output[-1] == view[-1] and not summaries


def test_stub_first_still_rewrites(open_engine, summaries):
    engine = open_engine(leaf_chunk_tokens=1000, fresh_tail_pressure_yield_enabled=False,
                         large_output_externalization_enabled=True,
                         large_output_externalization_threshold_chars=1_000_000,
                         large_output_active_replay_stubbing_enabled=True, deferred_maintenance_enabled=True)
    engine.threshold_tokens = 12_000
    view = tool_view()
    output = engine.compress(view, current_tokens=count_messages_tokens(view))
    assert engine._last_stub_first_exit and not summaries
    assert output != view and output[-2:] == view[-2:]
    assert all(row["content"].startswith(STUB) for row in output if row["role"] == "tool")


def test_survival_fit_still_cuts(open_engine, monkeypatch, summaries):
    engine = open_engine(fresh_tail_count=2)
    view = host_view()
    engine.ingest(view)
    ids = [row["store_id"] for row in stored_rows(engine)]
    engine._dag.add_node(SummaryNode(session_id="S", summary=SUMMARY, source_ids=ids[:-2]))
    engine._compaction_request.cleanup_only_due_to_boundary_cooldown = True
    monkeypatch.setattr(engine, "_survival_fit_budget", lambda *args, **kwargs: 700)
    output = engine.compress(view, current_tokens=count_messages_tokens(view))
    assert output != view and len(output) < len(view)
    assert engine._last_survival_fit and not summaries


def test_forced_overflow_still_recovers(open_engine, summaries):
    engine = open_engine(fresh_tail_count=64, max_assembly_tokens=70)
    engine.compression_count = 1
    view = [{"role": "system", "content": "system"}, user_row(0),
            {"role": "assistant", "content": "oversized" + PAD}]
    output = engine.compress(view, current_tokens=150)
    assert output != view and engine._last_compression_status == "overflow_recovery"
    assert engine._ingest_cursor == len(output) and not summaries


def test_committed_prefix_is_still_resumed(open_engine, summaries):
    engine = open_engine(survival_fit=False)
    view = host_view()
    first = engine.compress(view, current_tokens=24_000, force=True)
    calls = list(summaries)
    close_route(engine)
    output = engine.compress(deepcopy(view), current_tokens=24_000)
    assert output != view and output[1:] == first[1:]
    assert engine._last_compression_status == "compacted"
    assert engine._last_compress_leaves[1] == 0 and summaries == calls
    assert engine._ingest_cursor == len(output)


def test_quarantine_behind_replayed_head_is_reassembled(open_engine, uid_mode, summaries):
    engine = open_engine(survival_fit=False)
    view = [*published_head(engine), user_row(0),
            host_row({"role": "assistant", "content": _broken_assistant_output()}, "broken")]
    output = engine.compress(view, current_tokens=4_000)
    assert engine._last_compress_leaves[1] == 0
    assert output is not view
    assert "assistant output quarantined" in output[-1]["content"]
    assert output[-1]["content"] != view[-1]["content"]


@pytest.mark.parametrize("position", ["at-end", "before-user"])
def test_missing_result_behind_replayed_head_is_stubbed(open_engine, uid_mode, summaries, position):
    engine = open_engine(survival_fit=False)
    head = published_head(engine)
    call, _ = tool_pair("unanswered", "unused")
    call["content"] = ""
    view = [*head, user_row(0), host_row(call, "unanswered-call")]
    following = [user_row(1)] if position == "before-user" else []
    view.extend(following)
    snapshot = deepcopy(view)

    output = _compress_with_replay_observation(engine, view, uid_mode, 4_000)

    assert engine._last_compress_leaves[1] == 0
    assert output[len(head) + 1:] == [call, engine._missing_tool_result_stub("unanswered"), *following]
    assert output is not view
    assert view == snapshot


@pytest.mark.parametrize("result_count", [1, 2], ids=["one-result", "two-results"])
def test_aged_results_behind_replayed_head_are_stubbed(open_engine, uid_mode, summaries, result_count):
    engine = open_engine(survival_fit=False, threshold_full_sweep_enabled=False,
                         large_output_externalization_enabled=True,
                         large_output_active_replay_stubbing_enabled=True,
                         large_output_externalization_threshold_chars=1_000_000)
    view = list(published_head(engine))
    tool_ids = [f"aged-{index}" for index in range(result_count)]
    for index, tool_id in enumerate(tool_ids):
        call, result = tool_pair(tool_id, payload_of(3_000, f"word{index}"))
        call["content"] = ""
        reply = {"role": "assistant", "content": f"reply {index}"}
        view.extend((user_row(index), host_row(call, f"call-{index}"),
                     host_row(result, f"result-{index}"), host_row(reply, f"reply-{index}")))
    view.extend((user_row(result_count), {"role": "assistant", "content": "fresh reply"}))
    snapshot = deepcopy(view)

    output = _compress_with_replay_observation(engine, view, uid_mode, count_messages_tokens(view))

    assert engine._last_compress_leaves[1] == 0
    assert not engine._last_stub_first_exit
    assert [tool_content(output, tool_id).startswith(STUB) for tool_id in tool_ids] == [True] * result_count
    assert output is not view
    assert output[-2:] == view[-2:]
    assert view == snapshot


def test_assembly_cap_trims_tail_behind_replayed_head(open_engine, monkeypatch, uid_mode, summaries):
    engine = open_engine(survival_fit=False)
    head = published_head(engine)
    old_user = user_row(0)
    old_user["content"] += PAD * 10
    fresh = [user_row(1), {"role": "assistant", "content": "fresh reply"}]
    view = [*head, old_user, *fresh]
    snapshot = deepcopy(view)
    engine._config.max_assembly_tokens = 300
    assert count_messages_tokens(view) > engine._effective_assembly_token_cap()
    # An over-cap view normally takes the forced recovery exit. Exercise the ordinary tail selector here.
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **kwargs: False)

    output = engine.compress(view, current_tokens=count_messages_tokens(view))

    assert engine._last_compress_leaves[1] == 0
    assert old_user not in output and len(output) < len(view)
    assert count_messages_tokens(output) <= 300
    assert output is not view and output[-2:] == fresh
    assert view == snapshot


def test_paired_inline_result_behind_replayed_head_returns_input(open_engine, uid_mode, summaries):
    engine = open_engine(survival_fit=False, large_output_externalization_enabled=True,
                         large_output_active_replay_stubbing_enabled=True)
    head = published_head(engine)
    call, result = tool_pair("paired", "short result")
    call["content"] = ""
    view = [*head, user_row(0), host_row(call, "paired-call"), host_row(result, "paired-result"),
            user_row(1), {"role": "assistant", "content": "fresh reply"}]
    snapshot = deepcopy(view)

    output = engine.compress(view, current_tokens=count_messages_tokens(view))

    assert engine._last_compress_leaves[1] == 0
    assert_unchanged(engine, view, output, snapshot)


@pytest.mark.parametrize("cap_kind", ["reserve-floor", "explicit-max", "exact-boundary"])
def test_fitting_assembly_cap_behind_replayed_head_returns_input(
    open_engine, monkeypatch, uid_mode, summaries, cap_kind,
):
    options = {
        "reserve-floor": {"reserve_tokens_floor": 40_000},
        "explicit-max": {"max_assembly_tokens": 100_000},
        "exact-boundary": {},
    }
    engine = open_engine(survival_fit=False, context_length=1_000_000, **options[cap_kind])
    head = published_head(engine)
    head_bytes = json.dumps(head[:2], ensure_ascii=False).encode()
    view = [*head, user_row(0)]
    snapshot = deepcopy(view)
    tokens = count_messages_tokens(view)
    if cap_kind == "exact-boundary":
        engine._config.max_assembly_tokens = tokens
        # Forced recovery starts at equality. Isolate the ordinary selector, which keeps rows at equality.
        assert engine._should_force_overflow_recovery(observed_tokens=tokens, messages=view)
        monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **kwargs: False)
    assert engine._effective_assembly_token_cap() == {
        "reserve-floor": 960_000, "explicit-max": 100_000, "exact-boundary": tokens,
    }[cap_kind]
    assert tokens <= engine._effective_assembly_token_cap()
    calls = list(summaries)

    output = engine.compress(view, current_tokens=tokens)

    assert engine._last_compress_leaves[1] == 0 and summaries == calls
    assert_unchanged(engine, view, output, snapshot)
    assert json.dumps(output[:2], ensure_ascii=False).encode() == head_bytes
