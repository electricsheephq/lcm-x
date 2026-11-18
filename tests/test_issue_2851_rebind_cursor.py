"""A foreground compaction exit must not adopt a successor's ingest state."""

import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from hermes_lcm.tokens import count_messages_tokens
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _engine, _rows, _stage, _turn
from tests.test_issue_671_stub_first_exit import THRESHOLD, tool_view

WAIT = 10
STOP = "LCM compaction stopped because the session binding changed"

FIRST_ROWS = {
    "user-stamped": {"role": "user", "content": "Session B first row", "timestamp": 9000.0, "message_uid": "b-0"},
    "user-unstamped": {"role": "user", "content": "Session B first row"},
    "assistant": {"role": "assistant", "content": "Session B first row", "message_uid": "b-0"},
}

CURSOR_WRITERS = {
    "leaf-gap": "_finish_with_leaf",
    "no-leaf-gap": "_finish_without_leaf",
    "stub-first-gap": "_stub_first_exit",  # hotfix tree: no _finalize_stub_first_exit split; the cursor write is here
    "nonempty-gap": "_finish_with_leaf",
    "reset-gap": "_finish_with_leaf",
}


class _LeakedBindingState(AssertionError):
    """Only the cross-binding write should satisfy a strict xfail."""


def _view(turns):
    rows = [row for i in range(turns) for row in _turn(f"turn-{i}", 1000.0 + i * 10, tool=True)]
    for i, row in enumerate(rows):
        row.update(message_uid=f"uid-{i}", timestamp=1000.0 + i * 2)
        if row["role"] == "tool":
            row["content"] += " durable output" * 24
    return rows


def _stored(engine, session):
    rows = engine._store.get_session_messages(session, limit=1000)
    return [(row["role"], row["content"], row.get("tool_call_id"), row.get("tool_calls")) for row in rows]


def _bind(engine, session, conversation):
    engine.on_session_start(session, platform="telegram", context_length=200_000, conversation_id=conversation)


def _binding_state(engine):
    fields = (
        "_compress_host_rows_consumed", "_compress_hidden_rows_consumed", "_last_compress_leaves",
        "_last_compression_status", "_last_compression_noop_reason", "_last_survival_fit",
        "_compress_commit_reason", "_compress_commit_publication", "_survival_fit_emission", "_emission_handle",
        "_last_threshold_full_sweep", "_last_stub_first_exit", "_no_progress_hold", "_zero_leaf_streaks",
        "_pressure_yield_blocked_streak", "_pressure_yield_invocation_verdict", "_sweep_budget_hold_until",
        "_compression_boundary_active_placeholder_digest_budget", "_compression_boundary_active_placeholder_digest_ordinals",
    )
    return {
        **{field: deepcopy(getattr(engine, field)) for field in fields},
        "frontier": deepcopy(engine._frontier),
        "lifecycle": engine._lifecycle.get_by_conversation("convB"),
        "metadata": [tuple(row) for row in engine._store.connection.execute("SELECT * FROM metadata ORDER BY key")],
        "B rows": _stored(engine, "B"),
        "B nodes": engine._dag.get_session_nodes("B"),
    }


def _exercise(tmp_path, monkeypatch, where, first, *, anchor="default", grow=False, control=False, uid_mode=None):
    tmp_path.mkdir()
    _state_db(tmp_path, [("S", None, None), ("B", None, None)])
    config = {}
    if where == "condensation-gap":
        config.update(condensation_fanin=2, temporal_rollups_enabled=False, embeddings_enabled=False)
    if where == "fit-gap":
        config["context_length"] = 3_000
    if where == "stub-first-gap":
        config.update(
            leaf_chunk_tokens=1_000, fresh_tail_pressure_yield_enabled=False,
            deferred_maintenance_enabled=True, large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=1_000_000,
            large_output_active_replay_stubbing_enabled=True,
        )
    engine = _engine(tmp_path, **config)
    engine._hermes_home = engine._store._hermes_home = str(tmp_path)
    at_exit, resume = threading.Event(), threading.Event()
    reset = where == "reset-gap"
    successor = "S" if reset else "B"
    b_prefix = []
    if where == "nonempty-gap":
        b_prefix = [
            {"role": "user", "content": "Already stored B ask", "timestamp": 8100.0, "message_uid": "b-prefix-0"},
            {"role": "assistant", "content": "Already stored B reply", "timestamp": 8200.0, "message_uid": "b-prefix-1"},
        ]

    def bind_successor():
        if reset:
            epoch = engine._pressure_yield_reset_epoch
            engine.on_session_reset()
            assert (engine._session_id, engine._conversation_id) == ("S", "conv")
            assert engine._pressure_yield_reset_epoch != epoch
            return
        _bind(engine, "B", "convB")
        if b_prefix:
            assert len(_stored(engine, "B")) == len(b_prefix)
            engine.ingest(b_prefix)
            assert engine._ingest_cursor == len(b_prefix)

    def hold():
        if not control and not at_exit.is_set():
            at_exit.set()
            assert resume.wait(WAIT), "compaction did not resume"

    def summarize(**kwargs):
        if where in ("summary", "lock-order", "bind-back"):
            hold()
        return "Earlier turns.\nExpand for details about: turns", 1

    with monkeypatch.context() as patch:
        if uid_mode is None:
            patch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
        else:
            patch.setenv("LCM_HOST_MESSAGE_UID", uid_mode)
        patch.delenv("LCM_IDENTITY_ANCHOR", raising=False)
        patch.setattr(lcm_engine, "summarize_with_escalation", summarize)
        try:
            if b_prefix:
                _bind(engine, "B", "convB")
                engine.ingest(b_prefix)
                _bind(engine, "S", "conv")
            rows = _view(1 if where in ("no-leaf", "no-leaf-gap") else 7)
            current_tokens = 6_000 if where == "fit-gap" else 24_000
            if where == "stub-first-gap":
                rows = tool_view()
                engine.threshold_tokens = THRESHOLD
                current_tokens = count_messages_tokens(rows)
                assert current_tokens >= THRESHOLD
            else:
                engine.ingest(rows)
            if where in ("summary", "fallback", "legacy-finish", "lock-order", "condensation-gap",
                         "fit-gap", "leaf-gap", "nonempty-gap", "reset-gap", "bind-back"):
                store_ids = [row["store_id"] for row in _rows(engine)]
                _stage(engine, 0, store_ids[:8])
                engine._last_compacted_store_id = store_ids[7]
                for row in rows:
                    engine._store.append("S", row, conversation_id="conv")
            a_before = _stored(engine, "S")
            nodes_before = engine._dag.get_session_node_count("S")
            fit_lengths = []
            if where == "fit-gap":
                apply_fit = engine._apply_survival_fit

                def at_fit_adoption(result, plan, *args, **kwargs):
                    assert kwargs["persisted"] and not kwargs["after_exception"]
                    hold()
                    fitted = apply_fit(result, plan, *args, **kwargs)
                    fit_lengths.append((len(result), len(fitted)))
                    return fitted

                patch.setattr(engine, "_apply_survival_fit", at_fit_adoption)
            if where == "no-leaf-gap":
                sanitize = engine._sanitize_active_context_messages

                def clean_at_no_leaf_finish(messages, **kwargs):
                    cleaned = sanitize(messages, **kwargs)
                    # Force the sanitation-only finish to adopt a shorter view.
                    if sys._getframe(1).f_code.co_name == "_finish_without_leaf":
                        return cleaned[:-1]
                    return cleaned

                patch.setattr(engine, "_sanitize_active_context_messages", clean_at_no_leaf_finish)
            cursor_write = CURSOR_WRITERS.get(where)
            if cursor_write:
                cursor = type(engine)._ingest_cursor

                def before_cursor_write(owner, value):
                    # Interpose after the binding check and before its cursor assignment.
                    if owner is engine and sys._getframe(1).f_code.co_name == cursor_write:
                        hold()
                    cursor.fset(owner, value)

                patch.setattr(type(engine), "_ingest_cursor", cursor.setter(before_cursor_write))
            if where == "condensation-gap":
                source_time_window = engine._dag.get_source_time_window

                def before_condensed_node(*args, **kwargs):
                    # The model has returned and the binding check has passed.
                    # The next steps construct and publish the condensed node.
                    if sys._getframe(1).f_code.co_name == "_condense_summary_nodes":
                        hold()
                    return source_time_window(*args, **kwargs)

                patch.setattr(engine._dag, "get_source_time_window", before_condensed_node)
            if where == "fallback":
                def publication_conflict(*args, **kwargs):
                    raise LifecyclePublicationConflictError("synthetic publication conflict")

                patch.setattr(engine, "_publish_leaf_node", publication_conflict)

            method = {
                "fallback": "_fail_open_after_publication_failure",
                "published": "_advance_after_leaf",
                "leaf-finish": "_finish_with_leaf",
                "no-leaf": "_finish_without_leaf",
                "outer-exit": "_compress_impl",
                "exception": "_ingest_compression_view",
                "legacy-finish": "_note_fresh_tail_pressure_relieved",
            }.get(where)
            if method:
                original = getattr(engine, method)

                def at_checkpoint(*args, **kwargs):
                    if where in ("outer-exit", "exception"):
                        result = original(*args, **kwargs)
                        hold()
                        if where == "exception" and not control:
                            raise RuntimeError("synthetic compaction failure")
                        return result
                    if where == "no-leaf":
                        assert not args[0].leaf_compacted_this_turn
                    if where in ("published", "leaf-finish"):
                        assert engine._dag.get_session_node_count("S") > nodes_before
                    hold()
                    return original(*args, **kwargs)

                patch.setattr(engine, method, at_checkpoint)

            error = None
            caller_view = list(rows)
            if control:
                returned = engine.compress(caller_view, current_tokens)
                bind_successor()
                b_at_rebind = _binding_state(engine)
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    pass_a = pool.submit(engine.compress, caller_view, current_tokens)
                    try:
                        assert at_exit.wait(WAIT), f"compaction never reached {where}"
                        if where == "lock-order":
                            # Keep the stable-use lock held as shutdown does while waiting
                            # for an in-flight compaction to leave the gate.
                            with engine._exclusive_lifecycle("rebind"):
                                engine._on_session_start_unlocked(
                                    "B", platform="telegram", context_length=200_000, conversation_id="convB"
                                )
                                with engine._compaction_gate:
                                    assert engine._compaction_inflight == 1
                                b_at_rebind = _binding_state(engine)
                                resume.set()
                                returned = pass_a.result(timeout=WAIT)
                        else:
                            pool.submit(bind_successor).result(timeout=WAIT)
                            b_at_rebind = _binding_state(engine)
                    finally:
                        resume.set()
                    try:
                        returned = pass_a.result(timeout=WAIT)
                    except RuntimeError as exc:
                        error = str(exc)
                        returned = None

            b_after = _binding_state(engine)
            a_before_successor_ingest = _stored(engine, "S")
            a_reconcile_before_schedule = []
            a_reconcile_at_rebind = None
            if where == "bind-back":
                schedule = engine._schedule_ingest_cursor_reconciliation

                def observe_old_session_reconcile():
                    if engine._session_id == "S":
                        a_reconcile_before_schedule.append(engine._ingest_cursor_needs_reconcile)
                    schedule()

                patch.setattr(engine, "_schedule_ingest_cursor_reconciliation", observe_old_session_reconcile)
                _bind(engine, "S", "conv")
                a_reconcile_at_rebind = engine._ingest_cursor_needs_reconcile
                engine.ingest(rows)
                a_after = _stored(engine, "S")
                _bind(engine, "B", "convB")
            if anchor == "off":
                patch.setenv("LCM_IDENTITY_ANCHOR", "off")
            b_input = [*b_prefix, dict(FIRST_ROWS[first])]
            if grow:
                if where not in CURSOR_WRITERS and where != "fit-gap":
                    engine.ingest(b_input)
                b_input = b_input + [
                    {"role": "user" if i % 2 else "assistant", "content": f"Session B row {i}", "timestamp": 9100.0 + i}
                    for i in range(1, 9)
                ]
            engine.ingest(b_input)
            b_ingested_cursor = engine._ingest_cursor
            b_stored = _stored(engine, successor)
            a_nodes = engine._dag.get_session_nodes("S")
            b_nodes = engine._dag.get_session_nodes("B")
            patch.delenv("LCM_IDENTITY_ANCHOR", raising=False)
            if reset:
                a_after = b_stored[:len(a_before)]
                b_stored = b_stored[len(a_before):]
            elif where != "bind-back":
                _bind(engine, "S", "conv")
                engine.ingest(rows)
                a_after = _stored(engine, "S")

            changes = [key for key in b_at_rebind if b_at_rebind[key] != b_after[key]]
            bound_metadata = dict(b_at_rebind["metadata"])
            exit_metadata = dict(b_after["metadata"])
            return {
                "b_expected": [(row["role"], row["content"]) for row in b_input],
                "b_stored": [(role, content) for role, content, *_ in b_stored],
                "cursor": (b_at_rebind["frontier"].cursor, b_after["frontier"].cursor),
                "b_ingested_cursor": b_ingested_cursor,
                "b_changes": changes,
                "b_changed_metadata": {
                    key for key in bound_metadata.keys() | exit_metadata.keys()
                    if bound_metadata.get(key) != exit_metadata.get(key)
                },
                "b_frontier_at_rebind": b_at_rebind["frontier"],
                "b_frontier_after": b_after["frontier"],
                "b_nodes": [(node.node_id, node.source_type, node.source_ids) for node in b_nodes],
                "returned_original": returned is caller_view,
                "error": error,
                "a_before": a_before,
                "a_after": a_after,
                "a_before_successor_ingest": a_before_successor_ingest,
                "a_reconcile_before_schedule": a_reconcile_before_schedule,
                "a_reconcile_at_rebind": a_reconcile_at_rebind,
                "fit_lengths": fit_lengths,
                "a_leaf_ids": [node.node_id for node in a_nodes if node.source_type == "messages"],
                "a_has_condensation": any(node.source_type == "nodes" for node in a_nodes),
            }
        finally:
            engine.shutdown()


def _assert_rebound(tmp_path, monkeypatch, caplog, where, first="user-unstamped", **options):
    caplog.set_level(logging.INFO, logger="hermes_lcm.compaction")
    raced = _exercise(tmp_path / "rebound", monkeypatch, where, first, **options)
    control = _exercise(tmp_path / "control", monkeypatch, where, first, control=True, **options)
    facts = {key: value for key, value in raced.items() if not key.startswith("a_")}
    assert raced["b_stored"] == raced["b_expected"] and raced["cursor"] == (0, 0), facts
    assert raced["b_changes"] in ([], ["frontier"]), facts
    if raced["b_changes"] == ["frontier"]:
        assert raced["b_frontier_after"] == replace(raced["b_frontier_at_rebind"], needs_reconcile=True), facts
        assert raced["b_frontier_after"].needs_reconcile is True, facts
    assert raced["returned_original"] and raced["error"] is None, facts
    assert raced["a_before"] == raced["a_after"] == control["a_after"] == control["a_before"]
    assert [record.message for record in caplog.records if record.message == STOP] == [STOP]


@pytest.mark.parametrize("first", FIRST_ROWS)
@pytest.mark.parametrize("anchor", ["default", "off"])
def test_summary_call(tmp_path, monkeypatch, caplog, first, anchor):
    _assert_rebound(tmp_path, monkeypatch, caplog, "summary", first, anchor=anchor)


@pytest.mark.parametrize("uid_mode", [None, "off", "on"], ids=["shadow", "off", "on"])
def test_nine_row_view(tmp_path, monkeypatch, caplog, uid_mode):
    _assert_rebound(tmp_path, monkeypatch, caplog, "summary", grow=True, uid_mode=uid_mode)


@pytest.mark.parametrize("where", ["fallback", "published", "leaf-finish", "no-leaf", "outer-exit", "exception"])
def test_rebind_at_exit(tmp_path, monkeypatch, caplog, where):
    _assert_rebound(tmp_path, monkeypatch, caplog, where)


def test_release_while_stable_use_lock_is_held(tmp_path, monkeypatch, caplog):
    _assert_rebound(tmp_path, monkeypatch, caplog, "lock-order")


@pytest.mark.parametrize("first", FIRST_ROWS)
@pytest.mark.parametrize("anchor", ["default", "off"])
def test_legacy_finish_holds(tmp_path, monkeypatch, first, anchor):
    raced = _exercise(tmp_path / "rebound", monkeypatch, "legacy-finish", first, anchor=anchor)
    assert raced["b_stored"] == raced["b_expected"]
    assert raced["a_before"] == raced["a_after"]


def test_rebind_after_compaction(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_lcm.compaction")  # STOP is logged at INFO
    control = _exercise(tmp_path / "control", monkeypatch, "summary", "user-unstamped", control=True)
    assert control["b_stored"] == control["b_expected"]
    assert control["a_before"] == control["a_after"]
    assert not any(record.message == STOP for record in caplog.records)


def test_rebind_after_condensation_check(tmp_path, monkeypatch):
    raced = _exercise(tmp_path / "rebound", monkeypatch, "condensation-gap", "user-unstamped", grow=True)
    control = _exercise(tmp_path / "control", monkeypatch, "condensation-gap", "user-unstamped", grow=True, control=True)
    assert control["a_has_condensation"]
    assert raced["a_after"] == control["a_after"]
    facts = {key: raced[key] for key in ("cursor", "b_expected", "b_stored", "b_nodes", "a_leaf_ids")}
    facts["needs_reconcile"] = raced["b_frontier_after"].needs_reconcile
    assert raced["b_stored"] == raced["b_expected"], facts
    assert raced["cursor"] == (0, 0), facts
    assert raced["b_ingested_cursor"] == len(raced["b_expected"]), facts
    assert facts["needs_reconcile"] is True, facts


@pytest.mark.issue(2865)
@pytest.mark.xfail(strict=True, raises=_LeakedBindingState, reason=(
    "issue 2865, item 1: a switch between condensation's binding check and its node insert; same on main"
))
def test_condensed_node_keeps_its_source_session(tmp_path, monkeypatch):
    raced = _exercise(tmp_path / "rebound", monkeypatch, "condensation-gap", "user-unstamped", grow=True)
    old_leaves = set(raced["a_leaf_ids"])
    foreign_nodes = [
        (node_id, source_ids)
        for node_id, source_type, source_ids in raced["b_nodes"]
        if source_type == "nodes" and old_leaves.intersection(source_ids)
    ]
    if foreign_nodes:
        raise _LeakedBindingState(foreign_nodes)


def _assert_gap_rows(tmp_path, monkeypatch, where, first="user-unstamped"):
    raced = _exercise(tmp_path / "rebound", monkeypatch, where, first, grow=True)
    control = _exercise(tmp_path / "control", monkeypatch, where, first, grow=True, control=True)
    assert raced["error"] is None, raced["error"]
    assert control["b_stored"] == control["b_expected"]
    assert raced["a_after"] == control["a_after"]
    assert raced["b_stored"] == raced["b_expected"], {
        key: raced[key] for key in ("cursor", "b_stored", "b_expected", "b_changes")
    }
    return raced, control


@pytest.mark.parametrize("first", FIRST_ROWS)
def test_rebind_inside_trimming_survival_fit(tmp_path, monkeypatch, first):
    raced, control = _assert_gap_rows(tmp_path, monkeypatch, "fit-gap", first)
    for result in (raced, control):
        [(before, after)] = result["fit_lengths"]
        assert after < before
    assert raced["b_frontier_after"].needs_reconcile is True


@pytest.mark.parametrize("where", CURSOR_WRITERS)
def test_rebind_before_cursor_write(tmp_path, monkeypatch, where):
    raced, control = _assert_gap_rows(tmp_path, monkeypatch, where)
    assert raced["cursor"][0] == control["cursor"][0] == (2 if where == "nonempty-gap" else 0)
    assert raced["cursor"][1] > raced["cursor"][0]
    assert raced["b_frontier_after"].needs_reconcile is True
    if where == "reset-gap":
        assert raced["returned_original"]
        assert raced["a_before"] == raced["a_before_successor_ingest"] == raced["a_after"]


@pytest.mark.issue(2865)
@pytest.mark.xfail(strict=True, raises=_LeakedBindingState, reason="issue 2865, item 2")
@pytest.mark.parametrize("where,first", [
    *(("fit-gap", first) for first in FIRST_ROWS),
    ("leaf-gap", "user-unstamped"),
    ("no-leaf-gap", "user-unstamped"),
    ("stub-first-gap", "user-unstamped"),
], ids=[*FIRST_ROWS, "leaf", "no-leaf", "stub-first"])
def test_cursor_gap_keeps_diagnostics_and_replay_metadata_in_their_binding(tmp_path, monkeypatch, where, first):
    raced = _exercise(tmp_path / "rebound", monkeypatch, where, first, grow=True)
    assert raced["error"] is None
    diagnostics = {
        "_last_survival_fit", "_last_compression_status", "_pressure_yield_invocation_verdict", "_last_stub_first_exit",
    }.intersection(raced["b_changes"])
    metadata = {
        "survival_fit:counter", "ignored_active_replay_placeholder_hash_counts:B",
        "ignored_active_replay_placeholder_hash_ordinals:B",
    }.intersection(raced["b_changed_metadata"])
    if diagnostics or metadata:
        raise _LeakedBindingState(f"diagnostics={sorted(diagnostics)}, metadata={sorted(metadata)}")


def test_bind_back_reestablishes_old_session_reconcile_state(tmp_path, monkeypatch):
    raced, control = _assert_gap_rows(tmp_path, monkeypatch, "bind-back")
    assert raced["b_frontier_after"].needs_reconcile is True
    assert raced["a_reconcile_before_schedule"] == control["a_reconcile_before_schedule"] == [False]
    assert raced["a_reconcile_at_rebind"] == control["a_reconcile_at_rebind"] is True


def test_shutdown_with_a_switch_after_the_pass_matches_main(tmp_path, monkeypatch, caplog):
    """Mode unset: the binding check precedes the shutdown check, so the switch wins, as on main.

    Expected values were computed on main with this cell's inputs; the shutdown stop would leave
    ("noop", "engine shutting down") in the session served after the switch."""
    caplog.set_level(logging.INFO, logger="hermes_lcm.compaction")
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    _state_db(tmp_path, [("S", None, None), ("B", None, None)])
    engine = _engine(tmp_path)
    engine._hermes_home = engine._store._hermes_home = str(tmp_path)
    rows = _view(7)
    try:
        engine.ingest(rows)
        original = engine._compress_impl
        with ThreadPoolExecutor(max_workers=1) as pool:
            def at_outer_exit(*args, **kwargs):
                result = original(*args, **kwargs)
                pool.submit(_bind, engine, "B", "convB").result(timeout=WAIT)
                with engine._compaction_gate:
                    engine._compaction_closing = True
                return result

            monkeypatch.setattr(engine, "_compress_impl", at_outer_exit)
            caller_view = list(rows)
            returned = engine.compress(caller_view, 24_000)
        with engine._compaction_gate:
            engine._compaction_closing = False
        assert returned is caller_view
        assert (engine._session_id, engine._last_compression_status, engine._last_compression_noop_reason,
                engine._ingest_cursor, engine._ingest_cursor_needs_reconcile) == ("B", "idle", "", 0, True)
        assert [record.message for record in caplog.records if record.message == STOP] == [STOP]
    finally:
        engine.shutdown()
