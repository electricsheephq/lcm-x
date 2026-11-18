"""A foreground compaction exit must not adopt a successor's ingest state."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _engine, _rows, _stage, _turn

WAIT = 10
STOP = "LCM compaction stopped because the session binding changed"

FIRST_ROWS = {
    "user-stamped": {"role": "user", "content": "Session B first row", "timestamp": 9000.0, "message_uid": "b-0"},
    "user-unstamped": {"role": "user", "content": "Session B first row"},
    "assistant": {"role": "assistant", "content": "Session B first row", "message_uid": "b-0"},
}


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
    engine = _engine(tmp_path)
    engine._hermes_home = engine._store._hermes_home = str(tmp_path)
    at_exit, resume = threading.Event(), threading.Event()

    def hold():
        if not control and not at_exit.is_set():
            at_exit.set()
            assert resume.wait(WAIT), "compaction did not resume"

    def summarize(**kwargs):
        if where in ("summary", "lock-order"):
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
            rows = _view(1 if where == "no-leaf" else 7)
            engine.ingest(rows)
            if where in ("summary", "fallback", "legacy-finish", "lock-order"):
                store_ids = [row["store_id"] for row in _rows(engine)]
                _stage(engine, 0, store_ids[:8])
                engine._last_compacted_store_id = store_ids[7]
                for row in rows:
                    engine._store.append("S", row, conversation_id="conv")
            a_before = _stored(engine, "S")
            nodes_before = engine._dag.get_session_node_count("S")
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
                returned = engine.compress(caller_view, 24_000)
                _bind(engine, "B", "convB")
                b_at_rebind = _binding_state(engine)
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    pass_a = pool.submit(engine.compress, caller_view, 24_000)
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
                            pool.submit(_bind, engine, "B", "convB").result(timeout=WAIT)
                            b_at_rebind = _binding_state(engine)
                    finally:
                        resume.set()
                    try:
                        returned = pass_a.result(timeout=WAIT)
                    except RuntimeError as exc:
                        error = str(exc)
                        returned = None

            b_after = _binding_state(engine)
            if anchor == "off":
                patch.setenv("LCM_IDENTITY_ANCHOR", "off")
            b_input = [dict(FIRST_ROWS[first])]
            engine.ingest(b_input)
            if grow:
                b_input = b_input + [
                    {"role": "user" if i % 2 else "assistant", "content": f"Session B row {i}", "timestamp": 9100.0 + i}
                    for i in range(1, 9)
                ]
                engine.ingest(b_input)
            b_stored = _stored(engine, "B")
            patch.delenv("LCM_IDENTITY_ANCHOR", raising=False)
            _bind(engine, "S", "conv")
            engine.ingest(rows)
            a_after = _stored(engine, "S")

            changes = [key for key in b_at_rebind if b_at_rebind[key] != b_after[key]]
            return {
                "b_expected": [(row["role"], row["content"]) for row in b_input],
                "b_stored": [(role, content) for role, content, *_ in b_stored],
                "cursor": (b_at_rebind["frontier"].cursor, b_after["frontier"].cursor),
                "b_changes": changes,
                "returned_original": returned is caller_view,
                "error": error,
                "a_before": a_before,
                "a_after": a_after,
            }
        finally:
            engine.shutdown()


def _assert_rebound(tmp_path, monkeypatch, caplog, where, first="user-unstamped", **options):
    caplog.set_level(logging.INFO, logger="hermes_lcm.compaction")
    raced = _exercise(tmp_path / "rebound", monkeypatch, where, first, **options)
    control = _exercise(tmp_path / "control", monkeypatch, where, first, control=True, **options)
    facts = {key: value for key, value in raced.items() if not key.startswith("a_")}
    assert raced["b_stored"] == raced["b_expected"] and raced["cursor"] == (0, 0), facts
    assert raced["b_changes"] == [] and raced["returned_original"] and raced["error"] is None, facts
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
    control = _exercise(tmp_path / "control", monkeypatch, "summary", "user-unstamped", control=True)
    assert control["b_stored"] == control["b_expected"]
    assert control["a_before"] == control["a_after"]
    assert not any(record.message == STOP for record in caplog.records)
