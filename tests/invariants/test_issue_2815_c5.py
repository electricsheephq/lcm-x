"""Synthetic regressions for opt-in UID recognition and turn-path contention."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from threading import Event, get_ident

import pytest

from hermes_lcm import compaction, engine as engine_module
from hermes_lcm.engine import _summary_part_text
from hermes_lcm.reconcile import _PRESERVED_TODO_CONTEXT_PREFIX
from tests.invariants.test_issue_2815_c4 import STEER, _closing, _project, _stage_marked
from tests.invariants.test_issue_2815_uid_restart import _open, _seed_covered_view, _store_copies, _view
from tests.test_compaction_telemetry import _hot_usage
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _rows, summaries  # noqa: F401


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")


@pytest.mark.parametrize("site", ["resolve", "map", "drop", "cut", "copy", "publication", "engine"])
def test_shadow_uses_base_path(tmp_path, monkeypatch, site):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    _state_db(tmp_path, [("S", None, None)])
    with _closing(_open(tmp_path)) as engine:
        rows, ids = _seed_covered_view(engine)
        stored = _rows(engine)
        if site == "resolve":
            assert engine._host_uid_resolve(rows) == {}
        elif site == "map":
            assert engine._get_store_id_map_for_messages(rows) == engine._get_store_id_map_for_messages(
                rows, uid_first=False)
        elif site == "drop":
            engine._compress_host_rows_consumed = 0
            assert engine._drop_uid_covered_live_rows(rows) is rows
            assert engine._compress_host_rows_consumed == 0
        elif site == "cut":
            assert engine._host_uid_tail_cut(rows, stored[len(rows) - 1]) is None
        elif site == "copy":
            import hermes_lcm.store_complete as complete
            monkeypatch.setattr(complete, "_COPY_WINDOW", 1)
            twins, _ = engine._store_complete_copy_twins(stored[len(rows):], [])
            assert not any(twin in ids for sources in twins.values() for twin in sources)
        elif site == "publication":
            reads = []
            monkeypatch.setattr(engine, "_host_uid_lineage_key", lambda: (reads.append(True), None))
            assert not engine._store_complete_uid_copy_publication_match(None, None, None)
            assert not reads
        else:
            engine._engine_uids_pending = {"pending": ("different-profile", "summary")}
            assert engine._host_uid_engine_uids({"pending"}) == {"pending"}


@pytest.mark.parametrize("cold", [False, True])
def test_shadow_ingest_stores_the_base_occurrence(tmp_path, monkeypatch, cold):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    _state_db(tmp_path, [("S", None, None)])
    reply = {"role": "assistant", "content": "A known UID replayed after a fresh turn", "message_uid": "reply"}
    fresh = {"role": "user", "content": "Fresh turn", "timestamp": 1000.0}
    with _closing(_open(tmp_path)) as seed:
        seed.ingest([reply])
        if not cold:
            unstamped_prefix = {key: value for key, value in reply.items() if key != "message_uid"}
            seed.ingest([unstamped_prefix, fresh, reply])
            assert [row["content"] for row in _rows(seed)] == [reply["content"], fresh["content"], reply["content"]]
    if cold:
        with _closing(_open(tmp_path)) as replay:
            replay.ingest([fresh, reply])
            assert [row["content"] for row in _rows(replay)] == [reply["content"], fresh["content"], reply["content"]]
            assert "UID-bound" not in replay._last_ingest_reconciliation.get("reason", "")


@pytest.mark.parametrize("ledger", ["shadow", "off"])
def test_failed_located_cut_falls_back_to_tool_pair(tmp_path, monkeypatch, on, ledger):
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    _state_db(tmp_path, [("S", None, None)])
    _, call, result, reply = _project(_view(1))
    for row in (call, result):
        row.pop("message_uid")
    fresh = {"role": "user", "content": "Fresh user turn" + " synthetic padding" * 12, "timestamp": 1100.0}
    with _closing(_open(tmp_path)) as seed:
        seed.ingest([call, result, reply])
    with _closing(_open(tmp_path)) as cold:
        view = [call, result, fresh, reply]
        assert cold._reconcile_ingest_cursor_from_store(view) == 2
        cold.ingest(view)
        assert [(row["role"], row["content"] or "") for row in _rows(cold)] == [
            (row["role"], row["content"] or "") for row in [call, result, reply, fresh]
        ]


@pytest.mark.parametrize(("mode", "turns", "count"), [("on", 75, 64), ("shadow", 75, 301), ("on", 15, 61)])
def test_unlocated_cursor_cut_budget(tmp_path, monkeypatch, mode, turns, count):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    _state_db(tmp_path, [("S", None, None)])
    rows = _view(turns)
    with _closing(_open(tmp_path)) as seed:
        seed.ingest(rows)
        _store_copies(seed, [rows[-1]])
    view = _project(rows)
    prefix = int(count == 64)  # the budgeted cell needs a proven first row
    view[prefix].pop("message_uid")
    for i, row in enumerate(view[prefix:], start=prefix):
        if row["role"] == "user":
            row["timestamp"] += 0.5
        if row["role"] in {"user", "assistant"} and row["content"]:
            row["content"] += f" Edited host row {i}."
    with _closing(_open(tmp_path)) as cold:
        cuts = []
        project = cold._project_replay_candidate

        def count_cut(run, candidate):
            cuts.append(candidate.cursor)
            return project(run, candidate)

        monkeypatch.setattr(cold, "_project_replay_candidate", count_cut)
        cold._reconcile_ingest_cursor_from_store(view)
        assert cuts == list(range(len(view), len(view) - count, -1))


@pytest.mark.parametrize("where", ["summary", "ingest"])
def test_overlapping_compress_leaves_owner_state_alone(tmp_path, monkeypatch, where):
    owner_ready, release_owner = Event(), Event()
    caller = get_ident()

    def summarize(**kwargs):
        if get_ident() != caller:
            owner_ready.set()
            assert release_owner.wait(5)
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    solo_home = tmp_path / "solo"
    solo_home.mkdir()
    rows = _view(6)
    with _closing(_open(solo_home)) as solo:
        expected = solo.compress(deepcopy(rows), current_tokens=24000)
        expected_status = solo.last_compression_status
        expected_reason = solo.last_compression_noop_reason
    with _closing(_open(tmp_path)) as engine, ThreadPoolExecutor(max_workers=2) as workers:
        if where == "ingest":
            ingest = engine._ingest_messages_transaction

            def stall_ingest(*args, **kwargs):
                owner_ready.set()
                assert release_owner.wait(5)
                return ingest(*args, **kwargs)

            monkeypatch.setattr(engine, "_ingest_messages_transaction", stall_ingest)
        primary = workers.submit(engine.compress, deepcopy(rows), 24000)
        assert owner_ready.wait(5)
        status = (engine.last_compression_status, engine.last_compression_noop_reason, engine._compress_calls)
        fallback = deepcopy(rows)
        try:
            second = workers.submit(engine.compress, fallback, 24000)
            assert second.result(timeout=1) is fallback
            assert (engine.last_compression_status, engine.last_compression_noop_reason, engine._compress_calls) == status
            assert engine._compress_reentry_refusals == 1

            @contextmanager
            def bypassed():
                yield True

            with monkeypatch.context() as patch:
                patch.setattr(engine, "_bypass_compression_binding", bypassed)
                patch.setattr(engine, "_bypass_lcm_session_id", lambda: "child")
                child = [{"role": "assistant", "content": "Child ran"}]
                patch.setattr(engine, "_compress_delegate_child_bypassed", lambda *args: child)
                assert engine.compress(fallback) is child
                assert engine._compress_reentry_refusals == 1
        finally:
            release_owner.set()
        assert primary.result(timeout=5) == expected
        assert (engine.last_compression_status, engine.last_compression_noop_reason) == (
            expected_status, expected_reason)


@pytest.mark.parametrize("attribute", ["_no_progress_hold", "_hung_summary_strike"])
def test_hold_tuple_is_read_once(tmp_path, monkeypatch, attribute):
    with _closing(_open(tmp_path)) as engine:
        value = (10**12, "no_progress") if attribute == "_no_progress_hold" else (
            engine._hold_conversation_key(), "primary")

        def clear_on_read(instance):
            nonlocal value
            current, value = value, None
            return current

        monkeypatch.setattr(type(engine), attribute, property(clear_on_read), raising=False)
        if attribute == "_no_progress_hold":
            assert engine._no_progress_hold_active(managed_scope=True)
        else:
            budget = engine._new_foreground_budget()
            budget.hung_route = "primary"
            state = compaction._CompressionState()
            state.budget = budget
            engine._settle_leaf_stop(state, lambda: None)


def test_turn_telemetry_never_waits_for_compaction_lock(tmp_path):
    with _closing(_open(tmp_path)) as engine, ThreadPoolExecutor(max_workers=2) as workers:
        locked, release = Event(), Event()
        engine._compaction_telemetry_counter_rebaseline_pending = True
        engine._compaction_telemetry_turn_reset_pending = True

        def hold():
            with engine._store._write_lock:
                locked.set()
                release.wait(5)

        owner = workers.submit(hold)
        assert locked.wait(5)
        try:
            workers.submit(engine.update_from_response, _hot_usage()).result(timeout=1)
            assert engine._compaction_telemetry_skips == 1
            assert engine._compaction_telemetry_counter_rebaseline_pending
            assert engine._compaction_telemetry_turn_reset_pending
        finally:
            release.set()
        owner.result(timeout=1)
        engine.update_from_response(_hot_usage())
        assert not engine._compaction_telemetry_counter_rebaseline_pending
        assert not engine._compaction_telemetry_turn_reset_pending


@pytest.mark.parametrize("seconds", [60, 120])
def test_budget_stop_returns_pre_drop_view_and_next_pass_publishes(
    tmp_path, monkeypatch, on, summaries, seconds,  # noqa: F811
):
    _state_db(tmp_path, [("S", None, None)])
    with _closing(_open(tmp_path, threshold_full_sweep_enabled=True, context_threshold=0.1)) as engine:
        rows, _ = _seed_covered_view(engine)
        head = {"role": "user", "content": _summary_part_text(engine._dag.get_node(1)), "message_uid": "engine-head"}
        engine._store.add_host_uid_bindings(
            engine._host_uid_lineage_key()[0], [(0, "engine-head", "engine", "summary")])
        view = [head, *rows]
        real_clock = compaction.time.monotonic
        offset = 0
        prepare = engine._prepare_retained_user_anchor

        def spend(*args):
            nonlocal offset
            prepare(*args)
            offset += seconds
            return None

        with monkeypatch.context() as patch:
            patch.setattr(compaction.time, "monotonic", lambda: real_clock() + offset)
            patch.setattr(engine, "_prepare_retained_user_anchor", spend)
            new_budget = engine._new_foreground_budget

            def timed_budget():
                budget = new_budget()
                budget.hard = seconds
                return budget

            patch.setattr(engine, "_new_foreground_budget", timed_budget)
            expected = engine._sanitize_active_context_messages(view, insert_missing_tool_stubs=False)
            first = engine.compress(view, current_tokens=24000)
        assert first == expected
        assert not summaries
        assert engine._no_progress_hold is None
        engine.note_turn_complete()
        second = engine.compress(first, current_tokens=24000)
        assert summaries
        assert all(head["content"] not in summary for summary in summaries)
        assert len(second) < len(first)
        assert engine._no_progress_hold is None


@pytest.mark.parametrize("source", ["patterns", "records"])
def test_ignore_state_uses_base_path(tmp_path, on, source):
    _state_db(tmp_path, [("S", None, None)])
    settings = {"ignore_message_patterns": ["synthetic-unmatched"]} if source == "patterns" else {}
    with _closing(_open(tmp_path, **settings)) as engine:
        rows, _ = _seed_covered_view(engine)
        if source == "records":
            engine._write_generated_ignored_dependent_reply_records([{"store": "1234567890123456"}])
        expected = engine._get_store_id_map_for_messages(rows, uid_first=False)
        assert engine._get_store_id_map_for_messages(rows) == expected
        engine._compress_host_rows_consumed = 0
        assert engine._drop_uid_covered_live_rows(rows) is rows


def test_engine_uid_todo_with_new_paragraph_is_stored(tmp_path, on):
    _state_db(tmp_path, [("S", None, None)])
    seed_rows = _view(1)
    extension = "A fresh synthetic paragraph about a different request."
    todo = {"role": "user", "message_uid": "engine-todo", "timestamp": 5000.0,
            "content": f"{_PRESERVED_TODO_CONTEXT_PREFIX}\n- [ ] Synthetic item (pending)\n\n{extension}"}
    with _closing(_open(tmp_path)) as seed:
        seed.ingest(seed_rows)
        seed._store.add_host_uid_bindings(seed._host_uid_lineage_key()[0], [(0, "engine-todo", "engine", "summary")])
    with _closing(_open(tmp_path)) as cold:
        cold.ingest([*_project(seed_rows), todo])
        assert sum(extension in (row["content"] or "") for row in _rows(cold)) == 1


def test_located_matching_cut_needs_one_projection(tmp_path, monkeypatch, on):
    _state_db(tmp_path, [("S", None, None)])
    rows = _view(8)
    with _closing(_open(tmp_path)) as seed:
        seed.ingest(rows)
    view = deepcopy(rows)
    view[0].pop("message_uid")
    with _closing(_open(tmp_path)) as cold:
        projections = []
        project = cold._project_replay_candidate

        def count(run, candidate):
            projections.append(candidate.cursor)
            return project(run, candidate)

        monkeypatch.setattr(cold, "_project_replay_candidate", count)
        assert cold._reconcile_ingest_cursor_from_store(view) == len(view)
        assert projections == [len(view)]


def test_native_recovery_digest_below_cut_budget_is_still_checked(tmp_path, monkeypatch, on):
    _state_db(tmp_path, [("S", None, None)])
    rows = _view(20)
    with _closing(_open(tmp_path)) as seed:
        seed.ingest(rows)
        _store_copies(seed, rows[-1:])
    view = _project(rows)
    for row in view[1:]:  # keep a proven first row so the budget applies
        row.pop("message_uid")
        row["content"] = (row["content"] or "") + " Different synthetic row."
    with _closing(_open(tmp_path)) as cold:
        projections = []
        project = cold._project_replay_candidate

        def count(run, candidate):
            projections.append(candidate.cursor)
            return project(run, candidate)

        monkeypatch.setattr(cold, "_project_replay_candidate", count)
        monkeypatch.setattr(cold, "_load_native_recovery_replay_snapshot_digests", lambda: {"native"})
        monkeypatch.setattr(cold, "_replay_snapshot_prefix_digests", lambda *a, **kw: ({4: "native"}, {}, {}))
        assert cold._reconcile_ingest_cursor_from_store(view) == 4
        assert len(projections) == 64


def test_leaf_replaces_a_composite_engine_head(tmp_path, on, summaries):  # noqa: F811
    _state_db(tmp_path, [("S", None, None)])
    with _closing(_open(tmp_path)) as engine:
        rows, _ = _seed_covered_view(engine)
        old_summary = _summary_part_text(engine._dag.get_node(1))
        old_head = {"role": "user", "content": f"{old_summary}\n\n{old_summary}", "message_uid": "old-head"}
        engine._store.add_host_uid_bindings(
            engine._host_uid_lineage_key()[0], [(0, "old-head", "engine", "summary")])
        result = engine.compress([old_head, *rows], current_tokens=24000)
        assert summaries
        assert set(row.get("message_uid") for row in result).isdisjoint({"old-head"})
        assert sum(str(row.get("content")).count(old_summary) for row in result) == 1
        assert all(old_summary not in summary for summary in summaries)
        assert len(result) < len(rows) + 1
        assert engine._no_progress_hold is None


def test_ignored_record_does_not_suppress_a_distinct_bound_reply(tmp_path, on, summaries):  # noqa: F811
    _state_db(tmp_path, [("S", None, None)])
    reply_text = "The synthetic operation succeeded."
    with _closing(_open(tmp_path, leaf_chunk_tokens=8000, fresh_tail_count=1)) as engine:
        [ignored_id] = _store_copies(engine, [{"role": "assistant", "content": reply_text}])
        head = {"role": "user", "content": "Covered head", "message_uid": "head", "timestamp": 1000.0}
        question = {"role": "user", "content": "Please confirm", "message_uid": "question", "timestamp": 1010.0}
        reply = {"role": "assistant", "content": reply_text, "message_uid": "reply", "timestamp": 1011.0}
        view = [head, question, reply]
        engine.ingest(view)
        head_id = next(row["store_id"] for row in _rows(engine) if row["content"] == head["content"])
        _stage_marked(engine, [ignored_id, head_id])
        record = {"store": engine._ignored_dependent_reply_store_fingerprint(
            {"role": "assistant", "store_id": ignored_id})}
        engine._write_generated_ignored_dependent_reply_records([record])
        output = engine.compress(deepcopy(view), current_tokens=1, bypass_cooldown=True)
        assert reply in output
        assert not summaries
        assert engine._load_generated_ignored_dependent_reply_records() == [record]


def test_uid_match_requires_oob_blocks_on_both_sides(tmp_path, on):
    _state_db(tmp_path, [("S", None, None)])
    rows = _view(1)
    with _closing(_open(tmp_path)) as engine:
        tool = rows[2]
        [store_id] = _store_copies(engine, [dict(tool, content=tool["content"] + STEER)])
        engine._store.add_host_uid_bindings(engine._host_uid_lineage_key()[0], [
            (store_id, tool["message_uid"], "canonical", "stored_new")])
        assert engine._host_uid_resolve([tool]) == {}


def test_hold_status_survives_concurrent_clear(tmp_path, monkeypatch):
    with _closing(_open(tmp_path)) as engine:
        engine._no_progress_hold = (10**12, "hidden_only")

        def clear_after_check(*, managed_scope):
            engine._no_progress_hold = None
            return True

        monkeypatch.setattr(engine, "_no_progress_hold_active", clear_after_check)
        assert engine._no_progress_hold_status() is None
