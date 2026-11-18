"""Synthetic regressions for the #2815 c4 review and cold-bind cut point."""

import logging
import sqlite3
import time
from contextlib import closing, contextmanager
from copy import deepcopy

import pytest

from hermes_lcm import compaction, host_uid_emit
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER
from hermes_lcm.tokens import count_messages_tokens
from tests.invariants.test_issue_2815_uid_restart import _open, _store_copies, _view
from tests.test_host_uid_engine_uids import _compress, _host
from tests.test_host_uid_engine_uids import _open as _open_emitter
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import SummaryNode, _covered, _rows, summaries  # noqa: F401


@pytest.fixture(autouse=True)
def _enable_uid_recognition(monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")


MARK = "Staged coverage marker for the original head"
STEER = "\n\n[OUT-OF-BAND USER MESSAGE]\nKeep the synthetic instruction.\n[/OUT-OF-BAND USER MESSAGE]"


@contextmanager
def _closing(engine):
    try:
        yield engine
    finally:
        engine.shutdown()


def _project(rows):
    view = deepcopy(rows)
    for row in view:
        if row["role"] != "user":
            row.pop("timestamp", None)
            row.pop("observed", None)
    return view


def _dump(path):
    with closing(sqlite3.connect(path)) as conn:
        return tuple(conn.iterdump())


def _stage_marked(engine, ids):
    node = SummaryNode(session_id="S", summary=MARK, token_count=12, source_token_count=100, source_ids=ids)
    engine._dag.add_node(node, before_commit=lambda conn, node_id: engine._lifecycle.stage_compaction_publication(
        conn, "conv", "S", node_id, 0, ids))
    engine._last_compacted_store_id = ids[-1]


def _marked(output):
    return any(MARK in str(row.get("content") or "") for row in output)


@pytest.mark.parametrize("carrier", ["located", "unbound", "missing", "duplicate", "edited", "native"])
def test_cold_bind_projects_only_the_uid_located_cut(tmp_path, monkeypatch, carrier):
    _state_db(tmp_path, [("S", None, None)])
    rows = _view(75 if carrier == "located" else 6)
    with _closing(_open(tmp_path)) as seed:
        stored = []
        for index, row in enumerate(rows):
            if index >= 4 and (carrier == "located" or index in {len(rows) - 13, len(rows) - 7}):
                copy = rows[index - 4].copy()
                copy.pop("message_uid")
                stored.append(copy)
            stored.append(row)
        if carrier == "unbound":
            stored[-1] = dict(stored[-1], message_uid=None)
        seed.ingest(stored)
    view = _project(rows)
    for index, row in enumerate(view):
        if row["role"] == "tool" and index < 12:
            row["content"] = _HERMES_PRUNED_TOOL_PLACEHOLDER
        elif index % 4 == 3:
            row["timestamp"] = rows[index]["timestamp"] + 0.5
    view[12].pop("message_uid")
    if carrier == "missing":
        view[-1].pop("message_uid")
    elif carrier == "duplicate":
        view[12]["message_uid"] = view[-1]["message_uid"]
    elif carrier == "edited":
        view[-1]["content"] += " Edited host text."
    with _closing(_open(tmp_path)) as cold:
        if carrier == "native":
            view.append({"role": "user", "content": "Native recovery suffix"})
            monkeypatch.setattr(cold, "_load_native_recovery_replay_snapshot_digests", lambda: {"native"})
            monkeypatch.setattr(cold, "_replay_snapshot_prefix_digests",
                                lambda *a, **kw: ({len(view): "native"}, {}, {}))
        projections = []
        project = cold._project_replay_candidate

        def count(run, candidate):
            projections.append(candidate.cursor)
            return project(run, candidate)

        monkeypatch.setattr(cold, "_project_replay_candidate", count)
        before = _dump(tmp_path / "lcm.db")
        cursor = cold._reconcile_ingest_cursor_from_store(view)
        after = _dump(tmp_path / "lcm.db")
        assert before == after
        print(f"carrier={carrier} cursor={cursor} projections={len(projections)}")
        if carrier == "native":
            assert cursor == len(view)
            assert not projections
        elif carrier == "located":
            assert cursor == 12
            assert projections == list(range(len(view), len(view) - 64, -1))
        else:
            assert projections == list(range(len(view), -1, -1))


@pytest.mark.parametrize("where", ["index0_cold", "past_cursor_cold", "past_cursor_warm"])
def test_fresh_text_under_engine_uid_is_stored_once(tmp_path, where):
    _state_db(tmp_path, [("S", None, None)])
    rows = _view(5)
    with _closing(_open(tmp_path)) as seed:
        seed.ingest(rows)
        fresh = {"role": "user", "content": "New text under a generated UID", "message_uid": "engine-test",
                 "timestamp": 5000.0}
        seed._store.add_host_uid_bindings(seed._host_uid_lineage_key()[0], [(0, "engine-test", "engine", "summary")])
    with _closing(_open(tmp_path)) as engine:
        if where == "past_cursor_warm":
            engine.ingest(deepcopy(rows))
        view = _project(rows) if where != "past_cursor_warm" else deepcopy(rows)
        engine.ingest([fresh, *view] if where == "index0_cold" else [*view, fresh])
        assert sum(row["content"] == fresh["content"] for row in _rows(engine)) == 1


@pytest.mark.parametrize("tail", [1, 3])
@pytest.mark.parametrize("cut", ["through_glued_row", "whole"])
def test_glued_summary_under_engine_uid_stores_suffix(tmp_path, monkeypatch, tail, cut):
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    _state_db(tmp_path, [("S", None, None)])
    host = _host()
    with _closing(_open_emitter(tmp_path, tail=tail)) as engine:
        engine.ingest(deepcopy(host))
        output = _compress(engine, host)
        for i, row in enumerate(output):
            row.setdefault("message_uid", f"flushed-{i}")
        idx = next(i for i, row in enumerate(output) if row["role"] == "user"
                   and not row.get("_absorbed_message_uids")
                   and engine._host_uid_engine_uids({row["message_uid"]}))
        engine.ingest(deepcopy(output))
    view = deepcopy(output[:idx + 1] if cut == "through_glued_row" else output)
    suffix = "New synthetic user instruction"
    view[idx]["content"] += "\n\n" + suffix
    with _closing(_open_emitter(tmp_path, tail=tail)) as cold:
        cold.ingest(view)
        assert sum(suffix in str(row["content"]) for row in _rows(cold)) == 1


@pytest.mark.parametrize("pending", [False, True], ids=["recorded", "pending"])
@pytest.mark.parametrize("preseed_b", [False, True], ids=["rebind", "cold_reconcile"])
def test_engine_uid_does_not_cross_home_rebind(tmp_path, monkeypatch, pending, preseed_b):
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    home_a, home_b = tmp_path / "A", tmp_path / "B"
    home_a.mkdir()
    home_b.mkdir()
    _state_db(home_a, [("S", None, None)])
    _state_db(home_b, [("T", None, None)])
    seed_b = [{"role": "user", "content": "Existing home B question", "timestamp": 5000.0, "message_uid": "b0"}]
    if preseed_b:
        with _closing(_open(tmp_path, "T")) as seed:
            seed._hermes_home = seed._store._hermes_home = str(home_b)
            seed.ingest(deepcopy(seed_b))
    with _closing(_open_emitter(tmp_path, tail=3)) as engine:
        engine._hermes_home = engine._store._hermes_home = str(home_a)
        if pending:
            monkeypatch.setattr(engine, "_host_uid_record_engine", lambda output: None)
        host = _host()
        output = _compress(engine, host)
        generated = deepcopy(next(row for row in output if row["role"] == "user"
                                  and not row.get("_absorbed_message_uids")
                                  and engine._host_uid_engine_uids({row["message_uid"]})))
        engine.on_session_start("T", platform="cli", context_length=200_000, conversation_id="conv",
                                hermes_home=str(home_b))
        engine._hermes_home = engine._store._hermes_home = str(home_b)
        recognized = engine._host_uid_engine_uids({generated["message_uid"]})
        view = [generated, *(seed_b if preseed_b else []),
                {"role": "user", "content": "New home B question", "timestamp": 5200.0, "message_uid": "b1"}]
        engine.ingest(view)
        assert sum(row["content"] == generated["content"] for row in _rows(engine)) == 1
        assert not recognized


@pytest.mark.parametrize("exit_kind", ["hold", "cleanup"])
def test_covered_head_survives_no_summary_exit(tmp_path, monkeypatch, summaries, exit_kind):  # noqa: F811
    _state_db(tmp_path, [("S", None, None)])
    with _closing(_open(tmp_path)) as engine:
        rows = _view(7)
        engine.ingest(rows)
        _stage_marked(engine, [row["store_id"] for row in _rows(engine)[:8]])
        _store_copies(engine, rows)
        if exit_kind == "cleanup":
            engine._compaction_request.cleanup_only_due_to_boundary_cooldown = True
        else:
            engine._no_progress_hold = (10**12, "no_progress")
            monkeypatch.setattr(engine, "_no_progress_hold_blocks", lambda *a, **kw: True)
        output = engine.compress(rows, current_tokens=24000)
        assert not summaries
        assert _marked(output) or all(row in output for row in rows[:8])


@pytest.mark.parametrize("slack", [-1, -30, -90, -500])
def test_budget_before_first_leaf_preserves_uncovered_rows(tmp_path, monkeypatch, summaries, slack, caplog):  # noqa: F811
    _state_db(tmp_path, [("S", None, None)])
    rows = [{"role": "user", "content": "hi", "timestamp": 900.0, "message_uid": "head-user"},
            {"role": "assistant", "content": "ok", "timestamp": 902.0, "message_uid": "head-assistant"},
            *_view(6)]
    # The verifier used unstretched tool outputs to put the result at the cap.
    for row in rows:
        if row["role"] == "tool":
            row["content"] = row["content"].removesuffix(" durable output" * 24)
    view = [{"role": "system", "content": "system prompt"}, *rows]
    with _closing(_open(tmp_path, threshold_full_sweep_enabled=True, max_assembly_tokens=count_messages_tokens(view) - slack)) as engine:
        engine.ingest(view)
        stored = _rows(engine)
        _stage_marked(engine, [row["store_id"] for row in stored[:3]])
        _store_copies(engine, rows)
        original = engine._prepare_retained_user_anchor
        real_time = time.monotonic
        offset = 0

        monkeypatch.setattr(compaction.time, "monotonic", lambda: real_time() + offset)

        def spend_budget(*args):
            nonlocal offset
            result = original(*args)
            offset = 100_000
            return result

        monkeypatch.setattr(engine, "_prepare_retained_user_anchor", spend_budget)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            output = engine.compress(view, current_tokens=1)
        assert "spent its time budget before the first leaf" in caplog.text
        assert not summaries
        emitted = {row.get("message_uid") for row in output}
        text = "\n".join(str(row.get("content") or "") for row in output)
        missing = [row["message_uid"] for row in rows[2:]
                   if row["message_uid"] not in emitted and (row["content"] or "x") not in text]
        assert not missing
        assert _marked(output) or all(row in output for row in rows[:2])
        if slack == -500:
            assert all(row in output for row in rows[:2])


@pytest.mark.parametrize("blocks", [1, 2])
def test_steer_copy_stays_in_summarizer_input(tmp_path, summaries, blocks):  # noqa: F811
    _state_db(tmp_path, [("S", None, None)])
    with _closing(_open(tmp_path, leaf_chunk_tokens=8000)) as engine:
        rows = _view(2)
        engine.ingest(rows)
        _stage_marked(engine, [row["store_id"] for row in _rows(engine)])
        steer = dict(rows[2], content=rows[2]["content"] + STEER * blocks)
        [steer_id] = _store_copies(engine, [steer])
        engine.compress([{"role": "user", "content": "Fresh question"},
                         {"role": "assistant", "content": "Fresh reply"}], current_tokens=24000)
        assert steer_id in _covered(engine)
        assert sum(text.count("Keep the synthetic instruction.") for text in summaries) == blocks


@pytest.mark.parametrize("blocks", [1, 2])
def test_live_steer_does_not_resolve_to_bound_row_without_blocks(tmp_path, blocks):
    _state_db(tmp_path, [("S", None, None)])
    with _closing(_open(tmp_path)) as engine:
        rows = _view(3)
        engine.ingest(rows)
        _stage_marked(engine, [row["store_id"] for row in _rows(engine)[:4]])
        view = deepcopy(rows)
        view[2]["content"] += STEER * blocks
        _store_copies(engine, [view[2]])
        engine._compress_host_rows_consumed = 0
        assert 2 not in engine._host_uid_resolve(view)
        assert view[2] in engine._drop_uid_covered_live_rows(view)


def test_blank_call_id_copy_stays_in_summarizer_input(tmp_path, summaries):  # noqa: F811
    _state_db(tmp_path, [("S", None, None)])
    with _closing(_open(tmp_path, leaf_chunk_tokens=8000)) as engine:
        rows = _view(1)
        del rows[2]
        rows[1]["tool_calls"][0]["id"] = ""
        engine.ingest(rows)
        _stage_marked(engine, [row["store_id"] for row in _rows(engine)])
        [copy_id] = _store_copies(engine, [rows[1]])
        engine.compress([{"role": "user", "content": "Fresh question"},
                         {"role": "assistant", "content": "Fresh reply"}], current_tokens=24000)
        assert copy_id in _covered(engine)
        assert summaries


def test_uid_off_publication_copy_check_never_reads_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    with _closing(_open(tmp_path)) as engine:
        calls = []
        monkeypatch.setattr(engine, "_host_uid_lineage_key", lambda: (calls.append(True), None))
        assert not engine._store_complete_uid_copy_publication_match(None, None, None)
        assert not calls



@pytest.mark.parametrize("kind", ["tool", "_delegate_from"])
def test_delegate_child_cannot_resolve_parent_uids(tmp_path, kind):
    child = ("D", "S", None, "tool") if kind == "tool" else ("D", "S", None, "cli", '{"_delegate_from":"S"}')
    _state_db(tmp_path, [("S", None, "compression"), child])
    with _closing(_open(tmp_path)) as parent:
        parent.ingest(_view(4))
    with _closing(_open(tmp_path, "D")) as engine:
        view = _project(_view(4))
        assert not engine._host_uid_resolve(view)
        assert not engine._host_uid_resolve(view, prefix_only=True)
        assert engine._get_store_id_map_for_messages(view) == engine._get_store_id_map_for_messages(view, uid_first=False)
        assert engine._drop_uid_covered_live_rows(view) is view


def test_second_home_does_not_share_host_uid_bindings(tmp_path):
    home_b = tmp_path / "B"
    home_b.mkdir()
    _state_db(tmp_path, [("S", None, None)])
    _state_db(home_b, [("S", None, None)])
    view = _project(_view(4))
    with _closing(_open(tmp_path)) as first:
        first.ingest(_view(4))
        a = first._host_uid_lineage_key()[0]
        assert first._host_uid_resolve(view)
        first._hermes_home = first._store._hermes_home = str(home_b)
        assert first._host_uid_lineage_key()[0] != a
        assert not first._host_uid_resolve(view)
    with _closing(_open(tmp_path)) as second:
        second._hermes_home = second._store._hermes_home = str(home_b)
        assert not second._host_uid_resolve(view, prefix_only=True)
        assert not second._host_uid_resolve(view)
        assert second._get_store_id_map_for_messages(view) == second._get_store_id_map_for_messages(view, uid_first=False)


def test_partial_rotation_carry_resolves_only_owned_rows(tmp_path):
    _state_db(tmp_path, [("S1", None, "compression"), ("S2", "S1", None)])
    rows = _view(4)
    with _closing(_open(tmp_path, "S1")) as first:
        first.ingest(rows)
        carried = _rows(first)[8:]
    with _closing(_open(tmp_path, "S2")) as second:
        second._register_identity_anchor_carry(carried)
        assert {i: row["store_id"] for i, row in second._host_uid_resolve(_project(rows)).items()} == {
            i: row["store_id"] for i, row in enumerate(carried, 8)}


@pytest.mark.parametrize("session_end", [False, True], ids=["helpers", "frozen-session-end"])
def test_uid_view_does_not_write_database_in_read_only_pass(tmp_path, session_end):
    _state_db(tmp_path, [("S", None, None)])
    rows = _view(4)
    with _closing(_open(tmp_path)) as seed:
        seed.ingest(rows)
    with _closing(_open(tmp_path)) as engine:
        view = _project(rows)
        before = _dump(tmp_path / "lcm.db")
        if session_end:
            engine._ingest_read_only = RuntimeError("session-end evidence deferred")
            engine._session_end_flush_active = True
            engine._ingest_messages(view, allow_session_end_replay_proof=True)
        else:
            engine._host_uid_resolve(view)
            engine._host_uid_resolve(view, view, prefix_only=True)
            engine._get_store_id_map_for_messages(view)
            engine._compress_host_rows_consumed = 0
            engine._drop_uid_covered_live_rows(view)
            engine._store_complete_uid_copy_sources(engine._host_uid_lineage_key()[0],
                                                     engine._load_compression_carry_ranges())
            engine._is_generated_ignored_dependent_reply(view[3], view[3]["content"])
        assert _dump(tmp_path / "lcm.db") == before
