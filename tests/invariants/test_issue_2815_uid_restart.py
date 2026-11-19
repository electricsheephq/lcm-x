"""#2815/#2818. Synthetic projected host restarts and durable compaction attribution."""

from copy import deepcopy

import pytest

from hermes_lcm.engine import _summary_part_text
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER
from tests.test_host_uid_shadow import _counts, _state_db
from tests.test_issue_581_582_store_complete_survival import (
    _covered,
    _engine,
    _frontier,
    _rows,
    _stage,
    _turn,
    summaries,  # noqa: F401
)


@pytest.fixture(autouse=True)
def _enable_uid_recognition(monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")


def _open(path, session="S", **settings):
    engine = _engine(path, session=session, **settings)
    engine._hermes_home = engine._store._hermes_home = str(path)
    return engine


def _view(turns):
    rows = [row for i in range(turns) for row in _turn(f"turn-{i}", 1000.0 + i * 10, tool=True)]
    for i, row in enumerate(rows):
        row.update(message_uid=f"uid-{i}", timestamp=1000.0 + i * 2)
        if row["role"] == "tool":
            row["content"] += " durable output" * 24
    return rows


def _store_copies(engine, rows):
    return [engine._store.append("S", row, conversation_id="conv") for row in rows]


@pytest.mark.parametrize("restamped", [False, True], ids=["projected", "restamped"])
def test_2815_cold_engine_projected_view_stores_nothing(tmp_path, monkeypatch, restamped):
    _state_db(tmp_path, [("S", None, None)])
    rows = _view(15)
    engine = _open(tmp_path)
    try:
        seed = [item for i, row in enumerate(rows) for item in
                (row, {"role": "assistant", "content": f"hidden seed {i}"})]
        engine.ingest(seed)
        stored = _rows(engine)
    finally:
        engine.shutdown()
    projected = deepcopy(rows)
    for idx, row in enumerate(projected):
        if row["role"] != "user":
            if restamped:
                row["timestamp"] += 0.5
            else:
                row.pop("timestamp")
                row.pop("observed", None)
        if row["role"] == "tool" and idx < 24:
            row["content"] = _HERMES_PRUNED_TOOL_PLACEHOLDER
    cold = _open(tmp_path)
    try:
        searches = []
        original_search = cold._find_reconciled_cursor_for_store_tail

        def search(*args, **kwargs):
            searches.append(True)
            return original_search(*args, **kwargs)

        monkeypatch.setattr(cold, "_find_reconciled_cursor_for_store_tail", search)
        cold.ingest(projected)
        assert len(_rows(cold)) == len(stored)
        assert not _counts(cold).get("bound.disagree.stored_despite_match")
        assert searches == []
    finally:
        cold.shutdown()


def _seed_covered_view(engine):
    rows = _view(7)
    engine.ingest(rows)
    store_ids = [row["store_id"] for row in _rows(engine)]
    _stage(engine, 0, store_ids[:8])
    engine._last_compacted_store_id = store_ids[7]
    _store_copies(engine, rows)
    return rows, store_ids


def test_2815_pass_map_follows_uid_over_restart_copies(tmp_path):
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        rows, store_ids = _seed_covered_view(engine)
        assert engine._get_store_id_map_for_messages(rows) == {
            id(row): sid for row, sid in zip(rows[8:], store_ids[8:])
        }
    finally:
        engine.shutdown()


def test_2815_covered_live_rows_leave_the_list(tmp_path, monkeypatch, summaries):  # noqa: F811
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path)
    try:
        rows, store_ids = _seed_covered_view(engine)
        covering = _summary_part_text(engine._dag.get_node(1))
        mapping = engine._get_store_id_map_for_messages
        calls = []

        def count(*args, **kwargs):
            calls.append(kwargs.get("uid_first", True))
            return mapping(*args, **kwargs)

        monkeypatch.setattr(engine, "_get_store_id_map_for_messages", count)
        output = engine.compress(rows, current_tokens=24000)
        print(f"RT3 mapper calls={len(calls)} uid_first={sum(calls)} ordered={len(calls)-sum(calls)}")
        assert any(covering in str(row.get("content") or "") for row in output)
        emitted = {row.get("message_uid") for row in output}
        assert not emitted.intersection(row["message_uid"] for row in rows[:8])
        assert engine._compress_host_rows_consumed > 0
        assert not engine._no_progress_hold or engine._no_progress_hold[1] != "hidden_only"
        assert set(store_ids[:8]) <= set(_covered(engine))
        for i in range(1, len(rows), 4):
            assert (rows[i]["message_uid"] in emitted) == (rows[i + 1]["message_uid"] in emitted)
        assert all("turn-0" not in text and "turn-1]" not in text for text in summaries)
    finally:
        engine.shutdown()



@pytest.mark.parametrize("pruned", [False, True], ids=["literal", "pruned"])
def test_2815_restart_copies_are_not_resummarised(tmp_path, monkeypatch, summaries, pruned):  # noqa: F811
    from hermes_lcm import store_complete

    monkeypatch.setattr(store_complete, "_COPY_WINDOW", 1)
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path, leaf_chunk_tokens=8000)
    try:
        original = _view(4)
        engine.ingest(original)
        original_ids = [row["store_id"] for row in _rows(engine)]
        _stage(engine, 0, original_ids[:8])
        engine._last_compacted_store_id = original_ids[7]
        copies = deepcopy([row for row in original if row.get("tool_calls") or row["role"] == "tool"])
        if pruned:
            for row in copies:
                if row["role"] == "tool":
                    row["content"] = _HERMES_PRUNED_TOOL_PLACEHOLDER
        copy_ids = set(_store_copies(engine, copies * 3))
        extra = _view(2)
        for i, row in enumerate(extra):
            row["message_uid"] = f"extra-{i}"
            row["content"] += " extra"
            row["timestamp"] += 1000
            if row.get("tool_calls"):
                row["tool_calls"][0]["id"] += "-extra"
            if row.get("tool_call_id"):
                row["tool_call_id"] += "-extra"
        engine.compress(original + extra, current_tokens=24000)
        assert summaries
        assert not copy_ids.intersection(_covered(engine))
        assert copy_ids <= {row["store_id"] for row in _rows(engine)}
        assert _frontier(engine) > max(copy_ids)
        assert copy_ids <= set(engine._store_complete_copy_exclusion_proofs)
        assert all(_HERMES_PRUNED_TOOL_PLACEHOLDER not in text for text in summaries)
    finally:
        engine.shutdown()


def test_2815_distinct_or_unbound_tool_rows_stay_in_summary(tmp_path, summaries):  # noqa: F811
    _state_db(tmp_path, [("S", None, None)])
    engine = _open(tmp_path, leaf_chunk_tokens=8000)
    try:
        original = _view(2)
        engine.ingest(original)
        original_ids = [row["store_id"] for row in _rows(engine)]
        _stage(engine, 0, original_ids)
        engine._last_compacted_store_id = original_ids[-1]
        changed = dict(original[2], content="Different tool version")
        # Keep an unbound source and its sole pruned replay; their identities
        # differ, so the bounded literal-copy heuristic cannot exclude either.
        unbound = dict(original[6], tool_call_id="unbound")
        pruned = dict(unbound, content=_HERMES_PRUNED_TOOL_PLACEHOLDER)
        ids = _store_copies(engine, [changed, unbound, pruned])
        tail = [{"role": "user", "content": "fresh question"},
                {"role": "assistant", "content": "fresh response"}]
        engine.compress(tail, current_tokens=24000)
        assert summaries
        assert set(ids) <= set(_covered(engine))
    finally:
        engine.shutdown()


def test_2815_fork_does_not_use_parent_uid_copy_map(tmp_path):
    _state_db(tmp_path, [("S", None, "compression"), ("fork", "S", None, "cli", '{"_branched_from":"S"}')])
    parent = _open(tmp_path)
    try:
        parent.ingest(_view(2))
    finally:
        parent.shutdown()
    child = _open(tmp_path, "fork")
    try:
        # A copied host view cannot borrow the parent's UID binding.
        view = _view(2)
        assert not child._get_store_id_map_for_messages(view)
        for row in view:
            child._store.append("fork", row, conversation_id="conv")
        known, _ = child._store_complete_copy_twins(_rows(child), child._load_compression_carry_ranges())
        assert not known
    finally:
        child.shutdown()
