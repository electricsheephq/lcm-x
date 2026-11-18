"""Synthetic regressions for the opt-in cut budget, scaffold drop and pass binding."""

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import _summary_part_text
from hermes_lcm.reconcile import _PRESERVED_TODO_CONTEXT_PREFIX
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _rows, _stage, _turn, summaries  # noqa: F401

_BUDGET_LOG = "LCM UID cut budget exhausted:"
_REBIND_LOG = "LCM UID drop skipped after rebind:"
_HEADER = _PRESERVED_TODO_CONTEXT_PREFIX
_ITEM = "- [ ] Synthetic item (pending)"
_ASK = "Keep the synthetic instruction."
_TASK_ROWS = {
    "same-line": f"{_HEADER} {_ASK}",
    "second-line": f"{_HEADER}\n{_ASK}",
    "single-newline": f"{_HEADER}\n{_ITEM}\n{_ASK}",
    "blank-line": f"{_HEADER}\n{_ITEM}\n\n{_ASK}",
    "task-list": f"{_HEADER}\n{_ITEM}",
    "embedded": f"Synthetic note.\n\n{_HEADER}\n{_ITEM}",
}
_FIRST_ROWS = {
    "assistant": {"role": "assistant", "message_uid": "b-0"},
    "user-unstamped": {"role": "user"},
    "user-stamped": {"role": "user", "timestamp": 9000.0, "message_uid": "b-0"},
}


@contextmanager
def _open(path):
    config = LCMConfig(database_path=str(path / "lcm.db"), fresh_tail_count=2, leaf_chunk_tokens=400,
                       context_threshold=0.001, threshold_full_sweep_enabled=False)
    engine = engine_module.LCMEngine(config=config, hermes_home=str(path))
    try:
        engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
        yield engine
    finally:
        engine.shutdown()


def _mode(monkeypatch, on):
    if on:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    else:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)


def _view(turns, tag="turn"):
    rows = [row for i in range(turns) for row in _turn(f"{tag}-{i}", 1000.0 + i * 10, tool=True)]
    for i, row in enumerate(rows):
        row.update(message_uid=f"{tag}-{i}", timestamp=1000.0 + i * 2)
        if row["role"] == "tool":
            row["content"] += " durable output" * 24
    return rows


def _contents(rows):
    return [(row["role"], row.get("content") or "") for row in rows]


def _cold(path, monkeypatch, *, on, bound, added, legacy=(), unlocated=False):
    path.mkdir()
    _state_db(path, [("S", None, None)])
    call, result = _turn("old", 900.0, tool=True)[1:3]
    if bound:
        call["message_uid"], result["message_uid"] = "old-call", "old-result"
    prefix = [call, result, *legacy]
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    with _open(path) as seed:
        if bound:
            seed.ingest(deepcopy(prefix[:2]))
        for row in prefix[2 if bound else 0:]:
            seed._store.append("S", dict(row), conversation_id="conv")
        if unlocated:
            seed.ingest([call, result, {"role": "assistant", "content": "Bound row outside the view",
                                       "message_uid": "unlocated"}])
        seeded = _contents(_rows(seed))
    _mode(monkeypatch, on)
    with _open(path) as cold, monkeypatch.context() as patch:
        cuts = []
        project = cold._project_replay_candidate

        def count(run, candidate):
            cuts.append(candidate.cursor)
            return project(run, candidate)

        patch.setattr(cold, "_project_replay_candidate", count)
        cold.ingest(deepcopy([*prefix, *added]))
        return seeded, _contents(_rows(cold)), cuts


@pytest.mark.parametrize("new_count", [8, 64])
def test_unbound_tool_pair_uses_full_cold_walk(tmp_path, monkeypatch, caplog, new_count):
    caplog.set_level(logging.INFO, logger="hermes_lcm.reconcile")
    added = _view(new_count // 4, "new")
    seed, stored, cuts = _cold(tmp_path / "on", monkeypatch, on=True, bound=False, added=added)
    _, base, base_cuts = _cold(tmp_path / "unset", monkeypatch, on=False, bound=False, added=added)
    assert stored == base == seed + _contents(added)
    assert cuts == base_cuts == list(range(new_count + 2, 1, -1))
    assert _BUDGET_LOG not in caplog.text


def test_bound_prefix_with_200_unbound_rows_uses_full_cold_walk(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_lcm.reconcile")
    added = _view(50, "new")
    seed, stored, cuts = _cold(tmp_path / "on", monkeypatch, on=True, bound=True, added=added, unlocated=True)
    assert len(seed) == 3
    assert stored == seed + _contents(added)
    _, base, base_cuts = _cold(tmp_path / "unset", monkeypatch, on=False, bound=True, added=added, unlocated=True)
    assert stored == base
    assert cuts == base_cuts and len(cuts) > 64
    assert "LCM UID cut budget disabled: unbound_host_rows=137" in caplog.text


@pytest.mark.parametrize("shape", _TASK_ROWS)
@pytest.mark.parametrize("reply_after", [False, True], ids=["last", "reply-after"])
def test_host_task_row_survives_publishing_pass(tmp_path, monkeypatch, summaries, shape, reply_after):  # noqa: F811
    host_row = {"role": "user", "content": _TASK_ROWS[shape], "timestamp": 5000.0}
    view = _view(7) + [host_row]
    if reply_after:
        view.append({"role": "assistant", "content": "Synthetic reply", "message_uid": "after", "timestamp": 5001.0})
    outputs = []
    for on in (False, True):
        path = tmp_path / str(on)
        path.mkdir()
        _mode(monkeypatch, on)
        _state_db(path, [("S", None, None)])
        with _open(path) as engine:
            engine.ingest(deepcopy(view))
            assert _contents(_rows(engine)).count(("user", host_row["content"])) == 1
            output = engine.compress(deepcopy(view), current_tokens=24000)
            assert len(engine._dag.get_session_nodes("S")) == 1
            outputs.append(output)
    base, on = outputs
    for result in (base, on):
        assert sum(host_row["content"] in (row.get("content") or "") for row in result) == 1
    assert len(on) == len(base) == 3


@pytest.mark.parametrize("shape", _TASK_ROWS)
def test_scaffold_drop_keeps_host_task_row_before_fresh_tail(tmp_path, monkeypatch, shape):
    _mode(monkeypatch, True)
    _state_db(tmp_path, [("S", None, None)])
    host = {"role": "user", "content": _TASK_ROWS[shape]}
    view = [host, *_turn("fresh", 5000.0)]
    with _open(tmp_path) as engine:
        assert engine._fresh_tail_start(view) == 1
        assert engine._drop_uid_covered_live_rows(view) is view


def test_scaffold_drop_keeps_verified_head_inside_fresh_tail(tmp_path, monkeypatch):
    _mode(monkeypatch, True)
    _state_db(tmp_path, [("S", None, None)])
    with _open(tmp_path) as engine:
        engine.ingest(_view(1))
        _stage(engine, 0, [row["store_id"] for row in _rows(engine)])
        head = {"role": "user", "content": _summary_part_text(engine._dag.get_node(1))}
        older, fresh = dict(head), dict(head)
        reply = {"role": "assistant", "content": "Fresh synthetic reply"}
        view = [older, fresh, reply]
        assert engine._fresh_tail_start(view) == 1
        kept = engine._drop_uid_covered_live_rows(view)
        assert len(kept) == 2 and kept[0] is fresh and kept[1] is reply


def _compaction_rebind(path, monkeypatch, *, on, first, anchor, seam):
    path.mkdir()
    _mode(monkeypatch, on)
    _state_db(path, [("S", None, None), ("B", None, None)])
    view = _view(7)
    if seam == "summary":
        for row in view:
            row.pop("message_uid")
    with _open(path) as engine, monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as pool:
        engine.ingest(deepcopy(view))
        covered = [row["store_id"] for row in _rows(engine)[:8]]
        _stage(engine, 0, covered)
        engine._last_compacted_store_id = covered[-1]
        for row in view:
            engine._store.append("S", dict(row), conversation_id="conv")
        expected_nodes = len(engine._dag.get_session_nodes("S"))
        observed = []

        def rebind():
            if not observed:
                observed.append(len(engine._dag.get_session_nodes("S")) - expected_nodes)
                future = pool.submit(engine.on_session_start, "B", platform="telegram",
                                     context_length=200_000, conversation_id="convB")
                future.result(timeout=5)

        def summarize(**_kwargs):
            if seam == "summary":
                rebind()
            return "Earlier turns.\nExpand for details about: turns", 1

        def compress_impl(*args, **kwargs):
            result = original(*args, **kwargs)
            if seam == "after-leaf":
                rebind()
            return result

        original = engine._compress_impl
        patch.setattr(engine, "_compress_impl", compress_impl)
        patch.setattr(engine_module, "summarize_with_escalation", summarize)
        output = engine.compress(deepcopy(view), current_tokens=24000)
        if seam == "after-leaf":
            summary = (
                "[Recent Summary (d0, node 1)]\ns\n[Expand for details: ]\n\n---\n\n"
                "[Recent Summary (d0, node 2)]\nEarlier turns.\nExpand for details about: turns\n"
                "[Expand for details: turns]"
            )
            if on:
                objective = f"[Current user objective preserved from compacted history]\n{view[24]['content']}"
                assert output == [{"role": "user", "content": f"{objective}\n\n---\n\n{summary}"}, *view[25:]]
            else:
                assert output == [{"role": "user", "content": summary}, *view[5:]]
        before_rebind = (engine._ingest_cursor, len(output), engine._last_compression_status)
        if seam == "none":
            rebind()
        assert observed == [0 if seam == "summary" else 1]
        assert engine._session_id == "B"
        cursor = engine._ingest_cursor
        if anchor == "off":
            patch.setenv("LCM_IDENTITY_ANCHOR", "off")
        row = dict(_FIRST_ROWS[first], content="Session B first synthetic row")
        engine.ingest([row])
        return cursor, _contents(_rows(engine)), before_rebind


@pytest.mark.parametrize("first", _FIRST_ROWS)
@pytest.mark.parametrize("anchor", ["default", "off"])
def test_published_pass_skips_uid_guard_after_rebind(tmp_path, monkeypatch, caplog, first, anchor):
    caplog.set_level(logging.INFO, logger="hermes_lcm.store_complete")
    base = _compaction_rebind(tmp_path / "unset", monkeypatch, on=False, first=first, anchor=anchor,
                             seam="after-leaf")
    candidate = _compaction_rebind(tmp_path / "on", monkeypatch, on=True, first=first, anchor=anchor,
                                  seam="after-leaf")
    assert candidate[:2] == base[:2] == (0, [(_FIRST_ROWS[first]["role"], "Session B first synthetic row")])
    assert [r.message for r in caplog.records if r.message.startswith(_REBIND_LOG)] == [
        f"{_REBIND_LOG} input_rows=28"
    ]


def test_published_pass_without_midpass_rebind_keeps_uid_drop(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_lcm.store_complete")
    cursor, stored, after_pass = _compaction_rebind(tmp_path / "on", monkeypatch, on=True, first="user-unstamped",
                                                   anchor="default", seam="none")
    assert (cursor, stored, after_pass) == (0, [("user", "Session B first synthetic row")], (4, 4, "compacted"))
    assert _REBIND_LOG not in caplog.text


@pytest.mark.parametrize("first", _FIRST_ROWS)
@pytest.mark.parametrize("anchor", ["default", "off"])
def test_summary_call_rebind_keeps_base_publication_fallback(tmp_path, monkeypatch, first, anchor):
    base = _compaction_rebind(tmp_path / "unset", monkeypatch, on=False, first=first, anchor=anchor,
                             seam="summary")
    candidate = _compaction_rebind(tmp_path / "on", monkeypatch, on=True, first=first, anchor=anchor,
                                  seam="summary")
    print(f"summary rebind: {first=} {anchor=} unset={base} on={candidate}")
    assert candidate == base
    assert base[2] == (28, 28, "error")
