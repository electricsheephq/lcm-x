"""Synthetic regressions from the c10 review of the #2815 UID path."""

import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from threading import Event

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm import host_uid_emit
from hermes_lcm.config import LCMConfig
from hermes_lcm.store_complete import _UidDropState
from tests.invariants.test_issue_2815_c9 import _contents, _mode, _open, _view
from tests.invariants.test_issue_2815_c10 import _BUSY, _IDLE, _gate_state
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _frontier, _rows, _stage, _turn, summaries  # noqa: F401

_TASK_HEADER = "[Your active task list was preserved across context compression]"
_TASK = f"{_TASK_HEADER}\n- [ ] 1. Synthetic pending item (pending)"
_BIG = " ".join(f"line{i:05d} payload-{i * 7 % 1013}" for i in range(2600))


@contextmanager
def _managed_pass(tmp_path, monkeypatch):
    ready, release = Event(), Event()
    _state_db(tmp_path, [("S", None, None)])
    with _open(tmp_path) as engine, ThreadPoolExecutor(max_workers=1) as threads:
        def summarize(**_kwargs):
            ready.set()
            assert release.wait(10)
            return "Earlier turns.\nExpand for details about: turns", 1

        monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
        view = _view(6)
        future = threads.submit(engine.compress, view, 24000)

        def finish():
            release.set()
            assert len(future.result(timeout=10)) < len(view)

        try:
            assert ready.wait(10)
            yield engine, finish
        finally:
            finish()


@pytest.mark.parametrize("on", [False, True], ids=["unset", "on"])
def test_child_gate_during_managed_pass(tmp_path, monkeypatch, on):
    _mode(monkeypatch, on)
    with _managed_pass(tmp_path, monkeypatch) as (engine, _):
        engine._mark_thread_context_stateless("child")
        try:
            assert engine._bypasses_lcm_context_management()
            assert _gate_state(engine) == _IDLE
        finally:
            engine._clear_thread_context_stateless("child")


@pytest.mark.parametrize("on", [False, True], ids=["unset", "on"])
def test_split_gate_read_after_owner_finishes(tmp_path, monkeypatch, on):
    _mode(monkeypatch, on)
    with _managed_pass(tmp_path, monkeypatch) as (engine, finish), ThreadPoolExecutor(max_workers=1) as reader:
        # A second reader leaves a marker pending so the test also checks thread isolation and reset.
        assert reader.submit(engine._automatic_compression_blocked).result(timeout=10)

        def blocked_then_finish():
            blocked = engine._automatic_compression_blocked(ignore_cooldown=True)
            finish()
            return blocked

        assert blocked_then_finish()
        assert engine._compress_owner is None
        assert engine._compression_block_reason() == _BUSY
        assert engine._compression_block_reason() is None

        def reevaluate():
            assert not engine._automatic_compression_blocked()
            return engine._compression_block_reason()

        assert reader.submit(reevaluate).result(timeout=10) is None


def _emitted_nodes(engine, output):
    return {
        node
        for row in output
        if row.get("role") in {"user", "assistant"} and isinstance(content := row.get("content"), str)
        for header in engine._LCM_SUMMARY_PART_HEADER_RE.finditer(content)
        for node in engine._verified_lcm_summary_prefix(content[header.start():])[1]
    }


def _emitted_sources(engine, output):
    return {sid for node in _emitted_nodes(engine, output)
            for sid in engine._dag.source_message_ids(node, limit=2**63 - 1)}


def test_no_leaf_restore_updates_cursor_during_lifecycle_contention(tmp_path, monkeypatch):
    # Exercise cursor-based replay without the identity anchor's recovery of missed rows.
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "off")
    # Cleanup drops these orphan results, so a stale cursor can hide the next append.
    view = _view(7) + [
        {"role": "tool", "tool_call_id": f"orphan-{i}", "message_uid": f"orphan-{i}",
         "content": "Synthetic orphan result", "timestamp": 2000.0 + i}
        for i in range(2)
    ]
    added = {"role": "user", "message_uid": "new", "content": "New synthetic user turn", "timestamp": 3000.0}

    def run(on):
        _mode(monkeypatch, on)
        path = tmp_path / ("on" if on else "unset")
        path.mkdir()
        _state_db(path, [("S", None, None)])
        with _open(path) as engine, ThreadPoolExecutor(max_workers=1) as threads:
            engine.ingest(deepcopy(view))
            covered = _rows(engine)[0]["store_id"]
            _stage(engine, 0, [covered])
            engine._last_compacted_store_id = covered
            engine._config.threshold_full_sweep_enabled = True
            before = engine._dag.get_session_nodes("S")
            binding = engine._uid_recognition_binding()
            ready, release = Event(), Event()
            removed = set()
            drop = engine._drop_uid_covered_live_rows
            condense = engine._condense_before_leaves

            def expire_before_first_leaf(state, focus_topic):
                result = condense(state, focus_topic)
                state.sweep_deadline = 0.0
                return result

            def observe_drop(rows):
                trimmed = drop(rows)
                removed.update(row["message_uid"] for row in rows if row not in trimmed)
                return trimmed

            def hold_lifecycle():
                with engine._exclusive_lifecycle("test contention") as acquired:
                    assert acquired
                    ready.set()
                    assert release.wait(10)

            monkeypatch.setattr(engine, "_drop_uid_covered_live_rows", observe_drop)
            monkeypatch.setattr(engine, "_condense_before_leaves", expire_before_first_leaf)
            holder = threads.submit(hold_lifecycle)
            try:
                assert ready.wait(10)
                output = engine.compress(deepcopy(view), current_tokens=24000)
            finally:
                release.set()
            holder.result(timeout=10)

            assert engine._uid_recognition_binding() == binding
            assert engine._dag.get_session_nodes("S") == before
            assert engine.last_compression_status == "sanitized"
            assert engine._last_threshold_full_sweep["stop_reason"] == "time_budget_exhausted"
            assert removed == ({view[0]["message_uid"]} if on else set())
            assert len(output) == len(view) - 2
            cursor = engine._ingest_cursor
            stored = _rows(engine)
            engine.ingest(deepcopy(output + [added]))
            after = _rows(engine)
            assert after[:len(stored)] == stored
            return output, cursor, _contents(after[len(stored):])

    unset, _, _ = run(False)
    output, cursor, appended = run(True)
    assert output == unset
    assert (cursor, appended) == (len(output), _contents([added]))


@pytest.mark.parametrize("seam", ["guard-after-check", "outer-fit"])
def test_restore_then_rebind_keeps_next_session_cursor(tmp_path, monkeypatch, summaries, seam):  # noqa: F811
    _mode(monkeypatch, True)
    _state_db(tmp_path, [("S", None, None), ("B", None, None)])
    with _open(tmp_path) as engine, ThreadPoolExecutor(max_workers=1) as threads:
        view = _view(7)
        engine._config.sensitive_patterns_enabled = True
        engine.ingest(deepcopy(view))
        covered = _rows(engine)[0]["store_id"]
        _stage(engine, 0, [covered])
        engine._last_compacted_store_id = covered
        restored, rebound = Event(), Event()
        guard, matches, fit = engine._guard_uid_covered_drop, _UidDropState.matches_binding, engine._survival_fit
        fault_removed = []

        def bind():
            engine.on_session_start("B", platform="telegram", context_length=200_000, conversation_id="convB")
            assert engine._ingest_cursor == 0
            rebound.set()

        def rebind():
            if not rebound.is_set():
                threads.submit(bind).result(timeout=10)
                assert rebound.is_set()

        def faulted_guard(result):
            kept = [row for row in result if covered not in _emitted_sources(engine, [row])]
            fault_removed.append(len(result) - len(kept))
            return guard(kept)

        def matches_binding(scope):
            matched = matches(scope)
            caller = sys._getframe(1)
            if matched and caller.f_code.co_name == "_guard_uid_covered_drop" and "restored" in caller.f_locals:
                restored.set()
                if seam == "guard-after-check":
                    rebind()
            return matched

        def survival_fit(*args, **kwargs):
            if seam == "outer-fit" and restored.is_set():
                rebind()
            return fit(*args, **kwargs)

        monkeypatch.setattr(engine, "_guard_uid_covered_drop", faulted_guard)
        monkeypatch.setattr(_UidDropState, "matches_binding", matches_binding)
        monkeypatch.setattr(engine, "_survival_fit", survival_fit)
        engine.compress(deepcopy(view), current_tokens=24000)
        assert sum(fault_removed) and restored.is_set() and rebound.is_set() and engine._session_id == "B"
        cursor = engine._ingest_cursor
        first = {"role": "assistant", "message_uid": "b-0", "content": "Session B first synthetic row"}
        engine.ingest([first])
        assert (cursor, _contents(_rows(engine))) == (0, _contents([first]))


def _cold_cuts(engine, monkeypatch, view):
    cuts = []
    project = engine._project_replay_candidate

    def count(run, candidate):
        cuts.append(candidate.cursor)
        return project(run, candidate)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_project_replay_candidate", count)
        engine.ingest(deepcopy(view))
    return cuts


@pytest.mark.parametrize("seed_direct", [True, False], ids=["direct-bindings", "ingest-ledger"])
def test_bound_composite_before_64_new_rows(tmp_path, monkeypatch, seed_direct):
    _mode(monkeypatch, True)
    _state_db(tmp_path, [("S", None, None)])
    user = {"role": "user", "content": "question", "message_uid": "u"}
    assistant = {"role": "assistant", "content": "first\nsecond", "message_uid": "a"}
    prefix = [user, assistant]
    with _open(tmp_path) as seed:
        if seed_direct:
            ids = [seed._store.append("S", row, conversation_id="conv") for row in prefix]
            lineage = seed._host_uid_lineage_key()[0]
            seed._store.add_host_uid_bindings(lineage, [(sid, row["message_uid"], "canonical", "stored_new")
                                                       for sid, row in zip(ids, prefix)])
        else:
            seed.ingest(deepcopy(prefix))
        assert _contents(_rows(seed)) == _contents(prefix)
    added = _view(16, "new")
    with _open(tmp_path) as cold:
        view = [user, {**assistant, "_absorbed_message_uids": ["b"]}, *added]
        cuts = _cold_cuts(cold, monkeypatch, view)
        assert _contents(_rows(cold)) == _contents(prefix + added)
        assert cuts == list(range(66, 1, -1))


@pytest.mark.parametrize("on", [True, False], ids=["on", "unset"])
def test_covered_uid_keeps_task_annotation(tmp_path, monkeypatch, summaries, on):  # noqa: F811
    _mode(monkeypatch, on)
    _state_db(tmp_path, [("S", None, None)])
    view = _view(7)
    sentinel = "Synthetic rollback task text"
    with _open(tmp_path) as engine:
        engine.ingest(deepcopy(view))
        covered = [row["store_id"] for row in _rows(engine)[:4]]
        _stage(engine, 0, covered)
        engine._last_compacted_store_id = covered[-1]
        view[0]["content"] += f"\n\n{_TASK_HEADER} {sentinel}"
        # Break committed-replay adoption so base sends the annotated live row to the summarizer.
        view[3] = {"role": "assistant", "content": "a different synthetic reply" + " pad" * 40}
        output = engine.compress(deepcopy(view), current_tokens=24000)
        assert covered[0] in _emitted_sources(engine, output)
        assert any(sentinel in text for _, text in _contents(output)) or any(sentinel in text for text in summaries)


def _task_snapshot(tmp_path, monkeypatch, on, system):
    _mode(monkeypatch, on)
    _state_db(tmp_path, [("S", None, None)])
    head = [{"role": "system", "content": "Synthetic system prompt."}] if system else []
    covered, new = _view(2, "cov"), _view(5, "new")
    for index, row in enumerate(new):
        row["timestamp"] = 1100.0 + index * 2
    task = {"role": "user", "content": _TASK, "message_uid": "task-snap", "timestamp": 1020.0}
    return head + covered, head + covered + [task] + new


def _cover_task_prefix(engine):
    covered = [r["store_id"] for r in _rows(engine) if r["role"] != "system"][:8]
    assert len(covered) == 8
    leafed = {sid for node in engine._dag.get_session_nodes("S") for sid in node.source_ids}
    rest = [r["store_id"] for r in _rows(engine) if r["store_id"] <= covered[-1] and r["store_id"] not in leafed]
    if rest:
        _stage(engine, _frontier(engine), rest)
    engine._last_compacted_store_id = covered[-1]
    return covered[-1]


def _assert_task_kept(engine, result, task_id):
    assert any(_TASK in text for _, text in _contents(result)) or task_id in _emitted_sources(engine, result)


@pytest.mark.parametrize("on", [True, False], ids=["on", "unset"])
@pytest.mark.parametrize("state", ["live-no-proof", "restart-no-proof", "live-proof", "restart-proof"])
@pytest.mark.parametrize("system", [False, True], ids=["nosys", "sys"])
def test_uid_trim_keeps_task_snapshot(tmp_path, monkeypatch, summaries, on, state, system):  # noqa: F811
    head, view = _task_snapshot(tmp_path, monkeypatch, on, system)

    def check(engine):
        task_ids = [r["store_id"] for r in _rows(engine) if r["content"] == _TASK]
        assert len(task_ids) <= 1 and _frontier(engine) == frontier
        assert not task_ids or task_ids[0] > frontier
        before = {node.node_id for node in engine._dag.get_session_nodes("S")}
        summaries.clear()
        result = engine.compress(deepcopy(view), current_tokens=24000)
        published = {node.node_id for node in engine._dag.get_session_nodes("S")} - before
        assert published
        if (published & _emitted_nodes(engine, result)) and any(_TASK in text for text in summaries):
            return
        _assert_task_kept(engine, result, task_ids[0] if task_ids else None)

    proof = not state.endswith("no-proof")
    with _open(tmp_path) as seed:
        seed.ingest(deepcopy(head if proof else view))
        if proof:
            seed.compress(deepcopy(head), current_tokens=24000)
            assert seed._last_emission_descriptors
        frontier = _cover_task_prefix(seed)
        if proof:
            seed.ingest(deepcopy(view))
        if state.startswith("live"):
            check(seed)
    if state.startswith("restart"):
        with _open(tmp_path) as engine:
            check(engine)


@pytest.mark.parametrize("on", [True, False], ids=["on", "unset"])
def test_uid_trim_keeps_task_snapshot_after_followup_passes(tmp_path, monkeypatch, summaries, on):  # noqa: F811
    _, view = _task_snapshot(tmp_path, monkeypatch, on, False)
    with _open(tmp_path) as engine:
        engine.ingest(deepcopy(view))
        _cover_task_prefix(engine)
        task_id = next(r["store_id"] for r in _rows(engine) if r["content"] == _TASK)
        result = engine.compress(deepcopy(view), current_tokens=24000)
        for i in range(2):
            added = _view(4, f"more{i}")
            for index, row in enumerate(added):
                row["timestamp"] = 2000.0 + 100 * i + index * 2
            result = result + added
            engine.ingest(deepcopy(result))
            result = engine.compress(deepcopy(result), current_tokens=24000)
        _assert_task_kept(engine, result, task_id)


@contextmanager
def _fit_store(tmp_path, monkeypatch, *, externalize, shape, direct, bind_copies=False):
    _mode(monkeypatch, True)
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", False)
    _state_db(tmp_path, [("S", None, None)])
    settings = LCMConfig(database_path=str(tmp_path / "lcm.db"), fresh_tail_count=2 if direct else 24,
                         leaf_chunk_tokens=400, context_threshold=0.001 if direct else 0.5,
                         threshold_full_sweep_enabled=False, large_output_externalization_enabled=externalize)
    engine = engine_module.LCMEngine(config=settings, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("S", platform="telegram", context_length=4096, conversation_id="conv")
        if shape == "tool":
            newest = _turn("T", 5000.0, tool=True)[:3]
            newest[0]["content"], newest[2]["content"] = "[T] short prompt", f"result of T {_BIG}"
        else:
            newest = [{"role": "user", "content": f"[T] pasted log {_BIG}", "timestamp": 5000.0},
                      {"role": "assistant", "content": "[T] short reply"}]
        for i, row in enumerate(newest):
            row["message_uid"] = f"uid-T-{i}"
        older = [] if direct else [
            dict(row, message_uid=f"old-{i}-{j}")
            for i in range(6) for j, row in enumerate(_turn(f"old-{i}", 100.0 + i))
        ]
        view = older + newest
        engine.ingest(deepcopy(view))
        originals = [r["store_id"] for r in _rows(engine)]
        assert len(originals) == len(view)
        copies = [engine._store.append("S", {k: deepcopy(v) for k, v in row.items() if k != "message_uid"},
                                       conversation_id="conv") for row in newest]
        if bind_copies:
            lineage = engine._host_uid_lineage_key()[0]
            engine._store.add_host_uid_bindings(lineage, [(sid, row["message_uid"], "version", "probe")
                                                         for sid, row in zip(copies, newest)])
        _stage(engine, 0, originals)
        engine._last_compacted_store_id = originals[-1]
        assert _frontier(engine) == originals[-1] < min(copies)
        yield engine, view, originals
    finally:
        engine.shutdown()


def _assert_fitted_original(engine, result, live, original, budget, *, externalized=False):
    after = engine._survival_measure(result)
    assert after <= budget, (after, budget)
    projected = next(row for row in reversed(result) if row["role"] == live["role"])
    assert projected["content"] != live["content"]
    if externalized:
        # The durable payload pointer is short enough to return verbatim.
        assert projected["content"] == engine._store.get(original)["content"]
        return
    source = engine._survival_projection_source(projected, live["role"], projected["content"])
    assert source and source["store_id"] == original


@pytest.mark.parametrize("externalize", [False, True], ids=["ext-off", "ext-on"])
@pytest.mark.parametrize("bind_copies", [False, True], ids=["copies-unbound", "copies-bound"])
def test_uid_original_is_direct_survival_projection_source(tmp_path, monkeypatch, externalize, bind_copies):
    with _fit_store(tmp_path, monkeypatch, externalize=externalize, shape="tool",
                    direct=True, bind_copies=bind_copies) as (engine, view, ids):
        budget = engine._survival_fit_budget(view, 0, None, 2048)
        assert engine._survival_measure(view) > budget
        assert engine._host_uid_resolve(view)[len(view) - 1]["store_id"] == ids[-1]
        result = engine._survival_fit(view, view, 0, "recovery_attempt:probe", request_cap=2048)
        _assert_fitted_original(engine, result, view[-1], ids[-1], budget, externalized=externalize)


@pytest.mark.parametrize(("shape", "passes"), [("tool", 1), ("user", 1), ("tool", 2)])
def test_uid_original_is_recovery_projection_source(tmp_path, monkeypatch, summaries, shape, passes):  # noqa: F811
    with _fit_store(tmp_path, monkeypatch, externalize=shape == "user", shape=shape,
                    direct=False) as (engine, view, ids):
        target = -1 if shape == "tool" else -2
        stored = _contents(_rows(engine))
        for _ in range(passes):
            observed = engine._survival_measure(view)
            caps = engine._survival_fit_args(view, observed, "probe", True, automatic=False)
            budget = engine._survival_fit_budget(view, observed, caps.get("window_cap"), caps.get("request_cap"))
            result = engine.compress(deepcopy(view), current_tokens=observed, bypass_cooldown=True)
            _assert_fitted_original(engine, result, view[target], ids[target], budget)
            assert _contents(_rows(engine)) == stored


def test_anchor_off_uid_cold_cursor_uses_full_walk(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "off")
    older = _view(2, "old")
    older[-1] = {**older[-1], "content": "unstamped composite\nsecond part", "_absorbed_message_uids": ["extra"]}
    older[-1].pop("timestamp")
    previous = _view(20, "new")
    added = [{**row, "content": f"Updated payload {i}"} for i, row in enumerate(previous)]
    view = older + added
    outcomes = []
    for on in (False, True):
        path = tmp_path / str(on)
        path.mkdir()
        _state_db(path, [("S", None, None)])
        _mode(monkeypatch, on)
        with _open(path) as seed:
            seed.ingest(deepcopy(previous + older))
            seed._store.append("S", {"role": "assistant", "content": "Newest stored row outside the view"},
                               conversation_id="conv")
        with _open(path) as cold:
            outcomes.append(_cold_cuts(cold, monkeypatch, view))
    base, candidate = outcomes
    assert len(base) > 64
    assert candidate == base
