"""Synthetic regressions from the c10 review of the #2815 UID path."""

# Items C, E and F of round c12 are not on this branch.

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from threading import Event

import pytest

from hermes_lcm import engine as engine_module
from tests.invariants.test_issue_2815_c9 import _contents, _mode, _open, _view
from tests.invariants.test_issue_2815_c10 import _BUSY, _IDLE, _gate_state
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _rows


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
