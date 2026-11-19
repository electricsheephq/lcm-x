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
        # The worker clears its pending reason before the main reader consumes its own.
        assert reader.submit(engine._automatic_compression_blocked).result(timeout=10)

        def blocked_then_finish():
            blocked = engine._automatic_compression_blocked(ignore_cooldown=True)
            finish()
            return blocked

        assert blocked_then_finish()
        assert engine._compress_owner is None
        def reevaluate():
            assert not engine._automatic_compression_blocked()
            return engine._compression_block_reason()

        assert reader.submit(reevaluate).result(timeout=10) is None
        assert engine._compression_block_reason() == _BUSY
        assert engine._compression_block_reason() is None


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
