"""Synthetic regressions for overlapping compaction and turn telemetry."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from threading import Event, get_ident

import pytest

from hermes_lcm import compaction, engine as engine_module
from tests.test_compaction_telemetry import _hot_usage
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _engine
from tests.test_issue_2851_rebind_cursor import _bind, _binding_state, _view
from tests.test_issue_2854_no_summary_pass_unchanged import published_head, user_row


def _open(path):
    engine = _engine(path, embeddings_enabled=False, temporal_rollups_enabled=False)
    engine._hermes_home = engine._store._hermes_home = str(path)
    return engine


@contextmanager
def _closing(engine):
    try:
        yield engine
    finally:
        engine.shutdown()


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

            with monkeypatch.context() as patch:
                patch.setattr(engine, "_bypasses_lcm_context_management", lambda: True)
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


def test_hold_status_survives_concurrent_clear(tmp_path, monkeypatch):
    with _closing(_open(tmp_path)) as engine:
        engine._no_progress_hold = (10**12, "hidden_only")

        def clear_after_check(*, managed_scope):
            engine._no_progress_hold = None
            return True

        monkeypatch.setattr(engine, "_no_progress_hold_active", clear_after_check)
        assert engine._no_progress_hold_status() is None


@pytest.mark.parametrize("where", ["summary-after-rebind", "unchanged-list"])
def test_main_exit_refuses_overlapping_call(tmp_path, monkeypatch, where):
    _state_db(tmp_path, [("S", None, None), ("B", None, None)])
    ready, release = Event(), Event()
    shortcut = where == "unchanged-list"

    def wait_for_refusal():
        if not ready.is_set():
            ready.set()
            assert release.wait(10)

    def summarize(**kwargs):
        if not shortcut:
            wait_for_refusal()
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    with _closing(_open(tmp_path)) as engine, ThreadPoolExecutor(max_workers=2) as workers:
        def snapshot():
            return {
                **_binding_state(engine),
                "rows": engine._store.connection.execute("SELECT * FROM messages ORDER BY store_id").fetchall(),
                "calls": engine._compress_calls,
                "estimates": deepcopy(vars(engine._foreground_estimates)),
            }

        view = _view(7)
        if shortcut:
            view = [*published_head(engine), user_row(0)]
            covers = engine._replayed_head_covers_frontier

            def covered(state):
                result = covers(state)
                assert result
                wait_for_refusal()
                return result

            monkeypatch.setattr(engine, "_replayed_head_covers_frontier", covered)
        primary = workers.submit(engine.compress, view, 4_000 if shortcut else 24_000)
        try:
            assert ready.wait(10)
            if not shortcut:
                _bind(engine, "B", "convB")
            second = [{"role": "user", "content": "Second caller's synthetic turn"}]
            before = snapshot()
            refused = workers.submit(engine.compress, second, 4_000).result(timeout=2)
            assert snapshot() == before, "overlapping call changed owner or store state"
            assert refused is second
            assert engine._compress_reentry_refusals == 1
        finally:
            release.set()
        assert primary.result(timeout=10) is view
        if shortcut:
            assert engine._last_compress_leaves[1] == 0
        else:
            assert engine._ingest_cursor_needs_reconcile


def test_refusal_returns_original_before_canonicalizing(tmp_path, monkeypatch):
    ready, release = Event(), Event()
    caller_messages = [{"role": "user", "content": "Synthetic lone surrogate \ud800"}]
    _state_db(tmp_path, [("S", None, None)])
    with _closing(_open(tmp_path)) as engine, ThreadPoolExecutor(max_workers=1) as worker:
        canonicalize = engine._canonical_host_view
        assert canonicalize(caller_messages)[0] is not caller_messages
        attempted = []

        def canonical(raw):
            if raw is caller_messages:
                attempted.append(raw)
            return canonicalize(raw)

        def summarize(**_kwargs):
            if not ready.is_set():
                ready.set()
                assert release.wait(10)
            return "Earlier turns.\nExpand for details about: turns", 1

        monkeypatch.setattr(engine, "_canonical_host_view", canonical)
        monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
        first = worker.submit(engine.compress, _view(6), 24_000)
        try:
            assert ready.wait(10)
            result = engine.compress(caller_messages, 100)
            assert not attempted, "the second caller reached host-list canonicalization"
            assert result is caller_messages
        finally:
            release.set()
        first.result(timeout=10)


class _Interrupted(BaseException):
    pass


@pytest.mark.parametrize("exit_name", [
    "normal", "canonical-error", "closing-at-entry", "closing-after-impl", "closing-before-proof",
    "closing-exception", "rebind-after-impl", "rebind-after-fit", "rebind-before-proof",
    "rebind-exception", "rebind-after-exception-fit", "rebind-interrupt",
    "fitted-exception", "locked-exception", "exception", "interrupt", "setup-error", "finalizer-error",
])
def test_owner_is_released_at_every_exit(tmp_path, monkeypatch, exit_name):
    view = [{"role": "user", "content": "First caller's synthetic turn"}]
    other_view = [{"role": "user", "content": "Other caller's synthetic turn"}]
    thread = get_ident()
    with _closing(_open(tmp_path)) as engine, ThreadPoolExecutor(max_workers=1) as worker:
        overlap = []
        visited = []
        host_view_of = engine._canonical_host_view
        setup = engine._new_foreground_budget
        cleanup = engine._end_pre_leaf_phase
        stop_for_shutdown = engine._stop_pass_for_shutdown
        closing_checks = 0
        shutdown_stops = []

        def probe():
            before = engine._compress_calls
            result = worker.submit(engine.compress, other_view, 100).result(timeout=2)
            overlap.append((result is other_view, engine._compress_calls - before))

        def canonicalize(raw):
            if raw is view:
                probe()
                if exit_name == "canonical-error":
                    raise RuntimeError(exit_name)
            return host_view_of(raw)

        def is_closing():
            nonlocal closing_checks
            if get_ident() != thread:
                return False
            closing_checks += 1
            close_at = {"closing-at-entry": 1, "closing-after-impl": 2,
                        "closing-before-proof": 3, "closing-exception": 2}
            return closing_checks == close_at.get(exit_name)

        def rebind():
            engine._pressure_yield_reset_epoch += 1

        class RebindOnLen(list):
            def __len__(self):
                rebind()
                return super().__len__()

        def compact(messages, **_kwargs):
            visited.append(get_ident())
            if get_ident() != thread:
                return messages
            if exit_name in {"rebind-after-impl", "rebind-exception", "rebind-interrupt"}:
                rebind()
            if exit_name in {"interrupt", "rebind-interrupt"}:
                raise _Interrupted(exit_name)
            if exit_name == "locked-exception":
                raise sqlite3.OperationalError("database is locked")
            if exit_name.endswith("exception") or exit_name == "rebind-after-exception-fit":
                raise RuntimeError(exit_name)
            if exit_name == "rebind-before-proof":
                engine._no_progress_candidate = True
            return messages

        fitted = [*view, {"role": "assistant", "content": "Synthetic fitted result"}]

        def fit(messages, result, *_args, after_exception=False, **_kwargs):
            if get_ident() != thread:
                return result
            if exit_name in {"rebind-after-fit", "rebind-after-exception-fit"}:
                rebind()
            if exit_name == "fitted-exception" and after_exception:
                return fitted
            if exit_name == "rebind-before-proof":
                return RebindOnLen(result)
            return result

        def make_budget():
            if get_ident() == thread and exit_name == "setup-error":
                raise RuntimeError(exit_name)
            return setup()

        def finish(*args):
            result = cleanup(*args)
            if get_ident() == thread and exit_name == "finalizer-error":
                raise RuntimeError(exit_name)
            return result

        def stop():
            shutdown_stops.append(get_ident())
            stop_for_shutdown()

        with monkeypatch.context() as patch:
            patch.setattr(engine, "_canonical_host_view", canonicalize)
            patch.setattr(engine, "_compaction_is_closing", is_closing)
            patch.setattr(engine, "_compress_impl", compact)
            patch.setattr(engine, "_survival_fit", fit)
            patch.setattr(engine, "_new_foreground_budget", make_budget)
            patch.setattr(engine, "_end_pre_leaf_phase", finish)
            patch.setattr(engine, "_retained_stop_hold_active", lambda: False)
            patch.setattr(engine, "_hold_fit_only_applies", lambda *_args: False)
            patch.setattr(engine, "_stop_pass_for_shutdown", stop)
            patch.setattr(engine, "_should_force_overflow_recovery", lambda **_: False)
            patch.setattr(engine, "_survival_fit_budget", lambda *_args: 10_000)
            if exit_name in {"interrupt", "rebind-interrupt", "exception", "setup-error", "finalizer-error"}:
                error = _Interrupted if exit_name.endswith("interrupt") else RuntimeError
                with pytest.raises(error, match=exit_name):
                    engine.compress(view, 100)
            else:
                expected = fitted if exit_name == "fitted-exception" else view
                assert engine.compress(view, 100) is expected
            if exit_name.startswith("closing-") and exit_name != "closing-at-entry":
                assert shutdown_stops == [thread]

            # Exercise a different thread again so same-thread recursion cannot mask a leak.
            before = len(visited)
            assert worker.submit(engine.compress, other_view, 100).result(timeout=2) is other_view
            assert len(visited) == before + 1, "call after the exit did not enter compaction"
            assert overlap == [(True, 0)], "another caller entered before the first pass exited"
            assert engine._compress_owner is None
            assert engine._compress_owner_depth == 0


def _managed_runs(engine, worker, visited):
    """A managed pass from another thread enters the owned pass instead of being refused."""
    view = [{"role": "user", "content": "Other caller's synthetic turn"}]
    before, refusals = len(visited), engine._compress_reentry_refusals
    assert worker.submit(engine.compress, view, 100).result(timeout=2) is view
    return len(visited) - before, engine._compress_reentry_refusals - refusals


def test_late_bypass_releases_the_claim(tmp_path, monkeypatch):
    thread = get_ident()
    view = [{"role": "user", "content": "Late auxiliary caller's synthetic turn"}]
    child = [{"role": "assistant", "content": "Synthetic child result"}]
    with _closing(_open(tmp_path)) as engine, ThreadPoolExecutor(max_workers=1) as worker:
        late, seen, visited = [], [], []

        def reclassify():
            if get_ident() == thread:
                seen.append(("claimed", engine._compress_owner, engine._compress_owner_depth))
                late.append(True)

        def bypassed(*_args):
            seen.append(("bypass", engine._compress_owner, engine._compress_owner_depth))
            return child

        def owned(messages, *_args, **_kwargs):
            visited.append(get_ident())
            return messages

        monkeypatch.setattr(engine, "_bypasses_lcm_context_management", lambda: bool(late) and get_ident() == thread)
        monkeypatch.setattr(engine, "_maybe_reclassify_late_auxiliary_before_compaction_write", reclassify)
        monkeypatch.setattr(engine, "_bypass_lcm_session_id", lambda: "child")
        monkeypatch.setattr(engine, "_compress_delegate_child_bypassed", bypassed)
        monkeypatch.setattr(engine, "_compress_owned", owned)
        assert engine.compress(view, 100) is child
        assert seen == [("claimed", thread, 1), ("bypass", None, 0)], "the late bypass kept the claim"
        assert not visited
        late.clear()
        assert _managed_runs(engine, worker, visited) == (1, 0), "another thread was refused after a late bypass"
        assert (engine._compress_owner, engine._compress_owner_depth) == (None, 0)


def test_nested_release_is_depth_counted(tmp_path, monkeypatch):
    thread = get_ident()
    outer = [{"role": "user", "content": "Outer synthetic turn"}]
    inner = [{"role": "user", "content": "Inner synthetic turn"}]
    with _closing(_open(tmp_path)) as engine, ThreadPoolExecutor(max_workers=1) as worker:
        visited, during = [], {}

        def owned(messages, *_args, **_kwargs):
            visited.append(get_ident())
            if messages == outer and get_ident() == thread:
                assert engine.compress(inner, 100) == inner
                during["claim"] = (engine._compress_owner, engine._compress_owner_depth)
                during["other"] = _managed_runs(engine, worker, visited)
            return messages

        monkeypatch.setattr(engine, "_compress_owned", owned)
        assert engine.compress(outer, 100) == outer
        assert during["claim"] == (thread, 1), "the inner return dropped the outer claim"
        assert during["other"] == (0, 1), "another thread ran while the outer call held the claim"
        assert (engine._compress_owner, engine._compress_owner_depth) == (None, 0)
        assert _managed_runs(engine, worker, visited) == (1, 0), "another thread was refused after both returned"
