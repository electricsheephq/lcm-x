"""A bypassed child must leave its shared engine's foreground compress state alone."""

import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest

from hermes_lcm import message_patterns
from tests.harness import ScriptedSummarizer, assistant, engine_context, pin_char_counter, prose, user


# Audited child-path writes, plus the commit proof cleared before the bypass guard.
CHILD_WRITES = """
_compress_dropped_ignored_rows _compress_forced_overflow _compress_hidden_rows_consumed
_compress_host_rows_consumed _compress_occurrences _foreground_budget _hold_fit_only_requested
_host_fallback_compressor _host_fallback_session_id _ingest_cursor _last_compress_aborted
_last_compress_leaves _last_compression_noop_reason _last_compression_status _last_hidden_backlog
_last_stub_first_exit _no_progress_candidate _objective_only_noop _pending_emission_candidates
_preflight_automatic_request _preflight_below_threshold_cleanup_only
_preflight_cleanup_only_due_to_boundary_cooldown _pressure_yield_blocked_streak
_pressure_yield_invocation_verdict _pressure_yield_preflight_candidate _pressure_yield_streak_counted
_pressure_yield_tail_token_limit _stub_first_exit_now _survival_fit_emission _survival_fit_reason
_compress_commit_proof _current_compress_store_ids_by_message_id
""".split()


class ParentStateChanged(AssertionError):
    """Identify changes to the parent state captured around the child call."""


def _snapshot(engine):
    # Retain budget identity AND copy its fields, so in-place mutations are visible.
    snapshot = {name: (getattr(engine, name, None) if name in {
        "_foreground_budget", "_host_fallback_compressor"
    } else deepcopy(getattr(engine, name, None))) for name in CHILD_WRITES}
    budget = snapshot["_foreground_budget"]
    if budget is not None:
        snapshot["_foreground_budget"] = (
            budget, deepcopy({k: v for k, v in vars(budget).items() if k != "estimates"}),
            deepcopy(vars(budget.estimates)),
        )
    return snapshot


def _history():
    return [row for i in range(6) for row in (
        {**user(prose(800, seed=i)), "timestamp": float(2 * i + 1)},
        {**assistant(prose(800, seed=i + 10)), "timestamp": float(2 * i + 2)},
    )]


@pytest.mark.parametrize("schedule", ["sequential", "interleaved"])
@pytest.mark.parametrize("group", ["state", "proof", "status", "control"])
def test_child_preserves_parent_compress_state(tmp_path, monkeypatch, schedule, group):
    pin_char_counter(monkeypatch)
    summaries = ScriptedSummarizer().install(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="parent", conversation_id="parent",
                        fresh_tail_count=2, leaf_chunk_tokens=400,
                        large_output_externalization_enabled=False) as engine:
        # The native fallback is an external provider boundary; use its real deterministic trim.
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
        view = _history()
        snapshots = []

        def child():
            engine._mark_thread_context_stateless("child")
            try:
                assert engine._bypasses_lcm_context_management()
                result = engine.compress([user("child request"), assistant("child reply")], force=True)
                assert result and engine.current_session_id == "parent"
                assert engine._store.get_session_count("child") == 0
            finally:
                engine._clear_thread_context_stateless("child")

        if schedule == "sequential":
            result = engine.compress(view, force=True)
            snapshots.append(_snapshot(engine))
            if group != "control":
                child()
            snapshots.append(_snapshot(engine))
        else:
            ready, finished = Barrier(2, timeout=10), Barrier(2, timeout=10)
            record_proof = engine._record_compress_commit_proof

            def pause_parent(messages, result):
                record_proof(messages, result)
                if not engine._bypasses_lcm_context_management():
                    snapshots.append(_snapshot(engine))
                    ready.wait()
                    finished.wait()
                    snapshots.append(_snapshot(engine))

            monkeypatch.setattr(engine, "_record_compress_commit_proof", pause_parent)

            def run_child():
                ready.wait()
                try:
                    if group != "control":
                        child()
                finally:
                    finished.wait()

            with ThreadPoolExecutor(max_workers=2) as pool:
                parent = pool.submit(engine.compress, view, force=True)
                delegated = pool.submit(run_child)
                delegated.result(timeout=15)
                result = parent.result(timeout=15)

        assert summaries.calls and len(result) < len(view)
        before, after = snapshots
        assert before["_compress_commit_proof"] is not None
        assert before["_last_compression_status"] == "compacted"
        if schedule == "interleaved":
            assert before["_foreground_budget"] is not None
        names = (set(CHILD_WRITES) if group == "control" else
                 {"_compress_commit_proof"} if group == "proof" else
                 {"_last_compression_status", "_last_compression_noop_reason"} if group == "status" else
                 set(CHILD_WRITES) - {"_compress_commit_proof", "_last_compression_status",
                                      "_last_compression_noop_reason"})
        changed = {name: (before[name], after[name]) for name in names if before[name] != after[name]}
        if changed:
            raise ParentStateChanged(changed)


@pytest.mark.parametrize("host_fails", [False, True])
def test_interleaved_child_stop_reason_and_native_state(tmp_path, monkeypatch, caplog, host_fails):
    """The native path uses a child cache and clock while a parent's proof is live."""
    import logging
    import sys
    import threading
    import time
    from types import ModuleType

    pin_char_counter(monkeypatch)
    ScriptedSummarizer().install(monkeypatch)
    clock = threading.local()
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0 + getattr(clock, "elapsed", 0.0))
    created = []

    class NativeCompressor:
        compression_count = 0
        _last_compress_aborted = False

        def __init__(self, **kwargs):
            created.append(self)

        def compress(self, messages, **kwargs):
            clock.elapsed = 7.5
            if host_fails:
                raise RuntimeError("scripted native failure")
            self.compression_count += 1
            return messages

    native = ModuleType("agent.context_compressor")
    native.ContextCompressor = NativeCompressor
    monkeypatch.setitem(sys.modules, "agent.context_compressor", native)
    with engine_context(tmp_path, window=200_000, session_id="parent", conversation_id="parent",
                        fresh_tail_count=2, leaf_chunk_tokens=400, temporal_rollups_enabled=False,
                        large_output_externalization_enabled=False) as engine:
        engine.model = "test-model"
        engine._last_compress_aborted = "parent sentinel"
        ready, finished = Barrier(2, timeout=10), Barrier(2, timeout=10)
        record_proof = engine._record_compress_commit_proof
        snapshots, threads = [], {}

        def pause_parent(messages, result):
            record_proof(messages, result)
            if engine._bypasses_lcm_context_management():
                return
            snapshots.append(_snapshot(engine))
            ready.wait()
            finished.wait()
            snapshots.append(_snapshot(engine))

        monkeypatch.setattr(engine, "_record_compress_commit_proof", pause_parent)

        def parent():
            threads["parent"] = threading.get_ident()
            return engine.compress(_history(), force=True)

        def child():
            threads["child"] = threading.get_ident()
            ready.wait()
            engine._mark_thread_context_stateless("child")
            try:
                view = [user("child request"), assistant("child reply")]
                result = engine.compress(view, force=True)
                assert result is view
                assert engine.last_compression_status == "host_fallback"
                assert engine.get_status()["last_compression_noop_reason"] == "LCM bypassed: auxiliary thread context"
                record_proof(view, result)  # Its own guard must not clear the parent's proof either.
                engine.last_compression_status = ""  # The host's writable reset is child-local too.
                assert engine.last_compression_status == ""
                assert engine._thread_context.foreground_budget is None
            finally:
                engine._clear_thread_context_stateless("child")
                finished.wait()

        with caplog.at_level(logging.INFO, logger="hermes_lcm"), ThreadPoolExecutor(max_workers=2) as pool:
            foreground, delegated = pool.submit(parent), pool.submit(child)
            delegated.result(timeout=15)
            assert foreground.result(timeout=15)
        assert len(created) == 1
        stops = {record.thread: record.getMessage() for record in caplog.records
                 if record.getMessage().startswith("LCM compaction stop:")}
        assert len(stops) == 2
        assert "reason=host_fallback leaves=0" in stops[threads["child"]]
        assert "host=7.5s" in stops[threads["child"]]
        assert "reason=compacted" in stops[threads["parent"]]
        assert " host=" not in stops[threads["parent"]]
        assert snapshots[0] == snapshots[1]


# Legacy names also form #1242's parent-state snapshot contract.
PREFLIGHT_HANDOFFS = (
    "_preflight_cleanup_only_due_to_boundary_cooldown",
    "_preflight_below_threshold_cleanup_only",
    "_preflight_automatic_request",
    "_hold_fit_only_requested",
)


@pytest.mark.parametrize("mask", range(16))
@pytest.mark.parametrize("exit_path", ["empty", "ingest_error"])
def test_preflight_handoffs_consumed_before_early_exit(tmp_path, monkeypatch, mask, exit_path):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000) as engine:
        assert all(getattr(engine, name) is False for name in PREFLIGHT_HANDOFFS)
        for bit, name in enumerate(PREFLIGHT_HANDOFFS):
            setattr(engine, name, bool(mask & (1 << bit)))

        if exit_path == "empty":
            view = []
            assert engine._compress_impl(view) is view
            assert engine.last_compression_status == "noop"
            assert engine.last_compression_noop_reason == "empty message list"
        else:
            def fail_ingest(messages):
                assert all(getattr(engine, name) is False for name in PREFLIGHT_HANDOFFS)
                raise RuntimeError("scripted ingest failure")

            monkeypatch.setattr(engine, "_ingest_messages", fail_ingest)
            with pytest.raises(RuntimeError, match="scripted ingest failure"):
                engine._compress_impl([user("next turn")], current_tokens=10)
        assert all(getattr(engine, name) is False for name in PREFLIGHT_HANDOFFS)
        # A second invocation cannot inherit the previous request.
        assert engine._compress_impl([]) == []
        assert all(getattr(engine, name) is False for name in PREFLIGHT_HANDOFFS)


@pytest.mark.parametrize("reset", ["preflight", "session"])
def test_preflight_handoff_reset_masks(tmp_path, monkeypatch, reset):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000) as engine:
        for name in PREFLIGHT_HANDOFFS:
            setattr(engine, name, True)
        if reset == "preflight":
            assert engine.should_compress_preflight([]) is False
            expected = (False, False, False, True)
        else:
            engine._reset_session_scoped_runtime_state()
            expected = (True, False, False, True)
        assert tuple(getattr(engine, name) for name in PREFLIGHT_HANDOFFS) == expected


@pytest.mark.parametrize("threshold", [0, 100])
def test_maintenance_handoff_preserves_unrelated_flags(tmp_path, threshold):
    with engine_context(tmp_path, window=200_000) as engine:
        engine.threshold_tokens = threshold
        engine._preflight_cleanup_only_due_to_boundary_cooldown = True
        engine._hold_fit_only_requested = True
        assert engine._mark_below_threshold_maintenance() is True
        assert tuple(getattr(engine, name) for name in PREFLIGHT_HANDOFFS) == (
            True, threshold > 0, True, True,
        )
        assert engine.last_compression_status == "pending"
        assert engine.last_compression_noop_reason == ""


def test_child_preserves_armed_preflight_handoffs(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000) as engine:
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
        for name in PREFLIGHT_HANDOFFS:
            setattr(engine, name, True)
        before = _snapshot(engine)
        engine._mark_thread_context_stateless("child")
        try:
            assert engine.compress([], force=True) == []
        finally:
            engine._clear_thread_context_stateless("child")
        after = _snapshot(engine)
        for name in PREFLIGHT_HANDOFFS:
            assert before[name] is True and after[name] is True


class _TimeoutRegex:  # the optional ``regex`` engine's timeout-capable interface (as tests/test_host_uid_shadow.py)
    error = re.error

    class _Pattern:
        def __init__(self, pattern):
            self._compiled = re.compile(pattern)

        def search(self, text, *, timeout=None):
            return self._compiled.search(text)

    @classmethod
    def compile(cls, pattern):
        return cls._Pattern(pattern)


@pytest.mark.parametrize("site", ["ignored_backlog", "leaf_candidate", "ingest", "dependent_reply"])
@pytest.mark.parametrize("exit_path", ["return", "inner_error", "outer_error", "map_error"])
def test_store_id_scopes_restore_outer_identity(tmp_path, monkeypatch, site, exit_path):
    """Each temporary map restores by identity, including nested readers and failures."""
    pin_char_counter(monkeypatch)
    # CI installs no ``regex``; without a timeout-capable engine the ignore pattern is dropped and three sites never run.
    monkeypatch.setattr(message_patterns, "_regex_engine", _TimeoutRegex)
    with engine_context(tmp_path, window=200_000, ignore_message_patterns=["ignored"],
                        large_output_externalization_enabled=False) as engine:
        outer, inner = assistant("outer visible"), assistant("inner visible")
        original = {999: 999}
        maps = {id(outer): {id(outer): 101}, id(inner): {id(inner): 202}}
        engine._current_compress_store_ids_by_message_id = original
        monkeypatch.setattr(engine, "_fresh_tail_start", len)
        monkeypatch.setattr(engine, "_leading_anchor_count", lambda messages: 0)
        monkeypatch.setattr(engine, "_replay_occurrences", lambda messages: (list(messages), True))
        reads, builds, stack = [], [], []

        class FinishedIngest(Exception):
            """Stop after the ingest scope, before unrelated persistence work."""

        def after_ingest_scope(messages):
            raise FinishedIngest

        monkeypatch.setattr(engine, "_redact_active_replay_messages", after_ingest_scope)

        def build(messages, occurrences=None):
            assert len(messages) == 1
            assert occurrences == (messages if site in {"ignored_backlog", "leaf_candidate"} else None)
            builds.append(messages[0])
            if exit_path == "map_error":
                raise RuntimeError("map construction failed")
            return maps[id(messages[0])]

        monkeypatch.setattr(engine, "_get_store_id_map_for_messages", build)

        def stored_row(store_id):
            reads.append(store_id)
            assert store_id == maps[id(stack[-1])][id(stack[-1])]
            return assistant("ignored")

        monkeypatch.setattr(engine._store, "get", stored_row)
        mapped_reader = engine._mapped_stored_row_matches_ignore_message_patterns

        def invoke(message):
            stack.append(message)
            try:
                if site == "ignored_backlog":
                    engine._has_ignored_backlog_outside_fresh_tail([message])
                elif site == "leaf_candidate":
                    engine._leaf_compaction_candidate_status_once([message])
                elif site == "dependent_reply":
                    engine._drop_preexisting_generated_ignored_dependent_eof_replies([message], [{"store": "fixture"}])
                else:
                    with pytest.raises(FinishedIngest):
                        engine._ingest_messages([message])
            finally:
                stack.pop()

        def observe(message):
            active = maps[id(message)]
            assert engine._current_compress_store_ids_by_message_id is active
            assert mapped_reader(message) is True  # The real engine reader must use this scope's map.
            if message is inner:
                if exit_path == "inner_error":
                    raise RuntimeError("inner scope failed")
            else:
                if exit_path == "inner_error":
                    with pytest.raises(RuntimeError, match="inner scope failed"):
                        invoke(inner)
                else:
                    invoke(inner)
                assert engine._current_compress_store_ids_by_message_id is active
                # Restore the same object, not a copy of its old contents.
                active[303] = 404
                if exit_path == "outer_error":
                    raise RuntimeError("outer scope failed")
            return True

        monkeypatch.setattr(engine, "_mapped_stored_row_matches_ignore_message_patterns", observe)
        monkeypatch.setattr(engine, "_matches_preexisting_generated_ignored_dependent_reply",
                            lambda message, text, records: observe(message))
        if exit_path in {"outer_error", "map_error"}:
            with pytest.raises(RuntimeError, match="outer scope failed|map construction failed"):
                invoke(outer)
        else:
            invoke(outer)
        assert engine._current_compress_store_ids_by_message_id is original
        assert original == {999: 999}
        assert builds == ([outer] if exit_path == "map_error" else [outer, inner])
        assert reads == ([] if exit_path == "map_error" else [101, 202])
        if exit_path != "map_error":
            assert maps[id(outer)][303] == 404


def test_running_compress_reader_uses_pass_store_ids(tmp_path, monkeypatch):
    """The provider boundary observes the current pass map, not a preflight map."""
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, fresh_tail_count=2, leaf_chunk_tokens=400,
                        ignore_message_patterns=["never-match-fixture"],
                        large_output_externalization_enabled=False) as engine:
        objects, reads, observed = {}, [], []
        build = engine._get_store_id_map_for_messages
        get = engine._store.get

        def remember(messages, occurrences=None):
            objects.update((id(message), message) for message in messages)
            return build(messages, occurrences)

        def read(store_id):
            reads.append(store_id)
            return get(store_id)

        def summarize(*args, **kwargs):
            active = engine._current_compress_store_ids_by_message_id
            assert active
            message_id, store_id = next(iter(active.items()))
            reads.clear()
            assert engine._mapped_stored_row_matches_ignore_message_patterns(objects[message_id]) is False
            assert reads == [store_id]
            observed.append(active)
            return ScriptedSummarizer.healthy_reply

        monkeypatch.setattr(engine, "_get_store_id_map_for_messages", remember)
        monkeypatch.setattr(engine._store, "get", read)
        summaries = ScriptedSummarizer(mode="scripted", fn=summarize).install(monkeypatch)
        result = engine.compress(_history(), force=True)
        assert summaries.calls and observed and len(result) < len(_history())
        # The pass-level assignment is deliberately persistent, unlike temporary scopes.
        assert engine._current_compress_store_ids_by_message_id is observed[-1]
