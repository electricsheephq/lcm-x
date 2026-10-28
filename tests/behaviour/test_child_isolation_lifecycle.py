"""Child-only diagnostics and cross-thread native fallback lifecycle ownership."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import pytest

from tests.harness import assistant, engine_context, pin_char_counter, user


def _view():
    return [user("request"), assistant("reply")]


@pytest.fixture
def native(monkeypatch):
    created, ended, resets = [], [], []
    events = []

    class NativeCompressor:
        compression_count = 0
        _last_compress_aborted = True

        def __init__(self, **kwargs):
            self.index = len(created)
            self.events = events
            created.append(self)
            events.append(("create", self.index))

        def compress(self, messages, **kwargs):
            self.compression_count += 1
            events.append(("compress", self.index))
            return messages

        def on_session_end(self, sid, messages):
            ended.append((sid, self))
            events.append(("end", sid, self.index))

        def on_session_reset(self):
            resets.append(self)
            events.append(("reset", self.index))

    module = ModuleType("agent.context_compressor")
    module.ContextCompressor = NativeCompressor
    monkeypatch.setitem(sys.modules, "agent.context_compressor", module)
    return created, ended, resets


def _config(kind):
    return {f"{kind}_session_patterns": ["cron*"]}


@pytest.mark.parametrize("kind", ["ignore", "stateless"])
@pytest.mark.parametrize("case,status,reason", [
    ("empty", "noop", "empty message list"),
    ("below", "noop", "LCM bypassed below threshold"),
    ("forced", "host_fallback", "LCM bypassed:"),
])
def test_main_bypass_status_cross_thread(tmp_path, monkeypatch, kind, case, status, reason):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="cron_1", **_config(kind)) as engine:
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
        view = [] if case == "empty" else _view()
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(engine.compress, view, force=case == "forced").result() == view
        assert engine.last_compression_status == status
        assert engine.last_compression_noop_reason.startswith(reason)
        public = engine.get_status()
        assert public["last_compression_status"] == status
        assert public["last_compression_noop_reason"] == engine.last_compression_noop_reason
        assert engine.last_compression_was_noop == (status == "noop")


@pytest.mark.parametrize("kind", ["ignore", "stateless"])
def test_main_bypass_aborted_visible_on_loop_thread(tmp_path, monkeypatch, native, kind):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="cron_1", **_config(kind)) as engine:
        engine.model = "test-model"
        with ThreadPoolExecutor(1) as pool:
            pool.submit(engine.compress, _view(), force=True).result()
        assert getattr(engine, "_last_compress_aborted", False) is True


@pytest.mark.parametrize("kind", ["ignore", "stateless"])
def test_main_bypass_reuses_native_across_pool_threads(tmp_path, monkeypatch, native, kind):
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    with engine_context(tmp_path, window=200_000, session_id="cron_1", **_config(kind)) as engine:
        engine.model = "test-model"
        # Both workers stay alive, so distinct pools guarantee distinct threads.
        with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
            first.submit(engine.compress, _view(), force=True).result()
            second.submit(engine.compress, _view(), force=True).result()
        assert len(created) == 1
        assert created[0].compression_count == 2
        engine.on_session_end("cron_1", [])
        assert ended == [("cron_1", created[0])]
        assert resets == created


@pytest.mark.parametrize("kind", ["ignore", "stateless"])
def test_main_bypass_reset_recreates_native(tmp_path, monkeypatch, native, kind):
    pin_char_counter(monkeypatch)
    created, _, resets = native
    with engine_context(tmp_path, window=200_000, session_id="cron_1", **_config(kind)) as engine:
        engine.model = "test-model"
        with ThreadPoolExecutor(1) as pool:
            pool.submit(engine.compress, _view(), force=True).result()
            first = created[0]
            engine.on_session_reset()
            pool.submit(engine.compress, _view(), force=True).result()
        assert resets == [first]
        assert len(created) == 2
        assert created[1] is not first


@pytest.mark.parametrize("kind", ["ignore", "stateless", "child"])
@pytest.mark.parametrize("event", ["end", "rebind", "reset"])
def test_native_lifecycle_from_other_thread(tmp_path, monkeypatch, native, kind, event):
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    sid = "child" if kind == "child" else "cron_1"
    config = {} if kind == "child" else _config(kind)
    with engine_context(tmp_path, window=200_000, session_id="parent" if kind == "child" else sid,
                        **config) as engine:
        engine.model = "test-model"

        def compress():
            if kind == "child":
                engine._mark_thread_context_stateless(sid)
            try:
                engine.compress(_view(), force=True)
            finally:
                if kind == "child":
                    engine._clear_thread_context_stateless(sid)

        with ThreadPoolExecutor(1) as pool:
            pool.submit(compress).result()
            if event == "end":
                engine.on_session_end(sid, [])
            elif event == "reset":
                engine.on_session_reset()
            else:
                engine.on_session_start("next_parent", platform="cli", context_length=200_000)
        assert len(created) == 1
        assert ended == ([] if event == "reset" else [(sid, created[0])])
        assert resets == created


def test_next_child_starts_with_clean_diagnostics(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)

        def children():
            engine._mark_thread_context_stateless("first_child")
            engine.compress(_view(), force=True)
            engine._last_compress_aborted = True
            engine._clear_thread_context_stateless("first_child")
            engine._mark_thread_context_stateless("next_child")
            try:
                return (engine.last_compression_status, engine.last_compression_noop_reason,
                        getattr(engine, "_last_compress_aborted", None))
            finally:
                engine._clear_thread_context_stateless("next_child")

        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(children).result() == ("idle", "", None)


def test_late_reclassification_preserves_foreground_status(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="background") as engine:
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
        monkeypatch.setattr(engine, "_in_process_auxiliary_caller_generation", lambda sid, **kwargs: 1)
        engine._last_compression_status = "compacted"
        proof = engine._compress_commit_proof = {"sentinel": True}

        def compress():
            result = engine.compress(_view(), force=True)
            return result, engine.last_compression_status

        with ThreadPoolExecutor(1) as pool:
            result, status = pool.submit(compress).result()
        assert result == _view()
        assert status == "host_fallback"
        assert engine._last_compression_status == "compacted"
        assert engine._compress_commit_proof is proof


def test_profile_rebind_discards_child_native_on_other_thread(tmp_path, monkeypatch, native):
    """Profile reconfiguration must not leave a worker holding the old native cache."""
    pin_char_counter(monkeypatch)
    created, _, _ = native
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"

        def child():
            engine._mark_thread_context_stateless("child")
            try:
                engine.compress(_view(), force=True)
            finally:
                engine._clear_thread_context_stateless("child")

        with ThreadPoolExecutor(1) as pool:
            pool.submit(child).result()
            engine.on_session_start("next_parent", hermes_home=str(tmp_path / "next_home"),
                                    platform="cli", context_length=200_000)
            pool.submit(child).result()
        assert len(created) == 2
        assert created[0] is not created[1]


def _compress_child(engine, sid="child"):
    engine._mark_thread_context_stateless(sid)
    try:
        assert engine.compress(_view(), force=True) == _view()
    finally:
        engine._clear_thread_context_stateless(sid)


@pytest.mark.parametrize("event", ["reset", "rebind", "reconfigure"])
def test_same_child_id_two_threads_lifecycle_then_end(tmp_path, monkeypatch, native, event):
    """Every worker cache must receive the base lifecycle sequence."""
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"
        with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
            for pool in (first, second):
                pool.submit(_compress_child, engine).result()
            initial = list(created)
            assert len(initial) == 2
            if event == "reset":
                engine.on_session_reset()
            else:
                kwargs = {"hermes_home": str(tmp_path / "next_home")} if event == "reconfigure" else {}
                engine.on_session_start("next_parent", platform="cli", context_length=200_000, **kwargs)
            for pool in (first, second):
                pool.submit(_compress_child, engine).result()
            assert len(created) == 4, "each worker must create a fresh compressor after the boundary"
            engine.on_session_end("child", [])
        expected = [("create", 0), ("compress", 0), ("create", 1), ("compress", 1)]
        if event == "reset":
            expected += [("reset", 0), ("reset", 1)]
        elif event == "rebind":
            expected += [("end", "child", 0), ("reset", 0), ("end", "child", 1), ("reset", 1)]
        # Profile reconfiguration discards native caches without callbacks, as on base.
        expected += [("create", 2), ("compress", 2), ("create", 3), ("compress", 3),
                     ("end", "child", 2), ("reset", 2), ("end", "child", 3), ("reset", 3)]
        assert created[0].events == expected
        assert resets == (created[2:] if event == "reconfigure" else created)
        assert ended == [("child", item) for item in (created if event == "rebind" else created[2:])]
        with engine._auxiliary_session_lock:
            assert engine._child_host_fallback_states == {}


def test_same_child_id_two_threads_end_then_first_thread_reuses_id(tmp_path, monkeypatch, native):
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"
        with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
            first.submit(_compress_child, engine).result()
            second.submit(_compress_child, engine).result()
            engine.on_session_end("child", [])
            assert ended == [("child", item) for item in created]
            first.submit(_compress_child, engine).result()
            assert len(created) == 3, "the first worker must not reuse an ended compressor"
            engine.on_session_end("child", [])
        assert created[0].events == [
            ("create", 0), ("compress", 0), ("create", 1), ("compress", 1),
            ("end", "child", 0), ("reset", 0), ("end", "child", 1), ("reset", 1),
            ("create", 2), ("compress", 2), ("end", "child", 2), ("reset", 2),
        ]
        assert resets == created


def test_nested_child_clear_keeps_outer_diagnostics(tmp_path, monkeypatch, native):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"

        def diagnostics():
            return (engine.last_compression_status, engine.last_compression_noop_reason,
                    getattr(engine, "_last_compress_aborted", None))

        def nested():
            engine._mark_thread_context_stateless("outer")
            try:
                engine.compress(_view(), force=True)
                before = diagnostics()
                engine._mark_thread_context_stateless("inner")
                engine._clear_thread_context_stateless("inner")
                assert engine._thread_context_session_id() == "outer"
                assert diagnostics() == before == ("host_fallback", "LCM bypassed: auxiliary thread context", True)
            finally:
                engine._clear_thread_context_stateless()
            engine._mark_thread_context_stateless("next_child")
            try:
                assert diagnostics() == ("idle", "", None)
            finally:
                engine._clear_thread_context_stateless("next_child")

        with ThreadPoolExecutor(1) as pool:
            pool.submit(nested).result()


@pytest.mark.parametrize("event", ["reset", "end"])
def test_same_child_registry_barrier_interleavings(tmp_path, monkeypatch, native, event):
    """Exercise registry writes and lifecycle walks without taking a second lock."""
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    barrier = threading.Barrier(2, timeout=10)
    errors = []
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"

        def worker(parent):
            try:
                for _ in range(200):
                    # Seed both thread dictionaries for the same child before each race.
                    _compress_child(engine)
                    barrier.wait()
                    if parent:
                        with engine._auxiliary_session_lock:
                            if event == "reset":
                                engine.on_session_reset()
                            else:
                                engine.on_session_end("child", [])
                    else:
                        _compress_child(engine)
                    barrier.wait()
                    if parent:
                        # A compress admitted after the callback is ended at quiescence.
                        engine.on_session_end("child", [])
                    barrier.wait()
            except BaseException as exc:
                errors.append(exc)
                barrier.abort()

        threads = [threading.Thread(target=worker, args=(parent,), daemon=True) for parent in (False, True)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert not any(thread.is_alive() for thread in threads), "registry/lifecycle deadlock"
        assert not errors, errors
        assert len(created) >= 400
        assert sorted(item.index for item in resets) == list(range(len(created))), "lost or duplicate reset"
        ended_indices = [item.index for sid, item in ended if sid == "child"]
        assert len(ended_indices) == len(set(ended_indices)), "duplicate end"
        if event == "end":
            assert sorted(ended_indices) == list(range(len(created))), "lost end"
        with engine._auxiliary_session_lock:
            assert engine._child_host_fallback_states == {}


@pytest.mark.parametrize("marker_held", [False, True])
def test_retained_child_promoted_to_bypassed_main_ends_native(tmp_path, monkeypatch, native, marker_held):
    """Retained child state must not hide the same ID's foreground compressor."""
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    sid = "cron_child"
    with engine_context(tmp_path, window=200_000, session_id="parent", **_config("ignore")) as engine:
        engine.model = "test-model"
        with ThreadPoolExecutor(1) as pool:
            if marker_held:
                pool.submit(engine._mark_thread_context_stateless, sid).result()
                pool.submit(engine.compress, _view(), force=True).result()
            else:
                pool.submit(_compress_child, engine, sid).result()
            engine.on_session_start(sid, platform="cli", context_length=200_000)
            engine.compress(_view(), force=True)
            main = engine.__dict__["_host_fallback_compressor"]
            start = len(main.events)
            engine.on_session_end(sid, [])
            assert (sid, main) in ended
            assert main in resets
            engine.compress(_view(), force=True)
            fresh = engine.__dict__["_host_fallback_compressor"]
            assert fresh is not main
            # Compare MAIN's sequence with the shared-slot base, excluding child callbacks.
            main_events = [event for event in main.events[start:] if event[-1] in (main.index, fresh.index)]
            assert main_events == [("end", sid, main.index), ("reset", main.index),
                                   ("create", fresh.index), ("compress", fresh.index)]
            if marker_held:
                pool.submit(engine._clear_thread_context_stateless, sid).result()
        assert created[-1] is fresh


@pytest.mark.parametrize("kind", ["ignore", "stateless"])
def test_child_collides_with_bypassed_main_end(tmp_path, monkeypatch, native, kind):
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    sid = "cron_main"
    with engine_context(tmp_path, window=200_000, session_id=sid, **_config(kind)) as engine:
        engine.model = "test-model"
        engine.compress(_view(), force=True)
        main = created[0]
        with ThreadPoolExecutor(1) as pool:
            pool.submit(_compress_child, engine, sid).result()
            active = list(created)
            engine.on_session_end(sid, [])
        assert ended == [(sid, compressor) for compressor in active]
        assert resets == active
        for compressor in active:
            lifecycle = [event for event in main.events if event[0] in ("end", "reset")
                         and event[-1] == compressor.index]
            assert lifecycle == [("end", sid, compressor.index), ("reset", compressor.index)]
        engine.compress(_view(), force=True)
        assert created[-1] not in active


@pytest.mark.parametrize("bound", ["child", "other", ""])
def test_auxiliary_handoff_respects_main_native_binding(tmp_path, monkeypatch, native, bound):
    """The usage handoff's equality guard applies separately to every native state."""
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    with engine_context(tmp_path, window=200_000, session_id="cron_main", **_config("ignore")) as engine:
        engine.model = "test-model"
        engine.compress(_view(), force=True)
        main = created[0]
        with ThreadPoolExecutor(1) as pool:
            pool.submit(engine._mark_thread_context_stateless, "child").result()
            pool.submit(engine.compress, _view(), force=True).result()
            # Model the three pre-existing shared-slot bindings at generation handoff.
            engine.__dict__["_host_fallback_session_id"] = bound
            monkeypatch.setattr(engine, "_in_process_auxiliary_caller_generation", lambda *args, **kwargs: 7)
            pool.submit(engine.update_from_response, {"prompt_tokens": 123}).result()
            pool.submit(engine._clear_thread_context_stateless, "child").result()
        main_events = [event for event in main.events if event[0] in ("end", "reset") and event[-1] == main.index]
        assert main_events == ([("end", "child", main.index), ("reset", main.index)] if bound == "child" else [])
        assert (main in resets) == (bound == "child")
        assert (engine.__dict__["_host_fallback_compressor"] is main) == (bound != "child")
        assert ("child", created[1]) in ended


@pytest.mark.parametrize("schedule", ["host_constructor", "marker_lookup"])
def test_child_native_publication_after_end(tmp_path, monkeypatch, native, schedule):
    """An admitted child retains its identity even after its live marker is gone."""
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    reached, resume = threading.Event(), threading.Event()
    child_states = []
    native_class = sys.modules["agent.context_compressor"].ContextCompressor

    class HostChild:
        _subagent_id = "synthetic-child"
        _parent_session_id = "parent"
        session_id = "child"

        def compress(self, engine):
            run_child(engine)

    def pause():
        reached.set()
        assert resume.wait(10), "parent did not release publication"

    def run_child(engine):
        child_states.append(vars(engine._thread_context))
        _compress_child(engine)

    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"
        main_compressor = None
        engine.__dict__["_host_fallback_compressor"] = main_compressor
        engine.__dict__["_host_fallback_session_id"] = "parent"
        original_init = native_class.__init__
        original_get = engine._get_host_fallback_compressor
        original_end = native_class.on_session_end
        original_reset = native_class.on_session_reset

        def construct(self, **kwargs):
            original_init(self, **kwargs)
            if schedule == "host_constructor":
                pause()

        def get():
            if schedule == "marker_lookup":
                pause()
            return original_get()

        def end(self, sid, messages):
            assert not engine._auxiliary_session_lock._is_owned()
            assert messages == []
            original_end(self, sid, messages)

        def reset(self):
            assert not engine._auxiliary_session_lock._is_owned()
            original_reset(self)

        monkeypatch.setattr(native_class, "__init__", construct)
        monkeypatch.setattr(native_class, "on_session_end", end)
        monkeypatch.setattr(native_class, "on_session_reset", reset)
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", get)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(HostChild().compress if schedule == "host_constructor" else run_child, engine)
            try:
                assert reached.wait(10), "child did not reach publication boundary"
                engine.on_session_end("child", [])
                assert engine._child_host_fallback_states == {}
                assert ended == resets == []
            finally:
                resume.set()
            future.result(10)
        assert len(created) == 1
        assert ended == [("child", created[0])]
        assert resets == created
        assert created[0].compression_count == 0
        assert engine._child_host_fallback_states == {}
        assert child_states[0].get("_host_fallback_compressor") is None
        assert child_states[0].get("_host_fallback_session_id", "") == ""
        assert engine.__dict__["_host_fallback_compressor"] is main_compressor
        assert engine.__dict__["_host_fallback_session_id"] == "parent"
        # Repeated lifecycle cleanup cannot rediscover or dispose the loser twice.
        engine.on_session_end("child", [])
        engine.on_session_reset()
        assert ended == [("child", created[0])]
        assert resets == created


@pytest.mark.parametrize("entry", ["native", "end_callback", "reset_callback"])
def test_ended_child_nested_compress_inherits_binding(tmp_path, monkeypatch, native, entry):
    """An admitted child stays bypassed through nested native and disposal calls."""
    pin_char_counter(monkeypatch)
    reached, resume = threading.Event(), threading.Event()
    native_class = sys.modules["agent.context_compressor"].ContextCompressor
    observed, restored, cleared = [], [], []
    inside = False
    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"
        parent = (engine.__dict__["_host_fallback_compressor"], engine.__dict__["_host_fallback_session_id"])
        before = list(engine._store.connection.iterdump())
        cursor = engine._ingest_cursor
        original_get = engine._get_host_fallback_compressor
        original_bypass = engine._compress_lcm_bypassed_session
        original_init = native_class.__init__
        original_end = native_class.on_session_end
        original_reset = native_class.on_session_reset

        def pause():
            reached.set()
            assert resume.wait(10), "child was not released"

        def nested():
            nonlocal inside
            outer = engine._thread_context.bypass_compression_binding
            inside = True
            try:
                engine.compress(_view(), current_tokens=100, force=True)
                restored.append(engine._thread_context.bypass_compression_binding is outer)
            finally:
                inside = False

        def bypass(messages, **kwargs):
            if inside:
                observed.append((engine._bypass_lcm_session_id(), engine._bypass_compress_state(),
                                 engine._thread_context.bypass_compression_binding))
            return original_bypass(messages, **kwargs)

        def construct(self, **kwargs):
            original_init(self, **kwargs)
            if entry != "native":
                pause()

        def compress(self, messages, **kwargs):
            pause()
            nested()
            return messages

        def end(self, sid, messages):
            assert not engine._auxiliary_session_lock._is_owned()
            original_end(self, sid, messages)
            if entry == "end_callback":
                nested()

        def reset(self):
            assert not engine._auxiliary_session_lock._is_owned()
            original_reset(self)
            if entry == "reset_callback":
                nested()

        def child():
            _compress_child(engine)
            cleared.append(getattr(engine._thread_context, "bypass_compression_binding", None))

        monkeypatch.setattr(native_class, "__init__", construct)
        monkeypatch.setattr(native_class, "compress", compress)
        monkeypatch.setattr(native_class, "on_session_end", end)
        monkeypatch.setattr(native_class, "on_session_reset", reset)
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None if inside else original_get())
        monkeypatch.setattr(engine, "_compress_lcm_bypassed_session", bypass)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(child)
            try:
                assert reached.wait(10), "child did not reach the nested-call boundary"
                engine.on_session_end("child", [])
                assert "child" not in engine._auxiliary_session_ids
            finally:
                resume.set()
            future.result(10)
        assert len(observed) == 1, "nested compress must use the bypass path"
        sid, state, binding = observed[0]
        assert sid == "child" and state is binding[0] and binding[1] == "child"
        assert state is not engine.__dict__
        assert restored == [True] and cleared == [None]
        assert (engine.__dict__["_host_fallback_compressor"], engine.__dict__["_host_fallback_session_id"]) == parent
        assert list(engine._store.connection.iterdump()) == before, "child wrote parent messages, summaries or cursors"
        assert engine._ingest_cursor == cursor


def test_reused_child_generation_rejects_old_publication(tmp_path, monkeypatch, native):
    """An old admitted compressor cannot join a replacement with the same ID."""
    pin_char_counter(monkeypatch)
    created, ended, resets = native
    reached, resume = threading.Event(), threading.Event()
    states = []
    native_class = sys.modules["agent.context_compressor"].ContextCompressor

    class HostChild:
        _subagent_id = "synthetic-child"
        _parent_session_id = "parent"
        session_id = "child"

        def start(self, engine):
            engine.on_session_start(self.session_id, platform="cli", context_length=200_000)

        def compress(self, engine):
            states.append(vars(engine._thread_context))
            engine.compress(_view(), force=True)

    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        engine.model = "test-model"
        original_init = native_class.__init__
        original_end = native_class.on_session_end
        original_reset = native_class.on_session_reset

        def construct(self, **kwargs):
            original_init(self, **kwargs)
            if self.index == 0:
                reached.set()
                assert resume.wait(10), "old publication was not released"

        def end(self, sid, messages):
            assert not engine._auxiliary_session_lock._is_owned()
            assert messages == []
            original_end(self, sid, messages)

        def reset(self):
            assert not engine._auxiliary_session_lock._is_owned()
            original_reset(self)

        monkeypatch.setattr(native_class, "__init__", construct)
        monkeypatch.setattr(native_class, "on_session_end", end)
        monkeypatch.setattr(native_class, "on_session_reset", reset)
        old, replacement = HostChild(), HostChild()
        old.start(engine)
        generation = engine._auxiliary_session_generations["child"]
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(old.compress, engine)
            try:
                assert reached.wait(10), "old child did not reach publication"
                engine.on_session_end("child", [])
                replacement.start(engine)
                assert engine._auxiliary_session_generations["child"] != generation
                replacement.compress(engine)
                replacement_state = dict(states[1])
                registry = dict(engine._child_host_fallback_states["child"])
            finally:
                resume.set()
            future.result(10)
        assert engine._child_host_fallback_states["child"] == registry, "old generation was registered"
        assert states[1] == replacement_state
        assert ended == [("child", created[0])] and resets == [created[0]]
        assert states[0].get("_host_fallback_compressor") is None
        assert states[0].get("_host_fallback_session_id", "") == ""
        assert created[0].compression_count == 0
        assert all(state.get("_host_fallback_compressor") is not created[0]
                   for state in [*states, *engine._host_fallback_states()])
        engine.on_session_end("child", [])
        assert ended == [("child", item) for item in created] and resets == created
