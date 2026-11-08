# ruff: noqa: F811  # Imported pytest fixture is intentionally injected by name.
"""#2482: a delegated child interrupted before it finishes keeps every row the host handed LCM-X.

The host hands a child's whole list to plugins at ``pre_llm_call`` (once per turn, before the tool loop) and at
``pre_api_request`` (before every model call), both with ``conversation_history=list(messages)``. A child that is
killed or cut off by a clean host exit reaches neither ``post_llm_call`` nor ``on_session_end``. The three
interrupted shapes are the ones the v0.30.0 driven run (r7) lost; the guards pin that a child that finishes
normally stores the same rows exactly once, and that the parent's resume in a new process adds no copy.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.test_issue_1931_delegate_children import DELEGATE, PARENT_VIEW, Agent, child_rows, host  # noqa: F401

PROMPT = {"role": "user", "content": "Count the ledger rows and report the total."}


def call(i, tag=""):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"{tag}c{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]}


def result(i, tag=""):
    return {"role": "tool", "tool_call_id": f"{tag}c{i}", "content": f"{tag}ledger chunk {i}: 17 rows"}


def view(tool_calls, final=False, tag=""):
    rows = [dict(PROMPT, content=tag + PROMPT["content"])]
    for i in range(1, tool_calls + 1):
        rows += [call(i, tag), result(i, tag)]
    return rows + ([{"role": "assistant", "content": f"{tag}The ledger has 34 rows."}] if final else [])


class HostChild(Agent):
    """A delegated child whose hooks fire inside its own frame, with the host's payload fields."""

    tag = ""

    def fire(self, ctx, name, history, **extra):
        """Run the callbacks on a separate worker thread, as the host runs its timeout-bounded hooks."""
        payload = dict(session_id=self.session_id, platform="subagent", conversation_history=list(history), **extra)
        out, errors = [], []

        def run():
            try:
                out.extend(callback(**payload) for callback in ctx.hooks.get(name, []))
            except BaseException as exc:  # surfaced below: a hook must never raise
                errors.append(exc)

        worker = threading.Thread(target=run, name=f"hermes-hook-{name}", daemon=True)
        worker.start()
        worker.join(30)
        assert not worker.is_alive() and not errors, errors
        return out

    def spawn(self, ctx):
        """The parent fires ``subagent_start`` on its own thread before the child runs (tools/delegate_tool.py)."""
        for callback in ctx.hooks.get("subagent_start", []):
            callback(parent_session_id=self._parent_session_id, parent_turn_id="pt-1", parent_subagent_id="",
                     child_session_id=self.session_id, child_subagent_id=self._subagent_id, child_role="leaf",
                     child_goal="count the ledger")

    def llm_start(self, ctx, history):
        return self.fire(ctx, "pre_llm_call", history, user_message=history[0]["content"], is_first_turn=True,
                         parent_session_id=self._parent_session_id, model="m")

    def api_call(self, ctx, history, n):
        return self.fire(ctx, "pre_api_request", history, api_call_count=n, retry_count=0, model="m",
                         request_messages=list(history), message_count=len(history), task_id="t", turn_id="turn-1")

    def run_until(self, ctx, tool_calls):
        """Start the turn and issue model call k after k-1 tool rounds, up to ``tool_calls`` rounds done."""
        self.llm_start(ctx, view(0, tag=self.tag))
        for k in range(tool_calls + 1):
            self.api_call(ctx, view(k, tag=self.tag), k + 1)


def _setup(host, route):
    load, clone = host
    ctx, module = load()
    ctx.module = module
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    ctx.hooks["post_llm_call"][0](session_id="parent", platform="cli",
                                  conversation_history=PARENT_VIEW, context_compressor=parent)
    child = HostChild("child", **DELEGATE)
    child.spawn(ctx)
    engine = parent
    if route == "own_engine":
        engine = clone(ctx)
        child.start(engine)
    return load, clone, ctx, parent, child, engine


def _contents(rows):
    return [(row["role"], row["content"]) for row in rows]


ROUTES = ["own_engine", "parent_engine"]
# The host binds each child's own engine (on_session_start on the child's thread) before its first hook; the early
# store runs only for a child bound that way as a delegate. "parent_engine" (never bound) keeps main's behaviour.
BOUND = ["own_engine"]
# r7's three interrupted shapes: tool rounds completed before the interruption.
SHAPES = {"kill_third_model_call_in_flight": 2, "exit_after_one_call_and_result": 1, "exit_with_only_the_prompt": 0}


@pytest.mark.parametrize("route", BOUND)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_an_interrupted_child_keeps_every_row_handed_over(host, route, shape):
    _, _, ctx, parent, child, _ = _setup(host, route)
    child.run_until(ctx, SHAPES[shape])
    # Interrupted here: no post_llm_call, no on_session_end, no compaction.
    assert _contents(child_rows(parent)) == _contents(view(SHAPES[shape]))


@pytest.mark.parametrize("route", ROUTES)
def test_a_child_that_finishes_after_incremental_stores_has_no_copy(host, route):
    _, _, ctx, parent, child, engine = _setup(host, route)
    child.run_until(ctx, 2)
    final = view(2, final=True)
    child.turn(ctx, final, engine)
    child.end(engine, final)
    child.end(engine, final)  # a repeated end replays the same view
    assert _contents(child_rows(parent)) == _contents(final)


@pytest.mark.parametrize("route", ROUTES)
def test_a_child_that_finishes_normally_is_unchanged(host, route):
    _, _, ctx, parent, child, engine = _setup(host, route)
    final = view(2, final=True)
    child.turn(ctx, final, engine)
    child.end(engine, final)
    assert _contents(child_rows(parent)) == _contents(final)


@pytest.mark.parametrize("route", BOUND)
def test_the_parent_resume_after_a_kill_adds_no_copy(host, route):
    load, clone, ctx, parent, child, _ = _setup(host, route)
    child.run_until(ctx, 2)
    # Killed; a new process resumes the parent and runs one more parent turn.
    ctx2, _ = load()
    parent2 = clone(ctx2)
    parent2.on_session_start("parent", platform="cli", context_length=200_000)
    resumed = PARENT_VIEW + [{"role": "user", "content": "And the total?"},
                             {"role": "assistant", "content": "The subagent did not finish."}]
    ctx2.hooks["post_llm_call"][0](session_id="parent", platform="cli",
                                   conversation_history=resumed, context_compressor=parent2)
    assert _contents(child_rows(parent2)) == _contents(view(2))
    assert _contents(parent2._store.get_session_messages("parent")) == _contents(resumed)


def test_other_sessions_ignore_the_model_call_hook(host):
    _, _, ctx, parent, _, _ = _setup(host, "parent_engine")
    before = _contents(parent._store.get_session_messages("parent"))
    for callback in ctx.hooks.get("pre_api_request", []):
        callback(session_id="parent", platform="cli", conversation_history=PARENT_VIEW + [
            {"role": "user", "content": "a new prompt not yet stored"}], api_call_count=1)
    assert _contents(parent._store.get_session_messages("parent")) == before


@pytest.mark.parametrize("route", BOUND)
def test_a_retried_model_call_with_the_same_list_adds_nothing(host, route):
    _, _, ctx, parent, child, _ = _setup(host, route)
    child.run_until(ctx, 1)
    child.api_call(ctx, view(1), 2)  # the host retries the same request
    child.api_call(ctx, view(1), 2)
    assert _contents(child_rows(parent)) == _contents(view(1))


def test_parallel_children_of_one_parent_store_only_their_own_rows(host):
    _, clone, ctx, parent, _, _ = _setup(host, "parent_engine")
    before = _contents(parent._store.get_session_messages("parent"))
    children = []
    for n, tag in enumerate(("a:", "b:", "c:")):
        child = HostChild(f"child-{n}", **dict(DELEGATE, _subagent_id=f"sa-{n}", log_prefix=f"[subagent-{n}]"))
        child.tag = tag
        child.spawn(ctx)
        child.start(clone(ctx))
        children.append(child)
    start = threading.Barrier(len(children))

    def run(child):
        start.wait()
        child.run_until(ctx, 2)

    with ThreadPoolExecutor(len(children)) as pool:
        list(pool.map(run, children))
    for child in children:
        assert _contents(child_rows(parent, child.session_id)) == _contents(view(2, tag=child.tag))
    assert _contents(parent._store.get_session_messages("parent")) == before


@pytest.mark.parametrize("route", ROUTES)
def test_the_pre_llm_call_return_value_is_unchanged(host, route):
    _, _, ctx, parent, child, _ = _setup(host, route)
    # main 4e8c8c2d returns [None] here for the child and the parent; the child store adds no context.
    assert child.llm_start(ctx, view(0)) == [None] * len(ctx.hooks["pre_llm_call"])
    assert _contents(child_rows(parent)) == (_contents(view(0)) if route == "own_engine" else [])
    parent_returns = [callback(session_id="parent", platform="cli", conversation_history=PARENT_VIEW,
                               user_message="x", is_first_turn=False, model="m")
                      for callback in ctx.hooks["pre_llm_call"]]
    assert parent_returns == [None] * len(ctx.hooks["pre_llm_call"])
    assert all(value is None for value in child.api_call(ctx, view(0), 1))


def test_a_busy_store_lock_skips_the_store_within_the_bound(host, monkeypatch):
    _, _, ctx, parent, child, _ = _setup(host, "own_engine")
    aux = ctx.module("aux_session")
    monkeypatch.setattr(aux, "_LIVE_CHILD_STORE_WAIT_S", 0.2, raising=False)
    held, release = threading.Event(), threading.Event()

    def hold():
        with aux._DELEGATE_STORE_LOCK:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(5)
        began = time.monotonic()
        child.run_until(ctx, 1)  # pre_llm_call + two model calls, each skipped
        assert time.monotonic() - began < 2.0
        assert child_rows(parent) == []
    finally:
        release.set()
        holder.join(10)
    child.api_call(ctx, view(1), 3)  # the next model call stores everything handed over so far
    assert _contents(child_rows(parent)) == _contents(view(1))


def test_a_failing_store_never_raises_into_the_hook(host, monkeypatch):
    _, _, ctx, parent, child, _ = _setup(host, "own_engine")
    mixin = ctx.module("aux_session").DelegateChildMixin
    real = mixin._append_delegate_child_rows

    def boom(self, *args, **kwargs):
        raise RuntimeError("store failed")

    monkeypatch.setattr(mixin, "_append_delegate_child_rows", boom)
    assert child.llm_start(ctx, view(0)) == [None] * len(ctx.hooks["pre_llm_call"])
    assert all(value is None for value in child.api_call(ctx, view(0), 1))
    assert child_rows(parent) == []
    monkeypatch.setattr(mixin, "_append_delegate_child_rows", real)
    child.api_call(ctx, view(1), 2)
    assert _contents(child_rows(parent)) == _contents(view(1))


@pytest.mark.xfail(strict=True, reason="#2482 named loss limit (v0.30.0): a child interrupted while its tools run "
                   "keeps its rows up to its last model call; that round's tool-call row and finished results stay "
                   "only in the host (follow-up: recovery from the host's state.db)")
@pytest.mark.parametrize("route", BOUND)
def test_residual_a_child_interrupted_while_its_tools_run_keeps_that_round(host, route):
    _, _, ctx, parent, child, _ = _setup(host, route)
    child.run_until(ctx, 2)
    # The host persists the third tool-call row and its result while the tools run; no model call follows.
    assert _contents(child_rows(parent)) == _contents(view(3))


@pytest.mark.parametrize("route", ROUTES)
def test_without_a_subagent_start_record_the_model_call_hooks_store_nothing(host, route):
    # A host without the subagent_start hook (the legacy frame-walk fallback): behaviour stays at main's.
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    child = HostChild("child", **DELEGATE)  # no spawn(): no subagent_start record
    engine = parent
    if route == "own_engine":
        engine = clone(ctx)
        child.start(engine)
    child.run_until(ctx, 2)
    assert child_rows(parent) == []
    final = view(2, final=True)
    child.turn(ctx, final, engine)
    child.end(engine, final)
    assert _contents(child_rows(parent)) == _contents(final)


def test_when_another_context_engine_is_active_the_model_call_hooks_store_nothing(tmp_path, monkeypatch):
    # Main stores nothing for a child while LCM-X is not the context engine (post_llm_call returns early).
    from tests.test_issue_622_slow_load import _Ctx, _Manager, _OtherEngine, _install_host, _load_plugin

    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    _install_host(monkeypatch, tmp_path)
    module = _load_plugin("hermes_lcm_2482_inactive")
    ctx = _Ctx(_Manager(_OtherEngine()))
    module.register(ctx)
    prototype = ctx.offered
    try:
        child = HostChild("child", **DELEGATE)
        child.spawn(ctx)
        child.run_until(ctx, 2)
        assert prototype._store.get_session_messages("child") == []
    finally:
        prototype.shutdown()



@pytest.mark.parametrize("shape", list(SHAPES))
def test_a_child_that_never_bound_an_engine_keeps_mains_behaviour(host, shape):
    # Auxiliary detection needs the child's own frames, which only its bind on its own thread sees; the hook worker
    # thread cannot tell a delegate from an AUXILIARY child, so an unbound child is left to the end store, as on main.
    _, _, ctx, parent, child, engine = _setup(host, "parent_engine")
    child.run_until(ctx, SHAPES[shape])
    assert child_rows(parent) == []
    final = view(2, final=True)
    child.turn(ctx, final, engine)
    child.end(engine, final)
    assert _contents(child_rows(parent)) == _contents(final)


def test_an_auxiliary_child_with_a_subagent_start_record_stores_nothing(host):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    child = HostChild("child", **dict(DELEGATE, enabled_toolsets=["memory", "skills"]))
    child.spawn(ctx)
    engine = clone(ctx)
    child.start(engine)
    child.run_until(ctx, 2)
    final = view(2, final=True)
    child.turn(ctx, final, engine)
    child.end(engine, final)
    assert child_rows(parent) == []


def test_a_child_rotated_by_its_own_compaction_keeps_the_rows_handed_over_after_it(host, monkeypatch):
    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    big = [{"role": "user", "content": "start " + "payload " * 400}] + [
        {"role": "assistant" if i % 2 else "user", "content": f"turn {i} " + "payload " * 400} for i in range(1, 12)]
    child.llm_start(ctx, big[:1])
    child.api_call(ctx, big, 1)
    monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
    monkeypatch.setattr(engine, "_bypass_compaction_target_tokens", lambda **kwargs: 64)
    compacted = child.compress(engine, big)
    child.rotate(engine, "child-continued")
    after = [*compacted, {"role": "assistant", "content": "after rotation 0"}]
    child.api_call(ctx, after, 2)
    after = [*after, {"role": "assistant", "content": "after rotation 1"}]
    child.api_call(ctx, after, 3)
    # Interrupted here.
    assert _contents(child_rows(parent)) == _contents(big)
    assert _contents(child_rows(parent, "child-continued")) == [("assistant", "after rotation 0"),
                                                                 ("assistant", "after rotation 1")]


def test_a_slow_store_returns_to_the_host_within_the_bound_and_still_lands(host, monkeypatch):
    _, _, ctx, parent, child, _ = _setup(host, "own_engine")
    aux = ctx.module("aux_session")
    monkeypatch.setattr(aux, "_LIVE_CHILD_STORE_JOIN_S", 0.3, raising=False)
    mixin = aux.DelegateChildMixin
    real, calls, running, overlap = mixin._append_delegate_child_rows, [], [], []

    def slow(self, *args, **kwargs):
        calls.append(1)
        running.append(1)
        overlap.append(len(running))
        try:
            if len(calls) == 1:
                time.sleep(2.0)  # e.g. SQLite busy: a wait the lock bounds do not cover
            return real(self, *args, **kwargs)
        finally:
            running.pop()

    monkeypatch.setattr(mixin, "_append_delegate_child_rows", slow)
    began = time.monotonic()
    child.api_call(ctx, view(0), 1)
    child.api_call(ctx, view(1), 2)  # handed over while the first store is in flight: kept, stored after it
    assert time.monotonic() - began < 1.5
    # Interrupted here: no further hook and no end store.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and len(child_rows(parent)) < 3:
        time.sleep(0.05)
    time.sleep(0.2)
    assert _contents(child_rows(parent)) == _contents(view(1))
    assert len(calls) == 2 and max(overlap) == 1  # one store at a time per child


def test_a_list_queued_behind_a_store_is_stored_before_the_engines_close(host, monkeypatch):
    import sqlite3

    _, _, ctx, parent, child, engine = _setup(host, "own_engine")
    aux = ctx.module("aux_session")
    monkeypatch.setattr(aux, "_LIVE_CHILD_STORE_JOIN_S", 0.2, raising=False)
    mixin = aux.DelegateChildMixin
    real, calls = mixin._append_delegate_child_rows, []

    def slow_first(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            time.sleep(1.0)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(mixin, "_append_delegate_child_rows", slow_first)
    db_path = parent._store.db_path
    child.api_call(ctx, view(0), 1)
    child.api_call(ctx, view(1), 2)  # queued behind the in-flight store
    # Clean exit right away: the child's, the parent's and the prototype engines shut down.
    for closing in (engine, parent, ctx.offered):
        closing.shutdown()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY store_id",
                            ("child",)).fetchall()
    assert rows == _contents(view(1))
