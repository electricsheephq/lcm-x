"""#2482: a delegated child interrupted before it finishes keeps every row the host handed LCM-X.

The host hands a child's whole list to plugins at ``pre_llm_call`` (once per turn, before the tool loop) and at
``pre_api_request`` (before every model call), both with ``conversation_history=list(messages)``. A child that is
killed or cut off by a clean host exit reaches neither ``post_llm_call`` nor ``on_session_end``. The three
interrupted shapes are the ones the v0.30.0 driven run (r7) lost; the guards pin that a child that finishes
normally stores the same rows exactly once, and that the parent's resume in a new process adds no copy.
"""

import pytest

from tests.test_issue_1931_delegate_children import DELEGATE, PARENT_VIEW, Agent, child_rows, host  # noqa: F401

PROMPT = {"role": "user", "content": "Count the ledger rows and report the total."}


def call(i):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]}


def result(i):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": f"ledger chunk {i}: 17 rows"}


def view(tool_calls, final=False):
    rows = [PROMPT]
    for i in range(1, tool_calls + 1):
        rows += [call(i), result(i)]
    return rows + ([{"role": "assistant", "content": "The ledger has 34 rows."}] if final else [])


class HostChild(Agent):
    """A delegated child whose hooks fire inside its own frame, with the host's payload fields."""

    def fire(self, ctx, name, history, **extra):
        for callback in ctx.hooks.get(name, []):
            callback(session_id=self.session_id, platform="subagent", conversation_history=list(history), **extra)

    def llm_start(self, ctx, history):
        self.fire(ctx, "pre_llm_call", history, user_message=PROMPT["content"], is_first_turn=True,
                  parent_session_id=self._parent_session_id, model="m")

    def api_call(self, ctx, history, n):
        self.fire(ctx, "pre_api_request", history, api_call_count=n, retry_count=0, model="m",
                  request_messages=list(history), message_count=len(history), task_id="t", turn_id="turn-1")

    def run_until(self, ctx, tool_calls):
        """Start the turn and issue model call k after k-1 tool rounds, up to ``tool_calls`` rounds done."""
        self.llm_start(ctx, view(0))
        for k in range(tool_calls + 1):
            self.api_call(ctx, view(k), k + 1)


def _setup(host, route):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    ctx.hooks["post_llm_call"][0](session_id="parent", platform="cli",
                                  conversation_history=PARENT_VIEW, context_compressor=parent)
    child = HostChild("child", **DELEGATE)
    engine = parent
    if route == "own_engine":
        engine = clone(ctx)
        child.start(engine)
    return load, clone, ctx, parent, child, engine


def _contents(rows):
    return [(row["role"], row["content"]) for row in rows]


ROUTES = ["own_engine", "parent_engine"]
# r7's three interrupted shapes: tool rounds completed before the interruption.
SHAPES = {"kill_third_model_call_in_flight": 2, "exit_after_one_call_and_result": 1, "exit_with_only_the_prompt": 0}


@pytest.mark.parametrize("route", ROUTES)
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


@pytest.mark.parametrize("route", ROUTES)
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
