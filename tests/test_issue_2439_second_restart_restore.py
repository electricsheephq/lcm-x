"""#2439 item 1: after a delegated child's stored-tail anchor is rejected, a later restart can store the kept rows
again. That is surplus, never a loss. The rejection stores the rewritten head after the kept rows, so the stored tail
ends on a row in front of the anchor. Only the in-memory boundary protects the next turn, and a restart loses it. Here
the rejection runs in a cold process, so the shape needs a second restart; after a warm rejection one restart is
enough. The fix (a durable boundary, or reconciling the stored suffix) is v0.30.x work."""

import pytest

from tests.test_issue_1931_delegate_children import DELEGATE, Agent, child_rows, child_view
from tests.test_issue_1931_delegate_children import host  # noqa: F401


def _restart(load, clone):
    ctx, _module = load()  # fresh module and engines: no in-memory boundary or output record
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    return ctx, engine


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2439 item 1: a restart after a rejected anchor re-stores the kept rows (surplus, never a loss)")
def test_a_restart_after_a_rejected_anchor_stores_nothing_again(host):  # noqa: F811
    load, clone = host
    ctx, engine = _restart(load, clone)
    child = Agent("child", **DELEGATE)
    child.turn(ctx, child_view(2), engine)
    stored = [row["content"] for row in child_rows(engine)]
    if stored != [m["content"] for m in child_view(2)]:
        pytest.fail("setup: the child's first view was not stored as sent")
    # The host rewrote the head and kept the last two rows; no row follows them.
    view = [{"role": "user", "content": "[CONTEXT COMPACTION] rewritten head"}, *child_view(2)[-2:]]
    ctx, engine = _restart(load, clone)
    child.turn(ctx, view, engine)
    if [row["content"] for row in child_rows(engine)] != [*stored, view[0]["content"]]:
        pytest.fail("setup: the first restart did not store the rewritten head once, behind the kept rows")
    ctx, engine = _restart(load, clone)
    child.turn(ctx, view, engine)
    rows = [row["content"] for row in child_rows(engine)]
    desired = [*stored, view[0]["content"]]
    if rows == desired:
        return  # fixed: the strict marker turns this into a loud XPASS
    if rows == [*desired, *(m["content"] for m in view[1:])]:
        raise AssertionError("#2439 item 1: the kept rows were stored again")
    pytest.fail(f"not the #2439 surplus shape (a different defect, possibly a loss): {rows[len(stored):]}")


def test_a_second_restart_stores_nothing_again_when_a_new_row_follows_the_kept_rows(host):  # noqa: F811
    """The control: the same two restarts, with a new row after the kept rows. That row is the stored tail, so the
    anchor resumes after it and nothing is stored again."""
    load, clone = host
    ctx, engine = _restart(load, clone)
    child = Agent("child", **DELEGATE)
    child.turn(ctx, child_view(2), engine)
    stored = [row["content"] for row in child_rows(engine)]
    view = [{"role": "user", "content": "[CONTEXT COMPACTION] rewritten head"}, *child_view(2)[-2:],
            {"role": "assistant", "content": "after the rewrite"}]
    for _ in range(2):
        ctx, engine = _restart(load, clone)
        child.turn(ctx, view, engine)
    assert [row["content"] for row in child_rows(engine)] == [*stored, view[0]["content"], view[-1]["content"]]
