"""#2439 item 1: after a second restart with an identical rewritten view and no new row, a delegated child's kept
rows are stored again. That is surplus, never a loss. A rejected anchor stores the rewritten head after the kept
rows, so the stored tail ends on a row in front of the anchor. Only the in-memory boundary protects the next turn,
and a restart loses it. The fix (a durable boundary, or reconciling the stored suffix) is v0.30.x work."""

import pytest

from tests.test_issue_1931_delegate_children import DELEGATE, Agent, child_rows, child_view
from tests.test_issue_1931_delegate_children import host  # noqa: F401


def _restart(load, clone):
    ctx, _module = load()  # fresh module and engines: no in-memory boundary or output record
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    return ctx, engine


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2439 item 1: a second restart re-stores the kept rows (surplus, never a loss)")
def test_a_second_restart_with_the_same_rewritten_view_stores_nothing_again(host):  # noqa: F811
    load, clone = host
    ctx, engine = _restart(load, clone)
    child = Agent("child", **DELEGATE)
    child.turn(ctx, child_view(2), engine)
    stored = [row["content"] for row in child_rows(engine)]
    # The host rewrote the head and kept the last two rows; no row follows them.
    view = [{"role": "user", "content": "[CONTEXT COMPACTION] rewritten head"}, *child_view(2)[-2:]]
    ctx, engine = _restart(load, clone)
    child.turn(ctx, view, engine)
    if [row["content"] for row in child_rows(engine)] != [*stored, view[0]["content"]]:
        pytest.fail("setup: the first restart did not store the rewritten head once, behind the kept rows")
    ctx, engine = _restart(load, clone)
    child.turn(ctx, view, engine)
    assert [row["content"] for row in child_rows(engine)] == [*stored, view[0]["content"]]
