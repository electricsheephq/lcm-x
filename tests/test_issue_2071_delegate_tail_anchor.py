"""#2071 settling tests: a delegated child whose compression boundary is missing must not lose a new row X to the
shortened 3-2-1-row tail anchor when the row after X repeats the last stored row.
On main only the busy pre-store case is safe (#2306 skips that compaction); the others are strict xfails until the
#2071 fix.

The boundary goes missing in one child session when (a) the pre-compaction store fails, (b) recording the boundary
raises, or (c) a concurrent store moves the stored tail during the compaction. Each case compacts past the whole
stored tail, then sends [compacted head, X, a copy of the last stored row]; X must be stored once.
"""
import sqlite3

import pytest

from tests.test_issue_1931_delegate_children import DELEGATE, Agent, child_rows
from tests.test_issue_1931_delegate_children import host  # noqa: F401 (the fixture)

COMPACTED = [{"role": "user", "content": "compressed head"}]  # none of the stored tail rows remains
X = {"role": "user", "content": "new request X"}


def _child(harness):
    load, clone = harness
    ctx, module = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    stored = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "payload " * 400}
              for i in range(12)]
    child.turn(ctx, stored, engine)
    if [row["content"] for row in child_rows(engine)] != [row["content"] for row in stored]:
        pytest.fail("setup: the child's first turn was not stored")
    return ctx, module("aux_session"), engine, child, stored


def _once(original, error):
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise error
        return original(*args, **kwargs)

    return fail_once, calls


def _send_x_then_a_repeat(ctx, engine, child, result):
    repeat = {"role": child_rows(engine)[-1]["role"], "content": child_rows(engine)[-1]["content"]}
    for _ in range(2):
        child.turn(ctx, [*result, X, repeat], engine)
    return [row["content"] for row in child_rows(engine)].count(X["content"])


def test_a_busy_pre_store_skips_the_compaction_so_x_is_stored(host, monkeypatch):  # noqa: F811
    """(a), busy: #2306 keeps the view when the pre-store hits a busy lock, so no compaction drops the tail."""
    ctx, aux, engine, child, stored = _child(host)
    fail, calls = _once(engine._append_delegate_child_rows, sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(engine, "_append_delegate_child_rows", fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, stored)
    assert calls, "the pre-compaction store was not attempted"
    assert result == stored  # compaction skipped: input not stored
    assert _send_x_then_a_repeat(ctx, engine, child, result) == 1


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#2071: a non-busy pre-store failure still compacts and the shortened tail "
                                                              "anchor skips X")
def test_a_failed_pre_store_does_not_let_the_tail_anchor_skip_x(host, monkeypatch):  # noqa: F811
    """(a), any other error (disk I/O, disk full): the compaction runs with no boundary recorded."""
    ctx, aux, engine, child, stored = _child(host)
    fail, calls = _once(engine._append_delegate_child_rows, sqlite3.OperationalError("disk I/O error"))
    monkeypatch.setattr(engine, "_append_delegate_child_rows", fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, stored)
    if not calls or result != COMPACTED or engine._delegate_child_key("child") in aux._DELEGATE_RESUME_BOUNDARIES:
        pytest.fail("setup: the compaction did not run without a boundary")
    assert _send_x_then_a_repeat(ctx, engine, child, result) == 1


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#2071: when recording the boundary raises, the shortened tail anchor skips X")
def test_a_failed_boundary_record_does_not_let_the_tail_anchor_skip_x(host, monkeypatch):  # noqa: F811
    """(b): recording the boundary raises after the compaction."""
    ctx, aux, engine, child, stored = _child(host)
    fail, calls = _once(engine._remember_delegate_child_boundary, RuntimeError("synthetic boundary failure"))
    monkeypatch.setattr(engine, "_remember_delegate_child_boundary", fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, stored)
    if not calls or result != COMPACTED or engine._delegate_child_key("child") in aux._DELEGATE_RESUME_BOUNDARIES:
        pytest.fail("setup: the compaction did not run without a boundary")
    assert _send_x_then_a_repeat(ctx, engine, child, result) == 1


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#2071: a concurrent store drops the boundary, and the shortened tail anchor "
                                                              "skips X")
def test_a_concurrent_store_does_not_let_the_tail_anchor_skip_x(host, monkeypatch):  # noqa: F811
    """(c): another engine stores a row of the same child during the compaction, so the boundary is dropped."""
    ctx, aux, engine, child, stored = _child(host)
    load, clone = host
    other = clone(ctx)
    other.on_session_start("parent", platform="cli", context_length=200_000)
    concurrent = {"role": "assistant", "content": "concurrently stored row"}

    def compress_with_store(*args):
        assert (other._store_delegate_child_turn("child", [*stored, concurrent], platform="subagent") or 0) == 1
        return COMPACTED

    monkeypatch.setattr(engine, "_compress_bypassed", compress_with_store)
    result = child.compress(engine, stored)
    if result != COMPACTED or engine._delegate_child_key("child") in aux._DELEGATE_RESUME_BOUNDARIES:
        pytest.fail("setup: the compaction did not run without a boundary")
    assert _send_x_then_a_repeat(ctx, engine, child, result) == 1
