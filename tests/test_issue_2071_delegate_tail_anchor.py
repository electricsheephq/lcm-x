"""#2071 settling tests: a delegated child whose compression boundary is missing must not lose a new row X to the
shortened 3-2-1-row tail anchor when the row after X repeats the last stored row.
The busy pre-store case is safe through #2306 (that compaction is skipped). The others are safe through the #2071 anchor
check: every row in front of the stored-tail anchor must be stored or be one of the last compaction's outputs.

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


def test_a_concurrent_store_does_not_let_the_tail_anchor_skip_x(host, monkeypatch):  # noqa: F811
    """(c): another engine stores a row of the same child during the compaction, so the boundary is dropped."""
    ctx, aux, engine, child, stored = _child(host)
    load, clone = host
    other = clone(ctx)
    other.on_session_start("parent", platform="cli", context_length=200_000)
    concurrent = {"role": "assistant", "content": "concurrently stored row"}

    concurrent_stores = []

    def compress_with_store(*args):
        concurrent_stores.append(other._store_delegate_child_turn("child", [*stored, concurrent], platform="subagent") or 0)
        return COMPACTED

    monkeypatch.setattr(engine, "_compress_bypassed", compress_with_store)
    result = child.compress(engine, stored)
    if concurrent_stores != [1]:
        pytest.fail("setup: the other engine did not store the concurrent row")
    if result != COMPACTED or engine._delegate_child_key("child") in aux._DELEGATE_RESUME_BOUNDARIES:
        pytest.fail("setup: the compaction did not run without a boundary")
    assert _send_x_then_a_repeat(ctx, engine, child, result) == 1


def _compact_after_a_failed_pre_store(host, monkeypatch):
    ctx, aux, engine, child, stored = _child(host)
    fail, calls = _once(engine._append_delegate_child_rows, sqlite3.OperationalError("disk I/O error"))
    monkeypatch.setattr(engine, "_append_delegate_child_rows", fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, stored)
    if not calls or result != COMPACTED or engine._delegate_child_key("child") in aux._DELEGATE_RESUME_BOUNDARIES:
        pytest.fail("setup: the compaction did not run without a boundary")
    return ctx, aux, engine, child, stored, result


def test_the_compacted_head_is_not_stored_and_the_repeat_follows_x(host, monkeypatch):  # noqa: F811
    """The counted fallback consumes the compaction's own output: rows end [..., last, X, repeat]."""
    ctx, aux, engine, child, stored, result = _compact_after_a_failed_pre_store(host, monkeypatch)
    assert _send_x_then_a_repeat(ctx, engine, child, result) == 1
    assert [row["content"] for row in child_rows(engine)] == [
        *(row["content"] for row in stored), X["content"], stored[-1]["content"]]


def test_the_output_record_follows_a_compression_handoff(host, monkeypatch):  # noqa: F811
    """A failed boundary record, then the child rotates: the new id still knows the compacted head."""
    ctx, aux, engine, child, stored = _child(host)
    fail, calls = _once(engine._remember_delegate_child_boundary, RuntimeError("synthetic boundary failure"))
    monkeypatch.setattr(engine, "_remember_delegate_child_boundary", fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, stored)
    if not calls or result != COMPACTED:
        pytest.fail("setup: the compaction did not run")
    child.rotate(engine, "child-2")
    if engine._delegate_child_key("child-2") not in aux._DELEGATE_COMPACTION_OUTPUTS:
        pytest.fail("setup: the handoff did not run")
    repeat = {"role": stored[-1]["role"], "content": stored[-1]["content"]}
    for _ in range(2):
        child.turn(ctx, [*result, X, repeat], engine)
    assert [row["content"] for row in child_rows(engine, "child-2")] == [X["content"], repeat["content"]]


def test_after_a_restart_the_compacted_head_is_stored_once_and_x_once(host, monkeypatch):  # noqa: F811
    """The output record is in-process only: a new process stores the head once (a duplicate, never a loss)."""
    ctx, aux, engine, child, stored, result = _compact_after_a_failed_pre_store(host, monkeypatch)
    load, clone = host
    ctx2, _ = load()  # fresh module and engines: no in-memory boundary or output record
    resumed = clone(ctx2)
    resumed.on_session_start("parent", platform="cli", context_length=200_000)
    assert _send_x_then_a_repeat(ctx2, resumed, child, result) == 1
    assert [row["content"] for row in child_rows(resumed)] == [
        *(row["content"] for row in stored), COMPACTED[0]["content"], X["content"], stored[-1]["content"]]
