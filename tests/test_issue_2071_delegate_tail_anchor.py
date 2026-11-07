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


def _compact_after_a_failed_pre_store(host, monkeypatch):  # noqa: F811
    ctx, aux, engine, child, stored = _child(host)
    fail, calls = _once(engine._append_delegate_child_rows, sqlite3.OperationalError("disk I/O error"))
    monkeypatch.setattr(engine, "_append_delegate_child_rows", fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, stored)
    if not calls or result != COMPACTED or engine._delegate_child_key("child") in aux._DELEGATE_RESUME_BOUNDARIES:
        pytest.fail("setup: the compaction did not run without a boundary")
    return ctx, aux, engine, child, stored, result


def test_after_a_failed_pre_store_the_compacted_head_is_stored_once_and_x_once(host, monkeypatch):  # noqa: F811
    """After a failed pre-store the compaction's outputs are never recorded, even when the input was stored earlier: a
    copy of the last stored row behind an unstored one would read as covered (#2071). So the head is stored once, a
    visible surplus and never a loss, and X once. The copy of the last stored row is the anchor itself."""
    ctx, aux, engine, child, stored, result = _compact_after_a_failed_pre_store(host, monkeypatch)
    assert _send_x_then_a_repeat(ctx, engine, child, result) == 1
    assert [row["content"] for row in child_rows(engine)] == [
        *(row["content"] for row in stored), COMPACTED[0]["content"], X["content"]]


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
    assert [row["content"] for row in child_rows(engine, "child-2")] == [X["content"]]


def test_after_a_restart_the_compacted_head_is_stored_once_and_x_once(host, monkeypatch):  # noqa: F811
    """The output record is in-process only: a new process stores the head once (a duplicate, never a loss)."""
    ctx, aux, engine, child, stored, result = _compact_after_a_failed_pre_store(host, monkeypatch)
    load, clone = host
    ctx2, _ = load()  # fresh module and engines: no in-memory boundary or output record
    resumed = clone(ctx2)
    resumed.on_session_start("parent", platform="cli", context_length=200_000)
    assert _send_x_then_a_repeat(ctx2, resumed, child, result) == 1
    assert [row["content"] for row in child_rows(resumed)] == [
        *(row["content"] for row in stored), COMPACTED[0]["content"], X["content"]]


def test_a_failed_output_record_drops_the_earlier_compactions_outputs(host, monkeypatch):  # noqa: F811
    """A second compaction whose output and boundary records both fail must not leave the first compaction's head
    exempt: that head, sent again as a new row, is stored."""
    ctx, aux, engine, child, stored = _child(host)
    first, second = [{"role": "user", "content": "first compacted head"}], [{"role": "user", "content": "second head"}]
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: first)
    if child.compress(engine, stored) != first or not aux._DELEGATE_COMPACTION_OUTPUTS.get(engine._delegate_child_key("child")):
        pytest.fail("setup: the first compaction's outputs were not recorded")
    later = {"role": "assistant", "content": "a row after the first compaction"}
    child.turn(ctx, [*first, later], engine)  # stored before the failures, so the second pass's pre-store succeeds
    calls = {}
    for name in ("_remember_delegate_child_outputs", "_remember_delegate_child_boundary"):
        fail, calls[name] = _once(getattr(engine, name), RuntimeError(f"synthetic {name} failure"))
        monkeypatch.setattr(engine, name, fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: second)
    result = child.compress(engine, [*first, later])  # the first head is exempt, so it stays unstored
    rows = [row["content"] for row in child_rows(engine)]
    if result != second or rows.count(first[0]["content"]) or rows[-1] != later["content"]:
        pytest.fail("setup: the second compaction did not run over a stored input with an unstored first head")
    if not all(calls.values()):
        pytest.fail("setup: the second pass did not reach both the output record and the boundary record")
    after = [{"role": "assistant" if i % 2 else "user", "content": f"after the second compaction {i}"} for i in range(3)]
    child.turn(ctx, [*second, *after], engine)  # stored: the anchor below can cover the second head and these rows
    for _ in range(2):
        child.turn(ctx, [*second, *first, *after], engine)
    assert [row["content"] for row in child_rows(engine)].count(first[0]["content"]) == 1


def test_kept_first_rows_and_a_repeated_new_row_survive_a_restart(host):  # noqa: F811
    """The real compactors keep the first rows (protect_first_n) and add a summary. After a restart (no boundary, no
    output record) the anchor check finds the kept rows stored in order, stores only the summary in front, and keeps
    the new row behind the anchor even though it repeats an older stored row."""
    load, clone = host
    ctx, _module = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    stored = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(12)]
    stored[5] = {"role": "assistant", "content": "Let me check."}
    child.turn(ctx, stored, engine)
    if [row["content"] for row in child_rows(engine)] != [row["content"] for row in stored]:
        pytest.fail("setup: the child's first turn was not stored")
    ctx2, _ = load()  # fresh module and engines: no in-memory boundary or output record
    resumed = clone(ctx2)
    resumed.on_session_start("parent", platform="cli", context_length=200_000)
    summary = {"role": "user", "content": "[CONTEXT SUMMARY] turns 3 to 5"}
    repeat = {"role": "assistant", "content": "Let me check."}
    view = [*stored[:3], summary, *stored[6:], repeat]
    for _ in range(2):
        child.turn(ctx2, view, resumed)
    assert [row["content"] for row in child_rows(resumed)] == [
        *(row["content"] for row in stored), summary["content"], repeat["content"]]


def test_a_rejection_without_a_boundary_does_not_store_the_kept_tail_again(host):  # noqa: F811
    """After a restart (no boundary), a rejection stores the summary after the kept tail, out of view order. The view's
    end is then recorded as the boundary, so the next turn stores only its new row, not the kept tail a second time."""
    load, clone = host
    ctx, _module = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    stored = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(12)]
    child.turn(ctx, stored, engine)
    ctx2, _ = load()  # fresh module and engines: no in-memory boundary or output record
    resumed = clone(ctx2)
    resumed.on_session_start("parent", platform="cli", context_length=200_000)
    summary = {"role": "user", "content": "[CONTEXT SUMMARY] turns 3 to 5"}
    view = [*stored[:3], summary, *stored[6:]]
    child.turn(ctx2, view, resumed)
    new = {"role": "user", "content": "a new turn"}
    for _ in range(2):
        child.turn(ctx2, [*view, new], resumed)
    assert [row["content"] for row in child_rows(resumed)] == [
        *(row["content"] for row in stored), summary["content"], new["content"]]



@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#2422: with many equal rows, a boundary recorded after a "
                                                              "rejection can become ambiguous in a later view, and the "
                                                              "counted fallback then credits a hidden stored row against a "
                                                              "new one (main fails this sequence one view earlier)")
def test_a_view_that_extends_the_last_one_appends_exactly_the_extension(host):  # noqa: F811
    """Many equal rows after a restart (no boundary, no output record): each view extends the one before by a few new
    rows, and exactly those rows must be appended."""
    load, clone = host
    ctx, _module = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    child.turn(ctx, [{"role": "user", "content": c} for c in "CCBA"], engine)
    ctx2, _ = load()  # fresh module and engines: no in-memory boundary or output record
    resumed = clone(ctx2)
    resumed.on_session_start("parent", platform="cli", context_length=200_000)
    child.turn(ctx2, [{"role": "user", "content": c} for c in "CABABB"], resumed)
    previous = "CABABB"
    for letters in ("CABABBAB", "CABABBABCHC", "CABABBABCHCHCC"):
        before = "".join(row["content"] for row in child_rows(resumed))
        child.turn(ctx2, [{"role": "user", "content": c} for c in letters], resumed)
        after = "".join(row["content"] for row in child_rows(resumed))
        assert after == before + letters[len(previous):], (letters, before, after)
        previous = letters


def test_a_summary_of_an_unstored_input_row_is_stored(host, monkeypatch):  # noqa: F811
    """NeonDiff's case: the pre-store fails (not busy) while the compaction input holds a request that was never
    stored. The compaction replaces it with its summary, so the summary is that request's only survivor: it must not be
    treated as an output of stored rows, and the next turn stores it."""
    ctx, aux, engine, child, stored = _child(host)
    unstored = {"role": "user", "content": "request never stored"}
    fail, calls = _once(engine._append_delegate_child_rows, sqlite3.OperationalError("disk I/O error"))
    monkeypatch.setattr(engine, "_append_delegate_child_rows", fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, [*stored, unstored])
    contents = [row["content"] for row in child_rows(engine)]
    if not calls or result != COMPACTED or unstored["content"] in contents:
        pytest.fail("setup: the compaction did not run over an unstored input row")
    later = {"role": "assistant", "content": "reply after the summary"}
    child.turn(ctx, [*result, later], engine)
    contents = [row["content"] for row in child_rows(engine)]
    assert contents.count(COMPACTED[0]["content"]) == 1 and contents.count(later["content"]) == 1


def test_a_later_compactions_head_is_not_stored_after_a_failed_boundary_record(host, monkeypatch):  # noqa: F811
    """The first compaction's head is never stored (the boundary resumes after it), so it is in every later input. A
    second compaction over a covered input still records its outputs: when its boundary record fails, the next turn
    stores only the new row, not the second head."""
    ctx, aux, engine, child, stored = _child(host)
    first, second = [{"role": "user", "content": "first head"}], [{"role": "user", "content": "second head"}]
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: first)
    if child.compress(engine, stored) != first:
        pytest.fail("setup: the first compaction did not run")
    later = {"role": "assistant", "content": "later"}
    child.turn(ctx, [*first, later], engine)
    fail, calls = _once(engine._remember_delegate_child_boundary, RuntimeError("synthetic boundary failure"))
    monkeypatch.setattr(engine, "_remember_delegate_child_boundary", fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: second)
    result = child.compress(engine, [*first, later])
    if not calls or result != second:
        pytest.fail("setup: the second compaction did not run with a failed boundary record")
    after = {"role": "assistant", "content": "after the second compaction"}
    for _ in range(2):
        child.turn(ctx, [*result, after], engine)
    contents = [row["content"] for row in child_rows(engine)]
    assert contents.count(second[0]["content"]) == 0 and contents.count(after["content"]) == 1


def test_an_unstored_last_row_equal_to_an_older_row_keeps_its_summary(host, monkeypatch):  # noqa: F811
    """The input's last row is a new turn whose identity equals an older stored row, and the pre-store fails: the
    input is not covered (its last row is not the stored tail), so the summary is stored as that turn's only trace."""
    ctx, aux, engine, child, stored = _child(host)
    again = {"role": "user", "content": stored[0]["content"]}  # equal to a stored row, but a new turn
    fail, calls = _once(engine._append_delegate_child_rows, sqlite3.OperationalError("disk I/O error"))
    monkeypatch.setattr(engine, "_append_delegate_child_rows", fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, [*stored, again])
    if not calls or result != COMPACTED:
        pytest.fail("setup: the compaction did not run after a failed pre-store")
    later = {"role": "assistant", "content": "reply after the summary"}
    child.turn(ctx, [*result, later], engine)
    contents = [row["content"] for row in child_rows(engine)]
    assert contents.count(COMPACTED[0]["content"]) == 1 and contents.count(later["content"]) == 1
