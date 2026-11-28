"""#2440 (v0.30.0 named loss limit, fixed in v0.30.x): a delegated child's row compacted away after a non-busy
pre-compaction store failure is lost. The bypassed compressor writes nothing to LCM, and no later view holds the row.
Its summary is stored as the row's only trace (#2435). A busy failure skips the compaction instead (#2306)."""

import sqlite3

import pytest

from tests.test_issue_1931_delegate_children import child_rows
from tests.test_issue_1931_delegate_children import host  # noqa: F401 (the fixture)
from tests.test_issue_2071_delegate_tail_anchor import COMPACTED, X, _child, _once


def _expect_recovered_tail(tail, later):
    # X precedes the compaction output in host chronology; a fix may omit the stored summary.
    if tail in ([X["content"], later], [X["content"], COMPACTED[0]["content"], later]):
        return  # fixed: the strict marker turns this into a loud XPASS
    if tail == [COMPACTED[0]["content"], later]:
        raise AssertionError("#2440 known shape: X is lost; only its summary and the next row are stored")
    pytest.fail(f"not the #2440 shape: tail rows={len(tail)}, X copies={tail.count(X['content'])}, "
                f"summary copies={tail.count(COMPACTED[0]['content'])}, later copies={tail.count(later)}")


@pytest.mark.parametrize("tail,outcome", [
    pytest.param(["summary", "later"], AssertionError, id="known"),
    pytest.param(["X", "later"], None, id="fixed-without-summary"),
    pytest.param(["X", "summary", "later"], None, id="fixed-with-summary"),
    pytest.param(["later", "X"], pytest.fail.Exception, id="X-after-later"),
    pytest.param(["summary", "X", "later"], pytest.fail.Exception, id="X-after-summary"),
    pytest.param(["X", "summary", "summary", "later"], pytest.fail.Exception, id="head-stored-again"),
    pytest.param(["X", "X", "later"], pytest.fail.Exception, id="X-twice"),
    pytest.param(["X", "later", "later"], pytest.fail.Exception, id="later-twice"),
    pytest.param(["X", "later", "summary"], pytest.fail.Exception, id="later-not-last"),
    pytest.param(["X", "other", "later"], pytest.fail.Exception, id="unexpected-row"),
])
def test_recovered_tail_accepts_only_the_known_or_fixed_shape(tail, outcome):
    values = {"X": X["content"], "summary": COMPACTED[0]["content"], "later": "next reply", "other": "other"}
    stored = [values[row] for row in tail]
    if outcome is None:
        _expect_recovered_tail(stored, values["later"])
    else:
        with pytest.raises(outcome):
            _expect_recovered_tail(stored, values["later"])


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2440: a row compacted away after a non-busy pre-store failure is lost (its summary is kept)")
def test_a_row_compacted_away_after_a_failed_pre_store_is_stored(host, monkeypatch):  # noqa: F811
    ctx, _aux, engine, child, stored = _child(host)
    fail, calls = _once(engine._append_delegate_child_rows, sqlite3.OperationalError("disk I/O error"))
    monkeypatch.setattr(engine, "_append_delegate_child_rows", fail)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, [*stored, X])  # X is new; its pre-store fails, then the compaction drops it
    if not calls or result != COMPACTED:
        pytest.fail("setup: the compaction did not run after a failed, non-busy pre-store")
    later = {"role": "assistant", "content": "reply after the summary"}
    child.turn(ctx, [*result, later], engine)
    contents = [row["content"] for row in child_rows(engine)]
    head = [row["content"] for row in stored]
    if contents[:len(head)] != head:
        pytest.fail("setup: the stored rows changed")
    _expect_recovered_tail(contents[len(head):], later["content"])


def test_the_same_compaction_with_a_stored_input_keeps_the_row(host, monkeypatch):  # noqa: F811
    """The control: the pre-store succeeds, so X is stored before the compaction drops it from the view."""
    ctx, _aux, engine, child, stored = _child(host)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, [*stored, X])
    later = {"role": "assistant", "content": "reply after the summary"}
    child.turn(ctx, [*result, later], engine)
    contents = [row["content"] for row in child_rows(engine)]
    assert contents.count(X["content"]) == 1 and contents.count(later["content"]) == 1
