"""#2440 (v0.30.0 named loss limit, fixed in v0.30.x): a delegated child's row compacted away after a non-busy
pre-compaction store failure is lost. The bypassed compressor writes nothing to LCM, and no later view holds the row.
Its summary is stored as the row's only trace (#2435). A busy failure skips the compaction instead (#2306)."""

import sqlite3

import pytest

from tests.test_issue_1931_delegate_children import child_rows
from tests.test_issue_1931_delegate_children import host  # noqa: F401 (the fixture)
from tests.test_issue_2071_delegate_tail_anchor import COMPACTED, X, _child, _once


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
        pytest.fail(f"setup: the stored rows changed: {contents[:len(head)]}")
    tail = contents[len(head):]
    if tail.count(X["content"]) == 1 and tail.count(later["content"]) == 1:
        return  # fixed: the strict marker turns this into a loud XPASS
    if tail == [COMPACTED[0]["content"], later["content"]]:
        raise AssertionError("#2440 known shape: X is lost; only its summary and the next row are stored")
    pytest.fail(f"not the #2440 shape: {tail}")


def test_the_same_compaction_with_a_stored_input_keeps_the_row(host, monkeypatch):  # noqa: F811
    """The control: the pre-store succeeds, so X is stored before the compaction drops it from the view."""
    ctx, _aux, engine, child, stored = _child(host)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: COMPACTED)
    result = child.compress(engine, [*stored, X])
    later = {"role": "assistant", "content": "reply after the summary"}
    child.turn(ctx, [*result, later], engine)
    contents = [row["content"] for row in child_rows(engine)]
    assert contents.count(X["content"]) == 1 and contents.count(later["content"]) == 1
