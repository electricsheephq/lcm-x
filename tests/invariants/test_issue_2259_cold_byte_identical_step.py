"""#2259 named loss limit (v0.30.0): a fresh tool step byte-identical to stored rows is skipped on a cold start.

Engine-direct, the issue's steps: store ``[assistant call c, tool result c "ok", user "again"]``, restart
cold, then ingest a view holding a fresh occurrence of the same three rows with new host stamps or uids.
Main's legacy reconcile matches the fresh rows by replay identity only, so it stores neither the tool pair
nor the following user row. The strict xfail turns green with the P2 classifier fix (v0.30.x, with #2377
and #2414) and must then become an ordinary test. The control pins the replay that main correctly skips.
"""
from copy import deepcopy

import pytest

from tests.invariants.test_1232_classifier_twins import A, T, U, factory as factory, witness  # noqa: F401

pytestmark = pytest.mark.issue(2259)

FIRST = [(1, A()), (2, T()), (3, U("again"))]
FRESH = [(4, A()), (5, T()), (6, U("again"))]


def _cold(factory, kind, mode, second):
    engine = factory("case", mode)
    engine.ingest([witness(occ, row, kind) for occ, row in FIRST])
    if engine._ingest_failure_count:
        pytest.fail(f"setup: ingest failed: {engine._last_ingest_error}")
    engine = factory("case", mode, cold_of=engine)
    before = engine._store.get_session_messages("S")
    if len(before) != len(FIRST):
        pytest.fail("setup: the first view must store each row once")
    engine.ingest(deepcopy([witness(occ, row, kind) for occ, row in second]))
    if engine._ingest_failure_count:
        pytest.fail(f"setup: ingest failed: {engine._last_ingest_error}")
    after = engine._store.get_session_messages("S")
    if after[:len(before)] != before:
        pytest.fail("setup: the stored prefix changed")
    return [(row["role"], row.get("content") or "", row.get("tool_call_id")) for row in after[len(before):]]


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2259 named loss limit (v0.30.0): a fresh byte-identical tool step after a cold start "
                          "is skipped with the user row after it, even with new host stamps or uids")
@pytest.mark.parametrize("kind,mode", [("stamp", "shadow"), ("uid", "on"), ("gw", "on")])
def test_fresh_byte_identical_tool_step_after_a_cold_start_is_stored(factory, kind, mode):
    added = _cold(factory, kind, mode, FRESH)
    if added and added != [("assistant", "", None), ("tool", "ok", "c"), ("user", "again", None)]:
        pytest.fail(f"not the #2259 shape: {added}")
    # Main stores nothing: the tool pair AND the following user row are both lost.
    assert [r[0] for r in added] == ["assistant", "tool", "user"], f"#2259 known shape: added={added}"


@pytest.mark.parametrize("kind,mode", [("stamp", "shadow"), ("uid", "on"), ("gw", "on")])
def test_replay_of_the_stored_step_after_a_cold_start_is_skipped(factory, kind, mode):
    """Control (passes on main): the same occurrence replayed with its own witnesses stores nothing."""
    assert _cold(factory, kind, mode, FIRST) == []
