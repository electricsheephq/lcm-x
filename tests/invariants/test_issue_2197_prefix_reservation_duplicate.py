"""#2197 duplicate shape, a named surplus limit (v0.30.0): the whole-prefix reservation counts a prefix copy outside
the stored tail, so an already-stored tool group replayed after the cold-bind cursor is stored again with no link.

Engine-direct through ``LCMEngine.ingest``, as the issue's probe ran it: tool-call ids reused, the cold-bind cursor
forced to 2, the prefix copy of the pair outside the 64-row stored tail, and the host's view (stale, anchored on the
tail) replaying the stored post-cursor pair. The strict xfail turns green with the fix (v0.30.x) and must then become
an ordinary test. The control keeps the prefix copy inside the tail: the replayed pair is skipped.
"""
from copy import deepcopy

import pytest

from tests.invariants.test_issue_1529_warm_proof_salvage import build as build  # noqa: F401
from tests.invariants.test_issue_2163_compacted_cursor import _replay_links
from tests.test_issue_671_stub_first_exit import tool_pair

pytestmark = pytest.mark.issue(2197)


def _replay_after_a_forced_cursor(build, monkeypatch, filler):  # noqa: F811
    call, result = tool_pair("call_0", "OK", tool_name="poll")
    rows = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"filler row {i}"} for i in range(filler)]
    engine = build()
    engine.ingest(deepcopy([call, result, *rows, call, result]))
    before = engine._store.get_session_messages("S")
    if len(before) != filler + 4:
        pytest.fail("setup: every stored occurrence must be stored once")
    engine.shutdown()
    engine = build()
    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *a, **kw: 2)
    monkeypatch.setattr(engine, "_ledger_cold_cursor", lambda *a, **kw: dict(k=2, pairs={}))
    engine.ingest(deepcopy([call, result, call, result, {"role": "user", "content": "next"}]))
    if engine._ingest_failure_count:
        pytest.fail(f"setup: ingest failed: {engine._last_ingest_error}")
    after = engine._store.get_session_messages("S")
    if after[:len(before)] != before:
        pytest.fail("setup: the stored prefix changed")
    added = after[len(before):]
    if [row["content"] for row in added if row["role"] == "user"] != ["next"]:
        pytest.fail("not the #2197 shape: the new turn must be stored once")
    links = _replay_links(engine)
    unlinked = [row["store_id"] for row in added if row["role"] != "user" and row["store_id"] not in links]
    return added, unlinked


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2197 named surplus limit (v0.30.0): with the prefix copy outside the stored tail, a "
                          "replayed post-cursor tool group is stored again with no link")
def test_a_replayed_group_after_an_out_of_tail_prefix_copy_is_not_a_silent_copy(build, monkeypatch):  # noqa: F811
    added, unlinked = _replay_after_a_forced_cursor(build, monkeypatch, filler=70)
    if [row["role"] for row in added] not in (["user"], ["assistant", "tool", "user"]):
        pytest.fail(f"not the #2197 shape: {[row['role'] for row in added]}")
    assert unlinked == [], f"#2197 known shape: the replayed call and result are stored again unlinked {unlinked}"


def test_the_same_replay_with_the_prefix_copy_in_the_tail_is_skipped(build, monkeypatch):  # noqa: F811
    """Control (passes on main)."""
    added, unlinked = _replay_after_a_forced_cursor(build, monkeypatch, filler=0)
    assert [row["role"] for row in added] == ["user"] and unlinked == []
