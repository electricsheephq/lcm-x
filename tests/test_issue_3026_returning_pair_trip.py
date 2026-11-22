"""#3026 (v0.30.2 regression against v0.30.1): a stored unstamped tool pair leaves the host list, then comes back below
the cursor in the same session. The previous ingest's trip witness no longer holds it, so a tripped presence lookup
stores the pair a second time. v0.30.1 stores no copy.

Expectations fixed before any run: the returning pair keeps exactly one stored copy, and an unstamped pair the store
never held is still stored exactly once (no loss). The restart route named in #3026 is not covered here: a cold bind
never runs the changed-prefix audit, so it cannot reach the trip helper."""

from __future__ import annotations

import pytest

from hermes_lcm import identity_anchor
from tests.test_anchor_unstored_before_cursor import (  # noqa: F401 (autouse fixture)
    _a, _engine, _fresh_presence_notices, _key, _result, _u,
)
from tests.test_topic9_loss_cells_t1486 import _counts, _tool_history, _trips, _unstamped
from tests.test_topic9_loss_clause import _call


def _odd_shape_call(call_id) -> dict:
    """Arguments sent as a JSON object, not text: stored as is, a non-standard tool shape, so the presence lookup trips
    at its shipped budgets."""
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": {"path": "odd.txt"}}}]}


def _history(trip):
    rows = _tool_history()
    if trip == "odd_shape":
        rows = [*rows[:2], _odd_shape_call("call_odd"), _unstamped(_result("call_odd", "odd result", 0)), *rows[2:]]
    return rows


def _run(tmp_path, monkeypatch, trip, returning):
    engine = _engine(tmp_path)
    rows = _history(trip)
    engine.ingest(list(rows))
    at = next(i for i, row in enumerate(rows) if (row.get("tool_calls") or [{}])[0].get("id") == "call_d0")
    without = [*rows[:at], *rows[at + 2:]]
    leave = _a("a reply while the pair is gone", 1500)
    engine.ingest([*without, leave])  # the host dropped the pair; this ingest's witness no longer holds it
    before = _counts(engine)
    if trip == "window":  # a long session: the stored rows exceed the presence window
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 1, raising=False)
    trips = _trips(monkeypatch, engine)
    pair = returning(rows[at:at + 2])
    engine.ingest([*rows[:at], *pair, *rows[at + 2:], leave, _a("a new reply", 1600)])
    assert trips, "setup: the presence lookup tripped"
    return engine, before, pair


@pytest.mark.parametrize("trip", ["odd_shape", "window"])
def test_a_returning_stored_pair_is_not_stored_again(tmp_path, monkeypatch, trip):
    engine, before, pair = _run(tmp_path, monkeypatch, trip, lambda stored: stored)
    try:
        stored = _counts(engine)
        assert [stored[_key(row)] for row in pair] == [before[_key(row)] for row in pair] == [1, 1]
        assert not before - stored, "no stored row is lost"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("trip", ["odd_shape", "window"])
def test_a_never_stored_pair_in_the_same_slot_is_stored_once(tmp_path, monkeypatch, trip):
    """No-loss guard: the same route with a pair the store never held."""
    new = [_unstamped(_call("call_never", 0)), _unstamped(_result("call_never", "never stored", 0))]
    engine, before, pair = _run(tmp_path, monkeypatch, trip, lambda stored: new)
    try:
        stored = _counts(engine)
        assert [stored[_key(row)] for row in new] == [1, 1]
        assert not before - stored, "no stored row is lost"
    finally:
        engine.shutdown()
