"""Pins the #2377 v7 attribution oracle's scope (release manager condition on 2c80642f).

Per-row checks apply to the rows the product CAN tell apart: a ledger-bound occurrence, or a stamped
copy unique among the added rows of its key that has an equal-stamp original; such a replay must carry
its own link, to that original when exactly one exists. Every other replay pools with its scorer key,
across stamps (release-manager ruling: a group is never widened across stamps that carry evidence).
These tests run the oracle on synthetic rows, so a later edit cannot silently move either boundary.
"""

import sqlite3
from types import SimpleNamespace

from tests.invariants._ingest_shapes_2377 import Run


def _row(store_id, stamp=None, content="D"):
    return {"store_id": store_id, "role": "tool", "content": content, "tool_call_id": "d",
            "tool_calls": None, "tool_name": None, "conversation_id": "conversation",
            "observed_at": stamp}


def _run(before, added, inserted, links=(), pending=(), bound=()):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE message_relations (relation_id INTEGER PRIMARY KEY, store_id INTEGER, "
                 "kind TEXT, related_store_id INTEGER, ordinal INTEGER, observed_at REAL, created_at REAL)")
    conn.executemany("INSERT INTO message_relations(store_id,kind,related_store_id) VALUES (?,?,?)",
                     [(s, "possible_replay_of", t) for s, t in links] + [(s, "link_pending", None) for s in pending])
    engine = SimpleNamespace(_store=SimpleNamespace(_conn=conn))
    return Run(engine, before, added, inserted, set(), [], frozenset(bound))


def test_indistinguishable_rows_are_checked_as_a_multiset():
    # A2's shape: host 3 is genuinely new, host 6 a replay; both unstamped, one key, one target (5).
    # Store order links the new row; the product cannot tell the two apart, and the scorer counts per key.
    before, added = [_row(5)], [_row(8), _row(11)]
    run = _run(before, added, {3: 8, 6: 11}, links=[(8, 5)])
    assert run.violations({3}) == dict(missing=[], silent=[], invalid=[])
    # No link at all is still a silent copy, and the multiset never credits more than it holds.
    assert _run(before, added, {3: 8, 6: 11}).violations({3})["silent"] == [8]
    both_replays = _run(before + [_row(4)], added, {3: 8, 6: 11}, links=[(8, 5)])
    assert both_replays.violations(set())["silent"] == [11]


def test_distinguishable_rows_keep_the_per_row_check():
    # Distinct stamps: host 3 is new at stamp 7; host 6 replays the original stored at stamp 2.
    before, added = [_row(5, stamp=2.0)], [_row(8, stamp=7.0), _row(11, stamp=2.0)]
    wrong_row = _run(before, added, {3: 8, 6: 11}, links=[(8, 5)])
    assert wrong_row.violations({3})["silent"] == [11]  # the new row's link does NOT cover the replay
    right_row = _run(before, added, {3: 8, 6: 11}, links=[(11, 5)])
    assert right_row.violations({3}) == dict(missing=[], silent=[], invalid=[])
    # A stamped replay whose own equal-stamp original exists must link to it (§2.5 first tier).
    other = _run(before + [_row(6, stamp=9.0)], added, {3: 8, 6: 11}, links=[(11, 6)])
    assert other.violations({3})["invalid"] == [(11, [6], "replay not linked to its own equal-stamp original")]
    # A ledger-bound occurrence is distinguishable by proof even without a stamp.
    unstamped = [_row(8), _row(11)]
    bound = _run([_row(5)], unstamped, {3: 8, 6: 11}, links=[(8, 5)], bound={6})
    assert bound.violations({3})["silent"] == [11]


def test_restamped_replay_without_an_equal_stamp_original_pools_with_its_key():
    """A29 in miniature, with the link-removal control. The original is stored at stamp 2 (row 5); the
    host re-stamps the replay to 4 (row 8, host 6) and also sends a genuinely new unstamped key-mate
    (row 11, host 3). No original has stamp 4, so nothing tells the replay from the new row."""
    clean = dict(missing=[], silent=[], invalid=[])
    before, added = [_row(5, stamp=2.0)], [_row(8, stamp=4.0), _row(11)]
    assert _run(before, added, {6: 8, 3: 11}, links=[(11, 5)]).violations({3}) == clean
    # Control 1: removing that link leaves a real silent copy of the key, and it is flagged.
    assert _run(before, added, {6: 8, 3: 11}).violations({3})["silent"] == [8]
    # Control 2: a replay WITH a unique equal-stamp original is checked per row: a key-mate's link
    # does not cover it, and removing its own link is flagged.
    evident = [_row(8, stamp=2.0), _row(11)]
    assert _run(before, evident, {6: 8, 3: 11}, links=[(11, 5)]).violations({3})["silent"] == [8]
    assert _run(before, evident, {6: 8, 3: 11}).violations({3})["silent"] == [8]
    assert _run(before, evident, {6: 8, 3: 11}, links=[(8, 5)]).violations({3}) == clean
