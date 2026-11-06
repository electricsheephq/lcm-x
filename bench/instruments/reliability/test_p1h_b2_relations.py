"""P1h PR2: exact relation capacity, complete IDs, unchanged B2, fail-closed B2S."""
import copy
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from bench.instruments.reliability import ci
from bench.instruments.reliability.scorers import bars, multiset, tool_calls
from bench.instruments.reliability.test_ci_gate import PERSISTENT, rows_for
from tests import test_reliability_scorers as fixture

SPLIT = {"visible_surplus_rows", "silent_surplus_rows", "silent_surplus_examples"}


def score(tmp_path, rows, relations=None, **kwargs):
    before = fixture.make(tmp_path, rows=rows, events=kwargs.pop("events", []), bars=["B2"], **kwargs)
    db = tmp_path / "cell" / "db" / "lcm.db"
    if relations is not None:
        with sqlite3.connect(db) as con:
            con.execute("create table message_relations (store_id integer, kind text, related_store_id integer)")
            con.executemany("insert into message_relations values (?,?,?)", relations)
    raw = db.read_bytes()
    after = bars.score({"bars": ["B2"], "plugin": kwargs.get("plugin", {})}, tmp_path / "cell")
    assert db.read_bytes() == raw  # scoring reads relations without writing to the store
    assert after["verdict"] == before["verdict"]
    assert after["failed_bars"] == before["failed_bars"]
    assert {k: v for k, v in after["numbers"]["B2"].items() if k not in SPLIT} == {
        k: v for k, v in before["numbers"]["B2"].items() if k not in SPLIT}
    b2 = after["numbers"]["B2"]
    assert b2["visible_surplus_rows"] + b2["silent_surplus_rows"] == b2["surplus_rows"] + b2["tool_surplus_rows"]
    return b2


@pytest.mark.parametrize("pairs", [[(1, 2), (2, 3)], [(2, 1), (3, 1)]], ids=["chain", "star"])
def test_p1h_chain_and_star_explain_two(tmp_path, pairs):
    out = score(tmp_path, [("user", "same")] * 3 + [("assistant", fixture.R.format(1, 1))],
                [(a, "possible_replay_of", b) for a, b in pairs], events=fixture.turn_events(1, held="same"))
    assert (out["surplus_rows"], out["visible_surplus_rows"], out["silent_surplus_rows"]) == (2, 2, 0)


@pytest.mark.parametrize("kind", ["supersedes", "generated_unproven"])
def test_p1h_self_relation_and_pair_share_capacity(tmp_path, kind):
    # Three rows but only two explainable row identities, despite three relation records.
    out = score(tmp_path, [("user", "same")] * 3,
                [(1, kind, 99), (1, "possible_replay_of", 2), (2, "possible_replay_of", 1)])
    assert (out["visible_surplus_rows"], out["silent_surplus_rows"]) == (2, 1)
    assert out["silent_surplus_examples"] == [{"store_id": 3, "role": "user", "sha256": multiset.h("same")}]


def test_p1h_matching_can_reassign_a_pair_away_from_a_self_relation(tmp_path):
    out = score(tmp_path, [("user", "same")] * 2,
                [(1, "possible_replay_of", 2), (1, "supersedes", 99)])
    assert out["visible_surplus_rows"] == 2 and out["silent_surplus_rows"] == 0


@pytest.mark.parametrize("relations", [
    [(1, "possible_replay_of", 2)], [(2, "possible_replay_of", 1)],
    [(1, "possible_replay_of", 99)], [(1, "possible_replay_of", 1)],
    [(1, "composite", 2)], [(2, "split", 1)], [], None,
], ids=["cross-key", "cross-key-reverse", "missing-target", "self-pair", "composite", "split", "empty", "absent"])
def test_p1h_unproven_surplus_is_silent(tmp_path, relations):
    out = score(tmp_path, [("user", "alpha"), ("user", "beta")], relations)
    assert out["visible_surplus_rows"] == 0 and out["silent_surplus_rows"] == 2
    assert len(out["silent_surplus_examples"]) == 2


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_p1h_complete_ids_survive_all_display_caps(tmp_path, role):
    text = "same"
    out = score(tmp_path, [(role, text)] * 31,
                [(i, "generated_unproven", None) for i in range(12, 32)])
    assert (out["visible_surplus_rows"], out["silent_surplus_rows"]) == (20, 11)
    assert [e["store_id"] for e in out["silent_surplus_examples"]] == list(range(1, 11))


def test_p1h_complete_ids_for_a_duplicated_expected_key(tmp_path):
    out = score(tmp_path, [("user", "same")] * 31 + [("assistant", fixture.R.format(1, 1))],
                [(i, "supersedes", None) for i in range(21, 32)], events=fixture.turn_events(1, held="same"))
    assert (out["surplus_rows"], out["visible_surplus_rows"], out["silent_surplus_rows"]) == (30, 11, 19)


def test_p1h_host_licensed_surplus_is_excluded(tmp_path):
    out = score(tmp_path, [("user", "same")] * 3, [(1, "supersedes", None)],
                host=[("S0", "user", "same", 1)] * 3, plugin={"tree": str(Path(__file__).resolve().parents[3])})
    assert out["host_parity_licensed"]["rows"] == 3
    assert out["surplus_rows"] == out["visible_surplus_rows"] == out["silent_surplus_rows"] == 0


def test_p1h_partial_licence_caps_visibility(tmp_path):
    out = score(tmp_path, [("user", "same")] * 3,
                [(i, "supersedes", None) for i in range(1, 4)], host=[("S0", "user", "same", 1)] * 2,
                plugin={"tree": str(Path(__file__).resolve().parents[3])})
    assert out["host_parity_licensed"]["rows"] == 2
    assert (out["surplus_rows"], out["visible_surplus_rows"], out["silent_surplus_rows"]) == (1, 1, 0)


def test_p1h_pairs_do_not_cross_lineages(tmp_path):
    out = score(tmp_path, [("user", "same")] * 2, [(2, "possible_replay_of", 1)], sids=["S0", "foreign"])
    assert out["visible_surplus_rows"] == 0 and out["silent_surplus_rows"] == 2


@pytest.mark.parametrize("related", [True, False])
@pytest.mark.parametrize("kind", ["tool", "call", "malformed"])
def test_p1h_tool_surplus_with_and_without_relation(tmp_path, related, kind):
    row = {"tool": ("tool", "result", None, "call-1"),
           "call": ("assistant", "", json.dumps([{"id": "call-1", "function": {"name": "read", "arguments": "{}"}}])),
           "malformed": ("assistant", "", "bad-json")}[kind]
    out = score(tmp_path, [row, row], [(2, "supersedes", 1)] if related else [])
    assert out["tool_surplus_rows"] == 2 and out["surplus_rows"] == 0
    assert out["visible_surplus_rows"] == int(related)
    assert out["silent_surplus_rows"] == 2 - int(related)
    for example in out["silent_surplus_examples"]:
        assert example["role"] == ("tool" if kind == "tool" else "assistant")
        assert len(example["sha256"]) == 64


def test_p1h_tool_pair_and_loose_attempt_credit(tmp_path):
    events, rows = fixture.tool_turn(1, calls=fixture.PLAN[:1], results=("result",))
    # No turn_end: unfinished-attempt keys receive the existing one-copy allowance.
    events = fixture.turn_events(1)[:1] + events
    out = score(tmp_path, rows * 2, [(4, "possible_replay_of", 2), (3, "possible_replay_of", 1)], events=events)
    assert (out["tool_surplus_rows"], out["visible_surplus_rows"], out["silent_surplus_rows"]) == (2, 2, 0)


def test_p1h_one_stored_row_explains_one_copy_per_key(tmp_path):
    # Review finding 1: each tool entry is its own B2 key, so a marked row explains one copy of each.
    calls = [{"id": name, "function": {"name": name, "arguments": "{}"}} for name in ("a", "b")]
    out = score(tmp_path, [("assistant", "", json.dumps(calls))], [(1, "supersedes", None)])
    assert (out["tool_surplus_rows"], out["visible_surplus_rows"], out["silent_surplus_rows"]) == (2, 2, 0)


def test_p1h_marked_row_explains_its_content_and_its_call(tmp_path):
    calls = [{"id": "a", "function": {"name": "a", "arguments": "{}"}}]
    out = score(tmp_path, [("assistant", "hello", json.dumps(calls))], [(1, "supersedes", None)])
    assert out["surplus_rows"] >= 1 and out["tool_surplus_rows"] >= 1
    assert out["visible_surplus_rows"] == out["surplus_rows"] + out["tool_surplus_rows"]
    assert out["silent_surplus_rows"] == 0 and out["silent_surplus_examples"] == []


def test_p1h_one_row_explains_at_most_one_copy_within_a_key(tmp_path):
    out = score(tmp_path, [("user", "same")] * 3, [(3, "supersedes", None)])
    assert (out["surplus_rows"], out["visible_surplus_rows"], out["silent_surplus_rows"]) == (3, 1, 2)
    assert [e["store_id"] for e in out["silent_surplus_examples"]] == [1, 2]


def test_p1h_silent_examples_prefer_the_later_copy(tmp_path):
    # Review finding 5: with one expected copy and no relation, the example is the later row, not the original.
    out = score(tmp_path, [("user", "same")] * 2 + [("assistant", fixture.R.format(1, 1))],
                events=fixture.turn_events(1, held="same"))
    assert (out["surplus_rows"], out["silent_surplus_rows"]) == (1, 1)
    assert [e["store_id"] for e in out["silent_surplus_examples"]] == [2]


def test_p1h_tool_id_collection_keeps_full_keys_and_counts():
    rows = [(i, "S0", "tool", "result", None, "call-1") for i in range(1, 32)]
    ids = {}
    keys = tool_calls.stored_keys(rows, store_ids=ids)
    key = ("chat", "tool", "call-1", hashlib.sha256(b"result").hexdigest())
    assert keys == Counter({key: 31}) and ids == {key: list(range(1, 32))}


FULL = {"deficit_rows": 0, "silent_surplus_rows": 0, "visible_surplus_rows": 0, "surplus_rows": 0,
        "tool_surplus_rows": 0, "tool_missing_rows": 0, "externalized": {"deficit_rows": 0}}


@pytest.mark.parametrize("numbers,failed,verdict,want", [
    ({**FULL, "deficit_rows": 1}, ["B2"], "FAIL", True),
    ({**FULL, "silent_surplus_rows": 1, "surplus_rows": 1}, ["B2"], "FAIL", True),
    ({**FULL, "visible_surplus_rows": 2, "surplus_rows": 2}, ["B2"], "FAIL", False),
    ({**FULL, "visible_surplus_rows": 2, "surplus_rows": 1, "tool_surplus_rows": 1}, ["B2"], "FAIL", False),
    ({"deficit_rows": 0}, ["B2"], "FAIL", True),
    ({"deficit_rows": 0, "visible_surplus_rows": 2}, ["B2"], "FAIL", True),
    ({"deficit_rows": 0, "silent_surplus_rows": 0}, ["B2"], "FAIL", True),
    ({"deficit_rows": 0, "silent_surplus_rows": 0, "visible_surplus_rows": 0}, ["B2"], "FAIL", True),
    ({}, [], "PASS", True),
    (FULL, [], "PASS", False),
    ({**FULL, "silent_surplus_rows": -1}, ["B2"], "FAIL", True),
    ({**FULL, "silent_surplus_rows": False}, ["B2"], "FAIL", True),
    ({**FULL, "tool_missing_rows": 1}, ["B2"], "FAIL", True),
    ({**FULL, "externalized": {"deficit_rows": 1}}, ["B2"], "FAIL", True),
    # Review finding 2: a missing or None tool/externalized count fails closed.
    ({**FULL, "tool_missing_rows": None}, [], "PASS", True),
    ({**FULL, "externalized": {"deficit_rows": None}}, [], "PASS", True),
    ({**FULL, "externalized": {}}, [], "PASS", True),
    ({**FULL, "externalized": [1]}, [], "PASS", True),
    # Review finding 3: the split must cover all surplus.
    ({**FULL, "surplus_rows": 9}, ["B2"], "FAIL", True),
    ({**FULL, "visible_surplus_rows": 1, "surplus_rows": 2}, ["B2"], "FAIL", True),
    ({**FULL, "visible_surplus_rows": 3, "surplus_rows": 2}, ["B2"], "FAIL", True),
], ids=lambda value: None)
def test_p1h_b2s_gate_fails_closed(monkeypatch, numbers, failed, verdict, want):
    original = copy.deepcopy(ci.CELL_GATE)
    monkeypatch.setitem(ci.CELL_GATE, PERSISTENT, ("in-process", ("B2S",)))
    rows = rows_for(PERSISTENT, "fixture", "in-process", failed)
    row = next(r for r in rows if r["cell"] == PERSISTENT)
    row.update(verdict=verdict, numbers={"B2": numbers})
    problems = ci.gate(rows, set())
    assert bool(problems) is want
    if want:
        assert len(problems) == 1 and "['B2S']" in problems[0]
    monkeypatch.setattr(ci, "CELL_GATE", original)


@pytest.mark.parametrize("case", ["merge", "embedded", "projection"])
def test_p1h_existing_credits_do_not_create_extra_surplus(tmp_path, case):
    from bench.instruments.reliability.test_witness_scoring import COMPOSITE, EARLIER, HELD, merge_events

    plugin = {"tree": str(Path(__file__).resolve().parents[3])}
    if case == "merge":
        text, rows = "alpha\n\nbeta", [("user", "alpha"), ("user", "beta")]
        events = fixture.turn_events(1, held=text)[:1]
        events[0].update(content=text, persist=text)
        events += merge_events(("alpha", "beta"), session="S0")
        host = [("S0", "user", "alpha", 1), ("S0", "user", "beta", 1)]
        expected_surplus = 0
    elif case == "embedded":
        rows = [("user", COMPOSITE)]
        events = fixture.turn_events(1, held=HELD)[:1]
        events[0].update(content=HELD, persist=HELD)
        events += merge_events((EARLIER, HELD), session="S0")
        host = [("S0", "user", COMPOSITE, 1)]
        expected_surplus = 0
    else:
        original = "original objective"
        carrier = "[Current user objective preserved from compacted history]\n" + original
        rows = [("user", original), ("assistant", fixture.R.format(1, 1))]
        events = fixture.turn_events(1, held=carrier)
        events[0].update(content=original, persist=original)
        host, expected_surplus = None, 0
    out = score(tmp_path, rows, [(i, "supersedes", None) for i in range(1, len(rows) + 1)],
                events=events, host=host, plugin=plugin)
    assert out["deficit_rows"] == 0
    assert out["surplus_rows"] == expected_surplus
    assert out["visible_surplus_rows"] == out["silent_surplus_rows"] == 0


@pytest.mark.parametrize("verdict", ["UNSUPPORTED", "INCONCLUSIVE"])
def test_p1h_b2s_fails_closed_on_rows_without_numbers(monkeypatch, verdict):
    # Review finding 4: a B2S-gated row that carries no numbers fails closed (README: B2S).
    monkeypatch.setitem(ci.CELL_GATE, PERSISTENT, ("in-process", ("B2S",)))
    rows = rows_for(PERSISTENT, "fixture", "in-process", [])
    row = next(r for r in rows if r["cell"] == PERSISTENT)
    row.update(verdict=verdict, numbers={})
    problems = ci.gate(rows, set())
    assert any("['B2S']" in p for p in problems)


@pytest.mark.parametrize("verdict", ["UNSUPPORTED", "INCONCLUSIVE", "SKIPPED"])
def test_p1h_b2s_fails_closed_on_unproven_verdicts_with_clean_numbers(monkeypatch, verdict):
    # Bot review on #2291: a scenario that was not proven never satisfies B2S, however clean its counts.
    monkeypatch.setitem(ci.CELL_GATE, PERSISTENT, ("in-process", ("B2S",)))
    rows = rows_for(PERSISTENT, "fixture", "in-process", [])
    row = next(r for r in rows if r["cell"] == PERSISTENT)
    row.update(verdict=verdict, numbers={"B2": dict(FULL)})
    problems = ci.gate(rows, set())
    assert len(problems) == 1 and "['B2S']" in problems[0]


def test_p1h_b2s_proven_pass_with_clean_numbers_still_passes(monkeypatch):
    # The control for the case above: the same clean counts on a PASS row satisfy B2S.
    monkeypatch.setitem(ci.CELL_GATE, PERSISTENT, ("in-process", ("B2S",)))
    rows = rows_for(PERSISTENT, "fixture", "in-process", [])
    row = next(r for r in rows if r["cell"] == PERSISTENT)
    row.update(verdict="PASS", numbers={"B2": dict(FULL)})
    assert ci.gate(rows, set()) == []
