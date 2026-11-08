"""P1h PR2: exact relation capacity, complete IDs, unchanged B2, fail-closed B2S."""
import copy
import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pytest

from bench.instruments.reliability import ci
from bench.instruments.reliability.scorers import bars, multiset, tool_calls
from bench.instruments.reliability.test_ci_gate import PERSISTENT, rows_for
from tests import test_reliability_scorers as fixture

SPLIT = {"visible_surplus_rows", "silent_surplus_rows", "silent_surplus_examples"}


@pytest.mark.parametrize("kind", ["supersedes", "generated_unproven"])
def test_silent_surplus_examples_exclude_the_kept_copy(kind):
    key = ("assistant", "synthetic-digest")
    result = multiset.split_surplus([(key, 2, [1, 2, 2])], [(2, kind, None)])
    assert (result["visible_surplus_rows"], result["silent_surplus_rows"]) == (1, 1)
    assert result["silent_surplus_examples"] == [dict(store_id=2, role=key[0], sha256=key[1])]


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


def test_p1h_pairs_across_a_compression_chain_are_visible(tmp_path):
    # #2029: a rotation child's re-stored copy links to its predecessor's row; the host records the child under
    # its parent, so both rows are one lineage and the link explains the copy. A foreign session stays refused.
    out = score(tmp_path, [("user", "same")] * 2, [(2, "possible_replay_of", 1)], sids=["S0", "S1"],
                parents={"S0": None, "S1": "S0"})
    # No turn expects "same" here, so both rows are surplus (the foreign case above explains neither); the
    # cross-session link explains the later copy.
    assert (out["surplus_rows"], out["visible_surplus_rows"], out["silent_surplus_rows"]) == (2, 1, 1)


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


def _split(expected, rows, relations, host=None):
    keys = []
    out = multiset.score(expected, [(sid, "S", role, text) for sid, role, text in rows], host=host, surplus_keys=keys)
    return out, multiset.split_surplus(keys, relations)


@pytest.mark.parametrize("relations,want", [([], 0), ([(2, "possible_replay_of", 1)], 1),
                                          ([(1, "possible_replay_of", 2)], 0), ([(2, "carried_stamp", 1)], 0)])
def test_cursor0_stripped_content_requires_product_pair(relations, want):
    _out, split = _split([("assistant", "<think>old</think>answer")],
                         [(1, "assistant", "<think>old</think>answer"), (2, "assistant", "answer")], relations)
    assert split["visible_surplus_rows"] == want
    assert split["silent_surplus_rows"] == 1 - want


@pytest.mark.parametrize("old,new", [("<think>x</think>", " "), ("<think>x</think>answer", " answer"),
                                    ("<think>x</think>Answer", "answer"), ("<think>x</think>answer", "other")])
def test_cursor0_stripped_pair_rechecks_content_without_more_normalization(old, new):
    _out, split = _split([("assistant", old)], [(1, "assistant", old), (2, "assistant", new)],
                         [(2, "possible_replay_of", 1)])
    assert split["visible_surplus_rows"] == 0


def test_p1h_a_host_licensed_copy_makes_the_key_fail_closed():
    # Delta review on #2291: one expected copy plus one host-licensed copy; the silent re-store (row 2) sits before
    # the related host-mirrored copy (row 3), so the key's surplus stays silent.
    host = {("user", multiset.h("B")): {"n": 2, "ids": [101, 102]}}
    out, split = _split([("user", "B")], [(1, "user", "B"), (2, "user", "B"), (3, "user", "B")],
                        [(3, "possible_replay_of", 1)], host=host)
    assert out["surplus_rows"] == 1 and [r["licensed"] for r in out["host_parity_licensed"]] == [1]
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"]) == (0, 1)


@pytest.mark.parametrize("expected,rows,relations,want", [
    # Review on #2291: the expected copy's own relation (an edited revision's supersedes) never explains a later
    # unmarked re-store.
    ([("user", "B")], [(1, "user", "B"), (2, "user", "B")], [(1, "supersedes", 0)], (1, 0, 1, [2])),
    ([("user", "B")], [(1, "user", "B"), (2, "user", "B")], [(1, "generated_unproven", None)], (1, 0, 1, [2])),
    # A self relation on the expected copy plus a pair: the pair explains one later copy, the last stays silent.
    ([("user", "B")], [(1, "user", "B"), (2, "user", "B"), (3, "user", "B")],
     [(1, "supersedes", 9), (2, "possible_replay_of", 1)], (2, 1, 1, [3])),
    # A genuine repeat that the product marked as possible replay (two expected copies) does not explain a third,
    # unmarked copy: the pair links two expected copies.
    ([("user", "B")] * 2, [(1, "user", "B"), (2, "user", "B"), (3, "user", "B")],
     [(2, "possible_replay_of", 1)], (1, 0, 1, [3])),
    # Delta review on #2291: with two expected copies, a silent re-store (row 2) stored before the related genuine
    # repeat (row 3) cannot be told apart, so the key's surplus stays silent (fail closed).
    ([("user", "B")] * 2, [(1, "user", "B"), (2, "user", "B"), (3, "user", "B")],
     [(3, "possible_replay_of", 1)], (1, 0, 1, [3])),
    # The example is the unmarked row 2, the likely silent copy, not the marked row 3.
    ([("user", "B")] * 2, [(1, "user", "B"), (2, "user", "B"), (3, "user", "B")],
     [(3, "supersedes", None)], (1, 0, 1, [2])),
    # Controls: a relation on the later copy still explains it, by itself or through a pair to the original.
    ([("user", "B")], [(1, "user", "B"), (2, "user", "B")], [(2, "supersedes", None)], (1, 1, 0, [])),
    ([("user", "B")], [(1, "user", "B"), (2, "user", "B")], [(1, "possible_replay_of", 2)], (1, 1, 0, [])),
    ([("user", "B")], [(1, "user", "B"), (2, "user", "B")],
     [(1, "generated_unproven", None), (2, "generated_unproven", None)], (1, 1, 0, [])),
], ids=["supersedes-on-original", "unproven-on-original", "self-plus-pair", "pair-between-expected",
        "two-expected-pair-on-last", "two-expected-self-on-last",
        "later-self", "pair-to-original", "both-unproven"])
def test_p1h_only_later_copies_can_be_explained(expected, rows, relations, want):
    out, split = _split(expected, rows, relations)
    assert (out["surplus_rows"], split["visible_surplus_rows"], split["silent_surplus_rows"],
            [e["store_id"] for e in split["silent_surplus_examples"]]) == want


KEY = ("assistant", "k")


@pytest.mark.parametrize("groups,relations,want", [
    # Delta review on #2291: a tool key's ids repeat a row whose tool_calls holds one entry twice. Row 2 holds both
    # surplus copies and explains one of them; the other stays silent.
    ([(KEY, 2, [1, 2, 2])], [(1, "generated_unproven", None), (2, "generated_unproven", None)], (1, 1, [2])),
    # Counted as copies, [1, 1, 2] keeps two copies, so the key counts its worst case: the surplus copy can be one of
    # the unmarked row's.
    ([(KEY, 1, [1, 1, 2])], [(2, "supersedes", None)], (0, 1, [1])),
    ([(KEY, 1, [5, 5, 6])], [(6, "generated_unproven", None)], (0, 1, [5])),
    # Worst case over copies: row 2's copy and a second copy of row 1 can both be surplus; row 1 explains one.
    ([(KEY, 2, [1, 1, 2, 3, 4])], [(1, "supersedes", None), (3, "supersedes", None), (4, "supersedes", None)],
     (1, 1, [2])),
    # Controls: a surplus copy inside a self-related row is explained by that row, in either branch.
    ([(KEY, 1, [5, 5])], [(5, "generated_unproven", None)], (1, 0, [])),
    ([(KEY, 1, [5, 5, 6])], [(5, "generated_unproven", None), (6, "generated_unproven", None)], (1, 0, [])),
], ids=["repeated-later-row", "repeated-unmarked-row", "repeated-unmarked-row-unproven", "worst-case-over-copies",
        "one-kept-copy-in-a-marked-row", "every-row-marked"])
def test_p1h_split_counts_copies_not_rows(groups, relations, want):
    split = multiset.split_surplus(groups, relations)
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"],
            [e["store_id"] for e in split["silent_surplus_examples"]]) == want


def test_p1h_a_row_that_holds_a_call_twice_explains_one_copy():
    # Delta review on #2291, through the tool scorer: row 2's tool_calls holds the expected call twice, so the key's
    # ids repeat row 2. Row 2's relation explains one of its copies; the other is silent, so B2S fails.
    call = {"id": "call-1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    rows = [(1, "S0", "assistant", "", json.dumps([call]), None),
            (2, "S0", "assistant", "", json.dumps([call, call]), None)]
    ids, keys = {}, []
    stored = tool_calls.stored_keys(rows, store_ids=ids)
    out = tool_calls.compare(Counter({next(iter(stored)): 1}), set(), stored, store_ids=ids, surplus_keys=keys)
    assert out["tool_surplus_rows"] == 2 and [k[2] for k in keys] == [[1, 2, 2]]
    split = multiset.split_surplus(keys, [(1, "generated_unproven", None), (2, "generated_unproven", None)])
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"]) == (1, 1)


@pytest.mark.parametrize("composite,relations,want", [
    # Bot review on #2291: a host merge record consumes rows 1 and 2 as the expected composite's parts, and row 3
    # re-stores part "alpha" with a pair to row 1. The record picked its part among identical copies, so alpha takes
    # the worst case and the pair explains nothing: B2S fails closed (an accepted tradeoff; no gate cell has host
    # merge records).
    ("alpha\n\nbeta", [(3, "possible_replay_of", 1)], (0, 1)),
    ("alpha\n\nbeta", [(3, "supersedes", None)], (0, 1)),
    # Controls: a pair to the consumed copy of another key (row 2, "beta") explains nothing; so does no relation.
    ("alpha\n\nbeta", [(3, "possible_replay_of", 2)], (0, 1)),
    ("alpha\n\nbeta", [], (0, 1)),
    # Delta review on #2291: the record consumes two alpha copies, so alpha keeps two copies and takes its worst
    # case. A silent re-store (row 2) stored before the genuine part (row 3) cannot be told apart, so neither a pair
    # nor a self relation on row 3 explains the surplus.
    ("alpha\n\nalpha", [(3, "possible_replay_of", 1)], (0, 1)),
    ("alpha\n\nalpha", [(3, "supersedes", None)], (0, 1)),
], ids=["pair-to-consumed-part", "self-on-later-copy", "pair-to-other-part", "no-relation", "two-consumed-pair",
        "two-consumed-self"])
def test_p1h_merge_consumed_copies_stay_in_the_key(composite, relations, want):
    parts = [("user", multiset.h(p)) for p in composite.split("\n\n")]
    host = {("user", multiset.h(composite)): {"n": 1, "ids": [9], "merges": [{"host_row_id": 9, "parts": parts}]}}
    second = composite.split("\n\n")[1]
    out, split = _split([("user", composite)], [(1, "user", "alpha"), (2, "user", second), (3, "user", "alpha")],
                        relations, host=host)
    assert (out["deficit_rows"], out["surplus_rows"]) == (0, 1)
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"]) == want


@pytest.mark.parametrize("groups,relations,want", [
    # Delta review on #2291: an embedded credit reserved row 1, so the merge consumed the later copy (row 3) and row 1
    # is the key's surplus. A self relation on the consumed row must not explain it, and a key with a consumed copy
    # takes the worst case whichever row is marked.
    ([(KEY, 1, [1, 3], [3])], [(3, "supersedes", None)], (0, 1)),
    ([(KEY, 1, [1, 3], [3])], [(1, "supersedes", None)], (0, 1)),
    # Control: the same ids with no consumed copy follow store order, so a relation on the later copy explains it.
    ([(KEY, 1, [1, 3])], [(3, "supersedes", None)], (1, 0)),
    ([(KEY, 1, [1, 3], [])], [(3, "supersedes", None)], (1, 0)),
], ids=["consumed-later-copy-marked", "surplus-copy-marked", "no-merge", "empty-merge-list"])
def test_p1h_a_key_with_a_merge_consumed_copy_takes_the_worst_case(groups, relations, want):
    split = multiset.split_surplus(groups, relations)
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"]) == want


@pytest.mark.parametrize("relations", [[(3, "supersedes", None)], [(1, "supersedes", None)], []],
                         ids=["consumed-row-marked", "surplus-row-marked", "no-relation"])
def test_p1h_an_embedded_reserved_copy_sends_its_key_to_the_worst_case(relations):
    # Delta review on #2291: an embedded credit reserves row 1 for "alpha", so #804 takes the later A copy (row 3) as
    # C's licensed part and row 1 is A's surplus. Store order cannot name the kept copy, so no relation explains it.
    a, c = "alpha\n\nbeta", "alpha\n\nbeta\n\ngamma"
    spans = [{"byte_offset": 0, "bytes": 5, "sha256": multiset.h("alpha")},
             {"byte_offset": 7, "bytes": 4, "sha256": multiset.h("beta")}]
    witness = {"witness": {"event_index": 0, "phase": "p", "turn": 1}, "spans": spans, "bytes": 11, "kind": "host_merge"}
    host = {("user", multiset.h(a)): {"n": 1, "ids": [101], "span_witnesses": [witness], "span_capacity": 1},
            ("user", multiset.h("gamma")): {"n": 1, "ids": [102]}}
    out, split = _split([("user", "alpha"), ("user", c)], [(1, "user", a), (2, "user", "gamma"), (3, "user", a)],
                        relations, host=host)
    assert (out["deficit_rows"], out["surplus_rows"], out["split_keys"]) == (0, 1, 0)
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"]) == (0, 1)


def test_p1h_b2s_fails_when_only_the_original_carries_a_relation(monkeypatch):
    # Scorer to gate: the reviewer's probe fails B2S; the control with the relation on the later copy passes.
    for relations, want in (([(1, "supersedes", 0)], True), ([(2, "supersedes", None)], False)):
        out, split = _split([("user", "B")], [(1, "user", "B"), (2, "user", "B")], relations)
        numbers = {**FULL, **{k: out[k] for k in ("deficit_rows", "surplus_rows", "split_keys")},
                   "visible_surplus_rows": split["visible_surplus_rows"],
                   "silent_surplus_rows": split["silent_surplus_rows"]}
        monkeypatch.setitem(ci.CELL_GATE, PERSISTENT, ("in-process", ("B2S",)))
        rows = rows_for(PERSISTENT, "fixture", "in-process", ["B2"])
        row = next(r for r in rows if r["cell"] == PERSISTENT)
        row.update(verdict="FAIL", numbers={"B2": numbers})
        assert bool(ci.gate(rows, set())) is want


def test_p1h_tool_id_collection_keeps_full_keys_and_counts():
    rows = [(i, "S0", "tool", "result", None, "call-1") for i in range(1, 32)]
    ids = {}
    keys = tool_calls.stored_keys(rows, store_ids=ids)
    key = ("chat", "tool", "call-1", hashlib.sha256(b"result").hexdigest())
    assert keys == Counter({key: 31}) and ids == {key: list(range(1, 32))}


FULL = {"deficit_rows": 0, "silent_surplus_rows": 0, "visible_surplus_rows": 0, "surplus_rows": 0,
        "tool_surplus_rows": 0, "tool_missing_rows": 0, "externalized": {"deficit_rows": 0}, "split_keys": 0}


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
    # Bot review on #2291: a split reply fails closed, and so does a missing split count.
    ({**FULL, "split_keys": 1, "surplus_rows": 2, "visible_surplus_rows": 2}, ["B2"], "FAIL", True),
    ({**FULL, "split_keys": None}, [], "PASS", True),
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


def test_p1h_b2s_fails_a_split_reply_that_hides_a_lost_copy(monkeypatch):
    # Bot review on #2291, scorer to gate: two expected copies of one reply, one stored split pair. The scorer
    # counts no deficit for the split key, and both fragments are visible surplus; B2S must still fail.
    keys = []
    out = multiset.score([("assistant", "one two")] * 2,
                         [(1, "S", "assistant", "one "), (2, "S", "assistant", "two")], surplus_keys=keys)
    split = multiset.split_surplus(keys, [(1, "generated_unproven", None), (2, "generated_unproven", None)])
    assert (out["split_keys"], out["deficit_rows"], out["surplus_rows"]) == (1, 0, 2)
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"]) == (2, 0)
    numbers = {**FULL, **{k: out[k] for k in ("deficit_rows", "surplus_rows", "split_keys")},
               "visible_surplus_rows": split["visible_surplus_rows"],
               "silent_surplus_rows": split["silent_surplus_rows"]}
    monkeypatch.setitem(ci.CELL_GATE, PERSISTENT, ("in-process", ("B2S",)))
    rows = rows_for(PERSISTENT, "fixture", "in-process", ["B2"])
    row = next(r for r in rows if r["cell"] == PERSISTENT)
    row.update(verdict="FAIL", numbers={"B2": numbers})
    problems = ci.gate(rows, set())
    assert len(problems) == 1 and "['B2S']" in problems[0]


def test_p1h_b2s_proven_pass_with_clean_numbers_still_passes(monkeypatch):
    # The control for the case above: the same clean counts on a PASS row satisfy B2S.
    monkeypatch.setitem(ci.CELL_GATE, PERSISTENT, ("in-process", ("B2S",)))
    rows = rows_for(PERSISTENT, "fixture", "in-process", [])
    row = next(r for r in rows if r["cell"] == PERSISTENT)
    row.update(verdict="PASS", numbers={"B2": dict(FULL)})
    assert ci.gate(rows, set()) == []


def test_p1h_stripped_tool_result_link_uses_the_stored_rows():
    # #2335 review: a cursor-0 copy of a tool result linked through the stripped pass. Tool groups carry no row
    # evidence, so with no user/assistant surplus the scorer must read the stored rows the caller passes.
    rows = [(1, "S0", "tool", "<think>plan</think>answer", None, "call-1"),
            (2, "S0", "tool", "answer", None, "call-1")]
    # The two contents differ, so the copy is its own key: row 1 is expected, row 2 is the only surplus.
    keys = [(("tool", "digest-of-answer"), 1, [2])]
    relations = [(2, "possible_replay_of", 1)]
    split = multiset.split_surplus(keys, relations, {r[0]: (r[1], r[2], r[3]) for r in rows})
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"]) == (1, 0)
    # A relation between rows whose stripped content differs explains nothing.
    other = {1: ("S0", "tool", "<think>plan</think>answer"), 2: ("S0", "tool", "different")}
    split = multiset.split_surplus(keys, relations, other)
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"]) == (0, 1)


def test_p1h_shared_evidence_map_is_merged_once():
    # score() attaches one shared stored-row map to every surplus group; it must be read once, not once per group.
    # A Mapping, not a dict subclass: dict.update copies a dict subclass without calling any method we could count.
    class CountingMap(Mapping):
        reads = 0

        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, key):
            return self.rows[key]

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            CountingMap.reads += 1
            return iter(self.rows)

    shared = CountingMap({1: ("S0", "user", "a"), 2: ("S0", "user", "a")})
    groups = [(("user", f"k{i}"), 0, [1], [], shared) for i in range(50)]
    multiset.split_surplus(groups, [])
    assert CountingMap.reads == 1


def test_p1h_replay_links_are_checked_against_stored_content(tmp_path, monkeypatch):
    # #2335 review: the product links on stored bytes. When resolution expands only the older row's placeholder,
    # the split must still receive the stored contents. This checks the map score() passes; the effect on the
    # split is checked in the gate-kit audit's test_replay_links_are_checked_against_stored_content.
    fixture.make(tmp_path, rows=[("assistant", "[stored]"), ("assistant", "<think>r</think>[stored]")], events=[])
    resolve, seen = bars.externalized.resolve, {}

    def expand_first(rows, home):
        out, ext, ids = resolve(rows, home)
        return [(r[0], r[1], r[2], "expanded", *r[4:]) if r[0] == 1 else r for r in out], ext, ids

    def split(groups, relations, rows=None):
        seen.update(rows)
        return {}

    monkeypatch.setattr(bars.externalized, "resolve", expand_first)
    monkeypatch.setattr(bars.multiset, "split_surplus", split)
    bars.score({"bars": ["B2"], "plugin": {}}, tmp_path / "cell")
    assert seen == {1: ("S0", "assistant", "[stored]"), 2: ("S0", "assistant", "<think>r</think>[stored]")}


def test_silent_example_names_unexplained_surplus_copy():
    key = ("assistant", "digest")
    out = multiset.split_surplus([(key, 2, [1, 2, 2])], [(2, "generated_unproven", None)])
    assert (out["visible_surplus_rows"], out["silent_surplus_rows"]) == (1, 1)
    assert out["silent_surplus_examples"] == [{"store_id": 2, "role": "assistant", "sha256": "digest"}]


@pytest.mark.parametrize("relations,want", [([(2, "link_pending", None)], 1), ([], 0), ([(1, "link_pending", None)], 0),
                                            ([(2, "link_pending", 1)], 0)],
                         ids=["on-surplus", "none", "on-expected-copy", "related-not-null"])
def test_link_pending_credits_a_surplus_copy_and_counts_every_marked_row(relations, want):
    split = multiset.split_surplus([(("user", "k"), 1, [1, 2])], relations + [(3, "link_pending", None)])
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"], split["link_pending_rows"]) == (
        want, 1 - want, want)
    assert split["link_pending_rows_total"] == 1 + sum(o is None for _s, _k, o in relations)


BLOCK = "<todo block>"  # replaced by the todo injection header plus an item when a test runs
PARTS = {1: ("S", "user", "prompt one"), 2: ("S", "user", "prompt two"), 4: ("T", "user", "prompt two"),
         5: ("S", "user", BLOCK), 6: ("S", "user", "plain first")}


def todo_rows(rows):
    block = multiset.todo_header() + "\n- [ ] step"
    return {sid: (session, role, text.replace(BLOCK, block)) for sid, (session, role, text) in rows.items()}


@pytest.mark.parametrize("content,segments,want", [
    (BLOCK + "\n\nprompt one", [(0, None), (1, 1)], 1),
    (BLOCK + "\n\nprompt one\n\nprompt two", [(0, None), (1, 1), (2, 2)], 1),
    (BLOCK + "\n\nprompt onX", [(0, None), (1, 1)], 0),
    ("lead\n\nprompt one", [(0, None), (1, 1)], 0),
    (BLOCK + "\n\nprompt one\n\nprompt two", [(0, None), (1, 2), (2, 1)], 0),
    (BLOCK + "\n\nprompt one", [(0, None), (2, 1)], 0),
    (BLOCK + "\n\nprompt one", [(0, None), (1, None), (2, 1)], 0),
    (BLOCK + "\n\nprompt two", [(0, None), (1, 4)], 0),
    (BLOCK + "\n\nprompt one", [], 0),
    (BLOCK + "\nprompt one", [(0, None), (1, 1)], 0),
    (BLOCK + "\n\nprompt one", [(0, 5), (1, 1)], 1),
    ("plain first\n\nprompt one", [(0, 6), (1, 1)], 0),
    (BLOCK + "\n\nprompt two", [(0, 5), (1, 4)], 0),
    ("prompt one\n\n" + BLOCK, [(0, 1), (1, None)], 0),
], ids=["P1", "P2", "N1-byte", "N2-no-header", "N3-order", "N3-gap", "N4-two-null", "N5-other-session",
        "N6-none", "N7-single-newline", "P3-stored-todo", "N8-non-todo-first", "N9-other-session",
        "N10-scaffold-not-first"])
def test_composite_of_witness_credits_only_the_exact_shapes(content, segments, want):
    rows = todo_rows({**PARTS, 9: ("S", "user", content)})
    split = multiset.split_surplus([(("user", multiset.h(rows[9][2])), 1, [9])],
                                   [(9, "composite_of", r, o) for o, r in segments], rows)
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"], split["composite_rows"]) == (
        want, 1 - want, want)


def test_composite_of_witness_leaves_a_silent_part_silent():
    rows = todo_rows({**PARTS, 7: ("S", "user", "prompt one"), 9: ("S", "user", BLOCK + "\n\nprompt one")})
    groups = [(("user", multiset.h(rows[9][2])), 1, [9]), (("user", multiset.h("prompt one")), 1, [1, 7])]
    split = multiset.split_surplus(groups, [(9, "composite_of", None, 0), (9, "composite_of", 7, 1)], rows)
    assert (split["visible_surplus_rows"], split["silent_surplus_rows"], split["composite_rows"]) == (1, 1, 1)
    assert [e["store_id"] for e in split["silent_surplus_examples"]] == [7]


@pytest.mark.parametrize("case,credited,visible", [
    ("scaffold", {9}, 1), ("stored-todo", {9}, 1), ("repeated-part", {9}, 1),
    ("replay-link", {9}, 2), ("distinct-parts", {9, 10}, 2), ("partial-overlap", {9}, 1),
    ("later-part", set(), 0), ("non-user-part", set(), 0), ("stored-non-user-part", set(), 0),
    ("rejected-claims-nothing", {9, 12}, 2),
])
def test_composite_consume_once(case, credited, visible):
    rows = todo_rows({**PARTS, 7: ("S", "user", "prompt three"), 11: ("S", "user", "later")})
    composites = {9: [None, 1]}
    if case == "stored-todo":
        composites = {9: [5, 1]}
    elif case in ("repeated-part", "replay-link", "distinct-parts"):
        composites = {10: [None, 2 if case == "distinct-parts" else 1], 9: [None, 1]}
    elif case in ("partial-overlap", "rejected-claims-nothing"):
        composites = {10: [None, 2, 7], 9: [None, 1, 2]}
        if case == "rejected-claims-nothing":
            composites = {12: [None, 7], **composites}
    elif case == "later-part":
        composites = {9: [None, 11]}
    elif case in ("non-user-part", "stored-non-user-part"):
        rows[1] = ("S", "assistant", rows[1][2])
        if case == "stored-non-user-part":
            composites = {9: [5, 1]}
    relations = []
    for sid, parts in composites.items():  # higher composite first: credit must follow store order
        rows[sid] = ("S", "user", "\n\n".join(
            todo_rows({0: ("S", "user", BLOCK)})[0][2] if p is None else rows[p][2] for p in parts))
        relations += [(sid, "composite_of", p, i) for i, p in enumerate(parts)]
    if case == "replay-link":
        relations.append((10, "possible_replay_of", 9))
    assert multiset.composite_witness(relations, rows) == credited
    groups = []
    multiset.score([], [(sid, *rows[sid]) for sid in composites], surplus_keys=groups)
    split = multiset.split_surplus(groups, relations, rows)
    assert (split["composite_rows"], split["visible_surplus_rows"], split["silent_surplus_rows"]) == (
        len(credited), visible, len(composites) - visible)


@pytest.mark.parametrize("composite_id,reported,credited", [(3, [], 0), (4, [4], 1)],
                         ids=["expected-copy-not-reported", "later-copy-reported"])
def test_creditable_composite_ids_exclude_the_expected_copy(composite_id, reported, credited):
    """Review of #2495 (probe A): the host shows C once and a second copy is stored. Only the key's later copy can be
    credited, so a composite_of on the expected copy is witnessed but not reported as a credited row."""
    rows = todo_rows({1: ("S", "user", "prompt one"), 3: ("S", "user", BLOCK + "\n\nprompt one"),
                      4: ("S", "user", BLOCK + "\n\nprompt one")})
    c = rows[3][2]
    relations = [(composite_id, "composite_of", None, 0), (composite_id, "composite_of", 1, 1)]
    groups = []
    multiset.score([("user", "prompt one"), ("user", c)], [(s, *rows[s]) for s in sorted(rows)], surplus_keys=groups)
    forms, creditable = multiset.attribution_forms(relations, rows), set()
    split = multiset.split_surplus(groups, relations, rows, forms, creditable)
    assert forms["composite_rows"] == {composite_id}
    assert sorted(forms["composite_rows"] & {s for g in groups for s in g[2]}) == [composite_id]  # 22f25b19's list
    assert sorted(forms["composite_rows"] & creditable) == reported
    assert (split["composite_rows"], split["visible_surplus_rows"], split["silent_surplus_rows"]) == (
        credited, credited, 1 - credited)


def _composite_split(stored, expected, composites, extra=()):
    """Score ``stored`` (store_id -> text, session S, role user) against ``expected`` texts, with ``composites``
    (C -> parts, None for the todo scaffold) and ``extra`` relations; returns (split, reported composite ids)."""
    rows = todo_rows({sid: ("S", "user", text) for sid, text in stored.items()})
    relations = [(c, "composite_of", p, i) for c, parts in composites.items() for i, p in enumerate(parts)]
    relations += list(extra)
    groups = []
    multiset.score([("user", rows[s][2]) for s in expected], [(s, *rows[s]) for s in sorted(rows)],
                   surplus_keys=groups)
    forms, creditable = multiset.attribution_forms(relations, rows), set()
    split = multiset.split_surplus(groups, relations, rows, forms, creditable)
    return split, sorted(forms["composite_rows"] & creditable)


def test_composite_ids_exclude_a_key_with_zero_capacity():
    """Review of #2495 (NeonDiff P2, CodeRabbit): the host shows C twice and three copies are stored, so the unmarked
    copies take the one surplus copy first and the witnessed composite explains nothing. It is not reported."""
    c = BLOCK + "\n\nprompt one"
    split, reported = _composite_split({1: "prompt one", 3: c, 4: c, 5: c}, [1, 3, 4], {3: [None, 1]})
    assert reported == []
    assert (split["composite_rows"], split["visible_surplus_rows"], split["silent_surplus_rows"]) == (0, 0, 1)


def test_composite_ids_list_only_the_copies_the_matching_credits():
    """Three valid composites with distinct parts compete for one surplus copy: exactly one is reported."""
    c = BLOCK + "\n\nprompt one"
    split, reported = _composite_split({1: "prompt one", 2: "prompt one", 3: "prompt one", 4: c, 5: c, 6: c},
                                       [1, 2, 3, 4, 5], {4: [None, 1], 5: [None, 2], 6: [None, 3]})
    assert len(reported) == 1 and set(reported) <= {4, 5, 6}
    assert (split["composite_rows"], split["visible_surplus_rows"], split["silent_surplus_rows"]) == (1, 1, 0)


def test_composite_ids_when_a_replay_link_explains_the_same_copy():
    """A composite and a possible_replay_of pair compete for one copy: the copy is visible once, the form adds
    nothing beyond the pair (composite_rows 0), and at most that copy is reported."""
    c = BLOCK + "\n\nprompt one"
    split, reported = _composite_split({1: "prompt one", 3: c, 4: c}, [1, 3], {4: [None, 1]},
                                       [(4, "possible_replay_of", 3)])
    assert set(reported) <= {4}
    assert (split["composite_rows"], split["visible_surplus_rows"], split["silent_surplus_rows"]) == (0, 1, 0)


def test_composite_ids_exclude_a_reserved_merge_key_with_zero_capacity():
    """Review of #2495 (Astra, d1d1964c): a key with a reserved merge takes the worst case. Its one unmarked copy
    fills the one surplus copy, so the witnessed composite explains nothing and is not reported."""
    rows = todo_rows({1: ("S", "user", "prompt one"), 3: ("S", "user", BLOCK + "\n\nprompt one"),
                      4: ("S", "user", BLOCK + "\n\nprompt one")})
    relations = [(3, "composite_of", None, 0), (3, "composite_of", 1, 1)]
    forms, creditable = multiset.attribution_forms(relations, rows), set()
    split = multiset.split_surplus([(("user", multiset.h(rows[3][2])), 1, [3, 4], [3])], relations, rows, forms,
                                   creditable)
    assert forms["composite_rows"] == {3}
    assert sorted(forms["composite_rows"] & creditable) == []
    assert (split["composite_rows"], split["visible_surplus_rows"], split["silent_surplus_rows"]) == (0, 0, 1)
