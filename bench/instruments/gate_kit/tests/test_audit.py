"""Gate-kit audit, offline: tiny Hermes homes (the state.db tables the reliability scorers read, and an LCM-X store
created through the product's own MessageStore/SummaryDAG) scored end to end."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
import benchmarking.longmemeval as lme  # noqa: E402

lme._ensure_hermes_lcm_package()
from hermes_lcm.dag import SummaryDAG, SummaryNode  # noqa: E402
from hermes_lcm.store import MessageStore  # noqa: E402

from bench.instruments.gate_kit import audit  # noqa: E402

S = "ZQX-SENTINEL"  # in every content, argument and result field; it must never reach the output


def u(text):
    return {"role": "user", "content": f"{S} {text}"}


def a(text):
    return {"role": "assistant", "content": f"{S} {text}"}


def call(cid, name="read_file", arg="x"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps({"path": f"{S}/{arg}"})}}]}


def result(cid, text="out"):
    return {"role": "tool", "tool_call_id": cid, "content": f"{S} {text}"}


def turn(i, sid="S1"):
    return [(sid, u(f"question {i}")), (sid, a(f"answer {i}"))]


class Home:
    """``sessions``: {id: (source, parent, ended_at, end_reason)}; host and store rows: [(session, message)]. A host
    row's timestamp is its own unless the message names one (``_ts``): a carried handoff copy takes its original's."""

    def __init__(self, tmp_path, sessions, host, stored=None, *, relations=(), lineage=(), nodes=(), stamps=()):
        self.dir = tmp_path / "home"
        self.dir.mkdir()
        self.state, self.lcm = self.dir / "state.db", self.dir / "lcm.db"
        with sqlite3.connect(self.state) as con:
            con.execute("create table sessions (id text primary key, source text not null, parent_session_id text,"
                        " started_at real not null, ended_at real, end_reason text,"
                        " rewind_count integer not null default 0)")
            con.execute("create table messages (id integer primary key autoincrement, session_id text not null,"
                        " role text not null, content text, tool_call_id text, tool_calls text, tool_name text,"
                        " timestamp real not null, _compressed_summary integer not null default 0,"
                        " active integer not null default 1, compacted integer not null default 0)")
            for i, (sid, (source, parent, ended, reason, *started)) in enumerate(sessions.items()):
                con.execute("insert into sessions (id, source, parent_session_id, started_at, ended_at, end_reason)"
                            " values (?,?,?,?,?,?)",
                            (sid, source, parent, started[0] if started else 1000.0 + i, ended, reason))
            for n, (sid, m) in enumerate(host):
                con.execute("insert into messages (session_id, role, content, tool_call_id, tool_calls, timestamp,"
                            " active, compacted, _compressed_summary) values (?,?,?,?,?,?,?,?,?)",
                            (sid, m["role"], m.get("content"), m.get("tool_call_id"),
                             json.dumps(m["tool_calls"]) if m.get("tool_calls") else None, m.get("_ts", 2000.0 + n),
                             m.get("_active", 1), m.get("_compacted", 0), m.get("_compressed_summary", 0)))
        store = MessageStore(self.lcm)
        self.ids = [store.append(sid, {k: v for k, v in m.items() if not k.startswith("_")}
                                 | ({"timestamp": 1700000000.0 + n} if n in stamps else {})
                                 | ({"timestamp": m["_obs"]} if "_obs" in m else {}))
                    for n, (sid, m) in enumerate(host if stored is None else stored)]
        if relations:
            store.add_message_relations([[(s, kind, o, None, None)] for s, kind, o in relations])
        for child, parent in lineage:
            store.record_session_lineage(child, parent, source="subagent_start")
        store.close()
        if nodes:
            dag = SummaryDAG(self.lcm)
            for sid, ids in nodes:
                dag.add_node(SummaryNode(session_id=sid, summary=f"{S} leaf", source_ids=ids, created_at=1.0))
            dag.close()

    def run(self, logs=(), events=None, since=None, log_tz=None):
        paths = []
        for i, text in enumerate(logs):
            paths.append(self.dir / f"run-{i}.log")
            paths[-1].write_text(text)
        ev = None
        if events is not None:
            ev = self.dir / "events.jsonl"
            ev.write_text("".join(json.dumps(e) + "\n" for e in events))
        return audit.audit(self.state, self.lcm, paths, since, ev, log_tz)


ONE = {"S1": ("cli", None, 2000.0, "user_exit")}


def route(session, name, cursor, a, b, pairs=0, stripped=0):
    return (f"2026-11-06 10:00:00,000 INFO [{session}] hermes_lcm.engine: LCM ingest route: session={session} "
            f"route={name} cursor={cursor} reason=some reason text stored={a}-{b} rows={b - a + 1} "
            f"replay_pairs={pairs} stripped={stripped}\n")


def test_clean_session_passes_parity(tmp_path):
    out = Home(tmp_path, ONE, turn(1) + [("S1", call("c1")), ("S1", result("c1"))] + turn(2)).run()
    t = out["parity"]["totals"]
    assert (t["expected_rows"], t["stored_rows"], t["missing_rows"], t["surplus_rows"]) == (6, 6, 0, 0)
    bars = out["gate"]["bars"]
    assert all(bars[b]["pass"] for b in bars if b not in ("coverage_floors", "issue_1931_pass_A"))
    assert out["recorder"]["status"] == "recorder absent"
    assert out["gate"]["verdict"] == "GATE FAIL"  # #1931 PASS(A) needs n >= 5


def test_missing_row_fails_parity(tmp_path):
    host = turn(1) + turn(2)
    out = Home(tmp_path, ONE, host, stored=host[:3]).run()
    assert out["parity"]["totals"]["missing_rows"] == 1
    assert out["gate"]["bars"]["exact_content_deficits"] == {"value": 1, "pass": False}
    assert out["gate"]["verdict"] == "GATE FAIL"


def test_unexplained_duplicate_is_silent_and_a_replay_pair_makes_it_visible(tmp_path):
    host = turn(1)
    (tmp_path / "s").mkdir()
    silent = Home(tmp_path / "s", ONE, host, stored=host + [host[0]]).run()
    assert silent["parity"]["totals"]["silent_surplus_rows"] == 1
    assert silent["exempted_surplus"]["store_ids"]["unattributed"] == [3]
    assert silent["gate"]["bars"]["unattributed_silent_surplus"] == {"value": 1, "pass": False}
    (tmp_path / "v").mkdir()
    visible = Home(tmp_path / "v", ONE, host, stored=host + [host[0]], relations=[(3, "possible_replay_of", 1)]).run()
    t = visible["parity"]["totals"]
    assert (t["visible_surplus_rows"], t["silent_surplus_rows"]) == (1, 0)
    assert visible["m3"]["store_ids"] == [3]


CHILDREN = {"P": ("cli", None, 3000.0, "user_exit"), "A1": ("subagent", "P", 2500.0, "completed"),
            "B1": ("cron", "P", 2500.0, "completed")}


def test_child_buckets(tmp_path):
    host = turn(1, "P") + turn(2, "A1") + turn(3, "B1")
    out = Home(tmp_path, CHILDREN, host, stored=turn(1, "P")).run()
    a_, b_ = out["issue_1931"]["A"], out["issue_1931"]["B"]
    assert (a_["n"], a_["zero_row_children"], a_["missing_rows"]) == (1, ["A1"], 2)
    assert (b_["n"], b_["zero_row_children"], b_["missing_rows"]) == (1, ["B1"], 2)
    assert out["issue_1931"]["pass_A"] is False
    assert out["parity"]["zero_row_lineages"] == ["A1", "B1"]  # whole-run parity counts bucket B too
    assert out["parity"]["totals"]["missing_rows"] == 4


def test_a_b_child_with_a_missing_row_fails_the_gate(tmp_path):
    host = turn(1, "P") + turn(2, "A1") + turn(3, "B1")
    out = Home(tmp_path, CHILDREN, host, stored=host[:-1]).run()
    assert out["issue_1931"]["B"]["missing_rows"] == 1 and out["parity"]["totals"]["missing_rows"] == 1
    assert out["gate"]["bars"]["exact_content_deficits"] == {"value": 1, "pass": False}
    assert out["gate"]["verdict"] == "GATE FAIL"


def test_a_b_child_with_nothing_expected_is_reported_not_gated(tmp_path):
    host = turn(1, "P") + turn(2, "A1")
    out = Home(tmp_path, CHILDREN, host).run()
    assert out["issue_1931"]["B"]["zero_row_children"] == ["B1"]
    assert out["parity"]["zero_row_lineages"] == [] and out["gate"]["bars"]["zero_row_sessions"]["pass"]


def test_a_top_level_session_with_zero_rows_fails(tmp_path):
    out = Home(tmp_path, ONE, turn(1), stored=[]).run()
    assert out["parity"]["zero_row_lineages"] == ["S1"]
    assert out["gate"]["bars"]["zero_row_sessions"] == {"value": 1, "pass": False}


def test_lineage_row_counts_for_bucket_a_and_rotation_child_joins_its_parent(tmp_path):
    sessions = {"P": ("cli", None, 1000.5, "compression"), "P2": ("cli", "P", 3000.0, "user_exit"),
                "A1": ("subagent", "P2", 2500.0, "completed")}
    carried = [("P2", dict(m, _ts=2000.0 + n)) for n, (_s, m) in enumerate(turn(1, "P"))]  # the handoff copy
    host = turn(1, "P") + carried + turn(2, "P2") + [("P2", u("question 1"))] + turn(3, "A1")  # then a real repeat
    stored = turn(1, "P") + turn(2, "P2") + [("P2", u("question 1"))] + turn(3, "A1")
    out = Home(tmp_path, sessions, host, stored=stored, lineage=[("A1", "P2")]).run()
    assert out["parity"]["sessions"]["P2"]["lineage"] == "P"
    assert out["parity"]["lineages"]["P"]["expected_rows"] == 5
    assert out["parity"]["totals"]["missing_rows"] == out["parity"]["totals"]["surplus_rows"] == 0
    assert out["issue_1931"]["A"]["lineage_present"] == 1 and out["issue_1931"]["B"]["n"] == 0


def test_a_repeat_split_across_segments_is_expected_twice(tmp_path):
    # A parent's originals stay in the parent; the child holds only the handoff. The same text sent once in each
    # segment is two occurrences (the most copies in one segment would be one).
    sessions = {"P": ("cli", None, 1000.5, "compression"), "P2": ("cli", "P", 3000.0, "user_exit")}
    host = [("P", u("yes")), ("P2", u("yes"))]
    out = Home(tmp_path, sessions, host, stored=host[:1]).run()
    assert out["parity"]["totals"]["missing_rows"] == 1


def test_two_identical_sends_in_one_batch_both_count(tmp_path):
    # Two identical rows with one timestamp share a display identity (the host offsets each row of a batch by 1e-6 s,
    # so this is the worst case); both are still expected.
    host = [("S1", dict(u("yes"), _ts=5.0)), ("S1", dict(u("yes"), _ts=5.0)), ("S1", a("ok"))]
    out = Home(tmp_path, ONE, host, stored=host[1:]).run()
    assert out["parity"]["totals"]["expected_rows"] == 3 and out["parity"]["totals"]["missing_rows"] == 1


def test_in_place_rewound_rows_are_not_expected(tmp_path):
    host = turn(1) + [("S1", dict(u("question 2"), _active=0)),  # rewound: not displayed
                      ("S1", dict(u("question 3"), _active=0, _compacted=1))]  # archived history: displayed
    out = Home(tmp_path, ONE, host, stored=turn(1) + [("S1", u("question 3"))]).run()
    assert out["parity"]["totals"]["missing_rows"] == 0 and out["parity"]["totals"]["expected_rows"] == 3


@pytest.mark.parametrize("lose", [None, "original", "resend"])
def test_in_place_archived_tail_copies_keep_stamps_and_resend_counts(tmp_path, lose):
    tail = [u("retained question"), call("carried-call"), result("carried-call"), a("retained answer")]
    original = turn(0) + [("S1", dict(m, _ts=3000.0 + i)) for i, m in enumerate(tail)]
    archived = [(s, dict(m, _active=0, _compacted=1)) for s, m in original]
    carried = [("S1", dict(m, _ts=3000.0 + i)) for i, m in enumerate(tail)]  # copies keep the original's stamp
    # Tool-calling assistant identity uses call ids even when its rendered content changes.
    carried[1][1]["content"] = f"{S} carried rendering"
    resend = [("S1", dict(tail[0], _ts=5000.0)), ("S1", a("new answer"))]
    stored = original + resend
    if lose == "resend":
        stored = original + resend[1:]
    elif lose == "original":
        stored = original[:3] + original[4:] + resend  # lose the tool call, not just its copied rendering
    boundary = [("S1", dict(a("host summary"), _compressed_summary=1, _ts=4000.0))]
    out = Home(tmp_path, ONE, archived + boundary + carried + resend, stored=stored).run()
    totals = out["parity"]["totals"]
    assert totals["expected_rows"] == len(original) + len(resend)
    assert totals["missing_rows"] == int(lose is not None)
    assert totals["surplus_rows"] == 0
    assert S not in json.dumps(out)


def test_in_place_unmatched_user_ends_carry_block(tmp_path):
    archived = [("S1", dict(u("repeat"), _active=0, _compacted=1, _ts=3000.0))]
    # No carried tail here: a new user turn ends the insertion block. A later repeat is a real occurrence.
    live = [("S1", u("new turn")), ("S1", dict(u("repeat"), _ts=4000.0))]
    boundary = [("S1", dict(a("host summary"), _compressed_summary=1))]
    out = Home(tmp_path, ONE, archived + boundary + live, stored=archived + live[:1]).run()
    assert out["parity"]["totals"]["expected_rows"] == 3
    assert out["parity"]["totals"]["missing_rows"] == 1


def repeated_in_place():
    """Two summary boundaries; the first carried block was archived by the second compaction."""
    original = turn(0) + [("S1", dict(m, _ts=3000.0 + i)) for i, m in enumerate(
        [u("retained question"), call("carried-call"), result("carried-call"), a("retained answer")])]
    later = [(s, dict(m, _ts=3500.0 + i)) for i, (s, m) in enumerate(turn(2))]
    def archive(batch):
        return [(s, dict(m, _active=0, _compacted=1)) for s, m in batch]

    first = [(s, dict(m)) for s, m in original[2:]]
    first[1][1]["content"] = f"{S} first carried rendering"
    second = [(s, dict(m)) for s, m in first + later]  # a chained compaction copies the first block's copies
    second[1][1]["content"] = f"{S} second carried rendering"
    first_boundary = [("S1", dict(a("host summary"), _compressed_summary=1, _ts=4000.0))]
    boundary = [("S1", dict(a("host summary"), _compressed_summary=1, _ts=5000.0))]
    host = (archive(original) + archive(first_boundary) + archive(first) + archive(later) + boundary + second)
    return host, original + later


@pytest.mark.parametrize("lose", [False, True], ids=["kept", "missing-original"])
def test_in_place_multiple_compactions_deduplicate_archived_copies(tmp_path, lose):
    host, original = repeated_in_place()
    # Dropping a real tool call must remain visible even when both host carry blocks hold it.
    stored = original[:3] + original[4:] if lose else original
    out = Home(tmp_path, ONE, host, stored=stored).run()
    totals = out["parity"]["totals"]
    assert totals["expected_rows"] == len(original)
    assert totals["missing_rows"] == int(lose)
    assert totals["surplus_rows"] == 0
    assert S not in json.dumps(out)


@pytest.mark.parametrize("kind", ["merged", "pure", "prefix", "blank"])
def test_in_place_carrier_at_block_start_does_not_stop_matching(tmp_path, kind):
    original = [(s, dict(m, _ts=3000.0 + i)) for i, (s, m) in enumerate(turn(1))]
    archived = [(s, dict(m, _active=0, _compacted=1)) for s, m in original]
    text = {"merged": CARRIER + "\n\n" + u("question 1")["content"], "pure": CARRIER,
            "prefix": audit.plugin_tree.carrier_markers(REPO)[1][0] + S, "blank": "  "}[kind]
    boundary = [("S1", dict(a("host summary"), _compressed_summary=1))]
    host = archived + boundary + [("S1", {"role": "user", "content": text})] + original
    out = Home(tmp_path, ONE, host, stored=original, nodes=NODE).run()
    totals = out["parity"]["totals"]
    assert totals["expected_rows"] == 2
    assert totals["missing_rows"] == totals["surplus_rows"] == 0


@pytest.mark.parametrize("stamp", [3000.0, 6000.0], ids=["same-stamp", "new-stamp"])
def test_in_place_resend_after_second_block_still_counts(tmp_path, stamp):
    host, original = repeated_in_place()
    resend = ("S1", dict(u("retained question"), _ts=stamp))
    out = Home(tmp_path, ONE, host + [resend], stored=original).run()
    totals = out["parity"]["totals"]
    assert totals["expected_rows"] == len(original) + 1
    assert totals["missing_rows"] == 1
    assert totals["surplus_rows"] == 0


@pytest.mark.parametrize("signal", ["no-boundary", "no-compacted-column"])
def test_in_place_requires_boundary_and_compacted_evidence(tmp_path, signal):
    original = [("S1", dict(u("repeat"), _active=0, _compacted=1))]
    boundary = [] if signal == "no-boundary" else [("S1", dict(a("host summary"), _compressed_summary=1))]
    home = Home(tmp_path, ONE, original + boundary + [("S1", u("repeat"))], stored=original)
    if signal == "no-compacted-column":
        with sqlite3.connect(home.state) as con:
            con.execute("alter table messages drop column compacted")
    totals = home.run()["parity"]["totals"]
    assert totals["expected_rows"] == 2
    assert totals["missing_rows"] == 1


ROTATED = {"P": ("cli", None, 1000.5, "compression", 100.0), "P2": ("cli", "P", 3000.0, "user_exit", 1001.0)}


@pytest.mark.parametrize("lose", [False, True])
def test_a_lineage_partly_before_the_window_scores_its_window_members(tmp_path, lose):
    # The parent started before --since, its rotation child inside it: the child is scored, without the parent's rows
    # or the copy of them the child carries.
    carried = [("P2", dict(m, _ts=2000.0 + n)) for n, (_s, m) in enumerate(turn(1, "P"))]
    host = turn(1, "P") + carried + turn(2, "P2")
    stored = turn(1, "P") + turn(2, "P2")[: 1 if lose else 2]
    out = Home(tmp_path, ROTATED, host, stored=stored).run(since=1000.8)
    assert out["parity"]["lineages"]["P"]["expected_rows"] == 2
    assert out["parity"]["totals"]["missing_rows"] == int(lose) and out["parity"]["totals"]["surplus_rows"] == 0


def test_1674_needs_its_stamped_witness_in_the_rows_own_session(tmp_path):
    host = [("P", u("question 1")), ("P2", a("answer"))]
    stored = [("P", u("question 1")), ("P2", u("question 1")), ("P2", a("answer"))]
    events = [{"t": 0.0, "kind": "restart_cold", "session": "P2"}]
    out = Home(tmp_path, ROTATED, host, stored=stored, stamps={0}).run([route("P2", "advanced_cursor", 2, 2, 2)], events)
    assert out["exempted_surplus"]["store_ids"]["#1674"] == []  # the stamped copy is in the parent session
    assert out["exempted_surplus"]["store_ids"]["unattributed"] == [2]


T0 = 1700000100.0  # host timestamps of the undo tests (valid epoch seconds, so LCM-X keeps them as observed_at)


def undone(msg, ts):
    return dict(msg, _ts=ts, _active=0)  # rewind_to_message: active = 0, compacted left 0


def stamped(msg, ts):
    return dict(msg, _obs=ts)


def undo_home(tmp_path, extra=(), relations=()):
    """Turn 1, then turn 2 undone and re-sent: the host holds the undone rows inactive, LCM-X keeps them."""
    host = [("S1", dict(m, _ts=T0 + n)) for n, (_s, m) in enumerate(turn(1))] + [
        ("S1", undone(u("question 2"), T0 + 20)), ("S1", undone(a("answer 2"), T0 + 21)),
        ("S1", dict(u("question 2"), _ts=T0 + 30)), ("S1", dict(a("answer 2"), _ts=T0 + 31))]
    stored = [("S1", stamped(m, m["_ts"])) for _s, m in host] + list(extra)
    home = Home(tmp_path, ONE, host, stored=stored, relations=relations)
    with sqlite3.connect(home.state) as con:
        con.execute("update sessions set rewind_count = 1 where id = 'S1'")
    return home.run()


def test_undo_and_resend_is_retained_not_surplus(tmp_path):
    out = undo_home(tmp_path)
    t = out["parity"]["totals"]
    assert (t["expected_rows"], t["missing_rows"], t["surplus_rows"], t["silent_surplus_rows"]) == (4, 0, 0, 0)
    assert out["undo_retained"] == {"rows": 2, "store_ids": [3, 4], "host_rewind_count": {"S1": 1}}
    assert out["m3"]["denominator"] == 6


@pytest.mark.parametrize("copy", [u("question 2"), stamped(u("question 2"), T0 + 99)], ids=["null-observed", "other-observed"])
def test_a_replayed_copy_of_an_undone_row_stays_silent(tmp_path, copy):
    out = undo_home(tmp_path, extra=[("S1", copy)])
    assert out["undo_retained"]["store_ids"] == [3, 4]
    assert out["parity"]["totals"]["silent_surplus_rows"] == 1
    assert out["exempted_surplus"]["store_ids"]["unattributed"] == [7]


def test_one_undone_row_excuses_one_stored_copy(tmp_path):
    out = undo_home(tmp_path, extra=[("S1", stamped(u("question 2"), T0 + 20))])
    assert out["undo_retained"]["rows"] == 2  # one copy of the undone prompt and the undone reply
    assert out["parity"]["totals"]["silent_surplus_rows"] == 1


def test_undo_retained_rows_are_not_in_m3(tmp_path):
    out = undo_home(tmp_path, relations=[(3, "possible_replay_of", 5)])
    assert out["undo_retained"]["store_ids"] == [3, 4] and out["m3"]["numerator"] == 0


GENERATED = "[Recent Summary (d0, node 1)]\n" + S + " summary of the earlier turns"


@pytest.mark.parametrize("related", [True, False])
def test_a_generated_first_row_is_visible_only_with_its_relation(tmp_path, related):
    # #1286: LCM-X's own generated first row (a summary carrier) stored as a raw row. Its generated_unproven relation
    # makes it visible surplus and an M3 row; without the relation it is silent surplus.
    stored = [("S1", {"role": "user", "content": GENERATED})] + turn(1)
    out = Home(tmp_path, ONE, turn(1), stored=stored,
               relations=[(1, "generated_unproven", None)] if related else []).run()
    t = out["parity"]["totals"]
    assert (t["visible_surplus_rows"], t["silent_surplus_rows"]) == ((1, 0) if related else (0, 1))
    assert out["m3"]["store_ids"] == ([1] if related else [])
    assert out["exempted_surplus"]["store_ids"]["unattributed"] == ([] if related else [1])


def test_m3_rate_and_ids(tmp_path):
    host = turn(1) + turn(2)
    out = Home(tmp_path, ONE, host, relations=[(2, "generated_unproven", None), (4, "supersedes", None)]).run()
    assert out["m3"] == {"numerator": 1, "denominator": 4, "rate": 0.25, "store_ids": [2]}
    assert out["gate"]["bars"]["m3_rate"]["pass"] is False


def test_recorder_parsing_and_2305_rates(tmp_path):
    log = (route("S1", "persisted batch", 0, 1, 2, pairs=2, stripped=1) + route("S1", "persisted_batch", 3, 3, 3)
           + route("S1", "foreground", 0, 4, 4) + route("S1", "advanced_cursor", 2, 5, 5, pairs=2) + "noise line\n")
    out = Home(tmp_path, ONE, turn(1) + turn(2) + turn(3)).run([log])
    r = out["recorder"]
    assert r["present"] and r["ingests_per_route"] == {"advanced_cursor": 1, "foreground": 1, "persisted_batch": 2}
    assert (r["storing_ingests"], r["rate_2305"], r["stripped_per_replay_pair"]) == (4, 1 / 4, 0.25)


RESTART_BEFORE = [{"t": 0.0, "kind": "restart_cold", "session": "S1"}]


def attribution(tmp_path, log, stored_extra, stamps=frozenset({0}), host=None, events=RESTART_BEFORE):
    host = host or turn(1)
    return Home(tmp_path, ONE, host, stored=host + stored_extra, stamps=stamps).run(log, events)["exempted_surplus"]


@pytest.mark.parametrize("line", [None, ("persisted_batch", 0), ("advanced_cursor", 2), ("advanced_cursor", 0)])
def test_1674_needs_an_advanced_cursor_line_past_zero(tmp_path, line):
    ex = attribution(tmp_path, [route("S1", *line, 3, 3)] if line else [], [("S1", u("question 1"))])
    if line == ("advanced_cursor", 2):
        assert ex["store_ids"]["#1674"] == [3] and ex["unattributed_silent_rows"] == 0
        assert ex["fraction"] == 1 / 3 and ex["met"] is False
    else:
        assert ex["store_ids"]["#1674"] == [] and ex["store_ids"]["unattributed"] == [3]


def test_1674_needs_an_earlier_stamped_copy(tmp_path):
    ex = attribution(tmp_path, [route("S1", "advanced_cursor", 2, 3, 3)], [("S1", u("question 1"))], stamps=frozenset())
    assert ex["store_ids"]["#1674"] == [] and ex["store_ids"]["unattributed"] == [3]


@pytest.mark.parametrize("events,limit", [(None, "#1674_needs_manual_check"), ([], "unattributed"),
                                          ([{"t": 4e9, "kind": "restart_warm", "session": "S1"}], "unattributed"),
                                          ([{"t": 0.0, "kind": "restart_cold", "session": "S9"}], "unattributed"),
                                          (RESTART_BEFORE, "#1674"),
                                          ([{"t": 0.0, "kind": "restart_cold_shared", "session": "S1"}], "#1674")],
                         ids=["no-events", "none", "after", "other", "before", "shared"])
def test_1674_needs_a_restart_before_the_recorder_line(tmp_path, events, limit):
    ex = attribution(tmp_path, [route("S1", "advanced_cursor", 2, 3, 3)], [("S1", u("question 1"))], events=events)
    assert ex["store_ids"][limit] == [3]
    assert ex["exempted_rows"] == (limit == "#1674")


@pytest.mark.parametrize("lines", [[route("S1", "S3", 0, 3, 3), route("S1", "S3", 0, 2, 4)], [route("S9", "S3", 0, 3, 3)]],
                         ids=["overlapping", "other-session"])
def test_attribution_needs_exactly_one_line_of_the_rows_session(tmp_path, lines):
    ex = attribution(tmp_path, ["".join(lines)], [("S1", u("question 1"))])
    assert ex["store_ids"]["unattributed"] == [3]


@pytest.mark.parametrize("name,reason,limit", [("S3", "tail retry", "#2203"), ("cold", "append", "#2203"),
                                               ("session_end", "digest", "#2203"),
                                               ("session_end", "append", "unattributed"),
                                               ("warm_tail", "digest", "unattributed")])
def test_2203_routes(tmp_path, name, reason, limit):
    line = route("S1", name, 0, 3, 3).replace("reason=some reason text", f"reason={reason}")
    ex = attribution(tmp_path, [line], [("S1", u("question 1"))])
    assert ex["store_ids"][limit] == [3]


def test_a_recorder_line_logged_to_two_files_is_one_ingest(tmp_path):
    line = route("S1", "S3", 0, 3, 3)
    out = Home(tmp_path, ONE, turn(1), stored=turn(1) + [("S1", u("question 1"))], stamps={0}).run([line, line])
    assert out["recorder"]["storing_ingests"] == 1
    assert out["exempted_surplus"]["store_ids"]["#2203"] == [3]


def test_a_row_with_content_and_a_tool_call_counts_once_in_the_exempted_fraction(tmp_path):
    speak_and_call = dict(call("c1"), content=f"{S} let me read it")
    host = [("S1", u("question 1")), ("S1", speak_and_call), ("S1", result("c1"))]
    out = Home(tmp_path, ONE, host, stored=host + [("S1", speak_and_call)], stamps={0}).run(
        [route("S1", "S3", 0, 4, 4)])
    ex = out["exempted_surplus"]
    assert out["parity"]["totals"]["silent_surplus_rows"] == 2  # one copy each of its content key and its call key
    assert ex["store_ids"]["#2203"] == [4] and ex["exempted_rows"] == 1 and ex["fraction"] == 1 / 4
    assert ex["unattributed_silent_rows"] == 0


def test_2203_needs_an_unstamped_copy(tmp_path):
    ex = attribution(tmp_path, [route("S1", "S3", 0, 3, 3)], [("S1", u("question 1"))], stamps=frozenset({0, 2}))
    assert ex["store_ids"]["#2203"] == [] and ex["store_ids"]["unattributed"] == [3]


def test_849_and_1961_are_listed_for_a_hand_check_and_stay_silent(tmp_path):
    host = [("S1", u("first")), ("S1", u("second"))]
    composite = {"role": "user", "content": f"{S} first\n\n{S} second"}
    (tmp_path / "c").mkdir()
    ex = attribution(tmp_path / "c", [route("S1", "foreground", 2, 3, 3)], [("S1", composite)], host=host)
    assert ex["store_ids"]["#849_needs_manual_check"] == [3] and ex["unattributed_silent_rows"] == 1
    (tmp_path / "w").mkdir()
    ex = attribution(tmp_path / "w", [route("S1", "foreground", 0, 3, 3)], [("S1", u("question 1"))])
    assert ex["store_ids"]["#1961_needs_manual_check"] == [3] and ex["exempted_rows"] == 0


@pytest.mark.parametrize("line", [
    "2026-11-06 10:00:00,000 INFO [A1] hermes_lcm.engine: LCM reclassified session A1 as auxiliary at first ingest "
    "after bind-time detection missed\n",
    "2026-11-06 10:00:00,000 INFO hermes_lcm.engine: LCM session decision: session=A1 platform=subagent parent=P "
    "decision=auxiliary reason=background_review engine=lcm\n"], ids=["reclassified", "session-decision"])
def test_auxiliary_children_are_excluded(tmp_path, line):
    host = turn(1, "P") + turn(2, "A1")
    out = Home(tmp_path, {k: v for k, v in CHILDREN.items() if k != "B1"}, host, stored=turn(1, "P")).run([line])
    assert out["issue_1931"]["auxiliary_excluded"] == ["A1"] and out["issue_1931"]["A"]["n"] == 0
    assert out["parity"]["zero_row_lineages"] == [] and out["parity"]["totals"]["missing_rows"] == 0


def test_a_frame_marker_auxiliary_subagent_child_stays_scored(tmp_path):
    line = ("2026-11-06 10:00:00,000 INFO hermes_lcm.engine: LCM session decision: session=A1 platform=subagent "
            "parent=P decision=auxiliary reason=frame_marker engine=lcm\n")
    out = Home(tmp_path, {k: v for k, v in CHILDREN.items() if k != "B1"}, turn(1, "P") + turn(2, "A1"),
               stored=turn(1, "P")).run([line])
    assert out["issue_1931"]["auxiliary_excluded"] == [] and out["issue_1931"]["auxiliary_unproven"] == ["A1"]
    assert out["issue_1931"]["A"]["zero_row_children"] == ["A1"] and out["issue_1931"]["pass_A"] is False
    assert out["parity"]["zero_row_lineages"] == ["A1"]


def test_auxiliary_sources_do_not_meet_the_host_source_floor(tmp_path):
    sessions = dict(ONE, BG=("cron", None, 3000.0, "completed"))
    line = ("2026-11-06 10:00:00 INFO hermes_lcm.engine: LCM session decision: session=BG platform=cron "
            "parent=- decision=auxiliary reason=background_review engine=lcm\n")
    out = Home(tmp_path, sessions, turn(1) + turn(2, "BG"), stored=turn(1)).run([line])
    assert out["issue_1931"]["auxiliary_excluded"] == ["BG"]
    assert out["coverage"]["host_sources"] == {"value": 1, "floor": 2, "met": False}


def test_a_frame_marker_auxiliary_top_level_session_stays_scored(tmp_path):
    line = ("2026-11-06 10:00:00,000 INFO hermes_lcm.engine: LCM session decision: session=S1 platform=cli parent=- "
            "decision=auxiliary reason=frame_marker engine=lcm\n")
    out = Home(tmp_path, ONE, turn(1), stored=[]).run([line])
    assert out["issue_1931"]["auxiliary_unproven"] == ["S1"] and out["parity"]["zero_row_lineages"] == ["S1"]


def test_lines_of_other_loggers_or_mid_message_do_not_count(tmp_path):
    log = ("2026-11-06 10:00:00,000 INFO [S1] gateway.run: LCM ingest route: session=S1 route=S3 cursor=0 reason=x "
           "stored=3-3 rows=1 replay_pairs=0 stripped=0\n"
           "2026-11-06 10:00:00,000 INFO [S1] hermes_lcm.engine: quoted: LCM compaction #1: done\n"
           "2026-11-06 10:00:00,000 INFO hermes_lcm.engine: echo LCM session decision: session=S1 platform=cli parent=- "
           "decision=auxiliary reason=background_review engine=lcm\n")
    out = Home(tmp_path, ONE, turn(1)).run([log])
    assert out["recorder"]["present"] is False and out["coverage"]["compactions"]["value"] == 0
    assert out["issue_1931"]["auxiliary_excluded"] == [] and "S1" in out["parity"]["lineages"]


def test_compaction_lines_are_deduplicated_and_limited_to_scored_sessions_in_the_window(tmp_path):
    inside = datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc).timestamp()
    sessions = {"OLD": ("cli", None, None, None, inside - 7200), "S1": ("cli", None, None, None, inside - 60)}
    line = "{} INFO [{}] hermes_lcm.compaction: LCM compaction #{}: done\n"
    log = (line.format("2026-11-06 10:00:00,000", "S1", 1) + line.format("2026-11-06 10:00:05,000", "S1", 2)
           + line.format("2026-11-06 10:00:06,000", "OLD", 1) + line.format("2026-11-06 08:00:00,000", "S1", 1))
    out = Home(tmp_path, sessions, turn(1, "OLD") + turn(2, "S1")).run([log, log], since=inside - 3600, log_tz="UTC")
    assert out["coverage"]["compactions"]["value"] == 2  # S1's two in-window lines, each logged to both files


def test_a_store_decision_is_not_auxiliary(tmp_path):
    line = ("2026-11-06 10:00:00,000 INFO hermes_lcm.engine: LCM session decision: session=A1 platform=subagent "
            "parent=P decision=delegate_child reason=subagent_start engine=lcm\n")
    out = Home(tmp_path, {k: v for k, v in CHILDREN.items() if k != "B1"}, turn(2, "A1"), stored=[]).run([line])
    assert out["issue_1931"]["auxiliary_excluded"] == [] and out["issue_1931"]["A"]["zero_row_children"] == ["A1"]


@pytest.mark.parametrize("tz,counted", [("UTC", 0), ("Asia/Tokyo", 1)])
def test_log_tz_decides_whether_a_compaction_precedes_a_restart(tmp_path, tz, counted):
    restart = datetime(2026, 11, 6, 9, 0, tzinfo=timezone.utc).timestamp()  # 10:00 UTC is after it, 10:00 JST before
    log = "2026-11-06 10:00:00,000 INFO [S1] hermes_lcm.compaction: LCM compaction #1: done\n"
    out = Home(tmp_path, ONE, turn(1)).run([log], [{"t": restart, "kind": "restart_cold", "session": "S1"}], log_tz=tz)
    assert out["coverage"]["compactions_then_restart"]["value"] == counted


ACP_AT = datetime(2026, 11, 6, 10, 0, 30, tzinfo=timezone.utc).timestamp()
ACP_LINE = "2026-11-06 10:00:30 [INFO] hermes_plugins.hermes_lcm_x.compaction: LCM compaction #1: 9 messages → 4{}\n"


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("acp_first", [False, True])
def test_mixed_log_compactions_keep_distinct_sessions(tmp_path, count, acp_first):
    sessions = {f"S{i}": ONE["S1"] for i in range(1, count + 1)}
    file_log = "".join(f"2026-11-06 10:00:30 INFO [{sid}] hermes_lcm.compaction: "
                       "LCM compaction #1: 9 messages → 4\n" for sid in sessions)
    logs = [ACP_LINE.format(""), file_log] if acp_first else [file_log, ACP_LINE.format("")]
    events = [{"t": ACP_AT - 20, "kind": "compress_request", "session": "S1"}]
    events += [{"t": ACP_AT + 60, "kind": "restart_cold", "session": sid} for sid in sessions]
    home = Home(tmp_path, sessions, [row for sid in sessions for row in turn(1, sid)])
    out = home.run(logs, events, log_tz="UTC")
    assert out["coverage"]["compactions"]["value"] == count
    assert out["coverage"]["compactions_then_restart"]["value"] == count
    records = audit.read_logs(sorted(home.dir.glob("run-*.log")), "UTC")["compactions"]
    assert {sid for sid, _at, _msg in records} == set(sessions)


@pytest.mark.parametrize("request_at,suffix,attributed", [
    (ACP_AT - 20, "", True),  # the driver's compress_request 20 s before the line
    (ACP_AT - 600, "", False),  # too long before: no session, counted in compactions only
    (None, " session=S1", True),  # the message names its session
], ids=["nearest-request", "no-request-in-window", "session-in-message"])
def test_acp_stderr_compaction_lines(tmp_path, request_at, suffix, attributed):
    # acp_adapter/entry.py logs "<asctime> [<LEVEL>] <logger>: <message>" to stderr, with no session field.
    events = [{"t": ACP_AT + 60, "kind": "restart_cold", "session": "S1"}]
    if request_at is not None:
        events.append({"t": request_at, "kind": "compress_request", "session": "S1"})
    out = Home(tmp_path, ONE, turn(1)).run([ACP_LINE.format(suffix)], events, log_tz="UTC")
    assert out["coverage"]["compactions"]["value"] == 1
    assert out["coverage"]["compactions_then_restart"]["value"] == int(attributed)
    assert out["coverage_basis"]["compactions_without_session"] == int(not attributed)


def test_acp_stderr_lines_of_other_loggers_do_not_count(tmp_path):
    line = "2026-11-06 10:00:30 [INFO] agent.context_compressor: LCM compaction #1: 9 messages → 4\n"
    assert Home(tmp_path, ONE, turn(1)).run([line])["coverage"]["compactions"]["value"] == 0


def test_unknown_log_tz_is_an_input_error(tmp_path):
    home = Home(tmp_path, ONE, turn(1))
    with pytest.raises(SystemExit) as stop:
        audit.main(["--state-db", str(home.state), "--lcm-db", str(home.lcm), "--log-tz", "Nowhere/Zone",
                    "--out", str(tmp_path / "r.json")])
    assert stop.value.code == 2


def floor_run(tmp_path, *, compactions=20, recorder=True, replay=False):
    sessions = {"P": ("cli", None, 9000.0, "user_exit"), "T": ("telegram", None, 9000.0, "user_exit")}
    host, lineage = [], []
    for k in range(5):
        sessions[f"A{k}"] = ("subagent", "P", 5000.0, "completed")
        host += turn(k, f"A{k}")
        lineage.append((f"A{k}", "P"))
    for i in range(190):
        host += turn(i, "P")
    for i in range(50):
        host += [("T", call(f"c{i}", arg=str(i % 40))), ("T", result(f"c{i}", str(i)))]
    host += turn(0, "T")
    events = [{"t": 1e10, "kind": k, "session": "P"} for k in ("restart_cold", "restart_warm", "undo_resend")
              for _ in range(5)] + [{"t": 1e10, "kind": "undo_resend", "session": "P"}] * 5
    log = "".join(f"2026-11-06 10:00:{i:02d},000 INFO [P] hermes_lcm.compaction: LCM compaction #{i + 1}: done\n"
                  for i in range(compactions)) + (route("P", "foreground", 0, 1, 2) if recorder else "")
    stored, stamps = host, ()
    if replay:  # one #1674 copy of P's first prompt (host row 10), after a restart
        stored, stamps = host + [("P", u("question 0"))], {10}
        log += route("P", "advanced_cursor", 2, len(host) + 1, len(host) + 1)
        events = events + [{"t": 0.0, "kind": "restart_cold", "session": "P"}]
    home = Home(tmp_path, sessions, host, stored, lineage=lineage, nodes=[("P", [11])], stamps=stamps)
    return home.run([log], events)


def test_every_floor_met_gives_gate_pass_and_an_unmet_floor_gives_incomplete(tmp_path, monkeypatch):
    (tmp_path / "ok").mkdir()
    out = floor_run(tmp_path / "ok")
    assert {k: c["value"] for k, c in out["coverage"].items()} == {
        "turns": 196, "delegated_children": 5, "compactions": 20, "compactions_then_restart": 20, "restarts": 10,
        "restarts_cold": 5, "restarts_warm": 5, "undo_resend": 10, "tool_calls": 50, "tool_call_repeats": 10,
        "host_sources": 2}  # cli and telegram; subagent children are not entry points
    assert out["coverage_basis"]["store_depth0_message_leaves"] == 1
    failing = {k for k, c in out["coverage"].items() if not c["met"]}
    assert failing == {"turns", "delegated_children"}
    assert out["issue_1931"]["pass_A"] is True
    assert out["gate"]["verdict"] == "INCOMPLETE"
    monkeypatch.setitem(audit.FLOORS, "turns", 196)  # the same run with its two short floors lowered passes
    monkeypatch.setitem(audit.FLOORS, "delegated_children", 5)
    (tmp_path / "pass").mkdir()
    assert floor_run(tmp_path / "pass")["gate"]["verdict"] == "GATE PASS"
    (tmp_path / "absent").mkdir()
    absent = floor_run(tmp_path / "absent", recorder=False)
    assert absent["recorder"]["status"] == "recorder absent" and absent["gate"]["verdict"] == "INCOMPLETE"
    (tmp_path / "replay").mkdir()
    replay = floor_run(tmp_path / "replay", replay=True)
    assert replay["exempted_surplus"]["store_ids"]["#1674"] == [493]  # 492 host rows, then the copy
    assert replay["gate"]["verdict"] == "GATE PASS PENDING HAND CHECK"


def test_events_count_only_for_audited_sessions_in_the_window(tmp_path):
    events = [{"t": 1500.0, "kind": "restart_cold", "session": "S1"}, {"t": 1500.0, "kind": "restart_cold", "session": "ZZ"},
              {"t": 500.0, "kind": "restart_warm", "session": "S1"}, {"t": 1500.0, "kind": "undo_resend", "session": "S1"}]
    out = Home(tmp_path, ONE, turn(1)).run(events=events, since=999.0)
    assert {k: out["coverage"][k]["value"] for k in ("restarts_cold", "restarts_warm", "undo_resend")} == {
        "restarts_cold": 1, "restarts_warm": 0, "undo_resend": 1}


def test_no_events_reports_unknown_floors(tmp_path):
    out = Home(tmp_path, ONE, turn(1)).run()
    assert out["coverage"]["restarts"] == {"value": "unknown", "floor": 10, "met": False}
    assert out["coverage"]["compactions"]["value"] == "unknown"


def test_since_excludes_earlier_sessions(tmp_path):
    sessions = {"OLD": ("cli", None, 1500.0, "user_exit"), "NEW": ("cli", None, 1600.0, "user_exit")}
    out = Home(tmp_path, sessions, turn(1, "OLD") + turn(2, "NEW"), stored=turn(2, "NEW")).run(since=1000.5)
    assert sorted(out["parity"]["lineages"]) == ["NEW"] and out["parity"]["totals"]["missing_rows"] == 0


def test_output_holds_no_content_and_sources_open_read_only(tmp_path, capsys):
    host = turn(1) + [("S1", call("c1")), ("S1", result("c1"))] + turn(2)
    home = Home(tmp_path, ONE, host, stored=host[:3] + [host[0]], relations=[(4, "possible_replay_of", 1)])
    before = {p: p.read_bytes() for p in (home.state, home.lcm)}
    files = sorted(p.name for p in home.dir.iterdir())
    out = tmp_path / "result.json"
    assert audit.main(["--state-db", str(home.state), "--lcm-db", str(home.lcm), "--out", str(out)]) == 0
    text, printed = out.read_text(), capsys.readouterr().out
    assert S not in text and S not in printed and "verdict: GATE FAIL" in printed
    assert json.loads(text)["parity"]["totals"]["missing_rows"] == 3  # the result and turn 2
    assert {p: p.read_bytes() for p in before} == before
    assert sorted(p.name for p in home.dir.iterdir()) == files  # immutable=1: no -wal or -shm beside the sources


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_a_non_empty_wal_or_hot_journal_is_refused(tmp_path, suffix):
    home = Home(tmp_path, ONE, turn(1))
    Path(f"{home.lcm}{suffix}").write_bytes(b"pending pages")
    with pytest.raises(SystemExit) as stop:
        audit.main(["--state-db", str(home.state), "--lcm-db", str(home.lcm), "--out", str(tmp_path / "r.json")])
    assert stop.value.code == 2 and not (tmp_path / "r.json").exists()


def test_a_stored_only_session_whose_payloads_are_all_unreadable_is_kept(tmp_path):
    ref = "gone.json"
    placeholder = (f"[Externalized tool output: tool=read_file; tool_call_id=c9; chars=10; bytes=10; "
                   f'read it with lcm_expand(externalized_ref="{ref}"); ref={ref}]')
    out = Home(tmp_path, ONE, turn(1), stored=turn(1) + [("GHOST", dict(result("c9"), content=placeholder))]).run()
    assert out["parity"]["lineages"]["GHOST"]["externalized_unresolved"] == 1
    assert out["gate"]["bars"]["externalized_unresolved"] == {"value": 1, "pass": False}
    assert out["gate"]["verdict"] == "GATE FAIL"


def test_a_symlinked_database_is_checked_beside_its_target(tmp_path):
    home = Home(tmp_path, ONE, turn(1))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = elsewhere / "lcm.db"
    home.lcm.rename(target)
    home.lcm.symlink_to(target)
    Path(f"{target}-wal").write_bytes(b"pending pages")
    with pytest.raises(SystemExit) as stop:
        audit.main(["--state-db", str(home.state), "--lcm-db", str(home.lcm), "--out", str(tmp_path / "r.json")])
    assert stop.value.code == 2 and not (tmp_path / "r.json").exists()


def test_an_unresolved_externalized_payload_fails_its_own_bar(tmp_path):
    ref = "missing-payload.json"
    placeholder = (f"[Externalized tool output: tool=read_file; tool_call_id=c1; chars=10; bytes=10; "
                   f'read it with lcm_expand(externalized_ref="{ref}"); ref={ref}]')
    host = turn(1) + [("S1", call("c1")), ("S1", result("c1"))]
    out = Home(tmp_path, ONE, host, stored=host[:3] + [("S1", dict(result("c1"), content=placeholder))]).run()
    assert out["gate"]["bars"]["externalized_unresolved"] == {"value": 1, "pass": False}
    assert out["parity"]["totals"]["missing_rows"] == 1  # the host's result is a deficit too, as in B2


@pytest.mark.parametrize("copies", [1, 2])
def test_an_in_place_compaction_clone_does_not_excuse_a_stored_copy(tmp_path, copies):
    # In-place compaction rewinds the carried original (active 0, compacted 0) next to a live clone with its timestamp:
    # the original is not an undone row, and the one stored copy is the clone's.
    host = [("S1", undone(u("question 1"), T0)), ("S1", dict(u("question 1"), _ts=T0)), ("S1", dict(a("answer 1"), _ts=T0 + 1))]
    stored = [("S1", stamped(u("question 1"), T0))] * copies + [("S1", stamped(a("answer 1"), T0 + 1))]
    out = Home(tmp_path, ONE, host, stored=stored).run()
    t = out["parity"]["totals"]
    assert out["undo_retained"]["rows"] == 0
    assert (t["expected_rows"], t["missing_rows"], t["silent_surplus_rows"]) == (2, 0, copies - 1)


def test_a_rotation_child_holding_more_copies_than_its_earlier_parent_expects_the_rest(tmp_path):
    again = dict(u("again"), _ts=1500.0)  # the parent's row, carried to the child, and the child's second copy of it
    out = Home(tmp_path, ROTATED, [("P", again), ("P2", again), ("P2", again)], stored=[("P", u("again"))]).run(since=1000.8)
    assert out["parity"]["lineages"]["P"]["expected_rows"] == 1 and out["parity"]["totals"]["missing_rows"] == 1


CARRIER = "[Recent Summary (d0, node 1)]\n" + S + " leaf\n[Expand for details: ]"  # node 1 as Home(nodes=) writes it
NODE = [("S1", [1])]


def outcome(out):
    """The verdict on the data alone: the unit homes meet no coverage floor, hold no #1931 bucket A, and are so small
    that any exempted row is over the 1% bar."""
    bars = out["gate"]["bars"]
    if not all(v["pass"] for b, v in bars.items()
               if b not in ("coverage_floors", "issue_1931_pass_A", "exempted_surplus_fraction")):
        return "FAIL"
    return "HAND CHECK" if out["gate"].get("hand_checks") or out["exempted_surplus"]["exempted_rows"] else "PASS"


@pytest.mark.parametrize("separator", ["\n\n", "\n\n---\n\n"])
def test_a_user_request_merged_behind_a_summary_carrier_is_expected(tmp_path, separator):
    merged = {"role": "user", "content": CARRIER + separator + S + " the unique request"}
    host = [("S1", merged), ("S1", a("answer"))]
    (tmp_path / "lost").mkdir()
    lost = Home(tmp_path / "lost", ONE, host, stored=[("S1", a("answer"))], nodes=NODE).run()
    assert lost["parity"]["totals"]["missing_rows"] == 1 and outcome(lost) == "FAIL"
    (tmp_path / "kept").mkdir()
    kept = Home(tmp_path / "kept", ONE, host, stored=[("S1", u("the unique request")), ("S1", a("answer"))], nodes=NODE).run()
    assert outcome(kept) == "PASS" and kept["parity"]["lineages"]["S1"]["host_carrier_rows_excluded"] == 0


@pytest.mark.parametrize("later", [0.0, 100.0], ids=["same-time", "later"])
def test_a_carrier_repeating_a_kept_user_rows_text_is_never_a_pass(tmp_path, later):
    # Scout: "continue" at T, then a carrier merged with "continue" at T+100: a second occurrence, or the host's merged
    # copy of the first; a hand check. The same text at the same time is one occurrence (control: PASS).
    merged = {"role": "user", "content": CARRIER + "\n\n" + S + " continue", "_ts": T0 + later}
    host = [("S1", dict(u("continue"), _ts=T0)), ("S1", dict(a("ok"), _ts=T0 + 1)), ("S1", merged)]
    out = Home(tmp_path, ONE, host, stored=host[:2], nodes=NODE).run()
    assert outcome(out) == ("HAND CHECK" if later else "PASS")
    assert out["gate"]["hand_checks"] == ([{"reason": "merged_carrier_repeat", "lineage": "S1", "host_ids": [3]}]
                                          if later else [])


CHAIN = {"P": ("cli", None, 1000.5, "compression", 100.0), "P2": ("cli", "P", 1002.0, "compression", 1001.0),
         "P3": ("cli", "P2", 3000.0, "user_exit", 1002.5)}


@pytest.mark.parametrize("replay", [False, None, T0 + 99], ids=["no-replay", "null-observed", "other-observed"])
@pytest.mark.parametrize("carried", [True, False], ids=["carried-time", "own-time"])
def test_two_carriers_of_one_occurrence_leave_a_replay_surplus(tmp_path, replay, carried):
    # Astra: one plain occurrence and two rotation children's different carriers holding it; an unmarked replay of it
    # is surplus and fails. Without the replay (control) it passes, or is a hand check when the carriers have their own
    # times (each could be a resend).
    def merged(node, ts):
        return dict(role="user", content=CARRIER.replace("node 1", f"node {node}") + "\n\n" + S + " continue", _ts=ts)
    host = [("P", dict(u("continue"), _ts=T0)), ("P2", merged(1, T0 if carried else T0 + 50)),
            ("P3", merged(2, T0 if carried else T0 + 60))]
    stored = [("P", stamped(u("continue"), T0))] + (
        [] if replay is False else [("P3", u("continue") if replay is None else stamped(u("continue"), replay))])
    out = Home(tmp_path, CHAIN, host, stored=stored, nodes=[("P3", [1]), ("P3", [2])]).run()
    assert outcome(out) == ("FAIL" if replay is not False else "PASS" if carried else "HAND CHECK")
    assert out["exempted_surplus"]["store_ids"]["unattributed"] == ([] if replay is False else [2])


@pytest.mark.parametrize("node", [False, True], ids=["no-node", "node-in-store"])
def test_a_summary_shaped_row_is_a_carrier_only_when_its_node_is_in_the_store(tmp_path, node):
    # Astra: a user message in summary shape with no node behind it is the user's own content, so losing it fails.
    host = [("S1", {"role": "user", "content": CARRIER})] + turn(1)
    out = Home(tmp_path, ONE, host, stored=turn(1), nodes=NODE if node else ()).run()
    assert outcome(out) == ("PASS" if node else "FAIL")
    assert out["parity"]["lineages"]["S1"]["host_carrier_rows_excluded"] == int(node)


@pytest.mark.parametrize("line", [("S3", 0), ("advanced_cursor", 2)])
@pytest.mark.parametrize("in_order", [False, True], ids=["A-B-A-B", "A-A-B-B"])
def test_a_repeated_prompt_stored_out_of_order_is_not_exempted(tmp_path, line, in_order):
    # Astra: host A, A, B; stored A, B, A, B: the second A was stored after B, so the stored rows cannot follow the
    # host's in order and the duplicate B is unattributed. Stored A, A, B, B (control) keeps its exemption.
    go, done = u("go"), a("done")
    stored = [go, go, done, done] if in_order else [go, done, go, done]
    out = Home(tmp_path, ONE, [("S1", go), ("S1", go), ("S1", done)], stored=[("S1", m) for m in stored],
               stamps={2 if in_order else 1}).run([route("S1", *line, 3, 4)], RESTART_BEFORE)
    limit = "#2203" if line[0] == "S3" else "#1674"
    assert out["exempted_surplus"]["store_ids"][limit if in_order else "unattributed"] == [4]
    assert outcome(out) == ("HAND CHECK" if in_order else "FAIL")


def test_repeated_text_that_leaves_the_order_ambiguous_is_a_hand_check(tmp_path):
    # The host undid "go" and the user sent it again; the store holds one "go": which host row it is, is ambiguous.
    host = [("S1", undone(u("go"), T0)), ("S1", dict(u("go"), _ts=T0 + 1)), ("S1", dict(a("done"), _ts=T0 + 2))]
    out = Home(tmp_path, ONE, host, stored=[("S1", u("go")), ("S1", a("done")), ("S1", a("done"))]).run(
        [route("S1", "S3", 0, 3, 3)])
    assert out["exempted_surplus"]["store_ids"]["order_ambiguous"] == [3] and outcome(out) == "HAND CHECK"
    assert {"reason": "view_order_ambiguous", "store_ids": [3]} in out["gate"]["hand_checks"]


def test_a_symlinked_state_db_reads_payloads_beside_its_target(tmp_path):
    ref = "payload-c1.json"
    placeholder = (f"[Externalized tool output: tool=read_file; tool_call_id=c1; chars=10; bytes=10; "
                   f'read it with lcm_expand(externalized_ref="{ref}"); ref={ref}]')
    host = turn(1) + [("S1", call("c1")), ("S1", result("c1"))]
    home = Home(tmp_path, ONE, host, stored=host[:3] + [("S1", dict(result("c1"), content=placeholder))])
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / audit.externalized.PAYLOAD_DIRNAME).mkdir(parents=True)
    (elsewhere / audit.externalized.PAYLOAD_DIRNAME / ref).write_text(json.dumps({"content": result("c1")["content"]}))
    home.state.rename(elsewhere / "state.db")
    home.state.symlink_to(elsewhere / "state.db")
    out = home.run()
    assert out["gate"]["bars"]["externalized_unresolved"] == {"value": 0, "pass": True}
    assert out["parity"]["totals"]["missing_rows"] == 0


@pytest.mark.parametrize("line", [("S3", 0), ("advanced_cursor", 2)])
def test_a_host_insertion_stored_before_the_copy_is_neither_2203_nor_1674(tmp_path, line):
    # The host inserted X mid-history after question 2 was stored; a retry stores X and then question 2 again (#2305's
    # insert drained through a retry route): the stored rows no longer follow the host's in order.
    host = turn(1) + [("S1", u("inserted"))] + turn(2)
    stored = turn(1) + turn(2) + [("S1", u("inserted")), ("S1", u("question 2"))]
    out = Home(tmp_path, ONE, host, stored=stored, stamps={2}).run([route("S1", *line, 5, 6)], RESTART_BEFORE)
    ex = out["exempted_surplus"]
    assert out["parity"]["totals"]["silent_surplus_rows"] == 1
    assert ex["store_ids"]["#2203"] == ex["store_ids"]["#1674"] == [] and ex["store_ids"]["unattributed"] == [6]
    assert out["gate"]["verdict"] == "GATE FAIL"


def test_a_restart_earlier_in_the_same_second_precedes_the_ingest(tmp_path):
    second = datetime(2026, 11, 6, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    line = route("S1", "advanced_cursor", 2, 3, 3).replace("10:00:00,000", "10:00:00,900")
    out = Home(tmp_path, ONE, turn(1), stored=turn(1) + [("S1", u("question 1"))], stamps={0}).run(
        [line], [{"t": second + 0.1, "kind": "restart_cold", "session": "S1"}], log_tz="UTC")
    assert out["exempted_surplus"]["store_ids"]["#1674"] == [3]


@pytest.mark.parametrize("where", ["the-lcm-db", "a-link-to-it", "beside-it"])
def test_out_may_not_overwrite_or_sit_beside_an_input_database(tmp_path, where):
    home = Home(tmp_path, ONE, turn(1))
    before = home.lcm.read_bytes()
    out = {"the-lcm-db": home.lcm, "a-link-to-it": tmp_path / "r.json", "beside-it": home.dir / "r.json"}[where]
    if where == "a-link-to-it":
        out.symlink_to(home.lcm)
    with pytest.raises(SystemExit) as stop:
        audit.main(["--state-db", str(home.state), "--lcm-db", str(home.lcm), "--out", str(out)])
    assert stop.value.code == 2 and home.lcm.read_bytes() == before
    assert where == "a-link-to-it" or not (home.dir / "r.json").exists()


def test_out_is_written_whole_with_no_temporary_file_left(tmp_path):
    home = Home(tmp_path, ONE, turn(1))
    (tmp_path / "out").mkdir()
    assert audit.main(["--state-db", str(home.state), "--lcm-db", str(home.lcm), "--out", str(tmp_path / "out" / "r.json")]) == 0
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["r.json"]
    assert json.loads((tmp_path / "out" / "r.json").read_text())["instrument"] == "gate-kit-audit-v1"


@pytest.mark.parametrize("where", ["payload", "link", "resolved-payload", "home-subdir"])
def test_out_refuses_externalized_payloads_and_home_descendants(tmp_path, where):
    ref = "payload-c1.json"
    placeholder = (f"[Externalized tool output: tool=read_file; tool_call_id=c1; chars=10; bytes=10; "
                   f'read it with lcm_expand(externalized_ref="{ref}"); ref={ref}]')
    host = turn(1) + [("S1", call("c1")), ("S1", result("c1"))]
    home = Home(tmp_path, ONE, host, stored=host[:3] + [("S1", dict(result("c1"), content=placeholder))])
    payload_dir = home.dir / audit.externalized.PAYLOAD_DIRNAME
    payload_dir.mkdir()
    payload = payload_dir / ref
    payload.write_text(json.dumps({"content": result("c1")["content"]}))
    assert home.run()["gate"]["bars"]["externalized_unresolved"]["value"] == 0
    before = payload.read_bytes()
    out = payload
    if where == "link":
        out = tmp_path / "link.json"
        out.symlink_to(payload)
    elif where == "resolved-payload":
        out = tmp_path / ref
        payload.rename(out)
        payload.symlink_to(out)
    elif where == "home-subdir":
        out = home.dir / "nested" / "report.json"
        out.parent.mkdir()
    with pytest.raises(SystemExit) as stop:
        audit.main(["--state-db", str(home.state), "--lcm-db", str(home.lcm), "--out", str(out)])
    assert stop.value.code == 2 and payload.read_bytes() == before


def test_a_generated_unproven_row_stays_in_m3_when_it_matches_an_undone_row(tmp_path):
    out = undo_home(tmp_path, relations=[(3, "generated_unproven", None)])
    assert out["undo_retained"]["store_ids"] == [3, 4] and out["m3"]["store_ids"] == [3]


def test_unattributed_silent_rows_are_key_copies_and_distinct_rows_are_reported(tmp_path):
    speak_and_call = dict(call("c1"), content=f"{S} let me read it")
    host = [("S1", u("question 1")), ("S1", speak_and_call), ("S1", result("c1"))]
    ex = Home(tmp_path, ONE, host, stored=host + [("S1", speak_and_call)]).run()["exempted_surplus"]
    assert (ex["unattributed_silent_rows"], ex["unattributed_silent_distinct_rows"]) == (2, 1)
    assert ex["unattributed_silent_rows_unit"] == "key copies"


WINDOWED = {"P": ("cli", None, 100.0, "compression", 50.0), "Q": ("cli", "P", 3000.0, "user_exit", 100.0)}


@pytest.mark.parametrize("text", ["continue", "a new request"], ids=["parent-text", "control"])
def test_a_merged_carrier_repeating_an_out_of_window_parents_text_is_a_hand_check(tmp_path, text):
    # Astra: P (50-100) holds "continue"@90; its child Q holds a verified carrier plus "continue"@110; the store holds
    # P's original and an unmarked Q copy; with --since 100 parity passes, so the detector names both host rows.
    host = [("P", dict(u("continue"), _ts=90.0)),
            ("Q", {"role": "user", "content": CARRIER + "\n\n" + S + " " + text, "_ts": 110.0})]
    out = Home(tmp_path, WINDOWED, host, stored=[("P", u("continue")), ("Q", u(text))], nodes=[("Q", [1])]).run(
        since=100.0)
    windowed = [{"reason": "merged_carrier_window_repeat", "lineage": "P", "host_ids": [1, 2]}]
    assert out["gate"]["hand_checks"] == (windowed if text == "continue" else [])
    assert outcome(out) == ("HAND CHECK" if text == "continue" else "PASS")


@pytest.mark.parametrize("rows", [2, 1], ids=["two-carriers", "control"])
def test_merged_carriers_of_one_text_and_time_in_one_session_are_a_hand_check(tmp_path, rows):
    # Astra: two rows at one time, each a different verified carrier plus "continue", count as one occurrence.
    host = [("S1", {"role": "user", "content": CARRIER.replace("node 1", f"node {n}") + "\n\n" + S + " continue",
                    "_ts": T0 + 401}) for n in (1, 2)][:rows]
    out = Home(tmp_path, ONE, host, stored=[("S1", stamped(u("continue"), T0 + 401))],
               nodes=[("S1", [1]), ("S1", [2])]).run()
    same = [{"reason": "merged_carrier_same_stamp_multiplicity", "lineage": "S1", "host_ids": [1, 2]}]
    assert out["gate"]["hand_checks"] == (same if rows == 2 else []) and out["parity"]["totals"]["missing_rows"] == 0
    assert outcome(out) == ("HAND CHECK" if rows == 2 else "PASS")


@pytest.mark.parametrize("second", [T0, T0 + 5], ids=["both-at-T1", "control"])
def test_copies_stamped_off_the_hosts_times_are_a_stamp_swap(tmp_path, second):
    # Scout: the host holds "continue" at T1 and T2; the store holds it twice at T1. Parity matches by content.
    host = [("S1", dict(u("continue"), _ts=T0)), ("S1", dict(a("ok"), _ts=T0 + 1)), ("S1", dict(u("continue"), _ts=T0 + 5))]
    stored = [("S1", stamped(u("continue"), T0)), ("S1", stamped(a("ok"), T0 + 1)), ("S1", stamped(u("continue"), second))]
    out = Home(tmp_path, ONE, host, stored=stored).run()
    assert out["gate"]["hand_checks"] == ([{"reason": "stamp_swap", "session": "S1", "store_ids": [1, 3]}]
                                          if second == T0 else [])
    assert outcome(out) == ("HAND CHECK" if second == T0 else "PASS")


@pytest.mark.parametrize("lost", [True, False], ids=["undone-pair-twice", "control"])
def test_an_undone_pair_kept_twice_while_its_resend_is_lost_is_a_stamp_swap(tmp_path, lost):
    host = [("S1", dict(m, _ts=T0 + n)) for n, (_s, m) in enumerate(turn(1))] + [
        ("S1", undone(u("question 2"), T0 + 20)), ("S1", undone(a("answer 2"), T0 + 21)),
        ("S1", dict(u("question 2"), _ts=T0 + 30)), ("S1", dict(a("answer 2"), _ts=T0 + 31))]
    stored = [("S1", stamped(m, m["_ts"])) for _s, m in host[:4]] + (
        [("S1", stamped(m, m["_ts"])) for _s, m in host[2:4]] if lost else [("S1", stamped(m, m["_ts"])) for _s, m in host[4:]])
    out = Home(tmp_path, ONE, host, stored=stored).run()
    assert out["parity"]["totals"]["missing_rows"] == 0 and out["undo_retained"]["rows"] == 2
    swaps = [{"reason": "stamp_swap", "session": "S1", "store_ids": [3, 5]},
             {"reason": "stamp_swap", "session": "S1", "store_ids": [4, 6]}]
    assert out["gate"]["hand_checks"] == (swaps if lost else [])
    assert outcome(out) == ("HAND CHECK" if lost else "PASS")


def test_replay_links_are_checked_against_stored_content():
    # #2335 review: resolution can expand the older row only; the product linked the stored bytes.
    rows = [(1, "S", "assistant", "expanded", None, None, 1.0),
            (2, "S", "assistant", "<think>r</think>[stored]", None, None, 2.0)]
    stored = {1: ("S", "assistant", "[stored]"), 2: ("S", "assistant", "<think>r</think>[stored]")}
    relations = [(2, "possible_replay_of", 1)]
    numbers, silent = audit.score_lineage("S", [("assistant", "expanded")], {}, rows, relations, stored)
    assert (numbers["surplus_rows"], numbers["visible_surplus_rows"], numbers["silent_surplus_rows"]) == (1, 1, 0)
    assert silent == []
    # Checked against the resolved content instead, the same link explains nothing.
    resolved = {r[0]: (r[1], r[2], r[3]) for r in rows}
    numbers, _silent = audit.score_lineage("S", [("assistant", "expanded")], {}, rows, relations, resolved)
    assert (numbers["visible_surplus_rows"], numbers["silent_surplus_rows"]) == (0, 1)


def carrier_boundary_home(tmp_path, block, *, stored, history=None, summary=False):
    """This host pin's real in-place shape: no flagged summary row; the inserted block opens with LCM's pure summary
    carrier (a user row), then the host's copies of the retained tail. ``summary``: a host-flagged summary row instead."""
    history = history or [("S1", dict(m, _ts=3000.0 + i)) for i, (_s, m) in enumerate(turn(0) + turn(1))]
    archived = [(s, dict(m, _active=0, _compacted=1)) for s, m in history]
    opened = [("S1", dict(a("host summary"), _compressed_summary=1, _ts=3500.0) if summary else
               {"role": "user", "content": CARRIER, "_ts": 3500.0})]
    return Home(tmp_path, ONE, archived + opened + block, stored=stored, nodes=NODE)


@pytest.mark.parametrize("lose", [False, True], ids=["kept", "missing-original"])
def test_in_place_carrier_boundary_without_summary_row(tmp_path, lose):
    history = [("S1", dict(m, _ts=3000.0 + i)) for i, (_s, m) in enumerate(turn(0) + turn(1))]
    stored = history[:1] + history[2:] if lose else history
    totals = carrier_boundary_home(tmp_path, history[2:], stored=stored).run()["parity"]["totals"]
    assert totals["expected_rows"] == len(history)
    assert totals["missing_rows"] == int(lose)
    assert totals["surplus_rows"] == 0


def test_in_place_repeated_turns_keep_a_genuine_repeat(tmp_path):
    """Review: an unbounded longest-suffix match took a genuine new go/ok pair for copies. Stamps decide."""
    go = [("S1", dict(u("go"), _ts=3000.0)), ("S1", dict(a("ok"), _ts=3001.0)),
          ("S1", dict(u("go"), _ts=3002.0)), ("S1", dict(a("ok"), _ts=3003.0))]
    copies = go[2:]
    genuine = [("S1", dict(u("go"), _ts=3600.0)), ("S1", dict(a("ok"), _ts=3601.0))]
    totals = carrier_boundary_home(tmp_path, copies + genuine, stored=go, history=go).run()["parity"]["totals"]
    assert totals["missing_rows"] == 2


@pytest.mark.parametrize("summary", [False, True], ids=["carrier", "summary-row"])
@pytest.mark.parametrize("shape", ["unanswered-resend", "reused-call-id", "single-ok"])
def test_in_place_new_rows_with_their_own_stamps_are_never_copies(tmp_path, shape, summary):
    if shape == "unanswered-resend":
        history = turn(0) + [("S1", u("x"))]
        new = [("S1", dict(u("x"), _ts=3500.000001))]  # the resend lands just after the boundary's commit
    elif shape == "reused-call-id":
        history = turn(0) + [("S1", call("call_0")), ("S1", result("call_0"))]
        new = [("S1", dict(call("call_0"), _ts=3600.0)), ("S1", dict(result("call_0"), _ts=3601.0))]
    else:
        history = turn(0) + [("S1", a("ok"))]
        new = [("S1", {"role": "user", "content": CARRIER, "_ts": 3599.0}), ("S1", dict(a("ok"), _ts=3600.0))]
    history = [(s, dict(m, _ts=3000.0 + i)) for i, (s, m) in enumerate(history)]
    totals = carrier_boundary_home(tmp_path, new, stored=history, history=history,
                                   summary=summary).run()["parity"]["totals"]
    assert totals["missing_rows"] == sum(1 for _s, m in new if m["content"] != CARRIER)


@pytest.mark.parametrize("summary", [False, True], ids=["carrier", "summary-row"])
def test_in_place_restamped_copy_is_a_false_alarm_not_a_hidden_loss(tmp_path, summary):
    """A copy the host stamps anew is not provably a copy: it stays expected (missing), whatever its distance from
    the boundary."""
    history = [("S1", dict(m, _ts=3000.0 + i)) for i, (_s, m) in enumerate(turn(0) + turn(1))]
    restamped = [(s, dict(m, _ts=3500.000001 + i / 1e6)) for i, (s, m) in enumerate(history[2:])]
    totals = carrier_boundary_home(tmp_path, restamped, stored=history, history=history,
                                   summary=summary).run()["parity"]["totals"]
    assert totals["missing_rows"] == len(restamped)


def chained_in_place():
    """The r2 c85ae0 shape: three in-place compactions with no flagged summary row. The second copies the first
    block's copies, a prefix carrier and the new turn, with one copy dropped and one row moved; the third opens with a
    merged carrier and carries an empty carrier inside its block."""
    def at(batch, t0):
        return [(s, dict(m, _ts=t0 + i)) for i, (s, m) in enumerate(batch)]

    def archive(batch):
        return [(s, dict(m, _active=0, _compacted=1)) for s, m in batch]

    prefix = audit.plugin_tree.carrier_markers(REPO)[1][0] + S
    first_turns = at(turn(0) + turn(1) + [("S1", call("c1")), ("S1", result("c1")), ("S1", a("ok"))], 3000.0)
    block1 = [(s, dict(m)) for s, m in first_turns[2:]]
    new1 = at(turn(2) + [("S1", call("c2")), ("S1", result("c2")), ("S1", a("ok"))], 4000.0)
    pure = [("S1", {"role": "user", "content": CARRIER, "_ts": 3900.0})]
    pfx = [("S1", {"role": "user", "content": prefix, "_ts": 3900.5})]
    # Second compaction: drops one copy, moves the last, and carries the prefix carrier between.
    block2 = [(s, dict(m)) for s, m in block1[1:-1]] + pfx + [(s, dict(m)) for s, m in new1] + [block1[-1]]
    new2 = at(turn(3), 5000.0)
    merged = [("S1", {"role": "user", "content": CARRIER + "\n\n" + u("question 9")["content"], "_ts": 5900.0})]
    empty = [("S1", {"role": "user", "content": "", "_ts": 3900.5})]
    block3 = [(s, dict(m)) for s, m in new1[:2]] + empty + [(s, dict(m)) for s, m in new1[2:] + new2]
    host = (archive(first_turns) + archive(pure) + archive(block1) + archive(pfx) + archive(new1) + archive(pure)
            + archive(block2) + archive(new2) + merged + block3)
    genuine = first_turns + new1 + new2 + [("S1", dict(u("question 9"), _ts=5900.0))]
    return host, genuine


@pytest.mark.parametrize("lose", [False, True], ids=["kept", "missing-genuine"])
def test_in_place_chained_compactions_with_inner_carriers(tmp_path, lose):
    host, genuine = chained_in_place()
    stored = [r for r in genuine if not (lose and r[1].get("content") == u("question 3")["content"])]
    out = Home(tmp_path, ONE, host, stored=stored, nodes=NODE).run()
    totals = out["parity"]["totals"]
    assert totals["expected_rows"] == len(genuine)
    assert totals["missing_rows"] == int(lose)
    assert totals["surplus_rows"] == 0
