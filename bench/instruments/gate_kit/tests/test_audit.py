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
    """``sessions``: {id: (source, parent, ended_at, end_reason)}; host and store rows: [(session, message)]."""

    def __init__(self, tmp_path, sessions, host, stored=None, *, relations=(), lineage=(), nodes=(), stamps=()):
        self.dir = tmp_path / "home"
        self.dir.mkdir()
        self.state, self.lcm = self.dir / "state.db", self.dir / "lcm.db"
        with sqlite3.connect(self.state) as con:
            con.execute("create table sessions (id text primary key, source text not null, parent_session_id text,"
                        " started_at real not null, ended_at real, end_reason text)")
            con.execute("create table messages (id integer primary key autoincrement, session_id text not null,"
                        " role text not null, content text, tool_call_id text, tool_calls text, timestamp real not null,"
                        " active integer not null default 1)")
            for i, (sid, (source, parent, ended, reason)) in enumerate(sessions.items()):
                con.execute("insert into sessions values (?,?,?,?,?,?)", (sid, source, parent, 1000.0 + i, ended, reason))
            for sid, m in host:
                con.execute("insert into messages (session_id, role, content, tool_call_id, tool_calls, timestamp)"
                            " values (?,?,?,?,?,?)", (sid, m["role"], m.get("content"), m.get("tool_call_id"),
                                                       json.dumps(m["tool_calls"]) if m.get("tool_calls") else None, 1.0))
        store = MessageStore(self.lcm)
        self.ids = [store.append(sid, dict(m, **({"timestamp": 1700000000.0 + n} if n in stamps else {})))
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
    assert out["parity"]["zero_row_lineages"] == ["A1"]  # the B child is reported, not gated
    assert out["parity"]["lineages"]["B1"]["gates"] is False
    assert out["parity"]["totals"]["missing_rows"] == 2


def test_lineage_row_counts_for_bucket_a_and_rotation_child_joins_its_parent(tmp_path):
    sessions = {"P": ("cli", None, 1000.5, "compression"), "P2": ("cli", "P", 3000.0, "user_exit"),
                "A1": ("subagent", "P2", 2500.0, "completed")}
    host = turn(1, "P") + turn(1, "P2") + turn(2, "P2") + turn(3, "A1")  # P2 re-issues turn 1
    stored = turn(1, "P") + turn(2, "P2") + turn(3, "A1")
    out = Home(tmp_path, sessions, host, stored=stored, lineage=[("A1", "P2")]).run()
    assert out["parity"]["sessions"]["P2"]["lineage"] == "P"
    assert out["parity"]["totals"]["missing_rows"] == out["parity"]["totals"]["surplus_rows"] == 0
    assert out["issue_1931"]["A"]["lineage_present"] == 1 and out["issue_1931"]["B"]["n"] == 0


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


def attribution(tmp_path, log, stored_extra, stamps=frozenset({0}), host=None):
    host = host or turn(1)
    return Home(tmp_path, ONE, host, stored=host + stored_extra, stamps=stamps).run(log)["exempted_surplus"]


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


@pytest.mark.parametrize("name,reason,limit", [("S3", "tail retry", "#2203"), ("cold", "append", "#2203"),
                                               ("session_end", "digest", "#2203"),
                                               ("session_end", "append", "unattributed"),
                                               ("warm_tail", "digest", "unattributed")])
def test_2203_routes(tmp_path, name, reason, limit):
    line = route("S1", name, 0, 3, 3).replace("reason=some reason text", f"reason={reason}")
    ex = attribution(tmp_path, [line], [("S1", u("question 1"))])
    assert ex["store_ids"][limit] == [3]


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


def test_unknown_log_tz_is_an_input_error(tmp_path):
    home = Home(tmp_path, ONE, turn(1))
    with pytest.raises(SystemExit) as stop:
        audit.main(["--state-db", str(home.state), "--lcm-db", str(home.lcm), "--log-tz", "Nowhere/Zone",
                    "--out", str(tmp_path / "r.json")])
    assert stop.value.code == 2


def floor_run(tmp_path, *, compactions=20):
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
                  for i in range(compactions)) + route("P", "foreground", 0, 1, 2)
    home = Home(tmp_path, sessions, host, lineage=lineage, nodes=[("P", [11])])
    return home.run([log], events)


def test_every_floor_met_gives_gate_pass_and_an_unmet_floor_gives_incomplete(tmp_path, monkeypatch):
    (tmp_path / "ok").mkdir()
    out = floor_run(tmp_path / "ok")
    assert {k: c["value"] for k, c in out["coverage"].items()} == {
        "turns": 196, "delegated_children": 5, "compactions": 20, "compactions_then_restart": 20, "restarts": 10,
        "restarts_cold": 5, "restarts_warm": 5, "undo_resend": 10, "tool_calls": 50, "tool_call_repeats": 10,
        "host_sources": 3}
    assert out["coverage_basis"]["store_depth0_message_leaves"] == 1
    failing = {k for k, c in out["coverage"].items() if not c["met"]}
    assert failing == {"turns", "delegated_children"}
    assert out["issue_1931"]["pass_A"] is True
    assert out["gate"]["verdict"] == "INCOMPLETE"
    monkeypatch.setitem(audit.FLOORS, "turns", 196)  # the same run with its two short floors lowered passes
    monkeypatch.setitem(audit.FLOORS, "delegated_children", 5)
    (tmp_path / "pass").mkdir()
    assert floor_run(tmp_path / "pass")["gate"]["verdict"] == "GATE PASS"


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
    out = tmp_path / "result.json"
    assert audit.main(["--state-db", str(home.state), "--lcm-db", str(home.lcm), "--out", str(out)]) == 0
    text, printed = out.read_text(), capsys.readouterr().out
    assert S not in text and S not in printed and "verdict: GATE FAIL" in printed
    assert json.loads(text)["parity"]["totals"]["missing_rows"] == 3  # the result and turn 2
    assert {p: p.read_bytes() for p in before} == before
