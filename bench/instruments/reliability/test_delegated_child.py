"""#1931: the delegated-child cell's scenario and scorer (scorers/delegated.py), over synthetic cell dirs, no Hermes."""
import json
import sqlite3
import types

import pytest

from bench.instruments.reliability import cells, probe
from bench.instruments.reliability import process_cell as PC
from bench.instruments.reliability.scorers import bars
from tests import test_reliability_scorers as fixture

GOAL, REPLY = probe.child_goal(2), probe.child_reply(2)
CHILD_EVENTS = [
    {"phase": "A", "turn": 2, "event": "child_user_sent", "tag": "D02", "session": "C1", "content": GOAL},
    {"phase": "A", "turn": 2, "event": "child_emit", "tag": "D02", "text": REPLY},
    {"phase": "A", "turn": 2, "event": "child_turn_end", "tag": "D02", "session": "C1", "held": GOAL, "failed": False}]


def chat(sid_of=lambda t: "S0"):
    """Parent turns 1-3 (turn 2 issues delegate_task) and their stored rows with session ids."""
    ev, rows = fixture.tool_turn(2, calls=[("delegate_task", {"goal": GOAL})], results=('{"status": "dispatched"}',))
    events = fixture.clean_events()
    events[4:4] = ev  # inside T02, after its user_sent
    tagged = fixture.clean_rows(1) + [fixture.clean_rows(2)[2]] + rows + fixture.clean_rows(3)[3:]
    sids = [sid_of(1)] * 2 + [sid_of(2)] * 4 + [sid_of(3)] * 2
    return events, list(zip(tagged, sids))


def make(tmp_path, *, child_rows=(("user", GOAL), ("assistant", REPLY)), child_sid="C1", sessions=None, link="S0",
         events=None, extra=(), sid_of=lambda t: "S0"):
    parent_events, parent = chat(sid_of)
    rows = parent + [(r, child_sid) for r in child_rows] + list(extra)
    d = tmp_path / "cell"
    (d / "db").mkdir(parents=True)
    con = sqlite3.connect(d / "db" / "state.db")
    con.execute("create table sessions (id text primary key, parent_session_id text, source text)")
    con.executemany("insert into sessions values (?,?,?)",
                    sessions or [("S0", None, "acp"), ("C1", "S0", "subagent")])
    con.commit()
    con.close()
    con = sqlite3.connect(d / "db" / "lcm.db")
    con.execute("create table messages (store_id integer primary key, session_id text, role text, content text,"
                " tool_calls text, tool_call_id text, ingested_at real)")
    con.execute("create table summary_nodes (node_id integer primary key, session_id text, depth integer,"
                " source_ids text, source_type text, created_at real)")
    for i, ((role, content, *rest), sid) in enumerate(rows, 1):
        con.execute("insert into messages values (?,?,?,?,?,?,?)", (i, sid, role, content, *(rest + [None, None])[:2], i * 10.0))
    if link:
        con.execute("create table session_lineage (child_session_id text primary key, parent_session_id text not null,"
                    " child_subagent_id text, parent_subagent_id text, child_role text, source text not null, created_at real)")
        con.execute("insert into session_lineage values ('C1', ?, 'sa-0', NULL, 'leaf', 'subagent_start', 1.0)", (link,))
    con.commit()
    con.close()
    comp = [{"phase": "A", "turn": i + 1, "event": "compaction", "compression_status": "compacted", "depth0_nodes": i + 1}
            for i in range(2)]
    allev = parent_events + (CHILD_EVENTS if events is None else events) + comp
    (d / "transcript.jsonl").write_text("".join(json.dumps(e) + "\n" for e in allev))
    (d / "phase-A.json").write_text(json.dumps({
        "phase": "A", "log_counts": {}, "counters": {"failed": [], "orphan_drops": 0, "native_max": 1},
        "compactions_logged": 2, "final_check": {"published": True}}))
    cell = {"id": "t", "tool_plan": [], "native_recovery": False, "min_compactions": 2, "final_compaction_check": True,
            "bars": [*cells.BARS, *cells.CHILD_BARS], "delegate_plan": [{"turns": [2]}]}
    return bars.score(cell, d)


def test_child_is_its_own_lineage_and_passes(tmp_path):
    out = make(tmp_path)
    assert out["verdict"] == "PASS", out.get("reason") or out["failed_bars"]
    assert "B10" in out["applicable_bars"]
    assert out["numbers"]["B2"]["per_session"] == {"C1": "PASS", "chat": "PASS"}
    assert out["numbers"]["B1"]["per_session"] == {"C1": 0, "chat": 0}
    b10 = out["numbers"]["B10"]
    assert (b10["scripted"], b10["issued"], b10["host_children"]) == (1, 1, 1)
    assert b10["children"]["C1"] == {"host_parent": "S0", "stored_rows": 2, "link": ["S0", "subagent_start"]}


@pytest.mark.parametrize("child_rows,b10", [((("user", GOAL),), False), ((), True)])
def test_a_missing_child_row_fails(tmp_path, child_rows, b10):
    out = make(tmp_path, child_rows=child_rows)
    assert out["verdict"] == "FAIL"
    assert {"B1", "B2"} <= set(out["failed_bars"])
    assert out["numbers"]["B2"]["per_session"]["C1"] == "FAIL" and out["numbers"]["B2"]["per_session"]["chat"] == "PASS"
    assert ("B10" in out["failed_bars"]) is b10
    if b10:
        assert out["failed_bars"]["B10"]["children_without_rows"] == ["C1"]


def test_a_child_stored_under_the_parent_session_fails(tmp_path):
    out = make(tmp_path, child_sid="S0")
    assert out["verdict"] == "FAIL"
    assert {"B1", "B2", "B10"} <= set(out["failed_bars"])
    assert out["numbers"]["B2"]["per_session"] == {"C1": "FAIL", "chat": "FAIL"}
    assert out["failed_bars"]["B10"]["children_without_rows"] == ["C1"]
    assert len(out["failed_bars"]["B10"]["child_rows_outside_child_lineages"]) == 2


def test_a_parent_row_copied_into_the_child_fails(tmp_path):
    out = make(tmp_path, extra=[(("user", fixture.U.format(1, 1)), "C1")])
    assert out["verdict"] == "FAIL" and "B10" in out["failed_bars"]
    assert out["failed_bars"]["B10"]["parent_rows_copied_into_children"] == [11]
    assert out["numbers"]["B10"]["parent_rows_copied_into_children"] == 1
    # The goal is the child's own user row, never a parent copy, even though the parent's tool call carries it.
    assert out["numbers"]["B10"]["child_rows_outside_child_lineages"] == 0


@pytest.mark.parametrize("link", [None, "S9"], ids=["missing", "wrong-parent"])
def test_a_missing_or_wrong_parent_link_fails(tmp_path, link):
    out = make(tmp_path, link=link)
    assert out["verdict"] == "FAIL" and out["failed_bars"]["B10"]["children_without_parent_link"] == ["C1"]
    assert set(out["failed_bars"]) == {"B10"}


def test_host_child_count_must_match_the_scripted_delegates(tmp_path):
    out = make(tmp_path, sessions=[("S0", None, "acp"), ("C1", "S0", "subagent"), ("C2", "S0", "subagent")])
    assert out["verdict"] == "FAIL"
    assert out["failed_bars"]["B10"]["child_count_mismatch"] == {"issued": 1, "host_children": 2}


def test_a_compression_continuation_still_joins_chat(tmp_path):
    """Turn 3 runs on S1, the parent's rotation child (same source): one chat lineage; the child stays apart."""
    out = make(tmp_path, sessions=[("S0", None, "acp"), ("S1", "S0", "acp"), ("C1", "S0", "subagent")],
               sid_of=lambda t: "S1" if t == 3 else "S0")
    assert out["verdict"] == "PASS", out["failed_bars"]
    assert set(out["numbers"]["B2"]["per_session"]) == {"C1", "chat"}
    group = bars.lineage(tmp_path / "cell", "S0")
    assert (group("S1"), group("C1"), group("S0")) == ("chat", "C1", "chat")


def test_a_child_compression_continuation_joins_the_child():
    sessions = {"S0": (None, "acp"), "C1": ("S0", "subagent"), "C1b": ("C1", "subagent")}
    from bench.instruments.reliability.scorers import delegated
    assert [delegated.is_child_root(s, sessions) for s in ("S0", "C1", "C1b")] == [False, True, False]


@pytest.mark.parametrize("drop", ["all", "turn_end", "interrupted"])
def test_a_child_that_never_ran_or_completed_is_unsupported(tmp_path, drop):
    # The observer's own shape for a stopped child: interrupted, with failed set as well.
    events = [] if drop == "all" else CHILD_EVENTS[:2] if drop == "turn_end" else \
        CHILD_EVENTS[:2] + [dict(CHILD_EVENTS[2], interrupted=True, failed=True)]
    out = make(tmp_path, events=events)
    assert out["verdict"] == "UNSUPPORTED" and "B10: D02" in out["reason"]


def test_a_child_that_started_and_failed_fails(tmp_path):
    out = make(tmp_path, events=CHILD_EVENTS[:2] + [dict(CHILD_EVENTS[2], failed=True)])
    assert out["verdict"] == "FAIL" and out["failed_bars"]["B10"]["children_failed"] == ["D02"]


def test_a_child_the_host_did_not_record_as_a_subagent_is_unsupported(tmp_path):
    out = make(tmp_path, sessions=[("S0", None, "acp"), ("C1", "S0", "acp")])
    assert out["verdict"] == "UNSUPPORTED" and "not a subagent child" in out["reason"]


def test_child_requests_are_never_parent_steps():
    c = cells.select("delegated-child/in-place")[0]
    events = []
    run = types.SimpleNamespace(cell=c, text_tag="T05", files="/f", crash_due=lambda t: False,
                                event=lambda **ev: events.append(ev))
    scenario = PC.Scenario(run)
    scenario.begin(5, "normal", "A", 1)
    first = scenario.main([{"role": "user", "content": probe.user_text(c, "T", 5)}])
    assert first["tool_calls"][0]["name"] == "delegate_task"
    assert json.loads(first["tool_calls"][0]["arguments"]) == {"goal": probe.child_goal(5)}
    child = scenario.main([{"role": "system", "content": "child prompt " + probe.child_goal(5)},
                           {"role": "user", "content": probe.child_goal(5)}])
    assert child["content"] == probe.child_reply(5) and scenario.st["step"] == 1 and not scenario.unexpected
    assert events[-1] == {"turn": 5, "event": "child_emit", "tag": "D05", "session_prefix": "D", "text": probe.child_reply(5)}
    reply = scenario.main([{"role": "user", "content": probe.user_text(c, "T", 5)},
                           {"role": "tool", "tool_call_id": "call_T05_0_0", "content": "{}"}])
    assert reply["content"] == PC.reply_text(c, "T", 5) and scenario.st["step"] == 2


def test_registry_cells_are_acp_process_only_and_declared():
    ids = [x["id"] for x in cells.select("delegated-child/*")]
    assert ids == ["delegated-child/in-place", "delegated-child/rotation"]
    assert cells.ISSUES[1931][0] == ("B1", "B2", "B10") and cells.ISSUE_TRANSPORTS[1931] == ("acp-process",)
    assert all("delegation" in x["toolsets"] and x["targets"] == [1931] for x in cells.select("delegated-child/*"))
    assert not any(x.get("delegate_plan") or "B10" in x["bars"] for x in cells.select("all")
                   if not x["id"].startswith("delegated-child/"))


def test_b6_applies_to_delegate_cells(tmp_path):
    """The delegate call and its result are a tool group: B6 scores it even with an empty tool_plan."""
    out = make(tmp_path)
    assert "B6" in out["applicable_bars"]
