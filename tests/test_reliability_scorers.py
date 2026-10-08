"""Reliability harness R1 (bench/instruments/reliability): scorers, registry and host loader, no Hermes.

Every bar gets a PASS and a FAIL path over a tiny synthetic cell dir (transcript.jsonl, phase-A.json,
db/lcm.db) built here.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.instruments.reliability import ci, cells, controls, hosts, plugin_tree, probe, report, run_matrix  # noqa: E402
from bench.instruments.reliability.scorers import bars, dupes, multiset  # noqa: E402

U = "[T{:02d}] user turn {}: alpha beta end."
R = "reply to T{:02d}: noted item {}."


def turn_events(t, *, reply=True, held=None, tags=None):
    text = U.format(t, t)
    return [{"phase": "A", "turn": t, "event": "user_sent", "tag": f"T{t:02d}", "content": text, "persist": text},
            {"phase": "A", "turn": t, "event": "emit", "tag": f"T{t:02d}", "text": R.format(t, t)},
            {"phase": "A", "turn": t, "event": "turn_end", "tag": f"T{t:02d}", "held": held or text,
             "reply": R.format(t, t) if reply else None, "user_tags": tags or {f"T{t:02d}": 1}}]


PLAN = [("read_file", {"path": "p"}), ("lcm_expand", {"store_id": 1})]


def tool_turn(t, calls=PLAN, results=("r0", "r1"), **dispatch):
    """Probe events + stored rows for turn t, whose first step issues ``calls`` that the host ran for real."""
    ids = [f"call_T{t:02d}_0_{k}" for k in range(len(calls))]
    ev = [{"phase": "A", "turn": t, "event": "tool_issue", "tag": f"T{t:02d}", "id": i, "name": n, "args": a, "expect": {}}
          for i, (n, a) in zip(ids, calls)]
    ev += [{"phase": "A", "turn": t, "event": "tool_dispatch", "tag": f"T{t:02d}", "id": i, "name": n, "args": a, "ok": True,
            "chars": len(r), **dispatch} for i, (n, a), r in zip(ids, calls, results)]
    ev += [{"phase": "A", "turn": t, "event": "tool_seen", "id": i, "sha": hashlib.sha256(r.encode()).hexdigest()}
           for i, r in zip(ids, results)]
    call_row = ("assistant", "", json.dumps([{"id": i, "function": {"name": n, "arguments": json.dumps(a)}}
                                             for i, (n, a) in zip(ids, calls)]))
    return ev, [call_row] + [("tool", r, None, i) for i, r in zip(ids, results)]


def tool_cell(**dispatch):
    ev, rows = tool_turn(1, **dispatch)
    events = clean_events()
    events[1:1] = ev  # between T01's user_sent and its emit/turn_end
    return events, clean_rows(1) + rows + clean_rows(3)[2:], {"tool_plan": [{"turns": [1], "calls": [
        {"name": n, "args": a} for n, a in PLAN]}]}


def make(tmp_path, *, rows, events, nodes=(), phase=None, sids=None, parents=None, host=None, **cell_kw):
    d = tmp_path / "cell"
    (d / "db").mkdir(parents=True)
    if parents is not None or host is not None:  # host: state.db (session, role, content, active) rows
        con = sqlite3.connect(d / "db" / "state.db")
        con.execute("create table sessions (id text primary key, parent_session_id text)")
        con.executemany("insert into sessions values (?,?)", (parents or {"S0": None}).items())
        if host is not None:
            con.execute("create table messages (id integer primary key, session_id text, role text, content text, active integer)")
            con.executemany("insert into messages (session_id, role, content, active) values (?,?,?,?)", host)
        con.commit()
        con.close()
    con = sqlite3.connect(d / "db" / "lcm.db")
    con.execute("create table messages (store_id integer primary key, session_id text, role text, content text,"
                " tool_calls text, tool_call_id text, ingested_at real)")
    con.execute("create table summary_nodes (node_id integer primary key, session_id text, depth integer,"
                " source_ids text, source_type text, created_at real)")
    for i, (role, content, *rest) in enumerate(rows, 1):
        sid = sids[i - 1] if sids else "S0"
        con.execute("insert into messages values (?,?,?,?,?,?,?)", (i, sid, role, content, *(rest + [None, None])[:2], i * 10.0))
    for i, ids in enumerate(nodes, 1):
        con.execute("insert into summary_nodes values (?,?,?,?,?,?)", (i, "S0", 0, json.dumps(ids), "messages", 1.0))
    con.commit()
    con.close()
    comp = [{"phase": "A", "turn": i + 1, "event": "compaction", "compression_status": "compacted", "depth0_nodes": i + 1}
            for i in range(cell_kw.pop("passes", 2))] + cell_kw.pop("extra_events", [])
    (d / "transcript.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events + comp))
    base = {"phase": "A", "log_counts": {}, "counters": {"failed": [], "orphan_drops": 0, "native_max": 1},
            "compactions_logged": sum(e.get("compression_status") == "compacted" for e in comp), "final_check": {"published": True}}
    (d / "phase-A.json").write_text(json.dumps({**base, **(phase or {})}))
    cell = {"id": "t", "tool_plan": [], "native_recovery": False, "min_compactions": 2, "final_compaction_check": True,
            "bars": list(cells.BARS), **cell_kw}
    return bars.score(cell, d)


def clean_rows(n=3):
    return [r for t in range(1, n + 1) for r in (("user", U.format(t, t)), ("assistant", R.format(t, t)))]


def clean_events(n=3):
    return [e for t in range(1, n + 1) for e in turn_events(t)]


def test_clean_cell_passes_every_bar(tmp_path):
    out = make(tmp_path, rows=clean_rows(), events=clean_events())
    assert out["verdict"] == "PASS", out["failed_bars"]
    assert out["applicable_bars"] == ["B1", "B2", "B3", "B4", "B5", "B8"]


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("kind", ["placeholder", "note"])
def test_recovery_host_echo_never_licenses_stored_surplus(tmp_path, composite, kind):
    tree = Path(__file__).resolve().parents[1]
    prefixes = plugin_tree.recovery_prefixes(tree)
    assert len(prefixes) == 2 and set(prefixes) <= set(plugin_tree.carrier_markers(tree)[1])
    text = next(p for p in prefixes if ("latest message" in p) == (kind == "note"))
    text += "123 tokens) is stored." if kind == "note" else ""
    if composite:
        text = U.format(1, 1) + "\n\n" + text
    out = make(tmp_path, rows=clean_rows() + [("user", text)], events=clean_events(),
               host=[("S0", "user", text, 1)], plugin={"tree": str(tree)})
    assert out["verdict"] == "FAIL" and "B2" in out["failed_bars"]
    assert out["numbers"]["B2"]["stored_rows_not_expected"] == 1
    assert out["numbers"]["B2"]["host_parity_licensed"]["rows"] == 0
    if composite:
        assert "B1" in out["failed_bars"]


def test_recovery_exclusion_preserves_embedded_objective_licence(tmp_path):
    tree = Path(__file__).resolve().parents[1]
    text = "ordinary user text\n\n[Current user objective preserved from compacted history] quoted"
    out = make(tmp_path, rows=clean_rows() + [("user", text)], events=clean_events(),
               host=[("S0", "user", text, 1)], plugin={"tree": str(tree)})
    assert out["verdict"] == "PASS" and out["numbers"]["B2"]["host_parity_licensed"]["rows"] == 1


@pytest.mark.parametrize("marked,prior,verdict", [(False, True, "UNSUPPORTED"), (True, False, "UNSUPPORTED"),
                                                 (True, True, "PASS")])
def test_forced_recovery_requires_marker_and_prior_compaction(tmp_path, marked, prior, verdict):
    recovery = {"event": "compaction", "turn": 2, "compression_status": "overflow_recovery",
                "recovery_marker": marked, "prior_compaction": prior}
    out = make(tmp_path, rows=clean_rows(), events=clean_events(), bars=["B1", "B2"],
               faults=[{"kind": "forced_recovery", "turn": 2}], extra_events=[recovery])
    assert out["verdict"] == verdict
    if verdict == "UNSUPPORTED":
        assert "recovery" in out["reason"]


def test_overflow_cell_restart_and_bars_contract():
    for c in cells.select("overflow-recovery-restart/*"):
        assert c["bars"] == ["B1", "B2"] and c["targets"] == [534]
        assert c["faults"][1] == {"kind": "crash_after_compaction_before_reply", "after_status": "overflow_recovery", "offset": 1}
        assert c["faults"][2] == {"kind": "clean_exit_before_turn", "after_restart": 1}
        assert set(c["lcm_env"]) == set(cells.tight(c["window"]))
    assert cells.ISSUES[534] == (("B1", "B2"), "")


@pytest.mark.parametrize("stored_result,verdict", [("real host result", "PASS"), ("wrong result", "FAIL")])
def test_forced_recovery_scores_durable_result_against_dispatch(tmp_path, stored_result, verdict):
    ev, rows = tool_turn(1, calls=PLAN[:1], results=[stored_result],
                         recovery_result_sha256=hashlib.sha256(b"real host result").hexdigest())
    ev = [e for e in ev if e["event"] != "tool_seen"]
    events = clean_events()
    events[1:1] = ev
    recovery = {"event": "compaction", "turn": 1, "compression_status": "overflow_recovery",
                "recovery_marker": True, "prior_compaction": True}
    out = make(tmp_path, rows=clean_rows(1) + rows + clean_rows(3)[2:], events=events,
               faults=[{"kind": "forced_recovery", "turn": 1}], extra_events=[recovery], bars=["B1", "B2"],
               tool_plan=[{"turns": [1], "calls": [{"name": n, "args": a} for n, a in PLAN[:1]]}])
    assert out["verdict"] == verdict
    assert out["numbers"]["B6"]["tool_execution_failures"] == 1  # result never reached the provider


@pytest.mark.parametrize("dispatch_ok,ending,reply,gap", [
    (False, "complete", True, "recovery tool dispatch failed"),
    (True, "failed", True, "forced recovery turn did not complete"),
    (True, "missing", True, "forced recovery turn did not complete"),
    (True, "cancel", True, "forced recovery turn did not complete"),
    (True, "complete", False, "scripted reply"),
    (True, "complete", True, None),
])
def test_forced_recovery_requires_successful_dispatch_and_completed_reply(dispatch_ok, ending, reply, gap):
    events = turn_events(32)
    ev, _ = tool_turn(32, calls=[("write_file", {"path": "p", "content": "result"})], results=["result"],
                      ok=dispatch_ok, recovery_result_sha256=hashlib.sha256(b"result").hexdigest())
    events[1:1] = [e for e in ev if e["event"] != "tool_seen"]
    if ending == "failed":
        events[-1]["failed"] = True
    elif ending == "cancel":
        events[-1]["kind"] = "cancel"
    elif ending == "missing":
        events.pop()
    if not reply:
        events = [e for e in events if e["event"] != "emit"]
    events.append({"event": "compaction", "turn": 32, "compression_status": "overflow_recovery",
                   "recovery_marker": True, "prior_compaction": True})
    atts = bars.attempts(events)
    bound = bars.tool_calls.bind(atts)
    cell = {"bars": ["B1", "B2"], "faults": [{"kind": "forced_recovery", "turn": 32}],
            "tool_plan": [{"turns": [32], "calls": [{"name": "write_file"}]}]}
    gaps = bars.scenario_gaps(cell, events, atts, {"groups": 0}, bound)
    assert any(gap in g for g in gaps) if gap else not gaps


def test_b1_b2_duplicate_user_row_fails(tmp_path):
    out = make(tmp_path, rows=clean_rows() + [("user", U.format(2, 2) + " ")], events=clean_events())
    assert out["failed_bars"]["B1"] == {"T02": {"expected": 1, "stored": 2}}
    assert out["numbers"]["B2"]["surplus_rows"] == 1


def test_b2_catches_identical_reply_surplus_that_tags_cannot(tmp_path):
    events = clean_events()
    for e in events:
        if e["event"] == "emit":
            e["text"] = "same reply."
    rows = [r if r[0] == "user" else ("assistant", "same reply.") for r in clean_rows()] + [("assistant", "same reply.")]
    out = make(tmp_path, rows=rows, events=events)
    assert set(out["failed_bars"]) == {"B2"}
    assert out["numbers"]["B2"]["duplicated_keys"] == 1 and out["numbers"]["B2"]["surplus_rows"] == 1


def test_b2_loss_and_crash_merge_composite(tmp_path):
    lost = make(tmp_path, rows=clean_rows()[:-1], events=clean_events())
    assert lost["numbers"]["B2"]["deficit_rows"] == 1 and "B1" in lost["failed_bars"]
    # Turn 2 crashed before its reply; turn 3's prompt merged into the dangling row (host composite).
    composite = U.format(2, 2) + "\n\n" + U.format(3, 3)
    events = turn_events(1) + turn_events(2)[:1] + turn_events(3, held=composite, tags={"T02": 1, "T03": 1})
    rows = clean_rows(1) + [("user", composite), ("assistant", R.format(3, 3))]
    assert make(tmp_path / "m", rows=rows, events=events)["verdict"] == "PASS"


def test_b3_b4(tmp_path):
    out = make(tmp_path, rows=clean_rows(), events=clean_events(),
               phase={"log_counts": {"publication_invariant_conflict": 2}, "counters": {"failed": ["T03"]}})
    assert out["failed_bars"]["B3"] == {"publication_invariant_conflict": 2}
    assert out["failed_bars"]["B4"]["failed_turns"] == ["T03"]
    unpublished = make(tmp_path / "f", rows=clean_rows(), events=clean_events(), phase={"final_check": {"published": False}})
    assert set(unpublished["failed_bars"]) == {"B4"}
    cleanup_only = {"outcome": "inconclusive", "engine_status": "sanitized", "attempts": [{}, {}], "entry": "x"}
    unsure = make(tmp_path / "i", rows=clean_rows(), events=clean_events(), phase={"final_check": cleanup_only})
    assert unsure["verdict"] == "INCONCLUSIVE" and unsure["failed_bars"] == {} and set(unsure["inconclusive_bars"]) == {"B4"}


def test_b8_survival_fit_is_a_normal_cell_failure_and_not_scored_on_injected_failure_cells(tmp_path):
    """#582: B8 counts `LCM survival fit applied` like B3; a cell that lists its own bars (the
    publication-failure cells) does not score it."""
    fit = {"log_counts": {"survival_fit": 1}}
    out = make(tmp_path, rows=clean_rows(), events=clean_events(), phase=fit)
    assert out["verdict"] == "FAIL" and out["failed_bars"] == {"B8": {"survival_fit": 1}}
    assert report.signature(out) == "B8 survival_fits=1"
    injected = make(tmp_path / "i", rows=clean_rows(), events=clean_events(), phase=fit, bars=["B1", "B2", "B4"])
    assert injected["verdict"] == "PASS" and "B8" not in injected["applicable_bars"]
    clean = make(tmp_path / "c", rows=clean_rows(), events=clean_events())
    assert clean["verdict"] == "PASS" and clean["numbers"]["B8"] == {"survival_fit": 0}


def test_native_on_off_counts_every_candidate_event_and_reports_the_pre_publication_split():
    """native-on-off (R4 addendum d): the older ref's phase is not counted; EVERY candidate-phase conflict
    or fit is, before its first publication too; the pre-publication share is a diagnostic only."""
    old = {"exit": "plugin_switch", "log_counts": {"publication_invariant_conflict": 5}}
    before = {"exit": "crash", "log_counts": {"publication_invariant_conflict": 2}, "log_counts_after_commit": None}
    first = {"exit": "done", "log_counts": {"publication_invariant_conflict": 3, "survival_fit": 2},
             "log_counts_after_commit": {"publication_invariant_conflict": 1, "survival_fit": 0}}
    cell = {"from_ref": "v0.24.3"}
    assert bars.counted(cell, [old, before, first], "publication_invariant_conflict") == 5
    assert bars.counted(cell, [old, before, first, first], "survival_fit") == 4
    assert bars.counted({}, [old, before, first], "publication_invariant_conflict") == 10
    assert bars.pre_publication(cell, [old, before, first], "publication_invariant_conflict") == 4
    assert bars.pre_publication(cell, [old, before, first], "survival_fit") == 2
    assert bars.pre_publication(cell, [old, before, before], "publication_invariant_conflict") == 4  # never published
    stale = {"exit": "done", "log_counts": {"publication_invariant_conflict": 3}}
    assert bars.counted(cell, [old, stale], "publication_invariant_conflict") == 3  # counted without the diagnostic
    assert bars.pre_publication(cell, [old, stale], "publication_invariant_conflict") is None
    with pytest.raises(ValueError, match="no candidate phase"):
        bars.counted(cell, [dict(old, exit="done")], "survival_fit")  # no candidate phase at all: never 0
    assert all(c["faults"] == [{"kind": "plugin_switch", "turn": 31}] and c["native_recovery"]
               for c in cells.select("native-on-off/*"))


def test_r3_probe_phase_json_carries_post_publication_counts_so_b3_b8_fail(tmp_path):
    """R3 F4: the probe computes the post-publication counts BEFORE it writes the phase JSON, so a
    candidate phase with 3 conflicts and 5 fits after its first publication scores B3/B8 FAIL
    (it used to write them after, and bars saw no publication: B3 = B8 = 0, a false green)."""
    import inspect
    log = ("LCM compaction #1: ...\n" + "reason=publication_invariant_conflict\n" * 3
           + "LCM survival fit applied (reason=x)\n" * 5)
    fields = probe.phase_log_fields(log)
    assert fields["log_counts_after_commit"] == {"publication_invariant_conflict": 3, "survival_fit": 5}
    source = inspect.getsource(probe.main)
    assert source.index("phase_log_fields(") < source.index('phase-{phase}.json')
    cell = {"from_ref": "v0.24.3", "bars": ["B3", "B4", "B8"]}
    make(tmp_path, rows=clean_rows(), events=clean_events(), phase={"exit": "plugin_switch"}, bars=cell["bars"])
    (tmp_path / "cell" / "phase-B.json").write_text(json.dumps(
        {"phase": "B", "exit": "done", "counters": {"failed": []}, "final_check": {"published": True}, **fields}))
    out = bars.score({"id": "t", "tool_plan": [], "native_recovery": False, "min_compactions": 2,
                      "final_compaction_check": True, **cell}, tmp_path / "cell")
    assert out["verdict"] == "FAIL" and out["failed_bars"] == {
        "B3": {"publication_invariant_conflict": 3}, "B8": {"survival_fit": 5}}, out["failed_bars"]
    assert out["numbers"]["pre_publication_counts"] == {"publication_invariant_conflict": 0, "survival_fit": 0}


def test_r4_native_on_off_pre_publication_conflicts_and_fits_fail_b3_b8(tmp_path):
    """R4 addendum (d): a candidate phase whose 12 conflicts and 1 fit all precede its first publication
    fails B3 and B8 (they used to be forgiven: a false green); the split stays visible as a diagnostic."""
    log = ("reason=publication_invariant_conflict\n" * 12 + "LCM survival fit applied (reason=x)\n"
           + "LCM compaction #1: ...\n")
    fields = probe.phase_log_fields(log)
    assert fields["log_counts_after_commit"] == {"publication_invariant_conflict": 0, "survival_fit": 0}
    cell = {"from_ref": "v0.23.3", "bars": ["B3", "B4", "B8"]}
    make(tmp_path, rows=clean_rows(), events=clean_events(), phase={"exit": "plugin_switch"}, bars=cell["bars"])
    (tmp_path / "cell" / "phase-B.json").write_text(json.dumps(
        {"phase": "B", "exit": "done", "counters": {"failed": []}, "final_check": {"published": True}, **fields}))
    out = bars.score({"id": "t", "tool_plan": [], "native_recovery": False, "min_compactions": 2,
                      "final_compaction_check": True, **cell}, tmp_path / "cell")
    assert out["verdict"] == "FAIL" and out["failed_bars"] == {
        "B3": {"publication_invariant_conflict": 12}, "B8": {"survival_fit": 1}}, out["failed_bars"]
    assert out["numbers"]["pre_publication_counts"] == {"publication_invariant_conflict": 12, "survival_fit": 1}


def test_compact_transcript_fields_mean_held_equals_persist(tmp_path):
    events = clean_events()
    for e in events:
        if e["event"] == "turn_end":
            del e["held"]
            e["held_same"] = True
        else:
            e["persist"] = None
    assert make(tmp_path, rows=clean_rows(), events=events)["verdict"] == "PASS"


def test_b5_growth_minimum_and_log_parity(tmp_path):
    assert "B5" in make(tmp_path, rows=clean_rows(), events=clean_events(), passes=1)["failed_bars"]
    stall = clean_events() + [{"phase": "A", "turn": 3, "event": "compaction", "compression_status": "compacted",
                               "depth0_nodes": 1}]
    out = make(tmp_path / "s", rows=clean_rows(), events=stall, phase={"compactions_logged": 3})
    assert out["failed_bars"]["B5"]["non_growing_passes"] == [1]  # depth-0 sequence 1, 1, 2
    assert "B5" in make(tmp_path / "l", rows=clean_rows(), events=clean_events(), phase={"compactions_logged": 5})["failed_bars"]


def test_b6_split_group_and_orphans(tmp_path):
    calls = json.dumps([{"id": "c1"}, {"id": "c2"}])
    rows = clean_rows(1) + [("assistant", "", calls), ("tool", "a", None, "c1"), ("tool", "b", None, "c2")] + clean_rows(3)[2:]
    events, rows, kw = tool_cell()
    whole = make(tmp_path, rows=rows, events=events, nodes=[[1, 2, 3, 4, 5]], **kw)
    assert whole["verdict"] == "PASS", whole["failed_bars"]
    split = make(tmp_path / "s", rows=rows, events=events, nodes=[[1, 2, 3, 4]], **kw)
    assert split["failed_bars"]["B6"]["split_groups"] == 1
    orphan = make(tmp_path / "o", rows=rows, events=events, phase={"counters": {"failed": [], "orphan_drops": 1}}, **kw)
    assert orphan["failed_bars"]["B6"]["host_orphan_drops"] == 1


def native_events(n=3):
    events = clean_events(n)
    for e in events:
        if e["event"] == "turn_end":
            e["native_attempts"] = 1
    return events


def test_b7_native_health(tmp_path):
    ok = make(tmp_path, rows=clean_rows(), events=native_events(), native_recovery=True)
    assert ok["verdict"] == "PASS" and "B7" in ok["applicable_bars"]
    bad = make(tmp_path / "b", rows=clean_rows(), events=native_events(), native_recovery=True,
               phase={"log_counts": {"summary_generation_aborted": 1}, "counters": {"failed": [], "native_max": 2}})
    assert bad["failed_bars"]["B7"]["summary_generation_aborted"] == 1
    assert bad["failed_bars"]["B7"]["max_native_attempts_per_turn"] == 2


def test_multiset_normalisation_passes_and_a_split_reply_fails():  # R1.2 F1: a split is surplus, never PASS
    assert multiset.score([("user", "a  b\n")], [(1, "S", "user", " a  b")])["verdict"] == "PASS"  # edge strip only
    out = multiset.score([("user", "a  b\n"), ("assistant", "one two")],
                         [(1, "S", "user", "a  b"), (2, "S", "assistant", "one "), (3, "S", "assistant", "two")])
    assert out["verdict"] == "FAIL" and out["split_keys"] == 1 and out["surplus_rows"] == 2


def test_f1_stored_only_key_fails_b2(tmp_path):
    out = make(tmp_path, rows=clean_rows() + [("user", "synthetic row nobody sent")], events=clean_events())
    assert out["numbers"]["B2"]["stored_rows_not_expected"] == 1 and out["numbers"]["B2"]["surplus_rows"] == 1
    assert "B2" in out["failed_bars"] and "B1" not in out["failed_bars"]


def test_f2_native_pass_needs_its_own_host_commit(tmp_path):
    native = [{"phase": "A", "turn": 3, "event": "compaction", "compression_status": "host_native"}]
    final = [{"phase": "A", "turn": 3, "event": "compaction", "compression_status": "host_native", "final": True}]
    unproven = make(tmp_path, rows=clean_rows(), events=clean_events(), extra_events=native + final)
    assert unproven["failed_bars"]["B5"]["native_passes_without_host_commit"] == ["A:T3"]
    events = clean_events()
    events[-1]["host_commits"] = 1  # turn 3's turn_end saw one host "committed" telemetry line
    proven = make(tmp_path / "p", rows=clean_rows(), events=events, extra_events=native + final, passes=1, min_compactions=2)
    assert proven["verdict"] == "PASS", proven["failed_bars"]
    assert proven["numbers"]["B5"]["published"] == 2  # the final forced pass is B4 evidence, not counted here
    crash = [{"phase": "A", "turn": 3, "event": "crash", "fault": "crash_after_rotation_before_child_row", "host_commits": 0}]
    killed = make(tmp_path / "k", rows=clean_rows(), events=clean_events(), extra_events=native + crash)
    assert killed["verdict"] == "PASS" and killed["numbers"]["B5"]["native_passes_interrupted_by_crash"] == ["A:T3"]
    assert killed["numbers"]["B5"]["published"] == 2  # the interrupted native pass is not counted
    final_lcm = [{"phase": "A", "turn": 3, "event": "compaction", "compression_status": "compacted", "final": True, "depth0_nodes": 9}]
    logged = make(tmp_path / "l", rows=clean_rows(), events=clean_events(), extra_events=final_lcm, phase={"compactions_logged": 3})
    assert logged["verdict"] == "PASS" and logged["numbers"]["B5"]["published"] == 2  # final LCM pass logs, is not counted


def test_f3_unproven_scenarios_are_unsupported(tmp_path):
    events, rows, plan = tool_cell()
    undispatched = [e for e in events if not (e["event"] == "tool_dispatch" and e["name"] == "lcm_expand")]
    short = make(tmp_path, rows=rows, events=undispatched, nodes=[[1, 2, 3, 4, 5]], **plan)
    assert short["verdict"] == "UNSUPPORTED" and "never dispatched" in short["reason"]
    no_emit = make(tmp_path / "e", rows=clean_rows(), events=[e for e in clean_events() if e["event"] != "emit" or e["tag"] != "T02"])
    assert no_emit["verdict"] == "UNSUPPORTED" and "never returned its scripted reply" in no_emit["reason"]
    no_groups = make(tmp_path / "g", rows=clean_rows(), events=clean_events(), **plan)
    assert no_groups["verdict"] == "UNSUPPORTED" and "no tool-call group" in no_groups["reason"]
    no_native = make(tmp_path / "n", rows=clean_rows(), events=clean_events(), native_recovery=True)
    assert no_native["verdict"] == "UNSUPPORTED" and "zero native" in no_native["reason"]


NOTICE = "Your request was not processed. Send it again if you still want me to carry it out."


def test_r13_only_the_cited_notice_on_an_interrupted_attempt_is_held(tmp_path):
    rows = clean_rows() + [("assistant", NOTICE)]
    cited = {"failed_turn_notices": [NOTICE]}
    assert "B2" in make(tmp_path, rows=rows, events=clean_events(), phase=cited)["failed_bars"]
    events = clean_events()
    events[-1].update(host_notices=[NOTICE], interrupted=True)
    assert make(tmp_path / "h", rows=rows, events=events, phase=cited)["verdict"] == "PASS"
    events[-1]["interrupted"] = False  # the same row on an attempt the host did not interrupt
    assert "B2" in make(tmp_path / "n", rows=rows, events=events, phase=cited)["failed_bars"]
    other = rows[:-1] + [("assistant", "an arbitrary replayed assistant row")]
    events[-1].update(host_notices=["an arbitrary replayed assistant row"], interrupted=True)
    assert "B2" in make(tmp_path / "a", rows=other, events=events, phase=cited)["failed_bars"]


def test_r13_a_reply_the_host_dropped_is_a_deficit(tmp_path):
    events = clean_events()
    for e in events:
        if e["event"] == "turn_end":
            e["reply"] = None  # the host result no longer holds the reply the provider returned
    out = make(tmp_path, rows=[r for r in clean_rows() if r[0] == "user"], events=events)
    assert out["numbers"]["B2"]["deficit_rows"] == 3 and {"B1", "B2"} <= set(out["failed_bars"])


def test_r13_tools_bind_to_real_dispatch_and_durable_rows(tmp_path):
    events, rows, plan = tool_cell()
    assert make(tmp_path, rows=rows, events=events, nodes=[[1, 2, 3, 4, 5]], **plan)["verdict"] == "PASS"
    wrong, _r, _p = tool_cell(name="write_file")
    assert "planned read_file but the host ran write_file" in make(tmp_path / "w", rows=rows, events=wrong,
                                                                     nodes=[[1, 2, 3, 4, 5]], **plan)["failed_bars"]["B6"]["tool_failures"][0]
    bad, _r, _p = tool_cell(ok=False, detail="error: boom")
    assert "B6" in make(tmp_path / "f", rows=rows, events=bad, nodes=[[1, 2, 3, 4, 5]], **plan)["failed_bars"]
    lost = make(tmp_path / "l", rows=[r for r in rows if r[0] != "tool" or r[1] != "r1"], events=events, **plan)
    assert lost["numbers"]["B2"]["tool_missing_rows"] == 1 and "B2" in lost["failed_bars"]
    extra = list(events)
    extra.insert(5, dict(events[3], id=None, name="todo", args={}))  # an unplanned real dispatch in T01
    assert "unplanned host dispatch" in str(make(tmp_path / "u", rows=rows, events=extra, nodes=[[1, 2, 3, 4, 5]], **plan)["failed_bars"])


def test_f3_continue_rows_are_position_bound(tmp_path):
    events = clean_events(3)
    events[3]["content"] = events[3]["persist"] = events[5]["held"] = "continue"
    rows = [("user", U.format(1, 1)), ("assistant", R.format(1, 1)), ("user", "continue"), ("assistant", R.format(2, 2)),
            ("user", U.format(3, 3)), ("assistant", R.format(3, 3))]
    assert make(tmp_path, rows=rows, events=events)["verdict"] == "PASS"
    moved = [rows[0], rows[1], rows[3], rows[2], rows[4], rows[5]]  # same multiset, continue after the wrong reply
    out = make(tmp_path / "m", rows=moved, events=events)
    assert "continue" in out["failed_bars"]["B1"] and "B2" not in out["failed_bars"]


def test_f4_rows_are_scored_per_session_lineage(tmp_path):
    k = "[K01] user turn 1: alpha beta end."
    events = clean_events(2) + [
        {"phase": "A", "turn": 1, "event": "user_sent", "tag": "K01", "session_prefix": "K", "content": k, "persist": k},
        {"phase": "A", "turn": 1, "event": "turn_end", "tag": "K01", "session_prefix": "K", "held": k,
         "reply": "reply to K01: noted item 1.", "user_tags": {"K01": 1}, "session": "cron_job_01"}]
    events.insert(-1, {"phase": "A", "turn": 1, "event": "emit", "tag": "K01", "text": "reply to K01: noted item 1."})
    rows = clean_rows(2) + [("user", k), ("assistant", "reply to K01: noted item 1.")]
    right = ["S0", "S0", "child", "child", "cron_job_01", "cron_job_01"]  # turn 2 after a rotation to "child"
    ok = make(tmp_path, rows=rows, events=events, sids=right, parents={"S0": None, "child": "S0", "cron_job_01": None})
    assert ok["verdict"] == "PASS", ok["failed_bars"]
    swapped = ["S0", "S0", "cron_job_01", "cron_job_01", "child", "child"]  # same global multiset, lineages swapped
    out = make(tmp_path / "s", rows=rows, events=events, sids=swapped, parents={"S0": None, "child": "S0", "cron_job_01": None})
    assert {"B1", "B2"} <= set(out["failed_bars"])
    assert out["numbers"]["B2"]["per_session"] == {"chat": "FAIL", "cron_job_01": "FAIL"}


def test_dupes_counts_a_multi_origin_burst_after_compaction(tmp_path):
    db = tmp_path / "d.db"
    con = sqlite3.connect(db)
    con.execute("create table messages (store_id integer primary key, session_id text, role text, content text,"
                " tool_calls text, tool_call_id text, ingested_at real)")
    con.execute("create table summary_nodes (created_at real)")
    rows = [("user", "a", 1.0), ("assistant", "b", 5.0), ("user", "a", 20.0), ("assistant", "b", 20.1)]
    for i, (role, content, ts) in enumerate(rows, 1):
        con.execute("insert into messages values (?,?,?,?,?,?,?)", (i, "S", role, content, None, None, ts))
    con.execute("insert into summary_nodes values (10.0)")
    con.commit()
    con.close()
    assert dupes.count(db)["replayed_rows_after_compaction"] == 2


def test_registry_is_valid_and_selectable():
    everything = cells.select("all")
    assert len({c["id"] for c in everything}) == len(everything)
    assert [c["id"] for c in cells.select("baseline/*")] == ["baseline/in-place/acp", "baseline/rotation/acp"]
    json.dumps(everything)
    with pytest.raises(ValueError):
        cells.select("no-such-cell/*")
    targeted = {t for c in everything for t in c["targets"]}
    assert all(issue in targeted or cap for issue, (_b, cap) in cells.ISSUES.items())


def test_hosts_loader_refuses_the_live_hermes_dir(tmp_path):
    live, src = tmp_path / ".hermes", tmp_path / "src"
    (live / "hermes-agent").mkdir(parents=True)
    src.mkdir()
    good = {"python": str(tmp_path / "py"), "src": str(src), "sha": "abc"}
    path = tmp_path / "hosts.json"
    path.write_text(json.dumps({"hosts": {"ok": good, "live": {**good, "hermes_home": str(live / "profile")}}}))
    assert list(hosts.load(path, ["ok"], hermes_dir=live)) == ["ok"]
    with pytest.raises(ValueError, match="under the live"):
        hosts.load(path, ["live"], hermes_dir=live)
    path.write_text(json.dumps({"hosts": {"x": {**good, "src": str(live / "hermes-agent")}}}))
    with pytest.raises(ValueError, match="under the live"):
        hosts.load(path, hermes_dir=live)


def test_f5_symlinked_live_path_and_tree_manifest(tmp_path):
    live, outside = tmp_path / ".hermes", tmp_path / "outside"
    (live / "hermes-agent").mkdir(parents=True)
    outside.mkdir()
    (live / "link").symlink_to(outside)  # lexically under the live dir, resolves outside it
    (tmp_path / "back").symlink_to(live / "hermes-agent")  # lexically outside, resolves into it
    assert hosts.under_real_hermes(live / "link", hermes_dir=live)
    assert hosts.under_real_hermes(tmp_path / "back", hermes_dir=live)
    assert not hosts.under_real_hermes(outside, hermes_dir=live)
    repo, src = tmp_path / "repo", tmp_path / "export"
    repo.mkdir()
    (repo / "a.py").write_text("A = 1\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "x"], check=True)
    sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    (src / "tree").mkdir(parents=True)
    (src / "tree" / "a.py").write_text("A = 1\n")
    hosts.write_manifest(src / "tree", sha, repo)
    assert hosts.verify("x", {"src": str(src / "tree"), "sha": sha})["method"] == "tree-manifest"
    (src / "tree" / "a.py").write_text("A = 2\n")
    with pytest.raises(ValueError, match="tree hash"):
        hosts.verify("x", {"src": str(src / "tree"), "sha": sha})


def test_f6_final_check_never_falls_back_from_the_selected_api(monkeypatch):
    fallback = []

    def compress_now(*_a, **_k):
        raise ImportError("failure inside the selected path")
    for name, attrs in {"agent": {}, "agent.conversation_compression_manual": {"compress_now": compress_now,
                                                                            "parse_compress_args": lambda s: s},
                        "agent.conversation_compression": {"finalize_context_engine_compression_notification": print},
                        "acp_adapter": {}, "acp_adapter.commands": {"_estimate_tokens": lambda *a: 1}}.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    agent = types.SimpleNamespace(context_compressor=types.SimpleNamespace(_last_compression_status="compacted"),
                                  _compress_context=lambda *a, **k: fallback.append(1) or (a[0], None))
    out = probe.final_check(agent, [], probe.io.StringIO())
    assert out["outcome"] == "failed" and "failure inside" in out["exception"]
    assert out["entry"] == "compress_now" and not fallback and len(out["attempts"]) == 1


def test_r13_import_provenance_fails_closed_outside_the_host_tree(tmp_path, monkeypatch):
    src, tree, elsewhere = tmp_path / "src", tmp_path / "tree", tmp_path / "site-packages"
    for d in (src, tree, elsewhere):
        d.mkdir()
    cell = {"host_src": str(src), "host_python": sys.executable, "plugin": {"tree": str(tree), "module": "hermes_plugins.x"}}
    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(__file__=str(elsewhere / "run_agent.py")))
    monkeypatch.setitem(sys.modules, "hermes_state", types.SimpleNamespace(__file__=str(src / "hermes_state.py")))
    monkeypatch.setitem(sys.modules, "hermes_plugins.x", types.SimpleNamespace(__file__=str(src / "x.py")))
    bad = probe.provenance(cell)["violations"]
    assert any(v.startswith("run_agent ->") for v in bad) and any(v.startswith("hermes_plugins.x ->") for v in bad)
    assert not any(v.startswith("hermes_state ->") for v in bad)
    assert "sys.executable" in " ".join(probe.provenance({**cell, "host_python": "/nonexistent/python"})["violations"])


def test_d7_pc1_is_a_differential_encoded_as_data():
    pc1 = controls.CONTROLS["PC-1"]
    assert pc1["refs"] == ["47bd28e7", "ae1fb16d"]
    exp = pc1["expect"]
    assert exp[("47bd28e7", "baseline/in-place/acp")] == exp[("ae1fb16d", "baseline/in-place/acp")] == "PASS"
    assert exp[("47bd28e7", "acp-trailing/in-place")] == "FAIL" != exp[("ae1fb16d", "acp-trailing/in-place")]

    def rows(old_baseline):
        return [{"plugin_ref": ref, "cell": c, "host": h, "failed_bars": {"B1": {}, "B2": {}},
                 "verdict": exp[(ref, c)] if (ref, c) != ("47bd28e7", "baseline/in-place/acp")
                 else old_baseline} for ref in pc1["refs"] for c in pc1["cells"] for h in pc1["hosts"]]
    assert controls.check("PC-1", rows("PASS")) == []
    # round 1's PC-1 shape: the old tree fails the baseline too, so the red is not attributable to the transform
    assert len(controls.check("PC-1", rows("FAIL"))) == len(pc1["hosts"])


def test_plugin_identity_is_read_from_the_tree(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "plugin.yaml").write_text("name: hermes-lcm\n")
    (old / "engine.py").write_text("class E:\n    @property\n    def name(self) -> str:\n        return \"lcm\"\n")
    (new / "plugin.yaml").write_text("name: hermes-lcm-x\n")
    (new / "plugin_identity.py").write_text('ENGINE_NAME = "lcm-x"\n')
    assert plugin_tree.identity(old) == {"dir": "hermes-lcm", "enabled": "hermes-lcm", "engine": "lcm",
                                         "module": "hermes_plugins.hermes_lcm"}
    assert plugin_tree.identity(new)["engine"] == "lcm-x"


def fake_compress_host(monkeypatch, compress_now):
    for name, attrs in {"agent": {}, "agent.conversation_compression_manual": {"compress_now": compress_now,
                                                                            "parse_compress_args": lambda s: s},
                        "agent.conversation_compression": {"finalize_context_engine_compression_notification": lambda *a, **k: None},
                        "acp_adapter": {}, "acp_adapter.commands": {"_estimate_tokens": lambda *a: 1}}.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))


@pytest.mark.parametrize("host_status", ["skipped", "lock_skipped"])
def test_r14_a1_final_check_needs_a_fresh_compressed_pass(monkeypatch, host_status):
    engine = types.SimpleNamespace(_last_compression_status="compacted", _probe_status="compacted", _probe_calls=3)
    fake_compress_host(monkeypatch, lambda *a, **k: types.SimpleNamespace(status=host_status, after_messages=[]))
    stale = probe.final_check(types.SimpleNamespace(context_compressor=engine), [], probe.io.StringIO())
    assert stale["outcome"] != "published" and not stale["published"] and stale["engine_calls"] == 0

    def fresh(*_a, **_k):  # the host ran and the engine committed a pass inside this invocation
        engine._probe_calls, engine._probe_status = engine._probe_calls + 1, "compacted"
        return types.SimpleNamespace(status="compressed", after_messages=[])
    fake_compress_host(monkeypatch, fresh)
    assert probe.final_check(types.SimpleNamespace(context_compressor=engine), [], probe.io.StringIO())["outcome"] == "published"


def test_r14_a2_controls_never_hold_without_runs():
    assert controls.check("PC-2", []) and controls.check("PC-3", [])
    assert controls.check("PC-3", [], ["ref-0.21.5"])  # a requested host with no row is a problem
    assert controls.CONTROLS["PC-3"]["refs"] == ["508f893517f52a400c2bfe0b37f914e864ff806c"]
    row = {"plugin_ref": "v0.24.2", "host": "ref-0.21.5", "verdict": "FAIL", "failed_bars": {"B6": {}}}
    rows = [dict(row, cell=c) for c in controls.CONTROLS["PC-2"]["cells"]]
    assert controls.check("PC-2", rows, ["ref-0.21.5"]) == []
    assert controls.check("PC-2", rows, ["ref-0.21.5", "upstream-main"])  # "all" = requested hosts, not row hosts


def git_host(tmp_path):
    repo = tmp_path / "host"
    repo.mkdir()
    (repo / "run_agent.py").write_text("X = 1\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "x"], check=True)
    return repo, subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def test_r14_a3_git_host_identity_fails_closed(tmp_path):
    repo, sha = git_host(tmp_path)
    assert hosts.verify("h", {"src": str(repo), "sha": sha})["method"] == "git HEAD"
    (repo / "hermes_state.py").write_text("SHADOW = 1\n")  # untracked module that would be imported
    with pytest.raises(ValueError, match="untracked"):
        hosts.verify("h", {"src": str(repo), "sha": sha})
    (repo / "hermes_state.py").unlink()
    (repo / ".git" / "HEAD").write_text("garbage\n")  # every git command now fails
    with pytest.raises(ValueError, match="failed"):
        hosts.verify("h", {"src": str(repo), "sha": sha})


@pytest.mark.parametrize("missing", ["lcm.db", "state.db"])
def test_r14_a4_a_failed_db_copy_is_error_and_not_scored(tmp_path, monkeypatch, missing):
    home, dbdir = tmp_path / "home", tmp_path / "db"
    home.mkdir()
    dbdir.mkdir()
    for name in {"lcm.db", "state.db"} - {missing}:
        sqlite3.connect(home / name).close()
    errors = run_matrix.copy_dbs(home, dbdir)
    assert len(errors) == 1 and errors[0].startswith(missing)

    def never(*_a, **_k):
        raise AssertionError("scored despite a failed DB copy")
    monkeypatch.setattr(run_matrix.bars, "score", never)
    out = run_matrix.verdict_fields({"faults": []}, tmp_path, {"exit": "done"}, set(), {}, errors)
    assert out["verdict"] == "ERROR" and missing in out["reason"]


def test_985_a_failed_probe_keeps_its_own_reason_when_the_db_copy_fails(tmp_path, monkeypatch):
    """Regression (#985): a host that died before creating its databases reported only "database copy failed",
    which hid the probe's own error and stderr tail. The verdict stays ERROR and the reason now leads with the
    probe's failure; a finished probe with a failed copy keeps today's reason."""
    def never(*_a, **_k):
        raise AssertionError("scored despite a failed DB copy")
    monkeypatch.setattr(run_matrix.bars, "score", never)
    errors = ["lcm.db: FileNotFoundError('lcm.db does not exist')"]
    died = {"exit": "error", "reason": "host process ended: EOF; stderr: Traceback ... ImportError: x"}
    out = run_matrix.verdict_fields({"faults": []}, tmp_path, died, set(), {}, errors)
    assert out["verdict"] == "ERROR"
    assert out["reason"].startswith("probe failed before the databases were complete: host process ended: EOF")
    assert "ImportError: x" in out["reason"] and "database copy failed, not scored: lcm.db" in out["reason"]
    for last in ({"exit": "done"}, {"exit": "unsupported", "reason": "r"}):
        rec = run_matrix.verdict_fields({"faults": []}, tmp_path, last, set(), {}, errors)
        assert rec == {"verdict": "ERROR", "reason": "database copy failed, not scored: " + errors[0]}


def test_r14_t17_an_unapplied_bar_is_not_covered():
    row = {"cell": "native-long-prefix/rotation", "verdict": "PASS", "applicable_bars": ["B1", "B2", "B4", "B5", "B7"]}
    assert report.issue_status([row], "", ("B6",))[0].startswith("NOT COVERED on B6")
    assert report.issue_status([dict(row, applicable_bars=["B6"])], "", ("B6",))[0].startswith("target cells PASS")


def test_r14_t32_a_summary_node_across_lineages_fails(tmp_path):
    k = "[K01] user turn 1: alpha beta end."
    events = clean_events(2) + [
        {"phase": "A", "turn": 1, "event": "user_sent", "tag": "K01", "session_prefix": "K", "content": k, "persist": k},
        {"phase": "A", "turn": 1, "event": "emit", "tag": "K01", "text": "reply to K01: noted item 1."},
        {"phase": "A", "turn": 1, "event": "turn_end", "tag": "K01", "session_prefix": "K", "held": k,
         "user_tags": {"K01": 1}, "session": "cron_job_01"}]
    rows = clean_rows(2) + [("user", k), ("assistant", "reply to K01: noted item 1.")]
    sids, parents = ["S0"] * 4 + ["cron_job_01"] * 2, {"S0": None, "cron_job_01": None}
    ok = make(tmp_path, rows=rows, events=events, sids=sids, parents=parents, nodes=[[1, 2]])
    assert ok["verdict"] == "PASS", ok["failed_bars"]
    out = make(tmp_path / "x", rows=rows, events=events, sids=sids, parents=parents, nodes=[[1, 2, 5]])
    assert out["failed_bars"]["B5"]["cross_lineage_nodes"] == [1]


def test_r14_safety_refusals(tmp_path):
    for key in ("LCM_DATABASE_PATH", "LCM_EXPORT_DIR", "LCM_EMBEDDING_API_KEY", "LCM_AUTH_TOKEN", "HOME"):
        assert run_matrix.env_refusal({key: "x"}), key
    assert run_matrix.env_refusal({"LCM_CONTEXT_THRESHOLD": "0.5"}) is None
    # Token COUNT keys are tuning, not credentials (the fleet sets these two).
    for key in ("LCM_LEAF_CHUNK_TOKENS", "LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS", "LCM_RESERVE_TOKENS_FLOOR"):
        assert run_matrix.env_refusal({key: "8000"}) is None, key
    for key in ("LCM_API_KEY_TOKENS", "LCM_AUTH_TOKENS_SECRET", "LCM_ACCESS_TOKEN"):
        assert run_matrix.env_refusal({key: "x"}), key
    src = tmp_path / "src"
    src.mkdir()
    path = tmp_path / "hosts.json"
    for bad in ("../escape", "/abs", "a/b"):
        path.write_text(json.dumps({"hosts": {bad: {"python": "p", "src": str(src), "sha": "s"}}}))
        with pytest.raises(ValueError, match="safe path component"):
            hosts.load(path, hermes_dir=tmp_path / ".hermes")


def test_r15_a1_b2_is_exact_on_internal_whitespace(tmp_path):
    events = clean_events()
    sep = U.format(2, 2).replace(" alpha", "\n\nalpha")  # a prompt with planted paragraph separators (#545)
    for e in events:
        if e.get("tag") == "T02" and e["event"] == "user_sent":
            e["content"] = e["persist"] = sep
        if e.get("tag") == "T02" and e["event"] == "turn_end":
            e["held"] = sep
    rows = [r if r[1] != U.format(2, 2) else ("user", sep) for r in clean_rows()]
    assert make(tmp_path, rows=rows, events=events)["verdict"] == "PASS"
    collapsed = [r if r[1] != sep else ("user", sep.replace("\n\n", " ")) for r in rows]
    out = make(tmp_path / "c", rows=collapsed, events=events)
    assert "B2" in out["failed_bars"] and out["numbers"]["B2"]["missing_keys"] == 1


def test_r15_a2_tool_rows_are_scored_per_lineage(tmp_path):
    events, rows, plan = tool_cell()
    parents = {"S0": None, "cron_job_01": None}
    right = make(tmp_path, rows=rows, events=events, nodes=[[1, 2, 3, 4, 5]], sids=["S0"] * len(rows), parents=parents, **plan)
    assert right["verdict"] == "PASS", right["failed_bars"]
    wrong = ["S0", "S0", "cron_job_01", "cron_job_01", "cron_job_01"] + ["S0"] * (len(rows) - 5)  # tool rows in cron
    out = make(tmp_path / "w", rows=rows, events=events, nodes=[[1, 2]], sids=wrong, parents=parents, **plan)
    assert out["numbers"]["B2"]["tool_missing_rows"] == 4 and out["numbers"]["B2"]["tool_surplus_rows"] == 4


def test_r15_a3_a_modified_cached_export_is_never_reused(tmp_path):
    repo = tmp_path / "lcm"
    repo.mkdir()
    (repo / "plugin.yaml").write_text("name: hermes-lcm-x\n")
    (repo / "plugin_identity.py").write_text('ENGINE_NAME = "lcm-x"\n')
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "x"], check=True)
    first = plugin_tree.export(repo, "HEAD", tmp_path / "plugins")
    assert plugin_tree.export(repo, "HEAD", tmp_path / "plugins")["reused"] is True
    (Path(first["tree"]) / "plugin_identity.py").write_text('ENGINE_NAME = "tampered"\n')
    again = plugin_tree.export(repo, "HEAD", tmp_path / "plugins")
    assert again["reused"] is False and again["engine"] == "lcm-x"


def test_r15_a4_directory_symlinks_are_in_the_tree_hash(tmp_path):
    root = tmp_path / "tree"
    (root / "pkg").mkdir(parents=True)  # an empty package dir
    (root / "a.py").write_text("A = 1\n")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "shadow.py").write_text("X = 1\n")
    before = hosts.tree_hash(root)
    (root / "pkg").rmdir()
    (root / "pkg").symlink_to(tmp_path / "other")  # now a directory symlink to importable code outside the tree
    assert hosts.tree_hash(root) != before


def test_r15_a5_pc1_must_fail_its_intended_bars():
    pc1 = controls.CONTROLS["PC-1"]
    assert pc1["bars"] == ["B1", "B2"]
    rows = [{"plugin_ref": ref, "cell": c, "host": h, "verdict": pc1["expect"][(ref, c)],
             "failed_bars": {"B5": {}} if pc1["expect"][(ref, c)] == "FAIL" else {}}
            for ref in pc1["refs"] for c in pc1["cells"] for h in pc1["hosts"]]
    assert len(controls.check("PC-1", rows)) == len(pc1["hosts"])  # red for another reason does not HOLD


def test_r15_b_safety_and_precision(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "elsewhere").mkdir()
    (out / "cells").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        run_matrix.run_cell({"id": "x", "lcm_env": {}}, "h", {}, {"sha": "a" * 40}, out, 1, False)
    for key in ("LCM_database_path", "LCM_api_secret"):
        assert key in (run_matrix.env_refusal({key: "x"}) or "")
    rec = run_matrix.verdict_fields({"faults": []}, tmp_path, {"exit": "unsupported"}, set(), {}, ["lcm.db: missing"])
    assert rec["verdict"] == "ERROR"
    tree = Path(__file__).resolve().parent.parent  # this repo's own config.py: the plugin's parser
    for value, want in (("1", True), ("yes", True), ("On", True), ("true", True), ("false", False), ("garbage", False)):
        assert plugin_tree.parse_bool(tree, "LCM_NATIVE_RECOVERY", value)[0] is want, value


def test_r2a3_b5_a_counted_node_with_no_message_source_fails(tmp_path):
    """Regression (R2a.3, #567 thread): a message-sourced node with source_ids [] counted toward depth-0 growth
    and passed the cross-lineage check."""
    assert make(tmp_path, rows=clean_rows(), events=clean_events(), nodes=[[1, 2]])["verdict"] == "PASS"
    for i, bad in enumerate(([], "not-a-list", [99], [True])):  # a JSON true is not store id 1
        out = make(tmp_path / str(i), rows=clean_rows(), events=clean_events(), nodes=[[1, 2], bad])
        assert out["failed_bars"]["B5"]["empty_source_nodes"] == [2], bad
    assert bars.source_ids("{unparseable") == [] and bars.source_ids("null") == [] and bars.source_ids("[1]") == [1]


def test_r2a3_malformed_durable_tool_calls_are_b2_surplus(tmp_path):
    """Regression (R2a.3, #567 thread): malformed or non-list tool_calls JSON was silently "no calls"."""
    for i, calls in enumerate(("{not json", '{"id": "c1"}', '"x"', "[1]", "")):  # only NULL means no calls
        out = make(tmp_path / str(i), rows=clean_rows() + [("assistant", "", calls)], events=clean_events())
        assert out["failed_bars"]["B2"]["tool_surplus_rows"] == 1, calls
        assert out["numbers"]["B2"]["tool_surplus"] == [["chat", "malformed_tool_calls", 7]], calls
    assert make(tmp_path / "ok", rows=clean_rows() + [("assistant", "", "[]")], events=clean_events())["verdict"] == "PASS"


def test_r2a3_b1_covers_three_digit_turn_tags(tmp_path):
    """Regression (R2a.3, #570 thread): extend_turns can emit T101-T103 on a 100-turn cell; B1 parsed [A-Z]\\d\\d only."""
    ok = make(tmp_path, rows=clean_rows(101), events=clean_events(101))
    assert ok["verdict"] == "PASS" and ok["numbers"]["B1"]["user_tags"] == 101, ok["failed_bars"]
    rows = [r for r in clean_rows(101) if r[1] != U.format(101, 101)]
    out = make(tmp_path / "lost", rows=rows, events=clean_events(101))
    assert out["failed_bars"]["B1"] == {"T101": {"expected": 1, "stored": 0}}
    assert "T09" in bars.re.findall(r"\[([A-Z]\d{2,3})\] user", U.format(9, 9))  # leading zero kept below 100


def test_r2a3_issue_566_is_decided_by_b5_too():
    assert "B5" in cells.ISSUES[566][0]


TREE = {"tree": str(Path(__file__).resolve().parent.parent)}  # this checkout is an lcm-x plugin tree


def held(n=3, sid="S0"):
    return [(sid, role, text, 1) for role, text in clean_rows(n)]


def test_r2a4_a_host_parity_duplicate_is_licensed(tmp_path):
    dup = ("S0", "user", U.format(2, 2), 1)  # the host durably holds T02 twice in one view (H2 rows 105/106)
    out = make(tmp_path, rows=clean_rows() + [("user", U.format(2, 2))], events=clean_events(), host=held() + [dup], plugin=TREE)
    assert out["verdict"] == "PASS", out["failed_bars"]
    lic = out["numbers"]["B2"]["host_parity_licensed"]
    assert lic["rows"] == 1 and out["numbers"]["B1"]["host_parity_licensed"]["rows"] == 1
    assert lic["records"][0] | {"sha256": None} == {
        "role": "user", "sha256": None, "expected": 1, "stored": 2, "host": 2, "licensed": 1, "store_ids": [3, 7],
        "host_row_ids": [3, 7], "tags": ["T02"], "session": "chat"}
    assert "host-dup=B2:1/B1:1" in report.signature({"verdict": "PASS", **out})


def test_r2a4_b_an_lcm_only_duplicate_is_not_licensed(tmp_path):
    copies = [("S0", "user", U.format(2, 2), 0),  # an in-place generation copy (inactive)
              ("child", "user", U.format(2, 2), 1)]  # a rotation copy in a child session
    out = make(tmp_path, rows=clean_rows() + [("user", U.format(2, 2))], events=clean_events(),
               host=held() + copies, parents={"S0": None, "child": "S0"}, plugin=TREE)
    assert {"B1", "B2"} <= set(out["failed_bars"]) and out["numbers"]["B2"]["surplus_rows"] == 1
    assert out["numbers"]["B2"]["host_parity_licensed"]["rows"] == 0
    # H1 re-issues a row with a fresh timestamp per in-place generation: still one occurrence
    fresh = [("S0", "user", U.format(2, 2), 0)] * 3
    assert "B2" in make(tmp_path / "f", rows=clean_rows() + [("user", U.format(2, 2))], events=clean_events(),
                        host=held() + fresh, plugin=TREE)["failed_bars"]


def test_r2a4_c_a_deficit_is_never_licensed(tmp_path):
    out = make(tmp_path, rows=clean_rows()[:-2] + [clean_rows()[-1]], events=clean_events(),
               host=held() + [("S0", "user", U.format(3, 3), 1)], plugin=TREE)
    assert out["numbers"]["B2"]["deficit_rows"] == 1 and {"B1", "B2"} <= set(out["failed_bars"])


def _held_composite(tmp_path, *, parts_sid="S0", host_parts=(14, 15), inner=""):
    """#804 shape: the host held T14 + "\n\n" + T15 as one live composite but stored the parts apart; LCM stored the
    parts as two rows."""
    t14, t15 = U.format(14, 14) + inner, U.format(15, 15)
    host = [(parts_sid, "user", U.format(t, t) + (inner if t == 14 else ""), 1) for t in host_parts]
    parents = {"S0": None, **({parts_sid: None} if parts_sid != "S0" else {})}
    return make(tmp_path, rows=[("user", t14), ("user", t15), ("assistant", R.format(15, 15))],
                sids=[parts_sid, parts_sid, "S0"], events=turn_events(15, held=t14 + "\n\n" + t15),
                parents=parents, host=host, plugin=TREE, bars=["B2"])


def test_b2_held_composite_stored_as_host_parts_is_not_a_deficit(tmp_path):
    for i, inner in enumerate(("", "\n\nsecond paragraph\n\nthird")):  # a part may hold its own "\n\n"
        out = _held_composite(tmp_path / str(i), inner=inner)
        b2 = out["numbers"]["B2"]
        assert "B2" not in out["failed_bars"], out["failed_bars"]
        assert b2["missing_keys"] == b2["deficit_rows"] == b2["surplus_rows"] == 0
        assert b2["host_parity_licensed"]["rows"] == 0  # the licences were spent on the composite
        [composite] = b2["held_composites_as_parts"]
        assert composite["preview"].startswith("[T14]") and [p["store_ids"] for p in composite["parts"]] == [[1], [2]]


def test_b2_held_composite_needs_host_evidence_for_every_part(tmp_path):
    out = _held_composite(tmp_path, host_parts=(14,))
    b2 = out["numbers"]["B2"]
    assert "B2" in out["failed_bars"] and b2["missing_keys"] == 1 and not b2["held_composites_as_parts"]


def test_b2_held_composite_the_host_stored_durably_stays_a_deficit(tmp_path):
    """The host holds the composite itself as a durable row (and the parts): LCM should have stored it."""
    composite = U.format(14, 14) + "\n\n" + U.format(15, 15)
    out = make(tmp_path, rows=[("user", U.format(14, 14)), ("user", U.format(15, 15)), ("assistant", R.format(15, 15))],
               events=turn_events(15, held=composite), parents={"S0": None}, plugin=TREE, bars=["B2"],
               host=[("S0", "user", composite, 1), ("S0", "user", U.format(14, 14), 1), ("S0", "user", U.format(15, 15), 1)])
    b2 = out["numbers"]["B2"]
    assert "B2" in out["failed_bars"] and b2["missing_keys"] == 1 and not b2["held_composites_as_parts"]


def test_b2_composite_cover_respects_licence_capacity_while_searching():
    from bench.instruments.reliability.scorers import multiset
    a, b = "alpha part", "beta part"
    lic = {("user", multiset.h(x)): {"licensed": 1} for x in (a, a + "\n\n" + a, b)}
    assert multiset.licensed_parts(a + "\n\n" + a + "\n\n" + b, lic, 1) == {
        ("user", multiset.h(a + "\n\n" + a)): 1, ("user", multiset.h(b)): 1}
    assert multiset.licensed_parts(a + "\n\n" + b, {("user", multiset.h(a)): {"licensed": 1}}, 1) is None


def test_b2_composite_cover_is_fail_closed_on_a_pathological_input():
    """70 paragraphs, every single and adjacent pair licensed, and an unlicensed tail: the budget ends the search
    quickly and the composite stays a deficit."""
    import time
    from bench.instruments.reliability.scorers import multiset
    paras = [f"paragraph {i}" for i in range(70)]
    lic = {("user", multiset.h(p)): {"licensed": 1} for p in paras}
    lic.update({("user", multiset.h(a + "\n\n" + b)): {"licensed": 1} for a, b in zip(paras, paras[1:])})
    started = time.monotonic()
    assert multiset.licensed_parts("\n\n".join(paras + ["unlicensed tail"]), lic, 1) is None
    assert time.monotonic() - started < 10


def test_b2_composite_cover_cuts_inside_a_longer_newline_run():
    """A raw part may end or start with whitespace (the probe's trailing_ws turns): R + "\n" + "\n\n" + U + "\n" still
    splits into the two edge-stripped keys."""
    from bench.instruments.reliability.scorers import multiset
    a, b = "alpha part", "beta part"
    lic = {("user", multiset.h(x)): {"licensed": 1} for x in (a, b)}
    assert multiset.licensed_parts(a + "\n" + "\n\n" + b + "\n", lic, 1) == {
        ("user", multiset.h(a)): 1, ("user", multiset.h(b)): 1}


def test_b2_composite_cover_is_fail_closed_past_its_part_cap():
    from bench.instruments.reliability.scorers import multiset
    paras = [f"paragraph {i}" for i in range(multiset.COVER_PARTS + 1)]
    lic = {("user", multiset.h(p)): {"licensed": 1} for p in paras}
    assert multiset.licensed_parts("\n\n".join(paras[:-1]), lic, 1)  # exactly COVER_PARTS parts
    assert multiset.licensed_parts("\n\n".join(paras), lic, 1) is None


def test_b2_held_composite_pairs_only_the_occurrences_the_host_did_not_store_whole():
    """Expected twice, stored whole once: the other occurrence pairs with host-licensed parts unless the host stored
    both occurrences whole."""
    from bench.instruments.reliability.scorers import multiset
    a, b = "alpha part", "beta part"
    c = a + "\n\n" + b
    rows = [(1, "S0", "user", c), (2, "S0", "user", a), (3, "S0", "user", b)]
    def host(n):
        return {("user", multiset.h(x)): {"n": k, "ids": [x]} for x, k in ((a, 1), (b, 1), (c, n))}
    for held, verdict in ((0, "PASS"), (1, "PASS"), (2, "FAIL")):
        out = multiset.score([("user", c), ("user", c)], rows, host(held))
        assert out["verdict"] == verdict, (held, out)
        assert len(out["held_composites_as_parts"]) == (verdict == "PASS")
    [composite] = multiset.score([("user", c), ("user", c)], rows, host(1))["held_composites_as_parts"]
    assert composite["expected"] == 2 and composite["as_parts"] == 1


def test_b2_held_composite_parts_in_another_lineage_do_not_cover_it(tmp_path):
    out = _held_composite(tmp_path, parts_sid="cron_job_01")
    b2 = out["numbers"]["B2"]
    assert "B2" in out["failed_bars"] and b2["missing_keys"] == 1 and not b2["held_composites_as_parts"]


def test_r2a4_d_a_licence_is_per_lineage(tmp_path):
    other = ("cron_job_01", "user", U.format(2, 2), 1)  # the second copy is held in another lineage
    out = make(tmp_path, rows=clean_rows() + [("user", U.format(2, 2))], events=clean_events(),
               host=held() + [other], parents={"S0": None, "cron_job_01": None}, plugin=TREE)
    assert {"B1", "B2"} <= set(out["failed_bars"]) and out["numbers"]["B2"]["host_parity_licensed"]["rows"] == 0


def test_r2a4_no_host_evidence_no_licence(tmp_path):
    rows, events = clean_rows() + [("user", U.format(2, 2))], clean_events()
    for i, kw in enumerate(({}, {"parents": {"S0": None}},  # no state.db; a state.db with no messages table
                            {"host": held() + [("S0", "user", U.format(2, 2), 1)]})):  # no plugin tree: no carrier markers
        out = make(tmp_path / str(i), rows=rows, events=events, **kw)
        assert out["verdict"] == "FAIL" and out["numbers"]["B2"]["host_parity_licensed"]["unavailable"]


def test_r2a4_reports_licences_visibly(tmp_path):
    out = make(tmp_path, rows=clean_rows() + [("user", U.format(2, 2))], events=clean_events(),
               host=held() + [("S0", "user", U.format(2, 2), 1)], plugin=TREE)
    rec = {"cell": "baseline/in-place/acp", "host": "h", "plugin_ref": "r", "plugin_sha": "a" * 40, "host_sha": "b" * 40,
           "targets": [], **out}
    report.write(tmp_path, [rec], 1.0)
    matrix, im = (tmp_path / "MATRIX.md").read_text(), (tmp_path / "ISSUE-MAP.md").read_text()
    assert "| host-dup |" in matrix and "| h: B2 1/B1 1 |" in matrix
    assert "- `baseline/in-place/acp` on h: B2 1 / B1 1 rows licensed" in matrix
    assert "lcm [3, 7] host rows [3, 7]" in im and "tags T02" in im


def test_r2a4_541_gate_cell_injects_once_and_the_persistent_cell_is_data():
    by = {c["id"]: c for c in cells.select("publication-failure/*")}
    once, always = by["publication-failure/rotation-child"], by["publication-failure/rotation-child-persistent"]
    assert not once["faults"][0].get("persistent") and always["faults"][0]["persistent"] and not always["targets"]
    assert ci.in_gate_set(once["id"]) and not ci.in_gate_set(always["id"])
    for transport in (None, "acp-process"):
        assert always["id"] in ci.expected_cells(transport)
    src = Path(probe.__file__).read_text()
    assert 'pf.get("persistent") or pf["kind"] not in fired' in src


def test_r2a4_an_echoed_lcm_carrier_is_never_licensed(tmp_path):
    carrier = "[Recent Summary (d0, node 1)]\nStub summary #1 covers T01.\n\n" + U.format(2, 2)  # carrier-headed composite
    rows = clean_rows() + [("user", carrier)]
    host = held() + [("S0", "user", carrier, 1)]  # the host echoed it durably, active
    out = make(tmp_path, rows=rows, events=clean_events(), host=host, plugin=TREE)
    assert out["numbers"]["B2"]["host_parity_licensed"]["rows"] == 0 and {"B1", "B2"} <= set(out["failed_bars"])
    todo = "[Your active task list was preserved across context compression]\n- item"
    out = make(tmp_path / "t", rows=clean_rows() + [("user", todo)], events=clean_events(),
               host=held() + [("S0", "user", todo, 1)], plugin=TREE)
    assert out["numbers"]["B2"]["host_parity_licensed"]["rows"] == 0 and "B2" in out["failed_bars"]


def test_b9_only_cell_without_the_audit_is_unsupported_never_pass(tmp_path):
    """A P8 control proves nothing when B9 is UNSUPPORTED (audit off, or a host without the seams)."""
    out = make(tmp_path, rows=clean_rows(), events=clean_events(), bars=["B9"])
    assert out["verdict"] == "UNSUPPORTED" and out["applicable_bars"] == [], out
    assert out["reason"].startswith("B9: ")
