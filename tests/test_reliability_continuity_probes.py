"""#659 PR A: continuity probes P1 (host instruction), P2 (open todos), P5 (constraint before the fresh tail).
Fixture logs, fixture host modules and a fixture provider only; never launch a Hermes host."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.instruments.reliability import cells, ci, fake_provider as FP, process_cell as PC, report  # noqa: E402
from bench.instruments.reliability import run_matrix as RM  # noqa: E402
from bench.instruments.reliability.scorers import continuity as CT, tool_calls as TC  # noqa: E402

V1, V2 = "QXWZKVJMTR", "ZWVKQJMXRT"
CARRIERS = (re.compile(r"\[(?:Recent|Durable) Summary \(d\d+, node \d+\)\]"),
            ("[Current user objective preserved from compacted history]",
             "[Your active task list was preserved across context compression]"))


def probe_cell(shape, mode="in-place"):
    return cells.select(f"continuity/{shape}/{mode}", opt_in=PC.R2_PROBE_CELLS)[0]


# -- request-time markers ---------------------------------------------------------------------------------------
def test_nonce_markers_classify_every_carrier_without_request_text():
    messages = [
        {"role": "system", "content": f"identity {V1} line"},
        {"role": "user", "content": f"[T03] user turn 3: CONSTRAINT {V2}: fixture end."},
        {"role": "user", "content": f"[Recent Summary (d0, node 4)]\nsummary mentions {V2}"},
        {"role": "user", "content": f"[Current user objective preserved from compacted history]\n{V2}"},
        {"role": "user", "content": f"[T05] user turn 5: {V1} fixture.\n\n{CT.TODO_HEADER}\n- [>] 1. open {V1} (in_progress)"},
        {"role": "assistant", "content": f"reply mentions {V2}"},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"todos": [{"content": V2}]})}]
    data = CT.markers(messages, nonces=[V1, V2, "HWXQZTLGPN"], tool_args=[json.dumps({"content": V1})],
                      carriers=CARRIERS)
    assert data["nonces"][V1] == {"present": True, "count": 4,
                                  "carriers": {"system": 1, "user_raw": 1, "todo_fold": 1, "tool_args": 1}}
    assert data["nonces"][V2] == {"present": True, "count": 5, "carriers": {
        "user_raw": 1, "summary_or_head": 2, "assistant": 1, "tool_result": 1}}
    assert data["nonces"]["HWXQZTLGPN"] == {"present": False, "count": 0, "carriers": {}}
    leaked = json.dumps(data)
    assert "fixture" not in leaked and "CONSTRAINT" not in leaked and "summary mentions" not in leaked
    # Without probe nonces the record keeps today's keys.
    assert "nonces" not in CT.markers(messages)


def test_markers_without_plugin_carrier_markers_treat_summary_rows_as_raw_user_rows():
    msgs = [{"role": "user", "content": f"[Recent Summary (d0, node 4)] {V1}"}]
    assert CT.markers(msgs, nonces=[V1])["nonces"][V1]["carriers"] == {"user_raw": 1}


def test_provider_scans_tool_call_arguments_that_normalize_drops(tmp_path):
    body = {"model": "rel/main", "messages": [
        {"role": "user", "content": "[T03] user turn 3: x end."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "tool_call", "arguments": json.dumps({"name": "todo_list", "arguments": {"todos": [
                {"id": "1", "content": f"open {V1}", "status": "pending"}]}})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"}]}
    assert V1 not in json.dumps(FP.normalize(body, "openai"))
    assert any(V1 in a for a in FP.tool_call_args(body, "openai"))
    anthropic = {"messages": [{"role": "assistant", "content": [
        {"type": "tool_use", "id": "c1", "name": "tool_call", "input": {"name": "todo_list", "arguments": {"x": V2}}}]}]}
    assert any(V2 in a for a in FP.tool_call_args(anthropic, "anthropic"))
    p = FP.FakeProvider(tmp_path / "req.jsonl", main=lambda m: {"content": "ok"},
                        continuity_context=lambda: {"current": "T03", "nonces": [V1, V2]})
    try:
        p.json_reply = lambda *a: None
        p.serve(types.SimpleNamespace(path="/v1/chat/completions"), body, "openai")
    finally:
        p.server.server_close()
    [rec] = PC.read_jsonl(tmp_path / "req.jsonl")
    assert rec["continuity"]["nonces"][V1] == {"present": True, "count": 1, "carriers": {"tool_args": 1}}
    assert rec["continuity"]["nonces"][V2]["present"] is False and rec["current_user_tag"] == "T03"


# -- scoring: presence per compaction event -----------------------------------------------------------------------
def req(rid, ts, turn, nonces):
    return {"rid": rid, "ts": ts, "role": "main", "turn": turn,
            "continuity": {"F1": True, "user_tags": [], "reply_tags": [], "nonces": nonces}}


def hits(**carriers):
    return {"present": bool(carriers), "count": sum(carriers.values()), "carriers": carriers}


def commit(ts, turn):
    return {"kind": "compaction_committed", "ts": ts, "phase": "A", "turn": turn, "compaction_kind": "in-place"}


def test_host_instruction_rows_follow_the_rewrite_and_the_rebuild_control_scans_between():
    cell = probe_cell("probe-host-instruction")
    v1, v2 = cell["continuity"]["soul"]["nonce"], cell["continuity"]["soul"]["rewrite"]["nonce"]
    old, new, none = {v1: hits(system=1), v2: hits()}, {v1: hits(), v2: hits(system=1)}, {v1: hits(), v2: hits()}
    obs = [commit(1, 4), {"kind": "probe_rewrite", "ts": 5, "turn": 16}, commit(8, 18), commit(12, 22)]
    reqs = [req(1, 2, 4, old), req(2, 6, 16, old), req(3, 7, 17, {v1: hits(system=1), v2: hits(system=1)}),
            req(4, 9, 18, new), req(5, 13, 22, none)]
    scored = CT.score(cell, reqs, obs)
    p = scored["probes"]["host-instruction"]
    assert [e["ok"] for e in p["events"]] == [True, True, False]
    assert (p["met"], p["missed"], p["unknown"]) == (2, 1, 0)
    assert p["events"][0]["nonces"][v1] == {"expect": "present", "present": True, "carriers": {"system": 1}}
    rebuild = scored["probes"]["host-instruction-rebuild"]
    assert (rebuild["requests"], rebuild["met"], rebuild["missed"]) == (2, 1, 1)  # rid 3 already carries v2
    assert scored["scenario_observed"] is True


def test_open_todos_must_sit_in_the_fold_and_the_completed_item_never_does():
    cell = probe_cell("probe-todo")
    specs = {s["nonce"]: s for s in cell["continuity"]["probes"]}
    [done] = [n for n, s in specs.items() if s["expect"] == "absent"]
    opened = sorted(n for n in specs if n != done)
    folded = {**{n: hits(todo_fold=1) for n in opened}, done: hits(tool_result=1)}
    raw_only = {**{n: hits(tool_result=1) for n in opened}, done: hits()}
    leaked = {**{n: hits(todo_fold=1) for n in opened}, done: hits(todo_fold=1)}
    obs = [commit(1, 2), commit(3, 6), commit(5, 9), commit(7, 12)]
    reqs = [req(1, 2, 2, raw_only), req(2, 4, 6, folded), req(3, 6, 9, raw_only), req(4, 8, 12, leaked)]
    scored = CT.score(cell, reqs, obs)
    assert [e["ok"] for e in scored["probes"]["todo-open"]["events"]] == [True, False, True]
    assert scored["probes"]["todo-open"]["not_applicable"] == 1  # turn 2: before the todo call
    assert [e["ok"] for e in scored["probes"]["todo-done-control"]["events"]] == [True, True, False]


@pytest.mark.parametrize("shape", ["probe-constraint-head", "probe-constraint-mid", "probe-constraint-shape"])
def test_constraint_presence_per_event_with_unknown_rows_and_a_tail_control(shape):
    cell = probe_cell(shape)
    [n] = [s["nonce"] for s in cell["continuity"]["probes"] if s["probe"] == "constraint"]
    ctrl = {s["from_turn"]: s["nonce"] for s in cell["continuity"]["probes"] if s["probe"] == "constraint-control"}
    obs = [commit(1, 4), commit(3, 8), commit(5, 12), commit(7, 28), commit(9, 29)]
    reqs = [req(1, 2, 4, {n: hits(user_raw=1)}), req(2, 4, 8, {n: hits()}),
            {"rid": 3, "ts": 6, "role": "main", "turn": 12, "continuity": {}},
            req(4, 8, 28, {n: hits(), ctrl[28]: hits(user_raw=1)}),
            req(5, 10, 29, {n: hits(), ctrl[28]: hits(user_raw=1), ctrl[29]: hits()})]  # turn 29's own row dropped
    scored = CT.score(cell, reqs, obs)
    c = scored["probes"]["constraint"]
    assert [e["ok"] for e in c["events"]] == [True, False, None, False, False]
    assert (c["met"], c["missed"], c["unknown"]) == (1, 3, 1)
    control = scored["probes"]["constraint-control"]
    assert [e["ok"] for e in control["events"]] == [True, False]  # an older turn's control never stands in
    assert list(control["events"][1]["nonces"]) == [ctrl[29]] and control["not_applicable"] == 3
    # F1-F4 rows are unchanged beside the probes.
    assert [r["F1"] for r in scored["rows"]] == [True, True, None, True, True]


def test_each_current_turn_has_its_own_control_nonce():
    for shape in ("probe-constraint-head", "probe-constraint-mid", "probe-constraint-shape"):
        cell = probe_cell(shape)
        specs = [s for s in cell["continuity"]["probes"] if s["probe"] == "constraint-control"]
        assert [(s["from_turn"], s["to_turn"]) for s in specs] == [(t, t) for t in range(20, 31)]
        assert len({s["nonce"] for s in specs}) == 11 and len({len(s["nonce"]) for s in specs}) == 1
        for s in specs:
            assert s["nonce"] in PC.P1.user_text(cell, "T", s["from_turn"])
            assert s["nonce"] not in PC.P1.user_text(cell, "T", s["from_turn"] + 1)


def test_host_instruction_needs_an_observed_commit_after_the_rewrite():
    cell = probe_cell("probe-host-instruction")
    v1, v2 = cell["continuity"]["soul"]["nonce"], cell["continuity"]["soul"]["rewrite"]["nonce"]
    old = {v1: hits(system=1), v2: hits()}
    reqs = [req(1, 2, 4, old), req(2, 4, 8, old), req(3, 6, 12, old)]
    # Two commits, then the rewrite with no commit after it: the new line was never tested.
    late = CT.score(cell, reqs, [commit(1, 4), commit(3, 8), {"kind": "probe_rewrite", "ts": 5, "turn": 12}])
    assert late["scenario_observed"] is False
    assert late["probes"]["host-instruction"]["met"] == 2
    # No rewrite at all: every event precedes it, so the before-window specs still apply.
    never = CT.score(cell, reqs, [commit(1, 4), commit(3, 8)])
    assert never["scenario_observed"] is False
    p = never["probes"]["host-instruction"]
    assert (p["met"], p["missed"], p["not_applicable"]) == (2, 0, 0)
    assert set(p["events"][0]["nonces"]) == {v1, v2} and "host-instruction-rebuild" not in never["probes"]


def test_probe_scores_need_two_observed_commits_and_leave_other_cells_unchanged():
    cell = probe_cell("probe-constraint-head")
    one = CT.score(cell, [req(1, 2, 4, {})], [commit(1, 4)])
    assert one["scenario_observed"] is False
    plain = CT.score({}, [req(1, 2, 4, {})], [commit(1, 4)])
    assert "probes" not in plain and plain["scenario_observed"] is None


# -- recorded verdict -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("verdict", ["PASS", "FAIL", "UNSUPPORTED", "INCONCLUSIVE"])
def test_apply_probes_records_rates_and_never_touches_the_verdict(verdict):
    cell = probe_cell("probe-host-instruction")
    probes = {"host-instruction": {"events": [], "met": 3, "missed": 1, "unknown": 0, "not_applicable": 0},
              "host-instruction-rebuild": {"requests": 2, "met": 2, "missed": 0, "unknown": 0},
              "constraint": {"events": [], "met": 0, "missed": 0, "unknown": 2, "not_applicable": 0}}
    rec = {"verdict": verdict, "failed_bars": {"B2": {}}, "applicable_bars": ["B1"],
           "continuity": {"scenario_observed": True, "probes": probes}}
    before = json.loads(json.dumps(rec))
    PC.apply_probes(rec, cell)
    assert {k: rec[k] for k in before} == before
    got = rec["probes_recorded"]
    assert got["host-instruction"] == {"met": 3, "missed": 1, "unknown": 0, "rate": 0.75, "bar": 1.0,
                                       "recorded": "below bar"}
    assert got["host-instruction-rebuild"]["recorded"] == "meets bar" and got["host-instruction-rebuild"]["rate"] == 1.0
    assert got["constraint"]["rate"] is None and got["constraint"]["recorded"] == "not observed"
    unknown = {"host-instruction": {"events": [], "met": 2, "missed": 0, "unknown": 1, "not_applicable": 0},
               "host-instruction-rebuild": {"requests": 3, "met": 2, "missed": 1, "unknown": 1}}
    PC.apply_probes(rec := {"verdict": verdict, "continuity": {"scenario_observed": True, "probes": unknown}}, cell)
    assert rec["probes_recorded"]["host-instruction"]["recorded"] == "incomplete (unknown events)"  # never "meets bar"
    assert rec["probes_recorded"]["host-instruction-rebuild"]["recorded"] == "below bar"  # a miss decides
    error = {"verdict": "ERROR", "continuity": {"probes": probes}}
    PC.apply_probes(error, cell)
    assert "probes_recorded" not in error
    PC.apply_probes(rec := {"verdict": "PASS", "continuity": {"scenario_observed": False, "probes": probes}}, cell)
    assert {p["recorded"] for p in rec["probes_recorded"].values()} == {"scenario not observed"}


def test_matrix_renders_the_recorded_probe_section(tmp_path):
    row = {"host": "fixture-host", "host_sha": "h" * 40, "plugin_ref": "HEAD", "plugin_sha": "p" * 40,
           "cell": "continuity/probe-todo/in-place", "transport": "acp-process", "verdict": "FAIL",
           "probes_recorded": {"todo-open": {"met": 2, "missed": 1, "unknown": 0, "rate": 2 / 3, "bar": None,
                                             "recorded": "recorded"}}}
    report.write(tmp_path, [row], 1)
    text = (tmp_path / "MATRIX.md").read_text()
    assert "## #659 continuity probes (recorded, non-gating)" in text
    assert "| fixture-host | `continuity/probe-todo/in-place` | todo-open | 2 | 1 | 0 | 67% | - | recorded |" in text


# -- cells ------------------------------------------------------------------------------------------------------
IDS = {f"continuity/probe-{s}/{m}" for s in ("host-instruction", "todo", "constraint-head", "constraint-mid",
                                             "constraint-shape") for m in ("in-place", "rotation")}


def test_probe_cells_are_opt_in_r2_only_and_outside_the_nightly_gate():
    assert {c["id"] for c in PC.R2_PROBE_CELLS} == IDS
    assert not IDS & {c["id"] for c in cells.select("all", extra=PC.R2_CELLS)}
    assert not IDS & set(ci.expected_cells("acp-process")) and not IDS & set(ci.NON_GATE)
    assert not any(ci.in_gate_set(i) for i in IDS)
    assert {c["id"] for c in cells.select("continuity/probe-*", extra=PC.R2_CELLS, opt_in=PC.R2_PROBE_CELLS)} == IDS
    assert len(cells.select("continuity/*", extra=PC.R2_CELLS)) == 8  # the existing continuity set is unchanged
    with pytest.raises(ValueError):
        cells.select("continuity/probe-*", extra=PC.R2_CELLS)
    for c in PC.R2_PROBE_CELLS:
        cells.validate(c)
        assert PC.unsupported(c, "acp-process") is None and c["targets"] == []
        assert c["final_compaction_check"] is False and c["min_compactions"] >= 2
        assert float(c["lcm_env"]["LCM_CONTEXT_THRESHOLD"]) * c["window"] == 32000
        nonces = [s["nonce"] for s in c["continuity"]["probes"]]
        assert nonces and all(re.fullmatch(r"[G-Zg-z-]+", n) for n in nonces)  # no hex-shaped token


def test_run_matrix_selects_probe_cells_only_by_explicit_pattern():
    src = Path(RM.__file__).read_text()
    assert "opt_in=process_cell.R2_PROBE_CELLS if a.transport else ()" in src


def test_todo_cell_drives_the_default_tool_call_bridge():
    c = probe_cell("probe-todo", "rotation")
    [group] = c["tool_plan"]
    [call] = group["calls"]
    assert group["turns"] == [3] and call["name"] == "tool_call" and call["args"]["name"] == "todo_list"
    statuses = {t["status"] for t in call["args"]["arguments"]["todos"]}
    assert statuses == {"in_progress", "pending", "completed"}
    plugin = {"engine": "lcm", "enabled": "lcm", "dir": "lcm", "sha": "p" * 40}
    assert "tool_search" not in PC.config_yaml(c, plugin, "http://127.0.0.1:1/v1")  # the fleet defer list stands


def test_constraint_payload_sits_at_the_head_or_middle_and_keeps_the_turn_frame():
    head, mid = probe_cell("probe-constraint-head"), probe_cell("probe-constraint-mid")
    shape = probe_cell("probe-constraint-shape")
    n_head = head["user_text"]["payload"]["3"]["text"]
    text = PC.P1.user_text(head, "T", 3)
    assert text.startswith("[T03] user turn 3: " + n_head) and text.endswith(" end.")
    text = PC.P1.user_text(mid, "T", 3)
    k = text.index(mid["user_text"]["payload"]["3"]["text"])
    assert text.startswith("[T03] user turn 3: alpha") and 0.4 < k / len(text) < 0.6
    assert PC.P1.user_text(head, "T", 4) == PC.P1.user_text({"user_text": {"repeat": head["user_text"]["repeat"]}},
                                                         "T", 4)
    [n_shape] = [s["nonce"] for s in shape["continuity"]["probes"] if s["probe"] == "constraint"]
    assert n_shape.islower() and "-" in n_shape
    assert len(head["tool_plan"]) == 3 and all(g["turns"] == [6] for g in head["tool_plan"])


# -- host-side plumbing -----------------------------------------------------------------------------------------
def test_bind_matches_a_planned_bridge_call_to_the_unwrapped_dispatch():
    todos = {"todos": [{"id": "1", "content": "x", "status": "pending"}], "merge": False}
    planned = {"id": "call_T03_0_0", "name": "tool_call", "args": {"name": "todo_list", "arguments": todos}}
    att = {"tag": "T03", "ended": True, "end": {}, "tool_issues": [planned],
           "tool_seen": [{"id": "call_T03_0_0", "sha": "s"}],
           "tool_dispatch": [{"id": "call_T03_0_0", "name": "todo_list", "args": todos, "ok": True, "chars": 10}]}
    out = TC.bind([att])
    assert out["gaps"] == [] and out["failures"] == []
    assert ("chat", "call", "call_T03_0_0", "tool_call", TC.canon(planned["args"])) in out["expected"]
    batch = {**planned, "args": {"calls": [{"name": "todo_list", "arguments": json.dumps(todos)}]}}
    assert TC.bind([{**att, "tool_issues": [batch]}])["failures"] == []
    wrong = {**att, "tool_dispatch": [{**att["tool_dispatch"][0], "args": {"todos": []}}]}
    assert TC.bind([wrong])["failures"] == ["T03: call_T03_0_0 planned tool_call but the host ran todo_list"]


def load_observer(tmp_path, monkeypatch):
    from bench.instruments.reliability import probe
    spec = importlib.util.spec_from_file_location("_probe_observer", PC.OBSERVER / "rel_observer.py")
    obs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obs)
    monkeypatch.setattr(obs, "DIR", tmp_path)
    monkeypatch.setattr(obs, "_r1", probe)
    return obs


def test_observer_records_the_inline_todo_dispatch_with_its_call_id(tmp_path, monkeypatch):
    obs = load_observer(tmp_path, monkeypatch)
    seen = []
    mod = types.SimpleNamespace(INLINE_TOOL_EXECUTORS={
        "todo_list": lambda agent, args, ctx: seen.append(args) or json.dumps({"todos": args["todos"]}),
        "memory": lambda agent, args, ctx: "untraced"})
    memory = mod.INLINE_TOOL_EXECUTORS["memory"]
    assert obs.PATCHES["agent.inline_tool_executors"] is obs.patch_inline_tools
    obs.patch_inline_tools(mod)
    assert mod.INLINE_TOOL_EXECUTORS["memory"] is memory
    ctx = types.SimpleNamespace(tool_call_id="call_T03_0_0")
    out = mod.INLINE_TOOL_EXECUTORS["todo_list"](object(), {"todos": [{"id": "1"}]}, ctx)
    assert json.loads(out) == {"todos": [{"id": "1"}]} and seen == [{"todos": [{"id": "1"}]}]
    [ev] = PC.read_jsonl(tmp_path / "transcript.jsonl")
    assert (ev["event"], ev["id"], ev["name"], ev["via"], ev["ok"]) == (
        "tool_dispatch", "call_T03_0_0", "todo_list", "inline_tool_dispatch", True)
    obs.patch_inline_tools(types.SimpleNamespace())  # an older host without the table: no-op


def test_soul_written_before_start_and_rewritten_once_mid_run(tmp_path, monkeypatch):
    cell = probe_cell("probe-host-instruction")
    soul = cell["continuity"]["soul"]
    plugin = {"ref": "HEAD", "sha": "p" * 40, "tree": str(tmp_path), "dir": "fixture", "enabled": "fixture",
              "engine": "fixture"}
    host = {"src": str(tmp_path), "python": sys.executable, "sha": "h" * 40}
    seen = {}

    def phase(run, first):
        seen["start"] = (run.home / "SOUL.md").read_text()
        seen["context"] = run.continuity_context()
        PC.append(run.d / "observer.jsonl", {"phase": "A", "ts": 1.0, "kind": "compaction_committed", "turn": 9})
        PC.append(run.d / "observer.jsonl", {"phase": "A", "ts": 2.0, "kind": "compaction_committed", "turn": 15})
        for t in (10, 15, 16, 16, 17):  # 10 follows a commit but precedes from_turn; 16 is the first due turn
            run.soul_rewrite(t)
            seen[t] = (run.home / "SOUL.md").read_text()
        return {"exit": "done", "next_turn": None}

    monkeypatch.setattr(PC.ProcessCell, "run_phase", phase)
    monkeypatch.setattr(RM, "copy_dbs", lambda *a: [])
    monkeypatch.setattr(RM, "verdict_fields", lambda *a: {"verdict": "PASS", "failed_bars": {},
                                                       "applicable_bars": [], "numbers": {}})
    (tmp_path / "scratch").mkdir()
    rec = PC.run_cell_process(cell, "fixture", host, plugin, tmp_path / "out", 5, False,
                              identity={"method": "fixture"}, scratch_root=tmp_path / "scratch")
    assert soul["nonce"] in seen["start"] and soul["rewrite"]["nonce"] not in seen["start"]
    assert seen[10] == seen[15] == seen["start"] and soul["rewrite"]["nonce"] in seen[16] and seen[17] == seen[16]
    assert set(seen["context"]["nonces"]) == {s["nonce"] for s in cell["continuity"]["probes"]}
    notes = [n for n in PC.read_jsonl(Path(rec["dir"]) / "observer.jsonl") if n["kind"] == "probe_rewrite"]
    assert [n["turn"] for n in notes] == [16] and isinstance(notes[0]["ts"], float)
    assert "probes" in rec["continuity"] and "probes_recorded" in rec
