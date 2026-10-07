"""Continuity controls use fixture logs only; never launch a Hermes host."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.instruments.reliability import cells, ci, fake_provider as FP, process_cell as PC, report
from bench.instruments.reliability.scorers import continuity as CT


def request(rid=1, ts=4, *, note=True, previous=True, current=True):
    messages = [{"role": "system", "content": CT.NOTE if note else "ordinary system"},
                {"role": "user", "content": "[T01] user turn 1: fixture"}]
    if previous:
        messages.append({"role": "assistant", "content": "reply to T01: completed fixture"})
    if current:
        messages.append({"role": "user", "content": "[T02] user turn 2: fixture"})
    return {"rid": rid, "ts": ts, "role": "main", "turn": 2, "current_user_tag": "T02",
            "continuity": CT.markers(messages, anchor="T01", previous="T01", current="T02")}


def observations(kind="in-place", *, removed=True):
    return [{"kind": "turn_end", "ts": 1, "turn": 1, "reply_tag": "T01", "failed": False, "interrupted": False},
            {"kind": "compaction_committed", "ts": 3, "phase": "A", "turn": 2, "compaction_kind": kind,
             "compacted_reply_tags": ["T01"] if removed else []}]


@pytest.mark.parametrize("note,previous,current,expected", [
    (True, True, True, (True, True, True)), (False, True, True, (False, True, True)),
    (True, False, True, (True, False, True)), (True, True, False, (True, True, False))])
def test_fixture_positive_and_negative_controls(tmp_path, note, previous, current, expected):
    req = request(note=note, previous=previous, current=current)
    (tmp_path / "provider-requests.jsonl").write_text(json.dumps(req) + "\n")
    (tmp_path / "observer.jsonl").write_text("".join(json.dumps(o) + "\n" for o in observations()))
    result = CT.score_dir({"in_place": True}, tmp_path)
    [row] = result["rows"]
    assert tuple(row[f] for f in ("F1", "F3", "F4")) == expected
    assert row["previous_reply_in_compacted_span"] is True
    for f, present in zip(("F1", "F3", "F4"), expected):
        assert result["totals"][f]["present" if present else "absent"] == 1


def test_no_following_main_request_is_unknown_never_true():
    # A request sent before the commit and its later held/closed log entry
    # cannot stand in for a request received after the commit.
    early = request(ts=2)
    reqs = [early, {**early, "ts": 5, "phase": "client_closed"}, {**request(ts=4), "role": "lcm-summary"}]
    result = CT.score({"in_place": True}, reqs, observations())
    [row] = result["rows"]
    assert row["status"] == "no request" and result["no_request"] == 1
    assert row["F1"] is None and row["F3"] is None and row["F4"] is None
    assert all(t["present"] == 0 for t in result["totals"].values())


def test_first_main_request_in_tool_loop_and_across_restart_not_next_turn():
    obs = observations("rotation", removed=False)
    obs += [{"kind": "compaction_committed", "ts": 4.5, "phase": "B", "turn": 2,
             "compaction_kind": "survival fit", "survival_fit": True, "compacted_reply_tags": ["T01"]}]
    reqs = [request(3, 6), {**request(0, 3.5), "role": "aux"}, request(2, 5), request(1, 4)]
    result = CT.score({"in_place": False}, reqs, obs)
    assert [r["request_id"] for r in result["rows"]] == [1, 2]
    assert [r["previous_reply_in_compacted_span"] for r in result["rows"]] == [False, True]
    assert result["survival_fits"] == 1


def test_only_completed_successful_replies_before_commit_count():
    obs = observations() + [{"kind": "turn_end", "ts": 2, "reply_tag": "T99", "failed": True},
                            {"kind": "turn_end", "ts": 2.1, "reply_tag": "T98", "interrupted": True},
                            {"kind": "turn_end", "ts": 5, "reply_tag": "T02"}]
    assert CT.score({}, [request()], obs)["rows"][0]["previous_reply_tag"] == "T01"
    # The one-user tool loop has no prior completed reply at all.
    result = CT.score({"continuity": {"sole_user": True}}, [request()], obs[1:2])
    assert result["rows"][0]["F2"] is True and result["rows"][0]["F3"] is None
    assert result["totals"]["F3"]["not_applicable"] == 1


def test_old_logs_and_missing_markers_are_unknown():
    assert CT.score({}, [{"rid": 1, "role": "main"}], observations())["no_request"] == 1
    result = CT.score({}, [{"rid": 1, "role": "main", "ts": 4}], observations())
    assert result["rows"][0]["status"] == "no markers" and result["rows"][0]["F1"] is None


def test_marker_metadata_has_no_message_text_and_handles_anthropic_system_blocks():
    messages = FP.normalize({"system": [{"type": "text", "text": CT.NOTE}], "messages": [
        {"role": "user", "content": [{"type": "text", "text": "[T01] user turn 1: private fixture"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "reply to T01: private fixture"}]}]}, "anthropic")
    data = CT.markers(messages, anchor="T01", previous="T01", current="T01")
    assert all(data[f] is True for f in CT.CHECKS)
    assert data["user_tags"] == data["reply_tags"] == ["T01"]
    assert data["role_counts"] == {"system": 1, "user": 1, "assistant": 1}
    assert len(data["message_roles"]) == 3 and "private" not in json.dumps(data) and CT.NOTE not in json.dumps(data)
    assert not CT.markers([{"role": "user", "content": CT.NOTE}])["F1"]


def test_new_cells_are_r2_only_non_gating_and_express_existing_knobs():
    selected = cells.select("continuity/*", extra=PC.R2_CELLS)
    assert len(selected) == 8
    assert not any(c["id"].startswith("continuity/") for c in cells.registry())
    for c in selected:
        assert PC.unsupported(c, "acp-process") is None
        assert c["id"] in ci.NON_GATE and not ci.in_gate_set(c["id"])
        assert c["id"] in ci.expected_cells("acp-process")
        if c["continuity"].get("sole_user"):
            assert c["turns"] == 1 and len(c["tool_plan"]) == 12
            assert all(g["turns"] == [1] for g in c["tool_plan"])
        if c["id"].startswith("continuity/survival-fit/"):
            assert c["targets"] == [916] and cells.ISSUES[916][0] == ("F4",)
            assert c["lcm_env"]["LCM_SURVIVAL_RESERVE"] == "0.9"
            assert c["user_text"]["repeat_from"] == {"9": 100}
        if c["id"].startswith("continuity/survival-fit-older-turns/"):
            assert c["lcm_env"]["LCM_SURVIVAL_RESERVE"] == "0.5"
            assert c["window"] == 128000 and c["user_text"]["repeat_from"] == {"9": 1}
            assert len(PC.P1.user_text(c, "T", 8)) > 30000
            assert len(PC.P1.user_text(c, "T", 9)) < 100


@pytest.mark.parametrize("mode", ["in-place", "rotation"])
@pytest.mark.parametrize("shape", ["survival-fit", "survival-fit-older-turns"])
def test_survival_scenario_excludes_only_b8(shape, mode):
    c = cells.select(f"continuity/{shape}/{mode}", extra=PC.R2_CELLS)[0]
    assert c["continuity"]["require_survival_fit"] is True
    assert c["bars"] == ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9"]
    assert "B8" in cells.select(f"baseline/{mode}/acp")[0]["bars"]


@pytest.mark.parametrize("mode", ["in-place", "rotation"])
def test_sole_user_settings_use_distinct_reads_to_sustain_pressure(mode):
    # Identical reads are deduplicated/blocked before replay stubbing can matter.
    c = cells.select(f"continuity/sole-user-tool-loop/{mode}", extra=PC.R2_CELLS)[0]
    assert c["turns"] == 1 and c["min_compactions"] == 2
    assert c["big_files"] == len(c["tool_plan"]) == 12
    assert c["tool_plan"] == [{"turns": [1], "calls": [{"name": "read_file", "args": {
        "path": f"{{files}}/big-{k:02d}.txt"}, "expect": {"min_chars": 10000}}]} for k in range(1, 13)]
    env = c["lcm_env"]
    assert float(env["LCM_CONTEXT_THRESHOLD"]) * c["window"] == 15360
    assert env["LCM_FRESH_TAIL_COUNT"] == "4" and env["LCM_FRESH_TAIL_MAX_TOKENS"] == "2000"
    assert env["LCM_LEAF_CHUNK_TOKENS"] == "1000"
    assert "B5" in c["bars"] and c["final_compaction_check"] is False


@pytest.mark.parametrize("big_lines", [400, 3])
def test_shared_r1_r2_tool_fixture_writer_creates_distinct_files(tmp_path, big_lines):
    # Both transports call this helper; exercise fixture writing without a host.
    PC.P1.write_tool_files(tmp_path / "files", {"big_files": 12, "big_lines": big_lines})
    files = tmp_path / "files"
    payloads = [(files / f"big-{k:02d}.txt").read_bytes() for k in range(1, 13)]
    assert len(set(payloads)) == 12
    for k, payload in enumerate(payloads, 1):
        assert payload.decode().splitlines() == [
            f"f{k:02d} line {i:05d}: " + PC.P1.FILLER * 8 for i in range(big_lines)]
    assert (files / "big.txt").read_text() == "".join(
        f"line {i:05d}: " + PC.P1.FILLER * 8 + "\n" for i in range(big_lines))
    assert (files / "small.txt").read_text() == "small deterministic file\n"


def test_shared_tool_fixture_writer_defaults_leave_existing_files_unchanged(tmp_path):
    PC.P1.write_tool_files(tmp_path / "files", {})
    files = tmp_path / "files"
    assert sorted(p.name for p in files.iterdir()) == ["big.txt", "small.txt"]
    assert (files / "big.txt").read_text() == "".join(
        f"line {i:05d}: " + PC.P1.FILLER * 8 + "\n" for i in range(400))


def test_continuity_absences_and_survival_bar_failures_do_not_gate():
    rows = [{"cell": cid, "host": "fixture", "transport": "acp-process", "plugin_sha": "fixture",
             "verdict": "PASS", "targets": []} for cid in ci.expected_cells("acp-process")]
    for row in rows:
        if row["cell"] in ci.NON_GATE and row["cell"].startswith("continuity/"):
            row.update(verdict="FAIL", failed_bars={"B8": {"survival_fit": 1}},
                       continuity=CT.score({}, [request(note=False, previous=False, current=False)], observations()))
        if row["cell"].startswith("p8-control/") and not row["cell"].endswith("/none"):
            row.update(verdict="FAIL", failed_bars={"B9": {}})
    assert ci.gate(rows, set()) == []


def test_matrix_section_renders_per_host_counts_and_no_request(tmp_path):
    diagnostic = CT.score({}, [request(note=False, previous=False)], observations() + [
        {"kind": "compaction_committed", "ts": 6, "phase": "A", "turn": 3, "final": True}])
    row = {"host": "fixture-host", "host_sha": "h" * 40, "plugin_ref": "HEAD", "plugin_sha": "p" * 40,
           "cell": "baseline/in-place/acp", "transport": "acp-process", "verdict": "PASS", "continuity": diagnostic}
    (tmp_path / "results.jsonl").write_text(json.dumps(row) + "\n")
    report.write(tmp_path, [row], 1)
    text = (tmp_path / "MATRIX.md").read_text()
    assert "## Continuity (diagnostic, non-gating)" in text and "fixture-host" in text
    assert "0/1/1/0" in text and "no request" in text and "| forced |" in text
    assert "| in-place |" in text and "| False |" in text


def test_untagged_completed_reply_does_not_reuse_an_older_tag():
    obs = observations() + [{"kind": "turn_end", "ts": 2, "reply_tag": None}]
    assert CT.score({}, [request()], obs)["rows"][0]["previous_reply_tag"] is None


@pytest.mark.parametrize("cid", ["continuity/sole-user-tool-loop/in-place", "baseline/in-place/acp"])
def test_tool_result_is_seen_before_missing_current_tag_reply(cid):
    """A missing user tag must not hide tool results already received by the provider."""
    c = cells.select(cid, extra=PC.R2_CELLS)[0]
    events = []
    run = types.SimpleNamespace(cell=c, text_tag="T01", event=lambda **ev: events.append(ev))
    scenario = PC.Scenario(run)
    scenario.begin(1, "chat", "A", 1)
    tool_id, body = "call_T01_4_0", "synthetic tool result"
    scenario.st["issued"].add(tool_id)
    messages = [{"role": "user", "content": "untagged summary"},
                {"role": "tool", "tool_call_id": tool_id, "content": body}]

    reply = scenario.main(messages)

    assert reply["content"] == "ok" and reply["log"]["unexpected"] is True
    assert scenario.st["seen"] == {tool_id} and scenario.st["step"] == 0
    assert events == [{"turn": 1, "event": "tool_seen", "id": tool_id,
                       "sha": hashlib.sha256(body.encode()).hexdigest(), "chars": len(body)}]
    scenario.main(messages)
    assert len(events) == 1  # Replayed results retain the existing once-per-id semantics.


@pytest.mark.parametrize("cid,verdict", [("continuity/survival-fit/in-place", "FAIL"),
                                       ("baseline/in-place/acp", "ERROR")])
@pytest.mark.parametrize("projected", [True, False])
def test_missing_current_user_fixture_preserves_cell_boundary(tmp_path, monkeypatch, cid, verdict, projected):
    """Exercise request logging and final scoring with a fixture phase; no host."""
    from bench.instruments.reliability import run_matrix as RM
    c = cells.select(cid, extra=PC.R2_CELLS)[0]
    plugin = {"ref": "HEAD", "sha": "p" * 40, "tree": str(tmp_path), "dir": "fixture",
              "enabled": "fixture", "engine": "fixture"}
    host = {"src": str(tmp_path), "python": sys.executable, "sha": "h" * 40}
    message = "[LCM survival fit: this user message is stored verbatim]" if projected else "untagged fixture"

    def phase(run, first):
        run.text_tag = "T02"
        run.scenario.begin(2, "chat", "A", first)
        for o in observations("survival fit"):
            PC.append(run.d / "observer.jsonl", {"phase": "A", **o})
        # A tag in an older row must not hide its absence in the LAST user row.
        messages = [{"role": "user", "content": "[T02] older summary mention"},
                    {"role": "assistant", "content": "reply to T01: completed fixture"},
                    {"role": "user", "content": message}]
        handler = types.SimpleNamespace(path="/v1/chat/completions")
        run.provider.serve(handler, {"model": "rel/main", "messages": messages}, "openai")
        return {"exit": "done", "next_turn": None}

    monkeypatch.setattr(PC.ProcessCell, "run_phase", phase)
    monkeypatch.setattr(PC.FP.FakeProvider, "json_reply", lambda *a: None)
    monkeypatch.setattr(RM, "copy_dbs", lambda *a: [])
    monkeypatch.setattr(RM, "verdict_fields", lambda *a: {"verdict": "PASS", "failed_bars": {},
                                                       "applicable_bars": [], "numbers": {}})
    (tmp_path / "scratch").mkdir()
    rec = PC.run_cell_process(c, "fixture", host, plugin, tmp_path / "out", 5, False,
                               identity={"method": "fixture"}, scratch_root=tmp_path / "scratch")
    assert rec["verdict"] == verdict
    req = PC.read_jsonl(Path(rec["dir"]) / "provider-requests.jsonl")[0]
    assert req["continuity"]["current_user_projected"] is projected
    if verdict == "ERROR":
        assert rec["reason"].startswith("unexpected main-model requests:")
        assert req["continuity"]["F4"] is True  # existing non-continuity semantics
        return
    assert rec["targets"] == [916] and set(rec["failed_bars"]) == {"F4"}
    assert "current user tag missing" in rec["failed_bars"]["F4"]["reason"]
    assert req["continuity"]["F4"] is False and "last_user" not in req
    [row] = rec["continuity"]["rows"]
    assert row["F4"] is False and row["current_user_projected"] is projected
    assert rec["continuity"]["totals"]["F4"]["absent"] == 1
    rows = [{"cell": cid, "host": "fixture", "transport": "acp-process", "plugin_sha": "p" * 40,
             "verdict": "PASS", "targets": []} for cid in ci.expected_cells("acp-process")]
    for r in rows:
        if r["cell"].startswith("p8-control/") and not r["cell"].endswith("/none"):
            r.update(verdict="FAIL", failed_bars={"B9": {}})
    rows = [rec if r["cell"] == rec["cell"] else r for r in rows]
    assert ci.gate(rows, {916}) == []
    # The open-target rule also recognizes F4 if this diagnostic were gated.
    original = ci.in_gate_set
    monkeypatch.setattr(ci, "in_gate_set", lambda cid: cid == rec["cell"] or original(cid))
    assert ci.gate(rows, {916}) == []
    assert any("uncovered bars ['F4']" in p for p in ci.gate(rows, set()))
    report.write(Path(rec["dir"]), [rec], 1)
    assert "916 | F4 | `continuity/survival-fit/in-place`" in (Path(rec["dir"]) / "ISSUE-MAP.md").read_text()


@pytest.mark.parametrize("mode", ["in-place", "rotation"])
def test_older_turn_survival_fixture_keeps_current_and_measures_previous_reply(mode):
    c = cells.select(f"continuity/survival-fit-older-turns/{mode}", extra=PC.R2_CELLS)[0]
    result = CT.score(c, [request(previous=False)], observations("survival fit"))
    [row] = result["rows"]
    assert row["kind"] == "survival fit" and row["F4"] is True
    assert row["F3"] is False and row["current_user_projected"] is False
    assert result["scenario_observed"] is True


def test_optional_fault_reuses_compress_then_sends_an_ordinary_turn(tmp_path, monkeypatch):
    c = next(c for c in PC.R2_CELLS if c["id"] == "continuity/forced-compaction-then-turn/in-place")
    sequence = []

    class Peer:
        killed, sent_term = False, False

        def __init__(self, *a):
            pass

        def initialize(self, *a):
            pass

        def new_session(self, *a):
            return "fixture-session"

        def close(self):
            return 0
    monkeypatch.setattr(PC.AD, "AcpProcess", Peer)
    run = PC.ProcessCell(c, tmp_path, {"src": str(tmp_path), "python": sys.executable}, "acp-process", 10)

    def turn(t):
        sequence.append(t)
        run.last_turn = t

    def final():
        sequence.append("/compress")
        return {"published": True, "outcome": "published"}
    run.turn, run.final_check = turn, final
    try:
        assert run.run_phase(1)["exit"] == "done"
        assert sequence == [*range(1, 20), "/compress", 20]
        assert PC.read_jsonl(tmp_path / "faults-fired.jsonl")[0]["kind"] == "forced_compaction_then_turn"
    finally:
        run.proxy.stop()
        run.provider.server.server_close()


@pytest.mark.parametrize("status,fitted,kind", [("compacted", False, "compaction_committed"),
                                               ("noop", True, "survival_fit_committed")])
def test_observer_records_removed_reply_and_preserves_crash_trigger(tmp_path, monkeypatch, status, fitted, kind):
    import importlib.util
    from bench.instruments.reliability import probe
    spec = importlib.util.spec_from_file_location("_continuity_observer", PC.OBSERVER / "rel_observer.py")
    obs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obs)
    monkeypatch.setattr(obs, "DIR", tmp_path)
    monkeypatch.setattr(obs, "_r1", probe)
    monkeypatch.setattr(obs, "store_cover", lambda: (1, 1))
    monkeypatch.setattr(obs, "depth0", lambda: 1)
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.context_compressor", types.ModuleType("agent.context_compressor"))
    (tmp_path / "cell.json").write_text('{"in_place": true}')

    class Engine:
        def compress(self, messages):
            self._last_compression_status = status
            if fitted:
                obs.logstate["survival_fits"] += 1
                self._last_survival_fit = {"reason": "fixture"}
            return messages[-1:]

        def handle_tool_call(self, *a, **kw):
            return "{}"
    engine = Engine()
    obs.patch_engine(types.SimpleNamespace(context_compressor=engine))
    rows = [{"role": "assistant", "content": "reply to T01: prior"},
            {"role": "user", "content": "[T02] user turn 2: current"}]
    assert engine.compress(rows) == rows[-1:]
    [commit] = [n for n in PC.read_jsonl(tmp_path / "observer.jsonl") if n["kind"] == kind]
    assert commit["compacted_reply_tags"] == ["T01"]
    assert commit["compaction_kind"] == ("survival fit" if fitted else "in-place")
    assert obs.counters["compacted_turns"] == ([] if fitted else [0])


@pytest.mark.parametrize("verdict, observed, f4, unexpected, applicable, final", [
    ("UNSUPPORTED", True, None, [{"current_user_missing": True}], True, "FAIL"),  # a real F4 miss is never masked
    ("PASS", False, True, [], False, "PASS"),  # the required scenario never happened: F4 not covered
    ("PASS", True, True, [], True, "PASS"),
    ("PASS", None, True, [], True, "PASS"),  # cells without a scenario requirement keep today's rule
    ("PASS", True, None, [], False, "PASS"),  # scenario seen but no request after it: no request is never a pass
    ("UNSUPPORTED", True, True, [], False, "UNSUPPORTED"),
])
def test_f4_applies_only_to_an_observed_scenario_and_never_masks_a_miss(verdict, observed, f4, unexpected,
                                                                        applicable, final):
    rec = {"verdict": verdict, "continuity": {"scenario_observed": observed, "rows": [{"F4": f4, "status": "x"}]}}
    PC.apply_f4(rec, {"id": "continuity/survival-fit/in-place"}, unexpected)
    assert ("F4" in rec.get("applicable_bars", [])) is applicable
    assert rec["verdict"] == final
    assert ("F4" in rec.get("failed_bars", {})) is bool(unexpected)


@pytest.mark.parametrize("values,missing,status", [
    ([], False, "NOT COVERED on F4"),
    ([True, None], False, "NOT COVERED on F4"),
    ([None, True], False, "NOT COVERED on F4"),
    ([True, True], False, "target cells PASS on this bar"),
    ([True, None], True, "target cell FAILS"),
])
def test_f4_requires_every_continuity_row_measured(values, missing, status):
    cell = cells.select("continuity/survival-fit/in-place", extra=PC.R2_CELLS)[0]
    rec = {"cell": cell["id"], "verdict": "PASS", "continuity": {
        "scenario_observed": True, "rows": [{"F4": value} for value in values]}}
    unexpected = [{"current_user_missing": True}] if missing else []
    PC.apply_f4(rec, cell, unexpected)
    assert ("F4" in rec.get("applicable_bars", [])) is (bool(values) and None not in values or missing)
    assert rec["verdict"] == ("FAIL" if missing else "PASS")
    assert ("F4" in rec.get("failed_bars", {})) is missing
    assert report.issue_status([rec], "", cells.ISSUES[916][0])[0].startswith(status)


@pytest.mark.parametrize("after_fit", [False, True])
def test_f4_ordinary_compaction_cannot_cover_unmeasured_survival_fit(after_fit):
    cell = cells.select("continuity/survival-fit/in-place", extra=PC.R2_CELLS)[0]
    obs = observations() + [{"kind": "survival_fit_committed", "ts": 5, "phase": "A", "turn": 2,
                             "compaction_kind": "survival fit", "survival_fit": True}]
    reqs = [request()] + ([request(2, 6)] if after_fit else [])
    scored = CT.score(cell, reqs, obs)
    assert scored["scenario_observed"] is True
    assert [row["F4"] for row in scored["rows"]] == [True, True if after_fit else None]
    assert scored["no_request"] == (0 if after_fit else 1)
    rec = {"cell": cell["id"], "verdict": "PASS", "continuity": scored}
    PC.apply_f4(rec, cell, [])
    assert ("F4" in rec.get("applicable_bars", [])) is after_fit
    expected = "target cells PASS on this bar" if after_fit else "NOT COVERED on F4"
    assert report.issue_status([rec], "", cells.ISSUES[916][0])[0].startswith(expected)
