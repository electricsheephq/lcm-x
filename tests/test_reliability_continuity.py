"""Continuity controls use fixture logs only; never launch a Hermes host."""
from __future__ import annotations

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
    assert len(selected) == 6
    assert not any(c["id"].startswith("continuity/") for c in cells.registry())
    for c in selected:
        assert PC.unsupported(c, "acp-process") is None
        assert c["id"] in ci.NON_GATE and not ci.in_gate_set(c["id"])
        assert c["id"] in ci.expected_cells("acp-process")
        if c["continuity"].get("sole_user"):
            assert c["turns"] == 1 and len(c["tool_plan"]) == 24
            assert all(g["turns"] == [1] for g in c["tool_plan"])
        if c["continuity"].get("require_survival_fit"):
            assert c["lcm_env"]["LCM_SURVIVAL_RESERVE"] == "0.9"
            assert c["user_text"]["repeat_from"] == {"9": 100}


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
