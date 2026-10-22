"""The diagnostic cell cannot pass on a tool error or deferred catch-up alone."""
import copy
import json
from types import SimpleNamespace

import pytest

from bench.instruments.reliability import ci, process_cell as PC, stop_turn as ST


def test_stop_cell_is_nightly_non_gate_and_process_only():
    cell = next(c for c in PC.R2_CELLS if c.get("stop_after_tool"))
    assert cell["id"] == "stop-after-tool/no-retry/in-place"
    assert cell["id"] in ci.NON_GATE and cell["id"] in ci.expected_cells("acp-process")
    assert cell["id"] not in ci.expected_cells(None)
    assert PC.unsupported(cell, "acp-process") is None
    assert not ci.in_gate_set(cell["id"])
    assert cell["turns"] == 2 and not cell["final_compaction_check"]


def test_missing_sandbox_exec_uses_existing_process_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(PC, "SANDBOX_EXEC", str(tmp_path / "absent-sandbox-exec"))
    run = SimpleNamespace(cell={"stop_after_tool": True}, host={"python": str(tmp_path / "venv/bin/python")},
                          scratch=tmp_path / "scratch", d=tmp_path / "out")
    assert PC.ProcessCell.argv(run) == [str(tmp_path / "venv/bin/hermes"), "acp"]


@pytest.fixture
def measured(tmp_path):
    rows = ST.shape([{"role": "user", "content": "synthetic"},
                     {"role": "assistant", "content": "", "tool_calls": [{"id": "call", "function": {
                         "name": "read_file", "arguments": '{"path":"fixture"}'}}]},
                     {"role": "tool", "content": "successful read", "tool_call_id": "call"},
                     {"role": "assistant", "content": "interrupted"}])
    for k in ("immediate", "end", "catchup"):
        (tmp_path / f"stop-{k}.json").write_text(json.dumps({
            db: {"rows": rows, "owners_match": True} for db in ("lcm", "state")}))
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(json.dumps({"event": "tool_dispatch", "ok": True}) + "\n")
    (tmp_path / "provider-requests.jsonl").write_text(json.dumps({"role": "main", "turn": 1, "step": 0}) + "\n")
    notes = {"turn_end": [{"turn_kind": "cancel", "interrupted": True, "reason": "interrupted_by_user"}],
             "stop_tool_ready": [{}], "stop_interrupt_observed": [{"interrupted": True}]}
    return SimpleNamespace(d=tmp_path, transcript=transcript, notes=lambda k: notes[k]), rows, notes


@pytest.mark.parametrize("damage", [None, "missing-immediate", "duplicate", "order", "arguments", "owner", "tool-error", "not-interrupted", "second-request"])
def test_stop_scorer_rejects_deferred_rows_and_metadata_corruption(measured, damage):
    run, rows, notes = measured
    if damage == "tool-error":
        run.transcript.write_text(json.dumps({"event": "tool_dispatch", "ok": False}) + "\n")
    elif damage == "not-interrupted":
        notes["turn_end"][0]["reason"] = "completed"
    elif damage == "second-request":
        with (run.d / "provider-requests.jsonl").open("a") as f:
            f.write(json.dumps({"role": "main", "turn": 1, "step": 1}) + "\n")
    elif damage:
        phase = "catchup" if damage in ("duplicate", "order") else "immediate"
        path = run.d / f"stop-{phase}.json"
        data = json.loads(path.read_text())
        stored = copy.deepcopy(rows)
        if damage == "missing-immediate":
            stored = []
        elif damage == "duplicate":
            stored += rows
        elif damage == "order":
            stored.reverse()
        elif damage == "arguments":
            stored[1]["calls"][0]["arguments_sha256"] = "wrong"
        elif damage == "owner":
            data["lcm"]["owners_match"] = False
        data["lcm"]["rows"] = stored
        path.write_text(json.dumps(data))
    result = ST.score(run, {"exit": "done"})
    assert result["verdict"] == ("PASS" if damage is None else "ERROR" if damage in (
        "tool-error", "not-interrupted", "second-request") else "FAIL")


@pytest.mark.parametrize("damage", [False, True], ids=["preserved-prefix", "changed-prefix"])
def test_warm_stop_cell_checks_prefix_and_scores_only_stopped_rows(measured, damage):
    run, rows, _ = measured
    prefix = ST.shape([{"role": "user", "content": "warm question"},
                       {"role": "assistant", "content": "warm answer"}])
    warm = {db: {"rows": prefix, "owners_match": True} for db in ("lcm", "state")}
    (run.d / "stop-warmup.json").write_text(json.dumps(warm))
    for phase in ("immediate", "end", "catchup"):
        path = run.d / f"stop-{phase}.json"
        data = json.loads(path.read_text())
        for db in ("lcm", "state"):
            data[db]["rows"] = copy.deepcopy(prefix + rows)
        if damage and phase == "end":
            data["lcm"]["rows"][0]["content_sha256"] = "changed"
        path.write_text(json.dumps(data))
    result = ST.score(run, {"exit": "done"})
    assert result["verdict"] == ("FAIL" if damage else "PASS")
    assert result["stop_checks"]["warm_prefix"] is not damage
    assert result["stopped_rows"] == result["immediate_rows"] == len(rows)
