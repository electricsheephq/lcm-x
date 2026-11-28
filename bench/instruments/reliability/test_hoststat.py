"""Host receipt integration without sockets, subprocesses or a live Hermes host."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from bench.instruments.reliability import cells, process_cell as PC
from bench.instruments.reliability.scorers import bars
from tests import test_reliability_scorers as fixture


@pytest.fixture
def host_receipt_run(tmp_path, monkeypatch):
    cell = cells.select("baseline/in-place/acp")[0]
    tree = tmp_path / "tree"
    tree.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sequence = []
    def snapshot():
        sequence.append("snapshot")
        return {"monotonic": len(sequence), "loadavg": [1, 2, 3], "counters": None, "reason": "fixture"}
    def run_phase(first):
        sequence.append("phase")
        receipt_before.append(receipt.read_bytes())
        return {"exit": "done", "next_turn": None}
    def stop():
        sequence.append("stop")
    def process(*args, **kwargs):
        return SimpleNamespace(provider=SimpleNamespace(start=lambda: None, stop=stop, base_url="http://localhost"),
            proxy=SimpleNamespace(stop=lambda: None), sid="S0", run_phase=run_phase,
            phase_record=lambda *a: {"citations": {}}, scenario=SimpleNamespace(unexpected=[]), fired=set(), reply_fault=None)
    monkeypatch.setattr(PC, "ProcessCell", process)
    monkeypatch.setattr(PC.hoststat, "snapshot", snapshot)
    monkeypatch.setattr(PC, "accounting", lambda *a: {"unexpected_requests": [], "ok": True})
    monkeypatch.setattr(PC.RM, "copy_dbs", lambda *a: [])
    monkeypatch.setattr(PC.RM, "release_scratch", lambda *a: None)
    host = {"sha": "host", "src": str(tmp_path), "python": sys.executable}
    plugin = {"sha": "a" * 40, "ref": "fixture", "tree": str(tree), "dir": "fixture", "engine": "fixture", "enabled": "fixture"}
    receipt = (tmp_path / "out" / "cells" / "fixture" / plugin["sha"][:12] /
               f"{PC.RM.slug(cell['id'])}@acp-process" / "cell.json")
    receipt_before = []
    def run(verdict="PASS"):
        monkeypatch.setattr(PC.RM, "verdict_fields", lambda *a: {"verdict": verdict})
        return PC.run_cell_process(cell, "fixture", host, plugin, tmp_path / "out", 5, True,
                                   identity={"method": "fixture"}, scratch_root=scratch)
    return run, sequence, receipt_before


def test_process_cell_host_receipts(host_receipt_run):
    run, sequence, _ = host_receipt_run
    rec = run()
    assert rec["verdict"] == "PASS"
    assert sequence == ["snapshot", "phase", "stop", "snapshot"]
    directory = Path(rec["dir"])
    assert rec["host"] == "fixture"
    assert json.loads((directory / "cell.json").read_text())["host"] == "fixture"
    assert json.loads((directory / "cell.json").read_text())["host_stats"] == rec["host_stats"]
    assert json.loads((directory / "verdict.json").read_text())["host_stats"] == rec["host_stats"]
    assert rec["host_stats"]["elapsed_s"] == 3


def test_process_cell_receipt_rewrite_keeps_mode(host_receipt_run, monkeypatch, tmp_path):
    run, _, _ = host_receipt_run
    snapshot = PC.hoststat.snapshot
    def snapshot_after_phase():
        for receipt in (tmp_path / "out").rglob("cell.json"):
            receipt.chmod(0o640)  # a mode no default umask gives: the rewrite must keep it
        return snapshot()
    monkeypatch.setattr(PC.hoststat, "snapshot", snapshot_after_phase)
    rec = run()
    receipt = Path(rec["dir"]) / "cell.json"
    assert "host_stats" in json.loads(receipt.read_text())
    assert receipt.stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize("failure", ["write", "replace"])
@pytest.mark.parametrize("verdict", ["PASS", "FAIL"])
def test_process_cell_receipt_rewrite_failure_preserves_receipt_and_status(host_receipt_run, monkeypatch, caplog,
                                                                         failure, verdict):
    run, _, receipt_before = host_receipt_run
    original_open = io.open
    class FailingWrite:
        def __init__(self, stream):
            self.stream = stream
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def write(self, text):
            name = Path(self.stream.name).name
            if '"host_stats"' in text and (name == "cell.json" or name.startswith(".cell.json.")):
                self.stream.write(text[:11])
                self.stream.flush()
                raise OSError("injected receipt write failure")
            return self.stream.write(text)
    def failing_open(file, mode="r", *args, **kwargs):
        stream = original_open(file, mode, *args, **kwargs)
        return FailingWrite(stream) if mode == "w" else stream
    def fail_replace(source, destination):
        assert Path(source).parent == Path(destination).parent
        raise OSError("injected receipt replace failure")
    if failure == "write":
        monkeypatch.setattr(io, "open", failing_open)
    else:
        monkeypatch.setattr(PC.os, "replace", fail_replace)
    rec = run(verdict)
    directory = Path(rec["dir"])
    assert (directory / "cell.json").read_bytes() == receipt_before[0]
    assert rec["verdict"] == verdict
    assert json.loads((directory / "verdict.json").read_text())["verdict"] == verdict
    assert f"injected receipt {failure} failure" in caplog.text


@pytest.mark.parametrize("failing", [False, True])
def test_existing_scorer_ignores_host(tmp_path, failing):
    fixture.make(tmp_path, rows=fixture.clean_rows()[:-1] if failing else fixture.clean_rows(),
                 events=fixture.clean_events(), parents={"S0": None}, host=[])
    cell = {"bars": ["B1"], "min_compactions": 2}
    directory = tmp_path / "cell"
    before = bars.score(cell, directory)
    cell["host_stats"] = {"steal_pct": 100, "host_load_peak1m": 1000}
    (directory / "cell.json").write_text(json.dumps(cell))
    assert json.dumps(bars.score(cell, directory)) == json.dumps(before)
