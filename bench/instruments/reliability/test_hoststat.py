"""Host receipt integration without sockets, subprocesses or a live Hermes host."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from bench.instruments.reliability import cells, process_cell as PC
from bench.instruments.reliability.scorers import bars
from tests import test_reliability_scorers as fixture


def test_process_cell_host_receipts(tmp_path, monkeypatch):
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
    monkeypatch.setattr(PC.RM, "verdict_fields", lambda *a: {"verdict": "PASS"})
    monkeypatch.setattr(PC.RM, "release_scratch", lambda *a: None)
    host = {"sha": "host", "src": str(tmp_path), "python": sys.executable}
    plugin = {"sha": "a" * 40, "ref": "fixture", "tree": str(tree), "dir": "fixture", "engine": "fixture", "enabled": "fixture"}
    rec = PC.run_cell_process(cell, "fixture", host, plugin, tmp_path / "out", 5, True,
                              identity={"method": "fixture"}, scratch_root=scratch)
    assert rec["verdict"] == "PASS"
    assert sequence == ["snapshot", "phase", "stop", "snapshot"]
    directory = Path(rec["dir"])
    assert rec["host"] == "fixture"
    assert json.loads((directory / "cell.json").read_text())["host"] == "fixture"
    assert json.loads((directory / "cell.json").read_text())["host_stats"] == rec["host_stats"]
    assert json.loads((directory / "verdict.json").read_text())["host_stats"] == rec["host_stats"]
    assert rec["host_stats"]["elapsed_s"] == 3


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
