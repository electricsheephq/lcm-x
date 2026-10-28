"""Regression for PR 1689 legacy_h."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from .bot1_support import S2, function


def test_legacy_h_checkpoint_without_deep_receipts(tmp_path, monkeypatch):
    out = tmp_path / "runs/H/seed-1/1"
    cp = dict(id=1, row_index=1)
    class Run:
        def __init__(self, *args):
            self.snaps, self.events = [], [dict(event=0)]
            self.engine = None
            self.system, self.reader_calls, self.receipts_out = "", [], []
        def replay(self):
            directory = out / "cp-1"
            directory.mkdir()
            self.snaps.append(dict(dir=directory, row=1, view=[], n_events=1, checkpoint=cp))
        def ntok(self, _):
            return 1
        sysmsg = {}
    s2 = SimpleNamespace(DEFAULT_CTX=100, MATERIAL=tmp_path, RUNS=tmp_path / "runs", TS=tmp_path,
        CP=SimpleNamespace(select=lambda *a: [cp], late_checkpoint=lambda *a: 1), jload=lambda *a: dict(decision_checkpoint=dict(tokens=1)),
        jlines=lambda *a: [], replay_timeout=lambda *a: 1, sha=lambda *a: "material",
        probe=lambda *a: [], continuity=lambda *a: [], deep=SimpleNamespace(enabled=lambda _: False), S2=tmp_path / "source")
    s2.S2.mkdir()
    source = tmp_path / "host"
    source.mkdir()
    original_open = Path.open
    def deny_probe(path, mode="r", *args, **kwargs):
        if mode == "x":
            raise PermissionError()
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", deny_probe)
    monkeypatch.setattr(sys, "argv", ["runner", "--seed", "1", "--reader", "fake"])
    namespace = dict(argparse=__import__("argparse"), S2=s2, src=source, HOST_PROOF={}, HOST=dict(sha="host"),
        arms=SimpleNamespace(resolve=lambda _: {}), seam=SimpleNamespace(set_lane=lambda *a: None, call_llm=lambda **k: None),
        CC=SimpleNamespace(), estimate_messages_tokens_rough=len, estimate_tokens_rough=len, SimpleNamespace=SimpleNamespace,
        Run=Run, Engine=lambda *a: SimpleNamespace(real=SimpleNamespace(threshold_percent=.75, tail_mode="test"), threshold_tokens=75),
        json=json, reader=None)
    sys.path.insert(0, str(S2))
    assert function(S2 / "run_s_hermes_builtin.py", "main", namespace)() == 0
    assert json.loads((out / "cp-1/summary.json").read_text())["status"] == "DONE"
