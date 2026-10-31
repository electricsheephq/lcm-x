"""Offline contracts for the opt-in C1 throttle and #659 event probes."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent / "track_s" / "s2"))
from s2lib import c1_budget, deep  # noqa: E402


def test_throttle_counts_and_three_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("C1_BUDGET_ROOT", str(tmp_path))
    monkeypatch.setattr(c1_budget, "rate_limits", lambda: dict(weekly_used=56, windows=[]))
    monkeypatch.setattr(c1_budget.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(c1_budget.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="provider blocked"))
    for _ in range(2):
        assert c1_budget.run(["fixture"]).returncode == 1
    with pytest.raises(SystemExit, match="three consecutive"):
        c1_budget.run(["fixture"])
    state = json.loads((tmp_path / "counters.json").read_text())
    assert state["calls"] == 3 and state["terminal"]
    with pytest.raises(SystemExit):
        c1_budget.run(["fixture"])
    assert json.loads((tmp_path / "counters.json").read_text())["calls"] == 3


def test_429_and_weekly_pause(tmp_path, monkeypatch):
    monkeypatch.setenv("C1_BUDGET_ROOT", str(tmp_path))
    monkeypatch.setattr(c1_budget, "rate_limits", lambda: dict(weekly_used=85, windows=[]))
    monkeypatch.setattr(c1_budget.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    pauses = []
    def pause(state, path):
        if "pause_until" in state:
            pauses.append(state.pop("pause_until"))
    monkeypatch.setattr(c1_budget, "_pause", pause)
    monkeypatch.setattr(c1_budget.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="HTTP 429"))
    c1_budget.run(["fixture"])
    state = json.loads((tmp_path / "counters.json").read_text())
    assert len(pauses) == 1 and state["pause_until"] > c1_budget.time.time() + 590
    assert state["outcomes"][-1]["limited"]
    state["calls"], state["pause_until"] = 50, 0
    (tmp_path / "counters.json").write_text(json.dumps(state))
    c1_budget.run(["fixture"])
    state = json.loads((tmp_path / "counters.json").read_text())
    assert [r["call"] for r in state["rate_reads"]] == [0, 50]


def test_event_probe_requires_exact_quote_and_obedience(tmp_path, monkeypatch):
    specs = [dict(id=k, kind=k, text="fixture question", answer=k) for k in
             ("current_request", "constraint_quote", "constraint_obey", "host_instruction")]
    monkeypatch.setattr(deep.R, "SolLowReader", lambda _: SimpleNamespace(readback=dict(model="gpt-6.1-sol", effort="low")))
    answers = {p["id"]: p["answer"] for p in specs}
    answers["constraint_obey"] = "wrong action"
    monkeypatch.setattr(deep.R, "answer", lambda *a, **k: (answers, {}))
    run = SimpleNamespace(man=dict(material_version="track-s-deep-v1", continuity_probes=specs),
                          dir=tmp_path, system=" ".join(p["answer"] for p in specs),
                          m=SimpleNamespace(tokens=SimpleNamespace(count_tokens=len)))
    event = dict(event=0, row_index=200, is_compaction=True)
    deep.event_probes(run, event, [])
    assert [p["passed"] for p in event["deep_probes"]] == [True, True, False, True]
    assert (tmp_path / "deep-probes" / "event-0.json").exists()
