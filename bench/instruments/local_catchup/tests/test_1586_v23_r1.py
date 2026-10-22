"""Fail-closed regressions adapted from the independent v2.3 reviewer probes."""
from argparse import Namespace as NS
import json
import shutil
import sqlite3

import pytest

from bench.instruments.local_catchup import catchup, common, floor, measurement, recall, rescore
from bench.instruments.local_catchup.tests.test_1586_fix2 import window_batches
from bench.instruments.local_catchup.tests.test_1586_v2 import rows


@pytest.fixture
def recall_cell(tmp_path, monkeypatch, fake_store_disk_headroom):
    # probe_c/d/e: a child publishes honest PASS receipts before its parent stops it.
    monkeypatch.setattr(recall, "bootstrap", lambda v, f, seed: [f(v)])
    common.dump(tmp_path/"queries.json", [{}]*200)
    calibration = tmp_path/"0-ON-catching-up"
    calibration.mkdir()
    (calibration/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in window_batches()))
    sqlite3.connect(tmp_path/"baseline.db").close()
    a = NS(scratch=tmp_path, out=tmp_path, model_cache=tmp_path, lcmx_repo=tmp_path,
           hermes_source=tmp_path, smoke=False, label="fixture", max_seconds=1,
           seed=1586, standalone_calibration_s=.572, queries=200, block=0, blocks=1, rows=250000)
    control = NS(killed=True, breach=None)
    monkeypatch.setattr(measurement, "Supervisor", lambda _: NS(
        breach=control.breach, watch=lambda _: None, start=lambda: None,
        close=lambda: None, summary=lambda *a: {}))
    def stop(proc):
        proc.returncode = -9 if control.killed else 0
        return dict(stop_sigterm_at=1, stop_exit_at=61, stop_killed=control.killed,
                    stop_returncode=proc.returncode)
    monkeypatch.setattr(measurement, "stop", stop)
    def launch(*args, **kw):
        root = tmp_path/"0-OFF"
        samples = [dict(r, coverage_start={}, coverage_end={})
                   for r in rows()+rows(query_class="broad_nl")]
        pubs = [dict(start=0, end=10, node_id=i) for i in range(10)]
        for name, values in (("recalls", samples), ("publications", pubs)):
            (root/f"{name}.jsonl").write_text("".join(json.dumps(v)+"\n" for v in values))
        raw = recall.score(samples, pubs, calibration_s=.572)
        raw.update(measurement_valid=True, catchup_activity=False)
        assert raw["state"] == "PASS"
        common.dump(root/"raw-summary.json", raw)
        common.dump(root/"final-audit.json", dict(classes={}))
        proc = NS(pid=1, returncode=None if control.breach else 0)
        proc.poll = lambda: proc.returncode
        return proc
    monkeypatch.setattr(measurement.subprocess, "Popen", launch)
    return a, control


@pytest.mark.parametrize("killed", [True, False])
def test_finalization_keeps_parent_forced_stop(recall_cell, killed):
    a, control = recall_cell
    control.killed = killed
    result = measurement.cell(a, "OFF", 0, "recall")
    recall.finalize_block(a, 0, [dict(block=0, state="OFF", result=result)])
    expected = "INVALID" if killed else "PASS"
    assert result["state"] == expected
    for name in ("raw-summary.json", "summary.json"):
        saved = common.load(a.out/"0-OFF"/name)
        assert saved["state"] == expected
        if killed:
            assert saved["reason"] == "forced_stop_after_grace"


def test_direct_rescore_honours_parent_forced_stop(recall_cell):
    a, _ = recall_cell
    result = measurement.cell(a, "OFF", 0, "recall")
    assert result["state"] == "INVALID"
    assert common.load(a.out/"0-OFF/raw-summary.json")["state"] == "PASS"
    rescored = rescore.v2(a.out/"0-OFF")
    assert rescored["state"] == "INVALID"
    assert rescored["reason"] == "forced_stop_after_grace"


@pytest.mark.parametrize("breach", ["cpu_sustained_breach", "cell_deadline"])
def test_forced_stop_preserves_breach_and_stops_block(recall_cell, monkeypatch, breach):
    a, control = recall_cell
    control.breach = breach
    # Both paths stop a live child; the deadline case uses a fake clock.
    if breach == "cell_deadline":
        control.breach = None
        a.max_seconds = -1
        original = measurement.subprocess.Popen
        def launch(*args, **kw):
            proc = original(*args, **kw)
            proc.returncode = None
            return proc
        monkeypatch.setattr(measurement.subprocess, "Popen", launch)
    monkeypatch.setattr(recall, "prepare", lambda _: dict(canaries=[{}]))
    def first_cell(a, state, block, kind):
        assert state == "OFF", "block advanced after a forced-stop breach"
        return measurement.cell(a, state, block, kind)
    monkeypatch.setattr(recall, "cell", first_cell)
    with pytest.raises(RuntimeError, match=breach):
        recall.main(a)
    saved = common.load(a.out/"0-OFF/summary.json")
    assert saved["state"] == "INVALID"
    assert saved["reason"] == "forced_stop_after_grace"
    assert saved["breach_reason"] == breach
    assert not (a.out/"summary.json").exists()


@pytest.mark.parametrize("raw,recorded,expected", [
    (dict(state="INCOMPLETE", rows_s=.5), "PASS", "FAIL"),
    (dict(state="INVALID", reason="forced_stop_after_grace", rows_s=5), "PASS", "INVALID"),
    (dict(state="INCOMPLETE", rows_s=5), "FAIL", "FAIL"),
    (dict(state="INCOMPLETE", rows_s=5), "PASS", "INVALID"),
])
def test_disagreement_returns_worst_verdict(tmp_path, raw, recorded, expected):
    # probe_a: slow/invalid receipts, a recorded FAIL, and a non-state forgery.
    (tmp_path/"catchup").mkdir()
    raw.update(end_to_end_s=7300, audit=dict(complete=False))
    events = [dict(event="_embedding_backfill_summary_run", embedded=32, at=i, end=i+.572) for i in range(5)]
    common.dump(tmp_path/"manifest.json", dict(rows=250000))
    common.dump(tmp_path/"catchup/summary.json", raw)
    (tmp_path/"catchup/events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    saved = floor.rate(raw, 7300, 250000, events)
    saved.update(state=recorded)
    if raw["rows_s"] == 5 and recorded == "PASS" and raw["state"] == "INCOMPLETE":
        saved["measured_seconds"] = 1
    common.dump(tmp_path/"summary.json", saved)
    result = rescore.v2(tmp_path)
    assert result["state"] == expected
    assert result["rescore_disagreement"]


def test_bare_pass_is_invalid(tmp_path):
    common.dump(tmp_path/"summary.json", dict(state="PASS"))
    result = rescore.v2(tmp_path)
    assert result["state"] == "INVALID"
    assert result["rescore_disagreement"]


@pytest.fixture(scope="module")
def low_disk_before_function_fixtures():
    # Detect an autouse high-headroom stub that would conceal the real guard.
    usage = shutil.disk_usage(".")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(shutil, "disk_usage", lambda _: usage._replace(free=1))
        yield


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_real_disk_guard_refuses(low_disk_before_function_fixtures, tmp_path, monkeypatch, platform):
    monkeypatch.setattr(measurement.sys, "platform", platform)
    with pytest.raises(RuntimeError, match="disk headroom stop"):
        measurement.cell(NS(scratch=tmp_path), "OFF", 0, "recall")


def test_grace_assertion_includes_session_end_terms():
    from pathlib import Path
    import ast
    tree = ast.parse(Path(catchup.__file__).read_text())
    assertion = next(n for n in tree.body if isinstance(n, ast.Assert))
    # poll + two session-end waits + retirement + exit publication + margin.
    constants = [n.value for n in ast.walk(assertion) if isinstance(n, ast.Constant)]
    assert sum(v for v in constants if isinstance(v, (int, float))) >= 5+2+5+10+15
    assert catchup.STOP_GRACE_S > catchup.WORKER_POLL_S+5+2+5+10+15
