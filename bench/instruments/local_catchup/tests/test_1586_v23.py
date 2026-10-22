"""v2.3 instrument regressions: fake workers/receipts, no model downloads."""
from argparse import Namespace as NS
import json
import subprocess
import sys

import pytest

from bench.instruments.local_catchup import catchup, common, measurement, recall, rescore, rig
from bench.instruments.local_catchup.tests.test_1586_v2 import rows

pytest_plugins = ["bench.instruments.local_catchup.tests.test_fix_round3"]


def test_mixed_shutdown_grace_exceeds_product_exit_budget(tmp_path):
    marker = tmp_path/"dispatched"
    script = """
import pathlib, signal, sys, time
marker = pathlib.Path(sys.argv[1])
marker.write_text('dispatched')
def publish(*_):
    time.sleep(12)
    marker.unlink()
    sys.exit(0)
signal.signal(signal.SIGTERM, publish)
print('ready', flush=True)
while True: time.sleep(.1)
"""
    proc = subprocess.Popen([sys.executable, "-B", "-c", script, str(marker)], stdout=subprocess.PIPE)
    try:
        assert proc.stdout.readline() == b"ready\n"
        stopped = catchup.stop(proc)
        assert proc.returncode == 0, "supervisor killed the pending publication"
        assert stopped["stop_killed"] is False
        assert stopped["stop_returncode"] == 0
        assert stopped["stop_exit_at"]-stopped["stop_sigterm_at"] >= 12
        assert not marker.exists()
        assert catchup.STOP_GRACE_S > catchup.WORKER_POLL_S+2+10+15
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()


class IgnoresTerm:
    pid = 1
    returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def wait(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("fake-child", timeout)
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.mark.parametrize("restart", [False, True])
def test_forced_stop_is_invalid_not_integrity_failure(fake_catchup, monkeypatch, restart):
    a, checked, final, _, clock, _ = fake_catchup
    launch = catchup.launch
    def child(job, generation):
        launch(job, generation)
        proc = IgnoresTerm()
        wait = proc.wait
        def waited(timeout=None):
            clock[0] += timeout or 0
            return wait(timeout)
        proc.wait = waited
        return proc
    monkeypatch.setattr(catchup, "launch", child)
    monkeypatch.setattr(catchup, "stop", STOP)
    monkeypatch.setattr(catchup, "progress", lambda *a: dict(checked, complete=not restart))
    final["residue"] = {"dispatched": 32}
    result = catchup.run(a)
    assert (result["state"], result["reason"]) == ("INVALID", "forced_stop_after_grace")
    assert any(s["stop_killed"] for s in result["stops"])
    assert result["audit"]["residue"] == {"dispatched": 32}


STOP = catchup.stop


@pytest.mark.parametrize("live", [True, False])
def test_recall_watchdog_distinguishes_live_new_work_from_frozen_corpus(tmp_path, monkeypatch, live):
    a = NS(root=tmp_path, out=tmp_path, db=tmp_path/"db", home=tmp_path,
           model_cache=tmp_path, user_home=str(tmp_path), scratch=tmp_path,
           lcmx_repo=tmp_path, hermes_source=tmp_path, _ready=lambda: True)
    common.dump(tmp_path/"manifest.json", {})
    clock = [100.]
    monkeypatch.setattr(measurement.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(measurement, "foreign_files", lambda *a, **kw: [])
    monkeypatch.setattr(measurement, "progress", lambda *a: dict(complete=False, classes={"chunks": {"covered": 1}}, scan={}))
    measurement.safety(a)
    for tick in range(1, 22):
        clock[0] = 100.+tick*30
        if live:
            common.append(tmp_path/"events.jsonl", dict(event="_chunk_backfill_run", embedded=32, foreground=False, at=clock[0]-1, end=clock[0]))
        if not live and tick >= 20:
            with pytest.raises(RuntimeError, match="zero progress for ten minutes"):
                measurement.safety(a)
        else:
            measurement.safety(a)
    last = json.loads((tmp_path/"progress.jsonl").read_text().splitlines()[-1])
    assert last["fixed_corpus_frozen_s"] == 630
    assert last["global_live_commits_during_freeze"] == (21*32 if live else 0)
    if live:
        evidence = measurement.safety(a)
        assert evidence["fixed_corpus_frozen_s"] == 630
    else:
        # Provider unavailability must still prevent a dead-worker inference.
        a._ready = lambda: False
        clock[0] += 30
        measurement.safety(a)


def test_aborted_block_keeps_completed_state_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr(recall, "bootstrap", lambda v, f, seed: [f(v)])
    a = NS(out=tmp_path, queries=200, seed=1586, block=0, blocks=1, max_seconds=6300,
           standalone_calibration_s=.572)
    monkeypatch.setattr(recall, "prepare", lambda _: dict(canaries=[{}]))
    completed = []
    def cell(a, state, block, kind):
        root = a.out/f"{block}-{state}"
        root.mkdir()
        if state != "OFF":
            raise RuntimeError("later state abort")
        samples = [dict(r, coverage_start={}, coverage_end={}) for r in rows()+rows(query_class="broad_nl")]
        pubs = [dict(start=0, end=10, node_id=i) for i in range(10)]
        for name, values in (("recalls", samples), ("publications", pubs)):
            (root/f"{name}.jsonl").write_text("".join(json.dumps(v)+"\n" for v in values))
        raw = recall.score(samples, pubs)
        raw.update(measurement_valid=True, catchup_activity=False)
        completed.append(raw)
        for name in ("raw-summary.json", "summary.json"):
            common.dump(root/name, raw)
        common.dump(root/"request.json", dict(seed=1586, state=state, block=block, rows=250000, smoke=False))
        common.dump(root/"final-audit.json", dict(classes={}))
        return raw
    monkeypatch.setattr(recall, "cell", cell)
    with pytest.raises(RuntimeError, match="later state abort"):
        recall.main(a)
    root = tmp_path/"0-OFF"
    assert common.load(root/"summary.json")["reason"] == "calibration_window_short"
    assert "standalone_calibration_s" in completed[0]
    # A conflicting recorded verdict stays non-PASS and reports every difference.
    raw = common.load(root/"raw-summary.json")
    raw.update(reason="recorded_before_abort", host_check="recorded_unknown")
    common.dump(root/"raw-summary.json", raw)
    result = rescore.v2(root)
    assert result["state"] == "UNCALIBRATED"
    assert result["rescore_disagreement"] == {
        "reason": dict(recorded="recorded_before_abort", recomputed="calibration_window_short"),
        "host_check": dict(recorded="recorded_unknown", recomputed="UNCALIBRATED")}


def test_rig_records_every_stop_and_forced_stop_class(fake_rig, monkeypatch):
    a, _, final, _, _, clock = fake_rig
    final["residue"] = {"dispatched": 32}
    def stop(proc):
        clock[0] += 60
        return dict(stop_sigterm_at=clock[0]-60, stop_exit_at=clock[0],
                    stop_killed=True, stop_returncode=-9)
    monkeypatch.setattr(rig, "stop", stop)
    rig.main()
    result = common.load(a.out/"summary.json")
    assert (result["state"], result["reason"]) == ("INVALID", "forced_stop_after_grace")
    for progress in result["per_process_progress"]:
        assert progress["reason"] == "forced_stop_after_grace"
        assert all(s["stop_killed"] and s["stop_returncode"] == -9 for s in progress["stops"])


@pytest.mark.parametrize("killed", [True, False])
def test_measurement_records_stop_and_freeze_summary(tmp_path, monkeypatch, killed):
    import sqlite3
    a = NS(scratch=tmp_path, out=tmp_path/"out", model_cache=tmp_path,
           lcmx_repo=tmp_path, hermes_source=tmp_path, smoke=True, label="fake", max_seconds=1)
    a.out.mkdir()
    sqlite3.connect(tmp_path/"baseline.db").close()
    stopped = dict(stop_sigterm_at=1, stop_exit_at=61 if killed else 13,
                   stop_killed=killed, stop_returncode=-9 if killed else 0)
    monkeypatch.setattr(measurement, "stop", lambda _: stopped)
    monkeypatch.setattr(measurement, "Supervisor", lambda _: NS(
        breach=None, watch=lambda _: None, start=lambda: None, close=lambda: None, summary=lambda *a: {}))
    def launch(*args, **kw):
        root = a.out/"0-OFF"
        common.dump(root/"raw-summary.json", dict(state="UNCALIBRATED"))
        common.append(root/"progress.jsonl", dict(fixed_corpus_frozen_s=630, global_live_commits_during_freeze=672))
        return NS(pid=1, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(measurement.subprocess, "Popen", launch)
    result = measurement.cell(a, "OFF", 0, "recall")
    assert all(result[k] == v for k, v in stopped.items())
    assert result["fixed_corpus_frozen_s"] == 630
    assert result["global_live_commits_during_freeze"] == 672
    assert result["state"] == ("INVALID" if killed else "UNCALIBRATED")
    if killed:
        assert result["reason"] == "forced_stop_after_grace"
