"""A killed worker's truncated files cannot erase the parent INVALID receipt."""
from argparse import Namespace as NS
import json
import sqlite3

import pytest

from bench.instruments.local_catchup import catchup, common, measurement
from bench.instruments.local_catchup.tests.test_1586_v23_r2 import Child


@pytest.fixture
def cell_files(tmp_path, monkeypatch, fake_store_disk_headroom):
    sqlite3.connect(tmp_path/"baseline.db").close()
    a = NS(scratch=tmp_path, out=tmp_path, model_cache=tmp_path, lcmx_repo=tmp_path,
           hermes_source=tmp_path, smoke=False, label="fixture", max_seconds=-1)
    control = NS(killed=True, breach=None, files={})
    clock = [0.]
    monkeypatch.setattr(catchup, "STOP_GRACE_S", .01)
    monkeypatch.setattr(measurement, "Supervisor", lambda _: NS(
        breach=control.breach, watch=lambda _: None, start=lambda: None,
        close=lambda: None, summary=lambda *args: {}))
    def launch(*args, **kwargs):
        root = tmp_path/"0-OFF"
        common.dump(root/"raw-summary.json", dict(state="PASS"))
        for name, data in control.files.items():
            (root/name).write_bytes(data)
        return Child(1, clock, ignores_term=control.killed)
    monkeypatch.setattr(measurement.subprocess, "Popen", launch)
    return a, control, tmp_path/"0-OFF"


METRICS = dict(fixed_corpus_frozen_s=12, global_live_commits_during_freeze=7)
COMPLETE = (json.dumps(METRICS)+"\n").encode()
TAIL = b'{"unfinished":"\xff'


@pytest.mark.parametrize("name,data,metrics", [
    ("progress.jsonl", b"", {}),
    ("progress.jsonl", COMPLETE+TAIL, METRICS),
    ("progress.jsonl", TAIL, {}),
    ("raw-summary.json", b'{"state":', {}),
    ("recalls.jsonl", b'{"sample": 1}\n'+TAIL, {}),
])
@pytest.mark.parametrize("breach", [None, "sustained_cpu_breach"])
def test_forced_kill_always_writes_invalid_receipt(cell_files, name, data, metrics, breach):
    a, control, root = cell_files
    control.files, control.breach = {name: data}, breach
    with pytest.raises(RuntimeError, match=breach or "cell_deadline"):
        measurement.cell(a, "OFF", 0, "recall")
    saved = common.load(root/"summary.json")
    assert (saved["state"], saved["reason"]) == ("INVALID", "forced_stop_after_grace")
    assert saved["stop_killed"] is True
    assert saved["breach_reason"] == (breach or "cell_deadline")
    assert saved["worker_receipts_truncated"] == [name]
    assert {k: saved[k] for k in METRICS if k in saved} == metrics
    if name.endswith(".jsonl") and data:
        assert (root/name).read_bytes().endswith(TAIL)
        prefix = (root/name).read_bytes()[:-len(TAIL)]
        original = data[:-len(TAIL)]
        expected = b"".join((json.dumps(dict(json.loads(line), label="fixture"), sort_keys=True)+"\n").encode()
                            for line in original.splitlines())
        assert prefix == expected


@pytest.mark.parametrize("name", ["raw-summary.json", "recalls.jsonl"])
def test_graceful_truncated_worker_receipt_still_raises(cell_files, name):
    a, control, root = cell_files
    control.killed, control.files = False, {name: b'{"unfinished":'}
    with pytest.raises(json.JSONDecodeError):
        measurement.cell(a, "OFF", 0, "recall")
    assert not (root/"summary.json").exists()


@pytest.mark.parametrize("data,metrics", [(b"", {}), (COMPLETE, METRICS)])
def test_graceful_progress_and_labels_unchanged(cell_files, data, metrics):
    a, control, root = cell_files
    control.killed, control.files = False, {"progress.jsonl": data}
    with pytest.raises(RuntimeError, match="cell_deadline"):
        measurement.cell(a, "OFF", 0, "recall")
    saved = common.load(root/"summary.json")
    assert saved["state"] == "ABORTED"
    assert saved["stop_killed"] is False
    assert "worker_receipts_truncated" not in saved
    assert {k: saved[k] for k in METRICS if k in saved} == metrics
    expected = ((json.dumps(dict(METRICS, label="fixture"), sort_keys=True)+"\n").encode() if data else b"")
    assert (root/"progress.jsonl").read_bytes() == expected
