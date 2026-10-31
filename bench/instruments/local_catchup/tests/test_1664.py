"""Round-four regressions: guard attribution, read cost and exercised drain restart."""
from argparse import Namespace as NS
import json
import socket
import subprocess
import threading

import pytest

from bench.instruments.local_catchup import catchup, common, measurement, rig

pytest_plugins = ["bench.instruments.local_catchup.tests.test_fix_round2",
                  "bench.instruments.local_catchup.tests.test_fix_round3"]


def test_fresh_guard_event_does_not_hide_unrelated_error(tmp_path):
    request = tmp_path/"request.json"
    common.dump(request, dict(root=str(tmp_path)))
    def worker(_):
        common.append(tmp_path/"events.jsonl", {"violation": "network_attempt"})
        raise RuntimeError("unrelated")
    with pytest.raises(RuntimeError, match="^unrelated$"):
        measurement.guarded_worker(worker, request)
    assert not (tmp_path/"raw-summary.json").exists()


def test_verified_read_cost_sums_stores_and_polls(fake_rig, monkeypatch):
    a, checked, _, _, _, _ = fake_rig
    a.max_wall, a.restart_fraction = 700, 2
    monkeypatch.setattr(rig, "read_progress", lambda *args, **kw: checked.copy())
    ticks = iter(range(1000))
    monkeypatch.setattr(rig.time, "process_time", lambda: next(ticks))
    rig.main()
    summary = common.load(a.out/"summary.json")
    # Verified polls 10 and 20 plus the final poll, each reading two stores.
    assert summary["verified_read_cpu_s_total"] == 6
    assert summary["verified_read_cpu_s_max_per_read"] == 1


@pytest.mark.parametrize("attempt", ["cache", "socket", "subprocess", "foreground"])
def test_download_permission_reaches_cache_but_actual_attempts_fail(tmp_path, attempt):
    cache = tmp_path/"warm-model"
    cache.write_text("synthetic cached model")
    calls, errors = [], []
    def construct(self, *, allow_download):
        calls.append(allow_download)
        if attempt == "socket":
            socket.getaddrinfo("offline.invalid", 443)
        if attempt == "subprocess":
            subprocess.Popen(["offline-command"])
        return cache.read_text()
    provider = NS(_FASTEMBED_THREADS=2, FastembedProvider=NS(_construct=construct))
    command = NS(_chunk_backfill_run=lambda: {}, _embedding_backfill_summary_run=lambda: {})
    events = tmp_path/"events.jsonl"
    common.guards(lambda name: provider if name == "embedding_provider" else command, events)
    values = []
    def prepare():
        try:
            values.append(provider.FastembedProvider._construct(None, allow_download=True))
        except RuntimeError as exc:
            errors.append(str(exc))
    if attempt == "foreground":
        prepare()
    else:
        thread = threading.Thread(target=prepare)
        thread.start()
        thread.join()
    recorded = [json.loads(s) for s in events.read_text().splitlines()]
    assert recorded[0]["allow_download"] is True
    assert "download" not in recorded[0]
    if attempt == "cache":
        assert values == ["synthetic cached model"] and not errors
        a = NS(root=tmp_path, home=tmp_path, db=tmp_path/"db", model_cache=tmp_path,
               user_home=tmp_path, lcmx_repo=tmp_path, hermes_source=tmp_path,
               scratch=tmp_path, _next_progress=float("inf"))
        # Keep the consumer check isolated from actual file-descriptor inspection.
        from unittest.mock import patch
        with patch.object(measurement, "foreign_files", return_value=[]):
            assert measurement.safety(a)["violations"] == 0
    else:
        assert len(errors) == 1 and not values
        assert calls == ([] if attempt == "foreground" else [True])
        if attempt != "foreground":
            assert recorded[-1]["violation"] == ("network_attempt" if attempt == "socket" else "subprocess_launch")


@pytest.mark.parametrize("gen1", ["ready", "no_bind", "no_ready", "exit"])
def test_fast_drain_requires_generation_one_bind_ready_and_full_audit(fake_catchup, monkeypatch, gen1):
    a, checked, final, launched, clock, _ = fake_catchup
    a.drain96, a.max_wall = True, 90
    initial = {**checked, "classes": {"chunks": {"covered": 0, "expected": 2, "covered_tokens": 0}}}
    monkeypatch.setattr(catchup, "baseline", lambda *args: initial)
    full = {**final, "classes": {"chunks": {"covered": 2, "expected": 2, "covered_tokens": 2}}}
    monkeypatch.setattr(catchup, "progress", lambda *args: full.copy())
    audits = []
    def audit(*args):
        audits.append((len(launched), clock[0]))
        return full.copy()
    monkeypatch.setattr(catchup, "audit", audit)
    launch = catchup.launch
    def bind(job, generation):
        p = launch(job, generation)
        if generation == 1:
            if gen1 == "exit":
                p.poll, p.returncode = lambda: 1, 1
            if gen1 != "no_bind":
                common.dump(job.out/"worker-1.json", dict(pid=p.pid, effective_config={}, generation=1, one_bind=True))
            common.dump(job.out/"status-1.json", dict(ready=dict(available=gen1 == "ready")))
            common.append(job.out/"events.jsonl", dict(event="prepared", generation=1, foreground=False))
        return p
    monkeypatch.setattr(catchup, "launch", bind)
    summary = catchup.run(a)
    assert [g for _, g in launched] == [0, 1]
    if gen1 == "ready":
        assert summary["state"] == "COMPLETE" and summary["restarted"]
        assert common.load(a.out/"generation-1-audit.json")["complete"]
        assert len([n for n, _ in audits if n == 2]) >= 2  # Live full audit, then final audit.
    else:
        assert summary["state"] != "COMPLETE" and not summary["restarted"]
        assert clock[0] <= a.max_wall+20


def test_drain_without_covered_progress_never_completes_or_restarts(fake_catchup):
    a, _, _, launched, clock, _ = fake_catchup
    a.drain96, a.max_wall = True, 60
    summary = catchup.run(a)
    assert summary["state"] == "INCOMPLETE" and not summary["restarted"]
    assert [g for _, g in launched] == [0]
    assert clock[0] <= a.max_wall+20
