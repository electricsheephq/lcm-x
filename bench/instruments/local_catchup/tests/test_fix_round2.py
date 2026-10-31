"""SPEC I1-fix2: bounded regressions, fake workers and no model downloads."""
from argparse import Namespace
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import pytest

from bench.instruments.local_catchup import catchup, common, corpus, cpu, rig


@pytest.fixture
def fake_rig(tmp_path, monkeypatch):
    a = Namespace(scratch=tmp_path/"scratch", out=tmp_path/"out", model_cache=tmp_path/"cache",
                  processes=2, rows=2, shape="short", baseline="empty", max_wall=100,
                  stagger=0, peak_disk_gb=4, screen16=False, lcmx_repo=tmp_path,
                  hermes_source=tmp_path, restart_fraction=.4, restart_after=None)
    clock, calls, launched = [0.], [], []
    checked = dict(raw_unchanged=True, changed_vectors=[], integrity="ok", residue={},
                   complete=False, scan=[], classes={"chunks": {"covered": 1, "expected": 2}})
    final = {**checked, "complete": True}
    meter = Namespace(windows={}, breach=None, start=lambda: None, close=lambda: None)
    meter.summary = lambda *args, **kw: {"per_process": {}, "end": kw.get("end"),
                                        "aggregate_window": kw.get("aggregate_window")}
    monkeypatch.setattr(rig, "parser", lambda _: Namespace(add_argument=lambda *a, **kw: None,
                                                          parse_args=lambda: a))
    monkeypatch.setattr(rig, "immutable_cache", lambda _: None)
    monkeypatch.setattr(rig.shutil, "disk_usage", lambda _: Namespace(free=200*2**30))
    monkeypatch.setattr(rig, "build", lambda *args: {"database_bytes": 100, "rows": 2,
                                                  "expected": {"chunks": {"x": 1}}})
    monkeypatch.setattr(rig, "baseline", lambda db, m: calls.append(("start", db)) or checked.copy())
    monkeypatch.setattr(rig, "audit", lambda db, m: calls.append(("audit", db)) or final.copy())
    monkeypatch.setattr(rig, "read_progress", lambda db, m, **kw: {**checked, "complete": len(launched) >= 4})
    monkeypatch.setattr(rig, "supervisor", lambda *args: meter)
    monkeypatch.setattr(rig, "foreign_files", lambda *args, **kw: [])
    monkeypatch.setattr(rig, "connect", lambda _: nullcontext(Namespace(backup=lambda _: None)))
    monkeypatch.setattr(__import__("sqlite3"), "connect", lambda _: nullcontext(None))
    monkeypatch.setattr(rig.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rig.time, "sleep", lambda s: clock.__setitem__(0, clock[0]+s))
    def launch(job, generation):
        pid = len(launched)+1
        launched.append((job, generation))
        common.dump(job.out/f"worker-{generation}.json", {"pid": pid})
        return Namespace(pid=pid, poll=lambda: None)
    monkeypatch.setattr(rig, "launch", launch)
    monkeypatch.setattr(rig, "stop", lambda _: clock.__setitem__(0, clock[0]+10) or dict(stop_killed=False))
    meter.watch = lambda pid: meter.windows.update({pid: [clock[0], None]})
    return a, checked, final, calls, launched, clock


@pytest.mark.parametrize("bad", [dict(raw_unchanged=False), dict(changed_vectors=["old"]),
                                  dict(integrity="broken"), dict(residue={"uncertain": 1})])
def test_a_final_audit_overrides_complete(fake_rig, bad, monkeypatch):
    a, checked, final, _, _, _ = fake_rig
    monkeypatch.setattr(rig, "read_progress", lambda *args, **kw: {**checked, "complete": True})
    final.update(bad)
    rig.main()
    s = common.load(a.out/"summary.json")
    assert s["state"] == "ABORTED" and s["reason"] == "final_integrity_or_residue"
    assert all(r["state"] == "ABORTED" and r["reason"] == s["reason"]
               for r in s["per_process_progress"])


def test_a_full_audit_at_start_restart_end(fake_rig):
    a, _, _, calls, launched, _ = fake_rig
    rig.main()
    assert [g for _, g in launched] == [0, 0, 1, 1]
    summary = common.load(a.out/"summary.json")
    assert summary["cpu"]["aggregate_window"] == [0., 30.]
    for i in range(2):
        assert [(kind, db) for kind, db in calls if db == a.scratch/str(i)/"store.db"] == [
            ("start", a.scratch/str(i)/"store.db"),
            ("audit", a.scratch/str(i)/"store.db"),
            ("audit", a.scratch/str(i)/"store.db")]


def test_b_rig_records_loop_end_before_stops(fake_rig, monkeypatch):
    a, checked, _, _, _, clock = fake_rig
    monkeypatch.setattr(rig, "read_progress", lambda *args, **kw: checked.copy())
    rig.main()
    s = common.load(a.out/"summary.json")
    assert s["cpu"]["end"] <= a.max_wall
    assert s["cpu"]["end"] < clock[0]


def test_b_cpu_clips_unfinished_and_aggregate_at_loop_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "swap", lambda: 0)
    m = cpu.Supervisor(tmp_path/"cpu.jsonl")
    m.windows = {1: [0, None], 2: [10, None]}
    m.samples = {1: [(i, 1, 1 if i <= 60 else 9) for i in range(1, 81)],
                 2: [(i, 1, 1 if i <= 60 else 9) for i in range(11, 81)]}
    m.aggregate = [(i, 1, 2 if i <= 60 else 18) for i in range(11, 81)]
    monkeypatch.setattr(cpu.time, "monotonic", lambda: 100)
    s = m.summary(end=60)
    assert s["per_process"]["1"]["mean_cores"] == 1
    assert s["per_process"]["2"]["seconds"] == 50
    assert s["aggregate"]["mean_cores"] == 2 and s["aggregate"]["seconds"] == 50
    assert s["tail"]["seconds"] == 0


def test_c_own_outputs_allowed_sibling_paths_forbidden(tmp_path, monkeypatch):
    import psutil
    user, cell = tmp_path/"user", tmp_path/"user"/"cell"
    own = [cell/"out"/"status.json.tmp", cell/"scratch"/"final.db"]
    foreign = [user/"sibling"/"out"/"events.jsonl", user/"sibling"/"store.db",
               cell/"out"/".hermes"/"store.db"]
    monkeypatch.setattr(psutil, "Process", lambda _: Namespace(open_files=lambda: [
        Namespace(path=str(p)) for p in own+foreign]))
    assert common.foreign_files(1, cell/"home", cell/"store.db", user/"cache",
                                user_home=user, out=cell/"out", scratch=own[1]) == list(map(str, foreign))


def test_d_mixed_pilot_has_proportional_whole_shards():
    def shares(rows):
        counts = Counter()
        for shard, offset in enumerate(range(0, rows//2, 250)):
            counts[(20, 300, 1200)[shard % 3]] += min(250, rows//2-offset)
        return {k: v/(rows//2) for k, v in counts.items()}
    for size in (1000, 2000):
        rows = corpus.pilot_rows(size, 250000, "mixed")
        assert rows % 1500 == 0
        pilot, full = shares(rows), shares(250000)
        assert all(abs(pilot[k]/full[k]-1) <= .1 for k in full)


def test_e_copy_names_bind_cell_and_generation(tmp_path):
    a = Namespace(out=tmp_path/"run-a"/"catchup", scratch=tmp_path/"scratch")
    b = Namespace(out=tmp_path/"run-b"/"catchup", scratch=a.scratch)
    paths = [catchup.final_copy(x, g) for x in (a, b) for g in (0, 1)]
    assert len(set(paths)) == 4 and all(p.parent == a.scratch for p in paths)


def test_e_cpu_abort_tolerates_scheduler_jitter(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "swap", lambda: 0)
    ticks = [0.]
    monkeypatch.setattr(cpu.time, "monotonic", lambda: ticks[0])
    monkeypatch.setattr(cpu, "process", lambda _: (ticks[0]*13, 1, 1))
    m = cpu.Supervisor(tmp_path/"cpu.jsonl")
    m.watch(1)
    for i in range(1, 11):
        ticks[0] = i*.99999
        result = m.sample([1])
    assert result == "sustained_cpu_breach"


def test_e_documented_screen_budget():
    import shlex
    line = next(s for s in (Path(rig.__file__).parent/"README.md").read_text().splitlines()
                if "probes screen16" in s)
    args = shlex.split(line)
    assert "--peak-disk-gb" in args
    assert float(args[args.index("--peak-disk-gb")+1]) >= 90


def test_b_restarted_rig_keeps_aggregate_window(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "swap", lambda: 0)
    m = cpu.Supervisor(tmp_path/"cpu.jsonl")
    m.windows = {1: [0, 30], 2: [35, 100], 3: [10, 40], 4: [45, None]}
    m.aggregate = [(i, 1, 1) for i in range(1, 101)]
    s = m.summary(end=100, aggregate_window=(10, 100))
    assert s["aggregate"]["seconds"] == 90
    assert s["aggregate"]["mean_cores"] == 1
