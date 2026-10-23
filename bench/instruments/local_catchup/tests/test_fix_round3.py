"""SPEC I1-fix3: live restart residue, final coverage and CPU stop boundaries."""
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
import shlex

import pytest

from bench.instruments.local_catchup import audit, catchup, common, rig

pytest_plugins = ["bench.instruments.local_catchup.tests.test_fix_round2"]


def test_restart_inflight_residue_launches_gen1(fake_rig, monkeypatch):
    a, _, final, _, launched, _ = fake_rig
    calls = []
    def checked(db, _):
        calls.append(db)
        return {**final, "complete": False, "residue": {"dispatched": 32}} if calls.count(db) == 1 else final.copy()
    monkeypatch.setattr(rig, "audit", checked)
    rig.main()
    summary = common.load(a.out/"summary.json")
    assert summary["state"] == "COMPLETE"
    assert [g for _, g in launched] == [0, 0, 1, 1]


@pytest.mark.parametrize("incomplete_process", [None, 0, 1])
def test_rig_final_coverage_overrides_complete(fake_rig, monkeypatch, incomplete_process):
    a, checked, final, _, _, _ = fake_rig
    monkeypatch.setattr(rig, "read_progress", lambda *args, **kw: {**checked, "complete": True})
    def final_audit(db, _):
        if incomplete_process is not None and db.parent.name == str(incomplete_process):
            return {**final, "complete": False,
                    "classes": {"chunks": {"covered": 1, "expected": 2, "missing_count": 1}}}
        return final.copy()
    monkeypatch.setattr(rig, "audit", final_audit)
    rig.main()
    summary = common.load(a.out/"summary.json")
    if incomplete_process is None:
        assert (summary["state"], summary["reason"]) == ("COMPLETE", "complete")
    else:
        assert (summary["state"], summary["reason"]) == ("ABORTED", "final_coverage_regressed")
        row = summary["per_process_progress"][incomplete_process]
        assert (row["state"], row["reason"]) == ("ABORTED", "final_coverage_regressed")


@pytest.fixture
def fake_catchup(fake_rig, monkeypatch):
    a, checked, final, _, launched, clock = fake_rig
    a.db, a.home, a.manifest = a.scratch/"store.db", a.scratch/"homes", a.scratch/"manifest.json"
    a.progress_interval = 30
    a.scratch.mkdir()
    common.dump(a.manifest, {"rows": 2})
    for result in (checked, final):
        result.update(covered_rows=1)
        result["classes"] = {"chunks": {"covered": 1, "expected": 2, "covered_tokens": 1}}
    meter = rig.supervisor()
    monkeypatch.setattr(catchup, "baseline", lambda *args: checked.copy())
    monkeypatch.setattr(catchup, "immutable_cache", lambda _: None)
    monkeypatch.setattr(catchup, "supervisor", lambda *args: meter)
    monkeypatch.setattr(catchup, "launch", rig.launch)
    monkeypatch.setattr(catchup, "stop", rig.stop)
    monkeypatch.setattr(catchup, "foreign_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(catchup, "audit", lambda *args: final.copy())
    monkeypatch.setattr(catchup, "progress", lambda *args: {**checked, "complete": len(launched) >= 2})
    monkeypatch.setattr(audit, "connect", lambda _: nullcontext(Namespace(backup=lambda _: None)))
    original_launch = catchup.launch
    def launch(job, generation):
        p = original_launch(job, generation)
        common.dump(job.out/f"worker-{generation}.json", {"pid": p.pid, "effective_config": {}})
        return p
    monkeypatch.setattr(catchup, "launch", launch)
    return a, checked, final, launched, clock, meter


def test_catchup_restart_inflight_residue_launches_gen1(fake_catchup, monkeypatch):
    a, checked, final, launched, _, _ = fake_catchup
    monkeypatch.setattr(catchup, "audit", lambda *args:
                        {**checked, "residue": {"dispatched": 32}} if len(launched) == 1 else final.copy())
    summary = catchup.run(a)
    assert summary["state"] == "COMPLETE"
    assert summary["restarted"] and [g for _, g in launched] == [0, 1]


@pytest.mark.parametrize("complete", [True, False])
def test_catchup_final_coverage_overrides_complete(fake_catchup, monkeypatch, complete):
    a, checked, final, _, _, _ = fake_catchup
    monkeypatch.setattr(catchup, "progress", lambda *args: {**checked, "complete": True})
    final.update(complete=complete)
    final["classes"]["chunks"]["missing_count"] = 0 if complete else 1
    summary = catchup.run(a)
    expected = ("COMPLETE", "full_expected_id_audit") if complete else ("ABORTED", "final_coverage_regressed")
    assert (summary["state"], summary["reason"]) == expected


def test_catchup_restart_closes_cpu_window_before_stop(fake_catchup, monkeypatch):
    a, _, _, launched, clock, meter = fake_catchup
    stops = []
    def stop(p):
        stops.append((p.pid, clock[0], meter.windows[p.pid][1]))
        clock[0] += 10
        return dict(stop_killed=False)
    monkeypatch.setattr(catchup, "stop", stop)
    catchup.run(a)
    assert [g for _, g in launched] == [0, 1]
    assert stops[0] == (1, 0., 0.)
    assert meter.windows[1][1] < meter.windows[2][0]


def test_readme_explicit_mixed_command_fits_corpus_estimate():
    lines = (Path(catchup.__file__).parent/"README.md").read_text().splitlines()
    commands = [shlex.split(line) for line in lines if "local_catchup.corpus " in line]
    mixed = next((args for args in commands if args[args.index("--shape")+1] == "mixed"), None)
    assert mixed is not None, "README needs an explicit mixed corpus command"
    rows = int(mixed[mixed.index("--rows")+1])
    # Real 1,500-row whole-shard pilot measured by the fix2 reviewer.
    estimate = 3*(18051072+2002*2048)*rows/1500
    assert float(mixed[mixed.index("--peak-disk-gb")+1])*2**30 >= estimate
