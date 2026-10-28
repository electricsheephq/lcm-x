"""Forced stops preserve supervisor breaches and each rig worker's own verdict."""
from itertools import permutations
import subprocess

import pytest

from bench.instruments.local_catchup import catchup, common, rig

pytest_plugins = ["bench.instruments.local_catchup.tests.test_fix_round3"]
STOP = catchup.stop


class Child:
    def __init__(self, pid, clock, ignores_term=True):
        self.pid, self.clock, self.ignores_term = pid, clock, ignores_term
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        if not self.ignores_term:
            self.returncode = 0

    def wait(self, timeout=None):
        if timeout is not None and self.returncode is None:
            self.clock[0] += timeout
            raise subprocess.TimeoutExpired("fake-child", timeout)
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.mark.parametrize("breach", ["sustained_cpu_breach", None])
def test_catchup_forced_stop_keeps_breach_reason(fake_catchup, monkeypatch, breach):
    a, checked, _, _, clock, meter = fake_catchup
    launch = catchup.launch
    def child(job, generation):
        proc = launch(job, generation)
        return Child(proc.pid, clock)
    monkeypatch.setattr(catchup, "launch", child)
    monkeypatch.setattr(catchup, "stop", STOP)
    monkeypatch.setattr(catchup, "STOP_GRACE_S", .01)
    monkeypatch.setattr(catchup, "progress", lambda *args: dict(checked, complete=True))
    meter.breach = breach
    result = catchup.run(a)
    assert common.load(a.out/"summary.json") == result
    assert (result["state"], result["reason"]) == ("INVALID", "forced_stop_after_grace")
    assert any(s["stop_killed"] for s in result["stops"])
    if breach:
        assert result["breach_reason"] == breach
    else:
        assert "breach_reason" not in result


@pytest.mark.parametrize("breach", ["swap_increased", None])
def test_rig_forced_stop_keeps_aggregate_breach_reason(fake_rig, monkeypatch, breach):
    a, checked, _, _, _, clock = fake_rig
    meter = rig.supervisor()
    meter.breach = breach
    launch = rig.launch
    def child(job, generation):
        proc = launch(job, generation)
        return Child(proc.pid, clock)
    monkeypatch.setattr(rig, "launch", child)
    monkeypatch.setattr(rig, "stop", STOP)
    monkeypatch.setattr(catchup, "STOP_GRACE_S", .01)
    monkeypatch.setattr(rig, "read_progress", lambda *args, **kwargs: checked.copy())
    a.restart_fraction, a.restart_after = 2, 1000
    rig.main()
    result = common.load(a.out/"summary.json")
    assert (result["state"], result["reason"]) == ("INVALID", "forced_stop_after_grace")
    assert all(any(s["stop_killed"] for s in row["stops"]) for row in result["per_process_progress"])
    if breach:
        assert result["breach_reason"] == breach
    else:
        assert "breach_reason" not in result


@pytest.mark.parametrize("order", list(permutations(("killed", "graceful", "complete"))))
@pytest.mark.parametrize("breach", ["sustained_cpu_breach", None])
@pytest.mark.parametrize("screen", [False, True])
def test_rig_rows_keep_own_state_in_every_process_order(fake_rig, monkeypatch, order, breach, screen):
    a, checked, final, _, _, clock = fake_rig
    a.processes, a.screen16 = 3, screen
    # Screen mode always allocates 16 workers; keep the remaining workers graceful.
    roles = order+("graceful",)*13 if screen else order
    a.restart_fraction, a.restart_after = 2, 1000
    meter = rig.supervisor()
    meter.breach = breach
    launch = rig.launch
    def child(job, generation):
        proc = launch(job, generation)
        return Child(proc.pid, clock, roles[int(job.out.name)] == "killed")
    def audited(db, manifest):
        return {**final, "complete": roles[int(db.parent.name)] == "complete"}
    monkeypatch.setattr(rig, "launch", child)
    monkeypatch.setattr(rig, "stop", STOP)
    monkeypatch.setattr(catchup, "STOP_GRACE_S", .01)
    monkeypatch.setattr(rig, "read_progress", lambda *args, **kwargs: checked.copy())
    monkeypatch.setattr(rig, "audit", audited)
    rig.main()
    result = common.load(a.out/"summary.json")
    assert (result["state"], result["reason"]) == ("INVALID", "forced_stop_after_grace")
    for role, row in zip(roles, result["per_process_progress"]):
        expected = (("INVALID", "forced_stop_after_grace") if role == "killed" else
                    ("COMPLETE", "full_expected_id_audit") if role == "complete" else
                    ("ABORTED", breach) if breach else ("INCOMPLETE", "max_wall"))
        assert (row["state"], row["reason"]) == expected
        assert any(s["stop_killed"] for s in row["stops"]) == (role == "killed")
        if role == "killed" and breach:
            assert row["breach_reason"] == breach
        else:
            assert "breach_reason" not in row
