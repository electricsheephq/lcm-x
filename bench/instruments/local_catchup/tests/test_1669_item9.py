"""Issue #1669: round-five instrument claims, reproduced before implementation."""
from pathlib import Path
import signal
import subprocess

import pytest

from bench.instruments.local_catchup import common, floor

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]
WORKFLOW = Path(__file__).resolve().parents[4]/".github/workflows/lcmx-1586-floor.yml"


def floor_args(tmp_path, monkeypatch, limited=True, minutes=150):
    monkeypatch.setattr(floor.sys, "platform", "linux")
    monkeypatch.setattr(floor.sys, "argv", ["floor", "--lcmx-repo", str(tmp_path), "--hermes-source", str(tmp_path),
        "--scratch", str(tmp_path/"scratch"), "--out", str(tmp_path/"out"), "--model-cache", str(tmp_path),
        "--cgroup", str(tmp_path), "--cell", "coverage-rate", "--max-minutes", str(minutes),
        *(["--limited"] if limited else [])])

def test_item9_coverage_budget_fails_fast_before_build(tmp_path, monkeypatch):
    floor_args(tmp_path, monkeypatch, minutes=121)
    monkeypatch.setattr(floor.subprocess, "check_output", lambda *a, **kw: "CPU fixture\n")
    monkeypatch.setattr(floor, "standalone", lambda *a: {"standalone_calibration_s": .572})
    monkeypatch.setattr(floor, "measured_limits", lambda _: {})
    monkeypatch.setattr(floor, "isolated", lambda *a: None)
    def build(*a): raise AssertionError("build started with insufficient budget")
    monkeypatch.setattr(floor, "build", build)
    with pytest.raises(ValueError, match="coverage-rate.*150"):
        floor.main()

@pytest.mark.parametrize("error", [TimeoutError("floor deadline"), subprocess.TimeoutExpired("recall", 9000)])
def test_item9_inner_timeout_writes_labelled_summary(tmp_path, monkeypatch, error):
    floor_args(tmp_path, monkeypatch)
    monkeypatch.setattr(floor.subprocess, "check_output", lambda *a, **kw: "CPU fixture\n")
    monkeypatch.setattr(floor, "standalone", lambda *a: {"standalone_calibration_s": .572})
    monkeypatch.setattr(floor, "measured_limits", lambda _: {})
    monkeypatch.setattr(floor, "isolated", lambda *a: None)
    def build(*a):
        common.dump(tmp_path/"out/partial.json", {"state": "INCOMPLETE"})
        raise error
    monkeypatch.setattr(floor, "build", build)
    with pytest.raises(SystemExit) as stopped:
        floor.main()
    assert stopped.value.code == 3
    summary = common.load(tmp_path/"out/summary.json")
    assert summary["state"] == "ABORTED" and summary["reason"] == "floor_deadline"
    assert summary["label"] == floor.LABEL
    assert common.load(tmp_path/"out/partial.json")["label"] == floor.LABEL

@pytest.mark.parametrize("grace_expires", [False, True])
def test_item9_outer_timeout_terms_group_then_kills_after_grace(tmp_path, monkeypatch, grace_expires):
    floor_args(tmp_path, monkeypatch, limited=False)
    waits, signals = [], []
    class Proc:
        pid = 12345
        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1 or (len(waits) == 2 and grace_expires):
                raise subprocess.TimeoutExpired("floor", timeout)
            return -9 if grace_expires else 0
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Proc())
    monkeypatch.setattr(floor.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    # The old subprocess.run has no grace or process-group cleanup at all.
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("floor", kw["timeout"])))
    with pytest.raises(SystemExit) as stopped:
        floor.main()
    assert stopped.value.code == 3
    assert waits[0] > 150*60 and waits[1] >= 30
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert common.load(tmp_path/"out/summary.json") == dict(state="ABORTED", reason="outer_deadline", label=floor.LABEL)
