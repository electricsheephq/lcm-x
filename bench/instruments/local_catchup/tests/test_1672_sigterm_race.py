"""Issue #1672: an exited process group must not bypass timeout results."""
import signal
import subprocess

import pytest

from bench.instruments.local_catchup import common, floor
from bench.instruments.local_catchup.tests.test_1669_item9 import floor_args


def test_outer_timeout_sigterm_exit_race_labels_results(tmp_path, monkeypatch):
    floor_args(tmp_path, monkeypatch, limited=False)
    (tmp_path/"out").mkdir()
    common.dump(tmp_path/"out/partial.json", {"state": "INCOMPLETE"})
    waits, signals = [], []

    class Proc:
        pid = 12345

        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("floor", timeout)
            return 0

    def killpg(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGTERM:
            raise ProcessLookupError("process group already exited")

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Proc())
    monkeypatch.setattr(floor.os, "killpg", killpg)
    with pytest.raises(SystemExit) as stopped:
        floor.main()
    assert stopped.value.code == 3
    assert waits == [150*60+120, 30, None]
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert common.load(tmp_path/"out/summary.json") == dict(
        state="ABORTED", reason="outer_deadline", label=floor.LABEL)
    assert common.load(tmp_path/"out/partial.json")["label"] == floor.LABEL
