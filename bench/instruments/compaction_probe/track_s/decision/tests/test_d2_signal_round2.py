"""#1846 round 2: deterministic cleanup signals; only test-owned fake groups."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from test_d2_calibrated import HERE, load
from test_d2_driver import Clock
from test_d2_robustness import group_exists, wait_until
from test_d2_signal_round1 import cleanup_cell, owned_groups, ready_groups


@pytest.mark.parametrize(
    "first,second",
    [("INT", signal.SIGINT), ("INT", signal.SIGTERM), ("exception", signal.SIGTERM)],
)
@pytest.mark.parametrize("trial", range(6))
def test_second_signal_during_exception_cleanup(tmp_path, first, second, trial):
    child = "import time; time.sleep(60)"
    parent = (
        "import json,subprocess,sys,time; from pathlib import Path; "
        f"ps=[subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True) for _ in range(2)]; "
        "Path(sys.argv[1]).write_text(json.dumps([p.pid for p in ps])); time.sleep(60)"
    )
    harness = tmp_path / "driver.py"
    harness.write_text(
        "import json,signal,subprocess,sys,time; from pathlib import Path\n"
        f"sys.path.insert(0,{str(HERE)!r})\nimport d2_driver\n"
        "signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
        "signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
        "previous = {s: signal.getsignal(s) for s in signals}\n"
        "original = d2_driver.stop_process\n"
        "def slow_stop(p):\n"
        f"    Path({str(tmp_path / 'cleanup')!r}).touch()\n"
        "    time.sleep(.1)\n    original(p)\n"
        "d2_driver.stop_process = slow_stop\n"
        "def producer(row, attempt, path):\n"
        f"    p = subprocess.Popen([sys.executable,'-c',{parent!r},str(path/'groups.json')],start_new_session=True)\n"
        "    (path / 'producer.pid').write_text(str(p.pid))\n    return p\n"
        "def sleep(seconds):\n"
        "    end = time.monotonic() + seconds\n"
        "    while time.monotonic() < end:\n"
        f"        if Path({str(tmp_path / 'exception')!r}).exists(): raise ValueError('synthetic')\n"
        "        time.sleep(.002)\n"
        "try:\n"
        f"    d2_driver.run_comparison(dict(option='B',aa=False),Path({str(tmp_path / 'out')!r}),producer=producer,sleep=sleep)\n"
        "finally:\n"
        f"    Path({str(tmp_path / 'restored.json')!r}).write_text(json.dumps([signal.getsignal(s) == previous[s] for s in signals]))\n"
    )
    log = (tmp_path / "driver.log").open("w")
    process = subprocess.Popen([sys.executable, str(harness)], stdout=log, stderr=log)
    try:
        assert wait_until(lambda: ready_groups(tmp_path)), "six real groups required"
        if first == "INT":
            os.kill(process.pid, signal.SIGINT)
        else:
            (tmp_path / "exception").touch()
        assert wait_until(lambda: (tmp_path / "cleanup").exists())
        time.sleep(0.005 + (trial % 3) * 0.005)
        os.kill(process.pid, second)
        process.wait(timeout=10)
        groups = owned_groups(tmp_path)
        assert wait_until(
            lambda: not any(group_exists(g) for g in groups), timeout=1
        ), groups
        data = json.loads((tmp_path / "out/ledger.json").read_text())
        assert data["state"] == "interrupted"
        assert data["stop_reason"] == (
            "KeyboardInterrupt" if first == "INT" else "ValueError"
        )
        assert all(row["state"] != "running" for row in data["rows"])
        assert json.loads((tmp_path / "restored.json").read_text()) == [True] * 3
    finally:
        cleanup_cell(process, log, tmp_path)


@pytest.mark.parametrize("marker", [False, True])
def test_rate_exit_stops_peer_before_sleep(tmp_path, monkeypatch, marker):
    driver, clock, stopped, sleeps, polls = load("d2_driver"), Clock(), [], [], []
    out = tmp_path / "out"

    def producer(row, attempt, path):
        class Process:
            def poll(self):
                polls.append(row["arm"])
                if row["arm"] == "comparison":
                    return None
                if marker:
                    (path / "producer.log").write_text("usage-limit reached\n")
                return 0 if marker else 3

            def wait(self, timeout=None):
                pass

        return Process()

    def sleep(seconds):
        sleeps.append(json.loads((out / "ledger.json").read_text())["stop_reason"])
        clock.sleep(seconds)

    def stop(process):
        stopped.append((list(polls), list(sleeps)))

    monkeypatch.setattr(driver, "stop_process", stop)
    driver.run_comparison(
        dict(option="B", aa=False), out, producer=producer, clock=clock, sleep=sleep
    )
    assert sleeps == [], "rate exit must return to the stop at the next loop head"
    assert stopped == [(["candidate", "comparison"], [])]
    data = json.loads((out / "ledger.json").read_text())
    assert data["stop_reason"] == "rate_limited"
    assert data["rows"][0]["state"] == "rate_limited"
    assert data["rows"][1]["state"] == "truncated"
