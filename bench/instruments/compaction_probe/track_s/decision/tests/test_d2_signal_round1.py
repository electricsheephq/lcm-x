"""#1846 round 1: owned fake process groups only, no providers or credentials."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from test_d2_calibrated import HERE, load
from test_d2_driver import Clock, Producer
from test_d2_integrity import prepared  # noqa: F401
from test_d2_replays import producer_env  # noqa: F401
from test_d2_robustness import group_exists, wait_until


def signal_cell(tmp_path, *, nohup=False):
    child = "import time; time.sleep(60)"
    parent = ("import json,subprocess,sys,time; from pathlib import Path; "
              f"ps=[subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True) for _ in range(2)]; "
              "Path(sys.argv[1]).write_text(json.dumps([p.pid for p in ps])); time.sleep(60)")
    harness = tmp_path / "driver.py"
    harness.write_text(
        "import signal,subprocess,sys,time; from pathlib import Path\n"
        f"sys.path.insert(0,{str(HERE)!r})\nimport d2_driver\n"
        f"signal.signal(signal.SIGHUP, signal.{'SIG_IGN' if nohup else 'SIG_DFL'})\n"
        "signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
        "original = d2_driver.stop_process\n"
        "def slow_stop(p):\n"
        f"    Path({str(tmp_path / 'cleanup')!r}).touch()\n"
        "    time.sleep(.1)\n    original(p)\n"
        "d2_driver.stop_process = slow_stop\n"
        "def producer(row, attempt, path):\n"
        f"    p = subprocess.Popen([sys.executable,'-c',{parent!r},str(path/'groups.json')],start_new_session=True)\n"
        "    (path / 'producer.pid').write_text(str(p.pid))\n    return p\n"
        f"d2_driver.run_comparison(dict(option='B',aa=False),Path({str(tmp_path / 'out')!r}),producer=producer)\n")
    log = (tmp_path / "driver.log").open("w")
    process = subprocess.Popen([sys.executable, str(harness)], stdout=log, stderr=log)
    return process, log


def owned_groups(tmp_path):
    groups = []
    for path in (tmp_path / "out").rglob("groups.json"):
        groups.extend(json.loads(path.read_text()))
    for path in (tmp_path / "out").rglob("producer.pid"):
        groups.append(int(path.read_text()))
    return groups


def ready_groups(tmp_path):
    try:
        groups = owned_groups(tmp_path)
        return len(groups) == 6 and all(group_exists(g) for g in groups)
    except ValueError:
        return False


def cleanup_cell(process, log, tmp_path):
    # RED runs may orphan groups; kill only the exact groups this cell created.
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)
    for group in owned_groups(tmp_path):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    log.close()


def assert_clean_interrupt(process, tmp_path, first):
    process.wait(timeout=10)
    groups = owned_groups(tmp_path)
    survivors = [g for g in groups if group_exists(g)]
    assert wait_until(lambda: not any(group_exists(g) for g in groups), timeout=1), survivors
    data = json.loads((tmp_path / "out/ledger.json").read_text())
    assert data["state"] == "interrupted"
    assert data["stop_reason"] == signal.Signals(first).name
    assert all(r["state"] != "running" for r in data["rows"])


def test_nohup_survives_hup_then_term_cleans_all_groups(tmp_path):
    process, log = signal_cell(tmp_path, nohup=True)
    try:
        assert wait_until(lambda: ready_groups(tmp_path)), "fake producers must create six real groups"
        os.kill(process.pid, signal.SIGHUP)
        time.sleep(.15)
        assert process.poll() is None, "inherited ignored SIGHUP must remain ignored"
        assert all(group_exists(g) for g in owned_groups(tmp_path))
        assert not (tmp_path / "cleanup").exists()
        os.kill(process.pid, signal.SIGTERM)
        assert_clean_interrupt(process, tmp_path, signal.SIGTERM)
    finally:
        cleanup_cell(process, log, tmp_path)


@pytest.mark.parametrize("first,second", [(signal.SIGTERM, signal.SIGTERM),
                                         (signal.SIGHUP, signal.SIGHUP),
                                         (signal.SIGHUP, signal.SIGTERM)])
@pytest.mark.parametrize("trial", range(6))
def test_double_signal_finishes_every_group(tmp_path, first, second, trial):
    process, log = signal_cell(tmp_path)
    try:
        assert wait_until(lambda: ready_groups(tmp_path)), "fake producers must create six real groups"
        os.kill(process.pid, first)
        time.sleep(.005 + (trial % 3) * .005)
        os.kill(process.pid, second)
        assert wait_until(lambda: (tmp_path / "cleanup").exists())
        assert_clean_interrupt(process, tmp_path, first)
    finally:
        cleanup_cell(process, log, tmp_path)


@pytest.mark.parametrize("ignored", [signal.SIGTERM, signal.SIGHUP])
def test_ignored_disposition_and_cleanup_handlers_restored(ignored):
    driver = load("d2_driver")
    signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    previous = {s: signal.getsignal(s) for s in signals}
    try:
        signal.signal(ignored, signal.SIG_IGN)
        handled = signal.SIGHUP if ignored == signal.SIGTERM else signal.SIGTERM
        signal.signal(handled, signal.SIG_DFL)
        before = {s: signal.getsignal(s) for s in signals}
        with pytest.raises(driver.RunSignal):
            with driver.run_signals():
                assert signal.getsignal(ignored) == signal.SIG_IGN
                signal.getsignal(handled)(handled, None)
        assert {s: signal.getsignal(s) for s in signals} == before
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def test_first_signal_ignores_term_hup_and_int_during_cleanup():
    driver = load("d2_driver")
    with pytest.raises(driver.RunSignal):
        with driver.run_signals():
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    # Inspect inside the context too, before restoration.
    with driver.run_signals():
        with pytest.raises(driver.RunSignal):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert all(signal.getsignal(s) == signal.SIG_IGN
                   for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT))


@pytest.mark.parametrize("cause", ["interrupt", "limit"])
def test_stop_error_still_stops_remaining_producers(tmp_path, monkeypatch, cause):
    driver, fake, clock, stopped = load("d2_driver"), Producer(), Clock(), []
    original = driver.stop_process
    def stop(process):
        original(process)
        stopped.append(process)
        if len(stopped) == 1:
            raise RuntimeError("synthetic stop error after process exited")
    def sleep(seconds):
        if cause == "interrupt":
            raise KeyboardInterrupt()
        clock.now = driver.LIMIT_SECONDS
    monkeypatch.setattr(driver, "stop_process", stop)
    with pytest.raises(RuntimeError, match="synthetic stop error"):
        driver.run_comparison(dict(option="B", aa=False), tmp_path / "out", producer=fake,
                              validate=fake.validate, clock=clock, sleep=sleep)
    assert len(stopped) == 2 and not fake.active


@pytest.mark.parametrize("bad_tool", ["unset", "missing", "directory"])
def test_card_tool_preflight_launches_nothing(prepared, tmp_path, monkeypatch, bad_tool):  # noqa: F811
    driver, config, _, _ = prepared
    launched = []
    if bad_tool == "unset":
        monkeypatch.delenv("FMP_RESULT_CARD_TOOL", raising=False)
    else:
        monkeypatch.setenv("FMP_RESULT_CARD_TOOL", str(tmp_path if bad_tool == "directory" else tmp_path / "missing.py"))
    def producer(*args):
        launched.append(args)
        raise RuntimeError("producer launched before card preflight")
    with pytest.raises(ValueError, match="FMP_RESULT_CARD_TOOL"):
        driver.run_comparison(driver.prepare(config, "B", True), tmp_path / "out", producer=producer)
    assert not launched
    assert not (tmp_path / "out").exists()
