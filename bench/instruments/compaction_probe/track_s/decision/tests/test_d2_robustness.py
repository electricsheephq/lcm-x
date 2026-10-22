"""#1842: fake producers only; signal tests own every PID and process group."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from test_d2_calibrated import HERE, load
from test_d2_driver import Clock, Producer, run
from test_d2_integrity import prepared  # noqa: F401
from test_d2_replays import producer_env  # noqa: F401


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.02)
    return False


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_signal_cleans_real_descendant_groups_and_ledger(tmp_path, signum):
    # Each producer creates two independent child sessions, matching replay groups.
    child = "import time; time.sleep(60)"
    parent = ("import json,subprocess,sys,time; from pathlib import Path; "
              f"ps=[subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True) for _ in range(2)]; "
              "Path(sys.argv[1]).write_text(json.dumps([p.pid for p in ps])); time.sleep(60)")
    harness = tmp_path / "driver.py"
    harness.write_text(
        "import subprocess,sys; from pathlib import Path\n"
        f"sys.path.insert(0,{str(HERE)!r})\nimport d2_driver\n"
        "def producer(row, attempt, path):\n"
        f"    p = subprocess.Popen([sys.executable,'-c',{parent!r},str(path/'groups.json')],start_new_session=True)\n"
        "    (path / 'producer.pid').write_text(str(p.pid))\n    return p\n"
        f"d2_driver.run_comparison(dict(option='B',aa=False),Path({str(tmp_path / 'out')!r}),producer=producer)\n")
    process = subprocess.Popen([sys.executable, str(harness)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    groups = []
    try:
        def ready():
            nonlocal groups
            files = list((tmp_path / "out").rglob("groups.json"))
            try:
                groups = [g for f in files for g in json.loads(f.read_text())]
            except ValueError:
                return False
            return len(files) == 2 and len(groups) == 4
        assert wait_until(ready), "fake producer must spawn real child groups"
        assert all(group_exists(g) for g in groups)
        os.kill(process.pid, signum)
        process.wait(timeout=15)
        data = json.loads((tmp_path / "out/ledger.json").read_text())
        assert wait_until(lambda: not any(group_exists(g) for g in groups))
        assert data["state"] == "interrupted"
        assert signal.Signals(signum).name in data["stop_reason"]
        assert all(r["state"] != "running" for r in data["rows"])
    finally:
        # Base RED leaves orphans; cleanup only the exact test-owned groups.
        for g in groups:
            if group_exists(g):
                os.killpg(g, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        for f in (tmp_path / "out").rglob("producer.pid"):
            try:
                os.killpg(int(f.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_signal_handlers_restored(tmp_path):
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    seen = []
    fake = Producer()
    def producer(*args):
        seen.append({s: signal.getsignal(s) for s in previous})
        return fake(*args)
    driver, clock = load("d2_driver"), Clock()
    driver.run_comparison(dict(option="B", aa=False), tmp_path / "out", producer=producer,
                          validate=fake.validate, clock=clock, sleep=clock.sleep)
    assert all(seen[0][s] != previous[s] for s in previous)
    assert {s: signal.getsignal(s) for s in previous} == previous


def test_reaped_limit_producer_never_stopped(tmp_path, monkeypatch):
    driver, fake, stopped, clock = load("d2_driver"), Producer(rate_exit=True), [], Clock()
    original = driver.stop_process
    def stop(process):
        stopped.append(process)
        original(process)
    monkeypatch.setattr(driver, "stop_process", stop)
    driver.run_comparison(dict(option="B", aa=False), tmp_path / "out", producer=fake,
                          validate=fake.validate, clock=clock, sleep=clock.sleep)
    assert not stopped
    assert not fake.active


@pytest.mark.parametrize("legacy", [False, True])
def test_hash_ruling_is_separate_and_recorded(prepared, legacy):  # noqa: F811
    driver, config, _, _ = prepared
    config["expected_instrument_hash"] = "0" * 64
    if legacy:
        config["comparability_ruling"] = "https://example.invalid/route"
    with pytest.raises(ValueError, match="instrument_hash_ruling"):
        driver.prepare(config, "B", True)
    config["instrument_hash_ruling"] = "https://example.invalid/instrument"
    pin = driver.prepare(config, "B", True)
    assert pin["instrument_hash_ruling"] == config["instrument_hash_ruling"]
    assert pin["comparability_ruling"] == config.get("comparability_ruling")
    changed_route = copy.deepcopy(config)
    changed_route.pop("comparability_ruling", None)
    changed_route["summary"].update(provider="codex", model="gpt-6.1-sol")
    with pytest.raises(ValueError, match="comparability_ruling"):
        driver.prepare(changed_route, "B", True)


def test_print_hash_cli_matches_prepare_without_run(prepared, tmp_path):  # noqa: F811
    driver, config, kit, _ = prepared
    for name in ("d2_driver.py", "d2_score.py"):
        (kit / "decision" / name).write_bytes((HERE / name).read_bytes())
    subprocess.run(["git", "-C", str(kit), "add", "."], check=True)
    subprocess.run(["git", "-C", str(kit), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "driver fixture"], check=True)
    for build in config["builds"].values():
        build["sha"] = driver.git(kit, "rev-parse", "HEAD")
    config.pop("expected_instrument_hash")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, str(kit / "decision/d2_driver.py"), "--config", str(path),
                             "--print-instrument-hash", "--aa"], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 1 and len(result.stdout.strip()) == 64
    config["expected_instrument_hash"] = result.stdout.strip()
    assert driver.prepare(config, "B", True)["instrument_hash"] == result.stdout.strip()
    assert not list(tmp_path.rglob("ledger.json"))


def test_monitor_errors_bounded_and_dropped_counted(tmp_path, monkeypatch):
    driver, clock, fake = load("d2_driver"), Clock(), Producer()
    original = Path.open
    failures = []
    def unreadable(path, *args, **kwargs):
        if path.name == "producer.log" and args and args[0] == "rb":
            failures.append(path)
            raise PermissionError("synthetic unreadable log")
        return original(path, *args, **kwargs)
    def producer(*args):
        process = fake(*args)
        (args[2] / "producer.log").write_text("synthetic")
        return process
    monkeypatch.setattr(Path, "open", unreadable)
    driver.run_comparison(dict(option="B", aa=False), tmp_path / "out", producer=producer,
                          validate=fake.validate, clock=clock, sleep=clock.sleep)
    data = json.loads((tmp_path / "out/ledger.json").read_text())
    assert len(data["monitor_errors"]) == 50
    assert data["monitor_errors_dropped"] == len(failures) // 2 - 50
    assert data["monitor_errors_dropped"] > 0


def test_completed_fake_run_has_validated_result_card(tmp_path):
    run(tmp_path)
    root = tmp_path / "comparison"
    tool = os.environ["FMP_RESULT_CARD_TOOL"]
    assert (root / "RESULT.md").is_file()
    assert (root / "result.json").is_file() and (root / "result.md").is_file()
    cmd = [sys.executable, tool, "check", str(root), "--strict"]
    valid = subprocess.run(cmd, text=True, capture_output=True, timeout=10)
    assert valid.returncode == 0, valid.stdout + valid.stderr
    manifest = root / "RESULT-MANIFEST.json"
    data = json.loads(manifest.read_text())
    data.pop("code")
    manifest.write_text(json.dumps(data))
    corrupt = subprocess.run(cmd, text=True, capture_output=True, timeout=10)
    assert corrupt.returncode != 0


def test_new_untracked_file_stops_mid_run(prepared, tmp_path):  # noqa: F811
    driver, config, kit, _ = prepared
    pin, fake, clock = driver.prepare(config, "B", True), Producer(), Clock()
    def producer(*args):
        process = fake(*args)
        (kit / "decision/new_module.py").write_text("# synthetic drift\n")
        return process
    result = driver.run_comparison(pin, tmp_path / "out", producer=producer, validate=fake.validate,
                                   clock=clock, sleep=clock.sleep)
    data = json.loads((tmp_path / "out/ledger.json").read_text())
    assert result["raw_verdict"] == "INCONCLUSIVE"
    assert data["stop_reason"] == "instrument_changed"
    assert len(fake.started) == 1 and not fake.active


@pytest.mark.parametrize("cause", ["marker", "deadline"])
def test_post_launch_stop_precedes_any_sleep(tmp_path, cause):
    driver, fake, clock, sleeps = load("d2_driver"), Producer(rate=cause == "marker"), Clock(), []
    def producer(*args):
        process = fake(*args)
        if cause == "deadline":
            clock.now = driver.LIMIT_SECONDS
        return process
    def sleep(seconds):
        sleeps.append(seconds)
    driver.run_comparison(dict(option="B", aa=False), tmp_path / "out", producer=producer,
                          validate=fake.validate, clock=clock, sleep=sleep)
    data = json.loads((tmp_path / "out/ledger.json").read_text())
    assert not sleeps
    assert data["stop_reason"] == ("rate_limited" if cause == "marker" else "elapsed stop")
    assert not fake.active and len(fake.started) == 1
