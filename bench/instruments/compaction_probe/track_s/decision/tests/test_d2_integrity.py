"""The one integrity fix round: synthetic pins/processes only, no providers."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from test_d2_artifacts import artifacts, pins, write
from test_d2_calibrated import HERE, ledger, load
from test_d2_driver import Clock, Producer, run
from test_d2_replays import producer_env, produce  # noqa: F401


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_interrupt_kills_every_active_group_and_records_stop(tmp_path, exc):
    driver, processes = load("d2_driver"), []

    def producer(*args):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        processes.append(p)
        return p

    def interrupt(seconds):
        raise exc()

    try:
        with pytest.raises(exc):
            driver.run_comparison(dict(option="B", aa=False), tmp_path / "out", producer=producer,
                                  sleep=interrupt)
        assert len(processes) == 2 and all(p.poll() is not None for p in processes)
        data = json.loads((tmp_path / "out/ledger.json").read_text())
        assert exc.__name__ in data["stop_reason"]
        assert data["raw_verdict"] == "INCONCLUSIVE"
        assert all(r["state"] not in ("planned", "running") for r in data["rows"])
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=10)


@pytest.mark.parametrize("field", ["lane_model", "provider", "effort", "lane_effort", "tier", "service_tier"])
def test_served_route_mismatch_invalid(tmp_path, field):
    driver, pin = load("d2_driver"), pins()
    row = dict(arm="candidate", seed=1, k=1)
    root, _ = artifacts(row, 1, tmp_path, pin)
    summary = json.loads((root / "summary.json").read_text())
    summary["events"][0]["summariser_calls"][0][field] = "wrong"
    write(root / "summary.json", summary)
    assert driver.validate_artifacts(pin, row, 1, tmp_path)[0] == "invalid"


@pytest.fixture
def prepared(producer_env, monkeypatch):  # noqa: F811
    driver, env = load("d2_driver"), producer_env
    kit, material = Path(env["KIT_DIR"]), Path(env["MATERIAL"])
    monkeypatch.setattr(driver, "__file__", str(kit / "decision/d2_driver.py"))
    for seed in (1, 2, 3):
        folder = material / f"seed-{seed}"
        write(folder / "facts.json", [{"id": "f"}])
        (folder / "transcript.jsonl").write_text("synthetic\n")
        write(folder / "material.manifest.json", dict(shas={p: driver.digest(folder / p)
                                                         for p in ("facts.json", "transcript.jsonl")}))
    config = pins()
    config.update(material=str(material), builds={a: dict(worktree=str(kit), sha=env["PRODUCT_SHA"], arm="LCMX-fleet-rc3")
                                                 for a in ("candidate", "comparison")})
    config["expected_instrument_hash"] = __import__("hashlib").sha256(json.dumps(
        {p: driver.digest(kit / p) for p in driver.git(kit, "ls-files", "--", ".").splitlines()},
        sort_keys=True).encode()).hexdigest()
    return driver, config, kit, material


def test_prepare_requires_predeclared_kit_and_records_ruling(prepared):
    driver, config, _, _ = prepared
    config.pop("expected_instrument_hash")
    with pytest.raises(ValueError, match="instrument"):
        driver.prepare(config, "B", True)
    config["expected_instrument_hash"] = "0" * 64
    with pytest.raises(ValueError, match="instrument"):
        driver.prepare(config, "B", True)
    config["comparability_ruling"] = "https://example.invalid/ruling"
    pin = driver.prepare(config, "B", True)
    assert pin["comparability_ruling"] == config["comparability_ruling"]
    assert pin["expected_instrument_hash"] == "0" * 64


def test_prepare_refuses_option_a_aa(prepared):
    driver, config, _, _ = prepared
    config["funding_refusal"] = "https://example.invalid/refusal"
    with pytest.raises(ValueError, match="A/A"):
        driver.prepare(config, "A", True)


@pytest.mark.parametrize("kind", ["instrument", "material"])
@pytest.mark.parametrize("when", ["before_first", "during_replay", "before_score"])
def test_pin_drift_stops_comparison(prepared, tmp_path, kind, when):
    driver, config, kit, material = prepared
    pin, fake, clock = driver.prepare(config, "B", True), Producer(), Clock()
    target = kit / "s2/run_s_lcmx.py" if kind == "instrument" else material / "seed-1/transcript.jsonl"

    def mutate():
        target.write_text(target.read_text() + "changed\n")

    def producer(row, attempt, path):
        process = fake(row, attempt, path)
        if when == "during_replay" and len(fake.started) == 1:
            mutate()
        return process

    def validate(row, attempt, path):
        if when == "before_score" and len(fake.started) == 54 and row["slot"] == 2:
            mutate()
        return fake.validate(row, attempt, path)

    if when == "before_first":
        mutate()
    result = driver.run_comparison(pin, tmp_path / "out", producer=producer, validate=validate,
                                   clock=clock, sleep=clock.sleep)
    data = json.loads((tmp_path / "out/ledger.json").read_text())
    assert result["raw_verdict"] == "INCONCLUSIVE"
    assert data["stop_reason"] == kind + "_changed"
    assert not fake.active
    assert len(fake.started) == (0 if when == "before_first" else 1 if when == "during_replay" else 54)


def test_child_fault_environment_stripped(tmp_path, monkeypatch):
    driver, pin, captured = load("d2_driver"), pins(), {}
    pin.update(material="synthetic")
    pin["builds"]["candidate"]["worktree"] = "synthetic"
    for key in ("FAULT", "S2_FAULT_SLEEP_S", "S2_OTHER_FAULT"):
        monkeypatch.setenv(key, "injected")
    monkeypatch.setenv("D2_KEEP", "keep")
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **kw: captured.update(kw))
    driver.real_producer(pin)(dict(arm="candidate", seed=1, k=1), 1, tmp_path)
    assert not any(k == "FAULT" or k.startswith("S2_") for k in captured["env"])
    assert captured["env"]["D2_KEEP"] == "keep"


@pytest.mark.parametrize("k,attempt", [(1, 1), (9, 2)])
def test_fault_receipt_is_invalid(tmp_path, k, attempt):
    driver, pin, row = load("d2_driver"), pins(), dict(arm="candidate", seed=1, k=k)
    _, path = artifacts(row, attempt, tmp_path, pin)
    cell = next(path.glob("cell*.json"))
    write(cell, dict(fault="T"))
    assert driver.validate_artifacts(pin, row, attempt, tmp_path)[0] == "invalid"


def test_wall_clock_sleep_cannot_hide_elapsed(tmp_path):
    clock, wall = Clock(), Clock()
    clock.sleep = lambda seconds: setattr(wall, "now", 21600)
    result, data, producer = run(tmp_path, clock=clock, wall_clock=wall, wall_started=0)
    assert result["raw_verdict"] == "INCONCLUSIVE"
    assert data["elapsed_s"] == 21600 and len(producer.started) == 2
    assert not producer.active


def test_aa_cards_have_aa_claim(tmp_path):
    scorer = load("d2_score")
    result = scorer.score(ledger(aa=True))
    scorer.write_result(tmp_path, result)
    assert result["mode"] == "aa"
    assert "A/A" in result["claim"] and "non-inferiority" not in result["claim"]
    assert "A/A" in (tmp_path / "result.md").read_text()


def test_monitor_two_second_poll_and_atomic_ledger(tmp_path, monkeypatch):
    driver, clock, replaced = load("d2_driver"), Clock(), []
    replace = driver.os.replace

    def atomic(src, dst):
        json.loads(Path(src).read_text())
        replaced.append(Path(dst).name)
        replace(src, dst)

    monkeypatch.setattr(driver.os, "replace", atomic)
    sleeps = []
    clock.sleep = lambda seconds: sleeps.append(seconds)
    run(tmp_path, clock=clock)
    assert sleeps and set(sleeps) == {2}
    assert "ledger.json" in replaced


@pytest.mark.parametrize("failures", [1, 2])
def test_monitor_retries_once_and_records_error(tmp_path, monkeypatch, failures):
    fake, original_open, failed = Producer(rate=True), Path.open, []

    def flaky(path, *args, **kwargs):
        if path.name == "producer.log" and args and args[0] in ("r", "rb") and len(failed) < failures:
            failed.append(path)
            raise OSError("synthetic log race")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky)
    result, data, _ = run(tmp_path, fake)
    assert result["raw_verdict"] == "INCONCLUSIVE" and not fake.active
    assert len(failed) == failures
    if failures == 2:
        assert data["monitor_errors"]


def test_monitor_tails_offsets_and_split_marker(tmp_path, monkeypatch):
    fake, clock, seeks = Producer(), Clock(), []
    original, opened = fake.__call__, Path.open

    def producer(row, attempt, path):
        p = original(row, attempt, path)
        (path / "producer.log").write_text("x" * 10000 + "usage-")
        return p

    class Reader:
        def __init__(self, f):
            self.f = f
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.f.close()
        def __getattr__(self, key):
            return getattr(self.f, key)
        def seek(self, offset, *args):
            seeks.append(offset)
            return self.f.seek(offset, *args)

    def tracked(path, *args, **kwargs):
        f = opened(path, *args, **kwargs)
        return Reader(f) if path.name == "producer.log" and args and args[0] in ("r", "rb") else f

    def append(seconds):
        for path in tmp_path.rglob("producer.log"):
            with opened(path, "a") as f:
                f.write("limit reached\n")
        clock.sleep(seconds)

    monkeypatch.setattr(Path, "open", tracked)
    driver = load("d2_driver")
    result = driver.run_comparison(dict(option="B", aa=False), tmp_path / "out", producer=producer,
                                   validate=fake.validate, clock=clock, sleep=append)
    assert result["raw_verdict"] == "INCONCLUSIVE" and not fake.active
    assert 10006 in seeks


@pytest.mark.parametrize("existing", ["result.json", "result.md"])
def test_score_cli_exclusive_outputs(tmp_path, existing):
    write(tmp_path / "ledger.json", ledger())
    (tmp_path / existing).write_text("preserve")
    result = subprocess.run([sys.executable, str(HERE / "d2_score.py"), "--ledger", str(tmp_path / "ledger.json"),
                             "--out", str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert (tmp_path / existing).read_text() == "preserve"


@pytest.mark.parametrize("attempt", ["00", "01", "007"])
def test_leading_zero_attempt_refused(producer_env, attempt):  # noqa: F811
    result = produce(producer_env, 1, attempt)
    assert result.returncode == 2 and "attempt" in result.stderr
    assert not Path(producer_env["OUT"]).exists()


def test_scorer_runs_without_site_packages():
    result = subprocess.run([sys.executable, "-S", str(HERE / "d2_score.py"), "--help"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_option_a_spec_is_explicitly_two_sided():
    spec = HERE.parents[3] / "specs/RELEASE-READINESS-V1.md"
    text = spec.read_text().split("**Option-A contingency.**", 1)[1]
    assert "two-sided seed-stratified run-level permutation regression screen" in text
