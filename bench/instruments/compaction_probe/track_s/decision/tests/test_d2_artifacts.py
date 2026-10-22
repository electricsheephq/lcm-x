"""Exercise real pin preparation and artifact validation with synthetic files."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from test_d2_calibrated import load
from test_d2_driver import Clock, Producer
from test_d2_replays import producer_env  # noqa: F401


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pins():
    return dict(option="B", aa=True, builds={a: dict(sha="a"*40, arm="LCMX-fleet-rc3")
                                           for a in ("candidate", "comparison")},
                materials={str(s): dict(manifest="b"*64, fact_ids=["f"]) for s in (1, 2, 3)},
                summary=dict(provider="codex", model="gpt-6.1-sol", effort="low", tier="default"),
                reader="glm-5.3", policy=dict(arm="LCMX-fleet-rc3"))


def artifacts(row, attempt, path, pin):
    seed, k = row["seed"], row["k"]
    label = f"d{seed}-r{k}" + (f"-a{attempt}" if attempt != 1 else "")
    if k != 1 or attempt != 1:
        path = path / ("replay-" + label.split("-", 1)[1])
    run = path / f"lcmx-runs/LCMX-fleet-rc3/seed-{seed}/{label}"
    write(run / "summary.json", dict(status="DONE", worktree_head="a"*40, stop_row_index=304,
                                      material_sha256="b"*64, lane="codex-sol", reader="glm",
                                      arm=dict(env={"LCM_SUMMARY_REASONING_EFFORT": "low"})))
    receipt = path / f"decision/logs/s2-LCMX-fleet-rc3-{label}.log.wall"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text("start 0 2026-01-01T00:00:00Z\nexit 0 end 1 2026-01-01T00:00:01Z\n")
    for cp in (176, 304):
        write(path / f"decision/scores/cp-{cp}/LCMX-fleet-rc3.seed-{seed}.{label}.json",
              dict(reader_readback=dict(model="glm-5.3", pin_ok=True), reader_errors=0,
                   metrics=dict(facts_kept=dict(complete=True, denominator=1, value=1)), probes=dict(f=dict(success=True))))
    return run, path


@pytest.mark.parametrize("k,attempt", [(1, 1), (9, 1), (4, 2)])
@pytest.mark.parametrize("failure,state", [(None, "valid"), ("truncated", "truncated"),
                                         ("missing", "missing"), ("build", "invalid"),
                                         ("reader", "invalid"), ("incomplete", "invalid")])
def test_artifact_validity(tmp_path, k, attempt, failure, state):
    driver, pin = load("d2_driver"), pins()
    row = dict(arm="candidate", seed=1, k=k)
    run, path = artifacts(row, attempt, tmp_path, pin)
    score = path / f"decision/scores/cp-176/LCMX-fleet-rc3.seed-1.d1-r{k}{'-a2' if attempt == 2 else ''}.json"
    data = json.loads(score.read_text())
    if failure == "truncated":
        data["metrics"]["facts_kept"]["reader_truncated"] = 1
    elif failure == "incomplete":
        data["metrics"]["facts_kept"]["complete"] = False
    elif failure == "reader":
        data["reader_readback"]["model"] = "wrong"
    elif failure == "missing":
        (run / "summary.json").unlink()
    elif failure == "build":
        summary = json.loads((run / "summary.json").read_text())
        summary["worktree_head"] = "wrong"
        write(run / "summary.json", summary)
    write(score, data)
    assert driver.validate_artifacts(pin, row, attempt, tmp_path)[0] == state


def test_comparison_with_real_validator_and_stub_producer(tmp_path):
    driver, pin, clock, fake = load("d2_driver"), pins(), Clock(), Producer()

    def produce(row, attempt, path):
        process = fake(row, attempt, path)
        artifacts(row, attempt, path, pin)
        return process
    result = driver.run_comparison(pin, tmp_path / "comparison", producer=produce, clock=clock, sleep=clock.sleep)
    assert result["raw_verdict"] == "PASS" and len(fake.started) == 54


def test_prepare_resolves_every_pin_and_refuses_route_drift(producer_env, monkeypatch):  # noqa: F811
    driver = load("d2_driver")
    env = producer_env
    kit = Path(env["KIT_DIR"])
    monkeypatch.setattr(driver, "__file__", str(kit / "decision/d2_driver.py"))
    material = Path(env["MATERIAL"])
    for seed in (1, 2, 3):
        folder = material / f"seed-{seed}"
        write(folder / "facts.json", [{"id": "f"}])
        write(folder / "material.manifest.json", dict(shas={"facts.json": hashlib.sha256((folder / "facts.json").read_bytes()).hexdigest()}))
    config = pins()
    config.update(material=str(material), builds={a: dict(worktree=str(kit), sha=env["PRODUCT_SHA"], arm="LCMX-fleet-rc3")
                                                 for a in ("candidate", "comparison")})
    result = driver.prepare(config, "B", True)
    assert result["instrument_hash"] and result["kit_sha"] == env["PRODUCT_SHA"]
    assert all(r["fact_ids"] == ["f"] for r in result["materials"].values())
    assert result["retry"] and result["summary"] and result["policy"] and result["reader"]
    with pytest.raises(ValueError, match="funding_refusal"):
        driver.prepare(config, "A", False)
    for key in ("summary", "reader", "policy"):
        changed = copy.deepcopy(config)
        changed[key] = {"effort": "wrong"} if key == "summary" else "wrong" if key == "reader" else dict(arm="wrong")
        with pytest.raises(ValueError):
            driver.prepare(changed, "B", True)
