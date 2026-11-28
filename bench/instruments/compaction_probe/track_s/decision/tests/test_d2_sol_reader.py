"""Sol reader pins and dry cell arguments; synthetic material and runners only."""
import json
from pathlib import Path

import pytest

from test_d2_artifacts import artifacts, pins, write
from test_d2_calibrated import load
from test_d2_integrity import prepared  # noqa: F401
from test_d2_replays import produce, producer_env  # noqa: F401


def test_reader_flags():
    assert load("d2_driver").READER_FLAGS == {"glm-5.3": "glm", "gpt-6.1-sol": "sol-low"}


@pytest.mark.parametrize("reader,ruling,accepted", [
    ("gpt-6.1-sol", "synthetic ruling", True), ("glm-5.3", None, True),
    ("gpt-6.1-sol", None, False), ("gpt-6.1-sol", "", False),
    ("gpt-6.1-sol", "   ", False), ("gpt-6.1-sol", True, False),
    ("gpt-6-sol", "synthetic ruling", False), ("gpt-6-astra", "synthetic ruling", False),
    ("", "synthetic ruling", False),
])
def test_prepare_reader(prepared, reader, ruling, accepted):  # noqa: F811
    driver, config, _, _ = prepared
    config.update(reader=reader)
    if ruling is not None:
        config["reader_comparability_ruling"] = ruling
    if not accepted:
        with pytest.raises(ValueError):
            driver.prepare(config, "B", True)
        return
    result = driver.prepare(config, "B", True)
    assert result["reader"] == reader and "reader_flag" not in result
    if reader == "gpt-6.1-sol":
        assert result["reader_comparability_ruling"] == ruling
    else:
        assert "reader_comparability_ruling" not in result


@pytest.mark.parametrize("reader,flag,effort,model,pin_ok,state", [
    ("gpt-6.1-sol", "sol-low", "low", "gpt-6.1-sol", True, "valid"),
    ("gpt-6.1-sol", "glm", "low", "gpt-6.1-sol", True, "invalid"),
    ("glm-5.3", "sol-low", "low", "glm-5.3", True, "invalid"),
    ("gpt-6.1-sol", "sol-low", "high", "gpt-6.1-sol", True, "invalid"),
    ("gpt-6.1-sol", "sol-low", None, "gpt-6.1-sol", True, "invalid"),
    ("gpt-6.1-sol", "sol-low", "low", "wrong", True, "invalid"),
    ("gpt-6.1-sol", "sol-low", "low", "gpt-6.1-sol", False, "invalid"),
    ("unknown", "glm", "low", "unknown", True, "invalid"),
    ("unknown", None, "low", "unknown", True, "invalid"),
])
@pytest.mark.parametrize("cp", [176, 304])
def test_replay_reader(tmp_path, reader, flag, effort, model, pin_ok, state, cp):
    driver, pin = load("d2_driver"), pins()
    pin["reader"] = reader
    row = dict(arm="candidate", seed=1, k=1)
    run, path = artifacts(row, 1, tmp_path, pin)
    summary = json.loads((run / "summary.json").read_text())
    summary["reader"] = flag
    write(run / "summary.json", summary)
    for checkpoint in (176, 304):
        score = path / f"decision/scores/cp-{checkpoint}/LCMX-fleet-rc3.seed-1.d1-r1.json"
        data = json.loads(score.read_text())
        data["reader_readback"] = dict(model=reader, effort="low", pin_ok=True)
        if checkpoint == cp:
            data["reader_readback"] = dict(model=model, pin_ok=pin_ok)
            if effort is not None:
                data["reader_readback"]["effort"] = effort
        write(score, data)
    assert driver.validate_artifacts(pin, row, 1, tmp_path)[0] == state


@pytest.mark.parametrize("reader,flag", [("glm-5.3", "glm"), ("gpt-6.1-sol", "sol-low")])
def test_launch_reader(tmp_path, monkeypatch, reader, flag):
    driver, pin, captured = load("d2_driver"), pins(), {}
    pin.update(reader=reader, material="synthetic")
    pin["builds"]["candidate"]["worktree"] = "synthetic"
    monkeypatch.setattr(driver.os, "environ", {})
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **kw: captured.update(kw))
    driver.real_producer(pin)(dict(arm="candidate", seed=1, k=1), 1, tmp_path)
    assert captured["env"]["READER"] == flag


@pytest.mark.parametrize("lane,reader,key,rc", [
    ("codex-sol", "sol-low", False, 0), ("codex-sol", "sol-low", True, 0),
    ("codex-sol", "bogus", True, 2), ("glm", "sol-low", False, 2),
    ("codex-sol", "glm", False, 2), ("glm", "glm", False, 2),
    ("glm", "sol-low", True, 0), ("codex-sol", None, True, 0),
])
def test_shell_reader(producer_env, tmp_path, lane, reader, key, rc):  # noqa: F811
    env = dict(producer_env, SUMMARY_LANE=lane)
    nice = tmp_path / "nice"  # Scheduling is outside this dry argument check.
    nice.write_text('#!/bin/sh\nshift 2\nexec "$@"\n')
    nice.chmod(0o755)
    env["PATH"] = str(tmp_path) + ":" + env["PATH"]
    if reader is not None:
        env["READER"] = reader
    if not key:
        env.pop("GLM_API_KEY")
    result = produce(env)
    assert result.returncode == rc, result.stderr
    root = Path(env["OUT"])
    if rc == 0:
        log = root / f"decision/logs/s2-{env['ARM']}-d1-r1.log"
        args = json.loads(log.read_text())["argv"]
        assert args[args.index("--reader") + 1] == (reader or "glm")
    else:
        assert ("unsupported reader" if reader == "bogus" else "GLM_API_KEY") in result.stderr
        assert not root.exists()
