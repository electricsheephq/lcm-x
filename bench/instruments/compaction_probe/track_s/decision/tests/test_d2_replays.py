"""Replay selection and frozen default producer command/path golden."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_pmw_read import cell, invoke, ARM  # noqa: F401 (existing fixture, unchanged)
from test_d2_calibrated import load

HERE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("count", [4, 9])
def test_reader_selected_replays_and_legacy_default(cell, count):
    root, _ = cell
    for seed in (1, 2):
        source = root / "lcmx-runs" / ARM / f"seed-{seed}/d{seed}-r1"
        for k in range(2, count + 1):
            shutil.copytree(source, source.with_name(f"d{seed}-r{k}"))
            for cp in (176, 304):
                folder = root / "decision/scores" / f"cp-{cp}"
                shutil.copyfile(folder / f"{ARM}.seed-{seed}.d{seed}-r1.json",
                                folder / f"{ARM}.seed-{seed}.d{seed}-r{k}.json")
    code, result = invoke(cell, "--runs", *map(str, range(1, count + 1)))
    assert code == 0 and result["pooled"][ARM]["runs"] == 2 * count
    assert len(result["per_run"]) == 2 * count
    legacy = invoke(cell)[1]
    assert list(legacy["per_run"]) == [f"{ARM}|d{n}-r1" for n in (1, 2)]


@pytest.fixture
def producer_env(tmp_path):
    kit = tmp_path / "kit"
    (kit / "s2").mkdir(parents=True)
    (kit / "decision").mkdir()
    (kit / "s2/run_s_lcmx.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps({'argv':sys.argv[1:], 'out':os.environ['TRACK_S_OUT'], "
        "'tmp':os.environ['TMPDIR'], 'material':os.environ['TRACK_S_MATERIAL'], "
        "'product':os.environ['S2_PRODUCT_WORKTREE'], 'sha':os.environ['S2_PRODUCT_SHA']}))\n")
    for name in ("score_decision", "pmw_read"):
        (kit / f"decision/{name}.py").write_text(
            "import json, os, sys\nfrom pathlib import Path\n"
            f"with (Path(os.environ['OUT']) / 'commands.jsonl').open('a') as f: "
            f"f.write(json.dumps(['{name}', sys.argv[1:]]) + '\\n')\n")
    subprocess.run(["git", "init", "-q", str(kit)], check=True)
    subprocess.run(["git", "-C", str(kit), "add", "."], check=True)
    subprocess.run(["git", "-C", str(kit), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "stub"], check=True)
    sha = subprocess.check_output(["git", "-C", str(kit), "rev-parse", "HEAD"], text=True).strip()
    env = {k: os.environ[k] for k in ("PATH", "HOME")}
    env.update(PRODUCT_WT=str(kit), PRODUCT_SHA=sha, KIT_DIR=str(kit), MATERIAL=str(tmp_path / "material"),
               OUT=str(tmp_path / "out"), PY=sys.executable, GLM_API_KEY="synthetic-unused", FACTS_FLOOR="0.772",
               SEEDS="1", ARM="LCMX-fleet-rc3", PARALLEL="1")
    return env


def produce(env, *args):
    return subprocess.run(["bash", str(HERE / "run_pmw_cell.sh"), *map(str, args)],
                          env=env, text=True, capture_output=True, timeout=15)


def test_r1_frozen_base_golden(producer_env):
    env = producer_env
    result = produce(env)
    assert result.returncode == 0, result.stderr
    root, arm = Path(env["OUT"]), env["ARM"]
    # Frozen from 736af0f7: compare generated arguments and every generated path.
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) == [
        "cell.json", "commands.jsonl", f"decision/logs/s2-{arm}-d1-r1.log",
        f"decision/logs/s2-{arm}-d1-r1.log.wall", "score.log"]
    log = json.loads((root / f"decision/logs/s2-{arm}-d1-r1.log").read_text())
    assert log == dict(argv=["--arm", arm, "--seed", "1", "--run", "d1-r1", "--lane", "codex-sol",
                             "--reader", "glm", "--turn-end-notify", "--checkpoints", "176,304"],
                       out=env["OUT"], tmp=env["OUT"] + "/tmp", material=env["MATERIAL"],
                       product=env["PRODUCT_WT"], sha=env["PRODUCT_SHA"])
    assert [json.loads(s) for s in (root / "commands.jsonl").read_text().splitlines()] == [
        ["score_decision", ["--run-root", str(root / "lcmx-runs"), "--logs", str(root / "decision/logs"),
                            "--material", env["MATERIAL"], "--out", str(root / "decision"), "--arms", arm,
                            "--seeds", "1", "--checkpoints", "176", "304"]],
        ["pmw_read", ["--run-root", str(root), "--material", env["MATERIAL"], "--arm", arm,
                     "--seeds", "1", "--facts-floor", "0.772", "--json", str(root / "pmw-read.json")]]]


@pytest.mark.parametrize("k,attempt,label", [(2, 1, "d1-r2"), (2, 2, "d1-r2-a2"), (1, 2, "d1-r1-a2")])
def test_distinct_replay_attempt_paths_and_arguments(producer_env, k, attempt, label):
    env = producer_env
    result = produce(env, k, attempt)
    assert result.returncode == 0, result.stderr
    log = Path(env["OUT"]) / f"decision/logs/s2-{env['ARM']}-{label}.log"
    assert json.loads(log.read_text())["argv"][5] == label
    commands = (Path(env["OUT"]) / "commands.jsonl").read_text()
    assert '"--runs", "' + str(k) + '"' in commands
    assert '"--attempt", "' + str(attempt) + '"' in commands


def test_existing_paths_refused_before_any_overwrite(producer_env):
    env = producer_env
    assert produce(env).returncode == 0
    root = Path(env["OUT"])
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = produce(env)
    assert result.returncode != 0
    assert "existing" in result.stderr.lower()
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("args", [(0,), (10,), ("x",), (1, 0), (1, "x"), (1, 1, 1)])
def test_invalid_replay_arguments_refused(producer_env, args):
    assert produce(producer_env, *args).returncode == 2
    assert not Path(producer_env["OUT"]).exists()


def test_score_decision_selects_all_replays_and_attempts():
    import types
    module = load("score_decision")
    args = types.SimpleNamespace(material=Path("absent"), run_root=Path("runs"), external_root=Path("external"),
                                 arms=["LCMX-fleet-rc3"], checkpoints=[176, 304], runs=list(range(1, 10)), attempt=2)
    rows = list(module.runs(1, args))
    assert [r[1] for r in rows if r[0] == "LCMX-fleet-rc3"] == [f"d1-r{k}-a2" for k in range(1, 10)]
