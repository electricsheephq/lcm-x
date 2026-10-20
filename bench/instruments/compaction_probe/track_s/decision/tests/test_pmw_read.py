"""Synthetic PMW cell: no real material, model calls or credentials."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

DECISION = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pmw_read", DECISION / "pmw_read.py")
pmw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pmw)
ARM = "synthetic"
WANT = dict(leaf_chunk_tokens=24000, leaf_target_max_tokens=2000, summary_timeout_ms=120000)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def cell(tmp_path):
    root, material = tmp_path / "out", tmp_path / "material"
    for seed in (1, 2):
        transcript = material / f"seed-{seed}" / "transcript.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_text("\n".join(json.dumps(dict(turn=t)) for t in range(20)))
        run = root / "lcmx-runs" / ARM / f"seed-{seed}" / f"d{seed}-r1"
        events = [dict(turn=t, tokens_at_trigger=100, tokens_after=10,
                       compress_wall_s=1, leaves=1, levels=[dict(depth=0, source_tokens=90)],
                       summariser_calls=[dict(latency_s=0.5)]) for t in range(4)]
        write(run / "summary.json", dict(status="DONE", threshold_tokens=100,
                                        stop_row_index=19, events=events, replay_wall_s=5))
        write(run / "config.json", dict(config=WANT))
        (run / "engine.log").write_text("")
        for cp in (176, 304):
            write(root / "decision" / "scores" / f"cp-{cp}" / f"{ARM}.seed-{seed}.d{seed}-r1.json",
                  dict(metrics=dict(facts_kept=dict(value=0.583))))
    return root, material


def invoke(cell, *extra):
    root, material = cell
    output = root / "read.json"
    code = pmw.main(["--run-root", str(root), "--material", str(material), "--arm", ARM,
                     "--seeds", "1", "2", "--json", str(output), *extra])
    return code, json.loads(output.read_text())


def change_runs(cell, change):
    for path in cell[0].glob("lcmx-runs/*/seed-*/*/summary.json"):
        summary = json.loads(path.read_text())
        change(summary)
        write(path, summary)


def test_pass(cell):
    code, result = invoke(cell)
    assert code == 0
    assert result["verdict"]["result"] == "PASS"
    assert all(result["verdict"]["bars"].values())
    pooled = result["pooled"][ARM]
    assert pooled["c_per_100"] == 20
    assert pooled["waited_turn_share"] == 0.2
    assert pooled["mean_turn_wait_s"] == 0.2
    assert pooled["freed_per_event"] == 90


@pytest.mark.parametrize("bar", range(1, 7))
def test_each_bar_fails_alone(cell, bar):
    if bar == 1:
        change_runs(cell, lambda s: s["events"].extend([dict(s["events"][0]), dict(s["events"][0])]))
    elif bar == 2:
        change_runs(cell, lambda s: [e.update(compress_wall_s=80) for e in s["events"]])
    elif bar == 3:
        # Sub-second compactions sum to waited turns without adding C events.
        change_runs(cell, lambda s: s["events"].extend([
            dict(s["events"][0], turn=t, compress_wall_s=0.6) for t in (4, 4, 5, 5)]))
    elif bar == 4:
        for path in cell[0].glob("decision/scores/cp-304/*.json"):
            write(path, dict(metrics=dict(facts_kept=dict(value=0.582))))
    elif bar == 5:
        for path in cell[0].glob("lcmx-runs/*/seed-*/*/engine.log"):
            path.write_text("survival fit applied (reason=budget, conversation=synthetic, dropped_rows=1)\n")
    else:
        change_runs(cell, lambda s: s["events"][0].update(compress_wall_s=121))
    code, result = invoke(cell)
    assert code == 1
    assert result["verdict"]["result"] == "FAIL"
    assert [int(k[0]) for k, v in result["verdict"]["bars"].items() if not v] == [bar]


@pytest.mark.parametrize("key", WANT)
def test_config_mismatch_is_invalid(cell, key):
    path = next(cell[0].glob("lcmx-runs/*/seed-*/*/config.json"))
    write(path, dict(config=dict(WANT, **{key: 1})))
    code, result = invoke(cell)
    assert code == 2
    assert result["verdict"]["result"] == "INVALID"
    assert not result["config_check"]["valid"]


def test_missing_run_is_invalid(cell):
    next(cell[0].glob("lcmx-runs/*/seed-*/*/summary.json")).unlink()
    code, result = invoke(cell)
    assert code == 2
    assert result["verdict"]["result"] == "INVALID"


@pytest.mark.parametrize("mode", ["no-crossing", "zero-post-turns"])
def test_requested_run_requires_crossing_turns(cell, mode):
    # Leave the other run valid: one bad run invalidates the whole cell.
    path = cell[0] / "lcmx-runs" / ARM / "seed-1/d1-r1/summary.json"
    summary = json.loads(path.read_text())
    if mode == "no-crossing":
        summary["threshold_tokens"] = 101
    else:
        for event in summary["events"]:
            event["turn"] += 20
    write(path, summary)
    code, result = invoke(cell)
    assert code == 2
    assert result["verdict"]["result"] == "INVALID"
    assert f"{ARM}|d1-r1" in result["error"]
    assert result["verdict"]["facts_floor"] == 0.583
    assert ("no threshold crossing" if mode == "no-crossing" else "zero post-crossing turns") in result["error"]


@pytest.mark.parametrize("cp,expected", [(304, 2), (176, 0)])
def test_incomplete_checkpoint(cell, cp, expected):
    path = cell[0] / "decision/scores" / f"cp-{cp}" / f"{ARM}.seed-1.d1-r1.json"
    write(path, dict(metrics=dict(facts_kept=dict(value=0.583, complete=False))))
    code, result = invoke(cell)
    assert code == expected
    if cp == 304:
        assert result["verdict"]["result"] == "INVALID"
        assert f"{ARM}|d1-r1" in result["error"] and "incomplete cp-304" in result["error"]


@pytest.mark.parametrize("floor", [0.583, 0.584])
def test_receipt_records_effective_floor(cell, floor, capsys):
    _, result = invoke(cell, "--facts-floor", str(floor))
    assert result["verdict"]["facts_floor"] == floor
    assert json.loads(capsys.readouterr().out)["verdict"]["facts_floor"] == floor


@pytest.mark.parametrize("values,expected", [([], None), ([7], 7), (list(range(1, 11)), 9),
                                            ([11, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6], 10)])
def test_nearest_rank_p90(values, expected):
    assert pmw.p90(values) == expected


@pytest.mark.parametrize("value", [0, None, "missing"])
def test_cp176_never_gates(cell, value):
    for path in cell[0].glob("decision/scores/cp-176/*.json"):
        if value == "missing":
            path.unlink()
        else:
            write(path, dict(metrics=dict(facts_kept=dict(value=value))))
    code, result = invoke(cell)
    assert code == 0
    assert result["verdict"]["complete"]
    assert result["pooled"][ARM]["facts_176_mean"] == (0 if value == 0 else None)


def test_declared_floor_and_config(cell):
    for path in cell[0].glob("lcmx-runs/*/seed-*/*/config.json"):
        write(path, dict(config=dict(leaf_chunk_tokens=8000, leaf_target_max_tokens=1000, summary_timeout_ms=60000)))
    args = ("--leaf-chunk-tokens", "8000", "--leaf-target-max-tokens", "1000", "--summary-timeout-ms", "60000")
    assert invoke(cell, *args, "--facts-floor", "0.583")[0] == 0
    assert invoke(cell, *args, "--facts-floor", "0.584")[0] == 1


def test_pre_crossing_and_reported_figures(cell):
    def add_preflight(summary):
        summary["events"].insert(0, dict(turn=-1, tokens_at_trigger=50, compress_wall_s=999,
                                       summariser_calls=[dict(exception=True, error="timeout")]))
    change_runs(cell, add_preflight)
    code, result = invoke(cell)
    assert code == 0
    pooled = result["pooled"][ARM]
    assert pooled["timeouts"] == pooled["failures"] == 2
    assert pooled["wall_p90"] == 1
    assert pooled["survival_fits"] == 0


@pytest.mark.parametrize("mode,expected,fault", [("pass", 0, ""), ("pass", 0, "T"), ("pass", 0, "P"),
                                                ("fail", 7, ""), ("limit", 3, ""),
                                                ("dirty-tracked", 2, ""), ("dirty-untracked", 2, ""),
                                                ("short-sleep", 2, "T"), ("empty-sleep", 2, "P"),
                                                ("short-seeds", 0, ""), ("other-arm", 0, "")])
def test_runner_sequence_and_limit(tmp_path, mode, expected, fault):
    kit = tmp_path / "kit"
    runner = kit / "s2" / "run_s_lcmx.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("import sys, time\nprint('ARGS: ' + ' '.join(sys.argv[1:]), flush=True)\n"
                      "print('HTTP 429' if " + repr(mode) + " == 'limit' else 'replay', flush=True)\n"
                      "if " + repr(mode) + " == 'limit': time.sleep(20)\n"
                      "sys.exit(7 if " + repr(mode) + " == 'fail' else 0)\n")
    decision = kit / "decision"
    decision.mkdir()
    for filename in ("score_decision.py", "pmw_read.py"):
        (decision / filename).write_text(
            "import os\nfrom pathlib import Path\n"
            "import sys\n"
            f"with (Path(os.environ['OUT']) / 'sequence').open('a') as f: f.write('{filename} ' + ' '.join(sys.argv[1:]) + '\\n')\n")
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "SYSTEMROOT")}
    # Product and kit preflights use tiny committed repositories.
    subprocess.run(["git", "init", "-q", str(kit)], check=True)
    subprocess.run(["git", "-C", str(kit), "add", "s2", "decision"], check=True)
    subprocess.run(["git", "-C", str(kit), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-q", "-m", "synthetic kit"], check=True)
    kit_sha = subprocess.check_output(["git", "-C", str(kit), "rev-parse", "HEAD"], text=True).strip()
    if mode == "dirty-tracked":
        runner.write_text(runner.read_text() + "# dirty\n")
    elif mode == "dirty-untracked":
        (kit / "untracked.txt").write_text("dirty\n")
    product = tmp_path / "product"
    subprocess.run(["git", "init", "-q", str(product)], check=True)
    subprocess.run(["git", "-C", str(product), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-q", "--allow-empty", "-m", "fixture"], check=True)
    head = subprocess.check_output(["git", "-C", str(product), "rev-parse", "HEAD"], text=True).strip()
    env.update(PRODUCT_WT=str(product), PRODUCT_SHA=head, KIT_DIR=str(kit), MATERIAL=str(tmp_path / "material"),
               OUT=str(tmp_path / "out"), PY=sys.executable, GLM_API_KEY="synthetic-unused", SEEDS="1 2 3", PARALLEL="2",
               FACTS_FLOOR="0.772", FAULT=fault)
    if mode == "short-seeds":
        env["SEEDS"] = "1 2"
    elif mode == "other-arm":
        env["ARM"] = "synthetic"
    elif mode in ("short-sleep", "empty-sleep"):
        env["S2_FAULT_SLEEP_S"] = "0.01" if mode == "short-sleep" else ""
    result = subprocess.run(["bash", str(DECISION / "run_pmw_cell.sh")], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == expected, result.stderr
    sequence = Path(env["OUT"]) / "sequence"
    if mode in ("short-sleep", "empty-sleep"):
        assert "S2_FAULT_SLEEP_S" in result.stderr
        assert not Path(env["OUT"]).exists()
        return
    if mode.startswith("dirty-"):
        assert str(kit) in result.stderr
        assert not Path(env["OUT"]).exists()  # rejected before starting runs
        return
    receipt = json.loads((Path(env["OUT"]) / "cell.json").read_text())
    assert receipt == dict(product_sha=head, kit_sha=kit_sha, arm=env.get("ARM", "LCMX-fleet-rc3"),
                           seeds=[int(n) for n in env["SEEDS"].split()], facts_floor=0.772, fault=fault or None,
                           gating=mode not in ("short-seeds", "other-arm") and not fault)
    if mode in ("short-seeds", "other-arm") or fault:
        assert "NON-GATING:" in result.stderr
        assert ("FAULT" if fault else "SEEDS" if mode == "short-seeds" else "ARM") in result.stderr
    else:
        assert "NON-GATING:" not in result.stderr
    if expected == 0:
        steps = sequence.read_text().splitlines()
        assert [line.split()[0] for line in steps] == ["score_decision.py", "pmw_read.py"]
        assert "--facts-floor 0.772" in steps[1]  # the declared floor reaches the reader
        assert len(list((Path(env["OUT"]) / "decision/logs").glob("*.log.wall"))) == len(receipt["seeds"])
        for log in (Path(env["OUT"]) / "decision/logs").glob("*.log"):
            assert (f"--fault-hang {fault}" in log.read_text()) if fault else "--fault-hang" not in log.read_text()
    else:
        assert not sequence.exists()
    if mode == "limit":
        assert (Path(env["OUT"]) / "STOP").exists()
        assert not (Path(env["OUT"]) / "decision/logs/s2-LCMX-fleet-rc3-d3-r1.log").exists()


@pytest.mark.parametrize("fault", ["t", "X", "T P", "0"])
def test_runner_rejects_bad_fault(fault):
    result = subprocess.run(["bash", str(DECISION / "run_pmw_cell.sh")],
                            env=dict(PATH=os.environ["PATH"], FAULT=fault), capture_output=True, text=True)
    assert result.returncode == 2
    assert "FAULT" in result.stderr


def test_hang_metrics_are_reported_only(cell):
    def change(s):
        s["events"][0].update(compress_wall_s=100)
        s["events"][1].update(turn=0, compress_wall_s=101)
        s["events"].insert(0, dict(turn=-1, tokens_at_trigger=50, compress_wall_s=999))
        s["fault_hang"] = dict(mode="P", faulted_calls=2)
    change_runs(cell, change)
    _, result = invoke(cell)
    x = result["pooled"][ARM]
    assert (x["long_waits"], x["max_long_waits_in_one_turn"], x["max_turn_wait_s"]) == (4, 2, 201)
    assert (x["faulted_calls"], x["fault_mode"]) == (4, "P")
    for run in result["per_run"].values():
        assert (run["long_waits"], run["max_long_waits_in_one_turn"], run["max_turn_wait_s"]) == (2, 2, 201)
    assert len(result["verdict"]["bars"]) == 6


def test_missing_and_mixed_fault_modes(cell):
    _, result = invoke(cell)
    assert result["pooled"][ARM]["fault_mode"] is None
    assert result["pooled"][ARM]["faulted_calls"] == 0
    path = next(cell[0].glob("lcmx-runs/*/seed-*/*/summary.json"))
    s = json.loads(path.read_text())
    s["fault_hang"] = dict(mode="T", faulted_calls=1)
    write(path, s)
    _, result = invoke(cell)
    assert result["pooled"][ARM]["fault_mode"] == "mixed"


@pytest.mark.parametrize("missing", ["PRODUCT_WT", "PRODUCT_SHA", "KIT_DIR", "MATERIAL", "OUT", "PY", "GLM_API_KEY",
                                     "FACTS_FLOOR"])
def test_runner_requires_environment(missing):
    env = dict(PATH=os.environ["PATH"])
    env.update({key: "synthetic-unused" for key in ("PRODUCT_WT", "PRODUCT_SHA", "KIT_DIR", "MATERIAL", "OUT", "PY", "GLM_API_KEY",
                                                    "FACTS_FLOOR")})
    del env[missing]
    result = subprocess.run(["bash", str(DECISION / "run_pmw_cell.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert missing in result.stderr
