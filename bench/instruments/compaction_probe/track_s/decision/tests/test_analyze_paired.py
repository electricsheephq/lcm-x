"""Offline regression for committed, failed, and held-pass latency accounting."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("placebo,gate,expected", [
    ((0.01, 10), (1.0, 0), "INCONCLUSIVE"),
    ((0.01, -10), (1.0, 0), "INCONCLUSIVE"),
    ((0.5, -10), (0.01, -10), "BLOCK"),
    ((1.0, 0), (1.0, 0), "PASS"),
    ((0.01, 10), (0.01, -10), "INCONCLUSIVE"),
])
def test_d2_placebo_verdict(placebo, gate, expected, capsys):
    """Exercise the analyzer's output path with controlled run-level test results."""
    path = Path(__file__).resolve().parents[1] / "analyze_paired.py"
    tree = ast.parse(path.read_text())
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "paired")

    def pair_axes(cp):
        p, effect = placebo if cp == 176 else gate
        return {"status": "COMPLETE", "missing_pairs": [],
                "run_level": dict(test="synthetic", n_seeds=3, n_runs_per_side=6,
                                  p_exact=p, effect_pts_exact=effect, run_shares={}),
                "facts_per_fact": dict.fromkeys(("n_facts", "wins", "losses", "ties", "effect_pts_exact",
                                                "fact_sign_flip_p_exact", "sign_test_p_exact", "diagnostic_only")),
                "axes": {"facts_all": {"p_exact": p}}}

    stats = {arm: {"facts_r1_r2_spread_by_cp": {f"cp-{cp}": {f"seed-{n}": 0 for n in (1, 2, 3)}
                                               for cp in (176, 304)}} for arm in ("prev", "cand")}
    ns = dict(CPS=(176, 304), SEEDS=(1, 2, 3), ARMS=("prev", "cand"),
              pair_axes=pair_axes, per_arm=lambda: stats, losses=lambda cp: {}, json=json)
    exec(compile(ast.Module(body=tree.body[start:], type_ignores=[]), str(path), "exec"), ns)
    d2 = json.loads(capsys.readouterr().out)["d2"]
    assert d2["verdict"] == expected
    assert d2["cp-176"]["blocks"] == (placebo[0] < 0.05 and placebo[1] <= -5)
    assert d2["cp-304"]["blocks"] == (gate[0] < 0.05 and gate[1] <= -5)
    assert d2["cp-176"]["placebo_fires"] == (placebo[0] < 0.05 and abs(placebo[1]) >= 5)
    assert "placebo_fires" not in d2["cp-304"]
    assert d2["d2_rule"] == "cp-304 gates; cp-176 is the reader placebo (v0.28.0, #1128)"


def test_failed_sweeps_require_summariser_calls(tmp_path):
    runs, logs, material = (tmp_path / name for name in ("runs", "logs", "material"))
    logs.mkdir()
    (material / "seed-1").mkdir(parents=True)
    (material / "seed-1" / "facts.json").write_text("[]")
    events = [
        {"is_compaction": True, "compress_wall_s": 10, "summariser_calls": [{}]},
        {"is_compaction": False, "compress_wall_s": 5, "summariser_calls": []},
        {"is_compaction": False, "compress_wall_s": 40, "summariser_calls": [{}]},
        {"is_compaction": False, "compress_wall_s": 0, "summariser_calls": [{}]},
    ]
    for arm in ("first", "second"):
        run = runs / arm / "seed-1" / "d1-r1"
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps({"events": [{**e, "levels": []} for e in events]}))
        (logs / f"s2-{arm}-d1-r1.log.wall").write_text("start 1\nexit 0 end 2\n")
    command = [
        sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "analyze_paired.py"),
        "--arms", "first", "second", "--seeds", "1", "--checkpoints", "304",
        "--run-root", str(runs), "--logs", str(logs), "--material", str(material),
        "--decision-root", str(tmp_path / "decision"),
    ]
    result = json.loads(subprocess.check_output(command, text=True))
    for stats in result["per_arm"].values():
        assert stats["compaction_wall_s"] == {"n": 1, "p50": 10, "p90": 10, "max": 10}
        assert stats["failed_sweep_count"] == 2
        assert stats["failed_sweep_wall_s"] == {"n": 2, "p50": 20, "p90": 40, "max": 40}
        assert stats["compaction_including_failed_sweeps_wall_s"] == {"n": 3, "p50": 10, "p90": 40, "max": 40}
