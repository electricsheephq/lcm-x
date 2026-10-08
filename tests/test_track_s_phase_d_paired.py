"""Phase D (D2) release-over-release paired analysis: two trees, unrounded p, per-checkpoint spread, d2 verdict."""

import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1] / "bench/instruments/compaction_probe/track_s"
ANALYZE = TRACK / "decision/analyze_paired.py"
FACTS = [f"f{i}" for i in range(40)]
CPS = (176, 304)
SHAS = {"prev": "a1" * 20, "cand": "b2" * 20}


def kept_all(seed, run, cp):
    return set(FACTS)


def summary(arm, sha, seed, run):
    return {"events": [], "worktree_head": sha, "arm": {"name": arm, "kind": "plain", "env": {"LCM_X": "1"}},
            "fleet_keys_excluded": ["k"], "harness_overrides": {"h": 1}, "context_length": 200000,
            "lane": "glm", "reader": "glm"}


def build(root, arm, kept=kept_all, seeds=(1,), sha=SHAS["prev"], value=None, summarize=summary):
    """One tree: runs/<arm>/seed-N/dN-rX, decision/scores (admitted by manifest), logs/*.wall."""
    runs, decision, logs = root / "runs", root / "decision", root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest_path = decision / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema": "track-s-score-manifest-v2", "entries": {}}
    for seed in seeds:
        for run in ("r1", "r2"):
            rundir = runs / arm / f"seed-{seed}" / f"d{seed}-{run}"
            for cp in CPS:
                (rundir / f"cp-{cp}").mkdir(parents=True, exist_ok=True)
                (rundir / f"cp-{cp}" / "summary.json").write_text(json.dumps({"events": []}))
                ids = kept(seed, run, cp)
                payload = {
                    "probes": {f: {"class": "CORRECT" if f in ids else "WRONG"} for f in FACTS},
                    "metrics": {"facts_kept": {"complete": True,
                                               "value": value(seed, run, cp) if value else len(ids) / len(FACTS)},
                                "continuation": {"complete": True}, "continuity": {"complete": True, "grid": []}},
                    "stored_level3": {"level3": 0, "leaves": 1},
                }
                key = f"scores/cp-{cp}/{arm}.seed-{seed}.d{seed}-{run}.json"
                (decision / key).parent.mkdir(parents=True, exist_ok=True)
                (decision / key).write_text(json.dumps(payload))
                manifest["entries"][key] = {"sha256": hashlib.sha256((decision / key).read_bytes()).hexdigest()}
            (rundir / "summary.json").write_text(json.dumps(summarize(arm, sha, seed, run)))
            (logs / f"s2-{arm}-d{seed}-{run}.log.wall").write_text("start 1\nexit 0 end 2\n")
    manifest_path.write_text(json.dumps(manifest))
    return runs, decision, logs


def material(tmp_path, seeds=(1,)):
    root = tmp_path / "material"
    for seed in seeds:
        (root / f"seed-{seed}").mkdir(parents=True, exist_ok=True)
        (root / f"seed-{seed}" / "facts.json").write_text(json.dumps(
            [{"id": f, "placement": "head", "class": "early_user_constraint"} for f in FACTS]))
    return root


def analyze(tmp_path, arms, first, second=None, labels=None, check=True, shas=None):
    runs, decision, logs = first
    command = [sys.executable, "-B", str(ANALYZE), "--arms", *arms, "--seeds", "1",
               "--checkpoints", *map(str, CPS), "--run-root", str(runs), "--decision-root", str(decision),
               "--logs", str(logs), "--material", str(material(tmp_path))]
    if second:
        command += ["--second-tree", *map(str, second)]
    if labels:
        command += ["--labels", *labels]
    if shas:
        command += ["--expect-shas", *shas]
    result = subprocess.run(command, capture_output=True, text=True)
    if not check:
        return result
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def two_tree(tmp_path, cand_kept=kept_all, prev_kept=kept_all, shas=(SHAS["prev"][:7], SHAS["cand"][:12]), **cand_options):
    prev = build(tmp_path / "prev", "LCMX-fleet", prev_kept)
    cand = build(tmp_path / "cand", "LCMX-fleet", cand_kept, sha=SHAS["cand"], **cand_options)
    return analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"], shas=shas)


def test_two_tree_pairs_same_arm_across_roots(tmp_path):
    out = two_tree(tmp_path, cand_kept=lambda *a: set(FACTS) - {"f1", "f2"},
                   prev_kept=lambda *a: set(FACTS) - {"f0"})
    assert out["arms"] == ["LCMX-fleet", "LCMX-fleet"] and out["labels"] == ["prev", "cand"]
    assert out["trees"]["prev"]["arm"] == out["trees"]["cand"]["arm"] == "LCMX-fleet"
    assert out["trees"]["prev"]["decision_root"].endswith("prev/decision")
    assert out["trees"]["cand"]["run_root"].endswith("cand/runs")
    assert out["trees"]["cand"]["logs"].endswith("cand/logs")
    for cp in CPS:
        result = out[f"cp-{cp}"]
        assert result["pairs"] == ["seed-1/r1", "seed-1/r2"] and result["status"] == "COMPLETE"
        facts = result["axes"]["facts_all"]
        assert (facts["n"], facts["b_v1_only"], facts["c_v2_only"]) == (80, 4, 2)
        assert set(result["loss_classes"]) == {"prev", "cand"}
    assert set(out["per_arm"]) == {"prev", "cand"}


def test_two_tree_requires_distinct_labels(tmp_path):
    prev = build(tmp_path / "prev", "LCMX-fleet")
    cand = build(tmp_path / "cand", "LCMX-fleet")
    for labels in (None, ["same", "same"]):
        result = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=labels, check=False, shas=("abcdef0",) * 2)
        assert result.returncode != 0 and "labels must be distinct" in result.stderr


def test_p_exact_is_unrounded(tmp_path):
    tree = ast.parse(ANALYZE.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mcnemar"]
    ns = {"math": math}
    exec(compile(tree, str(ANALYZE), "exec"), ns)
    assert ns["mcnemar"](117, 150) < 0.05 and round(ns["mcnemar"](117, 150), 4) == 0.05
    out = two_tree(tmp_path, cand_kept=lambda *a: set(FACTS) - {"f1", "f2", "f3"})
    for cp in CPS:
        facts = out[f"cp-{cp}"]["axes"]["facts_all"]
        assert facts["p_exact"] == ns["mcnemar"](facts["b_v1_only"], facts["c_v2_only"])
        sign = facts["sign_test"]
        assert sign["p_exact"] == ns["mcnemar"](sign["wins"], sign["losses"])
        assert out["d2"][f"cp-{cp}"]["p_exact"] == facts["p_exact"]


def test_spread_is_reported_at_every_checkpoint(tmp_path):
    out = two_tree(tmp_path, cand_kept=lambda seed, run, cp: set(FACTS[:30]) if (cp, run) == (176, "r2") else set(FACTS))
    cand = out["per_arm"]["cand"]
    assert cand["facts_r1_r2_spread_by_cp"] == {"cp-176": {"seed-1": 0.25}, "cp-304": {"seed-1": 0.0}}
    assert cand["facts_cp304_r1_r2_spread"] == {"seed-1": 0.0}
    assert cand["spread_over_0.10"] == ["cp-176/seed-1"] and cand["spread_unmeasured"] == []
    assert out["per_arm"]["prev"]["spread_over_0.10"] == []
    assert out["d2"]["spread_over_0.10"] == {"prev": [], "cand": ["cp-176/seed-1"]}
    assert out["d2"]["verdict"] == "REPEAT_SEEDS"


def test_d2_blocks_on_a_significant_loss_of_five_points(tmp_path):
    out = two_tree(tmp_path, cand_kept=lambda *a: set(FACTS[:30]))
    for cp in CPS:
        d2 = out["d2"][f"cp-{cp}"]
        assert d2["effect_pts_exact"] == -25.0 and d2["p_exact"] < 0.05
        assert d2["net_loss_ge_5"] is True and d2["blocks"] is True
    assert out["d2"]["candidate"] == "cand" and out["d2"]["status"] == "COMPLETE"
    assert out["d2"]["verdict"] == "BLOCK"


def test_d2_passes_a_small_loss(tmp_path):
    out = two_tree(tmp_path, cand_kept=lambda seed, run, cp: set(FACTS[1:]) if run == "r1" else set(FACTS))
    for cp in CPS:
        d2 = out["d2"][f"cp-{cp}"]
        assert d2["effect_pts_exact"] == -1.25 and d2["p_exact"] == 1.0
        assert d2["net_loss_ge_5"] is False and d2["blocks"] is False
    assert out["d2"]["verdict"] == "PASS"


def test_d2_incomplete_when_a_pair_is_missing(tmp_path):
    prev = build(tmp_path / "prev", "LCMX-fleet")
    cand = build(tmp_path / "cand", "LCMX-fleet", lambda *a: set(FACTS[:30]), sha=SHAS["cand"])
    (cand[1] / "scores/cp-304/LCMX-fleet.seed-1.d1-r2.json").unlink()
    out = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"],
                  shas=(SHAS["prev"], SHAS["cand"]))
    assert out["status"] == "INCOMPLETE" and out["d2"]["verdict"] == "INCOMPLETE"


def test_single_tree_output_is_unchanged(tmp_path):
    root = tmp_path / "one"
    build(root, "A", lambda *a: set(FACTS) - {"f0"})
    tree = build(root, "B", lambda seed, run, cp: set(FACTS) - {"f1", "f2"} if run == "r1" else set(FACTS))
    out = analyze(tmp_path, ["A", "B"], tree)
    assert out["arms"] == ["A", "B"] and set(out["per_arm"]) == {"A", "B"}
    for cp in CPS:
        facts = out[f"cp-{cp}"]["axes"]["facts_all"]
        assert {k: facts[k] for k in ("n", "excluded", "v1", "v2", "effect_pts", "b_v1_only", "c_v2_only", "p")} == {
            "n": 80, "excluded": 0, "v1": 0.975, "v2": 0.975, "effect_pts": 0.0, "b_v1_only": 2, "c_v2_only": 2, "p": 1.0}
        assert {k: facts["sign_test"][k] for k in ("wins", "losses", "ties", "p")} == {
            "wins": 1, "losses": 1, "ties": 0, "p": 1.0}
        assert set(out[f"cp-{cp}"]["loss_classes"]) == {"A", "B"}
    assert out["per_arm"]["A"]["facts_cp304_r1_r2_spread"] == {"seed-1": 0.0}
    assert out["per_arm"]["B"]["facts_cp304_r1_r2_spread"] == {"seed-1": 0.05}
    # New in single-tree mode: labels default to the arms, d2 is reported, trees are not. Two different arms are
    # compared on purpose there, so the one-configuration rule (a D2, two-tree rule) is not checked.
    assert out["labels"] == ["A", "B"] and "trees" not in out
    assert out["d2"]["config_equal"] is None and "config_diff_keys" not in out["d2"]
    assert out["d2"]["expected_shas"] is None and out["d2"]["sha_mismatch"] == []


def test_spread_threshold_uses_the_unrounded_spread(tmp_path):
    out = two_tree(tmp_path, value=lambda seed, run, cp: 0.6004 if (cp, run) == (176, "r2") else 0.5)
    cand = out["per_arm"]["cand"]
    assert cand["facts_r1_r2_spread_by_cp"]["cp-176"] == {"seed-1": 0.1}
    assert cand["spread_over_0.10"] == ["cp-176/seed-1"] and out["d2"]["verdict"] == "REPEAT_SEEDS"


def test_d2_incomplete_when_receipts_are_missing(tmp_path):
    prev = build(tmp_path / "prev", "LCMX-fleet")
    runs, decision, _ = build(tmp_path / "cand", "LCMX-fleet", sha=SHAS["cand"])
    (tmp_path / "no-receipts").mkdir()
    out = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, (runs, decision, tmp_path / "no-receipts"),
                  labels=["prev", "cand"], shas=(SHAS["prev"], SHAS["cand"]))
    assert out["status"] == "COMPLETE"  # both score files are admitted; only the receipts are missing
    assert out["d2"]["missing_spread"] == ["cand/cp-176/seed-1", "cand/cp-304/seed-1"]
    assert out["d2"]["verdict"] == "INCOMPLETE"


def test_d2_binds_the_recorded_product_commits(tmp_path):
    out = two_tree(tmp_path)
    assert out["d2"]["expected_shas"] == {"prev": SHAS["prev"][:7], "cand": SHAS["cand"][:12]}
    assert out["d2"]["sha_mismatch"] == [] and out["d2"]["verdict"] == "PASS"
    out = two_tree(tmp_path / "wrong", shas=(SHAS["prev"], SHAS["prev"]))
    assert out["d2"]["sha_mismatch"] == [f"cand/seed-1/{run}: {SHAS['cand'][:12]}" for run in ("r1", "r2")]
    assert out["d2"]["verdict"] == "INCOMPLETE"
    prev, cand = build(tmp_path / "p", "LCMX-fleet"), build(tmp_path / "c", "LCMX-fleet")
    for shas, message in ((None, "requires --expect-shas"), (("abc", SHAS["cand"]), "hex commit prefixes"),
                          (("not-hex!", SHAS["cand"]), "hex commit prefixes")):
        result = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"], shas=shas, check=False)
        assert result.returncode != 0 and message in result.stderr


def test_d2_requires_the_same_effective_configuration(tmp_path):
    def summarize(arm, sha, seed, run):
        s = summary(arm, sha, seed, run)
        if run == "r2":
            s["arm"]["env"] = {"LCM_X": "2"}
        return s
    out = two_tree(tmp_path, summarize=summarize)
    assert out["d2"]["config_equal"] is False and out["d2"]["config_diff_keys"] == ["arm", "arm.env"]
    assert out["d2"]["verdict"] == "INCOMPLETE"
    assert two_tree(tmp_path / "same")["d2"]["config_equal"] is True
