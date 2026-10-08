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
D2 = (1, 2, 3)  # the D2 seed set; two-tree tests replay all of it


def facts_bytes():
    return json.dumps([{"id": f, "placement": "head", "class": "early_user_constraint"} for f in FACTS]).encode()


def manifest_bytes(seed):  # content-addresses its files, as gen_material.py's manifest does
    return json.dumps({"seed": seed, "facts": FACTS, "shas": {"facts.json": hashlib.sha256(facts_bytes()).hexdigest()}}).encode()


def kept_all(seed, run, cp):
    return set(FACTS)


def summary(arm, sha, seed, run):
    return {"events": [], "worktree_head": sha, "arm": {"name": arm, "kind": "plain", "env": {"LCM_X": "1"}},
            "fleet_keys_excluded": ["k"], "harness_overrides": {"h": 1}, "context_length": 200000,
            "lane": "glm", "reader": "glm", "tokenizer": "tiktoken",
            "material_sha256": hashlib.sha256(manifest_bytes(seed)).hexdigest()}


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
            receipt = logs / f"s2-{arm}-d{seed}-{run}.log.wall"  # per-tree start time: receipts differ across trees
            receipt.write_text(f"start {int(sha[:6], 16)}\nexit 0 end {int(sha[:6], 16) + 9}\n")
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
                manifest["entries"][key] = {"sha256": hashlib.sha256((decision / key).read_bytes()).hexdigest(),
                                            "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
            (rundir / "summary.json").write_text(json.dumps(summarize(arm, sha, seed, run)))
    manifest_path.write_text(json.dumps(manifest))
    return runs, decision, logs


def material(tmp_path, seeds=D2):
    root = tmp_path / "material"
    for seed in seeds:
        (root / f"seed-{seed}").mkdir(parents=True, exist_ok=True)
        (root / f"seed-{seed}" / "material.manifest.json").write_bytes(manifest_bytes(seed))
        (root / f"seed-{seed}" / "facts.json").write_bytes(facts_bytes())
    return root


def analyze(tmp_path, arms, first, second=None, labels=None, check=True, shas=None, seeds=(1,), material_root=None):
    runs, decision, logs = first
    command = [sys.executable, "-B", str(ANALYZE), "--arms", *arms, "--seeds", *map(str, seeds),
               "--checkpoints", *map(str, CPS), "--run-root", str(runs), "--decision-root", str(decision),
               "--logs", str(logs), "--material", str(material_root or material(tmp_path))]
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


def two_tree(tmp_path, cand_kept=kept_all, prev_kept=kept_all, shas=(SHAS["prev"][:7], SHAS["cand"][:12]), seeds=D2,
             material_root=None, check=True, **cand_options):
    prev = build(tmp_path / "prev", "LCMX-fleet", prev_kept, seeds=D2)
    cand = build(tmp_path / "cand", "LCMX-fleet", cand_kept, seeds=D2, sha=SHAS["cand"], **cand_options)
    return analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"], shas=shas, seeds=seeds,
                   material_root=material_root, check=check)


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
        assert result["pairs"] == [f"seed-{n}/{r}" for n in D2 for r in ("r1", "r2")] and result["status"] == "COMPLETE"
        facts = result["axes"]["facts_all"]
        assert (facts["n"], facts["b_v1_only"], facts["c_v2_only"]) == (240, 12, 6)
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
        assert out["d2"][f"cp-{cp}"]["occurrence_mcnemar_p_exact"] == facts["p_exact"]
        unit = out[f"cp-{cp}"]["facts_per_fact"]
        assert unit["p_exact"] == ns["mcnemar"](unit["wins"], unit["losses"]) and out["d2"][f"cp-{cp}"]["p_exact"] == unit["p_exact"]


def test_spread_is_reported_at_every_checkpoint(tmp_path):
    out = two_tree(tmp_path, cand_kept=lambda seed, run, cp: set(FACTS[:30]) if (cp, run) == (176, "r2") else set(FACTS))
    cand = out["per_arm"]["cand"]
    assert cand["facts_r1_r2_spread_by_cp"] == {"cp-176": {f"seed-{n}": 0.25 for n in D2},
                                                "cp-304": {f"seed-{n}": 0.0 for n in D2}}
    assert cand["facts_cp304_r1_r2_spread"] == {f"seed-{n}": 0.0 for n in D2}
    assert cand["spread_over_0.10"] == [f"cp-176/seed-{n}" for n in D2] and cand["spread_unmeasured"] == []
    assert out["per_arm"]["prev"]["spread_over_0.10"] == []
    assert out["d2"]["spread_over_0.10"] == {"prev": [], "cand": [f"cp-176/seed-{n}" for n in D2]}
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
        assert d2["effect_pts_exact"] == -1.25 and d2["p_exact"] == 1.0  # one fact lost in half its repetitions
        assert d2["occurrence_mcnemar_p_exact"] == 0.25 and (d2["n_facts"], d2["wins"], d2["losses"]) == (40, 0, 1)
        assert d2["net_loss_ge_5"] is False and d2["blocks"] is False
    assert out["d2"]["verdict"] == "PASS"


def test_d2_unit_is_the_fact_not_the_occurrence(tmp_path):
    # Four facts lost in every repetition (3 seeds x 2 runs): 24 discordant occurrences, all one way. Per occurrence that is
    # McNemar p = 2^-23 with a 10-point loss, which would block. The four facts are the independent units: a sign test on
    # 4 losses and 0 wins gives p = 0.125, so D2 does not block.
    out = two_tree(tmp_path, cand_kept=lambda *a: set(FACTS) - {"f0", "f1", "f2", "f3"})
    for cp in CPS:
        facts = out[f"cp-{cp}"]["axes"]["facts_all"]
        assert (facts["n"], facts["b_v1_only"], facts["c_v2_only"]) == (240, 24, 0) and facts["p_exact"] < 0.05
        d2 = out["d2"][f"cp-{cp}"]
        assert d2["unit"] == "fact" and d2["occurrence_mcnemar_p_exact"] == facts["p_exact"] == 2 / 2**24
        assert (d2["n_facts"], d2["wins"], d2["losses"], d2["ties"]) == (40, 0, 4, 36)  # pooled across seeds by fact id
        assert d2["effect_pts_exact"] == -10.0 and d2["p_exact"] == 0.125
        assert d2["net_loss_ge_5"] is True and d2["blocks"] is False
    assert out["d2"]["verdict"] == "PASS"


def test_d2_fact_share_counts_partial_repetitions(tmp_path):
    # A fact lost in only some repetitions moves its share, not a whole unit: f0..f9 lost in both runs of seed 1 only is a
    # one-third share loss on each of 10 facts (sign test p = 2/1024, mean -8.3 points), which blocks.
    out = two_tree(tmp_path, cand_kept=lambda seed, run, cp: set(FACTS[10:]) if seed == 1 else set(FACTS))
    for cp in CPS:
        unit = out[f"cp-{cp}"]["facts_per_fact"]
        assert (unit["v1_mean_share"], unit["v2_mean_share"], unit["losses"], unit["effect_pts"]) == (1.0, 0.917, 10, -8.3)
        d2 = out["d2"][f"cp-{cp}"]
        assert d2["effect_pts_exact"] == 100 * -20 / 240 and d2["p_exact"] == 2 / 1024 and d2["blocks"] is True
    assert out["d2"]["verdict"] == "BLOCK"


def test_d2_incomplete_when_a_pair_is_missing(tmp_path):
    prev = build(tmp_path / "prev", "LCMX-fleet", seeds=D2)
    cand = build(tmp_path / "cand", "LCMX-fleet", lambda *a: set(FACTS[:30]), seeds=D2, sha=SHAS["cand"])
    (cand[1] / "scores/cp-304/LCMX-fleet.seed-1.d1-r2.json").unlink()
    out = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"],
                  shas=(SHAS["prev"], SHAS["cand"]), seeds=D2)
    assert out["d2"]["missing_required"] == []
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
    assert cand["facts_r1_r2_spread_by_cp"]["cp-176"] == {f"seed-{n}": 0.1 for n in D2}
    assert cand["spread_over_0.10"] == [f"cp-176/seed-{n}" for n in D2] and out["d2"]["verdict"] == "REPEAT_SEEDS"


def test_a_spread_of_exactly_ten_points_does_not_repeat(tmp_path):
    # 48/60 vs 42/60: 0.8 - 0.7 is 0.10000000000000009 in binary floating point; the rule repeats only above 0.10.
    out = two_tree(tmp_path, value=lambda seed, run, cp: 0.8 if run == "r1" else 0.7)
    assert out["per_arm"]["cand"]["spread_over_0.10"] == [] and out["d2"]["verdict"] == "PASS"


def test_d2_roots_must_be_one_tree(tmp_path):
    # Review repros: the candidate's run root paired with the PREVIOUS tree's decision root (round 2), or with the previous
    # tree's decision AND logs roots (round 3), must not read the previous tree's scores as the candidate's.
    prev = build(tmp_path / "prev", "LCMX-fleet", seeds=D2)
    cand_runs, _, cand_logs = build(tmp_path / "cand", "LCMX-fleet", lambda *a: set(FACTS[:10]), seeds=D2, sha=SHAS["cand"])
    for second in ((cand_runs, prev[1], cand_logs), (cand_runs, prev[1], prev[2])):
        result = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, second, labels=["prev", "cand"],
                         shas=(SHAS["prev"][:7], SHAS["cand"][:7]), seeds=D2, check=False)
        assert result.returncode != 0 and "cand: the decision root and logs must sit in the run root's tree" in result.stderr
    # Moving the run root along with them reads the previous tree's runs, which record the previous commit.
    out = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, prev, labels=["prev", "cand"],
                  shas=(SHAS["prev"][:7], SHAS["cand"][:7]), seeds=D2)
    assert out["d2"]["sha_mismatch"] and out["d2"]["verdict"] == "INCOMPLETE"


def test_d2_binds_scores_to_the_selected_trees_receipts(tmp_path):
    # A run re-executed after scoring rewrites its receipt: the manifest's receipt hash no longer matches, so the score
    # cannot stand for that run.
    prev = build(tmp_path / "prev", "LCMX-fleet", seeds=D2)
    cand = build(tmp_path / "cand", "LCMX-fleet", seeds=D2, sha=SHAS["cand"])
    (cand[2] / "s2-LCMX-fleet-d2-r1.log.wall").write_text("start 1\nexit 0 end 2\n")
    out = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"],
                  shas=(SHAS["prev"][:7], SHAS["cand"][:7]), seeds=D2)
    assert out["d2"]["receipt_mismatch"] == ["cand/cp-176/seed-2/r1", "cand/cp-304/seed-2/r1"]
    assert out["d2"]["verdict"] == "INCOMPLETE"
    assert two_tree(tmp_path / "ok")["d2"]["receipt_mismatch"] == []


def test_d2_incomplete_when_receipts_are_missing(tmp_path):
    prev = build(tmp_path / "prev", "LCMX-fleet", seeds=D2)
    runs, decision, _ = build(tmp_path / "cand", "LCMX-fleet", seeds=D2, sha=SHAS["cand"])
    (tmp_path / "cand" / "no-receipts").mkdir()
    out = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, (runs, decision, tmp_path / "cand" / "no-receipts"),
                  labels=["prev", "cand"], shas=(SHAS["prev"], SHAS["cand"]), seeds=D2)
    assert out["status"] == "COMPLETE"  # every score file is admitted; only the receipts are missing
    assert out["d2"]["missing_spread"] == [f"cand/cp-{cp}/seed-{n}" for cp in CPS for n in D2]
    assert out["d2"]["verdict"] == "INCOMPLETE"


def test_d2_requires_the_declared_seed_set(tmp_path):
    out = two_tree(tmp_path, seeds=(1,))
    assert out["d2"]["missing_required"] == ["seed-2", "seed-3"] and out["d2"]["verdict"] == "INCOMPLETE"
    assert two_tree(tmp_path / "all")["d2"]["missing_required"] == []


def test_d2_requires_one_tokenizer(tmp_path):
    def summarize(arm, sha, seed, run):
        return {**summary(arm, sha, seed, run), "tokenizer": "char-estimate"}
    out = two_tree(tmp_path, summarize=summarize)
    assert out["d2"]["config_diff_keys"] == ["tokenizer"] and out["d2"]["verdict"] == "INCOMPLETE"


def test_d2_binds_runs_to_the_analysed_material(tmp_path):
    def other(arm, sha, seed, run):
        return {**summary(arm, sha, seed, run), "material_sha256": "0" * 64}

    def older(arm, sha, seed, run):  # a run from before the runner recorded its material
        return {k: v for k, v in summary(arm, sha, seed, run).items() if k != "material_sha256"}
    expected = [f"cand/seed-{n}/{r}" for n in D2 for r in ("r1", "r2")]
    for name, summarize in (("other", other), ("older", older)):
        out = two_tree(tmp_path / name, summarize=summarize)
        assert out["d2"]["material_mismatch"] == expected and out["d2"]["verdict"] == "INCOMPLETE"
    assert two_tree(tmp_path / "same")["d2"]["material_mismatch"] == []


def test_d2_binds_the_recorded_product_commits(tmp_path):
    out = two_tree(tmp_path)
    assert out["d2"]["expected_shas"] == {"prev": SHAS["prev"][:7], "cand": SHAS["cand"][:12]}
    assert out["d2"]["sha_mismatch"] == [] and out["d2"]["verdict"] == "PASS"
    out = two_tree(tmp_path / "wrong", shas=(SHAS["prev"], SHAS["prev"]))
    assert out["d2"]["sha_mismatch"] == [f"cand/seed-{n}/{run}: {SHAS['cand'][:12]}" for n in D2 for run in ("r1", "r2")]
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


def test_d2_rejects_repeated_seeds(tmp_path):
    result = two_tree(tmp_path, seeds=(1, 1, 2, 3), check=False)  # seed 1 twice would double its pairs in n and p
    assert result.returncode != 0 and "distinct" in result.stderr


def test_d2_counts_no_score_from_a_decision_root_without_a_manifest(tmp_path):
    prev = build(tmp_path / "prev", "LCMX-fleet", seeds=D2)
    cand = build(tmp_path / "cand", "LCMX-fleet", seeds=D2, sha=SHAS["cand"])
    (cand[1] / "manifest.json").unlink()  # a legacy decision root: its scores are not admitted, so none is counted
    out = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"],
                  shas=(SHAS["prev"][:7], SHAS["cand"][:12]), seeds=D2)
    assert out["d2"]["verdict"] == "INCOMPLETE" and out["d2"]["missing_spread"]
    assert not any(k.startswith("cand/") for k in out["d2"]["receipt_mismatch"])


def test_d2_checks_each_material_file_against_the_manifest(tmp_path):
    root = material(tmp_path)
    (root / "seed-2" / "facts.json").write_bytes(facts_bytes()[:-1])  # truncated: invalid JSON; the manifest is unchanged
    (root / "seed-3" / "facts.json").unlink()
    d2 = two_tree(tmp_path, material_root=root)["d2"]  # reported as a verdict, not a crash in the facts parse
    assert d2["material_mismatch"] == ["material/seed-2/facts.json", "material/seed-3/facts.json"]
    assert d2["verdict"] == "INCOMPLETE"


def test_d2_material_names_stay_inside_the_seed_directory(tmp_path):
    root = material(tmp_path)
    (tmp_path / "outside.json").write_text("{}")
    manifest = json.loads(manifest_bytes(1))
    manifest["shas"]["../../outside.json"] = hashlib.sha256(b"{}").hexdigest()
    (root / "seed-1" / "material.manifest.json").write_text(json.dumps(manifest))
    d2 = two_tree(tmp_path, material_root=root)["d2"]
    assert d2["material_mismatch"] == ["material/seed-1/../../outside.json: outside the seed directory"]


def test_d2_reports_effective_config_differences_without_gating(tmp_path):
    prev = build(tmp_path / "prev", "LCMX-fleet", seeds=D2)
    cand = build(tmp_path / "cand", "LCMX-fleet", seeds=D2, sha=SHAS["cand"])
    for tree, (runs, _, _) in (("prev", prev), ("cand", cand)):
        for seed in D2:
            for run in ("r1", "r2"):  # a product default that moved between the trees: the change under test
                (runs / "LCMX-fleet" / f"seed-{seed}" / f"d{seed}-{run}" / "config.json").write_text(json.dumps(
                    {"summary_prefix_target_tokens": 900 if tree == "prev" else 1200, "leaf_chunk_tokens": 8000}))
    d2 = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"],
                 shas=(SHAS["prev"][:7], SHAS["cand"][:12]), seeds=D2)["d2"]
    assert d2["effective_config_diff_keys"] == ["summary_prefix_target_tokens"] and d2["effective_config_unavailable"] == []
    assert d2["config_equal"] is True and d2["verdict"] == "PASS"


def test_d2_effective_config_diagnostic_is_unavailable_not_fatal(tmp_path):
    prev = build(tmp_path / "prev", "LCMX-fleet", seeds=D2)
    cand = build(tmp_path / "cand", "LCMX-fleet", seeds=D2, sha=SHAS["cand"])
    for seed in D2:  # only the previous tree recorded configs, and one of them is truncated
        for run in ("r1", "r2"):
            (prev[0] / "LCMX-fleet" / f"seed-{seed}" / f"d{seed}-{run}" / "config.json").write_text('{"a": 1}')
    (prev[0] / "LCMX-fleet" / "seed-1" / "d1-r1" / "config.json").write_text('{"a": ')
    d2 = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"],
                 shas=(SHAS["prev"][:7], SHAS["cand"][:12]), seeds=D2)["d2"]
    assert d2["effective_config_diff_keys"] is None and d2["verdict"] == "PASS"
    assert d2["effective_config_unavailable"] == ["prev/seed-1/r1"] + [f"cand/seed-{n}/{r}" for n in D2 for r in ("r1", "r2")]
