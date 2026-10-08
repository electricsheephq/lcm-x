"""Phase D (D2) release-over-release paired analysis: two trees, unrounded p, per-checkpoint spread, d2 verdict."""

import ast
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1] / "bench/instruments/compaction_probe/track_s"
ANALYZE = TRACK / "decision/analyze_paired.py"
FACTS = [f"f{i}" for i in range(40)]
CPS = (176, 304)
SHAS = {"prev": "a1" * 20, "cand": "b2" * 20}
D2 = (1, 2, 3)  # the D2 seed set; two-tree tests replay all of it


def facts_bytes(ids=FACTS):
    return json.dumps([{"id": f, "placement": "head", "class": "early_user_constraint"} for f in ids]).encode()


def manifest_bytes(seed, ids=FACTS):  # content-addresses its files, as gen_material.py's manifest does
    return json.dumps({"seed": seed, "facts": ids, "shas": {"facts.json": hashlib.sha256(facts_bytes(ids)).hexdigest()}}).encode()


def seed_ids(seed):  # the real v3 shape: 60 fact ids per seed, none shared across seeds
    return [f"S{seed}-F{j:02d}-0" for j in range(60)]


def kept_all(seed, run, cp):
    return set(FACTS)


def summary(arm, sha, seed, run):
    return {"events": [], "worktree_head": sha, "arm": {"name": arm, "kind": "plain", "env": {"LCM_X": "1"}},
            "fleet_keys_excluded": ["k"], "harness_overrides": {"h": 1}, "context_length": 200000,
            "lane": "glm", "reader": "glm", "tokenizer": "tiktoken",
            "material_sha256": hashlib.sha256(manifest_bytes(seed)).hexdigest()}


def build(root, arm, kept=kept_all, seeds=(1,), sha=SHAS["prev"], value=None, summarize=summary, ids=None):
    """One tree: runs/<arm>/seed-N/dN-rX, decision/scores (admitted by manifest), logs/*.wall. ids(seed) names each seed's
    facts (default: FACTS in every seed)."""
    runs, decision, logs = root / "runs", root / "decision", root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest_path = decision / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema": "track-s-score-manifest-v2", "entries": {}}
    for seed in seeds:
        facts = ids(seed) if ids else FACTS
        for run in ("r1", "r2"):
            rundir = runs / arm / f"seed-{seed}" / f"d{seed}-{run}"
            receipt = logs / f"s2-{arm}-d{seed}-{run}.log.wall"  # per-tree start time: receipts differ across trees
            receipt.write_text(f"start {int(sha[:6], 16)}\nexit 0 end {int(sha[:6], 16) + 9}\n")
            for cp in CPS:
                (rundir / f"cp-{cp}").mkdir(parents=True, exist_ok=True)
                (rundir / f"cp-{cp}" / "summary.json").write_text(json.dumps({"events": []}))
                kept_ids = kept(seed, run, cp)
                payload = {
                    "probes": {f: {"class": "CORRECT" if f in kept_ids else "WRONG"} for f in facts},
                    "metrics": {"facts_kept": {"complete": True,
                                               "value": value(seed, run, cp) if value else len(kept_ids) / len(facts)},
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


def material(tmp_path, seeds=D2, ids=None):
    root = tmp_path / "material"
    for seed in seeds:
        facts = ids(seed) if ids else FACTS
        (root / f"seed-{seed}").mkdir(parents=True, exist_ok=True)
        (root / f"seed-{seed}" / "material.manifest.json").write_bytes(manifest_bytes(seed, facts))
        (root / f"seed-{seed}" / "facts.json").write_bytes(facts_bytes(facts))
    return root


def analyze_functions(*names):
    tree = ast.parse(ANALYZE.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"math": math}
    exec(compile(tree, str(ANALYZE), "exec"), ns)
    return ns


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
    ns = analyze_functions("mcnemar", "sign_flip", "run_level_perm")
    assert ns["mcnemar"](117, 150) < 0.05 and round(ns["mcnemar"](117, 150), 4) == 0.05
    out = two_tree(tmp_path, cand_kept=lambda *a: set(FACTS) - {"f1", "f2", "f3"})
    for cp in CPS:
        facts = out[f"cp-{cp}"]["axes"]["facts_all"]
        assert facts["p_exact"] == ns["mcnemar"](facts["b_v1_only"], facts["c_v2_only"])
        sign = facts["sign_test"]
        assert sign["p_exact"] == ns["mcnemar"](sign["wins"], sign["losses"])
        unit, run, d2 = out[f"cp-{cp}"]["facts_per_fact"], out[f"cp-{cp}"]["run_level"], out["d2"][f"cp-{cp}"]
        diag = d2["facts_per_fact"]
        assert diag["occurrence_mcnemar_p_exact"] == facts["p_exact"]
        assert unit["test"] == "exact sign-flip on per-fact share differences"
        assert run["test"] == d2["test"] == "exact stratified run-level permutation"
        # f1..f3 are lost in all 6 repetitions: each per-fact difference is -6 over common = 6
        assert unit["fact_sign_flip_p_exact"] == ns["sign_flip"]([-6] * 3) == 0.25 == diag["fact_sign_flip_p_exact"]
        assert unit["sign_test_p_exact"] == ns["mcnemar"](unit["wins"], unit["losses"]) == diag["sign_test_p_exact"]
        assert unit["sign_test_p"] == round(unit["sign_test_p_exact"], 4)
        assert unit["fact_sign_flip_p"] == round(unit["fact_sign_flip_p_exact"], 4)
        # every run: 40/40 before, 37/40 after
        assert run["p_exact"] == ns["run_level_perm"]([([(40, 40)] * 2, [(37, 40)] * 2)] * 3)[0] == 2 / 216 == d2["p_exact"]
        assert run["p"] == round(run["p_exact"], 4) and run["effect_pts_exact"] == d2["effect_pts_exact"] == -7.5


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
        diag = d2["facts_per_fact"]
        # each seed: runs 40, 40 before and 39, 40 after; every split gives +-1/80, so |T*| = 3/80 in 2 x 27 of 216 splits
        assert d2["effect_pts_exact"] == -1.25 and d2["p_exact"] == 54 / 216
        assert diag["fact_sign_flip_p_exact"] == 1.0  # one fact lost in half its repetitions
        assert diag["occurrence_mcnemar_p_exact"] == 0.25 and (diag["n_facts"], diag["wins"], diag["losses"]) == (40, 0, 1)
        assert d2["net_loss_ge_5"] is False and d2["blocks"] is False
    assert out["d2"]["verdict"] == "PASS"


def test_d2_blocks_a_consistent_loss_that_the_fact_unit_misses(tmp_path):
    # Four facts lost in every repetition (3 seeds x 2 runs): in every seed both candidate runs keep 36/40 and both previous
    # runs 40/40, a 10-point loss. The run-level test reaches its minimum p (the observed split and its mirror in every seed:
    # 2/216) and blocks. The fact-level diagnostics see 4 facts (sign-flip p = 0.125) and 24 occurrences (McNemar 2^-23).
    out = two_tree(tmp_path, cand_kept=lambda *a: set(FACTS) - {"f0", "f1", "f2", "f3"})
    for cp in CPS:
        facts = out[f"cp-{cp}"]["axes"]["facts_all"]
        assert (facts["n"], facts["b_v1_only"], facts["c_v2_only"]) == (240, 24, 0) and facts["p_exact"] < 0.05
        d2 = out["d2"][f"cp-{cp}"]
        diag = d2["facts_per_fact"]
        assert d2["unit"] == "run (stratified by seed)" and (d2["n_seeds"], d2["n_runs_per_side"]) == (3, 6)
        assert d2["run_shares"] == {f"seed-{n}": {"prev": [1.0, 1.0], "cand": [0.9, 0.9]} for n in D2}
        assert d2["p_exact"] == 2 / 216 and d2["effect_pts_exact"] == -10.0
        assert d2["net_loss_ge_5"] is True and d2["blocks"] is True
        assert diag["occurrence_mcnemar_p_exact"] == facts["p_exact"] == 2 / 2**24
        assert (diag["n_facts"], diag["wins"], diag["losses"], diag["ties"]) == (40, 0, 4, 36)  # pooled across seeds by fact id
        assert diag["effect_pts_exact"] == -10.0 and diag["fact_sign_flip_p_exact"] == 0.125
        assert diag["sign_test_p_exact"] == diag["fact_sign_flip_p_exact"]  # one magnitude everywhere: sign-flip = sign test
        assert diag["diagnostic_only"] == "assumes independent facts; facts within a run are correlated"
    assert out["d2"]["verdict"] == "BLOCK"


def test_d2_does_not_block_when_one_seed_is_reversed(tmp_path):
    # Seeds 1 and 2: 40/40 before, 32/40 after (-20 points); seed 3 reversed: 36/40 before, 40/40 after (+10). The mean is
    # -10 points, but a sign the seeds do not share is not significant: |T*| >= 0.3 in 28 of 216 splits.
    out = two_tree(tmp_path, cand_kept=lambda seed, run, cp: set(FACTS) if seed == 3 else set(FACTS[8:]),
                   prev_kept=lambda seed, run, cp: set(FACTS[4:]) if seed == 3 else set(FACTS))
    for cp in CPS:
        d2 = out["d2"][f"cp-{cp}"]
        assert d2["effect_pts_exact"] == -10.0 and d2["p_exact"] == 28 / 216 and d2["p_exact"] > 0.05
        assert d2["net_loss_ge_5"] is True and d2["blocks"] is False
    assert out["d2"]["verdict"] == "PASS"


def test_d2_fact_share_counts_partial_repetitions(tmp_path):
    # A fact lost in only some repetitions moves its share, not a whole unit: f0..f9 lost in both runs of seed 1 only is a
    # one-third share loss on each of 10 facts (sign test p = 2/1024, mean -8.3 points). That diagnostic would block; the
    # gate does not, because one seed of three carries the whole loss: |T*| >= |T| in 2 x 36 of 216 splits.
    out = two_tree(tmp_path, cand_kept=lambda seed, run, cp: set(FACTS[10:]) if seed == 1 else set(FACTS))
    for cp in CPS:
        unit = out[f"cp-{cp}"]["facts_per_fact"]
        assert (unit["v1_mean_share"], unit["v2_mean_share"], unit["losses"], unit["effect_pts"]) == (1.0, 0.917, 10, -8.3)
        d2 = out["d2"][f"cp-{cp}"]
        diag = d2["facts_per_fact"]
        assert diag["effect_pts_exact"] == 100 * -20 / 240 and diag["fact_sign_flip_p_exact"] == 2 / 1024
        assert diag["sign_test_p_exact"] == diag["fact_sign_flip_p_exact"]  # one magnitude everywhere: sign-flip = sign test
        assert d2["effect_pts_exact"] == float(100 * Fraction(-1, 4) / 3) and d2["p_exact"] == 72 / 216
        assert d2["blocks"] is False
    assert out["d2"]["verdict"] == "PASS"


def test_sign_flip_dp_equals_brute_force_enumeration():
    sign_flip = analyze_functions("sign_flip")["sign_flip"]
    rng = random.Random(1007)
    vectors = [[rng.choice((-2, -1, 0, 1, 2)) for _ in range(rng.randint(1, 12))] for _ in range(8)]
    vectors += [[2, -1, 1, 2, -2, 1, 1, 2, -1, 2, 2, 1], [0, 0, 0], []]
    for d in vectors:
        nz = [x for x in d if x]
        t = abs(sum(d))
        hits = sum(abs(sum(s * abs(x) for s, x in zip(signs, nz))) >= t for signs in itertools.product((1, -1), repeat=len(nz)))
        assert sign_flip(d) == float(Fraction(hits, 2 ** len(nz))), d
    assert sign_flip([0, 0]) == sign_flip([]) == 1.0


def test_sign_flip_equals_the_sign_test_when_every_magnitude_is_equal():
    ns = analyze_functions("mcnemar", "sign_flip")
    rng = random.Random(7)
    for _ in range(40):
        a = rng.choice((1, 2, 3, 6))
        d = [a * rng.choice((-1, 0, 1)) for _ in range(rng.randint(0, 30))]
        assert ns["sign_flip"](d) == ns["mcnemar"](sum(x > 0 for x in d), sum(x < 0 for x in d)), d


def test_d2_sign_flip_blocks_a_concentrated_loss_the_sign_test_misses(tmp_path):
    # Real v3 shape: seed-specific fact ids, 3 seeds x 60 = 180 facts, each over its seed's 2 runs. Per seed: 4 facts kept in
    # both runs before and lost in both after (12 in all), 10 lose one of two runs (30), 10 gain one of two (30), 36 tied
    # (108). Which run moves alternates, so every seed's r1/r2 spread is 0.
    def cand_kept(seed, run, cp):
        moved = range(4, 9) if run == "r1" else range(9, 14)
        return {f for j, f in enumerate(seed_ids(seed)) if j >= 4 and j not in moved}

    def prev_kept(seed, run, cp):
        moved = range(14, 19) if run == "r1" else range(19, 24)
        return {f for j, f in enumerate(seed_ids(seed)) if j not in moved}

    def summarize(arm, sha, seed, run):
        return {**summary(arm, sha, seed, run), "material_sha256": hashlib.sha256(manifest_bytes(seed, seed_ids(seed))).hexdigest()}
    prev = build(tmp_path / "prev", "LCMX-fleet", prev_kept, seeds=D2, summarize=summarize, ids=seed_ids)
    cand = build(tmp_path / "cand", "LCMX-fleet", cand_kept, seeds=D2, sha=SHAS["cand"], summarize=summarize, ids=seed_ids)
    out = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"], shas=(SHAS["prev"], SHAS["cand"]),
                  seeds=D2, material_root=material(tmp_path, ids=seed_ids))
    # Derived independently of analyze_paired: the sign test on 30 wins vs 42 losses, and the sign-flip null of
    # T = 2A + B, A a sum of 12 signs (the outright losses, d = -2 over common = 2) and B of 60 (d = +-1); observed T = -24.
    sign_p = float(min(Fraction(1), Fraction(2 * sum(math.comb(72, i) for i in range(31)), 2 ** 72)))
    flip_p = float(Fraction(sum(math.comb(12, i) * math.comb(60, j) for i in range(13) for j in range(61)
                                if abs(2 * (2 * i - 12) + (2 * j - 60)) >= 24), 2 ** 72))
    assert sign_p >= 0.05 > flip_p
    for cp in CPS:
        d2 = out["d2"][f"cp-{cp}"]
        diag = d2["facts_per_fact"]
        assert (diag["n_facts"], diag["wins"], diag["losses"], diag["ties"]) == (180, 30, 42, 108)
        assert diag["effect_pts_exact"] == 100 * -24 / 360
        assert diag["sign_test_p_exact"] == sign_p and diag["fact_sign_flip_p_exact"] == flip_p
        # the gate: every run keeps 55/60 before and 51/60 after, in every seed
        assert d2["effect_pts_exact"] == 100 * -4 / 60 and d2["p_exact"] == 2 / 216
        assert d2["net_loss_ge_5"] is True and d2["blocks"] is True
    assert out["per_arm"]["cand"]["spread_over_0.10"] == [] and out["d2"]["verdict"] == "BLOCK"


def seed_specific_two_tree(tmp_path, prev_kept, cand_kept):  # the real v3 shape: seed_ids, 60 facts per seed
    def summarize(arm, sha, seed, run):
        return {**summary(arm, sha, seed, run), "material_sha256": hashlib.sha256(manifest_bytes(seed, seed_ids(seed))).hexdigest()}
    prev = build(tmp_path / "prev", "LCMX-fleet", prev_kept, seeds=D2, summarize=summarize, ids=seed_ids)
    cand = build(tmp_path / "cand", "LCMX-fleet", cand_kept, seeds=D2, sha=SHAS["cand"], summarize=summarize, ids=seed_ids)
    return analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"], shas=(SHAS["prev"], SHAS["cand"]),
                   seeds=D2, material_root=material(tmp_path, ids=seed_ids))


def test_d2_aa_placebo_with_correlated_facts_does_not_block(tmp_path):
    # The A/A shape seen on real runs: one seed's run shares move a lot between runs, and each run's facts move together.
    # Seed 1 keeps its first 39 and 33 facts on the previous runs (0.65, 0.55) and 27 and 21 on the candidate runs
    # (0.45, 0.35): each side's spread is exactly 0.10, so the spread rule does not repeat. Seeds 2 and 3 keep every fact.
    # Treating the 180 facts as independent gives a 6.7-point loss at p = 2/2^18, which would block; at the run level the
    # loss is one seed's, and |T*| >= |T| in 2 x 36 of 216 splits.
    def kept(counts):
        return lambda seed, run, cp: set(seed_ids(seed)[:counts[run]]) if seed == 1 else set(seed_ids(seed))
    out = seed_specific_two_tree(tmp_path, kept({"r1": 39, "r2": 33}), kept({"r1": 27, "r2": 21}))
    for cp in CPS:
        d2 = out["d2"][f"cp-{cp}"]
        diag = d2["facts_per_fact"]
        assert d2["run_shares"]["seed-1"] == {"prev": [0.65, 0.55], "cand": [0.45, 0.35]}
        assert d2["effect_pts_exact"] == float(100 * Fraction(-1, 5) / 3) and d2["p_exact"] == 72 / 216
        assert d2["blocks"] is False
        # the fact-level diagnostic on the same fixture: 18 facts, all losses (6 of them lost in both runs)
        assert diag["fact_sign_flip_p_exact"] == 2 / 2**18 < 0.05 and diag["effect_pts_exact"] <= -5
    assert out["per_arm"]["prev"]["spread_over_0.10"] == out["per_arm"]["cand"]["spread_over_0.10"] == []
    assert out["d2"]["verdict"] == "PASS"


def test_run_level_perm_equals_brute_force_enumeration():
    run_level_perm = analyze_functions("run_level_perm")["run_level_perm"]
    splits = list(itertools.combinations(range(4), 2))
    rng = random.Random(1007)
    def run():
        n = rng.choice((5, 6, 60))
        return rng.randint(0, n), n
    cases = [[([run(), run()], [run(), run()]) for _ in range(2)] for _ in range(25)]  # 2 seeds: 36 splits
    cases += [[([(3, 6), (2, 6)], [(1, 6), (4, 6)]), ([(5, 60), (5, 60)], [(5, 60), (5, 60)])],  # T = 0: every split ties
              [([(6, 6), (5, 6)], [(1, 6), (0, 6)]), ([(1, 2), (1, 2)], [(1, 2), (1, 2)])]]
    for seeds in cases:
        shares = [[Fraction(k, n) for k, n in (*first, *second)] for first, second in seeds]
        t = sum((r[2] + r[3] - r[0] - r[1]) / 2 for r in shares) / len(seeds)
        hits = 0
        for assign in itertools.product(splits, repeat=len(seeds)):
            star = sum(sum(r[i] for i in range(4) if i not in a) / 2 - sum(r[i] for i in a) / 2 for r, a in zip(shares, assign))
            hits += abs(star / len(seeds)) >= abs(t)
        assert run_level_perm(seeds) == (float(Fraction(hits, 6 ** len(seeds))), t), seeds
    assert run_level_perm([([(1, 1), (1, 1)], [(0, 1), (0, 1)])] * 3) == (2 / 216, -1)
    for bad in ([], [([(1, 1), (1, 1)], [(0, 1), (0, 1)])] * 8, [([(1, 1)], [(0, 1), (0, 1)])]):
        try:
            run_level_perm(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_d2_incomplete_when_a_pair_is_missing(tmp_path):
    prev = build(tmp_path / "prev", "LCMX-fleet", seeds=D2)
    cand = build(tmp_path / "cand", "LCMX-fleet", lambda *a: set(FACTS[:30]), seeds=D2, sha=SHAS["cand"])
    (cand[1] / "scores/cp-304/LCMX-fleet.seed-1.d1-r2.json").unlink()
    out = analyze(tmp_path, ["LCMX-fleet", "LCMX-fleet"], prev, cand, labels=["prev", "cand"],
                  shas=(SHAS["prev"], SHAS["cand"]), seeds=D2)
    assert out["d2"]["missing_required"] == []
    d2 = out["d2"]["cp-304"]  # seed 1 lacks a run: the gate has no p rather than a p over fewer runs
    assert (d2["p_exact"], d2["effect_pts_exact"], d2["blocks"], d2["n_seeds"]) == (None, None, False, 2)
    assert d2["run_shares"]["seed-1"] == {"prev": [1.0], "cand": [0.75]}
    assert out["d2"]["cp-176"]["blocks"] is True
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
