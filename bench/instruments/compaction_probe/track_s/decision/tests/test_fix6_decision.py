"""Round-3 regressions: synthetic, admitted checkpoint scores; no model calls."""
import ast
import hashlib
import json
from pathlib import Path
import shlex

import pytest
from score_s import accounting as count
from test_eval2 import e
from test_fix1_decision import write

TRACK = Path(__file__).resolve().parents[2]
SEEDS = list(range(1, 9))


def fixture(root, arms=("L0", "L1"), reader="glm-5.3"):
    scores, material = root / "decision/scores", root / "material"
    manifest = dict(schema=e.score_manifest.SCHEMA, entries={})
    for seed in SEEDS:
        mat = material / f"seed-{seed}"
        cps = [dict(id=f"S{seed}-CP{n}", row_index=n, tokens=n) for n in (10, 20, 30)]
        facts = [dict(id=f"f{i}", row_role="user", placement="head", **{"class": cls})
                 for i, cls in enumerate(["limit"] * 99 + ["decision"])]
        write(mat / "facts.json", facts)
        write(mat / "material.manifest.json", dict(seed=seed, checkpoints=cps, decision_checkpoint=cps[-1]))
        (mat / "lifecycle_probes.jsonl").write_text("".join(json.dumps(dict(
            id=f"life{n}", probe_token_position=n)) + "\n" for n in (10, 20)))
        for arm in arms:
            run = root / "runs" / arm / f"seed-{seed}" / "r1"
            summary = dict(events=[dict(summariser_calls=[dict(usage=dict(prompt_tokens=7, completion_tokens=3))])])
            write(run / "summary.json", summary)
            for n, cp in enumerate(cps, 1):
                # Distinct reader calls at each checkpoint, not cumulative.
                calls = [dict(calls=[dict(prompt_tokens=10*n, completion_tokens=n)])]
                checkpoint_summary = dict(summary, reader_calls=calls)
                write(run / f"cp-{cp['id']}/summary.json", checkpoint_summary)
                probes = {f["id"]: {"class": "MISS" if arm == "L0" else "CORRECT"} for f in facts}
                accounting = count(checkpoint_summary, probes)
                sc = dict(arm=arm, seed=f"seed-{seed}", reader=reader, context_length=272000,
                    worktree_head=arm, checkpoint_id=cp["id"], run_dir=str(run / f"cp-{cp['id']}"),
                    probes=probes, behaviour=dict(compactions=1), accounting=accounting,
                    reader_readback=dict(pin_ok=True, effort="low"), reader_errors=0,
                    summariser_usage=dict(events_present=True, calls=[dict(prompt_tokens=7, completion_tokens=3)]),
                    metrics=dict(facts_kept=dict(complete=True, denominator=100), lifecycle=dict(complete=True)))
                # Positive synthetic success denominator isolates the unchanged cost gate.
                sc["accounting"]["successful_probes"] = 1
                path = scores / f"cp-{cp['id']}" / f"{arm}.json"
                write(path, sc)
                manifest["entries"][path.relative_to(scores.parent).as_posix()] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write(scores.parent / "manifest.json", manifest)
    return scores, material


def modify(scores, change):
    manifest = json.loads((scores.parent / "manifest.json").read_text())
    for path in scores.glob("cp-*/*.json"):
        sc = json.loads(path.read_text())
        change(sc)
        write(path, sc)
        manifest["entries"][path.relative_to(scores.parent).as_posix()]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(scores.parent / "manifest.json", manifest)


def comparison(scores, material, other="L0"):
    return e.analyze(scores, material, SEEDS)["comparisons"][f"L1−{other}"]


@pytest.mark.parametrize("arm", ["L0", "L1", "L1-H", "L1-noptr"])
@pytest.mark.parametrize("bad", ["missing", "empty", "mixed", "mixed_context"])
def test_item2_every_real_product_arm_has_one_nonempty_pin(tmp_path, arm, bad):
    scores, material = fixture(tmp_path, arms=(arm,))
    def change(sc):
        if sc["seed"] == "seed-8" and sc["checkpoint_id"].endswith("CP30"):
            if bad == "missing":
                sc.pop("worktree_head")
            else:
                sc["worktree_head"] = "" if bad == "empty" else "other-head"
            if bad == "mixed_context":
                sc["context_length"] = 64000
    modify(scores, change)
    with pytest.raises(ValueError, match=f"{arm}.*worktree_head"):
        e.analyze(scores, material, SEEDS)


@pytest.mark.parametrize("reader", ["FAKE", "glm-5.3"])
def test_item2_valid_single_pins_and_fake_exemption(tmp_path, reader):
    scores, material = fixture(tmp_path, reader=reader)
    if reader == "FAKE":
        modify(scores, lambda sc: sc.pop("worktree_head"))
    assert comparison(scores, material)["verdict"] == "KEEP"


def test_item2_same_l0_l1_head_is_refused(tmp_path):
    scores, material = fixture(tmp_path)
    modify(scores, lambda sc: sc.update(worktree_head="same"))
    with pytest.raises(ValueError, match="same SHA"):
        comparison(scores, material)


def test_item3_eval2_codex_command_uses_file_dictation():
    line = next(s for s in (TRACK / "decision/run_decision.sh").read_text().splitlines()
                if '"s4-codex-native-d$N-astra"' in s)
    tokens = shlex.split(line)
    assert "--dictation" in tokens and tokens[tokens.index("--dictation") + 1] == "file"


@pytest.mark.parametrize("other,reader", [("L0", "glm-5.3"), ("H", "glm-5.3"), ("codex-native", "gpt-6-astra")])
def test_item4_three_checkpoints_sum_reader_once_and_summary_once(tmp_path, other, reader):
    scores, material = fixture(tmp_path, arms=("L1", other), reader=reader)
    # (11 + 22 + 33 + one whole-run summariser call of 10) / 3 successes.
    result = comparison(scores, material, "C" if other == "codex-native" else other)
    assert result["cost_per_success"] == {"L1": 76/3, other: 76/3}
    assert result["cost_ratio"] == 1


@pytest.mark.parametrize("path", ["s2/run_s_lcmx.py", "s2/run_s_hermes_builtin.py"])
def test_item4_s2_h_reset_reader_calls_inside_checkpoint_loop(path):
    tree = ast.parse((TRACK / path).read_text())
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "cp"]
    assert any(isinstance(n, ast.Assign) and isinstance(n.value, ast.List) and not n.value.elts and
               any(isinstance(t, ast.Attribute) and t.attr == "reader_calls" for t in n.targets) or
               isinstance(n, ast.Assign) and isinstance(n.value, ast.Tuple) and
               any(isinstance(t, ast.Tuple) and any(isinstance(v, ast.Attribute) and v.attr == "reader_calls" for v in t.elts)
                   for t in n.targets) and isinstance(n.value.elts[-1], ast.List) and not n.value.elts[-1].elts
               for loop in loops for n in ast.walk(loop))


@pytest.mark.parametrize("regression", [False, True])
def test_item7_failed_pair_blocks_keep_and_covered_kill(tmp_path, regression):
    scores, material = fixture(tmp_path)
    def change(sc):
        if regression and sc["arm"] == "L1":
            for p in sc["probes"].values():
                p["class"] = "MISS"
        if sc["seed"] == "seed-1" and sc["checkpoint_id"].endswith("CP20"):
            sc.update(reader_errors=1)
            sc["metrics"]["facts_kept"]["complete"] = False
            sc["accounting"]["successful_probes"] = 0
    modify(scores, change)
    result = comparison(scores, material)
    assert result["intervals"]["facts_user"]["n_seeds"] == 8
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["incomplete_pairs"] == [dict(seed=1, checkpoint="S1-CP20")]


def test_item7_failed_attempts_charged_without_adding_successes(tmp_path):
    scores, material = fixture(tmp_path)
    def change(sc):
        if sc["seed"] == "seed-1" and sc["checkpoint_id"].endswith("CP20"):
            sc.update(reader_errors=1)
            sc["accounting"].update(reader_input_tokens=100, reader_output_tokens=2, successful_probes=0,
                                    reader_estimated_attempts=2)
    modify(scores, change)
    result = comparison(scores, material)
    # 8 * (11+22+33+10), replace the failed 22-token read with 102; 23 successes.
    assert result["cost_per_success"] == {"L1": (8*76+80)/23, "L0": (8*76+80)/23}
    assert result["reader_estimated_attempts"] == {"L1": 2, "L0": 2}


def test_item8_class_floor_only_on_hermes_comparison(tmp_path):
    scores, material = fixture(tmp_path, arms=("L0", "L1", "H"))
    def change(sc):
        if sc["arm"] == "L1":
            sc["probes"]["f99"]["class"] = "MISS"
    modify(scores, change)
    gate = comparison(scores, material)
    assert gate["verdict"] == "KEEP"
    assert set(gate["intervals"]) == {"facts_user", "facts_all"}
    floor = comparison(scores, material, "H")
    assert floor["verdict"] == "BELOW FLOOR"
    assert floor["intervals"]["fact_class_decision"]["high"] == -100
    assert floor["intervals"]["facts_all"]["low"] == pytest.approx(-1)


@pytest.mark.parametrize("kind", ["facts", "corrections", "compactions"])
def test_item1_codex_horizon_excludes_only_affected_pairs_and_renders(tmp_path, kind):
    scores, material = fixture(tmp_path, arms=("L1", "codex-native"), reader="gpt-6-astra")
    def change(sc):
        if sc["arm"] == "codex-native":
            sc["beyond_declared"] = dict(count=1, row_indices=[sc["checkpoint_row"]+1] if "checkpoint_row" in sc else [21],
                                          facts=[], corrections=[], compactions=[])
            if sc["seed"] == "seed-1" and sc["checkpoint_id"].endswith("CP20"):
                sc["beyond_declared"][kind] = ["synthetic-id"]
    modify(scores, change)
    result = comparison(scores, material, "C")
    assert result["compared_window"]["1"] == ["S1-CP10", "S1-CP30"]
    assert len(result["horizon_exclusions"]) == 1
    exclusion = result["horizon_exclusions"][0]
    assert (exclusion["seed"], exclusion["checkpoint"]) == (1, "S1-CP20")
    assert exclusion["beyond_declared"][kind] == ["synthetic-id"]
    analysis, report = tmp_path / "analysis.json", tmp_path / "report.md"
    write(analysis, e.analyze(scores, material, SEEDS))
    e.main(["--analysis", str(analysis), "--out", str(report)])
    assert "beyond-declared exclusions" in report.read_text() and "synthetic-id" in report.read_text()
