"""Frozen Eval-2 amendment regressions, using real schedules and synthetic scores."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from test_eval2 import e


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture(scope="module")
def real_material(tmp_path_factory):
    root = tmp_path_factory.mktemp("real-schedules")
    path = Path(__file__).resolve().parents[3] / "gen_material.py"
    spec = importlib.util.spec_from_file_location("fix1_gen", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    for seed in range(1, 9):
        gen.generate(seed, root / f"seed-{seed}", v4=True)
    return root


def observations(root, material, *, pre=False, error=False, same_pin=False, mixed=False):
    scores = root / "decision/scores"
    admission = dict(schema=e.score_manifest.SCHEMA, entries={})
    for seed in range(1, 9):
        mat = material / f"seed-{seed}"
        facts = json.loads((mat / "facts.json").read_text())
        due = e.CP.due(mat)
        cps = e.CP.select(mat, "lifecycle")
        for arm in ("L0", "L1"):
            run = root / "runs" / arm / f"seed-{seed}" / "r1"
            write(run / "summary.json", dict(events=[dict(is_compaction=True, summariser_calls=[dict(usage=dict(prompt_tokens=10, completion_tokens=2))])]))
            for n, cp in enumerate(cps):
                # A single post-window checkpoint with a 10-point gain; pre-window loss dilutes it.
                count = sum(f["row_index"] <= cp["row_index"] for f in facts)
                ps = {f["id"]: {"class": "CORRECT" if arm == "L1" and (not pre or n == len(cps)-1) or
                              pre and arm == "L0" and n < len(cps)-1 else "MISS"} for f in facts if f["row_index"] <= cp["row_index"]}
                if pre and n == len(cps)-1:
                    for f in facts[:max(1, count // 10)]:
                        ps[f["id"]] = {"class": "CORRECT" if arm == "L1" else "MISS"}
                    for f in facts[max(1, count // 10):]:
                        ps[f["id"]] = {"class": "MISS"}
                for p in due.values():
                    if p["checkpoint_id"] == cp["id"]:
                        ps[p["id"]] = dict(kind=p["kind"], compaction_horizon=p["compaction_horizon"], actual_compactions=n,
                                          **{"class": "CORRECT" if arm == "L1" else "MISS"})
                sc = dict(arm=arm, seed=f"seed-{seed}", reader="glm-5.3", checkpoint_id=cp["id"], probes=ps,
                          context_length=64000 if mixed and arm == "L0" else 272000, worktree_head="a" if same_pin else arm,
                          behaviour=dict(compactions=int(not pre or n == len(cps)-1)),
                          accounting=dict(reader_input_tokens=10, reader_output_tokens=2,
                                          successful_probes=1),
                          run_dir=str(run / f"cp-{cp['id']}"), reader_errors=int(error and arm == "L1" and n == 0),
                          metrics=dict(facts_kept=dict(complete=True, denominator=count), lifecycle=dict(complete=True)))
                path = scores / f"cp-{cp['id']}" / f"{arm}.seed-{seed}.json"
                write(path, sc)
                admission["entries"][str(path.relative_to(scores.parent))] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write(scores.parent / "manifest.json", admission)
    return scores


def test_real_schedule_dominance_reaches_keep(tmp_path, real_material):
    scores = observations(tmp_path, real_material)
    # Equal positive synthetic costs isolate the retention gate.
    result = e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−L0"]
    assert all(result["intervals"][k]["n_seeds"] == 8 for k in ("stale_task", "corrected_value", "current_request"))
    assert result["verdict"] == "KEEP"
    assert not any("|" in k for k in result["intervals"])


def test_compared_window_and_fact_pooling(tmp_path, real_material):
    scores = observations(tmp_path, real_material, pre=True)
    r = e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−L0"]
    assert r["intervals"]["facts_user"]["low"] > 5
    assert all(len(v) == 1 for v in r["compared_window"].values())
    assert r["pre_window"]


def test_reader_errors_exclude_pair_symmetrically(tmp_path, real_material):
    scores = observations(tmp_path, real_material, error=True)
    r = e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−L0"]
    assert len(r["incomplete_pairs"]) == 8
    assert r["reader_errors"] == {"L1": 8, "L0": 0}
    assert all(len(v) == len(e.CP.select(real_material / f"seed-{seed}", "lifecycle"))-1 for seed, v in r["compared_window"].items())


def test_identical_product_pins_refused(tmp_path, real_material):
    scores = observations(tmp_path, real_material, same_pin=True)
    with pytest.raises(ValueError, match="same SHA"):
        e.analyze(scores, real_material, list(range(1, 9)))


def test_context_windows_never_mix(tmp_path, real_material):
    scores = observations(tmp_path, real_material, mixed=True)
    assert e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−L0"]["missing_seeds"] == list(range(1, 9))


def test_claim_classes_rendered(tmp_path, real_material):
    scores = observations(tmp_path, real_material)
    data = e.analyze(scores, real_material, list(range(1, 9)))
    analysis, table = tmp_path / "analysis.json", tmp_path / "table.md"
    write(analysis, data)
    e.main(["--analysis", str(analysis), "--out", str(table)])
    text = table.read_text()
    for claim in ("KEEP gate", "mechanism-level floor record", "descriptive ablations", "descriptive parity claim", "mechanism only"):
        assert claim in text


def test_cost_window_with_whole_run_summariser_tokens(tmp_path, real_material):
    scores = observations(tmp_path, real_material, pre=True)
    manifest = json.loads((scores.parent / "manifest.json").read_text())
    for path in scores.glob("cp-*/*.json"):
        sc = json.loads(path.read_text())
        # Reader cost 12, 10 successful window tasks. Both summary calls cost 12,
        # including a call before the window: (12 + 24) / 10 == 3.6.
        sc["accounting"]["successful_probes"] = 10
        write(path, sc)
        manifest["entries"][str(path.relative_to(scores.parent))]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write(Path(sc["run_dir"]).parent / "summary.json", dict(events=[dict(summariser_calls=[dict(usage=dict(prompt_tokens=10, completion_tokens=2))])] * 2))
    write(scores.parent / "manifest.json", manifest)
    r = e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−L0"]
    assert r["cost_per_success"] == {"L0": 3.6, "L1": 3.6}


def test_pool_facts_instead_of_checkpoint_percentages(tmp_path, real_material):
    scores = observations(tmp_path, real_material)
    manifest = json.loads((scores.parent / "manifest.json").read_text())
    expected = []
    for seed in range(1, 9):
        mat = real_material / f"seed-{seed}"
        last = e.CP.select(mat, "lifecycle")[-1]["id"]
        numerator, denominator = 0, 0
        for path in scores.glob(f"cp-*/L1.seed-{seed}.json"):
            sc = json.loads(path.read_text())
            for f in json.loads((mat / "facts.json").read_text()):
                if f["id"] in sc["probes"]:
                    sc["probes"][f["id"]]["class"] = "CORRECT" if sc["checkpoint_id"] == last else "MISS"
                    if f["row_role"] == "user":
                        numerator += sc["checkpoint_id"] == last
                        denominator += 1
            write(path, sc)
            manifest["entries"][str(path.relative_to(scores.parent))]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        expected.append(100 * numerator / denominator)
    write(scores.parent / "manifest.json", manifest)
    ci = e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−L0"]["intervals"]["facts_user"]
    assert ci["effect_pts"] == pytest.approx(sum(expected) / 8)


@pytest.mark.parametrize("checkpoint_retained", [False, True])
def test_continuity_axis_counts_checkpoint_items_not_compaction_frequency(checkpoint_retained):
    sc = dict(probes={}, metrics=dict(continuity=dict(value=.8,
        checkpoint_grid=[dict(item="host", strict=True), dict(item="request", strict=checkpoint_retained)], grid=[
        dict(item="host", strict=True), dict(item="request", strict=True),
        dict(item="host", strict=True), dict(item="request", strict=False)])))
    assert e.axes(sc, [])["continuity"] == [True, checkpoint_retained]
