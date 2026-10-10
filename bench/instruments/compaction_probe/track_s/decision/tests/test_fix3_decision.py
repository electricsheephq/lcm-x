"""Amendment 3 decisions: real first-compaction shape, coverage and Astra run identity."""
import hashlib
import json
from pathlib import Path

import pytest

from test_eval2 import e
import test_fix1_decision as fixtures
from test_fix2_decision import update

real_material, observations, write = fixtures.real_material, fixtures.observations, fixtures.write
FIRST = {272000: [112, 113, 113, 113, 112, 113, 109, 113], 64000: [21, 22, 22, 22, 20, 21, 22, 22]}


def realistic(scores, material, ctx):
    def shape(sc):
        seed = int(sc["seed"].removeprefix("seed-"))
        cp = next(c for c in e.CP.select(material / f"seed-{seed}", "lifecycle") if c["id"] == sc["checkpoint_id"])
        sc["behaviour"]["compactions"] = int(cp["row_index"] >= FIRST[ctx][seed-1])
        sc["context_length"] = ctx
    update(scores, shape)


@pytest.mark.parametrize("ctx", [272000, 64000])
def test_real_compaction_window_corrected_value_coverage_and_keep(tmp_path, real_material, ctx):
    scores = observations(tmp_path, real_material)
    realistic(scores, real_material, ctx)
    r = e.analyze(scores, real_material, list(range(1, 9)))["context_windows"][str(ctx)]["comparisons"]["L1−L0"]
    assert r["intervals"]["corrected_value"]["n_seeds"] >= 6
    assert r["verdict"] == "KEEP"
    assert all(r["compared_window"].values())
    if ctx == 272000:
        assert r["pre_window"]


@pytest.mark.parametrize("superiority,expected", [(False, "KILL"), (True, "INCONCLUSIVE")])
def test_coverage_blocks_keep_but_not_covered_kill(tmp_path, real_material, superiority, expected):
    scores = observations(tmp_path, real_material)
    realistic(scores, real_material, 272000)
    def limit(sc):
        if int(sc["seed"].removeprefix("seed-")) > 2:
            sc["probes"] = {pid: p for pid, p in sc["probes"].items() if not pid.endswith("@late")}
        if not superiority:
            for p in sc["probes"].values():
                if not p.get("kind"):
                    p["class"] = "CORRECT"
    update(scores, limit)
    r = e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−L0"]
    assert r["intervals"]["corrected_value"]["n_seeds"] < 6
    assert r["verdict"] == expected


def test_astra_runs_do_not_overwrite_and_codex_can_pair_different_names(tmp_path, real_material):
    scores = observations(tmp_path, real_material)
    def astra(sc):
        sc.update(reader="gpt-6-astra", reader_readback=dict(effort="low", pin_ok=True))
        for p in sc["probes"].values():
            p["class"] = "CORRECT"
        if sc["arm"] == "L0":
            sc.update(arm="codex-native", context_length=258400, summariser_usage=dict(events_present=True, calls=[]))
            sc["run_dir"] = sc["run_dir"].replace("/r1/", "/reference/")
            write(Path(sc["run_dir"]).parent / "summary.json", dict(events=[]))
    update(scores, astra)
    manifest = json.loads((scores.parent / "manifest.json").read_text())
    for path in list(scores.glob("cp-*/*.json")):
        sc = json.loads(path.read_text())
        if sc["arm"] != "L1":
            continue
        sc["run_dir"] = sc["run_dir"].replace("/r1/", "/r2/")
        sc["summariser_usage"] = dict(events_present=True, calls=[])
        for p in sc["probes"].values():
            p["class"] = "MISS"
        target = path.with_name(path.stem + ".r2.json")
        write(target, sc)
        write(Path(sc["run_dir"]).parent / "summary.json", dict(events=[]))
        manifest["entries"][str(target.relative_to(scores.parent))] = dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    write(scores.parent / "manifest.json", manifest)
    r = e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−C"]
    assert not r["missing_seeds"]
    assert r["intervals"]["facts_user"]["effect_pts"] == -50
    for seed, cps in r["compared_window"].items():
        assert len(cps) == 2 * sum(c["row_index"] >= fixtures.first_trigger(real_material / f"seed-{seed}") for c in e.CP.select(real_material / f"seed-{seed}", "lifecycle"))


def test_per_arm_stale_status_totals_render_descriptively(tmp_path):
    checkpoints = [dict(arm=arm, context_length=272000, checkpoint_id="fixture", facts_denominator=0,
        lifecycle={}, latency={}, status_line={"STATUS: LIVE": live, "STATUS: NOT LIVE": not_live})
        for arm, live, not_live in [("L0", 2, 3), ("L0", 4, 5), ("L1", 0, 9)]]
    for c in checkpoints:
        c["latency"] = dict(reader_latency=None, summariser_latency=None, compaction_wall=None)
    analysis, table = tmp_path / "analysis.json", tmp_path / "table.md"
    write(analysis, dict(context_length=272000, comparisons={}, per_checkpoint=checkpoints))
    e.main(["--analysis", str(analysis), "--out", str(table)])
    assert "| L0 (272000) | 6 | 8 | descriptive only |" in table.read_text()
    assert "| L1 (272000) | 0 | 9 | descriptive only |" in table.read_text()
