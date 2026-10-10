"""Amendment 2: real 272k windows, independent contexts and symmetric failure coverage."""
import hashlib
import json
from pathlib import Path

import pytest

from test_eval2 import e
import test_fix1_decision as fixtures

observations, real_material, write = fixtures.observations, fixtures.real_material, fixtures.write


def update(scores, change):
    manifest = json.loads((scores.parent / "manifest.json").read_text())
    for path in list(scores.glob("cp-*/*.json")):
        sc = json.loads(path.read_text())
        change(sc)
        write(path, sc)
        manifest["entries"][str(path.relative_to(scores.parent))]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(scores.parent / "manifest.json", manifest)


@pytest.mark.parametrize("n,expected", [(5, "INCONCLUSIVE"), (6, "KEEP"), (7, "KEEP")])
def test_axis_needs_six_seeds_and_second_failures_exclude_both_arms(tmp_path, real_material, n, expected):
    scores = observations(tmp_path, real_material)
    def fail(sc):
        bad = int(sc["seed"].removeprefix("seed-")) > n and sc["arm"] == "L0" and any(
            p.get("kind") == "current_request" for p in sc["probes"].values())
        sc.update(reader_errors=int(bad), reader_rereads=1 if bad else 0)
    update(scores, fail)
    r = e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−L0"]
    assert r["intervals"]["current_request"]["n_seeds"] == n
    assert r["intervals"]["current_request"]["verdict"] == ("INCONCLUSIVE" if n < 6 else "NON-INFERIOR")
    assert r["verdict"] == expected
    assert len(r["incomplete_pairs"]) == 8 - n
    assert r["reader_rereads"] == {"L0": 8-n, "L1": 0}
    for pair in r["incomplete_pairs"]:
        assert pair["checkpoint"] not in r["compared_window"][str(pair["seed"])]


def test_64k_never_changes_272k_and_is_reported_separately(tmp_path, real_material):
    scores = observations(tmp_path, real_material)
    before = e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−L0"]
    manifest = json.loads((scores.parent / "manifest.json").read_text())
    for path in list(scores.glob("cp-*/*.json")):
        sc = json.loads(path.read_text())
        sc["context_length"] = 64000
        sc["summariser_usage"] = dict(events_present=True, calls=[])
        sc["run_dir"] = sc["run_dir"].replace("/r1/", "/64k/")
        for p in sc["probes"].values():
            p["class"] = "MISS" if sc["arm"] == "L1" else "CORRECT"
        write(path.with_name(path.stem + ".64k.json"), sc)
        target = path.with_name(path.stem + ".64k.json")
        write(Path(sc["run_dir"]).parent / "summary.json", dict(events=[]))
        manifest["entries"][str(target.relative_to(scores.parent))] = dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    write(scores.parent / "manifest.json", manifest)
    data = e.analyze(scores, real_material, list(range(1, 9)))
    assert data["comparisons"]["L1−L0"] == before
    assert data["context_windows"]["64000"]["comparisons"]["L1−L0"]["verdict"] == "KILL"
    analysis, table = tmp_path / "analysis.json", tmp_path / "table.md"
    write(analysis, data)
    e.main(["--analysis", str(analysis), "--out", str(table)])
    assert "272000" in table.read_text() and "64000" in table.read_text()
    assert "n seeds" in table.read_text() and "status-line" in table.read_text()


def test_codex_pairs_seed_checkpoint_despite_own_window_and_run_name(tmp_path, real_material):
    scores = observations(tmp_path, real_material)
    def codex(sc):
        sc["reader"] = "gpt-6-astra"
        sc["reader_readback"] = dict(effort="low", pin_ok=True)
        if sc["arm"] == "L0":
            sc.update(arm="codex-native", context_length=258400, summariser_usage=dict(events_present=True, calls=[]))
            sc["run_dir"] = sc["run_dir"].replace("/r1/", "/codex-reference/")
            write(Path(sc["run_dir"]).parent / "summary.json", dict(events=[]))
    update(scores, codex)
    r = e.analyze(scores, real_material, list(range(1, 9)))["comparisons"]["L1−C"]
    assert r["verdict"] == "NON-INFERIOR" and not r["missing_seeds"]
    assert r["intervals"]["facts_user"]["n_seeds"] == 8
    assert all(r["compared_window"][str(seed)] for seed in range(1, 9))
