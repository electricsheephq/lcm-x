"""Regression for PR 1689 reuse."""
import json
import sys
from types import SimpleNamespace
import pytest
from .bot1_support import S2, load


@pytest.mark.parametrize("arm,field", [("LCMX-fleet-rc3", "worktree_head"), ("H", "host_sha")])
@pytest.mark.parametrize("recorded", ["old", None])
def test_c1_refuses_completed_run_with_wrong_code(tmp_path, monkeypatch, arm, field, recorded):
    mod = load(S2 / "run_c1.py")
    material = tmp_path / "material/seed-1"
    material.mkdir(parents=True)
    (material / "material.manifest.json").write_text("{}")
    (material / "deep_metadata.json").write_text(json.dumps(dict(checkpoint_answers={"1": []})))
    host = tmp_path / "hermes/src"
    host.mkdir(parents=True)
    (host.parent / "HOST.json").write_text(json.dumps(dict(sha="host")))
    run = tmp_path / "scratch/runs" / arm / "seed-1/d1-r1"
    (run / "cp-1").mkdir(parents=True)
    (run / "cp-1/summary.json").write_text("{}")
    summary = dict(status="DONE", material_sha256=mod.digest(material / "material.manifest.json"))
    if recorded is not None:
        summary[field] = recorded
    (run / "summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(mod.subprocess, "check_output", lambda *a, **k: mod.PRODUCT_SHA)
    monkeypatch.setattr(mod.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(sys, "argv", ["run_c1", "--scratch", str(tmp_path / "scratch"), "--product", str(tmp_path),
        "--material", str(material.parent), "--hermes-src", str(host), "--evidence", str(tmp_path / "evidence"), "--arms", arm, "--seeds", "1"])
    with pytest.raises(SystemExit, match="mismatched existing run"):
        mod.main()


@pytest.mark.parametrize("depth", [-1, 0, 1])
def test_round2_c1_refuses_shallow_completed_run(tmp_path, monkeypatch, depth):
    mod = load(S2 / "run_c1.py")
    material, host = tmp_path / "material/seed-1", tmp_path / "hermes/src"
    material.mkdir(parents=True)
    host.mkdir(parents=True)
    (material / "material.manifest.json").write_text("{}")
    (material / "deep_metadata.json").write_text(json.dumps(dict(checkpoint_answers={"1": []})))
    (host.parent / "HOST.json").write_text(json.dumps(dict(sha="host")))
    run = tmp_path / "scratch/runs/LCMX-fleet-rc3/seed-1/d1-r1"
    (run / "cp-1").mkdir(parents=True)
    (run / "cp-1/summary.json").write_text("{}")
    (run / "summary.json").write_text(json.dumps(dict(status="DONE", worktree_head=mod.PRODUCT_SHA,
        material_sha256=mod.digest(material / "material.manifest.json"), events=[dict(nodes=[dict(depth=depth)])])))
    monkeypatch.setattr(mod.subprocess, "check_output", lambda *a, **k: mod.PRODUCT_SHA)
    monkeypatch.setattr(mod.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(sys, "argv", ["run_c1", "--scratch", str(tmp_path / "scratch"), "--product", str(tmp_path),
        "--material", str(material.parent), "--hermes-src", str(host), "--evidence", str(tmp_path / "evidence"),
        "--arms", "LCMX-fleet-rc3", "--seeds", "1"])
    with pytest.raises(SystemExit, match="mismatched existing run"):
        mod.main()
