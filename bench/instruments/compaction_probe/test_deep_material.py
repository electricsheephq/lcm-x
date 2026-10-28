"""Offline deep material contract and immutable legacy serialization checks."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
GENERATOR = Path(__file__).with_name("gen_material.py")
BASE = "bda521376617b369c0421af88794e0c1acce3716"


def script(name):
    spec = importlib.util.spec_from_file_location(name, GENERATOR.with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen, verifier = script("gen_material"), script("verify_material")


def read(directory, name):
    return json.loads((directory / name).read_text())


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("v4", [False, True])
def test_legacy_all_files_byte_identical_to_base(tmp_path, seed, v4):
    source = subprocess.run(["git", "-C", str(ROOT), "show",
        f"{BASE}:bench/instruments/compaction_probe/gen_material.py"],
        check=True, capture_output=True, text=True).stdout
    base = ModuleType("base_material")
    base.__file__ = str(GENERATOR)
    exec(compile(source, str(GENERATOR), "exec"), base.__dict__)
    old, new = tmp_path / "base", tmp_path / "candidate"
    kwargs = dict(placements=True, classes12=True, v4=v4)
    base.generate(seed, old, **kwargs)
    gen.generate(seed, new, **kwargs)
    assert {p.name for p in old.iterdir()} == {p.name for p in new.iterdir()}
    assert all(p.read_bytes() == (new / p.name).read_bytes() for p in old.iterdir())
    assert verifier.verify(new)["status"] == "PASS"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_deep_scale_sources_checkpoints_and_determinism(tmp_path, seed):
    baseline, first, again = (tmp_path / n for n in ("baseline", "first", "again"))
    gen.generate(seed, baseline, placements=True, classes12=True)
    manifest = gen.generate(seed, first, deep=True)
    gen.generate(seed, again, deep=True)
    assert all(p.read_bytes() == (again / p.name).read_bytes() for p in first.iterdir())
    result, base = verifier.verify(first), verifier.verify(baseline)
    metadata = read(first, "deep_metadata.json")
    assert metadata["baseline_v3"] == dict(rows=base["rows"], tokens=base["tokens"])
    assert 2.5 <= result["rows"] / base["rows"] <= 4
    assert 2.5 <= result["tokens"] / base["tokens"] <= 4
    assert result["status"] == "PASS" and result["rows"] == 1024
    assert manifest["material_version"] == "track-s-deep-v2"
    assert [cp["requested_row_index"] for cp in manifest["checkpoints"]] == [176, 304, 600, 900, 1023]
    assert not (first / "lifecycle_probes.jsonl").exists()
    keys = metadata["checkpoint_answers"]
    assert len(keys) == 5 and all(len(facts) == 60 for facts in keys.values())
    policy = metadata["request_policy"]
    assert policy["initial_superseded_by"] is None
    initial = next(p for p in metadata["continuity_probes"] if p["kind"] == "current_request")
    assert policy["initial_row_id"] == initial["row_id"]
    assert policy["latest_rule"] == "latest user row, including fact corrections"
    assert any(a["value"] != b["value"] for a, b in zip(keys[min(keys, key=int)], keys[max(keys, key=int)]))
    assert verifier.main([str(first)]) == 0


@pytest.mark.parametrize("defect", ["checkpoint", "latest", "version", "action", "scale", "hash"])
def test_deep_verifier_rejects_semantic_defects(tmp_path, defect):
    manifest = gen.generate(1, tmp_path, deep=True)
    metadata = read(tmp_path, "deep_metadata.json")
    if defect == "checkpoint":
        manifest["checkpoints"][0]["tokens"] += 1
        message = "checkpoint admissibility"
    elif defect == "latest":
        metadata["checkpoint_answers"][str(manifest["checkpoints"][0]["id"])][0]["value"] = "future"
        message = "checkpoint latest value"
    elif defect == "version":
        facts = read(tmp_path, "facts.json")
        fact = next(f for f in facts if len(f["versions"]) > 1)
        fact["versions"][0]["row_id"] = "wrong-source"
        gen._json_write(tmp_path / "facts.json", facts)
        message = "version source"
    elif defect == "action":
        probe = next(p for p in metadata["continuity_probes"] if p["kind"] == "constraint_obey")
        probe["answer"] = "WRITE_FIXTURES"
        manifest["continuity_probes"] = metadata["continuity_probes"]
        message = "constraint/tool-heavy"
    elif defect == "scale":
        metadata["baseline_v3"]["tokens"] = 1
        message = "scale outside"
    else:
        (tmp_path / "traps.json").write_text((tmp_path / "traps.json").read_text() + " ")
        message = "digest mismatch"
    gen._json_write(tmp_path / "deep_metadata.json", metadata)
    if defect != "hash":
        manifest["shas"] = {name: gen._sha256(tmp_path / name) for name in manifest["shas"]}
    gen._json_write(tmp_path / "material.manifest.json", manifest)
    with pytest.raises(ValueError, match=message):
        verifier.verify(tmp_path)


def test_deep_cli_and_mode_guards(tmp_path):
    assert gen.main(["--seed", "1", "--out-dir", str(tmp_path / "cli"), "--deep"]) == 0
    assert verifier.verify(tmp_path / "cli")["status"] == "PASS"
    for kwargs in (dict(v4=True), dict(smoke=True), dict(turns=130), dict(min_events=3)):
        with pytest.raises(ValueError, match="deep"):
            gen.generate(1, tmp_path / "bad", deep=True, **kwargs)
    with pytest.raises(ValueError, match="empty"):
        gen.generate(1, tmp_path / "cli", deep=True)
