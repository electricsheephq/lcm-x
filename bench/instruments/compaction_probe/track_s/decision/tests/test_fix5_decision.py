"""Manifest admission and output preservation; synthetic material only."""
import hashlib
import json
import shutil
import sys

import pytest
from test_fix1_decision import write
from test_fix4_decision import decision_module

ARMS = ("L0", "L1", "L1-H", "L1-noptr", "H", "codex-native")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def admission(tmp_path, monkeypatch):
    def setup(arm):
        m = decision_module()
        source = tmp_path / "source/seed-1"
        write(source / "facts.json", [])
        (source / "lifecycle_probes.jsonl").write_text("")
        (source / "transcript.jsonl").write_text("synthetic transcript\n")
        write(source / "material.manifest.json", {"shas": {
            name: digest(source / name) for name in ("facts.json", "lifecycle_probes.jsonl", "transcript.jsonl")}})
        material = tmp_path / "material"
        shutil.copytree(source.parent, material)
        mat, run, out, logs = material / "seed-1", tmp_path / "run", tmp_path / "out", tmp_path / "logs"
        summary = dict(material_sha256=digest(mat / "material.manifest.json"), material_sha=digest(mat / "transcript.jsonl"))
        cps = {cp: f"cp-{cp}" for cp in ("first", "later")}
        for sub in cps.values():
            write(run / sub / "summary.json", summary)
        write(run / "summary.json", dict(events=[]))
        logs.mkdir()
        receipt = logs / f"{'s4' if arm == 'codex-native' else 's2'}-{arm}-fixture.log.wall"
        receipt.write_text("exit 0 end\n")
        monkeypatch.setattr(m, "runs", lambda *a: [(arm, "fixture", run, cps)])
        calls = []
        def score(*args):
            calls.append(args)
            return dict(probes={})
        monkeypatch.setattr(m, "score", score)
        monkeypatch.setattr(m, "classify_loss", lambda sc: {})
        monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: None)
        monkeypatch.setattr(sys, "argv", ["score_decision", "--run-root", str(tmp_path / "runs"),
            "--material", str(material), "--out", str(out), "--logs", str(logs), "--seeds", "1", "--arms", arm])
        return m, mat, run, out, receipt, calls
    return setup


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("name", ["facts.json", "lifecycle_probes.jsonl"])
def test_g1_changed_scoring_input_refuses_every_arm(admission, arm, name):
    m, mat, _, out, _, calls = admission(arm)
    transcript = (mat / "transcript.jsonl").read_bytes()
    (mat / name).write_text("changed synthetic input\n")
    with pytest.raises(ValueError, match=name):
        m.main()
    assert (mat / "transcript.jsonl").read_bytes() == transcript
    assert not calls and not list(out.rglob("scores/**/*.json"))


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("state", ["absent", "wrong"])
def test_g1_manifest_digest_required_for_every_arm(admission, arm, state):
    m, _, run, _, _, calls = admission(arm)
    path = run / "cp-first/summary.json"
    summary = json.loads(path.read_text())
    if state == "absent":
        del summary["material_sha256"]
    else:
        summary["material_sha256"] = "0" * 64
    write(path, summary)
    with pytest.raises(ValueError, match="material_sha256"):
        m.main()
    assert not calls


@pytest.mark.parametrize("refusal", ["later_digest", "receipt"])
def test_g2_refused_run_preserves_previous_outputs_byte_for_byte(admission, refusal):
    m, _, run, out, receipt, calls = admission("L1")
    name = "L1.seed-1.fixture.json"
    manifest = dict(schema=m.score_manifest.SCHEMA, entries={})
    for cp in ("first", "later"):
        for population in ("scores", "loss"):
            path = out / population / f"cp-{cp}" / name
            write(path, dict(previous=population, checkpoint=cp))
            manifest["entries"][path.relative_to(out).as_posix()] = dict(sha256=digest(path))
    m.score_manifest.write_atomic(out / "manifest.json", manifest)
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    if refusal == "later_digest":
        path = run / "cp-later/summary.json"
        summary = json.loads(path.read_text())
        summary["material_sha256"] = "0" * 64
        write(path, summary)
        with pytest.raises(ValueError, match="material_sha256"):
            m.main()
    else:
        receipt.write_text("exit 1 end\n")
        m.main()
    after = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert {p: after.get(p) for p in before} == before
    assert not calls


@pytest.mark.parametrize("probe", [None, {"class": "MISSING"}])
def test_g3_admitted_lost_precedes_absent_or_missing_probe(tmp_path, probe):
    m = decision_module()
    mat = tmp_path / "material"
    write(mat / "facts.json", [dict(id="fact", value="fixture", row_index=0, placement="head", **{"class": "limit"})])
    (mat / "lifecycle_probes.jsonl").write_text("")
    result = m.classify_loss(dict(material=str(mat), run_dir=str(tmp_path / "no-store"), arm="L1", seed="seed-1",
        run="fixture", checkpoint_row=0, probes={} if probe is None else {"fact": probe},
        metrics=dict(facts_kept=dict(lost_before_compaction=dict(ids=["fact"])))))
    assert result["counts"] == {"not-admitted": 1}
    assert result["facts"][0]["loss"] == "not-admitted"
