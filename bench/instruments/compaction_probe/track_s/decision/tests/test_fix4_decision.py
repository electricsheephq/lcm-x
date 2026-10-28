"""Fix 4 admission and cost accounting; all observations are synthetic."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from test_eval2 import e
from test_fix1_decision import write

HOST = "S1-CONT-host_instruction"


def usage(tokens):
    return dict(events_present=True, calls=[dict(prompt_tokens=tokens, completion_tokens=0)])


def paired(root, *, host=True, right_arm="codex-native"):
    scores, material = root / "decision/scores", root / "material"
    cp = dict(id="S1-CP20000", tokens=20000, row_index=4)
    write(material / "seed-1/facts.json", [dict(id="fact", row_role="user", placement="head", **{"class": "limit"})])
    write(material / "seed-1/material.manifest.json", dict(seed=1, checkpoints=[cp], decision_checkpoint=cp))
    (material / "seed-1/lifecycle_probes.jsonl").write_text("")
    manifest = dict(schema=e.score_manifest.SCHEMA, entries={})
    for arm, label in [("L1", "a"), ("L1", "b"), (right_arm, "c")]:
        run = root / "runs" / arm / label
        tokens = 20 if arm == "L1" else 30
        write(run / "summary.json", dict(events=[dict(summariser_calls=[dict(usage=dict(prompt_tokens=tokens, completion_tokens=0))])]))
        grid = [dict(item=HOST, strict=arm == "L1"), dict(item="active", strict=True)]
        sc = dict(arm=arm, worktree_head=arm, seed="seed-1", reader="gpt-6-astra", reader_readback=dict(effort="low", pin_ok=True),
                  context_length=272000, checkpoint_id=cp["id"], run_dir=str(run / f"cp-{cp['id']}"),
                  probes={"fact": {"class": "CORRECT"}}, behaviour=dict(compactions=1), summariser_usage=usage(tokens),
                  accounting=dict(reader_input_tokens=20 if arm == "L1" else 10, reader_output_tokens=0,
                                  successful_probes=2 if arm == "L1" else 1, reader_estimated_attempts=1),
                  metrics=dict(facts_kept=dict(complete=True, denominator=1), continuity=dict(value=.5, grid=grid)))
        if host:
            sc["host_instruction_presence"] = {HOST: True}
        path = scores / f"cp-{cp['id']}" / f"{arm}.{label}.json"
        write(path, sc)
        manifest["entries"][path.relative_to(scores.parent).as_posix()] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write(scores.parent / "manifest.json", manifest)
    return scores, material


def test_f1_physical_sessions_costed_once_in_two_to_one_pair(tmp_path):
    scores, material = paired(tmp_path)
    r = e.analyze(scores, material, [1])["comparisons"]["L1−C"]
    # L1: (20+20 + 20+20)/4 = 20; C: (10+30)/1 = 40.
    assert r["cost_per_success"] == {"L1": 20, "codex-native": 40}
    assert r["cost_ratio"] == .5
    assert r["reader_estimated_attempts"] == {"L1": 2, "codex-native": 1}


@pytest.mark.parametrize("change", ["tokens", "missing"])
def test_f3_changed_or_unadmitted_summariser_usage_refused(tmp_path, change):
    scores, material = paired(tmp_path)
    if change == "tokens":
        write(tmp_path / "runs/L1/a/summary.json", dict(events=[]))
    else:
        path = scores / "cp-S1-CP20000/L1.a.json"
        sc = json.loads(path.read_text())
        del sc["summariser_usage"]
        write(path, sc)
        manifest = json.loads((scores.parent / "manifest.json").read_text())
        manifest["entries"][path.relative_to(scores.parent).as_posix()]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write(scores.parent / "manifest.json", manifest)
    with pytest.raises(ValueError, match="admitted summariser usage"):
        e.analyze(scores, material, [1])


def test_f3_accounting_rewrite_does_not_change_admitted_usage(tmp_path):
    scores, material = paired(tmp_path)
    path = tmp_path / "runs/L1/a/summary.json"
    summ = json.loads(path.read_text())
    summ["accounting"] = dict(reader_input_tokens=999999)
    write(path, summ)
    assert e.analyze(scores, material, [1])["comparisons"]["L1−C"]["cost_ratio"] == .5


@pytest.mark.parametrize("host", [True, False])
def test_f4_host_visible_continuity_or_explicit_parity_only_exclusion(tmp_path, host):
    scores, material = paired(tmp_path, host=host)
    data = e.analyze(scores, material, [1])
    r = data["comparisons"]["L1−C"]
    assert r["intervals"]["continuity"]["effect_pts"] is None  # one seed, inspect pooled delta with two seeds below
    # Bootstrap retains its declared one-seed behavior; capture the supplied paired delta.
    left, right = [json.loads((scores / "cp-S1-CP20000" / name).read_text()) for name in ("L1.a.json", "codex-native.c.json")]
    excluded = [] if host else [HOST]
    assert r["continuity_exclusions"] == excluded
    assert e.axes(left, [], excluded)["continuity"] == e.axes(right, [], excluded)["continuity"]
    analysis, report = tmp_path / "analysis.json", tmp_path / "report.md"
    write(analysis, data)
    e.main(["--analysis", str(analysis), "--out", str(report)])
    assert "cost ratio / estimated attempts per arm" in report.read_text()
    assert f"continuity excluded (host-visible receipt absent) | {excluded}" in report.read_text()
    # Gate-arm strict booleans remain authoritative even if a host-presence field exists.
    right["arm"] = "L0"
    assert e.axes(right, [])["continuity"] == [False, True]


def decision_module():
    spec = importlib.util.spec_from_file_location("fix4_score_decision", Path(__file__).resolve().parents[1] / "score_decision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arm,field,source", [("L0", "material_sha256", "material.manifest.json"),
    ("H", "material_sha256", "material.manifest.json"), ("codex-native", "material_sha", "transcript.jsonl")])
@pytest.mark.parametrize("digest_state", ["wrong", "absent", "valid"])
def test_f8_material_admission_and_f3_usage_snapshot(tmp_path, monkeypatch, arm, field, source, digest_state):
    m = decision_module()
    material, run, out = tmp_path / "material", tmp_path / "run", tmp_path / "out"
    mat = material / "seed-1"
    write(mat / "material.manifest.json", {})
    (mat / "lifecycle_probes.jsonl").write_text("")
    (mat / "transcript.jsonl").write_text("synthetic transcript")
    summ = dict(events=[])
    if arm == "codex-native":
        summ["material_sha256"] = hashlib.sha256((mat / "material.manifest.json").read_bytes()).hexdigest()
    if digest_state != "absent":
        summ[field] = hashlib.sha256((mat / source).read_bytes()).hexdigest() if digest_state == "valid" else "0" * 64
    write(run / "cp-fixture/summary.json", summ)
    write(run / "summary.json", dict(events=[dict(summariser_calls=[dict(usage=dict(prompt_tokens=7, completion_tokens=3))])]))
    logs = tmp_path / "logs"
    logs.mkdir()
    lane = "s4" if arm == "codex-native" else "s2"
    (logs / f"{lane}-{arm}-fixture.log.wall").write_text("exit 0 end\n")
    monkeypatch.setattr(m, "runs", lambda *a: [(arm, "fixture", run, {"fixture": "cp-fixture"})])
    called = []
    def score(*args):
        called.append(args)
        return dict(probes={})
    monkeypatch.setattr(m, "score", score)
    monkeypatch.setattr(m, "classify_loss", lambda sc: {})
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["score_decision", "--run-root", str(tmp_path / "runs"), "--material", str(material),
                                      "--out", str(out), "--logs", str(logs), "--seeds", "1", "--arms", arm])
    if digest_state != "valid":
        with pytest.raises(ValueError, match="material digest mismatch or absent"):
            m.main()
        assert not called and not list((out / "scores").glob("cp-*/*.json"))
    else:
        m.main()
        admitted = json.loads((out / f"scores/cp-fixture/{arm}.seed-1.fixture.json").read_text())
        assert admitted["summariser_usage"] == dict(events_present=True, calls=[dict(prompt_tokens=7, completion_tokens=3)])
        assert json.loads((run / "summary.json").read_text())["accounting"]["summariser_input_tokens"] == 7
