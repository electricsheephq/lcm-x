"""Checkpoint/probe flow tests use fake lanes and a synthetic CLI; never authenticate or call models."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import fixtures as fx
import pytest
import score_s as sc

TRACK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACK))
import checkpoints as cp  # noqa: E402


@pytest.fixture
def material(tmp_path):
    mat = fx.material(tmp_path)
    facts = sc.jload(mat / "facts.json")
    for i, f in enumerate(facts):
        f.update(row_index=i, row_id=f"R{i}", row_role="user")
    facts[0]["correction_source"] = dict(value="old-alpha", row_index=0)
    fx.wj(mat / "facts.json", facts)
    cps = [dict(id=f"S1-CP{n * 20000}", row_index=i, tokens=n * 20000) for n, i in ((1, 4), (2, 8), (3, 11))]
    cps.append(dict(id="S1-CP340000", row_index=11, tokens=340000))
    man = sc.jload(mat / "material.manifest.json")
    man.update(seed=1, checkpoints=cps, decision_checkpoint=cps[-2], material_version="track-s-v4",
               lifecycle=[dict(task="old-task", resolution=dict(row_id="R2"))], shas={"transcript.jsonl": "fake"})
    fx.wj(mat / "material.manifest.json", man)
    batches = sc.jlines(mat / "probe_batches.jsonl")
    for b in batches:
        for p in b["probes"]:
            p["expect"] = "ABSTAIN" if p["kind"] == "trap" else "value"
    fx.wl(mat / "probe_batches.jsonl", batches)
    ps = [dict(id="stale", kind="stale_task", row_index=2, row_id="R2", text="Is old-task still to be done?",
               expect="value", answer="No; do new-task.", compaction_horizon=1, probe_token_position=20000),
          dict(id="X-F0-CORRECTION", kind="corrected_value", row_index=1, row_id="R1", text="Current value?",
               expect="value", answer=facts[0]["value"], compaction_horizon=3, probe_token_position=25000),
          dict(id="current", kind="current_request", row_index=8, row_id="R8", text="What now?",
               expect="value", answer="Please do new-task.", compaction_horizon=5, probe_token_position=40000)]
    fx.wl(mat / "lifecycle_probes.jsonl", ps)
    fx.wl(mat / "transcript.jsonl", [dict(id=f"R{i}", turn=i + 1, role="user", content=f"fixture-{i}", tool_call_id=None)
                                   for i in range(12)])
    return mat


def module(path, monkeypatch, tmp_path):
    monkeypatch.setenv("TRACK_S_OUT", str(tmp_path / "output"))
    monkeypatch.setenv("TRACK_S_MATERIAL", str(tmp_path))
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_checkpoint_ids_and_first_after_schedule(material):
    assert [c["id"] for c in cp.select(material, "lifecycle")] == ["S1-CP20000", "S1-CP40000", "S1-CP60000", "S1-CP340000"]
    assert cp.select(material, "CP40000") == cp.select(material, "S1-CP40000")
    assert cp.due(material)["stale"]["schedule"] == "exact"
    assert cp.due(material)["X-F0-CORRECTION"]["schedule"] == "first_after"
    with pytest.raises(ValueError, match="unknown checkpoint"):
        cp.select(material, "S1-CP99999")


@pytest.mark.parametrize("runner", ["s2/run_s_lcmx.py", "s4/run_s_codex.py"])
def test_both_writers_ask_due_lifecycle_and_no_future_facts(material, monkeypatch, tmp_path, runner):
    m = module(TRACK / runner, monkeypatch, tmp_path)
    selected = cp.select(material, "CP20000")[0]
    batches = m.batches_for(material, selected)
    ids = {p["id"] for b in batches for p in b["probes"]}
    assert "stale" in ids and "X-F0-CORRECTION" not in ids and "current" not in ids
    assert "X-F5" not in ids and not any(i.startswith("X-STATE") for i in ids)
    if runner.startswith("s2"):
        rows, _, stop = m.select_rows(material, "S1-CP20000", None)
        assert stop == 4 and len(rows) == 5


def test_v3_row_selectors_remain_unchanged(tmp_path, monkeypatch):
    mat = fx.material(tmp_path)
    fx.wl(mat / "transcript.jsonl", [dict(turn=i, role="user") for i in range(12)])
    man = sc.jload(mat / "material.manifest.json")
    man.update(checkpoints=[dict(row_index=4, tokens=20), dict(row_index=8, tokens=40)], decision_checkpoint=dict(row_index=8))
    fx.wj(mat / "material.manifest.json", man)
    m = module(TRACK / "s2/run_s_lcmx.py", monkeypatch, tmp_path)
    assert cp.select(mat, "4,8") == [dict(id=4, row_index=4), dict(id=8, row_index=8)]
    assert m.select_rows(mat, "20", None)[2] == 4 and m.select_rows(mat, "auto", None)[2] == 8
    assert cp.augment(mat, m.batches_for(mat)) == m.batches_for(mat)


def test_lifecycle_contract_and_scoring_denominators(material, tmp_path):
    run = fx.s2_run(tmp_path)
    summary = sc.jload(run / "summary.json")
    summary.update(checkpoint_id="S1-CP40000", stop_row_index=8)
    fx.wj(run / "summary.json", summary)
    rows = sc.jlines(run / "results.jsonl")
    rows += [dict(schema="s2-result-v1", arm="LCMX-a", kind="probe", probe_id=p["id"], probe_kind=p["kind"],
                  answer=p["answer"], status="OK", batch_id="X-BLIFE") for p in sc.jlines(material / "lifecycle_probes.jsonl")]
    fx.wl(run / "results.jsonl", rows)
    scored = sc.score(material, run, "LCMX-a")
    assert scored["metrics"]["lifecycle"]["by_kind_horizon"] == {"corrected_value|3": [1, 1], "current_request|5": [1, 1]}
    assert "stale" not in scored["probes"]
    man, facts = sc.jload(material / "material.manifest.json"), sc.jload(material / "facts.json")
    stale, corrected, current = sc.jlines(material / "lifecycle_probes.jsonl")
    assert sc.lifecycle_correct(stale, "STATUS: NOT LIVE\nNo; do new-task.", man, facts)
    assert not sc.lifecycle_correct(stale, "STATUS: LIVE\nYes; do old-task.", man, facts)
    assert not sc.lifecycle_correct(stale, "No; do new-task. Also do old-task.", man, facts)
    assert not sc.lifecycle_correct(corrected, corrected["answer"] + " old-alpha", man, facts)
    assert sc.lifecycle_correct(current, "new-task", man, facts)


def test_accounting_unknown_usage_stays_null_and_zero_room_is_per_compaction():
    probes = dict(a={"class": "CORRECT", "answer": "x"}, b={"class": "MISS", "answer": ""})
    summary = dict(events=[dict(is_compaction=True, empty_no_room=True, summariser_calls=[dict(usage=dict(prompt_tokens=50, completion_tokens=10))]),
                           dict(is_compaction=True, empty_no_room=False, summariser_calls=[])],
                   reader_calls=[dict(calls=[dict(input_tokens=30, output_tokens=10, cached_input_tokens=20, latency_s=2)])])
    a = sc.accounting(summary, probes)
    assert a["zero_room_rate"] == .5 and a["compactions_per_task"] == 2
    assert a["cost_tokens_per_successful_task"] == 100 and a["empty_replies"] == 1
    assert (a["reader_input_tokens"], a["reader_cached_tokens"], a["reader_uncached_tokens"]) == (30, 20, 10)
    assert sc.accounting({}, probes)["reader_input_tokens"] is None
    assert sc.accounting({}, probes)["zero_room_rate"] is None


def test_per_arm_pins_and_refusal_of_mismatched_or_dirty_fixture(tmp_path, monkeypatch):
    m = module(TRACK / "s2/run_s_lcmx.py", monkeypatch, tmp_path)
    product = tmp_path / "product"
    product.mkdir()
    subprocess.run(["git", "init", "-q", str(product)], check=True)
    (product / "fixture.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(product), "add", "."], check=True)
    subprocess.run(["git", "-C", str(product), "-c", "user.name=Fixture", "-c", "user.email=f@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    sha = subprocess.check_output(["git", "-C", str(product), "rev-parse", "HEAD"], text=True).strip()
    pins = dict(L0=dict(worktree=str(product), sha=sha), L1=dict(worktree=str(product), sha="0" * 40))
    monkeypatch.setenv("S2_ARM_PINS", json.dumps(pins))
    assert m.A.resolve("L0")["sha"] == sha and m.A.resolve("L1")["sha"] == "0" * 40
    monkeypatch.delitem(sys.modules, "hermes_lcm", raising=False)
    with pytest.raises(SystemExit, match="!= pinned"):
        m.seam.load_engine(m.A.resolve("L1"))
    (product / "fixture.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="tracked changes"):
        m.seam.load_engine(m.A.resolve("L0"))


@pytest.mark.parametrize("parent_changes", [False, True])
@pytest.mark.parametrize("reader_failures", [0, 1, 2])
def test_s4_probes_multiple_checkpoints_and_continues_one_parent(material, monkeypatch, tmp_path, parent_changes, reader_failures):
    m = module(TRACK / "s4/run_s_codex.py", monkeypatch, tmp_path)
    monkeypatch.setattr(m, "material_dir", lambda seed: material)
    monkeypatch.setattr(m, "setup_home", lambda: dict(login_status="logged_in", auth_copy_sha256_prefix="fake"))
    monkeypatch.setattr(m, "token_guard", lambda *a: None)
    parent = tmp_path / "parent.jsonl"
    parent.write_text("fixture-parent")
    monkeypatch.setattr(m, "rollout_path", lambda sid: parent)
    monkeypatch.setattr(m, "rollout_items", lambda sid: ([], [dict(model=m.MODEL, effort=m.EFFORT)]))
    monkeypatch.setattr(m, "survival", lambda *a: [])
    monkeypatch.setattr(m.PR, "parse_rollout", lambda *a: dict(compactions=[], token_series=[], model_context_window=64000, history_mode="fake"))
    monkeypatch.setattr(m, "codex_bin", lambda: Path("codex-fake"))
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="fake-version"))
    actual_sha = m.sha
    monkeypatch.setattr(m, "sha", lambda p: "fake" if p.name == "auth.json" else actual_sha(p))
    replay, forks, tries = [], [], {}
    def cli(cmd, prompt, cwd, stem):
        is_fork = cmd[2] == "fork"
        (forks if is_fork else replay).append(prompt)
        if is_fork and parent_changes:
            parent.write_text("changed-parent")
        key = (str(stem.parent), prompt)
        tries[key] = tries.get(key, 0) + 1
        if is_fork and tries[key] <= reader_failures:
            return dict(rc=124, thread_id=None, timed_out=True, wall_s=0,
                        agent_messages=[], tool_items=0, usage={})
        return dict(rc=0, thread_id="child" if is_fork else "parent", timed_out=False, wall_s=0,
                    agent_messages=["{}"] if is_fork else [], tool_items=0, usage={})
    monkeypatch.setattr(m, "run_cli", cli)
    args = SimpleNamespace(checkpoints="lifecycle", stop_row=None, seed="1", run="fake", auth_file=tmp_path / "auth.json",
                           slice=0, dictation="user", dry_run=False, readmit=False, force_event=False, batches=0)
    if parent_changes:
        with pytest.raises(SystemExit, match="FAILED: checkpoint fork changed parent; replay fallback forbidden"):
            m.main(args)
        assert args.run == "fake" and len(replay) == 5
        return
    assert m.main(args) == 0
    assert len(replay) == 12 and forks
    cps = list((tmp_path / "output/codex-runs/seed-1/fake").glob("cp-*/summary.json"))
    assert len(cps) == 4 and all(sc.jload(p)["isolation_ok"] for p in cps)
    assert parent.read_text() == "fixture-parent"
    rows = [r for p in cps for r in sc.jlines(p.parent / "results.jsonl")]
    assert sum(r["probe_id"].endswith("@late") for r in rows) == 1
    assert all(sc.jload(p)["late_corrected_value_checkpoint"]["id"] == "S1-CP340000" for p in cps)
    assert all(r["status"] == ("ERROR" if reader_failures == 2 else "OK") for r in rows)
    assert all(r["reader_rereads"] == int(reader_failures > 0) for r in rows)
    for p in cps:
        assert sc.jload(p)["reader_rereads"] == sum(r["reader_rereads"] for r in sc.jlines(p.parent / "results.jsonl"))
