"""Fix 4 loss, reader-failure cost and logical-turn checkpoint regressions."""
import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import score_s as sc
import fixtures as fx
import test_eval2_harness as harness

material = harness.material
HOST = "S1-CONT-host_instruction"


def test_f2_missing_due_fact_is_incomplete_not_dropped(material, tmp_path):
    spec = importlib.util.spec_from_file_location("fix4_loss", harness.TRACK / "decision/loss_class.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    facts = sc.jload(material / "facts.json")
    result = m.classify_loss(dict(material=str(material), run_dir=str(tmp_path / "no-store"), arm="L1", seed="seed-1",
        run="fixture", checkpoint_row=0, probes={}, metrics=dict(facts_kept={})))
    assert result["lost"] == 1 and result["counts"] == {"incomplete": 1}
    assert result["facts"][0]["id"] == facts[0]["id"]


@pytest.mark.parametrize("exception", [TimeoutError, ValueError])
def test_f7_raise_then_succeed_charges_exact_request_estimate(material, monkeypatch, tmp_path, exception):
    m = harness.module(harness.TRACK / "s2/run_s_lcmx.py", monkeypatch, tmp_path)
    selected = harness.cp.select(material, "CP20000")[0]
    batch = next(b for b in m.batches_for(material, selected) if any(p["kind"] == "stale_task" for p in b["probes"]))
    monkeypatch.setattr(m, "batches_for", lambda *a: [batch])
    prompts = []
    class Reader:
        readback = dict(model="FAKE", pin_ok=True)
        def ask(self, system, view, turns, replies):
            # Record the exact sent prompt just as the real reader lanes do.
            self.request_prompt = "exact request prefix\n" + system + "\n" + turns[0]
            prompts.append(self.request_prompt)
            if len(prompts) == 1:
                raise exception("synthetic failure")
            return json.dumps({p["id"]: "STATUS: NOT LIVE\nCancelled." for p in batch["probes"]}), dict(usage=dict(prompt_tokens=2, completion_tokens=3))
    counted = []
    def count(text):
        counted.append(text)
        return len(text)
    run = SimpleNamespace(sdir=material, arm=dict(name="L1", kind="fixture", open=False),
        receipts_out=[], m=SimpleNamespace(tokens=SimpleNamespace(count_tokens=count)), sysmsg=dict(role="system", content="fixture"),
        ntok=len, args=SimpleNamespace(reader="fake", batches=0, run="fake", lane="fake"), system="fixture",
        events=[], rows=[], seed="seed-1", run_id="fixture", timing_label="fixture", reader_calls=[])
    rows = m.probe(run, [], Reader(), False, dict(db=None, home=None, dir=tmp_path, row=4, checkpoint=selected))
    assert all(r["status"] == "OK" and r["reader_rereads"] == 1 for r in rows)
    calls = [c for b in run.reader_calls for c in b["calls"]]
    assert calls[0] == dict(prompt_tokens=len(prompts[0]), completion_tokens=0, estimated=True)
    assert counted == [prompts[0]] and prompts[0] == prompts[1]
    a = sc.accounting(dict(events=[], reader_calls=run.reader_calls), {"probe": {"class": "CORRECT"}})
    assert a["reader_input_tokens"] == len(prompts[0]) + 2
    assert a["cost_tokens_per_successful_task"] == len(prompts[0]) + 5
    assert a["reader_estimated_attempts"] == 1


def test_f4_scorer_reads_host_visible_receipt(tmp_path):
    mat, run = fx.material(tmp_path), fx.s4_run(tmp_path)
    man = sc.jload(mat / "material.manifest.json")
    man["continuity"] = [dict(id=HOST, value="host fixture", row_index=0)]
    fx.wj(mat / "material.manifest.json", man)
    summary = sc.jload(run / "summary.json")
    summary["survival"] = [dict(window_number=1, turn_last_row_index=4, continuity_verbatim={HOST: False})]
    fx.wj(run / "summary.json", summary)
    fx.wj(run / "continuity/event-0.json", dict(text="native replacement\nhost fixture"))
    scored = sc.score(mat, run, "codex-native")
    assert scored["host_instruction_presence"] == {HOST: True}
    assert scored["metrics"]["continuity"]["grid"][0]["strict"] is False


@pytest.fixture
def checkpoint_run(material, monkeypatch, tmp_path):
    m = harness.module(harness.TRACK / "s4/run_s_codex.py", monkeypatch, tmp_path)
    rows = [dict(id=f"R{i}", turn=i // 2 + 1, role="user" if i % 2 == 0 else "assistant",
                 content=f"fixture-{i}", tool_call_id=None) for i in range(12)]
    rows[0].update(role="system", content="host fixture")
    rows[1]["role"] = "user"
    fx.wl(material / "transcript.jsonl", rows)
    man = sc.jload(material / "material.manifest.json")
    man["continuity"] = [dict(id=HOST, value="host fixture", row_index=0)]
    man["shas"]["transcript.jsonl"] = m.sha(material / "transcript.jsonl")
    fx.wj(material / "material.manifest.json", man)
    monkeypatch.setattr(m, "material_dir", lambda seed: material)
    monkeypatch.setattr(m, "setup_home", lambda: dict(login_status="logged_in", auth_copy_sha256_prefix="fake"))
    monkeypatch.setattr(m, "token_guard", lambda *a: None)
    parent = tmp_path / "parent.jsonl"
    fx.wl(parent, [dict(type="compacted", payload=dict(window_number=1, replacement_history=[]))])
    monkeypatch.setattr(m, "rollout_path", lambda sid: parent)
    monkeypatch.setattr(m, "rollout_items", lambda sid: ([], [dict(model=m.MODEL, effort=m.EFFORT)]))
    monkeypatch.setattr(m.PR, "parse_rollout", lambda *a: dict(compactions=[dict(window_number=1)], token_series=[], model_context_window=64000, history_mode="fake"))
    monkeypatch.setattr(m, "codex_bin", lambda: Path("codex-fake"))
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="fake-version"))
    sha = m.sha
    monkeypatch.setattr(m, "sha", lambda p: "fake" if p.name == "auth.json" else sha(p))
    replay = []
    def cli(cmd, prompt, cwd, stem):
        is_fork = cmd[2] == "fork"
        if not is_fork:
            replay.append(stem.name)
        return dict(rc=0, thread_id="child" if is_fork else "parent", timed_out=False, wall_s=0,
                    agent_messages=["{}"] if is_fork else [], tool_items=0, usage={})
    monkeypatch.setattr(m, "run_cli", cli)
    args = SimpleNamespace(checkpoints="lifecycle", stop_row=None, seed="1", run="fake", auth_file=tmp_path / "auth.json",
                           slice=0, dictation="user", dry_run=False, readmit=False, force_event=False, batches=0)
    assert m.main(args) == 0
    return m, replay, tmp_path / "output/codex-runs/seed-1/fake"


def test_f5_checkpoint_survival_uses_real_workspace(checkpoint_run):
    m, _, root = checkpoint_run
    cp = root / "cp-S1-CP20000"
    assert not (cp / "workspace").exists()
    assert "host fixture" in sc.jload(cp / "continuity/event-0.json")["text"]
    assert sc.jload(cp / "summary.json")["survival"][0]["continuity_host_visible"][HOST] is True


def test_f6_mid_turn_checkpoint_never_creates_extra_host_turn(checkpoint_run):
    _, replay, root = checkpoint_run
    assert replay == [f"turn-{i:03d}" for i in range(1, 7)]
    for cp, declared, effective in [("CP20000", 4, 5), ("CP40000", 8, 9), ("CP60000", 11, 11), ("CP340000", 11, 11)]:
        summ = sc.jload(root / f"cp-S1-{cp}/summary.json")
        assert (summ["checkpoint_row"], summ["effective_row"]) == (declared, effective)
        assert summ["checkpoint"] == summ["stop_row_index"] == declared
