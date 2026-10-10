"""Structured stale status, failed-reader re-reads, and pre-spawn timeout validation."""
import json
import ast
import copy
import os
import subprocess
from types import SimpleNamespace

import pytest
import score_s as sc
import test_eval2_harness as harness
import test_fix1 as fixtures

TRACK, cp, material, module = harness.TRACK, harness.cp, harness.material, harness.module
real_seed = fixtures.real_seed


def test_h_reader_view_system_once(tmp_path):
    # Execute the actual snapshot method without importing or touching the host.
    tree = ast.parse((TRACK / "s2/run_s_hermes_builtin.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Run")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "snapshot")
    namespace = dict(copy=copy)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "H.snapshot", "exec"), namespace)
    run = SimpleNamespace(dir=tmp_path, snaps=[], events=[], sysmsg=dict(role="system", content="fixture"))
    namespace["snapshot"](run, dict(id="fixture", row_index=1), [dict(role="user", content="fixture")])
    assert sum(m["role"] == "system" for m in [run.sysmsg] + run.snaps[0]["view"]) == 1


PASS_TEXT = ["No; do {repl}.", "No, it was dropped.", "No, the user dropped it.",
             "No, it isn't needed anymore.", "No, it is not needed.",
             "No, you don't have to do {task} anymore; do {repl}.",
             "No - {task} was cancelled and {repl} was chosen instead.",
             "No, it isn't pending; {repl} replaced it.", "Not anymore.",
             "No. Instead of {task}, do {repl}.", "No - skip it.", "No; it was cancelled.",
             "No, there is nothing pending for it.", "No. Rather than {task}, do {repl}."]
LIVE_TEXT = ["Yes.", "{task} is still pending.", "It still needs to be done, along with {repl}.",
             "It hasn't been cancelled, so it is still on.", "Still outstanding; {repl} is also queued.",
             "It was not replaced; it still needs doing.", "Do {task} first, then {repl}.",
             "{task} remains open, as does {repl}.", "Not cancelled - still needed.",
             "Correct, it still needs to be done.", "Nothing superseded it; it is still required.",
             "It is still to be done."]


@pytest.mark.parametrize("reply,expected", [("\n STATUS: NOT LIVE\n" + x, True) for x in PASS_TEXT] +
    [("STATUS: LIVE\n" + x, False) for x in LIVE_TEXT] + [(x, False) for x in
    ["No; do {repl}.", "I don't know.", "No.", "No information about {task}.", "STATUS: UNKNOWN",
     "STATUS: NOT LIVE; cancelled", "Explanation first.\nSTATUS: NOT LIVE", "STATUS: LIVE\nSTATUS: NOT LIVE", "", None]] +
    [("STATUS: NOT LIVE\nSTATUS: LIVE", True)])
def test_review2_stale_adversarial_status_lines(real_seed, reply, expected):
    man, facts = sc.jload(real_seed / "material.manifest.json"), sc.jload(real_seed / "facts.json")
    for p in sc.jlines(real_seed / "lifecycle_probes.jsonl"):
        if p["kind"] != "stale_task":
            continue
        task = next(r["task"] for r in man["lifecycle"] if r.get("resolution", {}).get("row_id") == p["row_id"])
        repl = p["answer"].split("do ", 1)[1].rstrip(".")
        assert sc.lifecycle_correct(p, reply.format(task=task, repl=repl) if reply else reply, man, facts) is expected


def test_same_instruction_in_s2_h_and_s4_without_material_change(material, monkeypatch, tmp_path):
    before = (material / "lifecycle_probes.jsonl").read_bytes()
    texts = []
    for runner in ("s2/run_s_lcmx.py", "s4/run_s_codex.py"):
        m = module(TRACK / runner, monkeypatch, tmp_path)
        batches = m.batches_for(material, cp.select(material, "CP20000")[0])
        texts.append(next(p["text"] for b in batches for p in b["probes"] if p["kind"] == "stale_task"))
    assert texts[0] == texts[1]
    assert 'Begin your answer with exactly one line: `STATUS: LIVE` or `STATUS: NOT LIVE`, then explain.' in texts[0]
    assert 'S2.probe(' in (TRACK / "s2/run_s_hermes_builtin.py").read_text()
    assert (material / "lifecycle_probes.jsonl").read_bytes() == before


@pytest.mark.parametrize("failures,exception", [(1, TimeoutError), (2, TimeoutError), (1, ValueError), (2, ValueError)])
def test_s2_and_h_reader_rows_reread_once(material, monkeypatch, tmp_path, failures, exception):
    m = module(TRACK / "s2/run_s_lcmx.py", monkeypatch, tmp_path)
    selected = cp.select(material, "CP20000")[0]
    batch = next(b for b in m.batches_for(material, selected) if any(p["kind"] == "stale_task" for p in b["probes"]))
    monkeypatch.setattr(m, "batches_for", lambda *args: [batch])
    calls = []
    class Reader:
        readback = dict(model="FAKE", pin_ok=True)
        def ask(self, system, view, turns, replies):
            calls.append(turns[0])
            if len(calls) <= failures:
                raise exception("synthetic reader failure")
            return json.dumps({p["id"]: "STATUS: NOT LIVE\nCancelled." for p in batch["probes"]}), dict(usage=dict(prompt_tokens=2, completion_tokens=2))
    run = SimpleNamespace(sdir=material, arm=dict(name="L1", kind="fixture", open=False),
        receipts_out=[], m=SimpleNamespace(tokens=SimpleNamespace(count_tokens=len)), sysmsg=dict(role="system", content="fixture"),
        ntok=len, args=SimpleNamespace(reader="fake", batches=0, run="fake", lane="fake"), system="fixture",
        events=[], rows=[], seed="seed-1", run_id="fixture", timing_label="fixture", reader_calls=[])
    src = dict(db=None, home=None, dir=tmp_path, row=selected["row_index"], checkpoint=selected)
    rows = m.probe(run, [], Reader(), False, src)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert all(r["reader_rereads"] == 1 for r in rows)
    assert all(r["status"] == ("ERROR" if failures == 2 else "OK") for r in rows)
    assert sum(c["reread_rows"] for c in run.reader_calls) == len(rows)


@pytest.mark.parametrize("multiplier", ["2", "0", "-3", "bogus", "3.5", "nan", "999999999999999999999999999999999999999"])
def test_timeout_multiplier_rejected_before_any_child(tmp_path, multiplier):
    marker = tmp_path / "spawned"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    # The old run() launches this stub before checking the multiplier.
    stub = bindir / "perl"
    stub.write_text('#!/bin/sh\n: > "$SPAWN_MARKER"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"], SPAWN_MARKER=str(marker),
               TRACK_S_OUT=str(tmp_path / "out"), TRACK_S_MATERIAL=str(tmp_path), S2_TIMEOUT_MULTIPLIER=multiplier)
    result = subprocess.run(["bash", str(TRACK / "decision/run_decision.sh"), "glm", "1", "fixture"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert not marker.exists()
