"""Offline measurement-integrity regressions for the Track S run kit."""
import ast
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TRACK = Path(__file__).resolve().parents[1] / "bench/instruments/compaction_probe/track_s"


@pytest.mark.parametrize("replies,expected_calls,error", [
    ([({}, [8192]), ({}, [8192])], 2, True),
    ([({}, [8192]), ({"f": "kept"}, [10])], 2, False),
    ([({}, [8192]), ({}, [10])], 2, True),  # a cap-triggered retry that still leaves a probe unanswered stays truncated
    ([({"f": 42}, [10])], 1, False),  # a non-string answer is an answer, never an exception
    ([({"f": 0}, [8192])], 1, False),  # a falsy non-string answer completes the batch: no retry
    ([({}, [8192]), ({"f": ["a list"]}, [10])], 2, False),
    ([({"f": "kept"}, [8192])], 1, False),
    ([({}, [8192, 10])], 1, False),
    ([({}, [8191])], 1, False),
])
def test_reader_truncation_batch_retry(tmp_path, replies, expected_calls, error):
    import re

    for name in ("facts", "traps"):
        (tmp_path / f"{name}.json").write_text("[]")
    calls = []

    def answer(*args):
        calls.append(args)
        answers, tokens = replies[len(calls) - 1]
        usage = [{"completion_tokens": n} for n in tokens]
        return answers, {"reader_calls": usage, "usage": usage[-1]}

    reader = SimpleNamespace(readback={"model": "synthetic"})
    run = SimpleNamespace(db=None, home=None, dir=tmp_path, sdir=tmp_path,
        arm={"name": "LCMX-fleet", "kind": "plain", "open": False}, receipts_out=[],
        args=SimpleNamespace(reader="glm", batches=0, run="r1", lane="glm"),
        ntok=lambda _: 0, sysmsg={}, system="synthetic", seed=1, run_id="synthetic",
        timing_label="decision", reader_calls=[], m=SimpleNamespace(tokens=SimpleNamespace(count_tokens=len)))
    ns = functions("s2/run_s_lcmx.py", "probe", Run=SimpleNamespace,
        R=SimpleNamespace(answer=answer, ANSWER_RESERVE=1, READER_WINDOW={"glm": 100}),
        SUMMARY_RE=re.compile("never"), json=json, jload=lambda p: json.loads(p.read_text()),
        batches_for=lambda _: [{"id": "batch", "text": "Reply as JSON", "probes": [
            {"id": "f", "text": "q", "kind": "canary", "expect": "value"}]}])
    rows = ns["probe"](run, [], reader, False)
    assert len(calls) == expected_calls
    assert all(call == calls[0] for call in calls)
    assert rows[0]["status"] == ("ERROR" if error else "OK")
    assert bool(rows[0]["error"]) == error
    if error:
        assert rows[0]["error"] == "READER_TRUNCATED: completion cap 8192 reached"
        assert rows[0]["answered"] is False and rows[0]["timed_out"] is False
    saved = json.loads((tmp_path / "answers/batch.json").read_text())
    if expected_calls == 2:
        assert [a["usage"]["completion_tokens"] for a in saved["attempts"]] == [8192, replies[-1][1][-1]]
        assert len(run.reader_calls[0]["calls"]) == 2


@pytest.mark.parametrize("excluded_class", ["READER_TRUNCATED", "MISSING"])
def test_paired_reader_truncation_excludes_either_side(tmp_path, excluded_class):
    material = tmp_path / "seed-1"
    material.mkdir()
    (material / "facts.json").write_text(json.dumps([
        {"id": f, "placement": "head", "class": "early_user_constraint"} for f in ("a", "b", "c")]))

    def score(arm, *args):
        probes = {f: {"class": "CORRECT"} for f in ("a", "b", "c", "state.next", "state.status")}
        for pid in (("a", "state.next") if arm == "first" else ("b",)):
            probes[pid]["class"] = excluded_class
        return {"probes": probes, "metrics": {"facts_kept": {"complete": False},
            "continuation": {"complete": False, "correct_fields": ["status"], "denominator": 1},
            "continuity": {"complete": True}}}

    ns = functions("decision/analyze_paired.py", "pair_axes", "mcnemar", MATERIAL=tmp_path,
        SEEDS=[1], ARMS=["first", "second"], score=score, grid_by_row=lambda *a: {}, json=json,
        math=__import__("math"))
    axes = ns["pair_axes"](304)["axes"]
    for axis in ("facts_all", "facts_head", "constraint_class"):
        assert axes[axis]["n"] == 2 and axes[axis]["excluded"] == 4
        assert axes[axis]["v1"] == axes[axis]["v2"] == 1.0
        assert axes[axis]["b_v1_only"] == axes[axis]["c_v2_only"] == 0
    assert axes["continuation"]["n"] == axes["continuation"]["excluded"] == 2


def test_reader_truncation_retry_keeps_answers(tmp_path):
    import re

    for name in ("facts", "traps"):
        (tmp_path / f"{name}.json").write_text("[]")
    replies = [({"a": "first", "b": ""}, 8192), ({"a": "", "b": "second"}, 10)]
    calls = []

    def answer(*args):
        calls.append(args)
        answers, tokens = replies[len(calls) - 1]
        return answers, {"reader_calls": [{"completion_tokens": tokens}], "usage": {"completion_tokens": tokens},
                         "tool_calls": len(calls), "tokens_read_back": 10 * len(calls), "wall_s": 1.5}

    reader = SimpleNamespace(readback={"model": "synthetic"})
    run = SimpleNamespace(db=None, home=None, dir=tmp_path, sdir=tmp_path,
        arm={"name": "LCMX-fleet", "kind": "plain", "open": False}, receipts_out=[],
        args=SimpleNamespace(reader="glm", batches=0, run="r1", lane="glm"),
        ntok=lambda _: 0, sysmsg={}, system="synthetic", seed=1, run_id="synthetic",
        timing_label="decision", reader_calls=[], m=SimpleNamespace(tokens=SimpleNamespace(count_tokens=len)))
    ns = functions("s2/run_s_lcmx.py", "probe", Run=SimpleNamespace,
        R=SimpleNamespace(answer=answer, ANSWER_RESERVE=1, READER_WINDOW={"glm": 100}),
        SUMMARY_RE=re.compile("never"), json=json, jload=lambda p: json.loads(p.read_text()),
        batches_for=lambda _: [{"id": "batch", "text": "Reply as JSON", "probes": [
            {"id": p, "text": "q", "kind": "canary", "expect": "value"} for p in ("a", "b")]}])
    rows = ns["probe"](run, [], reader, False)
    assert len(calls) == 2
    assert {r["probe_id"]: r["answer"] for r in rows} == {"a": "first", "b": "second"}
    assert all(r["status"] == "OK" and r["error"] is None for r in rows)
    assert rows[0]["tool_calls_batch"] == 3 and rows[0]["tokens_read_back_batch"] == 30
    assert rows[0]["answer_wall_s_batch"] == 3.0
    saved = json.loads((tmp_path / "answers/batch.json").read_text())
    assert len(saved["attempts"]) == 2 and saved["error"] is None


def paired_axes(tmp_path, score, facts=("a", "b")):
    material = tmp_path / "seed-1"
    material.mkdir()
    (material / "facts.json").write_text(json.dumps([
        {"id": f, "placement": "head", "class": "early_user_constraint"} for f in facts]))
    ns = functions("decision/analyze_paired.py", "pair_axes", "mcnemar", MATERIAL=tmp_path,
        SEEDS=[1], ARMS=["first", "second"], score=score, grid_by_row=lambda *a: {}, json=json,
        math=__import__("math"), re=__import__("re"))
    return ns["pair_axes"](304)


TRUNCATED_ISSUE = "INCOMPLETE: READER_TRUNCATED: 1 probes excluded: ['x']"


def test_paired_admission_loss_outranks_truncation(tmp_path):
    def score(arm, *args):
        probes = {f: {"class": "CORRECT"} for f in ("a", "b")}
        facts_kept = {"complete": True, "lost_before_compaction": {"ids": []}}
        if arm == "first":
            probes["a"]["class"] = "READER_TRUNCATED"
            facts_kept = {"complete": False, "issues": [TRUNCATED_ISSUE], "lost_before_compaction": {"ids": ["a"]}}
        return {"probes": probes, "metrics": {"facts_kept": facts_kept,
            "continuation": {"complete": True}, "continuity": {"complete": True}}}

    axes = paired_axes(tmp_path, score)["axes"]
    assert axes["facts_all"]["n"] == 4 and axes["facts_all"]["excluded"] == 0
    assert axes["facts_all"]["c_v2_only"] == 2 and axes["facts_all"]["b_v1_only"] == 0


def test_loss_class_admission_loss_outranks_truncation(tmp_path):
    (tmp_path / "facts.json").write_text(json.dumps([
        {"id": f, "placement": "head", "class": "name", "value": f"value-{f}"} for f in ("a", "b")]))
    ns = functions("decision/loss_class.py", "classify_loss", "load", "ro", json=json, normalize=str.casefold)
    out = ns["classify_loss"]({"run_dir": str(tmp_path / "run"), "material": str(tmp_path), "arm": "x", "seed": "seed-1",
        "run": "r1", "checkpoint_row": 304, "probes": {"a": {"class": "READER_TRUNCATED"}, "b": {"class": "CORRECT"}},
        "metrics": {"facts_kept": {"lost_before_compaction": {"ids": ["a"]}}}})
    assert [(f["id"], f["loss"]) for f in out["facts"]] == [("a", "not-admitted")]


def test_paired_trap_only_truncation_is_incomplete(tmp_path):
    def score(arm, *args):
        probes = {f: {"class": "CORRECT"} for f in ("a", "b")} | {"t": {"class": "READER_TRUNCATED"}}
        incomplete = {"complete": False, "issues": [TRUNCATED_ISSUE]}
        return {"probes": probes, "metrics": {"facts_kept": incomplete, "continuation": incomplete,
                                              "continuity": {"complete": True}}}

    result = paired_axes(tmp_path, score)
    assert result["pairs"] == ["seed-1/r1", "seed-1/r2"] and result["missing_pairs"] == []
    assert not any(a["excluded"] for a in result["axes"].values())
    assert result["status"] == "INCOMPLETE"


def test_paired_unrun_score_with_missing_probe_is_rejected(tmp_path):
    def score(arm, *args):
        probes = {f: {"class": "CORRECT"} for f in ("a", "b")}
        facts_kept = {"complete": True}
        if arm == "first":
            probes["a"]["class"] = "MISSING"
            facts_kept = {"complete": False, "issues": [
                "UNRUN: run status FAILED", "INCOMPLETE: 1 of 2 scheduled facts have no result row (batches not run: B0)"]}
        return {"probes": probes, "metrics": {"facts_kept": facts_kept, "continuation": {"complete": True},
                                              "continuity": {"complete": True}}}

    result = paired_axes(tmp_path, score)
    assert result["pairs"] == [] and result["missing_pairs"] == [f"seed-1/{r} (incomplete score)" for r in ("r1", "r2")]
    assert result["status"] == "INCOMPLETE"


def functions(path, *names, **namespace):
    namespace.setdefault("Path", Path)
    tree = ast.parse((TRACK / path).read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def main_phase(path, start, **namespace):
    tree = ast.parse((TRACK / path).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    i = next(i for i, n in enumerate(main.body) if ast.unparse(n).startswith(start))
    main.body = main.body[i:]
    exec(compile(ast.Module(body=[main], type_ignores=[]), path, "exec"), namespace)
    return namespace["main"]


def test_absolute_requires_two_runs():
    absolute = functions("scorer/report_s.py", "absolute")["absolute"]
    run = {"run": 1, "metrics": {"continuity": {"status": "OK"}}}
    runs = {("arm", s): [run, run] for s in ("1", "2", "3")}
    assert absolute(runs, "arm", list("123"), "continuity", 3)[0] == "PASS"
    runs[("arm", "2")] = [run]
    assert absolute(runs, "arm", list("123"), "continuity", 3)[0] == "INCOMPLETE"


@pytest.mark.parametrize("receipt", [None, "exit 7 end 2\n", "exit 0 end 2\n"])
def test_skipped_scores_removed(tmp_path, monkeypatch, receipt):
    out, logs = tmp_path / "out", tmp_path / "logs"
    logs.mkdir()
    for population in ("scores", "loss"):
        directory = out / population / "cp-304"
        directory.mkdir(parents=True)
        (directory / "LCMX-fleet.seed-1.d1-r1.json").write_text("{}")
    if receipt:
        (logs / "s2-LCMX-fleet-d1-r1.log.wall").write_text(receipt)
    monkeypatch.setattr(sys, "argv", ["score_decision", "--run-root", str(tmp_path / "runs"),
        "--material", str(tmp_path), "--logs", str(logs), "--out", str(out),
        "--arms", "LCMX-fleet", "--seeds", "1", "--checkpoints", "304"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    runpy.run_path(str(TRACK / "decision/score_decision.py"), run_name="__main__")
    assert not list(out.glob("*/cp-304/*.json"))


def test_paired_admission_losses(tmp_path):
    material = tmp_path / "seed-1"
    material.mkdir()
    (material / "facts.json").write_text(json.dumps([
        {"id": f, "placement": "head", "class": "early_user_constraint"} for f in ("a", "b")]))
    def score(arm, *args):
        return {"probes": {f: {"class": "CORRECT"} for f in ("a", "b")}, "metrics": {
            "facts_kept": {"complete": True, "lost_before_compaction": {"ids": ["a" if arm == "first" else "b"]}},
            "continuation": {"complete": True, "correct_fields": [], "denominator": 0},
            "continuity": {"complete": True}}}
    ns = functions("decision/analyze_paired.py", "pair_axes", "mcnemar", MATERIAL=tmp_path,
        SEEDS=[1], ARMS=["first", "second"], score=score, grid_by_row=lambda *a: {}, json=json,
        math=__import__("math"))
    axes = ns["pair_axes"](304)["axes"]
    for axis in ("facts_all", "facts_head", "constraint_class"):
        assert axes[axis]["v1"] == axes[axis]["v2"] == 0.5
        assert axes[axis]["b_v1_only"] == axes[axis]["c_v2_only"] == 2


@pytest.mark.parametrize("pin,isolation,prior,expected", [
    (False, True, None, "FAILED"), (True, False, None, "FAILED"),
    (True, True, "FAILED", "FAILED"), (True, True, None, "COMPLETED")])
def test_s4_final_receipt(pin, isolation, prior, expected):
    # Exercise main's final receipt/exit phase without live replay or credentials.
    tree = ast.parse((TRACK / "s4/run_s_codex.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "summary['finished']")
    main.body = main.body[start:]
    summary = {"pin_ok": pin, "isolation_ok": isolation}
    if prior:
        summary["status"] = prior
    receipts = []
    ns = {"summary": summary, "time": __import__("time"), "rdir": Path("unused"),
          "jdump": lambda p, s: receipts.append(dict(s)), "json": json}
    exec(compile(ast.Module(body=[main], type_ignores=[]), "s4-final", "exec"), ns)
    rc = ns["main"]()
    assert receipts[0]["status"] == expected
    assert (rc != 0) == (expected == "FAILED")


def test_auth_replaced(tmp_path):
    home, source = tmp_path / "home", tmp_path / "synthetic.json"
    home.mkdir()
    (home / "auth.json").write_text('{}')
    source.write_text('{"last_refresh": "synthetic-second"}')
    ns = functions("s4/run_s_codex.py", "setup_home", "sha", HOME=home, USER_AUTH=source,
        os=os, shutil=__import__("shutil"), hashlib=__import__("hashlib"), json=json,
        codex_bin=lambda: "fake", env=lambda: {}, subprocess=__import__("types").SimpleNamespace(
            DEVNULL=-3, run=lambda *a, **k: __import__("types").SimpleNamespace(
                returncode=0, stdout="logged in chatgpt", stderr="")))
    ns["setup_home"]()
    assert (home / "auth.json").read_bytes() == source.read_bytes()
    assert (home / "auth.json").stat().st_mode & 0o777 == 0o600


def test_s1_dirty_checkout(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    git = fake / "git"
    git.write_text('#!/bin/sh\ncase "$3" in rev-parse) echo 988dee85592b9066ffe1c859542e8e18c23f2345;; diff) exit 1;; esac\n')
    git.chmod(0o755)
    result = subprocess.run(["bash", str(TRACK / "s1/run.sh"), str(tmp_path), "seed-1", "r1"],
        env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "TRACK_S_OUT": str(tmp_path),
             "TRACK_S_MATERIAL": str(tmp_path)}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 3
    assert "tracked modifications" in result.stderr


def test_tool_budget_multi_call():
    source = (TRACK / "s1/replay.ts").read_text()
    body = source.split("async function toolLoop(", 1)[1].split("function gateway(", 1)[0]
    body = body[body.index("  const schemas"):].replace("\n}\n", "\n", 1)
    js = '''const GUARD=2, READER_MAX_TOKENS=1, toolSchema=x=>x;
let dispatched=0, options=[], msgs=[], responses=[1,3,0];
const chat=async(p,m,o)=>{options.push(o); return {text:'done',toolCalls:Array.from({length:responses.shift()},(_,i)=>({id:String(i),name:'t',args:'{}'}))}};
const runTool=async()=>{dispatched++;return 'ok'};
async function toolLoop(purpose,msgs,tools,onCall,acct) {''' + body + '''}
toolLoop('probe',msgs,[{name:'t'}],()=>{}).then(r=>{
if(dispatched!==2 || !r.guardHit || options.at(-1).tools!==undefined || msgs.filter(m=>m.role==='tool').length!==4) process.exit(1);
});'''
    subprocess.run(["node", "-e", js], check=True, timeout=60)


@pytest.mark.parametrize("lane", ["glm", "s4"])
def test_decision_child_exit(tmp_path, lane):
    kit = tmp_path / "kit"
    for seam in ("decision", "s1", "s2", "s4"):
        (kit / seam).mkdir(parents=True)
    runner = kit / "decision/run_decision.sh"
    runner.write_text((TRACK / "decision/run_decision.sh").read_text().replace("sleep 10", "sleep 0.01").replace("sleep 30", "sleep 0.01"))
    child = kit / "s1/run.sh"
    child.write_text("#!/bin/sh\nexit 7\n")
    child.chmod(0o755)
    for script in ("s2/run_s_lcmx.py", "s4/run_s_codex.py"):
        (kit / script).write_text("raise SystemExit(7)\n")
    result = subprocess.run(["bash", str(runner), lane, "1", "fake", "synthetic"],
        env={**os.environ, "PYTHON": sys.executable, "TRACK_S_OUT": str(tmp_path),
             "TRACK_S_MATERIAL": str(tmp_path), "S2_PRODUCT_WORKTREE": "fake"}, timeout=60)
    assert result.returncode != 0
    assert all("exit 7 end" in p.read_text() for p in (tmp_path / "decision/logs").glob("*.wall"))


def test_decision_timeout_kills_group(tmp_path):
    child = tmp_path / "child.py"
    pidfile = tmp_path / "descendant.pid"
    child.write_text("import subprocess,sys,time\nfrom pathlib import Path\n"
        "p=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)'])\n"
        f"Path({str(pidfile)!r}).write_text(str(p.pid))\ntime.sleep(120)\n")
    prefix = (TRACK / "decision/run_decision.sh").read_text().split("PY=${PYTHON", 1)[0]
    prefix = prefix.replace("sleep 10", "sleep 0.1")
    command = prefix + '\nrun timeout 0 "$1" "$2"\n'
    try:
        result = subprocess.run(["bash", "-c", command, "test", sys.executable, str(child)],
            env={**os.environ, "TRACK_S_OUT": str(tmp_path), "TRACK_S_MATERIAL": str(tmp_path),
                 "S2_PRODUCT_WORKTREE": "fake"}, timeout=60)
        assert result.returncode == 124
        state = subprocess.run(["ps", "-o", "stat=", "-p", pidfile.read_text()], capture_output=True, text=True)
        assert not state.stdout.strip() or state.stdout.strip().startswith("Z")
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), 9)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(("arm", "count", "exercise"), [
    ("LCMX-fleet", 0, "VACUOUS"), ("LCMX-fleet", 3, "EXERCISED"),
    ("LCMX-fleet-agedoff", 0, "AGED-OFF"), ("LCMX-fleet-agedoff", 3, "CONTAMINATED")])
def test_stub2k_aged_off_control_must_not_stub(tmp_path, arm, count, exercise):
    import re
    gate = TRACK / "decision/score_stub2k_gate.py"
    stub_re = re.compile(r"^LCM active replay stubbing: replaced (\d+) evictable tool result\(s\),", re.M)
    ns = functions("decision/score_stub2k_gate.py", "run_record", "receipt_ok", "read", "aged_stubs",
                   ROOT=tmp_path, SHA="a" * 40, CPS=(176, 304), re=re, json=json, STUB_RE=stub_re)
    assert "CONTAMINATED" in gate.read_text()
    label = f"g9-{arm}-d1-r1"
    rdir = tmp_path / "runs" / arm / "seed-1" / label
    rdir.mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / f"{label}.log.wall").write_text("exit 0 end 2\n")
    expected = 2000 if arm == "LCMX-fleet" else 6000
    (rdir / "summary.json").write_text(json.dumps({"status": "DONE", "worktree_head": "a" * 40, "checkpoints": [176, 304],
                                                   "started": 1, "finished": 2}))
    (rdir / "config.json").write_text(json.dumps({"product_sha": "a" * 40, "aged_tier_resolved_tokens": expected,
                                                  "large_output_active_replay_stub_aged_threshold_tokens": expected}))
    (rdir / "engine.log").write_text(
        f"LCM active replay stubbing: replaced {count} evictable tool result(s), x\n" if count else "no stubbing\n")
    assert ns["run_record"](arm, 1, "r1")[1]["exercise"] == exercise


@pytest.mark.parametrize("missing_arm", [None, "first", "second"])
def test_paired_missing_scores_incomplete(tmp_path, missing_arm):
    (tmp_path / "seed-1").mkdir()
    (tmp_path / "seed-1/facts.json").write_text("[]")
    def score(arm, seed, rep, cp):
        if arm == missing_arm and rep == "r2":
            return None
        return {"probes": {}, "metrics": {"facts_kept": {"complete": True},
                "continuation": {"complete": True, "correct_fields": [], "denominator": 0},
                "continuity": {"complete": True}}}
    ns = functions("decision/analyze_paired.py", "pair_axes", "mcnemar", MATERIAL=tmp_path,
        SEEDS=[1], ARMS=["first", "second"], score=score, grid_by_row=lambda *a: {},
        json=json, math=__import__("math"))
    result = ns["pair_axes"](304)
    assert result["status"] == ("INCOMPLETE" if missing_arm else "COMPLETE")
    assert result.get("missing_pairs", []) == (["seed-1/r2"] if missing_arm else [])
    # Also check the script-level status across checkpoints, without run/material I/O.
    tree = ast.parse((TRACK / "decision/analyze_paired.py").read_text())
    output = []
    ns.update(CPS=[304], pair_axes=lambda cp: result, losses=lambda cp: {},
              per_arm=lambda: {}, print=output.append)
    tail = next(i for i, n in enumerate(tree.body) if
                (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "paired") or
                (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "print"))
    exec(compile(ast.Module(body=tree.body[tail:], type_ignores=[]), "paired-output", "exec"), ns)
    assert json.loads(output[0])["status"] == result["status"]


def s2_receipt_phase(tmp_path, *, snapshots=True, changed=False, reader=None):
    cpdir = tmp_path / "cp-304"
    cpdir.mkdir()
    cp = {"dir": cpdir, "db": cpdir / "db", "view": [], "row": 304,
          "receipts": [], "n_events": 0, "admission": {}}
    run = SimpleNamespace(snaps=[cp] if snapshots else [], receipts_out=[], reader_calls=[],
                          events=[], sysmsg={}, system="", db=tmp_path / "db", ntok=lambda view: 0)
    reader = reader or SimpleNamespace(readback={"model": "glm-5.3"})
    hashes = iter((["before"] if snapshots else []) + ["after" if changed else "before"])
    main = main_phase("s2/run_s_lcmx.py", "reader =", run=run, run_dir=tmp_path,
        args=SimpleNamespace(reader="glm"), R=SimpleNamespace(GLMReader=lambda: reader),
        is_store=True, probe=lambda *a: [], sha=lambda db: next(hashes), view=[], man={},
        summary={}, isolation={"db_sha256_before_probes": "before"}, json=json,
        time=__import__("time"), shutil=__import__("shutil"), continuity=lambda *a: [],
        db_ro=lambda db: SimpleNamespace(execute=lambda q: SimpleNamespace(fetchone=lambda: [0])))
    rc = main()
    return rc, json.loads((tmp_path / "summary.json").read_text()), cpdir


@pytest.mark.parametrize("snapshots,changed", [(True, True), (False, True), (True, False)])
def test_s2_probe_isolation_failure(tmp_path, snapshots, changed):
    rc, summary, cpdir = s2_receipt_phase(tmp_path, snapshots=snapshots, changed=changed)
    assert summary["status"] == ("FAILED" if changed else "DONE")
    assert bool(rc) == changed
    if snapshots:
        assert json.loads((cpdir / "summary.json").read_text())["status"] == summary["status"]


@pytest.mark.parametrize("model", ["other-model", None])
def test_reader_model_mismatch_fails_run(tmp_path, monkeypatch, model):
    import urllib.request
    tree = ast.parse((TRACK / "s2/s2lib/reader.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GLMReader"]
    ns = {"json": json, "urllib": __import__("urllib"), "READER_TIMEOUT_S": 1}
    exec(compile(tree, "reader", "exec"), ns)
    reader = ns["GLMReader"].__new__(ns["GLMReader"])
    reader._glm = SimpleNamespace(URL="https://offline.invalid", MODEL="glm-5.3")
    reader._k, reader.readback = "synthetic", {"model": None}
    models = iter(["glm-5.3", model, "glm-5.3"])
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps({"model": next(models), "choices": [{"message": {"content": "ok"}}]}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response())
    reader._post([])
    with pytest.raises(RuntimeError, match="reader pin mismatch"):
        reader._post([])
    assert reader.readback["model"] == model
    reader._post([])  # a later matching response cannot clear the run failure
    rc, summary, cpdir = s2_receipt_phase(tmp_path, reader=reader)
    assert rc != 0 and summary["status"] == "FAILED"
    assert json.loads((cpdir / "summary.json").read_text())["status"] == "FAILED"


@pytest.mark.parametrize("rc,timed_out", [(7, False), (0, True)])
def test_s4_forced_checkpoint_failure(rc, timed_out):
    summary = {}
    tree = ast.parse((TRACK / "s4/run_s_codex.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    block = next(n for n in main.body if isinstance(n, ast.If)
                 and ast.unparse(n.test).startswith("a.force_event and"))
    ns = {"a": SimpleNamespace(force_event=True), "summary": summary, "sid": "synthetic",
          "HOME": Path("unused"), "rdir": Path("unused"), "ws": Path("unused"), "turn_log": [],
          "PR": SimpleNamespace(parse_rollout=lambda *a: {"token_series": []}),
          "base_cmd": lambda *a: [], "run_cli": lambda *a: {"rc": rc, "timed_out": timed_out, "wall_s": 1}}
    exec(compile(ast.Module(body=[block], type_ignores=[]), "forced-checkpoint", "exec"), ns)
    assert summary["status"] == "FAILED"


def test_s4_auth_refresh_failure():
    receipts = []
    rc = main_phase("s4/run_s_codex.py", "summary['finished'] =",
        summary={"pin_ok": True, "isolation_ok": True, "auth_refreshed_in_run": True},
        time=__import__("time"), rdir=Path("unused"), json=json,
        jdump=lambda p, s: receipts.append(dict(s)))()
    assert receipts[0]["status"] == "FAILED" and rc != 0


def test_s4_reused_home_config_rejected(tmp_path):
    (tmp_path / "config.toml").write_text('model = "synthetic"\n')
    original_auth = tmp_path / "auth.json"
    original_auth.write_text("{}")
    calls = []
    ns = functions("s4/run_s_codex.py", "setup_home", HOME=tmp_path, USER_AUTH=tmp_path / "missing",
        os=os, shutil=SimpleNamespace(copy2=lambda *a: calls.append(a)), sha=lambda p: "synthetic",
        json=json, codex_bin=lambda: "fake", env=lambda: {},
        subprocess=SimpleNamespace(DEVNULL=-3, run=lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="logged in chatgpt", stderr="")))
    with pytest.raises(RuntimeError, match="config.toml"):
        ns["setup_home"]()
    assert not calls and original_auth.read_text() == "{}"


def test_scoring_manifest_filters_and_removes_runs(tmp_path, monkeypatch):
    import hashlib
    from datetime import datetime
    out, logs = tmp_path / "decision", tmp_path / "logs"
    logs.mkdir()
    for seed in (1, 2):
        rdir = tmp_path / "external/lc-runs" / f"seed-{seed}" / f"d{seed}-r1-default/cp-304"
        rdir.mkdir(parents=True)
        (rdir / "summary.json").write_text("{}")
        (logs / f"s1-lossless-claw-d{seed}-r1.log.wall").write_text("exit 0 end 2\n")
    ns = runpy.run_path(str(TRACK / "decision/score_decision.py"))
    ns["main"].__globals__.update(score=lambda *a: {"schema": "score-s-v1", "arm": "lossless-claw",
        "seed": a[0].name}, classify_loss=lambda s: {})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    argv = ["score_decision", "--run-root", str(tmp_path / "runs"), "--external-root", str(tmp_path / "external"),
            "--material", str(tmp_path), "--logs", str(logs), "--out", str(out),
            "--arms", "lossless-claw", "--checkpoints", "304", "--seeds"]
    for seed in (1, 2):
        monkeypatch.setattr(sys, "argv", argv + [str(seed)])
        ns["main"]()
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "track-s-score-manifest-v2"
    assert set(manifest["entries"]) == {f"{p}/cp-304/lossless-claw.seed-{s}.d{s}-r1.json"
                                       for p in ("scores", "loss") for s in (1, 2)}
    for key, entry in manifest["entries"].items():
        seed = key.split("seed-")[1][0]
        assert entry["sha256"] == hashlib.sha256((out / key).read_bytes()).hexdigest()
        assert entry["receipt_sha256"] == hashlib.sha256((logs / f"s1-lossless-claw-d{seed}-r1.log.wall").read_bytes()).hexdigest()
        datetime.fromisoformat(entry["scored_at"])
    scores = out / "scores/cp-304"
    (scores / "stray.json").write_text("not even JSON")
    report = runpy.run_path(str(TRACK / "scorer/report_s.py"))
    report["main"].__globals__["render"] = lambda runs, *a: ("report\n", {"seeds": sorted(s for _, s in runs)})
    report_args = ["--scores", str(scores), "--out", str(out / "report.md"), "--json", str(out / "report.json")]
    report["main"](report_args)
    result = json.loads((out / "report.json").read_text())
    assert result["seeds"] == ["seed-1", "seed-2"]
    assert result["ignored_unmanifested"] == ["stray.json"]
    (logs / "s1-lossless-claw-d1-r1.log.wall").write_text("exit 7 end 2\n")
    monkeypatch.setattr(sys, "argv", argv + ["1"])
    ns["main"]()
    assert set(json.loads(manifest_path.read_text())["entries"]) == {
        f"{p}/cp-304/lossless-claw.seed-2.d2-r1.json" for p in ("scores", "loss")}
    assert not (scores / "lossless-claw.seed-1.d1-r1.json").exists()
    manifest_path.unlink()
    (scores / "stray.json").write_text("{}")
    report["main"](report_args)
    assert json.loads((out / "report.json").read_text())["manifest"] == "absent"


@pytest.fixture
def manifest_api(monkeypatch):
    monkeypatch.syspath_prepend(str(TRACK / "scorer"))
    return SimpleNamespace(**runpy.run_path(str(TRACK / "scorer/score_manifest.py")))


def manifest_file(api, root, key, payload):
    path = root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    manifest = api.load(root / "manifest.json") or {"schema": api.SCHEMA, "entries": {}}
    manifest["entries"][key] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "receipt_sha256": "synthetic", "scored_at": "synthetic"}
    api.write_atomic(root / "manifest.json", manifest)
    return path


@pytest.fixture
def decision_scorer(tmp_path, monkeypatch, manifest_api):
    out, logs = tmp_path / "decision", tmp_path / "logs"
    logs.mkdir()
    for cp in (176, 304):
        directory = tmp_path / "external/lc-runs/seed-1/d1-r1-default" / f"cp-{cp}"
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text("{}")
    (logs / "s1-lossless-claw-d1-r1.log.wall").write_text("exit 0 end 2\n")
    main = runpy.run_path(str(TRACK / "decision/score_decision.py"))["main"]
    main.__globals__.update(score=lambda *a: {"schema": "score-s-v1", "arm": "lossless-claw", "seed": "seed-1"},
                            classify_loss=lambda s: {"counts": {"synthetic": 1}})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    argv = ["score_decision", "--run-root", str(tmp_path / "runs"), "--external-root", str(tmp_path / "external"),
            "--material", str(tmp_path), "--logs", str(logs), "--out", str(out),
            "--arms", "lossless-claw", "--seeds", "1", "--checkpoints"]
    def invoke(*cps):
        monkeypatch.setattr(sys, "argv", argv + [str(cp) for cp in cps])
        main()
    return out, main, invoke


def paired_analysis(root, material, api):
    directory = material / "seed-1"
    directory.mkdir(exist_ok=True)
    (directory / "facts.json").write_text("[]")
    return functions("decision/analyze_paired.py", "score", "losses", "pair_axes", "mcnemar",
        H=root, MATERIAL=material, SEEDS=[1], ARMS=["first", "second"], score_manifest=api,
        grid_by_row=lambda *a: {}, json=json, math=__import__("math"))


def test_manifest_admission_binds_file_bytes_and_checkpoint(tmp_path, manifest_api):
    key = "scores/cp-304/arm.seed-1.d1-r1.json"
    path = manifest_file(manifest_api, tmp_path, key, {"schema": "score-s-v1", "arm": "arm", "seed": "seed-1"})
    manifest = manifest_api.load(tmp_path / "manifest.json")
    assert manifest_api.admitted(tmp_path, manifest, path)
    other = tmp_path / "scores/cp-176" / path.name
    other.parent.mkdir()
    other.write_bytes(path.read_bytes())
    assert not manifest_api.admitted(tmp_path, manifest, other)
    path.write_text(path.read_text() + "\n")
    assert not manifest_api.admitted(tmp_path, manifest, path)
    report = functions("scorer/report_s.py", "load", json=json, score_manifest=manifest_api, PREFIX={})
    assert report["load"](path.parent) == report["load"](other.parent) == {}
    (tmp_path / "manifest.json").write_text(json.dumps({path.stem: {"receipt_sha256": "synthetic"}}))
    assert manifest_api.load(tmp_path / "manifest.json")["entries"] == {}
    assert not manifest_api.admitted(tmp_path, manifest_api.load(tmp_path / "manifest.json"), path)
    (tmp_path / "manifest.json").unlink()
    assert manifest_api.load(tmp_path / "manifest.json") is None


def test_incremental_scoring_revokes_other_checkpoints(decision_scorer, manifest_api):
    out, _, invoke = decision_scorer
    invoke(176, 304)
    invoke(304)
    manifest = manifest_api.load(out / "manifest.json")
    assert set(manifest["entries"]) == {f"{p}/cp-304/lossless-claw.seed-1.d1-r1.json" for p in ("scores", "loss")}
    for population in ("scores", "loss"):
        old = out / population / "cp-176/lossless-claw.seed-1.d1-r1.json"
        assert not old.exists() and not manifest_api.admitted(out, manifest, old)
        assert manifest_api.admitted(out, manifest, out / population / "cp-304" / old.name)


def test_scoring_revocation_persisted_before_delete_and_failure(decision_scorer, manifest_api, tmp_path, monkeypatch):
    out, main, invoke = decision_scorer
    invoke(176, 304)
    name, deleted = "lossless-claw.seed-1.d1-r1.json", []
    unlink = Path.unlink
    def checked_unlink(path, *args, **kwargs):
        if path.name == name:
            assert not any(Path(k).name == name for k in manifest_api.load(out / "manifest.json")["entries"])
            deleted.append(path)
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", checked_unlink)
    def fail_score(*args):
        raise RuntimeError("synthetic scoring interruption")
    main.__globals__["score"] = fail_score
    with pytest.raises(RuntimeError, match="synthetic scoring interruption"):
        invoke(304)
    assert len(deleted) == 4 and manifest_api.load(out / "manifest.json")["entries"] == {}
    report = functions("scorer/report_s.py", "load", json=json, score_manifest=manifest_api, PREFIX={})
    analysis = paired_analysis(out, tmp_path, manifest_api)
    analysis["ARMS"] = ["lossless-claw", "second"]
    # Even leftover valid bytes cannot regain admission after the interruption.
    leftover = out / "scores/cp-304" / name
    leftover.write_text(json.dumps({"schema": "score-s-v1", "arm": "lossless-claw", "seed": "seed-1"}))
    assert report["load"](leftover.parent) == {}
    assert analysis["score"]("lossless-claw", 1, "r1", 304) is None
    assert analysis["losses"](304)["lossless-claw"] == {}
    assert analysis["pair_axes"](304)["status"] == "INCOMPLETE"


@pytest.mark.parametrize("mode", ["absent", "unmanifested", "facts_kept", "continuation", "continuity"])
def test_paired_analysis_requires_admission_and_complete_metrics(tmp_path, manifest_api, mode):
    root = tmp_path / "decision"
    for arm in ("first", "second"):
        for run in ("r1", "r2"):
            payload = {"probes": {}, "metrics": {"facts_kept": {"complete": True},
                       "continuation": {"complete": True, "denominator": 0}, "continuity": {"complete": True}}}
            if mode in payload["metrics"] and arm == "first" and run == "r1":
                payload["metrics"][mode]["complete"] = False
            manifest_file(manifest_api, root, f"scores/cp-304/{arm}.seed-1.d1-{run}.json", payload)
    admitted_loss = manifest_file(manifest_api, root, "loss/cp-304/first.seed-1.d1-r1.json", {"counts": {"synthetic": 1}})
    (admitted_loss.parent / "first.seed-1.stray.json").write_text(json.dumps({"counts": {"synthetic": 99}}))
    if mode == "absent":
        (root / "manifest.json").unlink()
    elif mode == "unmanifested":
        manifest = manifest_api.load(root / "manifest.json")
        del manifest["entries"]["scores/cp-304/first.seed-1.d1-r1.json"]
        manifest_api.write_atomic(root / "manifest.json", manifest)
    analysis = paired_analysis(root, tmp_path, manifest_api)
    result = analysis["pair_axes"](304)
    assert result["status"] == "INCOMPLETE" and "seed-1/r1" not in result["pairs"]
    suffix = "" if mode in ("absent", "unmanifested") else " (incomplete score)"
    assert f"seed-1/r1{suffix}" in result["missing_pairs"]
    assert analysis["losses"](304)["first"] == ({} if mode == "absent" else {"synthetic": 1})


def test_report_keeps_unmanifested_prefix_timing(tmp_path, manifest_api):
    scores = tmp_path / "scores/cp-304"
    scores.mkdir(parents=True)
    manifest_api.write_atomic(tmp_path / "manifest.json", {"schema": manifest_api.SCHEMA, "entries": {}})
    (scores / "prefix.json").write_text(json.dumps({"schema": "score-s-v1", "population": "prefix60k", "arm": "arm",
        "seed": "seed-1", "run_dir": "synthetic", "metrics": {"latency": {"status": "OK", "timing_label": "synthetic",
        "events": {"count": 1, "p90": 1, "max": 1}, "calls": {"count": 1, "p90": 1, "max": 1}}}}))
    report = functions("scorer/report_s.py", "load", "prefix_cell", "fmt", json=json, score_manifest=manifest_api, PREFIX={})
    metadata = {}
    assert report["load"](scores, metadata) == {}
    assert metadata["prefix_unmanifested"] == ["prefix.json"] and metadata["ignored_unmanifested"] == []
    assert "seed-1 synthetic" in report["prefix_cell"]("arm")


def test_report_without_manifest_reads_every_score(tmp_path, manifest_api):
    """score_s.py output (run_smoke.sh) has no manifest: the report reads it and records the manifest as absent."""
    scores = tmp_path / "plain"
    scores.mkdir()
    (scores / "arm.seed-1.json").write_text(json.dumps({"schema": "score-s-v1", "arm": "arm", "seed": "seed-1",
                                                         "run": "r1", "metrics": {}}))
    report = functions("scorer/report_s.py", "load", json=json, score_manifest=manifest_api, PREFIX={})
    metadata = {}
    assert list(report["load"](scores, metadata)) == [("arm", "seed-1")]
    assert metadata["manifest"] == "absent" and metadata["ignored_unmanifested"] == []


def test_spread_is_unmeasured_when_a_run_has_no_fact_rate(tmp_path):
    logs, runs = tmp_path / "logs", tmp_path / "runs"
    logs.mkdir()
    for run in ("r1", "r2"):
        (logs / f"s2-A-d1-{run}.log.wall").write_text("start 1\nexit 0 end 2\n")
        summary = runs / "A" / "seed-1" / f"d1-{run}" / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({"events": []}))
    value = {"r1": None, "r2": 0.5}
    ns = functions("decision/analyze_paired.py", "per_arm", LOGS=logs, RUNS=runs, SEEDS=[1], ARMS=["A"], CPS=[304],
        loads=lambda: [], st=lambda xs: None, json=json, re=__import__("re"), score=lambda arm, seed, run, cp: {
            "stored_level3": {"level3": 0, "leaves": 1}, "metrics": {"facts_kept": {"value": value[run]}}})
    out = ns["per_arm"]()["A"]
    assert out["facts_cp304_r1_r2_spread"] == {"seed-1": None}
    assert out["spread_unmeasured"] == ["seed-1"] and out["spread_over_0.10"] == []
