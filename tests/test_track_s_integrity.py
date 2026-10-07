"""Offline measurement-integrity regressions for the Track S run kit."""
import ast
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

TRACK = Path(__file__).resolve().parents[1] / "bench/instruments/compaction_probe/track_s"


def functions(path, *names, **namespace):
    namespace.setdefault("Path", Path)
    tree = ast.parse((TRACK / path).read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


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
            "facts_kept": {"lost_before_compaction": {"ids": ["a" if arm == "first" else "b"]}},
            "continuation": {"correct_fields": [], "denominator": 0}}}
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
             "TRACK_S_MATERIAL": str(tmp_path)}, capture_output=True, text=True, timeout=5)
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
    subprocess.run(["node", "-e", js], check=True, timeout=5)


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
             "TRACK_S_MATERIAL": str(tmp_path), "S2_PRODUCT_WORKTREE": "fake"}, timeout=5)
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
                 "S2_PRODUCT_WORKTREE": "fake"}, timeout=5)
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
