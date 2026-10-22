"""v2.1 integrity regressions, adapted from the v2 review probes."""
from argparse import Namespace as NS
import itertools
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest
import yaml

from bench.instruments.local_catchup import common, floor, measurement, recall, rescore
from bench.instruments.local_catchup.tests.test_1586_fix1 import cover
from bench.instruments.local_catchup.tests.test_1586_fix2 import window_batches
from bench.instruments.local_catchup.tests.test_1586_v2 import rows
from bench.instruments.local_catchup.tests.test_corpus_audit import REPO

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]
STATES = ("OFF", "ON-catching-up", "ON-covered")


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(recall, "bootstrap", lambda v, f, seed: [f(v)])


def test_floor_real_home_reaches_recall_safety(tmp_path, monkeypatch):
    real = tmp_path/"real"
    real.mkdir()
    probe = real/"foreign.txt"
    probe.write_text("synthetic fixture")
    monkeypatch.setenv("HOME", str(real))
    a = NS(out=tmp_path/"out", scratch=tmp_path/"scratch", cgroup=tmp_path, cell="recall",
           lcmx_repo=REPO, hermes_source=REPO, rows=8, model_cache=tmp_path/"cache",
           shape="short", baseline="empty", block=0, covered_store=None, max_minutes=330)
    a.out.mkdir()
    monkeypatch.setattr(floor, "measured_limits", lambda _: {})
    monkeypatch.setattr(floor.subprocess, "check_output", lambda *a, **kw: "fixture")
    def calibration(a):
        monkeypatch.setenv("HOME", str(a.scratch/"calibration/home"))
        return dict(standalone_calibration_s=.572)
    monkeypatch.setattr(floor, "standalone", calibration)
    monkeypatch.setattr(measurement, "immutable_cache", lambda *a: None)
    monkeypatch.setattr(measurement, "progress", lambda *a: dict(complete=True, classes={}, scan={}))
    monkeypatch.setattr(measurement, "build", lambda *a: dict(database_bytes=1, expected=dict(chunks=[]), rows=8))
    def child(argv, **kw):
        user_home = argv[argv.index("--user-home")+1] if "--user-home" in argv else None
        child = NS(out=a.out/"recall", scratch=a.scratch/"recall", model_cache=a.model_cache,
            rows=8, blocks=1, samples=1, queries=1, lcmx_repo=REPO, hermes_source=REPO,
            shape="short", baseline="empty", user_home=user_home)
        measurement.prepare(child)
        child.root, child.home, child.db = child.out, child.scratch/"worker", child.scratch/"db"
        with probe.open():
            with pytest.raises(RuntimeError, match="cross-profile access"):
                measurement.safety(child)
        assert Path(child.user_home) == real
        common.dump(child.out/"summary.json", dict(state="INVALID"))
    monkeypatch.setattr(floor.subprocess, "run", child)
    floor.limited(a)


def test_floor_real_home_reaches_coverage(tmp_path, monkeypatch):
    real = tmp_path/"real"
    monkeypatch.setenv("HOME", str(real))
    a = NS(out=tmp_path, scratch=tmp_path/"scratch", cgroup=tmp_path, cell="coverage-rate", baseline="empty")
    monkeypatch.setattr(floor, "measured_limits", lambda _: {})
    monkeypatch.setattr(floor.subprocess, "check_output", lambda *a, **kw: "fixture")
    monkeypatch.setattr(floor, "standalone", lambda _: monkeypatch.setenv("HOME", str(tmp_path/"calibration")))
    def coverage(a):
        assert Path(getattr(a, "user_home", Path.home())) == real
    monkeypatch.setattr(floor, "coverage", coverage)
    floor.limited(a)


@pytest.mark.parametrize("problem", ["vector", "metadata", "preimage", "expected", "identities", "rows", "shape", "baseline", "seed", "missing", "unparsable", "honest"])
def test_persisted_covered_manifest(corpus, tmp_path, monkeypatch, problem):
    db, fresh, mod = corpus
    source, out = tmp_path/"covered", tmp_path/"out"
    source.mkdir()
    out.mkdir()
    with sqlite3.connect(db) as src, sqlite3.connect(source/"store.db") as dst:
        src.backup(dst)
    persisted = json.loads(json.dumps(fresh))
    cover(source/"store.db", persisted, mod)
    assert measurement.audit(source/"store.db", persisted)["complete"]
    common.dump(source/"manifest.json", persisted)
    common.dump(out/"manifest.json", fresh)  # No in-memory pinning on the recall side.
    if problem in ("vector", "metadata"):
        with sqlite3.connect(source/"store.db") as c:
            c.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'" if problem == "vector"
                      else "UPDATE lcm_chunk_meta SET archived=1 WHERE chunk_id='1:0'")
    elif problem == "missing":
        (source/"manifest.json").unlink()
    elif problem == "unparsable":
        (source/"manifest.json").write_text("{")
    elif problem != "honest":
        persisted[problem] = "mismatch"
        common.dump(source/"manifest.json", persisted)
    class Launched(Exception):
        pass
    def launch(*args, **kw):
        raise Launched
    monkeypatch.setattr(measurement.subprocess, "Popen", launch)
    a = NS(out=out, scratch=tmp_path, covered_store=source, lcmx_repo=REPO, hermes_source=REPO, model_cache=tmp_path/"cache")
    if problem == "honest":
        with pytest.raises(Launched):
            measurement.cell(a, "ON-covered", 0, "recall")
        updated = common.load(out/"0-ON-covered/manifest.json")
        assert updated["observed_vectors"] == persisted["observed_vectors"]
        assert not measurement.audit(db, common.load(out/"manifest.json"))["changed_vectors"]
        return
    try:
        result = measurement.cell(a, "ON-covered", 0, "recall")
    except Launched:
        pytest.fail("tampered or missing persisted manifest reached worker launch")
    assert result["state"] == "INVALID"
    expected = "covered_store_manifest_missing" if problem in ("missing", "unparsable") else "covered_store_changed"
    assert result["pre_audit"]["reason"] == expected
    assert rescore.v2(out/"0-ON-covered") == result


def write(path, values):
    path.write_text("".join(json.dumps(v)+"\n" for v in values))


def receipts(root, *, valid=True, activity=True, kinds=("pass", "pass"), host=(.572, 5), early=False):
    root.mkdir(exist_ok=True)
    a = NS(out=root, seed=1586, standalone_calibration_s=.572)
    pubs = [dict(start=20, end=30, node_id=i) for i in range(10)]
    samples = []
    for cls, kind in zip(("canary", "broad_nl"), kinds):
        data = rows(199 if kind == "short" else 198 if kind == "shorttimeout" else 200, query_class=cls)
        if "timeout" in kind:
            data += rows(1, query_class=cls)
            data[-1]["health"] = "timeout"
        if kind in ("slow", "scaledslow"):
            for r in data[-40:]:
                r["seconds"] = 9. if kind == "slow" else 4.5
        samples += data
    samples = [dict(r, start=21, end=22, incomplete=True, coverage_start=dict(chunks=0),
                    coverage_end=dict(chunks=int(activity))) for r in samples]
    results = []
    for state in STATES:
        cell = root/f"0-{state}"
        cell.mkdir()
        if state == "ON-covered" and early:
            raw = dict(state="INVALID", reason="covered_store_incomplete")
            write(cell/"events.jsonl", [dict(violation="covered_store_incomplete")])
        else:
            ev = window_batches(*host)
            if not valid:
                ev += [dict(event="volume_cap")]
            write(cell/"events.jsonl", ev)
            write(cell/"recalls.jsonl", samples)
            write(cell/"publications.jsonl", pubs)
            common.dump(cell/"request.json", dict(seed=1586, state=state, block=0, rows=250000, smoke=False))
            common.dump(root/"queries.json", [{}]*400)
            common.dump(cell/"final-audit.json", dict(classes=dict(chunks=dict(missing_count=1))))
            raw = recall.score(samples, pubs, state=state, activity=dict(catchup_activity=activity))
            raw.update(measurement_valid=valid and raw["state"] != "INVALID", catchup_activity=activity)
            if not valid:
                raw.update(state=common.rollup([raw["state"], "INVALID"]), reason="volume_cap")
        if "classes" in raw:
            raw["measurement_valid"] = raw["state"] != "INVALID"
        for name in ("raw-summary.json", "summary.json"):
            common.dump(cell/name, raw)
        results.append(dict(state=state, block=0, result=raw))
    recall.finalize_block(a, 0, results)
    common.dump(root/"summary.json", dict(state=common.rollup(r["result"]["state"] for r in results), blocks=results))
    return results


@pytest.mark.parametrize("forgery", ["measurement_valid", "catchup_activity", "dropped_block"])
def test_summary_only_forgeries_rejected(tmp_path, forgery):
    results = receipts(tmp_path, valid=forgery != "measurement_valid", activity=forgery != "catchup_activity")
    if forgery == "dropped_block":
        summary = common.load(tmp_path/"summary.json")
        summary["blocks"].pop()
        common.dump(tmp_path/"summary.json", summary)
    else:
        for entry in results:
            root = tmp_path/f"0-{entry['state']}"
            data = [json.loads(s) for s in (root/"recalls.jsonl").read_text().splitlines()]
            pubs = [json.loads(s) for s in (root/"publications.jsonl").read_text().splitlines()]
            forged = recall.score(data, pubs, state=entry["state"], activity=dict(catchup_activity=True), calibration_s=recall.window_calibration(window_batches()))
            raw = common.load(root/"raw-summary.json")
            raw.update(forged, measurement_valid=True, catchup_activity=True)
            raw.pop("reason", None)
            common.dump(root/"raw-summary.json", raw)
            entry["result"] = raw
        common.dump(tmp_path/"summary.json", dict(state="PASS", blocks=results))
    if forgery == "dropped_block":
        with pytest.raises(AssertionError):
            rescore.v2(tmp_path)
    else:
        assert any(c.get("rescore_disagreement") for c in rescore.v2(tmp_path)["blocks"])


def test_rescore_floor_output_root(tmp_path):
    receipts(tmp_path/"recall")
    common.dump(tmp_path/"summary.json", common.load(tmp_path/"recall/summary.json"))
    assert rescore.v2(tmp_path) == rescore.v2(tmp_path/"recall")


@pytest.mark.parametrize("state", STATES)
def test_observed_failure_outranks_short_volume_and_activity(state):
    data = rows(1)
    data[0]["health"] = "timeout"
    result = recall.score_class(data, [], state=state, activity=dict(catchup_activity=False))
    assert result["failures"] == 1 and result["state"] == "FAIL"


def test_workflow_rejects_budget_before_setup():
    jobs = yaml.safe_load((REPO/".github/workflows/lcmx-1586-floor.yml").read_text())["jobs"]
    assert "validate" in jobs
    for name in ("floor", "covered-store-1"):
        assert "validate" in jobs[name]["needs"]
    assert "needs.validate.result == 'success'" in jobs["floor"]["if"]
    step = jobs["validate"]["steps"][0]
    for cell, minutes, expected in (("recall", 329, 1), ("recall", 330, 0), ("coverage-pair", 299, 1),
            ("coverage-pair", 300, 0), ("coverage-rate", 149, 1), ("coverage-rate", 150, 0),
            ("recall", 331, 1), ("bad", 330, 1), ("coverage-rate", 150.5, 1)):
        proc = subprocess.run(["bash", "-c", step["run"]], env={**os.environ, "CELL": cell, "MINUTES": str(minutes)}, capture_output=True)
        assert proc.returncode == expected, (cell, minutes, proc.stderr)


@pytest.mark.parametrize("shortfall,failure,expected", [(False, False, "UNCALIBRATED"), (True, False, "INVALID"), (True, True, "FAIL")])
def test_late_final_publication_uses_final_receipts(tmp_path, monkeypatch, shortfall, failure, expected):
    manifest = dict(canaries=[dict(value="fixture")], expected=dict(chunks={}, leaves={}))
    common.dump(tmp_path/"manifest.json", manifest)
    pinned = [dict(id=str(i), query_class=c, canary=0, text="fixture") for i in range(200) for c in ("canary", "broad_nl")]
    common.dump(tmp_path/"queries.json", pinned)
    a = NS(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path, state="OFF", smoke=False,
           rows=250000, seed=1586, block=0, volume_seconds=1)
    calls = []
    ticks = iter([20., 30.]*9 + [21., 22.]*400 + [20., 30.])
    monkeypatch.setattr(recall.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(recall.time, "monotonic", lambda: 0. if len(calls) < 400 else 100.)
    dag = NS(add_node=lambda *a, **kw: len(calls))
    node_count = []
    def add(*args, **kw):
        node_count.append(1)
        return len(node_count)
    dag.add_node = add
    engine = NS(_dag=dag, _config=NS(recall_query_timeout_s=1), shutdown=lambda: None)
    tools = NS(_lcm_recall_fts_arm=lambda *a, **kw: ([], None))
    def query(*args, **kw):
        calls.append(1)
        return json.dumps(dict(timeout=failure, hits=[dict(arms=["fts"])], provenance=dict(coverage=dict(fts="full"))))
    tools.lcm_recall = query
    monkeypatch.setattr(recall, "runtime", lambda _: (a, lambda _: tools, engine, 0))
    def reader(*args):
        for _ in range(9):
            dag.add_node(NS(depth=0), before_commit=True)
        return NS(_dag=NS(close=lambda: None), _store=NS(close=lambda: None))
    monkeypatch.setattr(recall, "recall_reader", reader)
    class Thread:
        def __init__(self, target, **kw): self.target = target
        def start(self): self.target()
        def join(self, *a): pass
        def is_alive(self): return False
    monkeypatch.setattr(recall.threading, "Thread", Thread)
    def trace(*args):
        assert len(calls) == 400 and not args[3].is_set()
        if not shortfall:
            dag.add_node(NS(depth=0), before_commit=True)
        return [], 1, True
    monkeypatch.setattr(recall, "publication_trace", trace)
    monkeypatch.setattr(recall, "safety", lambda *a: {})
    monkeypatch.setattr(recall, "integrity", lambda *a: None)
    monkeypatch.setattr(recall, "audit", lambda *a: dict(classes={}))
    recall.worker(None)
    result = common.load(tmp_path/"raw-summary.json")
    assert result["state"] == expected
    assert (result.get("reason") == "volume_cap") is shortfall
    ev = [json.loads(s) for s in (tmp_path/"events.jsonl").read_text().splitlines()] if (tmp_path/"events.jsonl").exists() else []
    assert any(e.get("event") == "volume_cap" for e in ev) is shortfall


def test_864_precedence_combinations(tmp_path):
    kinds = ("pass", "slow", "scaledslow", "short", "shorttimeout", "timeout")
    hosts = ((.572/2.14, 5), (.572, 5), (.572/2.14, 4))
    for i, (canary, broad, host, valid, active, early) in enumerate(itertools.product(kinds, kinds, hosts, (True, False), (True, False), (False, True))):
        root = tmp_path/str(i)
        results = receipts(root, valid=valid, activity=active, kinds=(canary, broad), host=host, early=early)
        computed = rescore.v2(root)
        assert [r["result"]["state"] for r in results] == [r["state"] for r in computed["blocks"]]
        for entry in results:
            for cls in entry["result"].get("classes", {}).values():
                if cls["failures"]:
                    assert cls["state"] == "FAIL", (i, entry["state"], canary, broad)
