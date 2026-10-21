"""v2.2 rescore/audit regressions; fixture receipts make no floor timing claim."""
from argparse import Namespace as NS
import copy
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest
import yaml

from bench.instruments.local_catchup import catchup, common, floor, measurement, recall, rescore
from bench.instruments.local_catchup.tests.test_1586_fix1 import cover
from bench.instruments.local_catchup.tests.test_1586_fix2 import events
from bench.instruments.local_catchup.tests.test_1760 import receipts, write
from bench.instruments.local_catchup.tests.test_corpus_audit import REPO

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit",
                  "bench.instruments.local_catchup.tests.test_fix_round3"]


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(recall, "bootstrap", lambda v, f, seed: [f(v)])


def coverage_pair(root):
    cells = []
    for baseline, speed in (("empty", 10.), ("upgrade", .1)):
        cell = root/baseline
        (cell/"catchup").mkdir(parents=True)
        raw = dict(rows_s=speed, end_to_end_s=7300, state="INCOMPLETE", audit=dict(complete=False))
        common.dump(cell/"catchup/summary.json", raw)
        write(cell/"catchup/events.jsonl", events(.572))
        common.dump(cell/"manifest.json", dict(rows=250000))
        summary = dict(floor.rate(raw, 7300, 250000, events(.572)), baseline=baseline)
        common.dump(cell/"summary.json", summary)
        cells.append(summary)
    summary = dict(state="FAIL", cells=cells)
    common.dump(root/"summary.json", summary)
    return summary


@pytest.mark.parametrize("forgery", ["omit", "duplicate"])
def test_coverage_pair_summary_cannot_hide_disk_cell(tmp_path, forgery):
    summary = coverage_pair(tmp_path)
    assert rescore.v2(tmp_path)["state"] == "FAIL"
    summary.update(state="PASS", cells=[summary["cells"][0]]*(2 if forgery == "duplicate" else 1))
    common.dump(tmp_path/"summary.json", summary)
    with pytest.raises(AssertionError):
        rescore.v2(tmp_path)


@pytest.mark.parametrize("state", ["OFF", "ON-catching-up", "ON-covered"])
def test_early_invalid_cannot_hide_recall_receipts(tmp_path, state):
    receipts(tmp_path, kinds=("timeout", "pass"))
    cell = tmp_path/f"0-{state}"
    forged = dict(state="INVALID", reason="covered_store_incomplete")
    for name in ("summary.json", "raw-summary.json"):
        common.dump(cell/name, forged)
    with pytest.raises((AssertionError, ValueError)):
        rescore.v2(cell)


def test_early_invalid_summary_cannot_hide_raw_classes(tmp_path):
    cell = tmp_path/"0-ON-covered"
    cell.mkdir()
    common.dump(cell/"summary.json", dict(state="INVALID", reason="covered_store_incomplete"))
    common.dump(cell/"raw-summary.json", dict(state="FAIL", classes={}))
    with pytest.raises((AssertionError, ValueError)):
        rescore.v2(cell)


@pytest.mark.parametrize("key", ["cells", "blocks"])
def test_empty_root_is_not_pass(tmp_path, key):
    common.dump(tmp_path/"summary.json", dict(state="PASS", **{key: []}))
    with pytest.raises(AssertionError):
        rescore.v2(tmp_path)


def test_aborted_root_returns_aborted(tmp_path):
    summary = dict(state="ABORTED", reason="floor_deadline")
    common.dump(tmp_path/"summary.json", summary)
    assert rescore.v2(tmp_path) == summary
    proc = subprocess.run([os.sys.executable, "-B", "-m", rescore.__name__, str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0 and json.loads(proc.stdout) == summary


def test_missing_root_summary_is_clear(tmp_path):
    with pytest.raises(ValueError, match="missing root summary.json"):
        rescore.v2(tmp_path)
    proc = subprocess.run([os.sys.executable, "-B", "-m", rescore.__name__, str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode != 0 and "missing root summary.json" in proc.stderr


def test_zero_recall_rows_capped_is_invalid(tmp_path, monkeypatch):
    common.dump(tmp_path/"manifest.json", dict(canaries=[], expected=dict(chunks={}, leaves={})))
    common.dump(tmp_path/"queries.json", [dict(id=str(i), query_class="broad_nl", text="fixture") for i in range(200)])
    a = NS(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path, state="OFF",
           smoke=False, rows=250000, seed=1586, block=0)
    engine = NS(_dag=NS(add_node=lambda *a, **kw: None), _config=NS(recall_query_timeout_s=1), shutdown=lambda: None)
    monkeypatch.setattr(recall, "runtime", lambda _: (a, None, engine, 0))
    # The expired reader issues no calls, even with a full pinned query manifest.
    class Thread:
        def __init__(self, **kw): pass
        def start(self): pass
        def join(self, *args): pass
        def is_alive(self): return False
    monkeypatch.setattr(recall.threading, "Thread", Thread)
    monkeypatch.setattr(recall.threading.Event, "wait", lambda *a: True)
    monkeypatch.setattr(recall, "publication_trace", lambda *a: ([], 0, True))
    monkeypatch.setattr(recall, "safety", lambda *a: {})
    monkeypatch.setattr(recall, "integrity", lambda *a: None)
    monkeypatch.setattr(recall, "audit", lambda *a: dict(classes={}))
    recall.worker(None)
    raw = common.load(tmp_path/"raw-summary.json")
    assert (raw["state"], raw["reason"], raw["measurement_valid"]) == ("INVALID", "volume_cap", False)
    assert any(e["event"] == "volume_cap" for e in map(json.loads, (tmp_path/"events.jsonl").read_text().splitlines()))


@pytest.mark.parametrize("problem", ["absent", "empty", "one-pin", "metadata-absent", "metadata-empty", "metadata-one-pin", "integer", "null"])
def test_persisted_pins_cannot_be_recreated(corpus, tmp_path, monkeypatch, problem):
    db, fresh, mod = corpus
    source, out = tmp_path/"covered", tmp_path/"out"
    source.mkdir()
    out.mkdir()
    with sqlite3.connect(db) as src, sqlite3.connect(source/"store.db") as dst:
        src.backup(dst)
    persisted = copy.deepcopy(fresh)
    cover(source/"store.db", persisted, mod)
    assert measurement.audit(source/"store.db", persisted)["complete"]
    key = "observed_metadata" if problem.startswith("metadata-") else "observed_vectors"
    if problem.endswith("absent"):
        persisted.pop(key)
    elif problem.endswith("empty"):
        persisted[key] = {}
    elif problem.endswith("one-pin"):
        persisted[key].pop(next(iter(persisted[key])))
    elif problem == "integer":
        persisted = 1
    else:
        persisted = None
    common.dump(source/"manifest.json", persisted)
    common.dump(out/"manifest.json", fresh)
    before = (out/"manifest.json").read_bytes()
    monkeypatch.setattr(measurement.subprocess, "Popen", lambda *a, **kw: pytest.fail("invalid pins reached launch"))
    a = NS(out=out, scratch=tmp_path, covered_store=source)
    result = measurement.cell(a, "ON-covered", 0, "recall")
    assert result["state"] == "INVALID"
    assert result["pre_audit"]["reason"] == "covered_store_manifest_missing"
    assert (out/"manifest.json").read_bytes() == before
    assert rescore.v2(out/"0-ON-covered") == result


def test_workflow_budget_uses_decimal_bounded_input():
    jobs = yaml.safe_load((REPO/".github/workflows/lcmx-1586-floor.yml").read_text())["jobs"]
    script = jobs["validate"]["steps"][0]["run"]
    for minutes in ("0500", "0330", "18446744073709551946"):
        proc = subprocess.run(["bash", "-c", script], env={**os.environ, "CELL": "recall", "MINUTES": minutes}, capture_output=True)
        assert proc.returncode != 0, minutes
    for cell, low in (("coverage-rate", 150), ("coverage-pair", 300), ("recall", 330)):
        for minutes in range(low, 331):
            proc = subprocess.run(["bash", "-c", script], env={**os.environ, "CELL": cell, "MINUTES": str(minutes)}, capture_output=True)
            assert proc.returncode == 0, (cell, minutes)


@pytest.mark.parametrize("local", [True, False])
def test_worker_reads_cell_manifest_and_baseline_fallback(tmp_path, monkeypatch, local):
    root = tmp_path/"0-ON-covered"
    root.mkdir()
    baseline = dict(canaries=[], marker="baseline", observed_vectors={})
    cell_manifest = dict(canaries=[], marker="cell", observed_vectors={"fixture": "digest"})
    common.dump(tmp_path/"manifest.json", baseline)
    if local:
        common.dump(root/"manifest.json", cell_manifest)
    common.dump(tmp_path/"queries.json", [])
    a = NS(root=root, out=tmp_path, state="ON-covered")
    engine = NS(shutdown=lambda: None)
    monkeypatch.setattr(recall, "runtime", lambda _: (a, None, engine, 0))
    class Observed(Exception):
        pass
    def wait(a, mod, engine, manifest):
        assert manifest == (cell_manifest if local else baseline)
        assert a._manifest is manifest
        raise Observed
    monkeypatch.setattr(recall, "wait_covered", wait)
    with pytest.raises(Observed):
        recall.worker(None)


def test_catchup_foreign_files_receives_original_home(fake_catchup, tmp_path, monkeypatch):
    a, _, _, _, _, _ = fake_catchup
    a.user_home = str(tmp_path/"real-home")
    monkeypatch.setenv("HOME", str(tmp_path/"calibration-home"))
    seen = []
    def foreign(*args, **kw):
        seen.append(kw["user_home"])
        assert kw["user_home"] == Path(a.user_home)
        return []
    monkeypatch.setattr(catchup, "foreign_files", foreign)
    catchup.run(a)
    assert seen


def test_chain_assertion_rejects_substring_dependency(monkeypatch):
    from bench.instruments.local_catchup.tests.test_1586_v2 import test_artifact_chain_and_confinement_are_explicit
    original = yaml.safe_load
    def forged(text):
        workflow = original(text)
        if "covered-store-2" in workflow["jobs"]:
            workflow["jobs"]["covered-store-2"]["needs"] = "covered-store-1-suffix"
        return workflow
    monkeypatch.setattr(yaml, "safe_load", forged)
    with pytest.raises(AssertionError):
        test_artifact_chain_and_confinement_are_explicit()


def synthetic_receipts(root, case):
    """Exercise each non-smoke derivation independently with 400 recall rows."""
    results = receipts(root, activity=case != "inactive", valid=case != "volume_cap")
    if case in ("complete", "row-complete"):
        cell = root/"0-ON-catching-up"
        if case == "complete":
            common.dump(cell/"final-audit.json", dict(classes=dict(chunks=dict(missing_count=0))))
        else:
            data = list(map(json.loads, (cell/"recalls.jsonl").read_text().splitlines()))
            data[0]["incomplete"] = False
            write(cell/"recalls.jsonl", data)
        result = results[1]["result"]
        result.update(state="INVALID", measurement_valid=False)
        for name in ("raw-summary.json", "summary.json"):
            common.dump(cell/name, result)
        common.dump(root/"summary.json", dict(state="INVALID", blocks=results))
    return results


@pytest.mark.parametrize("case,expected", [("active", "PASS"), ("inactive", "INVALID"),
    ("complete", "INVALID"), ("row-complete", "INVALID"), ("volume_cap", "INVALID")])
def test_non_smoke_rescore_derivations(tmp_path, case, expected):
    synthetic_receipts(tmp_path, case)
    computed = rescore.v2(tmp_path)
    assert computed["state"] == expected
    cell = computed["blocks"][1]
    request = common.load(tmp_path/"0-ON-catching-up/request.json")
    assert request["rows"] >= 250000 and not request["smoke"]
    assert len(common.load(tmp_path/"queries.json")) >= 200
    assert cell["catchup_activity"] is (case != "inactive")
    assert cell["measurement_valid"] is (case == "active")
    assert (cell.get("reason") == "volume_cap") is (case == "volume_cap")
