"""v2.2 rescore/audit regressions; fixture receipts make no floor timing claim."""
from argparse import Namespace as NS
import copy
import json
import os
from pathlib import Path
import sqlite3
import shutil
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
    result = rescore.v2(cell)
    assert result["state"] == "FAIL"
    assert result["rescore_disagreement"]  # Forged early exit cannot hide the check.


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
    assert proc.returncode == 2 and "missing root summary.json" in proc.stderr
    assert str(tmp_path) in proc.stderr and "Traceback" not in proc.stderr
    assert len(proc.stderr.splitlines()) == 1


@pytest.mark.parametrize("state", ["OFF", "ON-catching-up", "ON-covered"])
def test_zero_recall_rows_capped_is_invalid(tmp_path, monkeypatch, state):
    out = tmp_path
    tmp_path = out/f"0-{state}"
    tmp_path.mkdir()
    common.dump(tmp_path/"manifest.json", dict(canaries=[], expected=dict(chunks={}, leaves={})))
    common.dump(out/"queries.json", [dict(id=str(i), query_class="broad_nl", text="fixture") for i in range(200)])
    a = NS(out=out, root=tmp_path, db=tmp_path/"db", home=tmp_path, state=state,
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
    monkeypatch.setattr(recall, "audit", lambda *a: dict(classes=dict(chunks=dict(missing_count=1), leaves=dict(missing_count=1))))
    common.dump(tmp_path/"request.json", dict(root=str(tmp_path), state=state, seed=1586, block=0, rows=250000, smoke=False))
    monkeypatch.setattr(recall, "wait_covered", lambda *a: None)
    monkeypatch.setattr(recall, "calibration_window", lambda *a: None)
    recall.worker(None)
    raw = common.load(tmp_path/"raw-summary.json")
    assert (raw["state"], raw["reason"], raw["measurement_valid"]) == ("INVALID", "volume_cap", False)
    assert any(e["event"] == "volume_cap" for e in map(json.loads, (tmp_path/"events.jsonl").read_text().splitlines()))
    entries = []
    for name in ("OFF", "ON-catching-up", "ON-covered"):
        cell = out/f"0-{name}"
        if name != state:
            shutil.copytree(tmp_path, cell)
            request = common.load(cell/"request.json")
            common.dump(cell/"request.json", dict(request, state=name))
        common.dump(cell/"summary.json", raw)
        entries.append(dict(state=name, block=0, result=dict(raw)))
    recall.finalize_block(NS(out=out, seed=1586, standalone_calibration_s=.572), 0, entries)
    common.dump(out/"summary.json", dict(state="INVALID", blocks=entries))
    assert rescore.v2(tmp_path)["reason"] == "volume_cap"
    assert rescore.v2(out)["state"] == "INVALID"
    write(tmp_path/"events.jsonl", [])
    with pytest.raises(AssertionError):
        rescore.v2(tmp_path)


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


@pytest.mark.parametrize("needs", ["covered-store-1-suffix", ["covered-store-1", "validate"]])
def test_chain_assertion_rejects_substring_dependency(monkeypatch, needs):
    from bench.instruments.local_catchup.tests.test_1586_v2 import test_artifact_chain_and_confinement_are_explicit
    original = yaml.safe_load
    def forged(text):
        workflow = original(text)
        if "covered-store-2" in workflow["jobs"]:
            workflow["jobs"]["covered-store-2"]["needs"] = needs
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


@pytest.mark.parametrize("floor_name", ["rows", "queries"])
def test_non_smoke_minimum_floors(tmp_path, floor_name):
    entries = receipts(tmp_path)
    if floor_name == "queries":
        common.dump(tmp_path/"queries.json", [{}]*199)
    for entry in entries:
        cell = tmp_path/f"0-{entry['state']}"
        if floor_name == "rows":
            request = common.load(cell/"request.json")
            request["rows"] = 249999
            common.dump(cell/"request.json", request)
        entry["result"].update(state="INVALID", measurement_valid=False)
        for name in ("raw-summary.json", "summary.json"):
            common.dump(cell/name, entry["result"])
    common.dump(tmp_path/"summary.json", dict(state="INVALID", blocks=entries))
    assert rescore.v2(tmp_path)["state"] == "INVALID"


def test_coverage_pair_requires_both_baselines(tmp_path):
    source, target = tmp_path/"source", tmp_path/"target"
    summary = coverage_pair(source)
    target.mkdir()
    shutil.copytree(source/"empty", target/"empty")
    common.dump(target/"summary.json", dict(state="PASS", cells=[summary["cells"][0]]))
    with pytest.raises(AssertionError):
        rescore.v2(target)


@pytest.mark.parametrize("auditor", ["rescore", "finalize"])
@pytest.mark.parametrize("source,target,block", [
    ("OFF", "ON-covered", 0), ("ON-covered", "OFF", 0), ("OFF", "OFF", 1),
])
def test_recall_aggregate_rejects_request_identity(tmp_path, auditor, source, target, block):
    entries = receipts(tmp_path)
    assert rescore.v2(tmp_path)["state"] == "PASS"
    cell = tmp_path/f"0-{target}"
    if source != target:
        shutil.copytree(tmp_path/f"0-{source}", cell, dirs_exist_ok=True)
    request = common.load(cell/"request.json")
    common.dump(cell/"request.json", dict(request, block=block))
    with pytest.raises(AssertionError, match="request identity"):
        if auditor == "rescore":
            rescore.v2(tmp_path)
        else:
            recall.finalize_block(NS(out=tmp_path, seed=1586, standalone_calibration_s=.572), 0, entries)


def test_off_receipts_cannot_pass_as_on_covered(tmp_path):
    entries = receipts(tmp_path)
    cell = tmp_path/"0-OFF"
    rows = [dict(json.loads(s), semantic_healthy=False) for s in (cell/"recalls.jsonl").read_text().splitlines()]
    write(cell/"recalls.jsonl", rows)
    pubs = list(map(json.loads, (cell/"publications.jsonl").read_text().splitlines()))
    assert recall.score_class(rows, pubs, state="ON-covered")["state"] == "FAIL"
    recall.finalize_block(NS(out=tmp_path, seed=1586, standalone_calibration_s=.572), 0, entries)
    assert entries[0]["result"]["state"] == "PASS"
    shutil.copytree(cell, tmp_path/"0-ON-covered", dirs_exist_ok=True)
    entries[2]["result"] = dict(entries[0]["result"])
    common.dump(tmp_path/"summary.json", dict(state="PASS", blocks=entries))
    with pytest.raises(AssertionError, match="request identity"):
        rescore.v2(tmp_path)


@pytest.mark.parametrize("block", [1, 2])
def test_single_block_root_and_request_identity(tmp_path, block):
    source, target = tmp_path/"source", tmp_path/"target"
    source.mkdir()
    entries = receipts(source)
    target.mkdir()
    shutil.copyfile(source/"queries.json", target/"queries.json")
    for entry in entries:
        state = entry["state"]
        shutil.copytree(source/f"0-{state}", target/f"{block}-{state}")
        entry["block"] = block
        request = common.load(target/f"{block}-{state}/request.json")
        common.dump(target/f"{block}-{state}/request.json", dict(request, block=block))
    common.dump(target/"summary.json", dict(state="PASS", blocks=entries))
    assert rescore.v2(target)["state"] == "PASS"
    common.dump(target/f"{block}-OFF/request.json", dict(request, block=0))
    with pytest.raises(AssertionError):
        rescore.v2(target)


@pytest.mark.parametrize("kind", ["recall", "pair", "floor"])
def test_aborted_root_cannot_hide_complete_fail(tmp_path, kind):
    if kind == "pair":
        coverage_pair(tmp_path)
    elif kind == "floor":
        receipts(tmp_path/"recall", kinds=("timeout", "pass"))
    else:
        receipts(tmp_path, kinds=("timeout", "pass"))
    common.dump(tmp_path/"summary.json", dict(state="ABORTED", reason="floor_deadline"))
    assert rescore.v2(tmp_path)["state"] == "FAIL"


@pytest.mark.parametrize("kind", ["recall", "pair"])
@pytest.mark.parametrize("state", ["OFF", "ON-catching-up", "ON-covered"])
def test_aborted_child_summary_rejected(tmp_path, kind, state):
    if kind == "pair":
        summary = coverage_pair(tmp_path)
        cell = tmp_path/"upgrade"
        summary["cells"][1]["state"] = "ABORTED"
        summary["state"] = "INVALID"
    else:
        entries = receipts(tmp_path)
        cell = tmp_path/f"0-{state}"
        for entry in entries:
            if entry["state"] == state:
                entry["result"] = dict(state="ABORTED", reason="worker_exit:1")
        summary = dict(state="INVALID", blocks=entries)
    common.dump(cell/"summary.json", dict(state="ABORTED", reason="worker_exit:1"))
    common.dump(tmp_path/"summary.json", summary)
    with pytest.raises(AssertionError):
        rescore.v2(tmp_path)


@pytest.mark.parametrize("shape", ["floor_deadline", "outer_deadline", "partial-recall", "partial-pair"])
def test_honest_aborted_entry_shapes(tmp_path, shape):
    if shape == "partial-recall":
        receipts(tmp_path/"recall")
        # Only cell receipts are needed; no recursive root summary is consulted.
    elif shape == "partial-pair":
        (tmp_path/"empty/catchup").mkdir(parents=True)
    summary = dict(state="ABORTED", reason="outer_deadline" if shape == "outer_deadline" else "floor_deadline")
    common.dump(tmp_path/"summary.json", summary)
    assert rescore.v2(tmp_path) == summary
    proc = subprocess.run([os.sys.executable, "-B", "-m", rescore.__name__, str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0 and json.loads(proc.stdout) == summary


GUARDS = [
    ("offline instrument denies network", dict(violation="network_attempt"), "offline_network_forbidden"),
    ("offline subprocess launch", dict(violation="subprocess_launch"), "offline_subprocess_launch"),
    ("foreground/download preparation forbidden", dict(event="prepare", foreground=True), "foreground_preparation_forbidden"),
    ("foreground backfill forbidden", dict(event="_chunk_backfill_run", foreground=True), "foreground_backfill_forbidden"),
    ("foreground embedding join forbidden", dict(violation="foreground_join"), "foreground_embedding_join_forbidden"),
]


@pytest.mark.parametrize("state", ["OFF", "ON-catching-up", "ON-covered"])
@pytest.mark.parametrize("health", ["timeout", "error"])
@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("aborted", [False, True])
def test_guard_preserves_preceding_failure(tmp_path, state, health, partial, aborted):
    entries = receipts(tmp_path)
    cell = tmp_path/f"0-{state}"
    rows = list(map(json.loads, (cell/"recalls.jsonl").read_text().splitlines()))
    rows[0]["health"] = health
    write(cell/"recalls.jsonl", rows[:1] if partial else rows)
    (cell/"final-audit.json").unlink()  # A guard can interrupt before the final audit.
    request = common.load(cell/"request.json")
    common.dump(cell/"request.json", dict(request, root=str(cell)))
    def denied(_):
        common.append(cell/"events.jsonl", dict(violation="network_attempt"))
        raise RuntimeError("offline instrument denies network")
    measurement.guarded_worker(denied, cell/"request.json")
    result = common.load(cell/"raw-summary.json")
    common.dump(cell/"summary.json", result)
    assert rescore.v2(cell)["state"] == "FAIL"
    for entry in entries:
        if entry["state"] == state:
            entry["result"] = result
    recall.finalize_block(NS(out=tmp_path, seed=1586, standalone_calibration_s=.572), 0, entries)
    assert result["state"] == rescore.v2(cell)["state"] == "FAIL"
    summary = dict(state="ABORTED", reason="floor_deadline") if aborted else dict(state="FAIL", blocks=entries)
    common.dump(tmp_path/"summary.json", summary)
    assert rescore.v2(tmp_path)["state"] == "FAIL"


@pytest.mark.parametrize("state", ["OFF", "ON-catching-up", "ON-covered"])
def test_aborted_root_preserves_failure_without_publications(tmp_path, state):
    entries = receipts(tmp_path)
    cell = tmp_path/f"0-{state}"
    rows = list(map(json.loads, (cell/"recalls.jsonl").read_text().splitlines()))
    rows[0]["health"] = "timeout"
    write(cell/"recalls.jsonl", rows)
    (cell/"publications.jsonl").unlink()  # publish() never creates an empty stream.
    result = recall.score(rows, [], state=state, activity=dict(catchup_activity=False))
    result.update(measurement_valid=True, catchup_activity=False)
    for name in ("raw-summary.json", "summary.json"):
        common.dump(cell/name, result)
    for entry in entries:
        if entry["state"] == state:
            entry["result"] = result
    recall.finalize_block(NS(out=tmp_path, seed=1586, standalone_calibration_s=.572), 0, entries)
    common.dump(tmp_path/"summary.json", dict(state="ABORTED", reason="floor_deadline"))
    assert rescore.v2(tmp_path)["state"] == "FAIL"
    assert rescore.v2(cell)["state"] == "FAIL"


@pytest.mark.parametrize("baseline,expected", [("empty", "ABORTED"), ("upgrade", "FAIL")])
def test_aborted_coverage_rate_entry_scores_directly(tmp_path, baseline, expected):
    source = tmp_path/"source"
    coverage_pair(source)
    root = tmp_path/"rate"
    shutil.copytree(source/baseline, root)
    common.dump(root/"summary.json", dict(state="ABORTED", reason="floor_deadline"))
    assert rescore.v2(root)["state"] == expected


@pytest.mark.parametrize("state", ["OFF", "ON-catching-up", "ON-covered"])
@pytest.mark.parametrize("kinds,expected", [(("timeout", "pass"), "FAIL"), (("pass", "pass"), "ABORTED")])
def test_aborted_recall_entry_scores_directly(tmp_path, state, kinds, expected):
    entries = receipts(tmp_path, kinds=kinds)
    cell = tmp_path/f"0-{state}"
    # receipts() finalizes the raw scores before the entry is interrupted.
    assert rescore.v2(cell)["state"] == next(e["result"]["state"] for e in entries if e["state"] == state)
    common.dump(cell/"summary.json", dict(state="ABORTED", reason="floor_deadline"))
    assert rescore.v2(cell)["state"] == expected


def test_aborted_guarded_recall_entry_scores_directly(tmp_path):
    test_guard_preserves_preceding_failure(tmp_path, "OFF", "timeout", True, True)
    cell = tmp_path/"0-OFF"
    common.dump(cell/"summary.json", dict(state="ABORTED", reason="floor_deadline"))
    assert rescore.v2(cell)["state"] == "FAIL"
    with pytest.raises(AssertionError, match="ABORTED is only valid at the entry root"):
        rescore.v2(tmp_path)


@pytest.mark.parametrize("speed,expected", [(.1, "FAIL"), (10., "ABORTED")])
def test_aborted_pair_scores_coverage_without_cell_summary(tmp_path, speed, expected):
    coverage_pair(tmp_path)
    cell = tmp_path/"upgrade"
    raw = common.load(cell/"catchup/summary.json")
    raw["rows_s"] = speed
    common.dump(cell/"catchup/summary.json", raw)
    # This is the same rate rule used by floor.coverage before its final write.
    assert floor.rate(raw, raw["end_to_end_s"], 250000, events(.572))["state"] == (
        "FAIL" if expected == "FAIL" else "PASS")
    (cell/"summary.json").unlink()
    common.dump(tmp_path/"summary.json", dict(state="ABORTED", reason="floor_deadline"))
    assert rescore.v2(tmp_path)["state"] == expected


@pytest.mark.parametrize("missing", ["manifest", "events", "truncated", "partial-json", "catchup-summary"])
def test_aborted_pair_incomplete_coverage_stays_aborted(tmp_path, missing):
    coverage_pair(tmp_path)
    cell = tmp_path/"upgrade"
    (cell/"summary.json").unlink()
    if missing == "truncated":
        write(cell/"catchup/events.jsonl", events(.572)[:4])
    elif missing == "partial-json":
        (cell/"catchup/events.jsonl").write_text('{"event":')
    else:
        (cell/{"manifest": "manifest.json", "events": "catchup/events.jsonl",
               "catchup-summary": "catchup/summary.json"}[missing]).unlink()
    summary = dict(state="ABORTED", reason="floor_deadline")
    common.dump(tmp_path/"summary.json", summary)
    assert rescore.v2(tmp_path) == summary


@pytest.mark.parametrize("state", ["OFF", "ON-catching-up", "ON-covered"])
@pytest.mark.parametrize("partial", [0, 5])
@pytest.mark.parametrize("message,event,reason", GUARDS)
def test_event_backed_guard_receipts(tmp_path, state, partial, message, event, reason):
    source, target = tmp_path/"source", tmp_path/"target"
    source.mkdir()
    entries = receipts(source)
    target.mkdir()
    cell = target/f"0-{state}"
    for path in source.iterdir():
        if path.name != cell.name:
            if path.is_dir():
                shutil.copytree(path, target/path.name)
            else:
                shutil.copyfile(path, target/path.name)
    cell.mkdir()
    old_cell = source/cell.name
    ev = list(map(json.loads, (old_cell/"events.jsonl").read_text().splitlines()))
    write(cell/"events.jsonl", ev)
    common.dump(cell/"request.json", dict(root=str(cell), seed=1586, state=state, block=0, rows=250000, smoke=False))
    def denied(_):
        if partial:
            rows = list(map(json.loads, (old_cell/"recalls.jsonl").read_text().splitlines()))[:partial]
            write(cell/"recalls.jsonl", rows)
            shutil.copyfile(old_cell/"publications.jsonl", cell/"publications.jsonl")
        common.append(cell/"events.jsonl", event)
        raise RuntimeError(message)
    measurement.guarded_worker(denied, cell/"request.json")
    result = common.load(cell/"raw-summary.json")
    common.dump(cell/"summary.json", result)
    for entry in entries:
        if entry["state"] == state:
            entry["result"] = result
    recall.finalize_block(NS(out=target, seed=1586, standalone_calibration_s=.572), 0, entries)
    common.dump(target/"summary.json", dict(state=common.rollup(e["result"]["state"] for e in entries), blocks=entries))
    assert result == dict(state="INVALID", reason=reason)
    assert rescore.v2(cell) == result
    assert rescore.v2(target)["state"] == "INVALID"
    for changed in (ev, ev+[dict(violation="covered_store_incomplete")]):
        write(cell/"events.jsonl", changed)
        with pytest.raises((AssertionError, FileNotFoundError)):
            rescore.v2(cell)
        with pytest.raises((AssertionError, FileNotFoundError)):
            rescore.v2(target)
