"""Round-two regressions use raw receipts and the production worker/launcher."""
from argparse import Namespace as NS
import json
import os
import random
import sqlite3
import sys
import threading

import pytest
import yaml

from bench.instruments.local_catchup import common, floor, measurement, recall, rescore
from bench.instruments.local_catchup.tests.test_1586_v2 import events, rows
from bench.instruments.local_catchup.tests.test_1586_fix1 import cover
from bench.instruments.local_catchup.tests.test_corpus_audit import REPO

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


def window_batches(seconds=.572, count=5):
    return ([dict(event="calibration_window_start", at=0)] + events(seconds)[:count]
            + [dict(event="calibration_window_end", at=10)])


@pytest.fixture(autouse=True)
def quick_bootstrap(monkeypatch):
    monkeypatch.setattr(recall, "bootstrap", lambda v, f, seed: [f(v)])


def block(tmp_path, monkeypatch, *, count=5, valid=True, slow=True):
    a = NS(out=tmp_path, seed=1586, standalone_calibration_s=99)
    pubs = [dict(start=20, end=30, node_id=i) for i in range(10)]
    data = rows(query_class="canary") + rows(160, query_class="broad_nl") + rows(
        40, seconds=4.5 if slow else .1, query_class="broad_nl")
    if not valid and slow:
        data = data[1:]  # A short canary class makes the worker's pre-scaling rollup INVALID.
    data = [dict(r, start=21, end=22, incomplete=True, coverage_start=dict(chunks=0), coverage_end=dict(chunks=1)) for r in data]
    results = []
    for state in ("OFF", "ON-catching-up", "ON-covered"):
        root = tmp_path/f"0-{state}"
        root.mkdir()
        loaded = [dict(e, at=e["at"]+20, end=e["end"]+20) for e in events(.45)]
        raw_events = window_batches(.572/2.14, count)+loaded + ([] if valid else [dict(event="volume_cap")])
        for name, values in (("events", raw_events), ("recalls", data), ("publications", pubs)):
            (root/f"{name}.jsonl").write_text("".join(json.dumps(v)+"\n" for v in values))
        raw = recall.score(data, pubs, state=state, activity=dict(catchup_activity=True))
        raw.update(measurement_valid=valid, catchup_activity=True)
        if not valid:
            raw.update(state="INVALID", reason="volume_cap")
        for name in ("raw-summary.json", "summary.json"):
            common.dump(root/name, raw)
        common.dump(root/"request.json", dict(seed=1586, state=state, block=0, rows=250000, smoke=False))
        common.dump(root/"final-audit.json", dict(classes=dict(chunks=dict(missing_count=1))))
        common.dump(tmp_path/"queries.json", [{}]*400)
        results.append(dict(state=state, block=0, result=raw))
    recall.finalize_block(a, 0, results)
    common.dump(tmp_path/"summary.json", dict(state=common.rollup(r["result"]["state"] for r in results), blocks=results))
    return results


def test_unloaded_window_controls_block_and_rescore(tmp_path, monkeypatch):
    results = block(tmp_path, monkeypatch)
    for entry in results:
        result = entry["result"]
        assert result["latency_scale"] == pytest.approx(2.14)
        assert result["classes"]["broad_nl"]["statistics"]["p90"]["seconds"] == 4.5
        assert result["classes"]["broad_nl"]["normalized_statistics"]["p90"]["seconds"] == pytest.approx(9.63)
        assert result["state"] == "FAIL"
    assert rescore.v2(tmp_path)["state"] == "FAIL"


def test_short_calibration_window(tmp_path, monkeypatch):
    results = block(tmp_path, monkeypatch, count=4, slow=False)
    assert all(r["result"]["state"] == "UNCALIBRATED" and
               r["result"]["reason"] == "calibration_window_short" for r in results)
    assert rescore.v2(tmp_path)["state"] == "UNCALIBRATED"


@pytest.mark.parametrize("count", [4, 5])
def test_calibration_window_records_bounds_and_spends_state_budget(tmp_path, monkeypatch, count):
    clock = [100.]
    monkeypatch.setattr(recall.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(recall, "safety", lambda *a: {})
    path = tmp_path/"events.jsonl"
    def sleep(_):
        for e in events(.267)[:count]:
            common.append(path, dict(e, at=e["at"]+100, end=e["end"]+100))
        clock[0] += 10 if count == 5 else 600
    monkeypatch.setattr(recall.time, "sleep", sleep)
    a = NS(root=tmp_path, volume_seconds=5700)
    recall.calibration_window(a)
    raw = [json.loads(s) for s in path.read_text().splitlines()]
    assert raw[0] == dict(event="calibration_window_start", at=100.)
    assert raw[-1] == dict(event="calibration_window_end", at=clock[0])
    assert a.volume_seconds == 5700-(clock[0]-100)
    assert recall.window_calibration(raw) == (pytest.approx(.267) if count == 5 else None)


@pytest.mark.parametrize("field,value", [("calibration_s", .572), ("reference_calibration_s", .3)])
def test_rescore_rejects_summary_calibration_tampering(tmp_path, monkeypatch, field, value):
    block(tmp_path, monkeypatch, slow=False)
    root = tmp_path/"0-OFF"
    raw = common.load(root/"raw-summary.json")
    params = dict(calibration_s=raw["calibration_s"], reference_calibration_s=raw["reference_calibration_s"])
    params[field] = value
    samples = [json.loads(s) for s in (root/"recalls.jsonl").read_text().splitlines()]
    pubs = [json.loads(s) for s in (root/"publications.jsonl").read_text().splitlines()]
    raw.update(recall.score(samples, pubs, **params))
    common.dump(root/"raw-summary.json", raw)
    assert rescore.v2(root)["rescore_disagreement"]


@pytest.mark.parametrize("slow,expected", [(False, "INVALID"), (True, "FAIL")])
def test_finalization_validity_and_fail_precedence(tmp_path, monkeypatch, slow, expected):
    result = block(tmp_path, monkeypatch, valid=False, slow=slow)
    assert all(r["result"]["state"] == expected for r in result)
    assert rescore.v2(tmp_path)["state"] == expected


def test_prepare_isolates_mac_setup_environment(corpus, tmp_path, monkeypatch):
    _, manifest, _ = corpus
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("CFFIXED_USER_HOME", raising=False)
    monkeypatch.setattr(measurement, "immutable_cache", lambda *a: None)
    observed = []
    def build(*a):
        observed.append((os.environ.get("ORT_DISABLE_TELEMETRY"), os.environ.get("CFFIXED_USER_HOME"), os.environ["HOME"]))
        return dict(manifest, database_bytes=1)
    monkeypatch.setattr(measurement, "build", build)
    a = NS(scratch=tmp_path/"scratch", out=tmp_path/"out", model_cache=tmp_path/"cache",
           rows=8, blocks=1, samples=1, queries=1, lcmx_repo=REPO, hermes_source=REPO, shape="short", baseline="empty")
    measurement.prepare(a)
    assert observed == [("1", str(a.scratch/"builder-home"), str(a.scratch/"builder-home"))]*2


def test_fill_failure_keeps_exception_message(tmp_path, monkeypatch):
    def fail(*a):
        raise RuntimeError("cross-profile access")
    monkeypatch.setattr(measurement, "audit", fail)
    with pytest.raises(RuntimeError, match="cross-profile access"):
        measurement.fill_covered(NS(db=tmp_path/"db", max_seconds=1), None, None, {})


def test_setup_diagnostic_allowlist():
    workflow = yaml.safe_load((REPO/".github/workflows/lcmx-1586-covered-store.yml").read_text())
    paths = next(s["with"]["path"] for s in workflow["jobs"]["build"]["steps"]
                 if "setup-evidence" in s.get("with", {}).get("name", "")).splitlines()
    assert {"evidence/covered/progress.jsonl", "evidence/covered/file-access.jsonl"} <= set(paths)
    assert not any(p.endswith("store.db") or p == "evidence/covered/" for p in paths)


@pytest.mark.parametrize("state", ["OFF", "ON-catching-up", "ON-covered"])
def test_guarded_worker_receipt_rescores(tmp_path, state):
    tmp_path = tmp_path/f"0-{state}"
    tmp_path.mkdir()
    request = tmp_path/"request.json"
    common.dump(request, dict(root=str(tmp_path)))
    def denied(*a):
        common.append(tmp_path/"events.jsonl", dict(violation="subprocess_launch"))
        raise RuntimeError("offline subprocess launch")
    measurement.guarded_worker(denied, request)
    assert rescore.v2(tmp_path) == dict(state="INVALID", reason="offline_subprocess_launch")


@pytest.mark.parametrize("problem", ["unfinished", "tampered", "missing-directory", "missing-db", "missing-table", "omitted"])
def test_cell_pre_audit_invalid_receipt(corpus, tmp_path, monkeypatch, problem):
    db, manifest, mod = corpus
    source = tmp_path/"covered"
    source.mkdir()
    if problem in ("unfinished", "tampered"):
        with sqlite3.connect(db) as src, sqlite3.connect(source/"store.db") as dst:
            src.backup(dst)
        if problem == "tampered":
            cover(source/"store.db", manifest, mod)
            measurement.audit(source/"store.db", manifest)
            common.dump(source/"manifest.json", manifest)  # Persist setup digests, not recall memory.
            with sqlite3.connect(source/"store.db") as c:
                c.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'")
        if problem == "unfinished":
            common.dump(source/"manifest.json", manifest)
    if problem == "missing-directory":
        source = tmp_path/"nonexistent"
    if problem == "missing-table":
        sqlite3.connect(source/"store.db").close()
    a = NS(out=tmp_path/"out", scratch=tmp_path, covered_store=None if problem == "omitted" else source)
    a.out.mkdir()
    common.dump(a.out/"manifest.json", manifest)
    result = measurement.cell(a, "ON-covered", 0, "recall")
    assert result["state"] == "INVALID" and result["reason"] == "covered_store_incomplete"
    assert rescore.v2(a.out/"0-ON-covered") == result
    if problem in ("unfinished", "tampered"):
        assert (result["pre_audit"]["missing_count"] > 0) is (problem == "unfinished")
        assert result["pre_audit"]["changed_vectors"] == (1 if problem == "tampered" else 0)
        assert result["pre_audit"]["reason"] == ("covered_store_changed" if problem == "tampered" else "covered_store_unfinished")
    else:
        assert result["pre_audit"]["reason"]


def test_recall_dispatch_omits_absent_store(tmp_path, monkeypatch):
    a = NS(out=tmp_path, scratch=tmp_path, cgroup=tmp_path, cell="recall", lcmx_repo=REPO,
           hermes_source=REPO, rows=250000, model_cache=tmp_path, shape="short", baseline="empty",
           block=0, covered_store=None, max_minutes=330)
    monkeypatch.setattr(floor, "measured_limits", lambda _: {})
    monkeypatch.setattr(floor.subprocess, "check_output", lambda *a, **kw: "fixture")
    monkeypatch.setattr(floor, "standalone", lambda _: dict(standalone_calibration_s=.572))
    def run(argv, **kw):
        assert "--covered-store" not in argv and "None" not in argv
        # Exercise the actual child cell guard on the omitted-store path.
        out = tmp_path/"recall"
        out.mkdir()
        common.dump(out/"manifest.json", {})
        result = measurement.cell(NS(out=out, scratch=tmp_path, covered_store=None), "ON-covered", 0, "recall")
        common.dump(out/"summary.json", result)
    monkeypatch.setattr(floor.subprocess, "run", run)
    floor.limited(a)
    assert common.load(tmp_path/"summary.json")["state"] == "INVALID"


def test_recall_dispatch_rejects_short_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["floor", "--lcmx-repo", str(REPO), "--hermes-source", str(REPO),
        "--scratch", str(tmp_path), "--out", str(tmp_path), "--model-cache", str(tmp_path),
        "--cgroup", str(tmp_path), "--cell", "recall", "--max-minutes", "329"])
    monkeypatch.setattr(floor.subprocess, "Popen", lambda *a, **kw: pytest.fail("short dispatch launched"))
    with pytest.raises(ValueError, match="recall requires at least 330 minutes"):
        floor.main()


def test_absent_covered_store_does_not_skip_other_states(tmp_path, monkeypatch):
    a = NS(out=tmp_path, scratch=tmp_path, queries=200, seed=1586, blocks=1, block=0,
        covered_store=tmp_path/"nonexistent", standalone_calibration_s=.572,
        smoke=False, label="fixture", shape="short", baseline="empty")
    monkeypatch.setattr(recall, "prepare", lambda _: dict(canaries=[dict(value="fixture")]))
    common.dump(tmp_path/"manifest.json", {})
    calls = []
    def cell(a, state, b, kind):
        calls.append(state)
        if state == "ON-covered":
            return measurement.cell(a, state, b, kind)
        (tmp_path/f"{b}-{state}").mkdir()
        return dict(state="INVALID", reason="fixture")
    monkeypatch.setattr(recall, "cell", cell)
    recall.main(a)
    assert calls == ["OFF", "ON-catching-up", "ON-covered"]
    result = common.load(tmp_path/"summary.json")
    assert result["state"] == "INVALID" and len(result["blocks"]) == 3
    assert result["blocks"][-1]["result"]["reason"] == "covered_store_incomplete"


def test_incremental_overlap_matches_full_recount():
    rng = random.Random(1586)
    data, pubs = [], []
    counter = recall.OverlapCounter("ON-catching-up")
    for i in range(300):
        start = i*.19
        if rng.random() < .4:
            # Completion can be observed a recall later than the publication start.
            pubs.append(dict(start=start-.2, end=start-.05, node_id=i//2))
        row = dict(rows(1, query_class="canary" if i % 2 else "broad_nl")[0],
                   start=start, end=start+.19, semantic_empty=rng.random() < .1)
        data.append(row)
        counter.add(row, pubs)
        for cls in ("canary", "broad_nl"):
            full = [r for r in recall.overlapping(data, pubs) if r["query_class"] == cls and not r["semantic_empty"]]
            assert len(counter.counted[cls]) == len(full)
            assert len(counter.publications[cls]) == recall.overlapping_publications(full, pubs)


@pytest.mark.parametrize("state,cap", [("OFF", False), ("OFF", True), ("ON-catching-up", False)])
def test_worker_volume_signal_stops_real_trace(tmp_path, monkeypatch, state, cap):
    """The reader itself signals enough; no test-side enough.set or cap mapping."""
    manifest = dict(canaries=[dict(value="fixture")], expected=dict(chunks={"1": 1}, leaves={"1": 1}))
    common.dump(tmp_path/"manifest.json", manifest)
    common.dump(tmp_path/"queries.json", [dict(id=f"{c}{i}", canary=0, text="fixture", query_class=c)
        for i in range(200) for c in ("canary", "broad_nl")])
    a = NS(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path, state=state,
           smoke=False, rows=250000, seed=1586, block=0, volume_seconds=.03 if cap else 8)
    calls, enough_seen = [], []
    tools = NS(_lcm_recall_fts_arm=lambda *a, **kw: ([], None))
    def query(*args, **kw):
        calls.append(1)
        # Yield briefly so the real publication trace can commit between calls.
        threading.Event().wait(.001)
        return json.dumps(dict(hits=[dict(arms=["fts"])], provenance=dict(
            arms_run=["chunk", "summary"], coverage=dict(fts="full", chunk="full", summary="full"))))
    tools.lcm_recall = query
    dag = NS()
    def publish(node, **kw):
        threading.Event().wait(.003)
        return len(calls)
    dag.add_node = publish
    engine = NS(_dag=dag, _config=NS(recall_query_timeout_s=1), shutdown=lambda: None,
        current_session_id="fixture", _conversation_id="fixture", ingest=lambda v: None,
        should_compress_preflight=lambda v: True,
        note_turn_complete=lambda: None, _prepare_leaves_after_turn=lambda v: None)
    # Match compress's host contract, preserving the current trace messages.
    engine.compress = lambda view, **kw: dag.add_node(NS(depth=0), before_commit=True) and view
    reader = NS(_dag=NS(close=lambda: None), _store=NS(close=lambda: None))
    def mod(name):
        return tools if name == "tools" else NS(count_messages_tokens=lambda v: 0)
    monkeypatch.setattr(recall, "runtime", lambda _: (a, mod, engine, 0))
    def open_reader(*args):
        if state == "ON-catching-up":
            raw = [json.loads(s) for s in (tmp_path/"events.jsonl").read_text().splitlines()]
            assert raw[-1]["event"] == "calibration_window_end"
            assert recall.window_calibration(raw) is not None
        return reader
    monkeypatch.setattr(recall, "recall_reader", open_reader)
    def calibrate(a):
        assert not calls and not (tmp_path/"turns.jsonl").exists()
        clock = [100.]
        def sleep(_):
            for i in range(5):
                common.append(tmp_path/"events.jsonl", dict(event="_embedding_backfill_summary_run",
                    embedded=32, at=100+i*.001, end=100+i*.001+.0001))
            clock[0] += .01
        with monkeypatch.context() as patch:
            patch.setattr(recall.time, "perf_counter", lambda: clock[0])
            patch.setattr(recall.time, "sleep", sleep)
            recall.calibration_window(a)
    # Keep the real window helper reachable while recording its ordering in worker().
    original_window = getattr(recall, "calibration_window", None)
    def window(a):
        with monkeypatch.context() as patch:
            patch.setattr(recall, "calibration_window", original_window)
            calibrate(a)
    monkeypatch.setattr(recall, "calibration_window", window, raising=False)
    monkeypatch.setattr(recall, "vector_counts", lambda *a: dict(chunks=0, leaves=0))
    monkeypatch.setattr(recall, "catchup_activity", lambda *a: dict(catchup_activity=True))
    monkeypatch.setattr(recall, "audit", lambda *a: dict(classes=dict(chunks=dict(missing_count=1), leaves=dict(missing_count=1))))
    monkeypatch.setattr(recall, "integrity", lambda *a: None)
    monkeypatch.setattr(recall, "safety", lambda *a: {})
    original = recall.publication_trace
    def trace(*args):
        result = original(*args)
        enough_seen.append(args[3].is_set())
        return result
    monkeypatch.setattr(recall, "publication_trace", trace)
    recall.worker(None)
    saved = common.load(tmp_path/"raw-summary.json")
    assert enough_seen == [not cap]
    if cap:
        assert saved["state"] == "INVALID" and saved["reason"] == "volume_cap"
    else:
        assert saved["measurement_valid"] is True
        assert all(c["overlapping"] >= 200 and c["overlapping_publications"] >= 10 for c in saved["classes"].values())
