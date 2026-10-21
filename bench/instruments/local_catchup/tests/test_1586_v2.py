"""Declaration-v2 regressions. Synthetic receipts only; no embedding downloads."""
from argparse import Namespace as NS
import json

import pytest

from bench.instruments.local_catchup import floor, measurement, recall, rescore

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


def events(seconds=.572):
    return [dict(event="_embedding_backfill_summary_run", embedded=32, at=i, end=i+seconds) for i in range(5)]


@pytest.mark.parametrize("seconds,normalized", [(.267, True), (.700, False)])
def test_normalized_rate(seconds, normalized):
    result = floor.rate(dict(state="INCOMPLETE", rows_s=2, audit=dict(complete=False)), 7200, 250000, events(seconds))
    assert result["implied_hours"] == 250000/2/3600
    assert result["reference_calibration_s"] == .572
    assert (result["implied_hours_gated"] > result["implied_hours"]) is normalized
    assert result["implied_hours_gated"] == max(result["implied_hours"], result["implied_hours_normalized"])


def test_missing_calibration_is_invalid():
    result = floor.rate(dict(state="INCOMPLETE", rows_s=10, audit=dict(complete=False)), 7200, 250000)
    assert (result["state"], result["reason"]) == ("INVALID", "no_calibration")


def test_only_exact_32_summary_batches_calibrate():
    other = [dict(event="_chunk_backfill_run", embedded=32, at=0, end=999),
             dict(event="_embedding_backfill_summary_run", embedded=31, at=0, end=999),
             dict(event="_embedding_backfill_summary_run", foreground=False)]
    assert floor.calibration(events()+other) == pytest.approx(.572)
    assert floor.calibration(events()[:4]+other) is None


@pytest.fixture
def publications():
    return [dict(start=0, end=10, node_id=i) for i in range(10)]


def rows(count=200, seconds=.1, query_class="canary", **kw):
    return [dict(start=1, end=2, seconds=seconds, query_class=query_class,
                 health="ok", fts_hits=True, fts_coverage="ok", honest_coverage=True,
                 semantic_healthy=True, **kw) for _ in range(count)]


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    # Constant fixtures have exact constant quantiles; keep these tests cheap.
    monkeypatch.setattr(recall, "bootstrap", lambda values, statistic, seed: [statistic(values)])


def test_each_class_needs_its_own_volume(publications):
    result = recall.score(rows(400)+rows(199, query_class="broad_nl"), publications)
    assert result["state"] == "INVALID"
    assert result["classes"]["broad_nl"]["overlapping"] == 199


def test_fast_class_cannot_rescue_slow_class(publications):
    result = recall.score(rows(1000)+rows(seconds=3, query_class="broad_nl"), publications)
    assert result["state"] == "FAIL"
    assert result["classes"]["canary"]["state"] == "PASS"
    assert result["classes"]["broad_nl"]["state"] == "FAIL"


def test_fts_none_fails_catching_up_even_with_hits(publications):
    samples = rows()+rows(query_class="broad_nl")
    samples[-1]["fts_coverage"] = "none"
    result = recall.score(samples, publications, state="ON-catching-up", activity=dict(catchup_activity=True))
    assert result["state"] == "FAIL"
    assert result["classes"]["broad_nl"]["failures"] == 1


def test_healthy_reference_basis_passes(publications):
    result = recall.score(rows()+rows(query_class="broad_nl"), publications, calibration_s=.1)
    assert result["reference_calibration_s"] == .572
    assert result["host_check"] == result["state"] == "PASS"


def test_host_scaling_preserves_raw_statistics(publications):
    result = recall.score(rows(seconds=1.5)+rows(seconds=1.5, query_class="broad_nl"), publications,
                          calibration_s=.25, reference_calibration_s=.5)
    assert result["state"] == "FAIL"
    for part in result["classes"].values():
        assert part["statistics"]["p50"]["seconds"] == 1.5
        assert part["normalized_statistics"]["p50"]["seconds"] == 3


@pytest.mark.parametrize("complete", [True, False])
def test_covered_store_first_check_only(tmp_path, monkeypatch, complete):
    checks = []
    monkeypatch.setattr(measurement, "audit", lambda *args: checks.append(1) or dict(complete=complete))
    def forbidden(*args):
        pytest.fail("covered recall tried to backfill")
    a = NS(db=tmp_path/"db", root=tmp_path)
    if complete:
        measurement.wait_covered(a, forbidden, None, {})
    else:
        with pytest.raises(RuntimeError, match="^covered_store_incomplete$"):
            measurement.wait_covered(a, forbidden, None, {})
    assert checks == [1]


def test_diagnostics_wrap_only_the_reader():
    reader, other = object(), object()
    tools = NS(_lcm_recall_fts_arm=lambda *a, **kw: ([{}, {}], None))
    observed, original = recall.diagnostics(tools, reader)
    tools._lcm_recall_fts_arm(other, deadline=100)
    assert observed == {}
    tools._lcm_recall_fts_arm(reader, deadline=recall.time.monotonic()+2)
    assert observed["fts_hit_count"] == 2
    assert observed["fts_seconds"] >= 0
    assert 0 < observed["fts_budget_s"] <= 2
    assert "fts_cap_reason" in observed
    tools._lcm_recall_fts_arm = original


def test_query_classes_alternate_and_canaries_keep_order(tmp_path, monkeypatch):
    a = NS(out=tmp_path, queries=4, seed=1586, blocks=3, block=1, build_covered=False,
           standalone_calibration_s=.5, smoke=False, label="floor surrogate", shape="short", baseline="empty")
    monkeypatch.setattr(recall, "prepare", lambda _: dict(canaries=[{}, {}, {}]))
    calls = []
    monkeypatch.setattr(recall, "cell", lambda a, s, b, kind: calls.append((s, b)) or dict(state="UNCALIBRATED"))
    recall.main(a)
    pinned = json.loads((tmp_path/"queries.json").read_text())
    assert [q["query_class"] for q in pinned] == ["canary", "broad_nl"]*4
    assert [q["id"] for q in pinned[::2]] == [f"q{i:04d}" for i in range(4)]
    assert [q["text"] for q in pinned[1::2]] == list(recall.BROAD_NL)+[recall.BROAD_NL[0]]
    assert recall.BROAD_NL == ("What turn was acknowledged", "Recall the filler from a turn", "Which turn contains a value")
    assert calls == [("OFF", 1), ("ON-catching-up", 1), ("ON-covered", 1)]


@pytest.mark.parametrize("shape,order", [("short", ["empty", "upgrade"]), ("mixed", ["upgrade", "empty"])])
def test_pair_order_and_separate_evidence(tmp_path, monkeypatch, shape, order):
    a = NS(out=tmp_path, scratch=tmp_path/"scratch", cgroup=tmp_path, cell="coverage-pair", shape=shape, baseline="empty")
    monkeypatch.setattr(floor, "measured_limits", lambda _: {})
    monkeypatch.setattr(floor.subprocess, "check_output", lambda *a, **kw: "CPU fixture\n")
    monkeypatch.setattr(floor, "standalone", lambda _: {})
    def measure(a):
        floor.dump(a.out/"summary.json", dict(state="FAIL" if a.baseline == "upgrade" else "PASS", baseline=a.baseline))
    monkeypatch.setattr(floor, "coverage", measure)
    floor.limited(a)
    result = json.loads((tmp_path/"summary.json").read_text())
    assert [r["baseline"] for r in result["cells"]] == order
    assert result["state"] == "FAIL"
    assert all((tmp_path/b/"summary.json").exists() for b in order)


def test_rescore_recomputes_and_checks_coverage(tmp_path):
    (tmp_path/"catchup").mkdir()
    summary = dict(state="INCOMPLETE", rows_s=2, audit=dict(complete=False), end_to_end_s=7200)
    floor.dump(tmp_path/"catchup/summary.json", summary)
    floor.dump(tmp_path/"manifest.json", dict(rows=250000))
    (tmp_path/"catchup/events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events()))
    floor.dump(tmp_path/"summary.json", floor.rate(summary, 7200, 250000, events()))
    assert rescore.v2(tmp_path)["state"] == "PASS"
    recorded = json.loads((tmp_path/"summary.json").read_text())
    recorded["implied_hours_gated"] = 0
    floor.dump(tmp_path/"summary.json", recorded)
    with pytest.raises(AssertionError):
        rescore.v2(tmp_path)


def test_standalone_five_fresh_32_leaf_batches(corpus, tmp_path, monkeypatch):
    from bench.instruments.local_catchup import calibration
    _, _, mod = corpus
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []
    def backfill(cfg, db, **kw):
        import sqlite3
        with sqlite3.connect(db) as c:
            assert c.execute("SELECT count(*) FROM summary_nodes WHERE depth=0").fetchone()[0] == 32
            assert c.execute("SELECT count(*) FROM lcm_embedding_vectors").fetchone()[0] == 0
        calls.append((db.name, kw))
        return dict(embedded=32)
    monkeypatch.setattr(mod("command"), "_embedding_backfill_summary_run", backfill)
    monkeypatch.setattr(mod("embedding_provider"), "resolve_provider", lambda *a, **kw: NS(_ensure_local=lambda **kw: None))
    a = NS(scratch=tmp_path/"cal", out=tmp_path, model_cache=tmp_path, lcmx_repo=tmp_path, hermes_source=tmp_path)
    monkeypatch.setattr(calibration, "product", lambda *a: mod)
    value = calibration.standalone(a)
    assert len(set(n for n, _ in calls)) == len(value["core_seconds"]) == 5
    assert all(k == dict(apply=True, limit=32, retry_uncertain=False, include_next_hint=False) for _, k in calls)
    assert value["standalone_calibration_s"] >= 0


@pytest.mark.parametrize("initialize", [True, False])
def test_measured_covered_worker_never_backfills_and_rows_have_diagnostics(tmp_path, monkeypatch, initialize):
    from bench.instruments.local_catchup import common
    a = NS(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path, state="ON-covered",
           smoke=True, rows=8, queries=1, seed=1586, block=0)
    common.dump(tmp_path/"manifest.json", dict(canaries=[dict(value="synthetic")], expected=dict(chunks={}, leaves={})))
    common.dump(tmp_path/"queries.json", [dict(id="q", canary=0, query_class="canary"),
        dict(id="b", text=recall.BROAD_NL[0], query_class="broad_nl")])
    closed = []
    store = NS(close=lambda: closed.append("reader"))
    engine = NS(_config=NS(recall_query_timeout_s=1), _dag=NS(add_node=lambda *a, **kw: 1), shutdown=lambda: closed.append("engine"))
    def forbidden(*a, **kw):
        pytest.fail("measured covered state called coverage construction")
    tools = NS(_lcm_recall_fts_arm=lambda *a, **kw: ([dict(arms=["fts"])], None))
    def call(*args, engine):
        hits, _ = tools._lcm_recall_fts_arm(engine, deadline=recall.time.monotonic()+2)
        return json.dumps(dict(hits=hits, provenance=dict(coverage=dict(fts="ok", chunk="full", summary="full"),
            arms_run=["fts", "chunk", "summary"], fts_budget_s=1.23)))
    tools.lcm_recall = call
    if not initialize:
        del tools._lcm_recall_fts_arm  # initialization failure must still close readers without masking the error
    monkeypatch.setattr(measurement, "fill_covered", forbidden)
    monkeypatch.setattr(recall, "runtime", lambda _: (a, lambda n: tools if n == "tools" else NS(
        _chunk_backfill_run=forbidden, _embedding_backfill_summary_run=forbidden), engine, 0))
    monkeypatch.setattr(recall, "recall_reader", lambda *a: NS(_dag=store, _store=store))
    monkeypatch.setattr(measurement, "audit", lambda *a: dict(complete=True))
    monkeypatch.setattr(recall, "audit", lambda *a: dict(complete=True))
    monkeypatch.setattr(recall, "vector_counts", lambda *a: dict(chunks=0, leaves=0))
    for name in ("trace", "integrity"):
        monkeypatch.setattr(recall, name, lambda *a: None)
    monkeypatch.setattr(recall, "safety", lambda *a: {})
    event = recall.threading.Event
    def stopped():
        value = event()
        value.set()
        return value
    class Thread:
        def __init__(self, target, **kw): self.target = target
        def start(self): self.target()
        def join(self, *a): pass
        def is_alive(self): return False
    monkeypatch.setattr(recall.threading, "Event", stopped)
    monkeypatch.setattr(recall.threading, "Thread", Thread)
    if not initialize:
        with pytest.raises(RuntimeError, match="recall worker failure.*AttributeError"):
            recall.worker(None)
    else:
        recall.worker(None)
        saved = [json.loads(s) for s in (tmp_path/"recalls.jsonl").read_text().splitlines()]
        assert [r["query_class"] for r in saved] == ["canary", "broad_nl"]
        assert all(set(("query_class", "fts_seconds", "fts_budget_s", "fts_hit_count", "fts_coverage", "fts_cap_reason")) <= r.keys() for r in saved)
        assert all(r["fts_budget_s"] == 1.23 and r["fts_hit_count"] == 1 and r["fts_coverage"] == "ok" for r in saved)
    assert closed == ["reader", "reader", "engine"]


@pytest.mark.parametrize("valid", [True, False])
def test_rescore_saved_recall_classes(tmp_path, publications, valid):
    samples = [dict(r, coverage_start={}, coverage_end={}) for r in rows()+rows(query_class="broad_nl")]
    recorded = recall.score(samples, publications, calibration_s=floor.calibration(events(.1)))
    recorded["measurement_valid"] = valid
    recorded["catchup_activity"] = False
    if not valid:
        recorded["state"] = "INVALID"
    for name, value in (("recalls", samples), ("publications", publications)):
        (tmp_path/f"{name}.jsonl").write_text("".join(json.dumps(v)+"\n" for v in value))
    floor.dump(tmp_path/"raw-summary.json", recorded)
    floor.dump(tmp_path/"request.json", dict(seed=1586, state="OFF", block=0, standalone_calibration_s=.1, rows=250000, smoke=not valid))
    floor.dump(tmp_path/"final-audit.json", dict(classes={}))
    floor.dump(tmp_path.parent/"queries.json", [{}]*400)
    (tmp_path/"events.jsonl").touch()
    catchup = tmp_path.parent/"0-ON-catching-up"
    catchup.mkdir(exist_ok=True)
    (catchup/"events.jsonl").write_text("".join(json.dumps(v)+"\n" for v in
        [dict(event="calibration_window_start", at=-1), *events(.1), dict(event="calibration_window_end", at=10)]))
    assert rescore.v2(tmp_path)["state"] == recorded["state"]
    recorded["classes"]["broad_nl"]["overlapping"] = 999
    floor.dump(tmp_path/"raw-summary.json", recorded)
    with pytest.raises(AssertionError):
        rescore.v2(tmp_path)


def test_setup_resume_preserves_partial_store(corpus, tmp_path, monkeypatch):
    from contextlib import closing
    import sqlite3
    from bench.instruments.local_catchup import common
    db, manifest, _ = corpus
    source = tmp_path/"source"
    source.mkdir()
    common.dump(source/"manifest.json", manifest)
    with closing(sqlite3.connect(db)) as src, closing(sqlite3.connect(source/"store.db")) as dst:
        src.backup(dst)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(measurement, "immutable_cache", lambda *a: None)
    monkeypatch.setattr(measurement, "build", lambda *a: pytest.fail("resume rebuilt corpus"))
    a = NS(scratch=tmp_path/"scratch", out=tmp_path/"evidence", model_cache=tmp_path/"cache",
        rows=8, shape="short", baseline="empty", seed=1586, blocks=1, samples=1, queries=1, resume_covered_store=source)
    assert measurement.prepare(a) == json.loads(json.dumps(manifest))
    from bench.instruments.local_catchup.audit import audit
    assert audit(a.scratch/"baseline.db", manifest)["raw_unchanged"]


def test_covered_runtime_retires_all_measured_maintenance(corpus, tmp_path, monkeypatch):
    from bench.instruments.local_catchup import common
    db, _, mod = corpus
    warmed = []
    mod("engine")
    mod("tools")  # Bind real provider imports before installing the local warm-up stub.
    monkeypatch.setattr(measurement, "product", lambda *a: mod)
    monkeypatch.setattr(measurement, "guards", lambda *a: None)
    monkeypatch.setattr(measurement.threading.Thread, "join", measurement.threading.Thread.join)
    monkeypatch.setattr(mod("embedding_provider"), "resolve_provider", lambda *a, **kw: NS(_ensure_local=lambda **kw: warmed.append(kw)))
    scheduled = []
    monkeypatch.setattr(mod("engine").LCMEngine, "_schedule_embedding_maintenance", lambda self: scheduled.append(1))
    def forbidden(*a, **kw): pytest.fail("ON-covered scheduled a backfill")
    for name in ("_chunk_backfill_run", "_embedding_backfill_summary_run"):
        monkeypatch.setattr(mod("command"), name, forbidden)
    request = dict(state="ON-covered", **{k: str(tmp_path) for k in ("root", "scratch", "out", "lcmx_repo", "hermes_source", "model_cache")}, db=str(db), home=str(tmp_path/"runtime-home"))
    common.dump(tmp_path/"request.json", request)
    _, _, engine, _ = measurement.runtime(tmp_path/"request.json")
    try:
        assert engine._embedding_maintenance_retired is True
        engine._embedding_maintenance_retired = False
        engine._schedule_embedding_maintenance()
        assert scheduled == []
        assert warmed == [dict(allow_download=False)]
    finally:
        engine.shutdown()


def test_artifact_chain_and_confinement_are_explicit():
    from pathlib import Path
    import yaml
    root = Path(floor.__file__).resolve().parents[3]
    workflow = yaml.safe_load((root/".github/workflows/lcmx-1586-floor.yml").read_text())
    jobs = workflow["jobs"]
    chain = ["covered-store-1", "covered-store-2", "covered-store-3", "covered-store", "floor"]
    for a, b in zip(chain, chain[1:]):
        needs = jobs[b]["needs"]
        assert ([needs] if isinstance(needs, str) else needs) == (["validate", a] if b == "floor" else [a])
    for name, stage, previous in zip(chain[:-1], ("1", "2", "3", "store"), ("", "1", "2", "3")):
        assert jobs[name]["uses"] == "./.github/workflows/lcmx-1586-covered-store.yml"
        assert jobs[name]["with"]["stage"] == stage
        assert jobs[name]["with"].get("previous_stage", "") == previous
    helper = yaml.safe_load((root/".github/workflows/lcmx-1586-covered-store.yml").read_text())
    build = helper["jobs"]["build"]
    steps = build["steps"]
    assert build["timeout-minutes"] == 350
    checkouts = [s for s in steps if s.get("uses", "").startswith("actions/checkout@")]
    assert len(checkouts) == 3
    assert checkouts[-1]["with"]["ref"] == "71d34bad1045610252eda02ee79549436896e784"
    calibrated = next(i for i, s in enumerate(steps) if "Calibrate inside" in s.get("name", ""))
    built = next(i for i, s in enumerate(steps) if "Build and audit" in s.get("name", ""))
    assert calibrated < built
    assert "cgroup.procs" in steps[calibrated]["run"] and "cgroup.procs" not in steps[built]["run"]
    assert "--resume-covered-store" in steps[built]["run"]
    assert '"$STAGE" != "store"' in steps[built]["run"]
