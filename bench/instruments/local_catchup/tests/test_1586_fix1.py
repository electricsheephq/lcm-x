"""Fix-round-1 receipts: deterministic stores, block calibration and volume caps."""
from argparse import Namespace as NS
from datetime import datetime, timezone
import json
import sqlite3
import threading
from unittest.mock import patch

import pytest
import yaml

from bench.instruments.local_catchup import audit, calibration, common, corpus as builder, floor, measurement, recall, rescore
from bench.instruments.local_catchup.tests.test_1586_v2 import events, rows
from bench.instruments.local_catchup.tests.test_corpus_audit import REPO

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


@pytest.fixture(autouse=True)
def quick_bootstrap(monkeypatch):
    monkeypatch.setattr(recall, "bootstrap", lambda values, statistic, seed: [statistic(values)])


@pytest.mark.parametrize("baseline", ["empty", "upgrade"])
def test_manifests_identical_across_job_clocks(corpus, tmp_path, baseline):
    manifests = []
    for i, clock in enumerate((1800000000, 1900000000)):
        with patch("time.time", return_value=clock), patch.object(corpus[2]("vector_store"), "datetime") as dt:
            dt.now.return_value = datetime.fromtimestamp(clock, timezone.utc)
            manifest = builder.build(REPO, REPO, tmp_path/f"job{i}.db", 8, baseline=baseline)
        common.dump(tmp_path/f"manifest{i}.json", manifest)
        manifests.append((tmp_path/f"manifest{i}.json").read_bytes())
    assert manifests[0] == manifests[1]


def cover(db, manifest, mod):
    """Write only missing vectors; preserve the upgrade's existing metadata."""
    vs = mod("vector_store").VectorStore(db)
    for kind, task in (("chunks", "chunk"), ("leaves", "summary")):
        identity = mod("vector_store").EmbeddingIdentity.canonical("fastembed", common.MODEL, "", 384, "float32", "little", task)
        for ident, meta in manifest["expected_metadata"][kind].items():
            if f"{kind}/{manifest['identities'][task]}/{ident}" in manifest["preimage"]["vectors"]:
                continue
            if kind == "chunks":
                vs.record_chunk_embedding(ident, common.MODEL, [1.]+[0.]*383, identity=identity,
                    **dict(zip(("store_id", "chunk_index", "char_start", "char_end", "token_estimate"), meta)))
            else:
                vs.record_embedding(ident, "summary", common.MODEL, [1.]+[0.]*383, identity=identity)
    vs.close()


def test_independent_recall_manifest_accepts_covered_upgrade(corpus, tmp_path):
    _, _, mod = corpus
    with patch("time.time", return_value=1800000000):
        manifest = builder.build(REPO, REPO, tmp_path/"covered.db", 8, baseline="upgrade")
    cover(tmp_path/"covered.db", manifest, mod)
    with patch("time.time", return_value=1900000000):
        independent = builder.build(REPO, REPO, tmp_path/"recall.db", 8, baseline="upgrade")
    assert audit.audit(tmp_path/"covered.db", independent)["complete"]
    with sqlite3.connect(tmp_path/"covered.db") as c:
        c.execute("UPDATE lcm_chunk_meta SET embedded_at=embedded_at+1 WHERE chunk_id='1:0'")
    assert not audit.audit(tmp_path/"covered.db", independent)["complete"]


def test_setup_persists_observed_digests_between_stages(corpus, tmp_path, monkeypatch):
    db, manifest, mod = corpus
    monkeypatch.setattr(measurement, "prepare", lambda _: manifest)
    monkeypatch.setattr(calibration, "product", lambda *a: mod)
    monkeypatch.setattr(measurement, "fill_covered", lambda *a: cover(db, manifest, mod))
    a = NS(scratch=tmp_path, out=tmp_path, lcmx_repo=REPO, hermes_source=REPO)
    # covered() expects baseline.db; use the fixture's DB as its existing input.
    db.rename(tmp_path/"baseline.db")
    db = tmp_path/"baseline.db"
    common.dump(tmp_path/"manifest.json", manifest)
    calibration.covered(a)
    persisted = common.load(tmp_path/"manifest.json")
    assert persisted["observed_vectors"] and persisted["observed_metadata"]
    with sqlite3.connect(tmp_path/"store.db") as c:
        c.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'")
        c.execute("UPDATE lcm_chunk_meta SET embedded_at=embedded_at+1 WHERE chunk_id='1:0'")
    later = audit.audit(tmp_path/"store.db", persisted)
    assert later["changed_vectors"] and later["changed_metadata"]


@pytest.mark.parametrize("batch_s,samples,expected", [(.572, 5, "PASS"), (.286, 5, "FAIL"), (.572, 4, "UNCALIBRATED")])
def test_production_block_calibration(tmp_path, monkeypatch, batch_s, samples, expected):
    a = NS(out=tmp_path, queries=200, seed=1586, blocks=1, block=0, build_covered=False,
        standalone_calibration_s=.00001, smoke=False, label="fixture", shape="short", baseline="empty")
    monkeypatch.setattr(recall, "prepare", lambda _: dict(canaries=[dict(value="fixture")]))
    pubs = [dict(start=0, end=10, node_id=i) for i in range(10)]
    # p50 stays below 2 s, but the fast runner's scaled p90 exceeds 8 s.
    data = []
    for cls in ("canary", "broad_nl"):
        data += rows(160, seconds=.5, query_class=cls)+rows(40, seconds=4.5, query_class=cls)
    def cell(args, state, block, kind):
        root = tmp_path/f"{block}-{state}"
        root.mkdir()
        for name, values in (("recalls", data), ("publications", pubs),
                ("events", ([dict(event="calibration_window_start", at=-1)]+events(batch_s)[:samples]
                    +[dict(event="calibration_window_end", at=10)]) if state == "ON-catching-up" else [])):
            (root/f"{name}.jsonl").write_text("".join(json.dumps(v)+"\n" for v in values))
        raw = recall.score(data, pubs, state=state, activity=dict(catchup_activity=True))
        raw.update(measurement_valid=True, catchup_activity=True)
        common.dump(root/"raw-summary.json", raw)
        common.dump(root/"request.json", dict(seed=1586, state=state, block=block))
        common.dump(root/"summary.json", raw)
        return raw
    monkeypatch.setattr(recall, "cell", cell)
    recall.main(a)
    summary = common.load(tmp_path/"summary.json")
    assert summary["state"] == expected
    assert rescore.v2(tmp_path)["state"] == expected
    for entry in summary["blocks"]:
        result = entry["result"]
        assert result["state"] == expected
        assert result["classes"]["canary"]["statistics"]["p90"]["seconds"] == 4.5
        if expected == "FAIL":
            assert result["classes"]["canary"]["normalized_statistics"]["p90"]["seconds"] == pytest.approx(9.)
        if expected == "UNCALIBRATED":
            assert result["reason"] == "calibration_window_short"
        assert rescore.v2(tmp_path/f"0-{entry['state']}")["state"] == expected


@pytest.mark.parametrize("cap,expected", [(2000, "PASS"), (100, "INVALID")])
def test_fast_canary_volume_and_cap(tmp_path, monkeypatch, cap, expected):
    clock = [0.]
    data, pubs, used, view_sizes = [], [], [], []
    enough, halt = threading.Event(), threading.Event()
    overlaps = dict(canary=0, broad_nl=0)
    a = NS(smoke=False, seed=1586, block=0, root=tmp_path, volume_seconds=cap)
    def tick():
        i = len(used)
        used.append(i)
        clock[0] += 1
        if i % 3 == 0:
            pubs.append(dict(start=i+.4, end=i+.45, node_id=i))
        # Continuous alternating .19s calls; only calls intersecting publication count.
        while len(data)*.19 < clock[0]:
            j = len(data)
            data.append(dict(rows(1, seconds=.19, query_class="canary" if j % 2 == 0 else "broad_nl")[0],
                start=j*.19, end=(j+1)*.19))
        if i % 3 == 0:
            for row in recall.overlapping(data[-6:], pubs[-1:]):
                overlaps[row["query_class"]] += 1
        if min(overlaps.values()) >= 200:
            enough.set()
    engine = NS(current_session_id="fixture", _conversation_id="fixture", ingest=lambda v: view_sizes.append(len(v)),
        should_compress_preflight=lambda v: False, note_turn_complete=tick, _prepare_leaves_after_turn=lambda v: None)
    monkeypatch.setattr(recall.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(recall, "safety", lambda *a: {})
    # Exercise the real seeded trace, with synthetic publications and a fake clock.
    if hasattr(recall, "publication_trace"):
        _, turns, capped = recall.publication_trace(a, engine, lambda _: NS(count_messages_tokens=lambda v: 0), enough, halt)
    else:
        measurement.trace(engine, lambda _: NS(count_messages_tokens=lambda v: 0), 600, a.seed, lambda _: None, lambda: None)
        turns, capped = len(used), False
    result = recall.score(data, pubs, 1586, "OFF", None, .572, .572)
    if capped:
        result.update(state="INVALID", reason="volume_cap")
    assert result["state"] == expected
    if expected == "PASS":
        assert turns > 600 and all(c["overlapping"] >= 200 and c["overlapping_publications"] >= 10 for c in result["classes"].values())
        assert max(view_sizes) > 1200  # Same view and query timeline extend past the old cutoff.
    else:
        assert result["reason"] == "volume_cap" and turns == cap


@pytest.mark.parametrize("states,expected", [(["PASS", "INVALID"], "INVALID"), (["FAIL", "INVALID"], "FAIL"), (["PASS", "PASS"], "PASS"), (["PASS", "ABORTED"], "INVALID"), (["PASS", "UNCALIBRATED"], "UNCALIBRATED")])
def test_pair_rollup_and_rescore(tmp_path, monkeypatch, states, expected):
    a = NS(out=tmp_path, scratch=tmp_path/"scratch", cgroup=tmp_path, cell="coverage-pair", shape="short", baseline="empty")
    monkeypatch.setattr(floor, "measured_limits", lambda _: {})
    monkeypatch.setattr(floor.subprocess, "check_output", lambda *a, **k: "fixture")
    monkeypatch.setattr(floor, "standalone", lambda _: {})
    values = iter(states)
    def coverage(a):
        common.dump(a.out/"summary.json", dict(state=next(values), reason="fixture", baseline=a.baseline))
    monkeypatch.setattr(floor, "coverage", coverage)
    floor.limited(a)
    assert common.load(tmp_path/"summary.json")["state"] == expected
    # Independently re-score children without introducing fake coverage statistics.
    original = rescore.v2
    monkeypatch.setattr(rescore, "v2", lambda path: common.load(path/"summary.json") if path != tmp_path else original(path))
    assert rescore.v2(tmp_path)["state"] == expected


@pytest.mark.parametrize("worker", [False, True])
def test_early_invalid_rescores_without_raw_files(tmp_path, worker):
    value = dict(state="INVALID", reason="covered_store_incomplete")
    common.dump(tmp_path/("raw-summary.json" if worker else "summary.json"), value)
    assert rescore.v2(tmp_path) == value


def test_rescore_rejects_tampered_duration(tmp_path):
    (tmp_path/"catchup").mkdir()
    raw = dict(state="INCOMPLETE", rows_s=2, audit=dict(complete=False), end_to_end_s=3600)
    common.dump(tmp_path/"catchup/summary.json", raw)
    common.dump(tmp_path/"manifest.json", dict(rows=250000))
    (tmp_path/"catchup/events.jsonl").write_text("".join(json.dumps(v)+"\n" for v in events()))
    common.dump(tmp_path/"summary.json", floor.rate(raw, 7200, 250000, events()))
    with pytest.raises(AssertionError):
        rescore.v2(tmp_path)


def test_incomplete_worker_closes_without_fill(tmp_path, monkeypatch):
    common.dump(tmp_path/"manifest.json", dict(canaries=[], expected=dict(chunks={}, leaves={})))
    common.dump(tmp_path/"queries.json", [])
    calls = []
    a = NS(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path, state="ON-covered")
    engine = NS(shutdown=lambda: calls.append("shutdown"))
    monkeypatch.setattr(recall, "runtime", lambda _: (a, lambda _: pytest.fail("backfill called"), engine, 0))
    monkeypatch.setattr(measurement, "fill_covered", lambda *a: pytest.fail("fill called"))
    monkeypatch.setattr(measurement, "audit", lambda *a: dict(complete=False))
    recall.worker(None)
    assert common.load(tmp_path/"raw-summary.json") == dict(state="INVALID", reason="covered_store_incomplete")
    assert calls == ["shutdown"]


def test_setup_upload_contains_metadata_only():
    path = REPO/".github/workflows/lcmx-1586-covered-store.yml"
    steps = yaml.safe_load(path.read_text())["jobs"]["build"]["steps"]
    setup = next(s for s in steps if "setup-evidence" in s.get("with", {}).get("name", ""))
    paths = setup["with"]["path"].splitlines()
    assert "evidence/" not in paths and "evidence/covered/" not in paths
    assert "evidence/covered/store.db" not in paths
    assert {"evidence/covered/audit.json", "evidence/covered/manifest.json", "evidence/calibration/"} <= set(paths)
