"""Failing-first regressions for the three #1660 false-PASS review findings."""
from argparse import Namespace as NS
import sqlite3
import subprocess
import sys

import pytest

from bench.instruments.local_catchup import audit, common, measurement, recall, turns
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


def cell_args(tmp_path, db):
    return NS(root=tmp_path, home=tmp_path, db=db, model_cache=tmp_path,
              user_home=str(tmp_path), scratch=tmp_path, out=tmp_path,
              lcmx_repo=tmp_path, hermes_source=tmp_path, _ready=lambda: True)


def test_safety_polls_preserve_first_vector_digest_for_final_audit(corpus, tmp_path, monkeypatch):
    db, manifest, mod = corpus
    common.dump(tmp_path/"manifest.json", manifest)
    a = cell_args(tmp_path, db)
    monkeypatch.setattr(measurement, "foreign_files", lambda *a, **kw: [])
    manifests = []
    def poll(db, manifest, verify):
        manifests.append(manifest)
        return audit.progress(db, manifest, verify)
    monkeypatch.setattr(measurement, "progress", poll)
    fill(db, manifest, mod)
    measurement.safety(a)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'")
    a._next_progress = 0
    measurement.safety(a)
    final = audit.audit(db, manifests[-1])
    assert final["changed_vectors"] and not final["complete"]
    assert manifests[0] is manifests[1]


def test_turns_final_integrity_sees_vectors_observed_by_safety(corpus, tmp_path, monkeypatch):
    db, manifest, mod = corpus
    common.dump(tmp_path/"manifest.json", manifest)
    a = cell_args(tmp_path, db)
    a.samples, a.seed, a.block, a.state = 1, 1, 0, "OFF"
    engine = NS(shutdown=lambda: None)
    monkeypatch.setattr(turns, "runtime", lambda _: (a, mod, engine, 0))
    monkeypatch.setattr(measurement, "foreign_files", lambda *a, **kw: [])
    monkeypatch.setattr(turns, "catchup_activity", lambda *a, **kw: {})
    def trace(engine, mod, count, seed, observe, check):
        fill(db, manifest, mod)
        observe(dict(ingest_preflight_ms=1, dispatch_ms=1, compression_due=True))
        check()
        with sqlite3.connect(db) as c:
            c.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'")
        a._next_progress = 0
        check()
    monkeypatch.setattr(turns, "trace", trace)
    with pytest.raises(RuntimeError, match="vector corruption"):
        turns.worker(None)


@pytest.mark.parametrize("reason", [
    "summary query timeout", "chunk query timeout",
    "session exclusion scope resolution timed out",
    "semantic latency budget exhausted",
    "lcm-summary exceeded the semantic latency budget",
    "provider resolution deadline exhausted",
    "chunk provider resolution deadline exhausted",
])
def test_full_coverage_timeout_cannot_pass_recall(reason):
    payload = dict(degraded=True, degraded_reason=reason,
                   provenance=dict(arms_run=["fts", "chunk", "summary"],
                                   coverage=dict(chunk="full", summary="full")))
    healthy = recall.semantic_healthy(payload)
    row = dict(start=0, end=1, seconds=.1, health="degraded", fts_hits=True,
               honest_coverage=True, semantic_healthy=healthy)
    result = recall.score([row]*200, [dict(start=0, end=1, node_id=i) for i in range(10)],
                          state="ON-covered")
    assert not healthy and result["state"] == "FAIL" and result["failures"] == 200


@pytest.mark.parametrize("caught", [False, True])
def test_offline_popen_invalidates_cell_even_if_product_catches_it(tmp_path, monkeypatch, caught):
    provider = NS(_FASTEMBED_THREADS=2, FastembedProvider=NS(_construct=lambda *a, **kw: None))
    command = NS(_chunk_backfill_run=lambda: {}, _embedding_backfill_summary_run=lambda: {})
    common.guards(lambda name: provider if name == "embedding_provider" else command,
                  tmp_path/"events.jsonl")
    a = cell_args(tmp_path, tmp_path/"db")
    a._next_progress = float("inf")
    monkeypatch.setattr(measurement, "foreign_files", lambda *a, **kw: [])
    request = tmp_path/"request.json"
    common.dump(request, dict(root=str(tmp_path)))
    def worker(_):
        try:
            subprocess.run([sys.executable, "-c", "pass"], check=True)
        except RuntimeError:
            if not caught:
                raise
        evidence = measurement.safety(a)
        common.dump(tmp_path/"raw-summary.json", dict(state="PASS", safety=evidence))
    measurement.guarded_worker(worker, request)
    result = common.load(tmp_path/"raw-summary.json")
    assert result == dict(state="INVALID", reason="offline_subprocess_launch")
    assert common.load(tmp_path/"events.jsonl")["violation"] == "subprocess_launch"
