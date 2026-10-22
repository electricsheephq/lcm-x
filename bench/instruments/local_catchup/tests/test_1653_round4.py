"""Round-4 verdict and observation boundary regressions."""
from argparse import Namespace as NS
import json
import sqlite3
import threading

import pytest

from bench.instruments.local_catchup import audit, common, measurement, recall
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


@pytest.mark.parametrize("arm", ["chunk", "summary"])
def test_honest_empty_is_healthy_but_not_a_latency_success(arm):
    payload = dict(degraded=True, degraded_reason=f"{arm} vectors are unavailable",
        provenance=dict(coverage={arm: "none", "summary" if arm == "chunk" else "chunk": "full"},
                        arms_run=["summary" if arm == "chunk" else "chunk"]))
    # The exemption needs measured incompleteness, and never applies to covered cells.
    assert not recall.semantic_healthy(payload)
    assert recall.semantic_healthy(payload, [arm])
    row = dict(start=0, end=1, seconds=.1, health="degraded", fts_hits=True,
               honest_coverage=True, semantic_healthy=True, semantic_empty=True)
    result = recall.score([row]*200, [dict(start=0, end=1, node_id=i) for i in range(10)],
                          state="ON-catching-up", activity=dict(catchup_activity=True))
    assert result["state"] == "INVALID" and result["failures"] == 0
    assert result["overlapping"] == 0
    payload["degraded_reason"] += "; chunk arm failed: boom"
    assert not recall.semantic_healthy(payload, [arm])


def test_no_semantic_hits_with_complete_vectors_is_failure():
    payload = dict(degraded=True, degraded_reason="chunk vectors are unavailable; summary vectors are unavailable",
                   provenance=dict(coverage=dict(chunk="none", summary="none"), arms_run=["fts"]))
    assert not recall.semantic_healthy(payload)


def test_health_failures_use_publication_overlap():
    row = dict(start=0, end=1, seconds=.1, health="ok", fts_hits=True,
               honest_coverage=True, semantic_healthy=True)
    bad = dict(row, start=2, end=3, health="error")
    result = recall.score([row]*200+[bad], [dict(start=0, end=1, node_id=i) for i in range(10)])
    assert result["state"] == "FAIL" and result["failures"] == 1  # v2 item 5: health covers every recall.


def test_request_scheduled_maintenance_captures_background_context(corpus, monkeypatch, tmp_path):
    db, manifest, mod = corpus
    em = mod("engine")
    jobs, seen = [], []
    monkeypatch.setattr(em._EMBEDDING_MAINTENANCE_SCHEDULER, "schedule",
                        lambda key, job, **kw: jobs.append(job))
    monkeypatch.setattr(em, "run_incremental_embedding_pass",
                        lambda *a, **kw: mod("command")._chunk_backfill_run())
    monkeypatch.setattr(mod("command"), "_chunk_backfill_run",
                        lambda: seen.append(common.REQUEST.get()) or {})
    for name in ("_chunk_backfill_run", "_embedding_backfill_summary_run"):
        monkeypatch.setattr(mod("command"), name, getattr(mod("command"), name))
    monkeypatch.setattr(mod("embedding_provider").FastembedProvider, "_construct",
                        mod("embedding_provider").FastembedProvider._construct)
    monkeypatch.setattr(em.LCMEngine, "_schedule_embedding_maintenance", em.LCMEngine._schedule_embedding_maintenance)
    events = tmp_path/"events.jsonl"
    common.guards(mod, events)
    engine = em.LCMEngine(config=common.config(mod, db), hermes_home=str(tmp_path))
    try:
        with common.foreground_request():
            engine._schedule_embedding_maintenance()
            assert common.REQUEST.get()
        thread = threading.Thread(target=jobs.pop())
        thread.start()
        thread.join(5)
        assert not thread.is_alive() and seen == [False]
        assert all(not e.get("foreground") for e in map(json.loads, events.read_text().splitlines()))
    finally:
        engine.shutdown()


def test_reader_shares_late_engine_provider_resolution(corpus, tmp_path):
    db, _, mod = corpus
    engine = NS(_config=common.config(mod, db))
    reader = recall.recall_reader(engine, mod, db, tmp_path)
    try:
        provider = mod("tools")._lcm_grep_resolve_provider(engine)
        assert mod("tools")._lcm_grep_resolve_provider(reader) is provider
        reader._lcm_chunk_provider_cache = ("key", provider)
        assert engine._lcm_chunk_provider_cache == ("key", provider)
    finally:
        reader._store.close()
        reader._dag.close()


def test_progress_hashes_only_new_vectors_and_final_audit_protects_them(corpus, monkeypatch):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    def forbidden(*args):
        pytest.fail("progress hashes the whole store")
    with monkeypatch.context() as patch:
        patch.setattr(audit, "preimage", forbidden)
        audit.progress(db, manifest)
        assert manifest["observed_vectors"]
        patch.setattr(audit.hashlib, "sha256", forbidden)
        audit.progress(db, manifest)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'")
    result = audit.audit(db, manifest)
    assert result["changed_vectors"] and not result["complete"]


def test_final_audit_protects_first_seen_vectors(corpus):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    assert audit.audit(db, manifest)["complete"]
    with sqlite3.connect(db) as c:
        c.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'")
    assert not audit.audit(db, manifest)["complete"]


def test_subprocess_claim_names_exact_observed_api(tmp_path, monkeypatch):
    common.dump(tmp_path/"manifest.json", {})
    monkeypatch.setattr(measurement, "foreign_files", lambda *a, **kw: [])
    a = NS(root=tmp_path, home=tmp_path, db=tmp_path/"db", model_cache=tmp_path,
           user_home=str(tmp_path), scratch=tmp_path, out=tmp_path, _next_progress=float("inf"))
    assert measurement.safety(a)["subprocess_launches_recorded"] == "subprocess.Popen"


def test_progress_first_observation_is_protected(corpus):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    audit.progress(db, manifest)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'")
    assert audit.audit(db, manifest)["changed_vectors"]
