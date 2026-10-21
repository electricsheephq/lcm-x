"""Review items 13–18: offline instrument regressions on the round-1 base."""
from argparse import Namespace
import json
import os
from pathlib import Path
import socket
import sqlite3
import threading

import pytest

from bench.instruments.local_catchup import audit, common, corpus as corpus_module, measurement, recall, turns
from bench.instruments.local_catchup.tests.test_corpus_audit import fill
from bench.instruments.local_catchup.tests.test_i2 import row

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


def test_13_recall_thread_provider_build_is_foreground(tmp_path, monkeypatch):
    for owner, names in ((socket.socket, ("connect", "connect_ex", "sendto")),
                         (socket, ("getaddrinfo", "gethostbyname", "gethostbyname_ex"))):
        for name in names:
            monkeypatch.setattr(owner, name, getattr(owner, name))
    provider = Namespace(_FASTEMBED_THREADS=2, FastembedProvider=Namespace(_construct=lambda *a, **kw: object()))
    command = Namespace(_chunk_backfill_run=lambda: {}, _embedding_backfill_summary_run=lambda: {})
    events, errors = tmp_path/"events.jsonl", []
    common.guards(lambda name: provider if name == "embedding_provider" else command, events)
    def build():
        try:
            with common.foreground_request():
                provider.FastembedProvider._construct(None, allow_download=False)
        except RuntimeError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=build, name="i2-recall")
    thread.start()
    thread.join()
    recorded = [json.loads(line) for line in events.read_text().splitlines()]
    assert recorded == [dict(event="prepare", foreground=True, allow_download=False, at=recorded[0]["at"])]
    assert errors == ["foreground/download preparation forbidden"]


def test_14_mixed_prepare_pilot_covers_every_shard(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[4]
    a = Namespace(scratch=tmp_path/"scratch", out=tmp_path/"out", model_cache=tmp_path,
                  rows=2000, blocks=1, samples=1, queries=1, shape="mixed", baseline="empty",
                  lcmx_repo=repo, hermes_source=Path(os.environ["INSTRUMENT_HERMES_SOURCE"]))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path/"home"))
    monkeypatch.setattr(measurement, "isolated", lambda *args: None)
    monkeypatch.setattr(measurement, "immutable_cache", lambda *args: None)
    pilots = []
    def build(*args):
        if args[2].name != "pilot.db":
            return {}
        result = corpus_module.build(*args)
        pilots.append(result)
        return result
    monkeypatch.setattr(measurement, "build", build)
    measurement.prepare(a)
    assert pilots[0]["rows"] == 1500
    assert len(pilots[0]["sessions"]) == 3
    assert all(any(f"SHARD{i:06d}" in c["id"] for c in pilots[0]["canaries"]) for i in range(3))


def test_15_turns_final_coverage_prevents_catching_up_throughout(tmp_path, monkeypatch):
    a = Namespace(out=tmp_path, root=tmp_path, db=tmp_path/"db", state="ON", samples=1, seed=1, block=0)
    common.dump(tmp_path/"manifest.json", {})
    monkeypatch.setattr(turns, "runtime", lambda _: (a, None, Namespace(shutdown=lambda: None), 1))
    reads = iter([False, False, True])
    monkeypatch.setattr(audit, "progress", lambda *args: dict(complete=next(reads), classes={}))
    monkeypatch.setattr(turns, "safety", lambda *args: {})
    monkeypatch.setattr(turns, "integrity", lambda *args: None)
    monkeypatch.setattr(turns, "catchup_activity", lambda *args, **kw: dict(catchup_activity=True))
    def trace(engine, mod, count, seed, observe, check):
        observe(dict(ingest_preflight_ms=1, dispatch_ms=1, compression_due=True))
        check()
    monkeypatch.setattr(turns, "trace", trace)
    turns.worker(None)
    assert common.load(tmp_path/"raw-summary.json")["catching_up"] is False


@pytest.mark.parametrize("table,column", [("lcm_chunk_meta", "char_end"), ("lcm_embedding_meta", "source_token_count")])
def test_16_upgrade_metadata_only_change_fails(corpus, tmp_path, table, column):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    with audit.connect(db) as c:
        manifest["preimage"] = audit.preimage(c)
    a = Namespace(db=db, root=tmp_path, baseline="upgrade")
    measurement.integrity(a, manifest)
    with sqlite3.connect(db) as c:
        c.execute(f"UPDATE {table} SET {column}={column}+1")
    with audit.connect(db) as c:
        assert audit.preimage(c)["vectors"] == manifest["preimage"]["vectors"]
    with pytest.raises(RuntimeError, match="pre-existing metadata corruption"):
        measurement.integrity(a, manifest)


@pytest.mark.parametrize("payload,expected", [
    ({"degraded": True, "degraded_reason": "summary query timeout"}, False),
    ({"degraded": True, "degraded_reason": "chunk semantic index catching up (2 rows not yet embedded)"}, True),
    ({"degraded": True, "degraded_reason": "summary semantic index catching up"}, False),
    ({"degraded": False, "degraded_reason": "chunk semantic index catching up"}, False),
    ({"degraded": True, "degraded_reason": "chunk semantic index has 2 rows awaiting /lcm embed backfill --apply --retry-uncertain --corpus chunks"}, True),
    ({"degraded": True, "degraded_reason": "chunk semantic index catching up"}, True),
    ({"degraded": True, "degraded_reason": "chunk arm coverage bounded: scored the 10 most-recent of 99 vectors (older vectors excluded)", "provenance": {"coverage": {"chunk": "bounded", "summary": "full"}}}, True),
    ({"degraded": True, "degraded_reason": "summary vectors are unavailable", "provenance": {"coverage": {"chunk": "full", "summary": "none"}}}, False),
    ({"provenance": {"coverage": {"chunk": "full"}}}, False),
])
def test_17_payload_disclosure_requires_semantic_coverage(tmp_path, monkeypatch, payload, expected):
    common.dump(tmp_path/"manifest.json", dict(expected=dict(chunks={"1": 1}, leaves={"1": 1}),
                                               canaries=[dict(value="query")]))
    common.dump(tmp_path/"queries.json", [dict(id="q", canary=0)])
    a = Namespace(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path,
                  state="ON-catching-up", smoke=True, rows=2, queries=1, seed=1, block=0)
    store = Namespace(close=lambda: None)
    engine = Namespace(_config=Namespace(recall_query_timeout_s=1), _dag=Namespace(add_node=lambda *a, **kw: 1), shutdown=lambda: None)
    def mod(name):
        return {"store": Namespace(MessageStore=lambda *args: store),
            "dag": Namespace(SummaryDAG=lambda *args: store),
            "tools": Namespace(lcm_recall=lambda *args, **kw: json.dumps(payload))}[name]
    # Declaration item 6: isolate the existing health probe from the new FTS observer.
    monkeypatch.setattr(recall, "diagnostics", lambda *a: ({}, None))
    monkeypatch.setattr(recall, "runtime", lambda _: (a, mod, engine, 1))
    monkeypatch.setattr(recall, "vector_counts", lambda *args: dict(chunks=0, leaves=1))
    monkeypatch.setattr(recall, "audit", lambda *args: {})
    monkeypatch.setattr(recall, "integrity", lambda *args: None)
    monkeypatch.setattr(recall, "safety", lambda *args: {})
    monkeypatch.setattr(recall, "trace", lambda *args: None)
    event = threading.Event
    def stopped_event():
        result = event()
        result.set()  # One pinned query runs synchronously, then stops.
        return result
    class Thread:
        def __init__(self, target, name):
            self.target = target
        def start(self):
            self.target()
        def join(self, *args):
            pass
        def is_alive(self):
            return False
    monkeypatch.setattr(threading, "Event", stopped_event)
    monkeypatch.setattr(threading, "Thread", Thread)
    recall.worker(None)
    assert common.load(tmp_path/"recalls.jsonl")["honest_coverage"] is expected


def test_18_quota_requires_ten_distinct_overlapping_publications():
    publications = [dict(node_id=0, start=0, end=3)] + [dict(node_id=i, start=10+i, end=11+i) for i in range(1, 10)]
    result = recall.score([row()]*200, publications)
    assert result["state"] == "INVALID"
    assert result["overlapping"] == 200 and result["overlapping_publications"] == 1
    assert recall.score([row()]*200, [publications[0]]*10)["state"] == "INVALID"
