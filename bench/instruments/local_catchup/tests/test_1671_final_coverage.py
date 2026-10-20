"""Final expected-corpus coverage stays authoritative after real trace ingestion."""
from argparse import Namespace as NS
import json
import sqlite3

import pytest

from bench.instruments.local_catchup import common, measurement, recall
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


@pytest.mark.parametrize("missing", [None, "chunks", "leaves"])
def test_final_expected_coverage_after_trace_ingestion(corpus, tmp_path, monkeypatch, missing):
    db, manifest, mod = corpus
    common.dump(tmp_path/"manifest.json", manifest)
    common.dump(tmp_path/"queries.json", [dict(id=f"q{i}", canary=0) for i in range(200)])
    # Satisfy sample gates independently; this is a policy test on an eight-row corpus.
    a = NS(out=tmp_path, root=tmp_path, db=db, home=tmp_path/"trace-home",
           state="ON-catching-up", smoke=False, rows=250000, queries=200, seed=1, block=0)
    cfg = mod("config").LCMConfig(database_path=str(db), embeddings_enabled=False,
        temporal_rollups_enabled=False, large_output_externalization_enabled=False)
    engine = mod("engine").LCMEngine(config=cfg, hermes_home=str(a.home))
    engine.on_session_start("i2-trace", platform="eval", conversation_id="i2-trace", context_length=6000)
    payload = dict(hits=[dict(arms=["fts"])], provenance=dict(arms_run=["chunk", "summary"],
        coverage=dict(chunk="full", summary="full")))
    monkeypatch.setattr(recall, "runtime", lambda _: (a, lambda _: NS(
        lcm_recall=lambda *a, **kw: json.dumps(payload)), engine, 0))
    store = NS(close=lambda: None)
    monkeypatch.setattr(recall, "recall_reader", lambda *a: NS(_dag=store, _store=store))
    monkeypatch.setattr(recall, "safety", lambda *a: {})
    monkeypatch.setattr(recall, "overlapping", lambda rows, _: rows)
    monkeypatch.setattr(recall, "overlapping_publications", lambda *a: 10)
    monkeypatch.setattr(recall, "catchup_activity", lambda *a: dict(catchup_activity=True))
    monkeypatch.setattr(recall, "score", lambda *a: dict(state="PASS"))
    class Thread:
        def __init__(self, target, **kw): self.target = target
        def start(self): self.target()
        def join(self, *a): pass
        def is_alive(self): return False
    monkeypatch.setattr(recall.threading, "Thread", Thread)
    def trace(*args):
        measurement.trace(engine, mod, 2, a.seed, args[4], args[5])
        fill(db, manifest, mod)
        if missing:
            table, meta, key, ident = ("lcm_chunk_vectors", "lcm_chunk_meta", "chunk_id", "1:0") if missing == "chunks" else (
                "lcm_embedding_vectors", "lcm_embedding_meta", "embedded_id", "1")
            with sqlite3.connect(db) as c:
                c.execute(f"DELETE FROM {table} WHERE {key}=?", (ident,))
                c.execute(f"DELETE FROM {meta} WHERE {key}=?", (ident,))
    monkeypatch.setattr(recall, "trace", trace)
    recall.worker(None)
    final = common.load(tmp_path/"final-audit.json")
    assert final["raw_unchanged"] is False
    assert final["complete"] is False
    assert all(v["covered"] == v["expected"] for v in final["classes"].values()) is (missing is None)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT count(*) FROM messages").fetchone()[0] > manifest["rows"]
    rows = [json.loads(line) for line in (tmp_path/"recalls.jsonl").read_text().splitlines()]
    assert len(rows) == 200 and all(r["incomplete"] for r in rows)
    assert common.load(tmp_path/"raw-summary.json")["state"] == ("INVALID" if missing is None else "PASS")
