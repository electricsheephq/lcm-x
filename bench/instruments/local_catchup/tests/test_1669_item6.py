"""Issue #1669: round-five instrument claims, reproduced before implementation."""
from argparse import Namespace as NS
from pathlib import Path
import sqlite3

import pytest

from bench.instruments.local_catchup import measurement
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]
WORKFLOW = Path(__file__).resolve().parents[4]/".github/workflows/lcmx-1586-floor.yml"


@pytest.mark.parametrize("failure", ["unpaired", "residue"])
def test_item6_judged_integrity_rejects_pairing_and_residue(corpus, tmp_path, failure):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    if failure == "unpaired":
        with sqlite3.connect(db) as c:
            c.execute("DELETE FROM lcm_chunk_meta WHERE chunk_id='1:0'")
    else:
        with sqlite3.connect(db) as c:
            mod("command")._ensure_inflight_table(c)
            c.execute("INSERT INTO lcm_embedding_backfill_inflight(embedded_id,identity_hash,state) VALUES('1:0',?,'uncertain')",
                      (manifest["identities"]["chunk"],))
    with pytest.raises(RuntimeError, match="pairing|residue"):
        measurement.integrity(NS(db=db, root=tmp_path), manifest)


@pytest.mark.parametrize("kind", ["turns", "recall"])
def test_item6_final_integrity_runs_after_shutdown(tmp_path, monkeypatch, kind):
    from bench.instruments.local_catchup import common, turns, recall
    module = turns if kind == "turns" else recall
    closed = []
    store = NS(close=lambda: None)
    engine = NS(shutdown=lambda: closed.append(True), _config=NS(recall_query_timeout_s=1),
                _dag=NS(add_node=lambda *a, **kw: 1))
    a = NS(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path, state="OFF",
           samples=1, seed=1, block=0, smoke=True, rows=2000, queries=1)
    common.dump(tmp_path/"manifest.json", dict(expected=dict(chunks={}, leaves={}), canaries=[dict(value="query")]))
    common.dump(tmp_path/"queries.json", [dict(id="q", canary=0)])
    payload = '{"hits":[{"arms":["fts"]}]}'
    def mod(_):
        return NS(lcm_recall=lambda *a, **kw: payload)
    monkeypatch.setattr(module, "runtime", lambda _: (a, mod, engine, 0))
    monkeypatch.setattr(module, "safety", lambda *a: {})
    monkeypatch.setattr(module, "catchup_activity", lambda *a, **kw: dict(catchup_activity=False))
    def trace(engine, mod, count, seed, observe, check, done=None):
        observe(dict(ingest_preflight_ms=1, dispatch_ms=1, compression_due=True))
        check()
    monkeypatch.setattr(module, "trace", trace)
    if kind == "recall":
        # Declaration item 6: this fixture tests teardown, not the FTS observer.
        monkeypatch.setattr(recall, "diagnostics", lambda *a: ({}, None))
        monkeypatch.setattr(module, "recall_reader", lambda *a: NS(_dag=store, _store=store))
        monkeypatch.setattr(module, "audit", lambda *a: dict(complete=False))
    else:
        from bench.instruments.local_catchup import audit
        monkeypatch.setattr(audit, "progress", lambda *a: dict(complete=False, classes={}))
    def integrity(*_):
        assert closed == [True], "final residue must be judged after shutdown"
    monkeypatch.setattr(module, "integrity", integrity)
    module.worker(None)
    assert closed == [True]
