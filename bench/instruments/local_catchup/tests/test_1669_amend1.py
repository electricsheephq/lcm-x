"""Recall honesty: provider disclosure and coverage completed during the call."""
from argparse import Namespace as NS
import json

import pytest

from bench.instruments.local_catchup import common, recall


@pytest.mark.parametrize("reason,coverage,before,after,honest", [
    ("embedding provider unavailable: local embedding model is preparing; retrieval uses full text",
     {"fts": "ok"}, (0, 0), (0, 0), True),
    ("embedding provider unavailable: local embedding model is unavailable (load failed); retrieval uses full text",
     {"fts": "ok"}, (0, 0), (0, 0), True),
    ("embedding provider unavailable: local embedding model is preparing; retrieval uses full text",
     {"chunk": "none", "summary": "none"}, (0, 0), (0, 0), True),
    ("chunk vectors are unavailable", {"chunk": "none", "summary": "full"}, (0, 0), (0, 4), True),
    ("chunk vectors are unavailable", {"chunk": "none", "summary": "full"}, (0, 0), (0, 3), False),
    ("chunk vectors are unavailable", {"chunk": "none"}, (0, 4), (0, 4), True),
    ("arm failed", {}, (0, 0), (0, 0), False),
    ("embedding provider unavailable: local embedding model is preparing; retrieval uses full text",
     {"chunk": "full", "summary": "none"}, (0, 0), (0, 0), False),
])
def test_honesty_uses_disclosure_and_before_after_counts(tmp_path, monkeypatch, reason, coverage, before, after, honest):
    common.dump(tmp_path/"manifest.json", dict(expected=dict(chunks={"1": 1}, leaves=dict.fromkeys("1234", 1)),
                                               canaries=[dict(value="query")]))
    common.dump(tmp_path/"queries.json", [dict(id="q0009", canary=0)])
    a = NS(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path,
           state="ON-catching-up", smoke=True, rows=2, queries=1, seed=1, block=0)
    store = NS(close=lambda: None)
    engine = NS(_config=NS(recall_query_timeout_s=1), _dag=NS(add_node=lambda *a, **kw: 1), shutdown=lambda: None)
    payload = dict(degraded=True, degraded_reason=reason, provenance=dict(coverage=coverage), hits=[])
    counts = iter([dict(zip(("chunks", "leaves"), values)) for values in (before, after)])
    # Declaration item 6: isolate the existing health probe from the new FTS observer.
    monkeypatch.setattr(recall, "diagnostics", lambda *a: ({}, None))
    monkeypatch.setattr(recall, "runtime", lambda _: (a, lambda _: NS(lcm_recall=lambda *a, **kw: json.dumps(payload)), engine, 0))
    monkeypatch.setattr(recall, "recall_reader", lambda *a: NS(_dag=store, _store=store))
    monkeypatch.setattr(recall, "vector_counts", lambda *a: next(counts))
    monkeypatch.setattr(recall, "audit", lambda *a: {})
    monkeypatch.setattr(recall, "integrity", lambda *a: None)
    monkeypatch.setattr(recall, "safety", lambda *a: {})
    monkeypatch.setattr(recall, "trace", lambda *a: None)
    class Thread:
        def __init__(self, target, **kw): self.target = target
        def start(self): self.target()
        def join(self, *a): pass
        def is_alive(self): return False
    monkeypatch.setattr(recall.threading, "Thread", Thread)
    event = recall.threading.Event
    def stopped_event():
        result = event()
        result.set()  # Exactly one pinned query, with no real background thread.
        return result
    monkeypatch.setattr(recall.threading, "Event", stopped_event)
    recall.worker(None)
    row = common.load(tmp_path/"recalls.jsonl")
    assert row["honest_coverage"] is honest
    assert row["coverage_start"] == dict(zip(("chunks", "leaves"), before))
    assert row["coverage_end"] == dict(zip(("chunks", "leaves"), after))
    assert row["incomplete"] is True  # The chunk arm remains incomplete in every shape.
    if "local embedding model is" in reason:
        assert row["semantic_healthy"] is False
        assert row["semantic_empty"] is False
