"""Issue #1669: round-five instrument claims, reproduced before implementation."""
from argparse import Namespace as NS
from pathlib import Path
import json
import threading

import pytest

from bench.instruments.local_catchup import common, recall

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]
WORKFLOW = Path(__file__).resolve().parents[4]/".github/workflows/lcmx-1586-floor.yml"


@pytest.mark.parametrize("post_complete,final_complete", [(True, False), (False, True)])
def test_item5_completion_during_recall_or_by_final_audit_invalidates(tmp_path, monkeypatch, post_complete, final_complete):
    common.dump(tmp_path/"manifest.json", dict(expected=dict(chunks={"1": 1}, leaves={"1": 1}),
                                               canaries=[dict(value="query")]))
    common.dump(tmp_path/"queries.json", [dict(id=f"q{i}", canary=0, query_class="canary" if i % 2 == 0 else "broad_nl", text="What turn was acknowledged") for i in range(400)])
    a = NS(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path,
           state="ON-catching-up", smoke=False, rows=250000, queries=200, seed=1, block=0)
    engine = NS(_config=NS(recall_query_timeout_s=1), _dag=NS(add_node=lambda *a, **kw: 1), shutdown=lambda: None)
    store = NS(close=lambda: None)
    payload = dict(hits=[dict(arms=["fts"])], provenance=dict(arms_run=["chunk", "summary"],
                    coverage=dict(chunk="full", summary="full")))
    # Declaration item 6: isolate the existing health probe from the new FTS observer.
    monkeypatch.setattr(recall, "diagnostics", lambda *a: ({}, None))
    monkeypatch.setattr(recall, "calibration_window", lambda *a: None)  # Separate raw-window regressions in fix2.
    monkeypatch.setattr(recall, "runtime", lambda _: (a, lambda _: NS(lcm_recall=lambda *a, **k: json.dumps(payload)), engine, 0))
    monkeypatch.setattr(recall, "recall_reader", lambda *a: NS(_dag=store, _store=store))
    calls = []
    def counts(*_):
        calls.append(1)
        return dict(chunks=int(post_complete and len(calls) % 2 == 0), leaves=1)
    monkeypatch.setattr(recall, "vector_counts", counts)
    monkeypatch.setattr(recall, "audit", lambda *a: dict(complete=final_complete,
        classes={kind: dict(covered=int(final_complete), expected=1, missing_count=int(not final_complete))
                 for kind in ("chunks", "leaves")}))
    monkeypatch.setattr(recall, "integrity", lambda *a: None)
    monkeypatch.setattr(recall, "safety", lambda *a: {})
    monkeypatch.setattr(recall, "trace", lambda *a: None)
    monkeypatch.setattr(recall, "overlapping", lambda rows, _: rows)
    monkeypatch.setattr(recall, "overlapping_publications", lambda *a: 10)
    monkeypatch.setattr(recall, "OverlapCounter", lambda _: NS(add=lambda *a: None,
        counted={c: set(range(200)) for c in ("canary", "broad_nl")},
        publications={c: set(range(10)) for c in ("canary", "broad_nl")}))
    monkeypatch.setattr(recall, "catchup_activity", lambda *a: dict(catchup_activity=True))
    monkeypatch.setattr(recall, "score", lambda *a: dict(state="PASS"))
    class Thread:
        def __init__(self, target, **kw): self.target = target
        def start(self): self.target()
        def join(self, *a): pass
        def is_alive(self): return False
    monkeypatch.setattr(threading, "Thread", Thread)
    recall.worker(None)
    assert common.load(tmp_path/"raw-summary.json")["state"] == "INVALID"
    if post_complete:
        assert all(not json.loads(s)["incomplete"] for s in (tmp_path/"recalls.jsonl").read_text().splitlines())
