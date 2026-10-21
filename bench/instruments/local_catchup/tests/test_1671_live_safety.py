"""Keep the live engine's file-access snapshot through teardown and summary."""
from argparse import Namespace as NS
import json

import pytest

from bench.instruments.local_catchup import audit, common, measurement, recall, turns


@pytest.mark.parametrize("module", [recall, turns], ids=["recall", "turns"])
@pytest.mark.parametrize("foreign", [True, False], ids=["foreign", "own"])
def test_worker_snapshots_handles_before_shutdown(tmp_path, monkeypatch, module, foreign):
    scratch = tmp_path/"scratch"
    a = NS(out=tmp_path/"evidence", root=tmp_path/"evidence", db=scratch/"cell.db",
           home=scratch/"0-OFF-home", scratch=scratch, model_cache=tmp_path/"cache",
           user_home=tmp_path/"runner-home", lcmx_repo=tmp_path/"repo", hermes_source=tmp_path/"host",
           state="OFF", smoke=True, rows=2, queries=1, samples=1, seed=1, block=0,
           _next_progress=float("inf"))
    a.root.mkdir()
    path = scratch/"sibling-home"/"state.db" if foreign else a.home/"state.db"
    path.parent.mkdir(parents=True)
    path.touch()
    handle = path.open("rb")
    calls = []
    def shutdown():
        calls.append("shutdown")
        handle.close()
    engine = NS(_config=NS(recall_query_timeout_s=1), _dag=NS(add_node=lambda *a, **kw: 1),
                shutdown=shutdown)
    common.dump(a.out/"manifest.json", dict(canaries=[dict(value="query")]))
    common.dump(a.out/"queries.json", [dict(id="q0", canary=0)])
    payload = dict(hits=[dict(arms=["fts"])], provenance={})
    monkeypatch.setattr(module, "runtime", lambda _: (a, lambda _: NS(
        lcm_recall=lambda *a, **kw: json.dumps(payload)), engine, 0))
    def safety(args):
        calls.append("live-safety" if not handle.closed else "closed-safety")
        return dict(measurement.safety(args), engine_live=not handle.closed)
    def integrity(*_):
        assert handle.closed, "integrity must follow shutdown"
        calls.append("integrity")
    monkeypatch.setattr(module, "safety", safety)
    monkeypatch.setattr(module, "integrity", integrity)
    monkeypatch.setattr(module, "catchup_activity", lambda *a, **kw: dict(catchup_activity=False))
    monkeypatch.setattr(module, "trace", lambda *a: a[4](dict(
        ingest_preflight_ms=1, dispatch_ms=1, compression_due=True)))
    if module is recall:
        store = NS(close=lambda: None)
        monkeypatch.setattr(recall, "recall_reader", lambda *a: NS(_dag=store, _store=store))
        monkeypatch.setattr(recall, "audit", lambda *a: {})
        class Thread:
            def __init__(self, target, **kw): self.target = target
            def start(self): self.target()
            def join(self, *a): pass
            def is_alive(self): return False
        monkeypatch.setattr(recall.threading, "Thread", Thread)
        event = recall.threading.Event
        def stopped_event():
            result = event()
            result.set()
            return result
        monkeypatch.setattr(recall.threading, "Event", stopped_event)
    else:
        monkeypatch.setattr(audit, "progress", lambda *a: dict(complete=False, classes={}))
    try:
        if foreign:
            with pytest.raises(RuntimeError, match="cross-profile access"):
                module.worker(None)
            assert common.load(a.root/"file-access.jsonl")["unexpected_path_hashes"]
            assert calls == ["live-safety", "shutdown"]
        else:
            module.worker(None)
            summary = common.load(a.root/"raw-summary.json")
            evidence = summary["safety" if module is recall else "join_evidence"]
            assert evidence["engine_live"] is True
            assert calls == ["live-safety", "shutdown", "integrity"]
    finally:
        handle.close()
