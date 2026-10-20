"""Bar 3: real recall intervals overlapping committed compaction publications."""
import json
import random
import threading
import time

from .audit import audit, progress
from .common import append, dump, load
from .cpu import percentile
from .measurement import SMOKE, arguments, bootstrap, cell, integrity, prepare, runtime, safety, trace, wait_covered


def overlapping(rows, publications):
    return [r for r in rows if any(r["start"] < p["end"] and p["start"] < r["end"] for p in publications)]


def score(rows, publications, seed=1586):
    counted = overlapping(rows, publications)
    stats = {}
    for name, quantile, limit in (("p50", .5, 2.), ("p90", .9, 8.)):
        values = [r["seconds"] for r in counted]
        point = percentile(values, quantile)
        upper = percentile(bootstrap(values, lambda v: percentile(v, quantile), seed), .95) if values else None
        stats[name] = dict(seconds=point, upper95_s=upper, limit_s=limit)
    failures = sum(r["health"] in ("error", "timeout") or not r["fts_hits"] or not r["honest_coverage"] for r in rows)
    valid = len(counted) >= 200 and len(publications) >= 10
    passed = not failures and all(v["seconds"] <= v["limit_s"] and v["upper95_s"] <= v["limit_s"] for v in stats.values()) if counted else False
    return dict(state="INVALID" if not valid else "PASS" if passed else "FAIL", overlapping=len(counted),
                publications=len(publications), failures=failures, statistics=stats, seed=seed, resamples=10000)


def worker(request):
    a, mod, engine, bind = runtime(request)
    manifest = load(a.out/"manifest.json")
    pinned = load(a.out/"queries.json")
    queries = {q["id"]: manifest["canaries"][q["canary"]]["value"] for q in pinned}
    rows, publications, errors = [], [], []
    halt, started = threading.Event(), threading.Event()
    reader = None
    try:
        if a.state == "ON-covered":
            wait_covered(a, mod, engine, manifest)
        initial = audit(a.db, manifest)
        dump(a.root/"initial-audit.json", initial)
        original = engine._dag.add_node
        def publish(node, **kw):
            start = time.perf_counter()
            result = original(node, **kw)  # returns only after SQLite commit
            if node.depth == 0 and kw.get("before_commit") is not None:
                row = dict(start=start, end=time.perf_counter(), node_id=result)
                publications.append(row)
                append(a.root/"publications.jsonl", row)
            return result
        engine._dag.add_node = publish
        def recall_loop():
            nonlocal reader
            try:
                # Separate SQLite connections owned by the recall thread, as scale389 does.
                from types import SimpleNamespace
                reader = SimpleNamespace(_config=engine._config, _hermes_home=str(a.home),
                    _store=mod("store").MessageStore(a.db), _dag=mod("dag").SummaryDAG(a.db),
                    current_session_id="i2-recall")
                started.set()
                i = 0
                while not halt.is_set() or i < len(pinned):
                    q = pinned[i % len(pinned)]
                    incomplete = not progress(a.db, manifest)["complete"] if a.state != "OFF" else False
                    start = time.perf_counter()
                    payload = json.loads(mod("tools").lcm_recall({"query": queries[q["id"]]}, engine=reader))
                    end = time.perf_counter()
                    coverage = payload.get("provenance", {}).get("coverage", {})
                    health = "timeout" if payload.get("timeout") else "error" if "error" in payload else "degraded" if payload.get("degraded") else "ok"
                    row = dict(query_id=q["id"], start=start, end=end, seconds=end-start, health=health,
                        reason=payload.get("degraded_reason"), semantic_coverage=coverage,
                        fts_hits=any("fts" in h.get("arms", []) for h in payload.get("hits", [])),
                        honest_coverage=not incomplete or bool(payload.get("degraded") and payload.get("degraded_reason")),
                        incomplete=incomplete)
                    rows.append(row)
                    append(a.root/"recalls.jsonl", row)
                    i += 1
                    if i >= len(pinned) and len(publications) >= 10 and len(overlapping(rows, publications)) >= 200:
                        break
            except Exception as exc:
                errors.append(type(exc).__name__)
                started.set()
            finally:
                if reader:
                    reader._dag.close()
                    reader._store.close()
        thread = threading.Thread(target=recall_loop, name="i2-recall")
        thread.start()
        if not started.wait(30):
            raise RuntimeError("recall reader start timeout")
        trace(engine, mod, 32 if a.smoke else max(600, a.queries*3), a.seed+a.block,
              lambda row: append(a.root/"turns.jsonl", row), lambda: safety(a))
        halt.set()
        thread.join(engine._config.recall_query_timeout_s*len(pinned)+30)
        if thread.is_alive() or errors:
            raise RuntimeError(f"recall worker failure: {errors}")
        integrity(a, manifest)
        result = score(rows, publications, a.seed)
        if a.smoke or a.rows < 250000 or len(pinned) < 200 or (a.state == "ON-catching-up" and not all(r["incomplete"] for r in overlapping(rows, publications))):
            result["state"] = "INVALID"
        result.update(bind_ms=bind, safety=safety(a), queries=len(pinned))
        dump(a.root/"raw-summary.json", result)
    finally:
        halt.set()
        engine.shutdown()


def main(a):
    manifest = prepare(a)
    order = list(range(len(manifest["canaries"])))
    random.Random(a.seed).shuffle(order)
    pinned = [{"id": f"q{i:04d}", "canary": order[i % len(order)]} for i in range(a.queries)]
    # IDs only: query text derives from the pinned corpus manifest, never raw transcripts.
    dump(a.out/"queries.json", pinned)
    results = [dict(block=b, state=s, result=cell(a, s, b, "recall"))
               for b in range(a.blocks) for s in ("OFF", "ON-catching-up", "ON-covered")]
    states = [r["result"]["state"] for r in results]
    dump(a.out/"summary.json", dict(state="INVALID" if "INVALID" in states else "FAIL" if "FAIL" in states else "PASS",
        label=SMOKE if a.smoke else a.label, blocks=results, seed=a.seed))


if __name__ == "__main__":
    a = arguments(__doc__)
    worker(a.worker) if a.worker else main(a)
