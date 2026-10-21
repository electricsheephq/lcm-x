"""Bar 3: real recall intervals overlapping committed compaction publications."""
import json
import random
import re
import threading
import time

from .audit import audit, vector_counts
from .common import append, dump, foreground_request, load
from .cpu import percentile
from .measurement import SMOKE, arguments, bootstrap, catchup_activity, cell, guarded_worker, integrity, prepare, runtime, safety, trace, wait_covered

INCOMPLETE_SEMANTIC_REASONS = (
    r"\b{arm} semantic index catching up\b",
    r"\b{arm} semantic index has \d+ rows awaiting /lcm embed backfill --apply --retry-uncertain --corpus {corpus}\b",
)
BROAD_NL = ("What turn was acknowledged", "Recall the filler from a turn", "Which turn contains a value")


def diagnostics(tools, reader):
    observed, original = {}, tools._lcm_recall_fts_arm
    def arm(engine, *args, **kw):
        if engine is not reader:
            return original(engine, *args, **kw)
        start = time.monotonic()
        observed["fts_budget_s"] = max(0., kw["deadline"]-start)
        try:
            hits, error = original(engine, *args, **kw)
            observed.update(fts_hit_count=len(hits), fts_cap_reason=(error or {}).get("error"))
            return hits, error
        finally:
            observed["fts_seconds"] = time.monotonic()-start
    tools._lcm_recall_fts_arm = arm
    return observed, original


def honest_coverage(payload, incomplete_arms):
    reason = payload.get("degraded_reason", "")
    coverage = payload.get("provenance", {}).get("coverage", {})
    if (payload.get("degraded") is True and re.search(r"\blocal embedding model is (preparing|unavailable)\b", reason)
            and all(coverage.get(arm) in (None, "none") for arm in ("chunk", "summary"))):
        return True  # Provider unavailability discloses coverage, not a legitimately empty index.
    return all(coverage.get(arm) in ("bounded", "full_approx", "none", "disabled") or
        (payload.get("degraded") and any(re.search(pattern.format(arm=arm,
            corpus="chunks" if arm == "chunk" else "summary"), reason)
            for pattern in INCOMPLETE_SEMANTIC_REASONS)) for arm in incomplete_arms)


def empty_arms(payload, incomplete_arms):
    if payload.get("degraded") is not True:
        return []
    coverage = payload.get("provenance", {}).get("coverage", {})
    reason = payload.get("degraded_reason", "")
    return [arm for arm in incomplete_arms if coverage.get(arm) == "none" and
        (re.search(rf"\b{arm} vectors are unavailable\b", reason) or any(
            re.search(pattern.format(arm=arm, corpus="chunks" if arm == "chunk" else "summary"), reason)
            for pattern in INCOMPLETE_SEMANTIC_REASONS))]


def semantic_healthy(payload, incomplete_arms=()):
    provenance = payload.get("provenance", {})
    reason = payload.get("degraded_reason", "")
    return (not payload.get("timeout") and not re.search(
        r"arm failed|provider unavailable|query embedding failed|cooling down|identity is stale|timeout|timed out|deadline exhausted|latency budget", reason)
        and all(arm in empty_arms(payload, incomplete_arms) or (arm in provenance.get("arms_run", []) and
            provenance.get("coverage", {}).get(arm) in ("full", "bounded", "full_approx"))
            for arm in ("chunk", "summary")))


def overlapping(rows, publications):
    return [r for r in rows if any(r["start"] < p["end"] and p["start"] < r["end"] for p in publications)]


def overlapping_publications(counted, publications):
    return len({p.get("node_id", i) for i, p in enumerate(publications) if overlapping([p], counted)})


def score(rows, publications, seed=1586, state="OFF", activity=None, calibration_s=None, reference_standalone_s=None):
    if not any("query_class" in r for r in rows):
        return score_class(rows, publications, seed, state, activity)
    ratio = reference_standalone_s/calibration_s if reference_standalone_s and calibration_s else None
    classes = {c: score_class([r for r in rows if r.get("query_class") == c], publications, seed, state, activity)
               for c in ("canary", "broad_nl")}
    if ratio and ratio > 1.25:
        for c, result in classes.items():
            scaled = score_class([dict(r, seconds=r["seconds"]*ratio) for r in rows if r.get("query_class") == c], publications, seed, state, activity)
            result["normalized_statistics"] = scaled["statistics"]
            if scaled["state"] == "FAIL":
                result["state"] = "FAIL"
    states = [r["state"] for r in classes.values()]
    host = "UNCALIBRATED" if ratio is None else "FAIL" if ratio > 1.25 and "FAIL" in states else "PASS"
    return dict(state="FAIL" if "FAIL" in states else "INVALID" if "INVALID" in states else host,
        classes=classes, standalone_calibration_s=calibration_s, reference_standalone_s=reference_standalone_s,
        host_check=host, latency_scale=ratio if ratio and ratio > 1.25 else 1.)


def score_class(rows, publications, seed=1586, state="OFF", activity=None):
    overlapped = overlapping(rows, publications)
    counted = [r for r in overlapped if not (state == "ON-catching-up" and
        r.get("semantic_healthy") and r.get("semantic_empty"))]
    distinct = overlapping_publications(counted, publications)
    stats = {}
    for name, quantile, limit in (("p50", .5, 2.), ("p90", .9, 8.)):
        values = [r["seconds"] for r in counted]
        point = percentile(values, quantile)
        upper = percentile(bootstrap(values, lambda v: percentile(v, quantile), seed), .95) if values else None
        stats[name] = dict(seconds=point, upper95_s=upper, limit_s=limit)
    failures = sum(r["health"] in ("error", "timeout") or not r["fts_hits"] or not r["honest_coverage"] or (state == "ON-catching-up" and r.get("fts_coverage") == "none") or (state != "OFF" and not r.get("semantic_healthy", False)) for r in rows)
    valid = len(counted) >= 200 and distinct >= 10
    passed = not failures and all(v["seconds"] <= v["limit_s"] and v["upper95_s"] <= v["limit_s"] for v in stats.values()) if counted else False
    result = dict(state="INVALID" if not valid else "PASS" if passed else "FAIL", overlapping=len(counted),
                publications=len(publications), overlapping_publications=distinct,
                failures=failures, statistics=stats, seed=seed, resamples=10000)
    if state == "ON-catching-up" and not (activity or {}).get("catchup_activity"):
        result.update(state="INVALID", reason="on_no_catchup_activity")
    return result


def recall_reader(engine, mod, db, home):
    from types import SimpleNamespace
    class Reader(SimpleNamespace):
        def __getattr__(self, name):
            if name.endswith("_provider_cache"):
                return getattr(engine, name)
            raise AttributeError(name)
        def __setattr__(self, name, value):
            if name.endswith("_provider_cache"):
                setattr(engine, name, value)
            else:
                super().__setattr__(name, value)
    return Reader(_config=engine._config, _hermes_home=str(home),
        _store=mod("store").MessageStore(db), _dag=mod("dag").SummaryDAG(db), current_session_id="i2-recall")


def worker(request):
    a, mod, engine, bind = runtime(request)
    manifest = a._manifest = load(a.out/"manifest.json")
    pinned = load(a.out/"queries.json")
    queries = {q["id"]: q["text"] if q.get("query_class") == "broad_nl" else manifest["canaries"][q["canary"]]["value"] for q in pinned}
    rows, publications, errors = [], [], []
    halt, started = threading.Event(), threading.Event()
    reader, original_fts = None, None
    closed = False
    try:
        if a.state == "ON-covered":
            try:
                wait_covered(a, mod, engine, manifest)
            except RuntimeError as exc:
                if str(exc) != "covered_store_incomplete":
                    raise
                dump(a.root/"raw-summary.json", dict(state="INVALID", reason=str(exc)))
                return
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
            nonlocal reader, original_fts
            try:
                # Separate SQLite connections owned by the recall thread, as scale389 does.
                reader = recall_reader(engine, mod, a.db, a.home)
                fts, original_fts = diagnostics(mod("tools"), reader)
                started.set()
                i = 0
                while not halt.is_set() or i < len(pinned):
                    q = pinned[i % len(pinned)]
                    before = vector_counts(a.db, manifest) if a.state != "OFF" else {}
                    incomplete_arms = [arm for kind, arm in (("chunks", "chunk"), ("leaves", "summary"))
                                       if before and before[kind] < len(manifest["expected"][kind])]
                    start = time.perf_counter()
                    fts.clear()
                    with foreground_request():
                        payload = json.loads(mod("tools").lcm_recall({"query": queries[q["id"]]}, engine=reader))
                    end = time.perf_counter()
                    after = vector_counts(a.db, manifest) if before else {}
                    honesty_arms = [arm for kind, arm in (("chunks", "chunk"), ("leaves", "summary"))
                                    if arm in incomplete_arms and after[kind] < len(manifest["expected"][kind])]
                    coverage = payload.get("provenance", {}).get("coverage", {})
                    health = "timeout" if payload.get("timeout") else "error" if "error" in payload else "degraded" if payload.get("degraded") else "ok"
                    row = dict(query_id=q["id"], query_class=q.get("query_class", "canary"), start=start, end=end, seconds=end-start, health=health,
                        **{k: payload.get("provenance", {}).get(k, fts.get(k, 0. if k == "fts_seconds" else 0 if k == "fts_hit_count" else None))
                           for k in ("fts_seconds", "fts_budget_s", "fts_hit_count", "fts_cap_reason")}, fts_coverage=coverage.get("fts"),
                        reason=payload.get("degraded_reasons", payload.get("degraded_reason")), semantic_coverage=coverage,
                        fts_hits=any("fts" in h.get("arms", []) for h in payload.get("hits", [])),
                        semantic_healthy=semantic_healthy(payload, incomplete_arms if a.state == "ON-catching-up" else ()),
                        semantic_empty=bool(empty_arms(payload, incomplete_arms)), honest_coverage=honest_coverage(payload, honesty_arms),
                        incomplete=bool(incomplete_arms) and any(after[k] < len(v) for k, v in manifest["expected"].items()),
                        coverage_start=before, coverage_end=after)
                    rows.append(row)
                    append(a.root/"recalls.jsonl", row)
                    i += 1
                    counted = [r for r in overlapping(rows, publications) if not (a.state == "ON-catching-up" and r["semantic_healthy"] and r["semantic_empty"])]
                    if i >= len(pinned) and all(len(part := [r for r in counted if r["query_class"] == c]) >= 200
                            and overlapping_publications(part, publications) >= 10 for c in ("canary", "broad_nl")):
                        break
            except Exception as exc:
                errors.append(type(exc).__name__)
                started.set()
            finally:
                if reader:
                    if original_fts is not None:
                        mod("tools")._lcm_recall_fts_arm = original_fts
                    reader._dag.close()
                    reader._store.close()
        thread = threading.Thread(target=recall_loop, name="i2-recall")
        thread.start()
        if not started.wait(30):
            raise RuntimeError("recall reader start timeout")
        manifest["trace_rows"] = trace(engine, mod, 32 if a.smoke else max(600, a.queries*3), a.seed+a.block,
              lambda row: append(a.root/"turns.jsonl", row), lambda: safety(a))
        halt.set()
        thread.join(engine._config.recall_query_timeout_s*len(pinned)+30)
        if thread.is_alive() or errors:
            raise RuntimeError(f"recall worker failure: {errors}")
        evidence = safety(a)
        engine.shutdown()
        closed = True
        integrity(a, manifest)
        final = audit(a.db, manifest)
        dump(a.root/"final-audit.json", final)
        counted = overlapping(rows, publications)
        activity = catchup_activity(a.root, min(r["start"] for r in counted), max(r["end"] for r in counted),
            min(counted, key=lambda r: r["start"])["coverage_start"],
            max(counted, key=lambda r: r["end"])["coverage_end"]) if counted else dict(catchup_activity=False)
        result = score(rows, publications, a.seed, a.state, activity, getattr(a, "standalone_calibration_s", None))
        if a.smoke or a.rows < 250000 or len(pinned) < 200 or (a.state == "ON-catching-up" and
                (all(final["classes"][kind]["missing_count"] == 0 for kind in manifest["expected"])
                 or not all(r["incomplete"] for r in overlapping(rows, publications)))):
            result["state"] = "INVALID"
        result.update(measurement_valid=result["state"] != "INVALID", bind_ms=bind, safety=evidence, queries=len(pinned), **activity)
        dump(a.root/"raw-summary.json", result)
    finally:
        halt.set()
        if not closed:
            engine.shutdown()


def main(a):
    if getattr(a, "build_covered", False):
        from .calibration import covered
        return covered(a)
    manifest = prepare(a)
    from .calibration import standalone
    if getattr(a, "standalone_calibration_s", None) is None:
        a.standalone_calibration_s = standalone(a)["standalone_calibration_s"]
    dump(a.out/"calibration.json", dict(standalone_calibration_s=a.standalone_calibration_s, reference_standalone_s=None))
    order = list(range(len(manifest["canaries"])))
    random.Random(a.seed).shuffle(order)
    pinned = [q for i in range(a.queries) for q in (
        {"id": f"q{i:04d}", "canary": order[i % len(order)], "query_class": "canary"},
        {"id": f"b{i:04d}", "text": BROAD_NL[i % 3], "query_class": "broad_nl"})]
    # IDs only: query text derives from the pinned corpus manifest, never raw transcripts.
    dump(a.out/"queries.json", pinned)
    results = [dict(block=b, state=s, result=cell(a, s, b, "recall"))
               for b in ([a.block] if getattr(a, "block", None) is not None else range(a.blocks)) for s in ("OFF", "ON-catching-up", "ON-covered")]
    states = [r["result"]["state"] for r in results]
    dump(a.out/"summary.json", dict(state="FAIL" if "FAIL" in states else "INVALID" if "INVALID" in states else "UNCALIBRATED" if "UNCALIBRATED" in states else "PASS",
        label=SMOKE if a.smoke else a.label, blocks=results, seed=a.seed, shape=a.shape, baseline=a.baseline))


if __name__ == "__main__":
    a = arguments(__doc__)
    guarded_worker(worker, a.worker) if a.worker else main(a)
