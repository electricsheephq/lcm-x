"""Bar 3: real recall intervals overlapping committed compaction publications."""
import json
from bisect import bisect_right
import random
import re
import threading
import time

from .audit import audit, vector_counts
from .common import REFERENCE_CALIBRATION_S, append, dump, foreground_request, load, rollup
from .floor import calibration
from .cpu import percentile
from .measurement import SMOKE, arguments, bootstrap, catchup_activity, cell, guard_reason, guarded_worker, integrity, prepare, runtime, safety, trace, wait_covered

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


class OverlapCounter:
    """Serial recall/commit intervals; visit only new intervals and their overlaps."""
    def __init__(self, state):
        self.state, self.rows, self.ends, self.pubs, self.pub_ends = state, [], [], [], []
        self.counted = {c: set() for c in ("canary", "broad_nl")}
        self.publications = {c: set() for c in self.counted}

    def match(self, i, j):
        row, pub = self.rows[i], self.pubs[j]
        if (row["start"] < pub["end"] and pub["start"] < row["end"] and not
                (self.state == "ON-catching-up" and row["semantic_healthy"] and row["semantic_empty"])):
            self.counted[row["query_class"]].add(i)
            self.publications[row["query_class"]].add(pub.get("node_id", j))

    def add(self, row, publications):
        self.rows.append(row)
        self.ends.append(row["end"])
        for j in range(bisect_right(self.pub_ends, row["start"]), len(self.pubs)):
            if self.pubs[j]["start"] >= row["end"]:
                break
            self.match(len(self.rows)-1, j)
        for pub in publications[len(self.pubs):]:
            self.pubs.append(pub)
            self.pub_ends.append(pub["end"])
            for i in range(bisect_right(self.ends, pub["start"]), len(self.rows)):
                if self.rows[i]["start"] >= pub["end"]:
                    break
                self.match(i, len(self.pubs)-1)


def window_calibration(events):
    starts = [e["at"] for e in events if e.get("event") == "calibration_window_start"]
    ends = [e["at"] for e in events if e.get("event") == "calibration_window_end"]
    return calibration([e for e in events if starts and ends and
        starts[0] <= e.get("at", -1) < e.get("end", -1) <= ends[0]])


def calibration_window(a):
    path, start = a.root/"events.jsonl", time.perf_counter()
    append(path, dict(event="calibration_window_start", at=start))
    while time.perf_counter()-start < 600:
        events = [json.loads(s) for s in path.read_text().splitlines()]
        if calibration([e for e in events if e.get("at", -1) >= start]) is not None:
            break
        safety(a)
        time.sleep(.1)
    end = time.perf_counter()
    append(path, dict(event="calibration_window_end", at=min(end, start+600)))
    # This unloaded window consumes the ON-catching-up state's fixed 100-minute budget.
    a.volume_seconds = max(0, getattr(a, "volume_seconds", VOLUME_SECONDS)-(end-start))


def score(rows, publications, seed=1586, state="OFF", activity=None, calibration_s=None, reference_calibration_s=REFERENCE_CALIBRATION_S):
    if not any("query_class" in r for r in rows):
        return score_class(rows, publications, seed, state, activity)
    ratio = reference_calibration_s/calibration_s if reference_calibration_s and calibration_s else None
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
    return dict(state=rollup([*states, host]),
        classes=classes, calibration_s=calibration_s, reference_calibration_s=reference_calibration_s,
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
    result = dict(state="FAIL" if failures else "INVALID" if not valid else "PASS" if passed else "FAIL", overlapping=len(counted),
                publications=len(publications), overlapping_publications=distinct,
                failures=failures, statistics=stats, seed=seed, resamples=10000)
    if not failures and state == "ON-catching-up" and not (activity or {}).get("catchup_activity"):
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


# A 330-minute block reserves 30 minutes for corpus/setup and 100 per state.
# Of each state, 95 minutes collect volume and 5 allow audits and shutdown.
STATE_SECONDS = 100*60
VOLUME_SECONDS = 95*60


def publication_trace(a, engine, mod, enough, halt):
    deadline = getattr(a, "_volume_deadline", None) or time.monotonic()+getattr(a, "volume_seconds", VOLUME_SECONDS)
    turns = 0
    def observe(row):
        nonlocal turns
        turns += 1
        append(a.root/"turns.jsonl", row)
    expected = trace(engine, mod, 32 if a.smoke else None, a.seed+a.block,
        observe, lambda: safety(a),
        lambda: enough.is_set() or halt.is_set() or time.monotonic() >= deadline)
    return expected, turns, not enough.is_set() and time.monotonic() >= deadline


def worker(request):
    a, mod, engine, bind = runtime(request)
    manifest = a._manifest = load(a.root/"manifest.json" if (a.root/"manifest.json").exists() else a.out/"manifest.json")
    pinned = load(a.out/"queries.json")
    queries = {q["id"]: q["text"] if q.get("query_class") == "broad_nl" else manifest["canaries"][q["canary"]]["value"] for q in pinned}
    rows, publications, errors = [], [], []
    halt, started = threading.Event(), threading.Event()
    enough = threading.Event()
    reader, original_fts = None, None
    closed = False
    try:
        if a.state == "ON-covered":
            try:
                wait_covered(a, mod, engine, manifest)
            except RuntimeError as exc:
                if str(exc) != "covered_store_incomplete":
                    raise
                append(a.root/"events.jsonl", dict(violation="covered_store_incomplete"))
                dump(a.root/"raw-summary.json", dict(state="INVALID", reason=str(exc)))
                return
        initial = audit(a.db, manifest)
        dump(a.root/"initial-audit.json", initial)
        if a.state == "ON-catching-up":
            calibration_window(a)
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
                counter = OverlapCounter(a.state)
                deadline = time.monotonic()+getattr(a, "volume_seconds", VOLUME_SECONDS)
                a._volume_deadline = deadline
                while time.monotonic() < deadline and (not halt.is_set() or i < len(pinned)):
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
                    counter.add(row, publications)
                    if i >= len(pinned) and all(len(counter.counted[c]) >= 200
                            and len(counter.publications[c]) >= 10 for c in counter.counted):
                        enough.set()
                        break
            except Exception as exc:
                errors.append(type(exc).__name__)
                halt.set()
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
        manifest["trace_rows"], turns_used, capped = publication_trace(a, engine, mod, enough, halt)
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
        result = score(rows, publications, a.seed, a.state, activity)
        if a.smoke or a.rows < 250000 or len(pinned) < 200 or (a.state == "ON-catching-up" and
                (all(final["classes"][kind]["missing_count"] == 0 for kind in manifest["expected"])
                 or not all(r["incomplete"] for r in overlapping(rows, publications)))):
            result["state"] = rollup([result["state"], "INVALID"])
        capped = capped and (not rows or any(c["overlapping"] < 200 or c["overlapping_publications"] < 10 for c in result["classes"].values()))
        if capped:
            append(a.root/"events.jsonl", dict(event="volume_cap"))
            result.update(state=rollup([result["state"], "INVALID"]), reason="volume_cap")
        result.update(turns_used=turns_used, measurement_valid=result["state"] != "INVALID", bind_ms=bind, safety=evidence, queries=len(pinned), **activity)
        dump(a.root/"raw-summary.json", result)
    finally:
        halt.set()
        if not closed:
            engine.shutdown()


def guarded_result(root, recorded):
    """Keep a guard denial INVALID unless its existing receipts prove FAIL."""
    def lines(path):
        return [json.loads(s) for s in path.read_text().splitlines()] if path.is_file() else []
    reason = recorded.get("reason")
    if not reason or guard_reason(lines(root/"events.jsonl")) != reason:
        return None
    rows = lines(root/"recalls.jsonl")
    if not rows:
        return recorded
    request = load(root/"request.json")
    pubs = lines(root/"publications.jsonl")
    counted = overlapping(rows, pubs)
    activity = catchup_activity(root, min(r["start"] for r in counted), max(r["end"] for r in counted),
        min(counted, key=lambda r: r["start"])["coverage_start"],
        max(counted, key=lambda r: r["end"])["coverage_end"]) if counted else dict(catchup_activity=False)
    calibrated = window_calibration(lines(root.parent/f"{request['block']}-ON-catching-up/events.jsonl"))
    computed = score(rows, pubs, request["seed"], request["state"], activity, calibrated)
    return dict(recorded, state=rollup([computed["state"], "INVALID"]))


def finalize_block(a, block, results):
    root = a.out/f"{block}-ON-catching-up"
    def lines(path):
        return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    calibrated = window_calibration(lines(root/"events.jsonl"))
    for entry in results:
        result = entry["result"]
        root = a.out/f"{block}-{entry['state']}"
        if (root/"request.json").is_file():
            request = load(root/"request.json")
            assert entry["block"] == block and (request["block"], request["state"]) == (block, entry["state"]), "request identity mismatch"
        if "classes" not in result:
            guarded = guarded_result(root, result)
            if guarded is not None:
                result.update(guarded)
                for name in ("raw-summary.json", "summary.json"):
                    dump(root/name, dict(load(root/name), **result))
            continue
        scored = score(lines(root/"recalls.jsonl"), lines(root/"publications.jsonl"),
            a.seed, entry["state"], result, calibrated)
        if result.get("measurement_valid") is False:
            scored["state"] = rollup([scored["state"], "INVALID"])
        if calibrated is None and result.get("measurement_valid") is not False:
            scored["reason"] = "calibration_window_short"
        parent_state = result["state"]
        # An uncalibrated child is provisional until this block's window is applied.
        if parent_state == "UNCALIBRATED" and result.get("calibration_s") is None and not result.get("reason"):
            parent_state = rollup(c["state"] for c in result["classes"].values())
        scored["state"] = rollup([scored["state"], parent_state, "INVALID" if result.get("stop_killed") else "PASS"])
        if result.get("stop_killed"):
            scored["reason"] = "forced_stop_after_grace"
        result.update(scored, calibration_s=calibrated,
            standalone_calibration_s=a.standalone_calibration_s)
        for name in ("raw-summary.json", "summary.json"):
            dump(root/name, dict(load(root/name), **result))


def main(a):
    if getattr(a, "build_covered", False):
        from .calibration import covered
        return covered(a)
    manifest = prepare(a)
    from .calibration import standalone
    if getattr(a, "standalone_calibration_s", None) is None:
        a.standalone_calibration_s = standalone(a)["standalone_calibration_s"]
    dump(a.out/"calibration.json", dict(standalone_calibration_s=a.standalone_calibration_s, reference_calibration_s=REFERENCE_CALIBRATION_S))
    a.max_seconds = min(getattr(a, "max_seconds", STATE_SECONDS), STATE_SECONDS)
    a.volume_seconds = min(VOLUME_SECONDS, max(0, a.max_seconds-300))
    order = list(range(len(manifest["canaries"])))
    random.Random(a.seed).shuffle(order)
    pinned = [q for i in range(a.queries) for q in (
        {"id": f"q{i:04d}", "canary": order[i % len(order)], "query_class": "canary"},
        {"id": f"b{i:04d}", "text": BROAD_NL[i % 3], "query_class": "broad_nl"})]
    # IDs only: query text derives from the pinned corpus manifest, never raw transcripts.
    dump(a.out/"queries.json", pinned)
    results = []
    for b in ([a.block] if getattr(a, "block", None) is not None else range(a.blocks)):
        block = []
        try:
            for s in ("OFF", "ON-catching-up", "ON-covered"):
                block.append(dict(block=b, state=s, result=cell(a, s, b, "recall")))
        finally:
            finalize_block(a, b, block)
        results.extend(block)
    states = [r["result"]["state"] for r in results]
    dump(a.out/"summary.json", dict(state=rollup(states),
        label=SMOKE if a.smoke else a.label, blocks=results, seed=a.seed, shape=a.shape, baseline=a.baseline))


if __name__ == "__main__":
    a = arguments(__doc__)
    guarded_worker(worker, a.worker) if a.worker else main(a)
