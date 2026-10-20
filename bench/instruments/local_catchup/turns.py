"""Bar 2: paired block bootstrap of host turn latency during LOCAL catch-up."""
import time

from .common import append, dump, load
from .cpu import percentile
from .measurement import SMOKE, arguments, bootstrap, catchup_activity, cell, guarded_worker, integrity, prepare, runtime, safety, trace


def paired_bar(pairs, seed=1586):
    """Resample matched blocks, never independent turns; inputs are (OFF, ON) p90 ms."""
    off = sum(x for x, _ in pairs)/len(pairs)
    delta = sum(y-x for x, y in pairs)/len(pairs)
    upper = percentile(bootstrap(pairs, lambda rows: sum(y-x for x, y in rows)/len(rows), seed), .95)
    limit = max(50., .1*off)
    return dict(off_p90_ms=off, delta_ms=delta, upper95_ms=upper, limit_ms=limit,
                state="PASS" if upper <= limit else "FAIL", resamples=10000, seed=seed)


def worker(request):
    a, mod, engine, bind = runtime(request)
    manifest = load(a.out/"manifest.json")
    rows, coverage = [], []
    try:
        def observe(row):
            rows.append(row)
            append(a.root/"turns.jsonl", row)
        from .audit import progress
        start = time.perf_counter()
        initial = progress(a.db, manifest, False)
        coverage.append(initial["complete"])
        def check():
            safety(a)
            coverage.append(progress(a.db, manifest, False)["complete"])
        trace(engine, mod, a.samples, a.seed+a.block, observe, check)
        final = progress(a.db, manifest, False)
        coverage.append(final["complete"])
        end = time.perf_counter()
        activity = catchup_activity(a.root, start, end,
            {k: v["covered"] for k, v in initial["classes"].items()},
            {k: v["covered"] for k, v in final["classes"].items()}, require_prepared=True)
        integrity(a, manifest)
        evidence = safety(a)
        summary = dict(state="NOT_JUDGED", bind_ms=bind, samples=len(rows),
            p90={k: percentile([r[k] for r in rows], .9) for k in ("ingest_preflight_ms", "dispatch_ms")},
            compression_turns=sum(r["compression_due"] for r in rows), join_evidence=evidence,
            catching_up=not any(coverage) if a.state == "ON" else None, **activity)
        if a.state == "ON" and not activity["catchup_activity"]:
            summary.update(state="INVALID", reason="on_no_catchup_activity")
        dump(a.root/"raw-summary.json", summary)
    finally:
        engine.shutdown()


def main(a):
    prepare(a)
    import random
    order = ["OFF", "ON"] if random.Random(a.seed).randrange(2) == 0 else ["ON", "OFF"]
    results = []
    dump(a.out/"order.json", dict(seed=a.seed, order=order*a.blocks))
    for block in range(a.blocks):
        results.append({state: cell(a, state, block, "turns") for state in order})
    if any(r.get("state") == "INVALID" for block in results for r in block.values()):
        dump(a.out/"summary.json", dict(state="INVALID", reason="cell_invalid", blocks=results,
            label=SMOKE if a.smoke else a.label, seed=a.seed, shape=a.shape, baseline=a.baseline))
        return
    bars = {k: paired_bar([(r["OFF"]["p90"][k], r["ON"]["p90"][k]) for r in results], a.seed)
            for k in ("ingest_preflight_ms", "dispatch_ms")}
    binds = [dict(block=i, added_ms=r["ON"]["bind_ms"]-r["OFF"]["bind_ms"]) for i, r in enumerate(results)]
    valid = a.samples >= 600 and a.blocks >= 2 and all(
        r["ON"]["catching_up"] and r["ON"]["catchup_activity"] and all(v["compression_turns"] for v in r.values()) for r in results)
    passed = all(v["state"] == "PASS" for v in bars.values()) and all(v["added_ms"] <= 500 for v in binds)
    state = "INVALID" if a.smoke or not valid else "PASS" if passed else "FAIL"
    if state == "INVALID":
        for bar in bars.values():
            bar["state"] = "NOT_JUDGED"
    dump(a.out/"summary.json", dict(state=state, label=SMOKE if a.smoke else a.label,
        shape=a.shape, baseline=a.baseline,
        blocks=results, order=order*a.blocks, seed=a.seed, bars=bars, first_bind=binds,
        reason="on_no_catchup_activity" if any(not r["ON"]["catchup_activity"] for r in results) else
            "on_coverage_completed_during_block" if any(not r["ON"]["catching_up"] for r in results) else
            "smoke_or_insufficient_samples_blocks_or_compaction" if state == "INVALID" else None,
        validity="600 samples/state/block, >=2 paired blocks, compaction, ON incomplete throughout with positive coverage delta and background preparation"))


if __name__ == "__main__":
    a = arguments(__doc__)
    guarded_worker(worker, a.worker) if a.worker else main(a)
