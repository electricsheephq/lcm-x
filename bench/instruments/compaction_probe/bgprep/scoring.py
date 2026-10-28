"""Pure, snapshot-only scoring for background-prep instrument v1."""
from collections import Counter
import math


def coverage(probe):
    """Keep multiplicity: sets alone would hide duplicate storage or leaf coverage."""
    rows = probe["rows"]
    expected = Counter(probe["expected"])
    stored = Counter(r["identity"] for r in rows)
    ids = Counter(r["store_id"] for r in rows)
    leaf_ids = [i for leaf in probe["leaf_sources"] for i in leaf]
    covered = Counter(leaf_ids)
    tail_rows = [r for r in sorted(rows, key=lambda r: r["store_id"])
                 if r["store_id"] > probe["frontier"] and not covered[r["store_id"]]]
    tail_identities = [r.get("active_identity", r["identity"]) for r in tail_rows]
    non_tail_identities = {r.get("active_identity", r["identity"]) for r in rows
                           if r["store_id"] <= probe["frontier"] or covered[r["store_id"]]}
    tail_counts = Counter(tail_identities)
    live_tail = [identity for identity in probe["active_view"] if identity in tail_counts]
    live = Counter(probe["active_view"])
    tail = []
    for row in tail_rows:
        # Active projections/carrier remainders may omit timestamps; storage identity may not.
        active_identity = row.get("active_identity", row["identity"])
        if live[active_identity]:
            tail.append(row["store_id"])
            live[active_identity] -= 1
    lost = [r["store_id"] for r in rows if r["store_id"] not in tail and not covered[r["store_id"]]]
    return {
        "lost": len(lost), "lost_ids": lost, "active_tail_ids": tail,
        "missing_rows": sum((expected - stored).values()),
        "duplicate_stored": sum((stored - expected).values()),
        "duplicate_ids": sum(n - 1 for n in ids.values() if n > 1),
        "duplicate_coverage": sum(n - 1 for n in covered.values() if n > 1),
        "duplicate_tail": sum((Counter(live_tail) - tail_counts).values()),
        "covered_in_active": sum(live[identity] for identity in non_tail_identities),
        "tail_order_preserved": live_tail == tail_identities,
        "leaf_order_preserved": all(a < b for a, b in zip(leaf_ids, leaf_ids[1:])),
        "dangling_sources": sorted(set(covered) - set(ids)),
        "ordered_identity_preserved": [r["identity"] for r in rows] == probe["expected"],
    }


def intact(score):
    return all(score[k] for k in (
        "ordered_identity_preserved", "tail_order_preserved", "leaf_order_preserved")) and not any(score[k] for k in (
        "lost", "missing_rows", "duplicate_stored", "duplicate_ids", "duplicate_coverage", "duplicate_tail",
        "covered_in_active", "dangling_sources"))


def promotion(events):
    crossings = [e for e in events if e["crossing"]]
    numerator = sum(e["promoted"] for e in crossings)
    denominator = sum(e["leaf_compactions"] for e in crossings)
    if not 0 <= numerator <= denominator:
        raise ValueError("promotions must be a subset of threshold-crossing leaf operations")
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def at_least(candidate, reference):
    """Exact pooled fractions, never a mean of seed rates or rounded floats."""
    if not candidate["denominator"] or not reference["denominator"]:
        return False
    return candidate["numerator"] * reference["denominator"] >= reference["numerator"] * candidate["denominator"]


def waits(events):
    cross = next((e["turn"] for e in events if e["crossing"]), None)
    values = [e["visible_wait_s"] for e in events if cross is not None and e["turn"] >= cross]
    waited = sum(v >= 1.0 for v in values)
    return {"p90_s": sorted(values)[math.ceil(.9 * len(values)) - 1] if values else None,
            "turns": len(values), "waited_turns": waited,
            "waited_turn_share": waited / len(values) if values else None,
            "waited_definition": ">=1 second; post-first-crossing turns, including zero waits"}


def score_cell(raw):
    scores = [coverage(p) for p in raw["probes"]]
    activity = raw["probes"][-1]["activity"]
    off_quiet = all(not any(p["activity"][k] for k in (
        "worker_starts", "worker_constructions", "enqueue_attempts", "jobs_enqueued",
        "worker_summary_calls", "promotions", "promotion_lookups")) for p in raw["probes"])
    rate = promotion(raw["events"])
    return {"integrity": bool(scores) and all(map(intact, scores)), "coverage": scores,
            "off_quiet": off_quiet, "activity": activity, "promotion": rate,
            "wait": waits(raw["events"]), "crossings": sum(e["crossing"] for e in raw["events"]),
            "positive_control": all(activity[k] > 0 for k in (
                "worker_starts", "worker_constructions", "enqueue_attempts", "jobs_enqueued",
                "worker_summary_calls", "promotions", "promotion_lookups")) and rate["denominator"] > 0,
            "facts_kept": {"value": None, "numerator": None, "denominator": None,
                           "reason": "synthetic workload, no Track S checkpoint material or reader"}}
