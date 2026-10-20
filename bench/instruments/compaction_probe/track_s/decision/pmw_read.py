#!/usr/bin/env python3
"""Read Track S summaries, engine logs, checkpoint scores and material turns.

The six pooled bars are C/100 <= 25, mean per-message wait <= 15 s,
waited share <= 25%, cp-304 facts kept >= the declared floor (default 0.583),
dropped rows = 0, and nearest-rank compaction wall p90 <= 120 s.
Exit 0: PASS; 1: FAIL; 2: INVALID (missing/incomplete run or config mismatch).
cp-176 is reported only: under the fleet policy it is pre-crossing, a reader A/A (#1128)

First crossing is the first event with tokens_at_trigger >= threshold_tokens;
post-crossing material turns run from that turn through the stop row. Sanitized
preflight events remain below the crossing. No model calls are made.
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path


def p90(xs):
    xs = sorted(xs)
    return xs[max(0, math.ceil(0.9 * len(xs)) - 1)] if xs else None


def med(xs):
    return statistics.median(xs) if xs else None


def mean(xs):
    return statistics.mean(xs) if xs else None


_turns = {}


def material_turns(material, n, stop):
    key = (material, n, stop)
    if key not in _turns:
        rows = [json.loads(line) for line in (material / f"seed-{n}" / "transcript.jsonl").read_text().splitlines() if line.strip()]
        _turns[key] = sorted({r["turn"] for r in rows[:stop + 1]})
    return _turns[key]


def run_metrics(run_root, material, arm, n, r):
    d = run_root / "lcmx-runs" / arm / f"seed-{n}" / f"d{n}-r{r}"
    summ = d / "summary.json"
    if not summ.exists():
        return None
    s = json.loads(summ.read_text())
    if s.get("status") != "DONE":
        return {"status": s.get("status")}
    ev = s.get("events") or []
    thr = s["threshold_tokens"]
    cross = next((e for e in ev if e["tokens_at_trigger"] >= thr), None)
    if cross is None:
        raise ValueError(f"{arm}|d{n}-r{r}: no threshold crossing")
    turns = material_turns(material, n, s["stop_row_index"])
    post = [t for t in turns if cross and t >= cross["turn"]]
    if not post:
        raise ValueError(f"{arm}|d{n}-r{r}: zero post-crossing turns")
    wait = {t: 0.0 for t in post}
    long_waits = {t: 0 for t in post}
    c_events, leaves, freed, walls, lat, timeouts, fails, zero_leaf = 0, [], [], [], [], 0, 0, 0
    src = []
    for e in ev:
        for c in e.get("summariser_calls") or []:
            if c.get("exception"):
                fails += 1
                if "timeout" in str(c.get("error", "")).lower():
                    timeouts += 1
            elif c.get("latency_s") is not None:
                lat.append(float(c["latency_s"]))
        for lv in e.get("levels") or []:
            if lv.get("depth") == 0 and lv.get("source_tokens") is not None:
                src.append(int(lv["source_tokens"]))
        if not cross or e["turn"] < cross["turn"]:
            continue
        w = float(e.get("compress_wall_s") or 0)
        if e["turn"] in wait:
            wait[e["turn"]] += w
            long_waits[e["turn"]] += int(w >= 100.0)
        if w >= 1.0:
            c_events += 1
            walls.append(w)
            leaves.append(int(e.get("leaves") or 0))
            freed.append(int(e["tokens_at_trigger"]) - int(e["tokens_after"]))
            if not e.get("leaves"):
                zero_leaf += 1
    eng = (d / "engine.log").read_text(errors="replace") if (d / "engine.log").exists() else ""
    holds = len(re.findall(r"automatic compaction held for \d+s: no_progress", eng))
    fits = re.findall(r"survival fit applied \(reason=[^,]*, conversation=[^,]*, dropped_rows=(\d+)", eng)
    env = bool(re.search(r"HTTP 429|HTTP Error 429|usage limit|too many requests", eng, re.I))
    facts = {}
    for cp in ("176", "304"):
        f = run_root / "decision" / "scores" / f"cp-{cp}" / f"{arm}.seed-{n}.d{n}-r{r}.json"
        if f.exists():
            fk = (json.loads(f.read_text()).get("metrics") or {}).get("facts_kept") or {}
            if cp == "304" and fk.get("complete") is False:
                raise ValueError(f"{arm}|d{n}-r{r}: incomplete cp-304 facts_kept score")
            facts[cp] = fk.get("value")
    return {"status": "DONE", "crossing_turn": cross["turn"] if cross else None, "post_turns": len(post),
            "c_events": c_events, "c_per_100": (100.0 * c_events / len(post)) if post else None,
            "mean_turn_wait_s": mean(list(wait.values())), "waited_turn_share": (sum(1 for v in wait.values() if v >= 1.0) / len(post)) if post else None,
            "leaves_per_event": mean(leaves), "freed_per_event": mean(freed), "leaf_source_tokens_median": med(src),
            "call_p50": med(lat), "call_p90": p90(lat), "_lat": lat, "_walls": walls,
            "wall_p50": med(walls), "wall_p90": p90(walls), "wall_max": max(walls) if walls else None,
            "timeouts": timeouts, "failures": fails, "zero_leaf_events": zero_leaf, "no_progress_holds": holds,
            "survival_fits": len(fits), "dropped_rows": sum(int(a) for a in fits), "env": env,
            "facts_kept": facts, "replay_wall_s": s.get("replay_wall_s"),
            "long_waits": sum(long_waits.values()), "max_long_waits_in_one_turn": max(long_waits.values(), default=0),
            "max_turn_wait_s": max(wait.values(), default=0.0),
            "faulted_calls": (s.get("fault_hang") or {}).get("faulted_calls", 0),
            "fault_mode": (s.get("fault_hang") or {}).get("mode")}



def pooled(out, a, runs):
    rs = [out["per_run"][f"{a}|d{n}-r{r}"] for n, r in runs]
    rs = [x for x in rs if x and x.get("status") == "DONE" and not x["env"]]
    if not rs:
        return {"runs": 0}
    pt = sum(x["post_turns"] for x in rs)
    ce = sum(x["c_events"] for x in rs)
    walls = [w for x in rs for w in x["_walls"]]
    lat = [v for x in rs for v in x["_lat"]]
    f304 = [x["facts_kept"].get("304") for x in rs if x["facts_kept"].get("304") is not None]
    f176 = [x["facts_kept"].get("176") for x in rs if x["facts_kept"].get("176") is not None]
    modes = {x["fault_mode"] for x in rs}
    return {"runs": len(rs), "post_turns": pt, "c_per_100": 100.0 * ce / pt if pt else None,
            "mean_turn_wait_s": mean([x["mean_turn_wait_s"] for x in rs if x["mean_turn_wait_s"] is not None]),
            "waited_turn_share": mean([x["waited_turn_share"] for x in rs if x["waited_turn_share"] is not None]),
            "leaves_per_event": mean([x["leaves_per_event"] for x in rs if x["leaves_per_event"] is not None]),
            "freed_per_event": mean([x["freed_per_event"] for x in rs if x["freed_per_event"] is not None]),
            "call_p50": med(lat), "call_p90": p90(lat), "wall_p50": med(walls), "wall_p90": p90(walls), "wall_max": max(walls) if walls else None,
            "timeouts": sum(x["timeouts"] for x in rs), "failures": sum(x["failures"] for x in rs),
            "zero_leaf_events": sum(x["zero_leaf_events"] for x in rs), "no_progress_holds": sum(x["no_progress_holds"] for x in rs),
            "survival_fits": sum(x["survival_fits"] for x in rs), "dropped_rows": sum(x["dropped_rows"] for x in rs),
            "facts_304_mean": mean(f304), "facts_176_mean": mean(f176), "facts_n": (len(f176), len(f304)),
            "long_waits": sum(x["long_waits"] for x in rs),
            "max_long_waits_in_one_turn": max(x["max_long_waits_in_one_turn"] for x in rs),
            "max_turn_wait_s": max(x["max_turn_wait_s"] for x in rs),
            "faulted_calls": sum(x["faulted_calls"] for x in rs),
            "fault_mode": next(iter(modes)) if len(modes) == 1 else "mixed"}


def find(d, k):
    if isinstance(d, dict):
        for kk, v in d.items():
            if kk == k:
                return v
            r = find(v, k)
            if r is not None:
                return r
    return None


def read(args):
    arm = args.arm
    runs = [(n, 1) for n in args.seeds]
    out = {"per_run": {}, "pooled": {}}
    for n, r in runs:
        out["per_run"][f"{arm}|d{n}-r{r}"] = run_metrics(args.run_root, args.material, arm, n, r)
    x = out["pooled"][arm] = pooled(out, arm, runs)
    want = {"leaf_chunk_tokens": args.leaf_chunk_tokens,
            "leaf_target_max_tokens": args.leaf_target_max_tokens,
            "summary_timeout_ms": args.summary_timeout_ms}
    cfg = {}
    for n, r in runs:
        path = args.run_root / "lcmx-runs" / arm / f"seed-{n}" / f"d{n}-r{r}" / "config.json"
        config = json.loads(path.read_text()) if path.exists() else {}
        cfg[f"d{n}-r{r}"] = {k: find(config, k) for k in want}
    valid = all(v == want for v in cfg.values())
    bars = {
        "1_c100_le_25": x.get("c_per_100") is not None and x["c_per_100"] <= 25,
        "2_wait_le_15s": x.get("mean_turn_wait_s") is not None and x["mean_turn_wait_s"] <= 15,
        "3_share_le_25pct": x.get("waited_turn_share") is not None and x["waited_turn_share"] <= 0.25,
        # Keep the historical key for receipt compatibility; the declared floor is configurable.
        "4_facts304_ge_583": None if x.get("facts_304_mean") is None else x["facts_304_mean"] >= args.facts_floor,
        "5_dropped_eq_0": x.get("dropped_rows") == 0,
        "6_wall_p90_le_120": x.get("wall_p90") is not None and x["wall_p90"] <= 120.0,
    }
    complete = x.get("runs") == len(runs) and (x.get("facts_n") or (0, 0))[1] == len(runs)
    out["config_check"] = {"per_run": cfg, "valid": valid}
    out["verdict"] = {"bars": bars, "complete": complete, "facts_floor": args.facts_floor,
                      "result": "INVALID" if not valid or not complete else "PASS" if all(bars.values()) else "FAIL"}
    for metrics in out["per_run"].values():
        if metrics:
            metrics.pop("_lat", None)
            metrics.pop("_walls", None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True, help="contains lcmx-runs/ and decision/scores/")
    ap.add_argument("--material", type=Path, required=True, help="Track S material directory")
    ap.add_argument("--arm", default="LCMX-fleet-rc3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--json", type=Path, required=True, metavar="OUT")
    ap.add_argument("--facts-floor", type=float, default=0.583)
    ap.add_argument("--leaf-chunk-tokens", type=int, default=24000)
    ap.add_argument("--leaf-target-max-tokens", type=int, default=2000)
    ap.add_argument("--summary-timeout-ms", type=int, default=120000)
    args = ap.parse_args(argv)
    try:
        out = read(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out = {"verdict": {"result": "INVALID", "facts_floor": args.facts_floor}, "error": str(exc)}
    args.json.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "per_run"}, indent=1, default=str))
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[out["verdict"]["result"]]


if __name__ == "__main__":
    raise SystemExit(main())
