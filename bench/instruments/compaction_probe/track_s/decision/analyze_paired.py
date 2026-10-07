"""D8 analysis, model-free, from decision/scores + decision/loss + lcmx-runs summaries + load.log. Prints JSON only.
Pairs: same seed, same run index (r1 with r1, r2 with r2). Exact two-sided McNemar on discordant pairs per axis.
Axes: facts kept (all, head/middle/tail), constraint class (early_user_constraint facts), continuation fields,
strict continuity (grid items aligned by the event's row_index; both arms must have an event at that row),
active_constraint continuity items (same alignment). b = first arm kept & second arm lost, c = second arm kept & first arm lost.
The v1/v2 column names denote first/second CLI arms; sign-test wins favor the second arm.
Each run pair votes once per axis; ties are excluded from the exact two-sided sign p-value."""

import argparse
import json
import math
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

AP = argparse.ArgumentParser(description=__doc__)
AP.add_argument("--arms", nargs=2, required=True, metavar=("FIRST", "SECOND"))
AP.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
AP.add_argument("--checkpoints", type=int, nargs="+", default=[176, 304])
AP.add_argument("--run-root", type=Path, required=True, help="existing lcmx-runs directory")
AP.add_argument(
    "--decision-root", type=Path, default=Path(os.environ["TRACK_S_OUT"]) / "decision" if os.environ.get("TRACK_S_OUT") else None
)
AP.add_argument("--material", type=Path, default=os.environ.get("TRACK_S_MATERIAL"))
AP.add_argument("--logs", type=Path, help="existing decision/logs receipts")
AP.add_argument("--load-log", type=Path, help="optional timestamped host-load log")
ARGS = AP.parse_args()
if ARGS.decision_root is None or ARGS.material is None:
    AP.error("provide --decision-root/--material or TRACK_S_OUT/TRACK_S_MATERIAL")
H, RUNS, MATERIAL = ARGS.decision_root, ARGS.run_root, ARGS.material
ARMS, CPS, SEEDS = ARGS.arms, ARGS.checkpoints, ARGS.seeds
LOGS = ARGS.logs or RUNS.parent / "decision" / "logs"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def nr(xs, p):
    xs = sorted(xs)
    return xs[max(0, -(-int(p * 100) * len(xs) // 100) - 1)] if xs else None


def st(xs):
    return {"n": len(xs), "p50": statistics.median(xs) if xs else None, "p90": nr(xs, 0.9), "max": max(xs) if xs else None}


def score(arm, seed, run, cp):
    p = H / "scores" / f"cp-{cp}" / f"{arm}.seed-{seed}.d{seed}-{run}.json"
    return json.loads(p.read_text()) if p.exists() else None


def grid_by_row(sc, arm, seed, run, cp):
    s = json.loads((RUNS / arm / f"seed-{seed}" / f"d{seed}-{run}" / f"cp-{cp}" / "summary.json").read_text())
    rows = {e["event"]: e["row_index"] for e in s["events"]}
    return {(rows[g["event"]], g["item"]): g["strict"] for g in sc["metrics"]["continuity"]["grid"] if g["event"] in rows}


def pair_axes(cp):
    facts = {n: {f["id"]: f for f in json.loads((MATERIAL / f"seed-{n}" / "facts.json").read_text())} for n in SEEDS}
    axes = {
        k: [0, 0, 0, 0]
        for k in (
            "facts_all",
            "facts_head",
            "facts_middle",
            "facts_tail",
            "constraint_class",
            "continuation",
            "strict_continuity",
            "active_constraint_items",
        )
    }  # [b, c, v1 kept, v2 kept] + n below
    n_items = {k: 0 for k in axes}
    pairs = []
    signs = {k: {"wins": 0, "losses": 0, "ties": 0} for k in axes}
    for seed in SEEDS:
        for run in ("r1", "r2"):
            s1, s2 = score(ARMS[0], seed, run, cp), score(ARMS[1], seed, run, cp)
            if not (s1 and s2):
                continue
            pairs.append(f"seed-{seed}/{run}")
            run_delta = {k: 0 for k in axes}

            def add(ax, k1, k2):
                run_delta[ax] += int(k2) - int(k1)
                axes[ax][0] += k1 and not k2
                axes[ax][1] += k2 and not k1
                axes[ax][2] += k1
                axes[ax][3] += k2
                n_items[ax] += 1

            lost1 = set(s1["metrics"]["facts_kept"].get("lost_before_compaction", {}).get("ids", []))
            lost2 = set(s2["metrics"]["facts_kept"].get("lost_before_compaction", {}).get("ids", []))
            for fid, f in facts[seed].items():
                k1 = s1["probes"].get(fid, {}).get("class") == "CORRECT" and fid not in lost1
                k2 = s2["probes"].get(fid, {}).get("class") == "CORRECT" and fid not in lost2
                add("facts_all", k1, k2)
                add(f"facts_{f['placement']}", k1, k2)
                if f["class"] == "early_user_constraint":
                    add("constraint_class", k1, k2)
            c1 = set(s1["metrics"]["continuation"].get("correct_fields") or [])
            c2 = set(s2["metrics"]["continuation"].get("correct_fields") or [])
            den = s1["metrics"]["continuation"]["denominator"]
            fields = sorted(c1 | c2)
            fields += [f"_miss{i}" for i in range(den - len(fields))]
            for fld in fields:
                add("continuation", fld in c1, fld in c2)
            g1, g2 = grid_by_row(s1, ARMS[0], seed, run, cp), grid_by_row(s2, ARMS[1], seed, run, cp)
            for key in sorted(set(g1) & set(g2)):
                if key[1].endswith("host_instruction"):
                    continue  # system slot, identical by construction
                add("strict_continuity", g1[key], g2[key])
                if key[1].endswith("active_constraint"):
                    add("active_constraint_items", g1[key], g2[key])
            for axis, delta in run_delta.items():
                signs[axis]["wins" if delta > 0 else "losses" if delta < 0 else "ties"] += 1
    out = {}
    for ax, (b, c, k1, k2) in axes.items():
        n = n_items[ax]
        out[ax] = {
            "n": n,
            "v1": round(k1 / n, 3) if n else None,
            "v2": round(k2 / n, 3) if n else None,
            "effect_pts": round(100 * (k2 - k1) / n, 1) if n else None,
            "b_v1_only": b,
            "c_v2_only": c,
            "p": round(mcnemar(b, c), 4),
            "sign_test": {**signs[ax], "p": round(mcnemar(signs[ax]["wins"], signs[ax]["losses"]), 4)},
        }
    missing = [f"seed-{seed}/{run}" for seed in SEEDS for run in ("r1", "r2")
               if f"seed-{seed}/{run}" not in pairs]
    return {"status": "INCOMPLETE" if missing else "COMPLETE", "missing_pairs": missing, "pairs": pairs, "axes": out}


def loads():
    rows = []
    if ARGS.load_log is None:
        return rows
    for line in ARGS.load_log.read_text().splitlines():
        m = re.match(r"(\S+Z) .*load averages?: ([\d.]+)", line)
        if m:
            rows.append((datetime.strptime(m[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp(), float(m[2])))
    return rows


def per_arm():
    L = loads()
    out = {}
    for arm in ARMS:
        leaf, comp, failed, compl, timeouts, l3, leaves, hiload, facts = [], [], [], [], 0, 0, 0, [], {}
        for seed in SEEDS:
            for run in ("r1", "r2"):
                w = LOGS / f"s2-{arm}-d{seed}-{run}.log.wall"
                if not w.exists() or not re.search(r"^exit 0 end", w.read_text(), re.M):
                    continue
                t0 = int(re.search(r"^start (\d+)", w.read_text(), re.M)[1])
                t1 = int(re.search(r"^exit 0 end (\d+)", w.read_text(), re.M)[1])
                mx = max([v for t, v in L if t0 - 600 <= t <= t1] or [0.0])
                if mx > 16:
                    hiload.append(f"seed-{seed}/{run} (max 1-min load {mx})")
                s = json.loads((RUNS / arm / f"seed-{seed}" / f"d{seed}-{run}" / "summary.json").read_text())
                for e in s["events"]:
                    if e["is_compaction"]:
                        comp.append(e["compress_wall_s"])
                    elif e["summariser_calls"]:
                        failed.append(e["compress_wall_s"])
                    for x in e["levels"]:
                        if x.get("depth") == 0 and x.get("wall_s") is not None:
                            leaf.append(x["wall_s"])
                    for c in e["summariser_calls"]:
                        if c.get("usage"):
                            compl.append(c["usage"]["completion_tokens"])
                        if "timeout" in (str(c.get("error") or "") + str(c.get("exception") or "")).lower():
                            timeouts += 1
                sc = score(arm, seed, run, max(CPS))
                if sc:
                    l3 += sc["stored_level3"]["level3"]
                    leaves += sc["stored_level3"]["leaves"]
                    facts.setdefault(seed, {})[run] = sc["metrics"]["facts_kept"]["value"]
        sp = {f"seed-{k}": round(abs(v["r1"] - v["r2"]), 3) for k, v in facts.items() if len(v) == 2}
        out[arm] = {
            "leaf_call_wall_s": st(leaf),
            "compaction_wall_s": st(comp),
            "failed_sweep_count": len(failed),
            "failed_sweep_wall_s": st(failed),
            "compaction_including_failed_sweeps_wall_s": st(comp + failed),
            "completion_tokens": st(compl),
            "timeouts": timeouts,
            "level3_stored_nodes": l3,
            "leaves": leaves,
            "level3_rate": round(l3 / leaves, 4) if leaves else None,
            f"facts_cp{max(CPS)}_r1_r2_spread": sp,
            "spread_over_0.10": [k for k, v in sp.items() if v > 0.10],
            "runs_with_load_over_16": hiload,
        }
    return out


def losses(cp):
    out = {}
    for arm in ARMS:
        tot = {}
        for seed in SEEDS:
            for p in (H / "loss" / f"cp-{cp}").glob(f"{arm}.seed-{seed}.*.json"):
                for k, v in json.loads(p.read_text())["counts"].items():
                    tot[k] = tot.get(k, 0) + v
        out[arm] = tot
    return out


paired = {f"cp-{cp}": {**pair_axes(cp), "loss_classes": losses(cp)} for cp in CPS}
print(
    json.dumps(
        {
            "arms": ARMS,
            "seeds": SEEDS,
            "checkpoints": CPS,
            "status": "INCOMPLETE" if any(p["status"] == "INCOMPLETE" for p in paired.values()) else "COMPLETE",
            "missing_pairs": {cp: p["missing_pairs"] for cp, p in paired.items() if p["missing_pairs"]},
            **paired,
            "per_arm": per_arm(),
        },
        indent=1,
    )
)
