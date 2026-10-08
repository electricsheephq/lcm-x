"""D8 analysis, model-free, from decision/scores + decision/loss + lcmx-runs summaries + load.log. Prints JSON only.
Pairs: same seed, same run index (r1 with r1, r2 with r2). Exact two-sided McNemar on discordant pairs per axis.
Axes: facts kept (all, head/middle/tail), constraint class (early_user_constraint facts), continuation fields,
strict continuity (grid items aligned by the event's row_index; both arms must have an event at that row),
active_constraint continuity items (same alignment). b = first arm kept & second arm lost, c = second arm kept & first arm lost.
The v1/v2 column names denote first/second CLI arms; sign-test wins favor the second arm.
Each run pair votes once per axis; ties are excluded from the exact two-sided sign p-value.
With --second-tree the second arm is read from another run/decision/logs root (release over release, D2);
results are keyed by --labels. Every p also has an unrounded p_exact; the d2 block uses only unrounded values."""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scorer"))
import score_manifest  # noqa: E402

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
AP.add_argument("--second-tree", type=Path, nargs=3, metavar=("RUN_ROOT", "DECISION_ROOT", "LOGS"),
                help="read the SECOND arm from this run root, decision root and logs root (release over release)")
AP.add_argument("--labels", nargs=2, metavar=("FIRST", "SECOND"), help="result keys for the two arms (default: --arms)")
AP.add_argument("--expect-shas", nargs=2, metavar=("FIRST", "SECOND"),
                help="product commit (hex prefix, >= 7) every counted run of each arm must record; required with --second-tree")
ARGS = AP.parse_args()
if ARGS.decision_root is None or ARGS.material is None:
    AP.error("provide --decision-root/--material or TRACK_S_OUT/TRACK_S_MATERIAL")
H, RUNS, MATERIAL = ARGS.decision_root, ARGS.run_root, ARGS.material
ARMS, CPS, SEEDS = ARGS.arms, ARGS.checkpoints, ARGS.seeds
LOGS = ARGS.logs or RUNS.parent / "decision" / "logs"
LABELS = ARGS.labels or ARMS
if LABELS[0] == LABELS[1]:
    AP.error("labels must be distinct (pass --labels when both trees use the same arm)")
# label -> sources; functions fall back to RUNS/H/LOGS and label == arm when a label has no entry
TREES = {LABELS[0]: {"arm": ARMS[0], "run_root": RUNS, "decision_root": H, "logs": LOGS}}
TREES[LABELS[1]] = {"arm": ARMS[1], **dict(zip(("run_root", "decision_root", "logs"), ARGS.second_tree or (RUNS, H, LOGS)))}
TWO_TREE = ARGS.second_tree is not None
EXPECT_SHAS = [x.lower() for x in ARGS.expect_shas] if ARGS.expect_shas else None
if EXPECT_SHAS and not all(re.fullmatch(r"[0-9a-f]{7,40}", x) for x in EXPECT_SHAS):
    AP.error("--expect-shas takes two hex commit prefixes of at least 7 characters")
if TWO_TREE and not EXPECT_SHAS:
    AP.error("--second-tree requires --expect-shas (the product commit of each tree)")


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


def score(label, seed, run, cp):
    src = globals().get("TREES", {}).get(label, {})
    root, arm = src.get("decision_root", H), src.get("arm", label)
    p = root / "scores" / f"cp-{cp}" / f"{arm}.seed-{seed}.d{seed}-{run}.json"
    manifest = score_manifest.load(root / "manifest.json")
    return json.loads(p.read_text()) if score_manifest.admitted(root, manifest, p) else None


def grid_by_row(sc, label, seed, run, cp):
    src = globals().get("TREES", {}).get(label, {})
    runs, arm = src.get("run_root", RUNS), src.get("arm", label)
    s = json.loads((runs / arm / f"seed-{seed}" / f"d{seed}-{run}" / f"cp-{cp}" / "summary.json").read_text())
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
    n_items, excluded = {k: 0 for k in axes}, {k: 0 for k in axes}
    pairs, missing, partial = [], [], False
    signs = {k: {"wins": 0, "losses": 0, "ties": 0} for k in axes}
    first, second = globals().get("LABELS", ARMS)
    for seed in SEEDS:
        for run in ("r1", "r2"):
            s1, s2 = score(first, seed, run, cp), score(second, seed, run, cp)
            if not (s1 and s2):
                missing.append(f"seed-{seed}/{run}")
                continue
            def partial_only(s, m):  # incomplete only through reader truncation or missing result rows
                return m != "continuity" and any(p.get("class") in ("MISSING", "READER_TRUNCATED") for p in s["probes"].values()) and all(
                    i.startswith(("INCOMPLETE: READER_TRUNCATED: ", "INCOMPLETE: no continuation_field row ")) or
                    (i.startswith("INCOMPLETE: ") and "scheduled facts have no result row" in i) for i in s.get("metrics", {}).get(m, {}).get("issues", []))
            if not all(s.get("metrics", {}).get(m, {}).get("complete") is True or partial_only(s, m)
                       for s in (s1, s2) for m in ("facts_kept", "continuation", "continuity")):
                missing.append(f"seed-{seed}/{run} (incomplete score)")
                continue
            pairs.append(f"seed-{seed}/{run}")
            partial = partial or any(p.get("class") in ("MISSING", "READER_TRUNCATED") for s in (s1, s2) for p in s["probes"].values())
            run_delta = {k: 0 for k in axes}

            def add(ax, k1, k2):
                if k1 is None or k2 is None:
                    excluded[ax] += 1
                    return
                run_delta[ax] += int(k2) - int(k1)
                axes[ax][0] += k1 and not k2
                axes[ax][1] += k2 and not k1
                axes[ax][2] += k1
                axes[ax][3] += k2
                n_items[ax] += 1

            def kept(s, pid):
                cls = s["probes"].get(pid, {}).get("class", "MISSING")
                return None if cls in ("MISSING", "READER_TRUNCATED") else cls == "CORRECT"
            lost1 = set(s1["metrics"]["facts_kept"].get("lost_before_compaction", {}).get("ids", []))
            lost2 = set(s2["metrics"]["facts_kept"].get("lost_before_compaction", {}).get("ids", []))
            for fid, f in facts[seed].items():
                k1 = False if fid in lost1 else kept(s1, fid)  # an admission-proven loss outranks truncation
                k2 = False if fid in lost2 else kept(s2, fid)
                add("facts_all", k1, k2)
                add(f"facts_{f['placement']}", k1, k2)
                if f["class"] == "early_user_constraint":
                    add("constraint_class", k1, k2)
            fields = {pid for s in (s1, s2) for pid in s["probes"] if "." in pid}
            for pid in fields:
                add("continuation", kept(s1, pid), kept(s2, pid))
            g1, g2 = grid_by_row(s1, first, seed, run, cp), grid_by_row(s2, second, seed, run, cp)
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
            "n": n, "excluded": excluded[ax],
            "v1": round(k1 / n, 3) if n else None,
            "v2": round(k2 / n, 3) if n else None,
            "effect_pts": round(100 * (k2 - k1) / n, 1) if n else None,
            "b_v1_only": b,
            "c_v2_only": c,
            "p": round(mcnemar(b, c), 4),
            "p_exact": mcnemar(b, c),
            "sign_test": {**signs[ax], "p": round(mcnemar(signs[ax]["wins"], signs[ax]["losses"]), 4),
                          "p_exact": mcnemar(signs[ax]["wins"], signs[ax]["losses"])},
        }
    return {"status": "INCOMPLETE" if missing or partial or any(excluded.values()) else "COMPLETE", "missing_pairs": missing, "pairs": pairs, "axes": out}


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
    for label in globals().get("LABELS", ARMS):
        src = globals().get("TREES", {}).get(label, {})
        arm, runs, logs = src.get("arm", label), src.get("run_root", RUNS), src.get("logs", LOGS)
        leaf, comp, failed, compl, timeouts, l3, leaves, hiload, facts = [], [], [], [], 0, 0, 0, [], {}
        heads, configs = {}, []  # per counted run: recorded product commit; distinct effective configurations
        for seed in SEEDS:
            for run in ("r1", "r2"):
                w = logs / f"s2-{arm}-d{seed}-{run}.log.wall"
                if not w.exists() or not re.search(r"^exit 0 end", w.read_text(), re.M):
                    continue
                t0 = int(re.search(r"^start (\d+)", w.read_text(), re.M)[1])
                t1 = int(re.search(r"^exit 0 end (\d+)", w.read_text(), re.M)[1])
                mx = max([v for t, v in L if t0 - 600 <= t <= t1] or [0.0])
                if mx > 16:
                    hiload.append(f"seed-{seed}/{run} (max 1-min load {mx})")
                s = json.loads((runs / arm / f"seed-{seed}" / f"d{seed}-{run}" / "summary.json").read_text())
                heads[f"seed-{seed}/{run}"] = s.get("worktree_head")
                cfg = {k: s.get(k) for k in ("arm", "fleet_keys_excluded", "harness_overrides", "context_length", "lane", "reader")}
                configs += [cfg] if cfg not in configs else []
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
                for cp in CPS:
                    sc = score(label, seed, run, cp)
                    if sc and cp == max(CPS):
                        l3 += sc["stored_level3"]["level3"]
                        leaves += sc["stored_level3"]["leaves"]
                    if sc:
                        facts.setdefault(cp, {}).setdefault(seed, {})[run] = sc["metrics"]["facts_kept"]["value"]
        raw = {f"cp-{cp}": {f"seed-{k}": None if None in (v["r1"], v["r2"]) else abs(v["r1"] - v["r2"])
                            for k, v in facts.get(cp, {}).items() if len(v) == 2}
               for cp in CPS}  # None: a run had no scorable fact (all reader-truncated)
        by_cp = {cp: {k: None if v is None else round(v, 3) for k, v in sd.items()} for cp, sd in raw.items()}  # display
        sp = by_cp[f"cp-{max(CPS)}"]
        out[label] = {
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
            "facts_r1_r2_spread_by_cp": by_cp,
            "spread_over_0.10": [f"{cp}/{k}" for cp, s in raw.items() for k, v in s.items() if v is not None and round(v, 9) > 0.10],  # exact at the boundary
            "spread_unmeasured": [f"{cp}/{k}" for cp, s in raw.items() for k, v in s.items() if v is None],
            "runs_with_load_over_16": hiload,
            "worktree_heads": heads,
            "run_configs": configs,
        }
    return out


def losses(cp):
    out = {}
    for label in globals().get("LABELS", ARMS):
        src = globals().get("TREES", {}).get(label, {})
        root, arm = src.get("decision_root", H), src.get("arm", label)
        manifest = score_manifest.load(root / "manifest.json")
        tot = {}
        for seed in SEEDS:
            for p in (root / "loss" / f"cp-{cp}").glob(f"{arm}.seed-{seed}.*.json"):
                if not score_manifest.admitted(root, manifest, p):
                    continue
                for k, v in json.loads(p.read_text())["counts"].items():
                    tot[k] = tot.get(k, 0) + v
        out[label] = tot
    return out


paired = {f"cp-{cp}": {**pair_axes(cp), "loss_classes": losses(cp)} for cp in CPS}
labels, arm_stats = globals().get("LABELS", ARMS), per_arm()
status = "INCOMPLETE" if any(p["status"] == "INCOMPLETE" for p in paired.values()) else "COMPLETE"
d2 = {cp: {"p_exact": a["p_exact"], "effect_pts_exact": 100 * (a["c_v2_only"] - a["b_v1_only"]) / a["n"] if a["n"] else None}
      for cp, a in ((cp, p["axes"]["facts_all"]) for cp, p in paired.items())}  # k2 - k1 == c - b on paired items
for v in d2.values():
    v["net_loss_ge_5"] = v["effect_pts_exact"] is not None and v["effect_pts_exact"] <= -5
    v["blocks"] = v["p_exact"] < 0.05 and v["net_loss_ge_5"]
spread = {label: arm_stats.get(label, {}).get("spread_over_0.10", []) for label in labels}
missing_spread = [f"{label}/cp-{cp}/seed-{n}" for label in labels for cp in CPS for n in SEEDS  # receipt-skipped or unscored
                  if arm_stats.get(label, {}).get("facts_r1_r2_spread_by_cp", {}).get(f"cp-{cp}", {}).get(f"seed-{n}") is None]
expect = globals().get("EXPECT_SHAS")
sha_mismatch = [f"{label}/{k}: {(h or 'missing')[:12]}" for label, sha in zip(labels, expect or ())
                for k, h in arm_stats.get(label, {}).get("worktree_heads", {}).items() if not str(h or "").startswith(sha)]


def receipt_mismatch():
    """Each admitted score must have been scored from THIS tree's run: its manifest receipt hash must equal the hash of the
    wall receipt in the label's logs root (a decision root from another tree fails here)."""
    out, trees = [], globals().get("TREES")
    if trees is None:  # extracted for an offline test without the CLI's tree map: nothing to bind
        return out
    for label in labels:
        src = trees.get(label, {})
        root, logs, arm = src.get("decision_root", H), src.get("logs", LOGS), src.get("arm", label)
        manifest = score_manifest.load(root / "manifest.json")
        for cp in CPS:
            for n in SEEDS:
                for run in ("r1", "r2"):
                    key = f"scores/cp-{cp}/{arm}.seed-{n}.d{n}-{run}.json"
                    entry = (manifest or {}).get("entries", {}).get(key)
                    if entry is None or not score_manifest.admitted(root, manifest, root / key):
                        continue
                    w = logs / f"s2-{arm}-d{n}-{run}.log.wall"
                    have = hashlib.sha256(w.read_bytes()).hexdigest() if w.is_file() else None
                    if entry.get("receipt_sha256") != have:
                        out.append(f"{label}/cp-{cp}/seed-{n}/{run}")
    return out


receipts = receipt_mismatch()
two_tree = bool(globals().get("TWO_TREE"))  # one configuration across trees is a D2 rule; single-tree arms differ by design
configs = [c for label in labels for c in arm_stats.get(label, {}).get("run_configs", [])] if two_tree else []
diff = sorted({k for c in configs for k in c if c[k] != configs[0][k]})
diff += sorted({f"arm.{k}" for c in configs if "arm" in diff and isinstance(c["arm"], dict) and isinstance(configs[0]["arm"], dict)
                for k in {*c["arm"], *configs[0]["arm"]} if c["arm"].get(k) != configs[0]["arm"].get(k)})
verdict = ("INCOMPLETE" if status == "INCOMPLETE" or missing_spread or sha_mismatch or receipts or diff
           else "REPEAT_SEEDS" if any(spread.values()) else "BLOCK" if any(v["blocks"] for v in d2.values()) else "PASS")
d2.update({"candidate": labels[1], "spread_over_0.10": spread, "missing_spread": missing_spread,
           "expected_shas": dict(zip(labels, expect)) if expect else None, "sha_mismatch": sha_mismatch, "receipt_mismatch": receipts,
           "config_equal": (not diff) if two_tree else None, **({"config_diff_keys": diff} if diff else {}), "status": status, "verdict": verdict})
print(
    json.dumps(
        {
            "arms": ARMS,
            "labels": labels,
            **({"trees": {k: {f: str(x) for f, x in v.items()} for k, v in TREES.items()}} if globals().get("TWO_TREE") else {}),
            "seeds": SEEDS,
            "checkpoints": CPS,
            "status": status,
            "missing_pairs": {cp: p["missing_pairs"] for cp, p in paired.items() if p["missing_pairs"]},
            **paired,
            "per_arm": arm_stats,
            "d2": d2,
        },
        indent=1,
    )
)
