"""Eval-2 session-clustered paired bootstrap and frozen margin decisions; stdlib only."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scorer"))
import checkpoints as CP  # noqa: E402
import score_manifest  # noqa: E402

COMPARISONS = ("L0", "H", "L1-H", "L1-noptr", "codex-native")
CLAIMS = {"L0": "KEEP gate", "H": "mechanism-level floor record",
          "L1-H": "descriptive ablations", "L1-noptr": "descriptive ablations", "codex-native": "descriptive parity claim"}


def bootstrap(deltas, resamples=10000, rng_seed=1064):
    if len(deltas) < 2:
        return dict(effect_pts=None, low=None, high=None, n_seeds=len(deltas))
    rng, n = random.Random(rng_seed), len(deltas)
    samples = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(resamples))
    return dict(effect_pts=sum(deltas) / n, low=samples[int(.025 * resamples)],
                high=samples[int(.975 * resamples) - 1], n_seeds=n)


def verdict(intervals, cost_ratio=None, complete=True):
    """KEEP requires every lower bound; KILL requires a failing upper bound. Crossing a bar is inconclusive."""
    if not complete or "facts_user" not in intervals or any(v.get("low") is None for v in intervals.values()) or cost_ratio is None:
        return "INCONCLUSIVE"
    if cost_ratio > 1.10:
        return "KILL"
    bars = {k: 5 if k == "facts_user" else -2 for k in intervals}
    if any(v["high"] <= bars[k] for k, v in intervals.items()):
        return "KILL"
    return "KEEP" if all(v["low"] > bars[k] for k, v in intervals.items()) else "INCONCLUSIVE"


def axes(sc, facts):
    out = {}
    for f in facts:
        p = sc["probes"].get(f["id"])
        if p is None:
            continue  # future-source facts have no score at this checkpoint
        keys = ["facts_all", f"facts_{f['row_role']}"]
        if f["row_role"] == "tool":
            keys.append(f"facts_tool_{f['placement']}")
        if f["class"] in ("name", "path", "build_id", "prefix", "file_change"):
            keys.append("identifier")
        for k in keys:
            out.setdefault(k, []).append(p["class"] == "CORRECT" and f["id"] not in
                                         sc["metrics"]["facts_kept"].get("lost_before_compaction", {}).get("ids", []))
    for pid, p in sc["probes"].items():
        if p.get("kind"):
            out.setdefault(p["kind"], []).append(p["class"] == "CORRECT")
        elif pid.endswith(".next_action"):
            out.setdefault("next_action", []).append(p["class"] == "CORRECT")
    for k in ("continuity", "continuation", "trap_abstention", "stale_rate"):
        v = sc["metrics"].get(k, {}).get("value")
        if v is not None:
            out[k] = list({g["item"]: bool(g["strict"]) for g in sc["metrics"][k].get("checkpoint_grid", sc["metrics"][k]["grid"])}.values()) if k == "continuity" else [1 - v if k == "stale_rate" else v]
    return out


def analyze(scores, material, seeds):
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")
    groups, counts, pins = {}, [], {}
    admission = score_manifest.load(scores.parent / "manifest.json")
    for file in sorted(scores.glob("cp-*/*.json")):
        if not score_manifest.admitted(scores.parent, admission, file):
            continue
        sc = json.loads(file.read_text())
        seed, arm = int(sc["seed"].removeprefix("seed-")), sc["arm"]
        if seed not in seeds:
            continue
        model = (sc.get("reader") or "").split("·")[0]
        run = Path(sc["run_dir"]).parent
        key = (sc["checkpoint_id"], model, sc.get("context_length"), run.name)
        groups.setdefault((arm, seed), {})[key] = sc
        if sc.get("worktree_head"):
            pins.setdefault(arm, set()).add(sc["worktree_head"])
        cells = {}
        for p in sc["probes"].values():
            if p.get("kind") and p["class"] != "INCOMPLETE":
                cell = cells.setdefault(f"{p['kind']}|{p.get('actual_compactions', 'unknown')}", [0, 0])
                cell[0] += p["class"] == "CORRECT"
                cell[1] += 1
        counts.append(dict(arm=arm, seed=seed, context_length=sc.get("context_length"), checkpoint_id=sc["checkpoint_id"],
                           facts_denominator=sc["metrics"]["facts_kept"]["denominator"], lifecycle=cells,
                           latency={k: sc.get("accounting", {}).get(k) for k in ("reader_latency", "summariser_latency", "compaction_wall")}))
    if pins.get("L0", set()) & pins.get("L1", set()):
        raise ValueError("L0 vs L1 carries the same SHA")
    out = {}
    pairs = [("L1", other) for other in COMPARISONS]
    if not any(ar == "L1" for ar, _ in groups):
        pairs.append(("L0", "H"))
    for left_arm, other in pairs:
        required = set()
        by_seed, missing, window, pre, incomplete, errors = {}, [], {}, [], [], {left_arm: 0, other: 0}
        totals = {left_arm: [0, 0, True], other: [0, 0, True]}
        allowed = ("gpt-6-astra",) if other == "codex-native" else ("glm-5.3", "FAKE")
        for seed in seeds:
            left = {k: v for k, v in groups.get((left_arm, seed), {}).items() if k[1] in allowed}
            right = {k: v for k, v in groups.get((other, seed), {}).items() if k[1] in allowed}
            required.update(p["kind"] for p in CP.due(material / f"seed-{seed}").values() if p.get("kind"))
            expected = {c["id"] for c in CP.select(material / f"seed-{seed}", "lifecycle")}
            if not left or left.keys() != right.keys() or {k[0] for k in left} != expected:
                missing.append(seed)
                continue
            tally, run_usage = {}, {}
            window[str(seed)] = []
            facts = json.loads((material / f"seed-{seed}" / "facts.json").read_text())
            for key in sorted(left):
                pair = (left[key], right[key])
                for arm, sc in zip((left_arm, other), pair, strict=True):
                    errors[arm] += sc.get("reader_errors", 0)
                if any(sc.get("reader_errors") or not sc["metrics"]["facts_kept"]["complete"] or
                       not sc["metrics"].get("lifecycle", {}).get("complete", True) or
                       sc.get("reader_readback", {}).get("pin_ok") is False or
                       key[1] == "gpt-6-astra" and sc.get("reader_readback", {}).get("effort") != "low" for sc in pair):
                    incomplete.append(dict(seed=seed, checkpoint=key[0]))
                    continue
                if not all(sc.get("behaviour", {}).get("compactions", 0) >= 1 for sc in pair):
                    pre.append(dict(seed=seed, checkpoint=key[0], axes=[axes(sc, facts) for sc in pair]))
                    continue
                window[str(seed)].append(key[0])
                a, b = (axes(sc, facts) for sc in pair)
                for k in a.keys() & b.keys():
                    cell = tally.setdefault(k, [0, 0, 0, 0])
                    for i, values in enumerate((a[k], b[k])):
                        cell[2*i] += sum(values)
                        cell[2*i+1] += len(values)
                for arm, sc in zip((left_arm, other), pair, strict=True):
                    usage, total = sc.get("accounting", {}), totals[arm]
                    tokens = [usage.get(k) for k in ("reader_input_tokens", "reader_output_tokens")]
                    total[2] &= all(v is not None for v in tokens)
                    total[0] += sum(v or 0 for v in tokens)
                    total[1] += usage.get("successful_probes", 0)
                    root = Path(sc["run_dir"]).parent / "summary.json"
                    run_usage[(arm, str(root))] = root
            for (arm, _), root in run_usage.items():
                summ = json.loads(root.read_text()) if root.exists() else {}
                calls = [{**c, **(c.get("usage") or {})} for ev in summ.get("events", []) for c in ev.get("summariser_calls", [])]
                tokens = [c.get(k) for c in calls for k in ("prompt_tokens", "completion_tokens")]
                totals[arm][2] &= "events" in summ and all(v is not None for v in tokens)
                totals[arm][0] += sum(v or 0 for v in tokens)
            if not tally:
                missing.append(seed)
            else:
                by_seed[seed] = {k: 100*(v[0]/v[1] - v[2]/v[3]) for k, v in tally.items() if v[1] and v[3]}
        intervals = {k: bootstrap([v[k] for v in by_seed.values() if k in v]) for k in sorted({k for v in by_seed.values() for k in v})}
        costs = {arm: t[0]/t[1] if t[2] and t[1] else None for arm, t in totals.items()}
        ratio = costs[left_arm]/costs[other] if costs[left_arm] is not None and costs[other] else None
        complete = not missing and len(seeds) >= 8 and bool(intervals) and required <= intervals.keys() and all(v["n_seeds"] == len(seeds) for v in intervals.values())
        claim = CLAIMS[other] if left_arm == "L1" else "descriptive fake comparison"
        if claim == "KEEP gate":
            result = verdict(intervals, ratio, complete)
        elif other in ("H", "codex-native") and left_arm == "L1":
            result = "NON-INFERIOR" if complete and all(v["low"] > -2 for v in intervals.values()) else "BELOW FLOOR" if complete and any(v["high"] <= -2 for v in intervals.values()) else "INCONCLUSIVE"
        else:
            result = "DESCRIPTIVE"
        out[f"{left_arm}−{'C' if other == 'codex-native' else other}"] = dict(intervals=intervals, missing_seeds=missing, cost_ratio=ratio,
            cost_per_success=costs, compared_window=window, pre_window=pre, incomplete_pairs=incomplete, reader_errors=errors,
            verdict=result, claim_class=claim, claim="mechanism only; overall KEEP also needs product comparison")
    return dict(schema="eval2-v2", unit="session/seed", resamples=10000, rng_seed=1064, seeds=seeds, comparisons=out, per_checkpoint=counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--material", type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 9)))
    ap.add_argument("--analysis", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.analysis:
        data = json.loads(a.analysis.read_text())
        rows = ["| comparison | axis | effect pt | 95% interval | verdict | claim class | scope |", "|---|---|---|---|---|---|---|"]
        for pair, value in data["comparisons"].items():
            for axis, ci in value["intervals"].items() or {"unmeasured": dict(effect_pts=None, low=None, high=None)}.items():
                rows.append(f"| {pair} | {axis} | {ci['effect_pts']} | {ci['low']} … {ci['high']} | {value['verdict']} | {value['claim_class']} | {value['claim']} |")
        rows += ["", "| pair | seed | compared-window checkpoints |", "|---|---|---|"]
        for pair, value in data["comparisons"].items():
            for seed, cps in value["compared_window"].items():
                rows.append(f"| {pair} | {seed} | {', '.join(cps)} |")
            rows.append(f"| {pair} | INCOMPLETE | {value['incomplete_pairs']} |")
        rows += ["", "| arm | checkpoint | facts denominator | lifecycle kind/actual compactions | correct/denominator |", "|---|---|---|---|---|"]
        for c in data["per_checkpoint"]:
            for kind, cell in c["lifecycle"].items() or {"none due": [0, 0]}.items():
                rows.append(f"| {c['arm']} | {c['checkpoint_id']} | {c['facts_denominator']} | {kind.replace('|', ' / ')} | {cell[0]}/{cell[1]} |")
        rows += ["", "| arm | checkpoint | reader latency | summariser latency | compaction wall |", "|---|---|---|---|---|"]
        for c in data["per_checkpoint"]:
            rows.append(f"| {c['arm']} | {c['checkpoint_id']} | " + " | ".join(str(c['latency'][k]) for k in ("reader_latency", "summariser_latency", "compaction_wall")) + " |")
        a.out.write_text("\n".join(rows) + "\n")
    else:
        a.out.write_text(json.dumps(analyze(a.scores, a.material, a.seeds), indent=1) + "\n")


if __name__ == "__main__":
    main()
