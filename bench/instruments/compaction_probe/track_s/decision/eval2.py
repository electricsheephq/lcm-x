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
            out.setdefault(f"{p['kind']}|{p['compaction_horizon']}", []).append(p["class"] == "CORRECT")
        elif pid.endswith(".next_action"):
            out.setdefault("next_action", []).append(p["class"] == "CORRECT")
    for k in ("continuity", "continuation", "trap_abstention", "stale_rate"):
        v = sc["metrics"].get(k, {}).get("value")
        if v is not None:
            out[k] = [1 - v if k == "stale_rate" else v]
    return out


def analyze(scores, material, seeds):
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")
    groups, costs, readers, counts, invalid = {}, {}, {}, [], set()
    admission = score_manifest.load(scores.parent / "manifest.json")
    for file in sorted(scores.glob("cp-*/*.json")):
        if not score_manifest.admitted(scores.parent, admission, file):
            continue
        sc = json.loads(file.read_text())
        seed, arm = int(sc["seed"].removeprefix("seed-")), sc["arm"]
        if seed not in seeds:
            continue
        if not sc["metrics"]["facts_kept"]["complete"] or not sc["metrics"].get("lifecycle", {}).get("complete", True):
            invalid.add((arm, seed))
        if "gpt-6-astra" in (sc.get("reader") or "") and sc.get("reader_readback", {}).get("effort") != "low":
            invalid.add((arm, seed))
        facts = json.loads((material / f"seed-{seed}" / "facts.json").read_text())
        model = (sc.get("reader") or "").split("·")[0]
        key = (arm, seed, file.parent.name.split("-CP", 1)[-1], model)
        groups.setdefault(key, []).append(axes(sc, facts))
        root = Path(sc["run_dir"]).parent / "summary.json"
        run_cost = json.loads(root.read_text()).get("accounting", {}).get("cost_tokens_per_successful_task") if root.exists() else None
        costs.setdefault((arm, seed, model), {})[sc["run_dir"].rsplit("/", 1)[0]] = run_cost
        counts.append(dict(arm=arm, seed=seed, checkpoint_id=sc["checkpoint_id"],
                           facts_denominator=sc["metrics"]["facts_kept"]["denominator"],
                           lifecycle=sc["metrics"].get("lifecycle", {}).get("by_kind_horizon", {})))
        readers.setdefault(arm, set()).add(model)
    out = {}
    for other in COMPARISONS:
        by_seed, missing = {}, []
        for seed in seeds:
            allowed = ("gpt-6-astra",) if other == "codex-native" else ("glm-5.3", "FAKE")
            left = {cp: rs for (arm, n, cp, reader), rs in groups.items() if arm == "L1" and n == seed and reader in allowed}
            right = {cp: rs for (arm, n, cp, reader), rs in groups.items() if arm == other and n == seed and reader in allowed}
            expected = {c["id"].split("-CP", 1)[-1] for c in CP.select(material / f"seed-{seed}", "lifecycle")} if left else set()
            if not left or left.keys() != expected or left.keys() != right.keys() or any(len(left[c]) != len(right[c]) for c in left):
                missing.append(seed)
                continue
            tally = {}
            for cp in left:
                for a, b in zip(left[cp], right[cp], strict=True):
                    for k in a.keys() & b.keys():
                        tally.setdefault(k, []).append(100 * (sum(a[k]) / len(a[k]) - sum(b[k]) / len(b[k])))
            by_seed[seed] = {k: sum(v) / len(v) for k, v in tally.items()}
        keys = {k for v in by_seed.values() for k in v}
        intervals = {k: bootstrap([v[k] for v in by_seed.values() if k in v]) for k in sorted(keys)}
        def cost(arm):
            vs = [v for (ar, n, reader), values in costs.items() for v in values.values() if ar == arm and n in seeds and reader in allowed]
            return sum(vs) / len(vs) if vs and all(v is not None for v in vs) else None
        candidate_cost, baseline_cost = cost("L1"), cost(other)
        ratio = candidate_cost / baseline_cost if candidate_cost is not None and baseline_cost else None
        complete = not missing and len(seeds) >= 8 and not any((arm, n) in invalid for arm in ("L1", other) for n in seeds) and all(v["n_seeds"] == len(seeds) for v in intervals.values())
        if other == "codex-native":
            complete = complete and "gpt-6-astra" in readers.get("L1", set()) and readers.get(other) == {"gpt-6-astra"}
        out[f"L1−{other}"] = dict(intervals=intervals, missing_seeds=missing, cost_ratio=ratio,
            verdict=verdict(intervals, ratio, complete), claim="mechanism only; overall KEEP also needs product comparison")
    return dict(schema="eval2-v1", unit="session/seed", resamples=10000, rng_seed=1064, seeds=seeds, comparisons=out, per_checkpoint=counts)


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
        rows = ["| comparison | axis | effect pt | 95% interval | verdict |", "|---|---|---|---|---|"]
        for pair, value in data["comparisons"].items():
            for axis, ci in value["intervals"].items() or {"unmeasured": dict(effect_pts=None, low=None, high=None)}.items():
                rows.append(f"| {pair} | {axis} | {ci['effect_pts']} | {ci['low']} … {ci['high']} | {value['verdict']} |")
        rows += ["", "| arm | checkpoint | facts denominator | lifecycle kind/horizon | correct/denominator |", "|---|---|---|---|---|"]
        for c in data["per_checkpoint"]:
            for kind, cell in c["lifecycle"].items() or {"none due": [0, 0]}.items():
                rows.append(f"| {c['arm']} | {c['checkpoint_id']} | {c['facts_denominator']} | {kind.replace('|', ' / ')} | {cell[0]}/{cell[1]} |")
        a.out.write_text("\n".join(rows) + "\n")
    else:
        a.out.write_text(json.dumps(analyze(a.scores, a.material, a.seeds), indent=1) + "\n")


if __name__ == "__main__":
    main()
