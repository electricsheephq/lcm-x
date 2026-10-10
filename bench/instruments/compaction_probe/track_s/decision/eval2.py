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
    if not complete or "facts_user" not in intervals:
        return "INCONCLUSIVE"
    covered = {k: v for k, v in intervals.items() if v.get("n_seeds", 8) >= 6 and v.get("low") is not None}
    bars = {k: 5 if k == "facts_user" else -2 for k in intervals}
    if any(v["high"] <= bars[k] for k, v in covered.items()):
        return "KILL"
    if cost_ratio is not None and cost_ratio > 1.10:
        return "KILL"
    return "KEEP" if cost_ratio is not None and len(covered) == len(intervals) and all(v["low"] > bars[k] for k, v in covered.items()) else "INCONCLUSIVE"


def axes(sc, facts, continuity_excluded=(), fact_classes=False):
    out = {}
    for f in facts:
        p = sc["probes"].get(f["id"])
        if p is None:
            continue  # future-source facts have no score at this checkpoint
        keys = ["facts_all", f"facts_{f['row_role']}"]
        if fact_classes:
            keys.append(f"fact_class_{f['class']}")
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
            if k == "continuity":
                grid = {g["item"]: bool(g["strict"]) for g in sc["metrics"][k].get("checkpoint_grid", sc["metrics"][k]["grid"])
                        if g["item"] not in continuity_excluded}
                if sc.get("arm") == "codex-native":
                    grid.update({it: present for it, present in sc.get("host_instruction_presence", {}).items() if present is not None and it in grid})
                out[k] = list(grid.values())
            else:
                out[k] = [1 - v if k == "stale_rate" else v]
    return out


def analyze_window(scores, material, seeds, context_length):
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
        if model != "FAKE" and arm in ("L0", "L1", "L1-H", "L1-noptr"):
            head = sc.get("worktree_head")
            if not isinstance(head, str) or not head.strip():
                raise ValueError(f"{arm}: every real score requires non-empty worktree_head")
            pins.setdefault(arm, set()).add(head)
        if arm != "codex-native" and sc.get("context_length") != context_length:
            continue
        run = Path(sc["run_dir"]).parent
        admitted_usage = sc.get("summariser_usage")
        summary_file = run / "summary.json"
        current = score_manifest.summariser_usage(json.loads(summary_file.read_text())) if summary_file.exists() else None
        if admitted_usage is None or admitted_usage != current:
            raise ValueError(f"admitted summariser usage missing or disagrees: {run}")
        key = (sc["checkpoint_id"], model, None if model == "gpt-6-astra" else context_length, run.name)
        groups.setdefault((arm, seed), {})[key] = sc
        cells = {}
        for p in sc["probes"].values():
            if p.get("kind") and p["class"] != "INCOMPLETE":
                cell = cells.setdefault(f"{p['kind']}|{p.get('actual_compactions', 'unknown')}", [0, 0])
                cell[0] += p["class"] == "CORRECT"
                cell[1] += 1
        counts.append(dict(arm=arm, seed=seed, context_length=sc.get("context_length"), checkpoint_id=sc["checkpoint_id"],
                           facts_denominator=sc["metrics"]["facts_kept"]["denominator"], lifecycle=cells,
                           status_line={s: sum(p.get("status_line") == s for p in sc["probes"].values())
                                        for s in ("STATUS: NOT LIVE", "STATUS: LIVE", "INVALID/MISSING/ABSTAIN")},
                           latency={k: sc.get("accounting", {}).get(k) for k in ("reader_latency", "summariser_latency", "compaction_wall")}))
    for arm, heads in pins.items():
        if len(heads) != 1:
            raise ValueError(f"{arm}: exactly one worktree_head required across all seeds/windows; found {len(heads)}")
    if pins.get("L0", set()) & pins.get("L1", set()):
        raise ValueError("L0 vs L1 carries the same SHA")
    out = {}
    pairs = [("L1", other) for other in COMPARISONS]
    if not any(ar == "L1" for ar, _ in groups):
        pairs.append(("L0", "H"))
    for left_arm, other in pairs:
        required = set()
        by_seed, missing, window, pre, incomplete, errors = {}, [], {}, [], [], {left_arm: 0, other: 0}
        rereads = {left_arm: 0, other: 0}
        totals = {left_arm: [0, 0, True], other: [0, 0, True]}
        seen, summary_seen, exclusions, horizon_exclusions = set(), set(), set(), []
        estimated = {left_arm: 0, other: 0}
        allowed = ("gpt-6-astra",) if other == "codex-native" else ("glm-5.3", "FAKE")
        for seed in seeds:
            left = {k: v for k, v in groups.get((left_arm, seed), {}).items() if k[1] in allowed}
            right = {k: v for k, v in groups.get((other, seed), {}).items() if k[1] in allowed}
            if other == "codex-native":
                references = right
                right = {}
                for key in left:
                    matches = [v for k, v in references.items() if k[:2] == key[:2]]
                    if key in references or len(matches) == 1:
                        right[key] = references[key] if key in references else matches[0]
            required.update(p["kind"] for p in CP.due(material / f"seed-{seed}").values() if p.get("kind"))
            expected = {c["id"] for c in CP.select(material / f"seed-{seed}", "lifecycle")}
            if not left or left.keys() != right.keys() or {k[0] for k in left} != expected:
                missing.append(seed)
                continue
            tally, run_usage = {}, {}
            window[str(seed)] = []
            facts = json.loads((material / f"seed-{seed}" / "facts.json").read_text())
            if other == "H" and left_arm == "L1":
                required.update(f"fact_class_{f['class']}" for f in facts)
            for key in sorted(left):
                pair = (left[key], right[key])
                excluded = {g["item"] for g in right[key]["metrics"].get("continuity", {}).get("grid", [])
                            if other == "codex-native" and g["item"].endswith("host_instruction") and
                            right[key].get("host_instruction_presence", {}).get(g["item"]) is None}
                exclusions.update(excluded)
                for arm, sc in zip((left_arm, other), pair, strict=True):
                    errors[arm] += sc.get("reader_errors", 0)
                    rereads[arm] += sc.get("reader_rereads", 0)
                beyond = right[key].get("beyond_declared", {})
                if other == "codex-native" and any(beyond.get(k) for k in ("facts", "corrections", "compactions", "continuity")):
                    horizon_exclusions.append(dict(seed=seed, checkpoint=key[0], beyond_declared=beyond))
                    continue
                if all(sc.get("behaviour", {}).get("compactions", 0) >= 1 for sc in pair):
                    for arm, sc in zip((left_arm, other), pair, strict=True):
                        identity = (arm, str(Path(sc["run_dir"]).resolve()))
                        if identity in seen:
                            continue
                        seen.add(identity)
                        usage, total = sc.get("accounting", {}), totals[arm]
                        estimated[arm] += usage.get("reader_estimated_attempts", 0)
                        tokens = [usage.get(k) for k in ("reader_input_tokens", "reader_output_tokens")]
                        total[2] &= all(v is not None for v in tokens)
                        total[0] += sum(v or 0 for v in tokens)
                        total[1] += usage.get("successful_probes", 0)
                        root = Path(sc["run_dir"]).parent / "summary.json"
                        run_usage[(arm, str(root.resolve()))] = sc["summariser_usage"]
                if any(sc.get("reader_errors") or not sc["metrics"]["facts_kept"]["complete"] or
                       not sc["metrics"].get("lifecycle", {}).get("complete", True) or
                       sc.get("reader_readback", {}).get("pin_ok") is False or
                       key[1] == "gpt-6-astra" and sc.get("reader_readback", {}).get("effort") != "low" for sc in pair):
                    incomplete.append(dict(seed=seed, checkpoint=key[0]))
                    continue
                if not all(sc.get("behaviour", {}).get("compactions", 0) >= 1 for sc in pair):
                    pre.append(dict(seed=seed, checkpoint=key[0], axes=[axes(sc, facts, excluded) for sc in pair]))
                    continue
                window[str(seed)].append(key[0])
                a, b = (axes(sc, facts, excluded, fact_classes=other == "H" and left_arm == "L1") for sc in pair)
                for k in a.keys() & b.keys():
                    cell = tally.setdefault(k, [0, 0, 0, 0])
                    for i, values in enumerate((a[k], b[k])):
                        cell[2*i] += sum(values)
                        cell[2*i+1] += len(values)
            for identity, usage in run_usage.items():
                if identity in summary_seen:
                    continue
                summary_seen.add(identity)
                arm = identity[0]
                tokens = [c[k] for c in usage["calls"] for k in ("prompt_tokens", "completion_tokens")]
                totals[arm][2] &= usage["events_present"] and all(v is not None for v in tokens)
                totals[arm][0] += sum(v or 0 for v in tokens)
            if tally:
                by_seed[seed] = {k: 100*(v[0]/v[1] - v[2]/v[3]) for k, v in tally.items() if v[1] and v[3]}
        intervals = {k: bootstrap([v[k] for v in by_seed.values() if k in v]) for k in sorted(required | {k for v in by_seed.values() for k in v})}
        for k, ci in intervals.items():
            bar = 5 if other == "L0" and k == "facts_user" else -2
            ci["verdict"] = "INCONCLUSIVE" if ci["n_seeds"] < 6 or ci["low"] is None else "SUPERIOR" if ci["low"] > bar and bar == 5 else "NON-INFERIOR" if ci["low"] > bar else "BELOW BAR" if ci["high"] <= bar else "INCONCLUSIVE"
        costs = {arm: t[0]/t[1] if t[2] and t[1] else None for arm, t in totals.items()}
        ratio = costs[left_arm]/costs[other] if costs[left_arm] is not None and costs[other] else None
        complete = not missing and not incomplete and len(seeds) >= 8 and bool(intervals)
        covered = complete and all(v["n_seeds"] >= 6 and v["low"] is not None for v in intervals.values())
        claim = CLAIMS[other] if left_arm == "L1" else "descriptive fake comparison"
        if claim == "KEEP gate":
            result = verdict(intervals, ratio, complete)
        elif other in ("H", "codex-native") and left_arm == "L1":
            result = "NON-INFERIOR" if covered and all(v["low"] > -2 for v in intervals.values()) else "BELOW FLOOR" if covered and any(v["high"] <= -2 for v in intervals.values()) else "INCONCLUSIVE"
        else:
            result = "DESCRIPTIVE"
        out[f"{left_arm}−{'C' if other == 'codex-native' else other}"] = dict(intervals=intervals, missing_seeds=missing, cost_ratio=ratio,
            cost_per_success=costs, compared_window=window, pre_window=pre, incomplete_pairs=incomplete, reader_errors=errors, reader_rereads=rereads,
            reader_estimated_attempts=estimated, continuity_exclusions=sorted(exclusions), horizon_exclusions=horizon_exclusions,
            verdict=result, claim_class=claim, claim="mechanism only; overall KEEP also needs product comparison")
    return dict(schema="eval2-v2", context_length=context_length, unit="session/seed", resamples=10000, rng_seed=1064, seeds=seeds, comparisons=out, per_checkpoint=counts)


def analyze(scores, material, seeds):
    admission = score_manifest.load(scores.parent / "manifest.json")
    contexts = {sc.get("context_length") for f in scores.glob("cp-*/*.json") if score_manifest.admitted(scores.parent, admission, f)
                for sc in [json.loads(f.read_text())] if sc["arm"] != "codex-native" and int(sc["seed"].removeprefix("seed-")) in seeds}
    windows = {str(ctx): analyze_window(scores, material, seeds, ctx) for ctx in sorted(contexts or {272000}, key=str)}
    # Preserve the default-window API; every other window has its own independent record.
    primary = windows.get("272000", next(iter(windows.values())))
    return dict(primary, context_windows=windows)


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
        windows = data.get("context_windows", {str(data.get("context_length")): data})
        comparisons = {f"{pair} ({ctx})": v for ctx, w in windows.items() for pair, v in w["comparisons"].items()}
        checkpoints = [c for w in windows.values() for c in w["per_checkpoint"]]
        statuses = {}
        for ctx, window in windows.items():
            for c in window["per_checkpoint"]:
                cell = statuses.setdefault((c["arm"], ctx), {"STATUS: LIVE": 0, "STATUS: NOT LIVE": 0})
                for status in cell:
                    cell[status] += c.get("status_line", {}).get(status, 0)
        rows = ["| comparison (context) | axis | n seeds | effect pt | 95% interval | axis verdict | verdict | claim class | scope |", "|---|---|---|---|---|---|---|---|---|"]
        for pair, value in comparisons.items():
            for axis, ci in value["intervals"].items() or {"unmeasured": dict(effect_pts=None, low=None, high=None)}.items():
                rows.append(f"| {pair} | {axis} | {ci.get('n_seeds', 0)} | {ci['effect_pts']} | {ci['low']} … {ci['high']} | {ci.get('verdict', 'INCONCLUSIVE')} | {value['verdict']} | {value['claim_class']} | {value['claim']} |")
        rows += ["", "| arm (context) | LIVE answers | NOT LIVE answers | claim class |", "|---|---|---|---|"]
        rows += [f"| {arm} ({ctx}) | {v['STATUS: LIVE']} | {v['STATUS: NOT LIVE']} | descriptive only |" for (arm, ctx), v in statuses.items()]
        rows += ["", "| pair | seed | compared-window checkpoints |", "|---|---|---|"]
        for pair, value in comparisons.items():
            for seed, cps in value["compared_window"].items():
                rows.append(f"| {pair} | {seed} | {', '.join(cps)} |")
            rows.append(f"| {pair} | INCOMPLETE | {value['incomplete_pairs']} |")
            rows.append(f"| {pair} | reader re-reads per arm | {value.get('reader_rereads', {})} |")
            rows.append(f"| {pair} | cost ratio / estimated attempts per arm | {value.get('cost_ratio')} / {value.get('reader_estimated_attempts', {})} |")
            rows.append(f"| {pair} | beyond-declared exclusions | {value.get('horizon_exclusions', [])} |")
            rows.append(f"| {pair} | continuity excluded (host-visible receipt absent) | {value.get('continuity_exclusions', [])} |")
        rows += ["", "| arm (context) | checkpoint | facts denominator | lifecycle kind/actual compactions | correct/denominator | status-line scoring (NOT LIVE passes) |", "|---|---|---|---|---|---|"]
        for c in checkpoints:
            for kind, cell in c["lifecycle"].items() or {"none due": [0, 0]}.items():
                rows.append(f"| {c['arm']} ({c.get('context_length')}) | {c['checkpoint_id']} | {c['facts_denominator']} | {kind.replace('|', ' / ')} | {cell[0]}/{cell[1]} | {c.get('status_line', {})} |")
        rows += ["", "| arm | checkpoint | reader latency | summariser latency | compaction wall |", "|---|---|---|---|---|"]
        for c in checkpoints:
            rows.append(f"| {c['arm']} | {c['checkpoint_id']} | " + " | ".join(str(c['latency'][k]) for k in ("reader_latency", "summariser_latency", "compaction_wall")) + " |")
        a.out.write_text("\n".join(rows) + "\n")
    else:
        a.out.write_text(json.dumps(analyze(a.scores, a.material, a.seeds), indent=1) + "\n")


if __name__ == "__main__":
    main()
