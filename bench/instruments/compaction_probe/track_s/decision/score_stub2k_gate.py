#!/usr/bin/env python3
"""Score the rc3 gate using S8's unchanged score() and existing scored baseline (offline).

Outputs stay under stub2k-gate. Missing/failed/incomplete runs cannot pass.
Strict continuity uses the existing `continuity` rate; its absolute status is preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import os

from score_decision import CPS, score

T = Path(os.environ["TRACK_S_OUT"]).resolve()
MATERIAL = Path(os.environ["TRACK_S_MATERIAL"]).resolve()

ROOT = T / "stub2k-gate"
SHA = "0b26ef7d6ce63ad430f4e676464eb4eaae428c92"
ARMS = ("LCMX-fleet", "LCMX-fleet-agedoff")
METRICS = ("facts_kept", "continuation", "continuity")
STUB_RE = re.compile(r"^LCM active replay stubbing: replaced (\d+) evictable tool result\(s\),", re.M)


def read(path):
    return json.loads(path.read_text())


def receipt_ok(path):
    return path.exists() and re.findall(r"^exit (\d+) end", path.read_text(), re.M) == ["0"]


def aged_stubs(path):
    """Engine assembly-only log; first-sight ingest logs use a different message.

    Counts replacement observations, not unique row IDs (the engine log has no row IDs).
    """
    if not path.exists():
        return None
    return sum(int(n) for n in STUB_RE.findall(path.read_text()))


def value(scored, metric):
    m = scored["metrics"][metric]
    if not m.get("complete") or m.get("value") is None:
        raise ValueError(f"{metric}: incomplete metric")
    return m["value"]


def baseline(cp):
    spreads = {m: [] for m in METRICS}
    refs = []
    for seed in (1, 2, 3):
        pair = []
        for rep in (1, 2):
            path = T / "decision/scores" / f"cp-{cp}" / f"LCMX-fleet.seed-{seed}.d{seed}-r{rep}.json"
            s = read(path)
            if s.get("schema") != "score-s-v1" or s.get("arm") != "LCMX-fleet" or s.get("seed") != f"seed-{seed}" or s.get("smoke"):
                raise ValueError("wrong S8 scored baseline identity")
            pair.append(s)
            refs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for metric in METRICS:
            spreads[metric].append(abs(value(pair[0], metric) - value(pair[1], metric)))
    return {m: {"spread": max(v), "per_seed_spread": v} for m, v in spreads.items()}, refs


def run_record(arm, seed, rep, smoke=False):
    label = f"g9-{arm}-{'smoke-1' if smoke else 'd' + str(seed)}-{rep}"
    seed_name = "smoke-1" if smoke else f"seed-{seed}"
    rdir = ROOT / "runs" / arm / seed_name / label
    wall = ROOT / "logs" / f"{label}.log.wall"
    if not receipt_ok(wall) or not (rdir / "summary.json").exists():
        raise ValueError(f"{label}: missing/failed completion receipt")
    summary, cfg = read(rdir / "summary.json"), read(rdir / "config.json")
    expected = 2000 if arm == "LCMX-fleet" else 6000
    if summary.get("status") != "DONE" or summary.get("worktree_head") != SHA or cfg.get("product_sha") != SHA:
        raise ValueError(f"{label}: wrong product/completion identity")
    if cfg.get("aged_tier_resolved_tokens") != expected or cfg.get("large_output_active_replay_stub_aged_threshold_tokens") != expected:
        raise ValueError(f"{label}: wrong aged threshold")
    if not smoke and summary.get("checkpoints") != list(CPS):
        raise ValueError(f"{label}: wrong checkpoints")
    count = aged_stubs(ROOT / "logs" / f"{label}.log") if not (rdir / "engine.log").exists() else aged_stubs(rdir / "engine.log")
    if count is None:
        raise ValueError(f"{label}: missing engine log")
    return rdir, {"id": label, "arm": arm, "seed": seed_name, "product_sha": SHA,
                  "aged_tier_resolved_tokens": expected, "aged_stub_replacements": count,
                  "exercise": ("VACUOUS" if count == 0 else "EXERCISED") if arm == "LCMX-fleet"
                  else "CONTAMINATED" if count else "AGED-OFF",
                  "wall_s": round(summary["finished"] - summary["started"], 2)}


def verdict(drop, spread, vacuous):
    return "FAIL" if drop > spread else "VACUOUS" if vacuous else "PASS"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rep", choices=("r1", "r2"), default="r1")
    ap.add_argument("--smoke", action="store_true", help="wiring metrics only; never a GA verdict")
    args = ap.parse_args()
    out = {"schema": "stub2k-gate-v1", "product_sha": SHA, "rep": args.rep,
           "claim": "WIRING-ONLY" if args.smoke else "aged-tier quality comparison",
           "stub_count_source": "engine.log assembly stubbing line; replacement observations, not deduplicated row IDs",
           "runs": [], "checkpoints": {}}
    if args.smoke:
        for arm in ARMS:
            rdir, rec = run_record(arm, 1, args.rep, smoke=True)
            s = score(MATERIAL / "smoke-seed-1", rdir, arm)
            rec["metrics"] = {m: {k: s["metrics"][m].get(k) for k in ("value", "status", "complete")} for m in METRICS}
            out["runs"].append(rec)
        on, off = out["runs"]
        out["status"] = "VACUOUS" if on["aged_stub_replacements"] == off["aged_stub_replacements"] == 0 else (
            "COMPLETE" if on["aged_stub_replacements"] > 0 and off["aged_stub_replacements"] == 0 else "FAIL")
    else:
        loaded, gaps = {}, []
        for seed in (1, 2, 3):
            for arm in ARMS:
                try:
                    rdir, rec = run_record(arm, seed, args.rep)
                    loaded[arm, seed] = rdir, rec
                    out["runs"].append(rec)
                except (OSError, ValueError, KeyError) as exc:
                    gaps.append(str(exc))
        # an aged-off control that still stubbed is no control: the comparison is incomplete, as in the smoke rule
        gaps += [f"{r['id']}: aged-off control recorded {r['aged_stub_replacements']} aged stub replacement(s)"
                 for r in out["runs"] if r["exercise"] == "CONTAMINATED"]
        for cp in CPS:
            b, refs = baseline(cp)
            cells = {m: {"per_seed": {}, **b[m]} for m in METRICS}
            for (arm, seed), (rdir, rec) in loaded.items():
                try:
                    probe = rdir / f"cp-{cp}"
                    summary = read(probe / "summary.json")
                    if summary.get("status") != "DONE" or summary.get("stop_row_index") != cp:
                        raise ValueError("missing checkpoint completion")
                    s = score(MATERIAL / f"seed-{seed}", probe, arm)
                    dest = ROOT / "scores" / f"cp-{cp}"
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / f"{rec['id']}.json").write_text(json.dumps(s, indent=1) + "\n")
                    for m in METRICS:
                        cells[m]["per_seed"].setdefault(f"seed-{seed}", {})[arm] = {
                            "value": value(s, m), "status": s["metrics"][m]["status"]}
                except (OSError, ValueError, KeyError) as exc:
                    gaps.append(f"{rec['id']} cp-{cp}: {exc}")
            for m, cell in cells.items():
                complete = all(set(cell["per_seed"].get(f"seed-{n}", {})) == set(ARMS) for n in (1, 2, 3))
                if not complete:
                    cell["verdict"] = "INCOMPLETE"
                    continue
                cell["means"] = {arm: statistics.mean(cell["per_seed"][f"seed-{n}"][arm]["value"] for n in (1, 2, 3)) for arm in ARMS}
                cell["agedoff_minus_stub2k"] = cell["means"][ARMS[1]] - cell["means"][ARMS[0]]
                vacuous = any(r["exercise"] == "VACUOUS" for r in out["runs"])
                cell["verdict"] = verdict(cell["agedoff_minus_stub2k"], cell["spread"], vacuous)
            out["checkpoints"][f"cp-{cp}"] = {"metrics": cells, "s8_scored_refs": refs}
        out["gaps"] = gaps
        states = [c["verdict"] for cp in out["checkpoints"].values() for c in cp["metrics"].values()]
        out["status"] = next((s for s in ("INCOMPLETE", "FAIL", "VACUOUS") if s in states or (s == "INCOMPLETE" and gaps)), "PASS")
    ROOT.mkdir(exist_ok=True)
    path = ROOT / ("smoke-report.json" if args.smoke else f"gate-report-{args.rep}.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
