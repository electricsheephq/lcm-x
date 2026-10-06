#!/usr/bin/env python3
"""Track S report: score_s.py JSONs -> decision tables (r3 §R3; r4 F2 mean-of-two + band algebra, F5 direction).

report_s.py --scores <dir of score JSONs> --out RESULTS-S.md [--json out.json] [--seeds-expected 3]
Stdlib only. The report COMPUTES; adjudication against the gate table is a separate, blind step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

QUALITY = ("facts_kept", "stale_rate", "trap_abstention", "continuation", "recall")
LOWER = {"stale_rate", "trap_failure_rate", "level3", "latency"}
NOT_IN_S = (("Reliability matrix", "bench/instruments/reliability receipts on three hosts (G-REL-1)"),
            ("Cross-session recall (Phase 4)", "RUN-SHEET-V1M-REBANK; not executed in this program's scope (r4 F5)"),
            ("Field window (Phase 0/5)", "release lane field pass: field-pass.sh + aggregate-field.py (r4 F4/F6)"),
            ("Visible wait (Phase 3)", "hosted LCM-X via the Hermes gateway or the local gauntlet host"))
PREFIX = {}  # arm -> prefix60k score JSONs (r4 F1 timing population (i)); kept out of every other table


def load(d: Path):
    runs = {}
    for p in sorted(d.glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if s.get("schema") == "score-s-v1" and s.get("population") == "prefix60k":
            PREFIX.setdefault(s["arm"], []).append(s)
        elif s.get("schema") == "score-s-v1":
            runs.setdefault((s["arm"], s["seed"]), []).append(s)
    return {k: sorted(v, key=lambda s: str(s.get("run") or 0)) for k, v in runs.items()}


def prefix_cell(arm):
    """The frozen-60k-prefix population: summariser calls and events with their actual sample counts."""
    return "; ".join(f"{s['seed']} {Path(s['run_dir']).name}: " + " / ".join(f"{k} n={lt[k]['count']} p90 {fmt(lt[k]['p90'])} max "
                     f"{fmt(lt[k]['max'])}" for k in ("events", "calls")) + f" [{lt['status']}; {lt['timing_label']}]"
                     for s in PREFIX.get(arm, []) for lt in [s["metrics"]["latency"]]) or "INCOMPLETE (no prefix60k run)"


def seed_value(runs, metric):
    """F2: the arithmetic mean of the two run-level rates -> (mean, |run1 - run2|, reason when not computable)."""
    if len(runs) < 2:
        return None, None, f"{len(runs)} of 2 runs"
    ms = [r["metrics"][metric] for r in runs[:2]]
    bad = [next((i for i in m.get("issues", []) if not i.startswith("FAIL")), m["status"])
           for m in ms if not m.get("complete") or m.get("value") is None]
    if bad:
        return None, None, bad[0] if len(bad[0]) <= 140 else bad[0][:137] + "..."
    a, b = (m["value"] for m in ms)
    return (a + b) / 2, abs(a - b), None


def seed_verdict(cand, comp, rule, lower):
    """cand/comp from seed_value. Band = max of the two arms' spreads; d = cand - comp (F2) or comp - cand (F5)."""
    if cand[2] or comp[2]:
        return {"verdict": "INCOMPLETE", "reason": f"candidate: {cand[2]}" if cand[2] else f"comparator: {comp[2]}"}
    band = max(cand[1], comp[1])
    d = comp[0] - cand[0] if lower else cand[0] - comp[0]
    ok = d >= -band if rule == "parity" else d > band
    return {"verdict": "PASS" if ok else "FAIL", "d": d, "band": band, "beyond_2band": d < -2 * band}


def overall(verdicts, rule, expected):
    """Parity: >= 2 of 3 seeds pass and no seed fails beyond 2 x band. Superiority: >= 2 of 3 seeds with d > +band."""
    inc = [v.get("reason") for v in verdicts if v["verdict"] == "INCOMPLETE"]
    if len(verdicts) < expected or inc:
        return f"INCOMPLETE({len(verdicts)} of {expected} seeds" + (f"; {inc[0]}" if inc else "") + ")"
    passes, need = sum(v["verdict"] == "PASS" for v in verdicts), min(2, expected)  # "2 of 3"; a 1-seed test run needs 1
    if rule == "parity":
        return "PASS" if passes >= need and not any(v["beyond_2band"] for v in verdicts) else "FAIL"
    return "PASS" if passes >= need else "FAIL"


def compare(runs, cand, comp, metric, rule, seeds, expected):
    per = {}
    for seed in seeds:
        c, k = runs.get((cand, seed), []), runs.get((comp, seed), [])
        per[seed] = seed_verdict(seed_value(c, metric), seed_value(k, metric) if k else (None, None, f"{comp} missing"),
                                 rule, metric in LOWER)
    return {"seeds": per, "overall": overall(list(per.values()), rule, expected)}


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def verdict_cell(v):
    return v["verdict"] + (f" d={fmt(v['d'])} band={fmt(v['band'])}" if "d" in v else f" ({v['reason']})")


def absolute(runs, arm, seeds, metric, expected):
    sts = [(seed, r.get("run"), r["metrics"][metric]["status"]) for seed in seeds for r in runs.get((arm, seed), [])]
    fails = [s for s in sts if s[2].startswith("FAIL")]
    if fails:
        return "FAIL", sts
    if not sts or len({s for s, _, _ in sts}) < expected or any(not s[2].startswith("OK") for s in sts):
        return "INCOMPLETE", sts
    return "PASS", sts


def render(runs, expected, title="Track S results"):
    arms = sorted({a for a, _ in runs})
    seeds = sorted({s for _, s in runs})
    lcmx = [a for a in arms if a.startswith("LCMX") and not a.endswith("-open")]
    lcmx_open = [a for a in arms if a.startswith("LCMX") and a.endswith("-open")]
    out, js = [], {"arms": arms, "seeds": seeds, "gates": {}}
    w = out.append
    w(f"# {title} (computed by report_s.py; adjudication is separate)\n")
    w(f"Seeds: {', '.join(seeds)} (expected {expected}). Arms: {', '.join(arms)}. Value per arm x seed = mean of two runs (r4 F2); "
      "spread = |run1 - run2|; band for a comparison = max spread of the two arms. Lower-is-better metrics use "
      "d = comparator - candidate (r4 F5). Any missing run, comparator, receipt or field = INCOMPLETE, never a pass. Recall legend (Q2): "
      "an unreachable clipped-middle / externalized-file fact = FAIL only on S1 public path + S2 `-open` arms (the recall gate row); a "
      "FAIL tag in a context-only arm's per-arm recall cell is an ordinary miss.\n")
    w("## Per arm x seed\n")
    cols = ("facts_kept", "stale_rate", "trap_abstention", "trap_failure_rate", "continuation", "recall", "continuity",
            "latency", "level3")
    w("| arm | seed | runs | reader / provenance | " + " | ".join(cols) + " |")
    w("|---" * (len(cols) + 4) + "|")
    for (arm, seed), rs in sorted(runs.items()):
        cells = []
        for m in cols:
            mean, spread, why = seed_value(rs, m)
            sts = "; ".join(sorted({r["metrics"][m]["status"].split("(")[0] for r in rs}))
            vals = "/".join(fmt(r["metrics"][m].get("value")) for r in rs)
            cells.append(f"{fmt(mean)} ±{fmt(spread)} [{sts}]" if why is None else f"runs {vals} [{sts}]")
        w(f"| {arm} | {seed} | {len(rs)} | {rs[0].get('reader')} / {rs[0].get('provenance')} | " + " | ".join(cells) + " |")
    w("\n## r3 §R3 gate rows produced by Track S\n")
    w("### Quality >= lossless-claw default path (parity rule, F2)\n")
    w("| candidate | metric | " + " | ".join(seeds) + " | overall |")
    w("|---" * (len(seeds) + 3) + "|")
    for a in lcmx or ["(no LCM-X arm scored)"]:
        for m in QUALITY:
            r = compare(runs, a, "lossless-claw", m, "parity", seeds, expected)
            js["gates"][f"quality_vs_lc/{a}/{m}"] = r
            w(f"| {a} | {m} | " + " | ".join(verdict_cell(r["seeds"][s]) for s in seeds) + f" | {r['overall']} |")
    w("\n### Quality vs Codex native (gap only; no pass/fail)\n")
    w("| candidate | metric | " + " | ".join(seeds) + " |")
    w("|---" * (len(seeds) + 2) + "|")
    for a in lcmx:
        for m in QUALITY:
            r = compare(runs, a, "codex-native", m, "parity", seeds, expected)
            w(f"| {a} | {m} | " + " | ".join(
                f"gap d={fmt(v['d'])} band={fmt(v['band'])}" if "d" in v else f"INCOMPLETE ({v['reason']})"
                for v in r["seeds"].values()) + " |")
    w("\n### Instruction continuity (ABSOLUTE: every item verbatim after every event; one strict miss = FAIL)\n")
    w("| arm | result | per run (seed, run, status) |\n|---|---|---|")
    for a in lcmx + lcmx_open + ["lossless-claw"]:
        res, sts = absolute(runs, a, seeds, "continuity", expected)
        js["gates"][f"continuity/{a}"] = res
        w(f"| {a} | {res} | " + "; ".join(f"{s} r{n}: {st}" for s, n, st in sts) + " |")
    w("\n### Latency p90 <= 60 s, never > 120 s (event wall; host-visible wait is not measured here)\n")
    w("| arm | wall label | 60k prefix | full stream per run (p90 / max / gate events / status) | result |\n|---|---|---|---|---|")
    for a in lcmx + lcmx_open + ["lossless-claw", "codex-native"]:
        res, sts = absolute(runs, a, seeds, "latency", expected)
        cells = []
        for seed in seeds:
            for r in runs.get((a, seed), []):
                lt = r["metrics"]["latency"]
                cells.append(f"{seed} r{r.get('run')}: {fmt(lt['gate_events']['p90'])} / {fmt(lt['gate_events']['max'])} / "
                             f"{lt['gate_events']['count']} / {lt['status']}")
                if res == "PASS" and lt.get("p90_le_60") is False:
                    res = "FAIL (p90 > 60 s)"
        label = next((r["metrics"]["latency"].get("wall_label") for s in seeds for r in runs.get((a, s), [])), None)
        ref = " (labelled reference)" if a == "codex-native" else ""
        js["gates"][f"latency/{a}"] = res
        w(f"| {a}{ref} | {label} | {prefix_cell(a) if a != 'codex-native' else 'n/a (r3: native cannot trigger inside 60k)'} | "
          + "; ".join(cells) + f" | {res} |")
    w("\n### Level-3 incidence (inside the band of zero = mean <= own run-to-run band; ABSOLUTE: answered route = FAIL)\n")
    w("| arm | " + " | ".join(seeds) + " | absolute |\n" + "|---" * (len(seeds) + 2) + "|")
    for a in lcmx + lcmx_open:
        cells = []
        for seed in seeds:
            mean, spread, why = seed_value(runs.get((a, seed), []), "level3")
            cells.append(f"INCOMPLETE ({why})" if why else f"{fmt(mean)} vs band {fmt(spread)}: "
                         + ("inside" if mean <= spread else "outside"))
        res, _ = absolute(runs, a, seeds, "level3", expected)
        w(f"| {a} | " + " | ".join(cells) + f" | {res} |")
    w("\n### Tool-assisted recall (correct / all scheduled clipped-middle + externalized-file + superseded-decision probes)\n")
    w("| candidate (reader) | comparison | rule | " + " | ".join(seeds) + " | overall |")
    w("|---" * (len(seeds) + 4) + "|")
    for a in lcmx_open or ["(no LCMX -open arm scored)"]:
        rd = next((r.get("reader") for s in seeds for r in runs.get((a, s), [])), None)
        for comp, rule in (("lossless-claw-open", "parity"), ("codex-native", "superiority")):
            r = compare(runs, a, comp, "recall", rule, seeds, expected)
            crd = next((x.get("reader") for s in seeds for x in runs.get((comp, s), [])), None)
            js["gates"][f"recall/{a}/{comp}"] = r
            w(f"| {a} ({rd}) | vs {comp} ({crd}) | {rule} | " + " | ".join(verdict_cell(r["seeds"][s]) for s in seeds)
              + f" | {r['overall']} |")
        res, sts = absolute(runs, a, seeds, "recall", expected)
        w(f"| {a} | ABSOLUTE clipped-middle / externalized-file reachable | absolute | "
          + " | ".join("; ".join(st for s2, _, st in sts if s2 == s) or "no run" for s in seeds) + f" | {res} |")
    w("\n### Rows Track S does not produce\n\n| gate | status | producing unit |\n|---|---|---|")
    for g, u in NOT_IN_S:
        w(f"| {g} | NOT IN TRACK S | {u} |")
    w("\n## Behaviour per arm x seed x run\n")
    w("| arm | seed | run | compactions | events (turn, tokens, wall s, timing) | wall p50/p90/max | summariser p50/p90/max "
      "| levels / fallbacks | tokens kept |\n" + "|---" * 9 + "|")
    for (arm, seed), rs in sorted(runs.items()):
        for r in rs:
            b = r["behaviour"]
            pub = lambda e: f" (+publication {fmt(e['publication_s'])} s, beside)" if e.get("publication_s") is not None else ""  # noqa: E731
            ev = "; ".join(f"t{e['turn']} {e['tokens']} {fmt(e['wall_s'])} {e['timing']}{pub(e)}" for e in b["events"] if e["compaction"])
            trip = lambda s: "/".join(fmt(s[k]) for k in ("p50", "p90", "max"))  # noqa: E731
            w(f"| {arm} | {seed} | {r.get('run')} | {b['compactions']} | {ev or '—'} | {trip(b['wall'])} ({b['wall_label']}) | "
              f"{trip(b['summariser_wall'])} | {b['levels']} | {b['tokens_kept_at_checkpoint']} |")
    w("\n## Recall path labels (context / search only / expand) per arm\n")
    for arm in arms:
        r0 = next(r for (a, _), rs in runs.items() if a == arm for r in rs)
        split = {k: v for k, v in r0["metrics"]["recall"].get("by_label", {}).items()}
        w(f"- **{arm}**: " + "; ".join(f"`{k}` -> {v}" for k, v in r0["label_map"].items()) + f". First run split: {split}")
    w("\n## Field gaps named by the scorer (never filled with invented values)\n")
    for arm in arms:
        r0 = next(r for (a, _), rs in runs.items() if a == arm for r in rs)
        w(f"- **{arm}** ({r0['writer']}): " + " | ".join(r0["gaps"]))
    return "\n".join(out) + "\n", js


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--seeds-expected", type=int, default=3)
    ap.add_argument("--title", default="Track S results")
    a = ap.parse_args(argv)
    md, js = render(load(a.scores), a.seeds_expected, a.title)
    a.out.write_text(md, encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps(js, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
