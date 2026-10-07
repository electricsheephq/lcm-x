#!/usr/bin/env python3
"""S8: scores + loss + run dirs + frozen report-unit-notes.md/json -> DECISION-RUN.md, DECISION-RUN.json.

Aggregation (r4 F2): per arm x seed the mean of the two runs; then the mean across the three seeds. A seed with fewer than
2 runs makes the cell INCOMPLETE (the available per-run values are shown, never averaged). Pooled rows (engine wall, tokens at
trigger, loss classes, labels) list every run behind them with n. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import math
from collections import Counter
from pathlib import Path

H = Path(os.environ["TRACK_S_OUT"]).resolve() / "decision"
ARMS = ("LCMX-fleet", "lossless-claw", "lossless-claw-tuned", "codex-native")
OPEN = ("LCMX-fleet-open", "lossless-claw-open", "lossless-claw-tuned-open")
SEEDS = ("seed-1", "seed-2", "seed-3")
GUARD_CALLS, GUARD_WINDOW = 24, 600.0  # engine default spend guard (config.py summary_spend_max_calls / window)


def f3(x):
    return "—" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(q / 100 * len(xs)) - 1)] if xs else None


def load(cp):
    out = {}
    for p in sorted((H / "scores" / f"cp-{cp}").glob("*.json")):
        s = json.loads(p.read_text())
        s["_loss"] = json.loads((H / "loss" / f"cp-{cp}" / p.name).read_text())
        s["_summary"] = json.loads((Path(s["run_dir"]) / "summary.json").read_text())
        out.setdefault(s["arm"], {}).setdefault(s["seed"], []).append(s)
    return out


def cell(runs_by_seed, fn, need=2):
    """-> (display, value|None, n per seed, max run-to-run spread)"""
    means, ns, spreads, raw = [], [], [], []
    for seed in SEEDS:
        rs = runs_by_seed.get(seed, [])
        vals = [fn(r) for r in rs]
        ns.append(len(rs))
        raw.append("/".join(f3(v) for v in vals) or "none")
        if len(rs) >= need and all(v is not None for v in vals[:need]):
            means.append(sum(vals[:need]) / need)
            spreads.append(max(vals[:need]) - min(vals[:need]))
    n = "/".join(map(str, ns))
    if len(means) == len(SEEDS):
        return f"{f3(sum(means) / len(means))} (n={n})", sum(means) / len(means), ns, max(spreads)
    return f"INCOMPLETE (n={n}; runs {'<br>'.join(raw)})", None, ns, None


def placement(r, p):
    g = r["metrics"]["facts_kept"]["by_class_placement"]
    ok = sum(v[0] for k, v in g.items() if k.endswith("|" + p))
    tot = sum(v[1] for k, v in g.items() if k.endswith("|" + p))
    return ok / tot if tot else None


def diag(r):
    g = [x["diagnostic"] for x in r["metrics"]["continuity"].get("grid", []) if x["diagnostic"] is not None]
    return sum(map(bool, g)) / len(g) if g else None


def ancestry(r):
    rc = [x for x in r["_summary"].get("receipts", []) if x.get("kind") == "ancestry"]
    return sum(x.get("status") == "PASS" for x in rc) / len(rc) if rc else None


def spend_window(summary):
    """Largest number of summariser calls inside any 600 s window; the call at which 24/600 s would have tripped."""
    ts = sorted(c["t_start"] for e in summary.get("events", []) for c in e.get("summariser_calls", []) if c.get("t_start"))
    best, trip, j = 0, None, 0
    for i, t in enumerate(ts):
        while ts[j] <= t - GUARD_WINDOW:
            j += 1
        best = max(best, i - j + 1)
        if trip is None and i - j + 1 >= GUARD_CALLS:
            trip = i + 1
    return {"calls": len(ts), "max_in_600s": best, "default_guard_trip_at_call": trip or "never"}


def rows_for(data, arms, cp):
    lines, js = [], {}
    cols = [("facts kept", lambda r: r["metrics"]["facts_kept"]["value"]),
            ("facts head", lambda r: placement(r, "head")), ("facts middle", lambda r: placement(r, "middle")),
            ("facts tail", lambda r: placement(r, "tail")),
            ("trap abst.", lambda r: r["metrics"]["trap_abstention"]["value"]),
            ("continuation", lambda r: r["metrics"]["continuation"]["value"]),
            ("continuity strict", lambda r: r["metrics"]["continuity"]["value"]),
            ("continuity diag", diag), ("ancestry PASS", ancestry),
            ("recall", lambda r: r["metrics"]["recall"]["value"])]
    lines.append("| arm | " + " | ".join(c for c, _ in cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for a in arms:
        if a not in data:
            lines.append(f"| {a} | " + " | ".join("UNSUPPORTED (n=0/0/0; end-only probe)" if a == "codex-native" and cp == 176 else "INCOMPLETE (n=0/0/0)" for _ in cols) + " |")
            continue
        need = 1 if a in OPEN else 2
        cs = {c: cell(data.get(a + "-open", {}) if c == "recall" and a != "codex-native" and a not in OPEN else data[a], fn, need=2) for c, fn in cols}
        js[a] = {c: {"display": v[0], "value": v[1], "n": v[2], "spread": v[3]} for c, v in cs.items()}
        js[a]["recall_source"] = a + "-open" if a != "codex-native" and a not in OPEN else a
        if need == 1:  # -open: one run per seed by design -> INCOMPLETE under the 2-run rule; per-seed values shown
            js[a]["note"] = "one run per seed (D6); every cell INCOMPLETE under the 2-run rule"
        lines.append(f"| {a} | " + " | ".join(cs[c][0] for c, _ in cols) + " |")
    return lines, js


def behaviour(data, arms):
    lines = ["| arm | runs | labels | summary blocks to reader | compactions per run | tokens at trigger p50 / max | "
             "engine wall p50 / p90 / max (gate events) | events > 120 s | reader usage per call (prompt / compl) | "
             "level-3 truncated / verbatim / leaves | lost facts by class |", "|---" * 11 + "|"]
    js = {}
    for a in arms:
        rs = [r for s in SEEDS for r in data.get(a, {}).get(s, [])]
        if not rs:
            lines.append(f"| {a} | 0 |" + " — |" * 9)
            continue
        ev = [e for r in rs for e in r["behaviour"]["events"] if e["compaction"] and e.get("timing") != "WIRING-ONLY"]
        walls, toks = [e["wall_s"] for e in ev], [e["tokens"] for e in ev if e.get("tokens") is not None]
        labels = Counter(lb for r in rs for lb in r["labels"]) + Counter({"—": sum(not r["labels"] for r in rs)})
        blocks = Counter(b for r in rs for b in r["summary_blocks"])
        calls = sum((r.get("reader_usage") or {}).get("calls", 0) for r in rs)
        pt = sum((r.get("reader_usage") or {}).get("prompt_tokens", 0) for r in rs)
        ct = sum((r.get("reader_usage") or {}).get("completion_tokens", 0) for r in rs)
        l3 = [r.get("stored_level3", {}) for r in rs]
        l3s = "n/a" if not a.startswith("LCMX") else (f"{sum(m.get('truncated', 0) for m in l3)} / "
                                                         f"{sum(m.get('verbatim', 0) for m in l3)} / {sum(m.get('leaves', 0) for m in l3)}")
        loss = Counter(k for r in rs for k, v in r["_loss"]["counts"].items() for _ in range(v))
        comp = [r["behaviour"]["compactions"] for r in rs]
        row = {"runs": len(rs), "labels": dict(labels), "summary_blocks": dict(blocks), "compactions": comp,
               "tokens_at_trigger": {"p50": pct(toks, 50), "max": max(toks) if toks else None},
               "engine_wall": {"n": sum(w is not None for w in walls), "missing": sum(w is None for w in walls), "p50": pct(walls, 50), "p90": pct(walls, 90), "max": max(filter(None, walls), default=None)},
               "events_over_120s": sum((w or 0) > 120 for w in walls),
               "reader_per_call": {"prompt": pt / calls if calls else None, "completion": ct / calls if calls else None, "calls": calls},
               "level3": l3s, "loss": dict(loss)}
        js[a] = row
        lines.append(f"| {a} | {len(rs)} | {dict(labels)} | {dict(blocks)} | {comp} | {f3(row['tokens_at_trigger']['p50'])} / "
                     f"{f3(row['tokens_at_trigger']['max'])} | {f3(row['engine_wall']['p50'])} / {f3(row['engine_wall']['p90'])} / "
                     f"{f3(row['engine_wall']['max'])} (n={row['engine_wall']['n']}; missing={row['engine_wall']['missing']}; "
                     f"{'INCOMPLETE' if not row['engine_wall']['n'] or row['engine_wall']['missing'] or any(len(data.get(a, {}).get(s, [])) < 2 for s in SEEDS) else 'COMPLETE'}) | {row['events_over_120s']} | "
                     f"{f3(row['reader_per_call']['prompt'] and round(row['reader_per_call']['prompt']))} / "
                     f"{f3(row['reader_per_call']['completion'] and round(row['reader_per_call']['completion']))} (calls {calls if calls else 'INCOMPLETE: not recorded'}) | "
                     f"{l3s} | {dict(loss)} |")
    return lines, js


def parity(js_q, cp):
    """LCM-X vs each arm: per-seed band = max(|r1-r2|) over the two arms; parity+ = LCMX mean >= other mean - band."""
    lines = [f"| metric (cp-{cp}) | vs arm | LCMX-fleet | other | spread used | verdict |", "|---|---|---|---|---|---|"]
    out = {}
    for m in ("facts kept", "continuation", "continuity strict"):
        for other in ARMS[1:]:
            a, b = js_q.get("LCMX-fleet", {}).get(m), js_q.get(other, {}).get(m)
            if not a or not b or a["value"] is None or b["value"] is None:
                v = "INCOMPLETE"
                lines.append(f"| {m} | {other} | {a and a['display']} | {b and b['display']} | — | {v} |")
            else:
                band = max(a["spread"], b["spread"])
                v = "parity+" if a["value"] >= b["value"] - band else "below"
                lines.append(f"| {m} | {other} | {f3(a['value'])} | {f3(b['value'])} | {f3(band)} | {v} |")
            out[f"{m}/{other}"] = v
    return lines, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-notes", type=Path, required=True, help="directory containing frozen report-unit-notes.md/json")
    args = ap.parse_args()
    md, js = ["# S8 decision run: computed tables (render_decision.py)\n"], {}
    for cp in (176, 304):
        data = load(cp)
        q, jq = rows_for(data, ARMS, cp)
        o, jo = rows_for(data, OPEN, cp)
        b, jb = behaviour(data, ARMS + OPEN)
        p, jp = parity(jq, cp)
        md += [f"\n## Checkpoint cp-{cp} (material row_index {cp})\n", "### Quality (mean of two runs per seed, then across seeds)\n",
               *q, "\n### -open runs (tools on; one run per seed)\n", *o, "\n### Behaviour (pooled over every run behind the row)\n", *b,
               "\n### Parity against the bar (LCM-X vs each arm)\n", *p]
        js[f"cp-{cp}"] = {"quality": jq, "open": jo, "behaviour": jb, "parity": jp}
    sw = {}
    for p in sorted((H.parent / "lcmx-runs").glob("LCMX-fleet*/seed-*/d*")):
        if (p / "summary.json").exists():
            sw[f"{p.parent.parent.name}/{p.parent.name}/{p.name}"] = spend_window(json.loads((p / "summary.json").read_text()))
    md += ["\n## Summariser calls per 600 s window (LCM-X runs; guard off, LCM_SUMMARY_SPEND_MAX_CALLS=0)\n",
           "| run | summariser calls | max in any 600 s | default 24/600 s guard would trip at call |", "|---|---|---|---|"]
    md += [f"| {k} | {v['calls']} | {v['max_in_600s']} | {v['default_guard_trip_at_call']} |" for k, v in sw.items()]
    js["spend_window"] = sw
    md += [(args.report_notes / "report-unit-notes.md").read_text()]
    js["report_unit"] = json.loads((args.report_notes / "report-unit-notes.json").read_text())
    (H / "DECISION-RUN.md").write_text("\n".join(md) + "\n")
    (H / "DECISION-RUN.json").write_text(json.dumps(js, indent=1, default=str) + "\n")
    print("wrote DECISION-RUN.md, DECISION-RUN.json")


if __name__ == "__main__":
    main()
