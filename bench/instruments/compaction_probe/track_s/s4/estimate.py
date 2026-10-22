#!/usr/bin/env python3
"""Estimate turns / tokens / wall to the first native auto-compaction on a seed, calibrated on the S4 smoke.

Context tokens after turn t ~= offset + k * C(t), C = cumulative chars the CLI keeps (user text incl. dictation, the echoed
notes, tool outputs capped at the smoke's observed truncation). Wall per turn ~= a + b * note_chars + c * calls.
Both fits use the smoke's own rollout (token_count per turn end) and turn log. Estimate only; no model calls.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

TS = Path(os.environ["TRACK_S_OUT"]).resolve()
MATERIAL = Path(os.environ["TRACK_S_MATERIAL"]).resolve()
RUN = TS / "codex-runs" / "smoke-seed-1" / "1"
TRIGGER = 244_800


def lstsq(X, y):  # tiny normal-equations solver (stdlib only)
    n = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[p], b[i], b[p] = A[p], A[i], b[p], b[i]
        for r in range(n):
            if r != i and A[i][i]:
                f = A[r][i] / A[i][i]
                A[r] = [x - f * z for x, z in zip(A[r], A[i])]
                b[r] -= f * b[i]
    return [b[i] / A[i][i] for i in range(n)]


def turn_chars(rows, cap):
    out = {}
    for r in rows:
        t = out.setdefault(r["turn"], {"chars": 0, "note_chars": 0, "calls": 0})
        c = len(r["content"])
        if r["role"] == "user":
            t["chars"] += c
        elif r["role"] == "assistant" and not r.get("tool_call_id"):
            t["chars"] += 2 * c            # dictated in the user turn + reproduced by the agent
            t["note_chars"] += c
        elif r["role"] == "tool":
            t["chars"] += min(c, cap) + 300
            t["calls"] += 1
    return out


def main(seed="1"):
    s = json.loads((RUN / "summary.json").read_text())
    parsed = json.loads((RUN / "rollout-parse.json").read_text())
    adm = json.loads((RUN / "admission.json").read_text())
    cap = max(r["output_chars"] for r in adm["rows"] if r["role"] == "tool" and r.get("runtime_truncated"))
    mat = MATERIAL / "smoke-seed-1"
    rows = [json.loads(x) for x in (mat / "transcript.jsonl").read_text().splitlines() if x.strip()]
    tc = turn_chars(rows, cap)
    ts = [(datetime.fromisoformat(p["ts"].replace("Z", "+00:00")).timestamp(), p["last_input"])
          for p in parsed["token_series"]]
    X, y, cum = [], [], 0
    for t in s["turn_log"]:
        if t["turn"] == "checkpoint":
            continue
        cum += tc[t["turn"]]["chars"]
        pts = [v for x, v in ts if x <= t["end_ts"]]
        if pts:
            X.append([1.0, float(cum)])
            y.append(float(pts[-1]))
    offset, k = lstsq(X, y)
    W = [[1.0, tc[t["turn"]]["note_chars"], tc[t["turn"]]["calls"]] for t in s["turn_log"] if t["turn"] != "checkpoint"]
    a, b, c = lstsq(W, [t["wall_s"] for t in s["turn_log"] if t["turn"] != "checkpoint"])
    srows = [json.loads(x) for x in (MATERIAL / f"seed-{seed}" / "transcript.jsonl").read_text().splitlines()
             if x.strip()]
    stc, cum, wall, hit = turn_chars(srows, cap), 0, 0.0, None
    for n in sorted(stc):
        cum += stc[n]["chars"]
        wall += a + b * stc[n]["note_chars"] + c * stc[n]["calls"]
        if offset + k * cum >= TRIGGER:
            hit = {"turn": n, "predicted_context_tokens": round(offset + k * cum), "cum_wall_s": round(wall)}
            break
    man = json.loads((MATERIAL / f"seed-{seed}" / "material.manifest.json").read_text())
    dc_turn = srows[man["decision_checkpoint"]["row_index"]]["turn"]
    wall_dc = sum(a + b * stc[n]["note_chars"] + c * stc[n]["calls"] for n in stc if n <= dc_turn)
    out = {"fit_context": {"O_tokens": round(offset), "tokens_per_char": round(k, 4), "points": len(y)},
           "fit_wall": {"a_s": round(a, 2), "s_per_note_char": round(b, 5), "s_per_call": round(c, 2)},
           "tool_output_cap_chars_observed": cap, "trigger": TRIGGER, "seed": seed, "first_trigger": hit,
           "decision_checkpoint_turn": dc_turn, "predicted_wall_to_decision_checkpoint_s": round(wall_dc),
           "predicted_context_at_decision_checkpoint_tokens_if_no_compaction":
               round(offset + k * sum(stc[n]["chars"] for n in stc if n <= dc_turn))}
    (TS / f"estimate-seed-{seed}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
