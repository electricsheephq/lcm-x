#!/usr/bin/env python3
"""S8 per-fact loss class (packet correction 4). loss_class.py <score json> -> JSON on stdout. Stdlib only; read-only.

For every fact the score marks as not kept, where its value is at the checkpoint:
  not-admitted             the value never reached the store (writer admission list)
  no-summary               not in any stored summary and not in the reader's view
  stored-not-delivered     in a stored summary the reader did not receive
  delivered-not-carried    in a summary the reader received; the answer did not carry it
  raw-view-not-carried     not in a received summary but in a raw row of the reader's view; the answer did not carry it
Values are matched after the kit's normalize() (casefold, no punctuation/whitespace). No corpus text is printed.
Sources: S2 cp-<r>/store/lcm.db summary_nodes + cp-<r>/assembled_context.json (summary blocks = `Summary (dN, node N)` headers);
S1 cp-<r>/lcm.db summaries + cp-<r>/reader-view.txt (a summary counts as received when its id or its first 200 normalized
characters are in the view). S4: no store (encrypted server-side summary) -> not classified.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scorer"))
from score_s import normalize  # noqa: E402


def ro(db: Path):
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def load(run: Path):
    """-> ([(summary_id, text, received)], view_text_normalized) or None."""
    if (run / "store" / "lcm.db").exists():  # S2
        ctx = json.loads((run / "assembled_context.json").read_text())
        view = "\n".join(m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content")) for m in ctx["view"])
        got = {int(n) for n in re.findall(r"Summary \(d\d+, node (\d+)\)\]", view)}
        with ro(run / "store" / "lcm.db") as c:
            sums = [(f"node {i}", s, i in got) for i, s in c.execute("SELECT node_id, summary FROM summary_nodes")]
        return sums, normalize(ctx.get("system", "") + "\n" + view)
    if (run / "reader-view.txt").exists() and (run / "lcm.db").exists():  # S1
        view = (run / "reader-view.txt").read_text()
        nv = normalize(view)
        with ro(run / "lcm.db") as c:
            sums = [(i, s, i in view or normalize(s)[:200] in nv) for i, s in c.execute("SELECT summary_id, content FROM summaries")]
        return sums, nv
    return None


def classify_loss(score: dict) -> dict:
    run, mat = Path(score["run_dir"]), Path(score["material"])
    facts = {f["id"]: f for f in json.loads((mat / "facts.json").read_text())}
    admitted_lost = set(score["metrics"]["facts_kept"].get("lost_before_compaction", {}).get("ids") or [])
    lost = [fid for fid, f in facts.items() if score["probes"].get(fid, {}).get("class") != "CORRECT" or fid in admitted_lost]
    data = load(run)
    out = []
    for fid in lost:
        f, v = facts[fid], normalize(facts[fid]["value"])
        rec = {"id": fid, "placement": f["placement"], "class": f["class"], "answer_class": score["probes"].get(fid, {}).get("class")}
        if fid in admitted_lost:
            rec["loss"] = "not-admitted"
        elif data is None:
            rec["loss"] = "unclassified (no store / reader view recorded)"
        else:
            sums, view = data
            hit = [(i, r) for i, s, r in sums if v and v in normalize(s)]
            rec["summaries_holding_value"] = [i for i, _ in hit]
            rec["loss"] = ("delivered-not-carried" if any(r for _, r in hit) else "raw-view-not-carried" if v in view
                           else "stored-not-delivered" if hit else "no-summary")
        out.append(rec)
    counts = {}
    for r in out:
        counts[r["loss"]] = counts.get(r["loss"], 0) + 1
    return {"arm": score["arm"], "seed": score["seed"], "run": score["run"], "checkpoint": score["checkpoint_row"],
            "lost": len(out), "counts": counts, "facts": out}


if __name__ == "__main__":
    print(json.dumps(classify_loss(json.loads(Path(sys.argv[1]).read_text())), indent=1))
