#!/usr/bin/env python3
"""S8: score every decision run at cp-176 and cp-304 (score_s.py), classify each lost fact (loss_class.py), and aggregate
per checkpoint with report_s.py --seeds-expected 3. score_decision.py --run-root DIR --material DIR --out DIR [--seeds 1 2 3]. Runs that are absent are
skipped here and show up as INCOMPLETE in the reports. S4 probes only at the end of its run (row 304): cp-176 is UNSUPPORTED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

H = Path(__file__).resolve().parent
T = H.parent
SCORER = T / "scorer"
sys.path.insert(0, str(SCORER))
sys.path.insert(0, str(H))
sys.path.insert(0, str(T))
import checkpoints as CP  # noqa: E402
from loss_class import classify_loss  # noqa: E402
import score_manifest  # noqa: E402
from score_s import score, accounting, jload  # noqa: E402

CPS = (176, 304)


def runs(n: int, args):
    """(arm, run label, run dir, checkpoint -> probe dir or None)"""
    s = f"seed-{n}"
    if (args.material / s / "lifecycle_probes.jsonl").exists():
        cps = {c["id"]: f"cp-{c['id']}" for c in CP.select(args.material / s, ",".join(args.checkpoints) if args.checkpoints else "lifecycle")}
        for arm in args.arms or ("L0", "L1", "L1-H", "L1-noptr", "H", "codex-native"):
            root = args.external_root / "codex-runs" / s if arm == "codex-native" else args.run_root / arm / s
            for rdir in sorted(root.glob("*")):
                if rdir.is_dir():
                    yield arm, rdir.name, rdir, cps
        return
    for k in (1, 2):
        r = f"d{n}-r{k}"
        for arm in (args.arms or ["LCMX-fleet", "LCMX-fleet-v2"]):
            if arm.startswith("LCMX") and not arm.endswith("-open"):
                yield arm, r, args.run_root / arm / s / r, {c: f"cp-{c}" for c in args.checkpoints}
        yield "lossless-claw", r, args.external_root / "lc-runs" / s / f"{r}-default", {c: f"cp-{c}" for c in args.checkpoints}
        yield "lossless-claw-tuned", r, args.external_root / "lc-runs" / s / f"{r}-tuned", {c: f"cp-{c}" for c in args.checkpoints}
        yield "codex-native", r, args.external_root / "codex-runs" / s / r, {176: None, 304: "."}
    o = f"d{n}-open"
    yield "LCMX-fleet-open", o, args.run_root / "LCMX-fleet-open" / s / o, {c: f"cp-{c}" for c in args.checkpoints}
    yield "lossless-claw-open", o, args.external_root / "lc-runs" / s / f"{o}-default", {c: f"cp-{c}" for c in args.checkpoints}
    yield "lossless-claw-tuned-open", o, args.external_root / "lc-runs" / s / f"{o}-tuned", {c: f"cp-{c}" for c in args.checkpoints}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True, help="existing lcmx-runs directory")
    ap.add_argument("--external-root", type=Path, help="root containing lc-runs and codex-runs")
    ap.add_argument("--logs", type=Path, help="existing decision/logs wall receipts")
    ap.add_argument("--material", type=Path, default=os.environ.get("TRACK_S_MATERIAL"))
    ap.add_argument("--out", type=Path, default=Path(os.environ["TRACK_S_OUT"]) / "decision" if os.environ.get("TRACK_S_OUT") else None)
    ap.add_argument("--arms", nargs="+", help="score only these arms")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--checkpoints", nargs="+", default=None, help="v3 rows or v4 ids/lifecycle")
    args = ap.parse_args()
    if args.material is None or args.out is None:
        ap.error("provide --material/--out or TRACK_S_MATERIAL/TRACK_S_OUT")
    args.external_root = args.external_root or args.run_root.parent
    args.logs = args.logs or args.run_root.parent / "decision" / "logs"
    v4 = (args.material / f"seed-{args.seeds[0] if args.seeds else 1}" / "lifecycle_probes.jsonl").exists()
    args.seeds = args.seeds or list(range(1, 9) if v4 else range(1, 4))
    if not v4:
        args.checkpoints = [int(x) for arg in args.checkpoints for x in arg.split(",")] if args.checkpoints else list(CPS)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = score_manifest.load(manifest_path) or {"schema": score_manifest.SCHEMA, "entries": {}}
    done = []
    for n in args.seeds:
        for arm, label, rdir, cps in runs(n, args):
            if args.arms and arm not in args.arms:
                continue
            name = f"{arm}.seed-{n}.{label}.json"
            manifest["entries"] = {k: v for k, v in manifest["entries"].items() if Path(k).name != name}
            score_manifest.write_atomic(manifest_path, manifest)
            for population in ("scores", "loss"):
                for directory in (args.out / population).glob("cp-*"):
                    (directory / name).unlink(missing_ok=True)
            lane = "s2" if arm.startswith("LCMX") or arm in ("L0", "L1", "L1-H", "L1-noptr", "H") else "s4" if arm == "codex-native" else "s1"
            receipt = args.logs / f"{lane}-{arm}-{label}.log.wall"
            if not receipt.exists() or re.findall(r"^exit (\d+) end", receipt.read_text(), re.M) != ["0"]:
                continue  # missing/failed runs are never measured as completed
            run_calls, run_probes = [], {}
            for cp, sub in cps.items():
                if sub is None or not (rdir / sub / "summary.json").exists():
                    continue
                out = score(args.material / f"seed-{n}", rdir / sub, arm)
                if v4:
                    run_calls += jload(rdir / sub / "summary.json").get("reader_calls", [])
                    run_probes.update({f"{cp}/{pid}": p for pid, p in out["probes"].items()})
                if arm.startswith("LCMX"):
                    events = json.loads((rdir / sub / "summary.json").read_text())["events"]
                    recorded = {nd["node_id"]: nd for e in events for nd in e.get("nodes", [])}
                    with sqlite3.connect(f"file:{rdir / sub / 'store' / 'lcm.db'}?mode=ro", uri=True) as db:
                        tables = {t[0] for t in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                        nodes = list(db.execute("SELECT n.node_id, n.depth, p.escalation_level FROM summary_nodes n LEFT JOIN "
                                                "summary_node_provenance p USING(node_id)")) if "summary_node_provenance" in tables else [
                            (nd["node_id"], nd["depth"], nd.get("level")) for nd in recorded.values()]
                    l3 = [nid for nid, depth, level in nodes if level == 3]
                    verbatim = sum(recorded[nid].get("l3_kind") == "verbatim" for nid in l3)
                    out["stored_level3"] = {"nodes": len(nodes), "leaves": sum(d == 0 for _, d, _ in nodes),
                        "level3": len(l3), "verbatim": verbatim, "truncated": len(l3) - verbatim,
                        "histogram_returns": sum(v for e in events for k, v in e.get("level_histogram", {}).items() if k.endswith(":L3")),
                        "source": "summary_node_provenance.escalation_level" if "summary_node_provenance" in tables else "recorded stored node.level"}
                out["run"] = label  # the decision run id (writers record their own run field)
                (args.out / "scores" / f"cp-{cp}").mkdir(parents=True, exist_ok=True)
                (args.out / "loss" / f"cp-{cp}").mkdir(parents=True, exist_ok=True)
                for population, payload in (("scores", out), ("loss", classify_loss(out))):
                    file_path = args.out / population / f"cp-{cp}" / name
                    file_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
                    manifest["entries"][file_path.relative_to(args.out).as_posix()] = {
                        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                        "scored_at": datetime.now(timezone.utc).isoformat()}
                done.append(f"cp-{cp} {name}")
            if v4 and (rdir / "summary.json").exists():
                summ = jload(rdir / "summary.json")
                summ["accounting"] = accounting(dict(summ, reader_calls=run_calls), run_probes)
                (rdir / "summary.json").write_text(json.dumps(summ, indent=1))
            score_manifest.write_atomic(manifest_path, manifest)
    score_manifest.write_atomic(manifest_path, manifest)
    for cp in (sorted(p.name[3:] for p in (args.out / "scores").glob("cp-*")) if v4 else args.checkpoints):
        if (args.out / "scores" / f"cp-{cp}").exists():
            subprocess.run([sys.executable, str(SCORER / "report_s.py"), "--scores", str(args.out / "scores" / f"cp-{cp}"),
                            "--out", str(args.out / f"full-report-cp-{cp}.md"), "--json", str(args.out / f"full-report-cp-{cp}.json"),
                            "--seeds-expected", str(len(args.seeds)), "--title", f"Track S decision run, checkpoint row_index {cp}"], check=True)
    print("\n".join(done))


if __name__ == "__main__":
    main()
