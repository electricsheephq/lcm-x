"""Recompute coverage verdict and zero-progress timer from immutable progress rows."""
import json
from pathlib import Path
from .catchup import state
from .common import REFERENCE_CALIBRATION_S, load, rollup


def verdict(path, max_wall):
    terminal = path.with_name("summary.json")
    if terminal.exists() and (recorded := load(terminal)).get("state") == "ABORTED" and recorded.get("reason"):
        return "ABORTED"
    if path.with_name("cross-profile.json").exists() and load(path.with_name("cross-profile.json")).get("paths"):
        return "ABORTED"
    previous, last_change, generation, final = None, 0., 0, "RUNNING"
    for row in map(json.loads, path.read_text().splitlines()):
        fingerprint = str(([v["covered"] for v in row["audit"]["classes"].values()], row["audit"]["scan"]))
        if fingerprint != previous or row.get("generation", 0) != generation:
            last_change, previous, generation = row["elapsed_s"], fingerprint, row.get("generation", 0)
        final = state(row["audit"], row["elapsed_s"], row["elapsed_s"]-last_change, row["ready"], max_wall)
    return final


def v2(root):
    from .floor import rate
    from .recall import score, window_calibration, overlapping
    from .measurement import catchup_activity
    def lines(path):
        return [json.loads(s) for s in path.read_text().splitlines()]
    for name in ("raw-summary.json", "summary.json"):
        if (root/name).exists():
            early = load(root/name)
            if early.get("state") == "INVALID" and "classes" not in early and early.get("reason") in (
                    "covered_store_incomplete", "foreground_preparation_backfill_join_or_network",
                    "offline_subprocess_launch", "foreground_preparation_forbidden", "foreground_backfill_forbidden",
                    "foreground_embedding_join_forbidden", "offline_network_forbidden"):
                return early
    if (root/"summary.json").exists() and "cells" in load(root/"summary.json"):
        recorded = load(root/"summary.json")
        cells = [v2(root/c["baseline"]) for c in recorded["cells"]]
        assert recorded["state"] == rollup(c["state"] for c in cells)
        return dict(state=recorded["state"], cells=cells)
    if (root/"summary.json").exists() and "blocks" in load(root/"summary.json"):
        recorded = load(root/"summary.json")
        base = root if (root/"0-OFF").exists() else root/"recall" if (root/"recall").is_dir() else root
        present = {p.name for p in base.iterdir() if p.is_dir() and any(p.name.endswith("-"+s) for s in ("OFF", "ON-catching-up", "ON-covered"))}
        cells = {name: v2(base/name) for name in sorted(present)}
        names = [f"{c['block']}-{c['state']}" for c in recorded["blocks"]]
        assert len(names) == len(set(names)) and set(names) == present
        assert all({f"{b}-{s}" for s in ("OFF", "ON-catching-up", "ON-covered")} <= present
                   for b in {name.split("-", 1)[0] for name in present})
        cells = [cells[name] for name in names]
        assert all(c["result"]["state"] == v["state"] for c, v in zip(recorded["blocks"], cells))
        assert recorded["state"] == rollup(c["state"] for c in cells)
        return dict(state=recorded["state"], blocks=cells)
    if (root/"catchup").is_dir():
        recorded = load(root/"summary.json")
        raw = load(root/"catchup/summary.json")
        seconds = raw.get("end_to_end_s")
        if seconds is None:
            seconds = max(r["elapsed_s"] for r in lines(root/"catchup/progress.jsonl"))
        computed = rate(raw, seconds,
            load(root/"manifest.json")["rows"], lines(root/"catchup/events.jsonl"))
        assert all(recorded[k] == v for k, v in computed.items())
    else:
        recorded = load(root/"raw-summary.json")
        request = load(root/"request.json")
        calibrated = window_calibration(lines(root.parent/f"{request['block']}-ON-catching-up/events.jsonl"))
        rows, pubs = lines(root/"recalls.jsonl"), lines(root/"publications.jsonl")
        counted = overlapping(rows, pubs)
        activity = catchup_activity(root, min(r["start"] for r in counted), max(r["end"] for r in counted),
            min(counted, key=lambda r: r["start"])["coverage_start"],
            max(counted, key=lambda r: r["end"])["coverage_end"]) if counted else dict(catchup_activity=False)
        raw = score(rows, pubs, request["seed"], request["state"], activity)
        final = load(root/"final-audit.json")
        invalid = (request["smoke"] or request["rows"] < 250000 or len(load(root.parent/"queries.json")) < 200
            or (request["state"] == "ON-catching-up" and (all(c["missing_count"] == 0 for c in final["classes"].values())
                or not all(r["incomplete"] for r in counted))))
        capped = any(e.get("event") == "volume_cap" for e in lines(root/"events.jsonl")) if (root/"events.jsonl").exists() else False
        if invalid or capped:
            raw["state"] = rollup([raw["state"], "INVALID"])
        valid = raw["state"] != "INVALID"
        computed = score(rows, pubs, request["seed"], request["state"], activity, calibrated, REFERENCE_CALIBRATION_S)
        computed.update(measurement_valid=valid, catchup_activity=activity["catchup_activity"])
        if capped:
            computed["reason"] = "volume_cap"
        if not valid:
            computed["state"] = rollup([computed["state"], "INVALID"])
        if calibrated is None and valid:
            computed["reason"] = "calibration_window_short"
        assert all(recorded[k] == v for k, v in computed.items())
    return computed


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        root = Path(name)
        if (root/"catchup").is_dir() or (root/"raw-summary.json").exists() or any(
                k in load(root/"summary.json") for k in ("cells", "blocks")) or load(root/"summary.json").get("state") == "INVALID":
            print(json.dumps(v2(root)))
            continue
        summary, audit = load(root/"summary.json"), load(root/"final-audit.json")
        final = verdict(root/"progress.jsonl", summary["max_wall"])
        coverage = sum(v["covered"] for v in audit["classes"].values())/sum(v["expected"] for v in audit["classes"].values())
        assert final == summary["state"] and coverage == summary["coverage_fraction"]
        print(json.dumps({"cell": root.name, "state": final, "coverage_fraction": coverage}))
