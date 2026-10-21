"""Recompute coverage verdict and zero-progress timer from immutable progress rows."""
import json
from pathlib import Path
from .catchup import state
from .common import load


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
    from .recall import score
    def lines(path):
        return [json.loads(s) for s in path.read_text().splitlines()]
    if (root/"summary.json").exists() and "cells" in load(root/"summary.json"):
        recorded = load(root/"summary.json")
        cells = [v2(root/c["baseline"]) for c in recorded["cells"]]
        assert recorded["state"] == ("PASS" if all(c["state"] == "PASS" for c in cells) else "FAIL")
        return dict(state=recorded["state"], cells=cells)
    if (root/"catchup").is_dir():
        recorded = load(root/"summary.json")
        computed = rate(load(root/"catchup/summary.json"), recorded["measured_seconds"],
            load(root/"manifest.json")["rows"], lines(root/"catchup/events.jsonl"))
        assert all(recorded[k] == v for k, v in computed.items())
    else:
        recorded = load(root/"raw-summary.json")
        request = load(root/"request.json")
        computed = score(lines(root/"recalls.jsonl"), lines(root/"publications.jsonl"),
            request["seed"], request["state"], recorded, request.get("standalone_calibration_s"),
            recorded["reference_standalone_s"])
        if recorded.get("measurement_valid") is False:
            computed["state"] = "INVALID"
        assert all(recorded[k] == v for k, v in computed.items())
    return computed


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        root = Path(name)
        if (root/"catchup").is_dir() or (root/"recalls.jsonl").exists() or "cells" in load(root/"summary.json"):
            print(json.dumps(v2(root)))
            continue
        summary, audit = load(root/"summary.json"), load(root/"final-audit.json")
        final = verdict(root/"progress.jsonl", summary["max_wall"])
        coverage = sum(v["covered"] for v in audit["classes"].values())/sum(v["expected"] for v in audit["classes"].values())
        assert final == summary["state"] and coverage == summary["coverage_fraction"]
        print(json.dumps({"cell": root.name, "state": final, "coverage_fraction": coverage}))
