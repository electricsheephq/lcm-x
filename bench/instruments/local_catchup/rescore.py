"""Recompute coverage verdict and zero-progress timer from immutable progress rows."""
import json
from pathlib import Path
from .catchup import state
from .common import load


def verdict(path, max_wall):
    previous, last_change, generation, final = None, 0., 0, "RUNNING"
    for row in map(json.loads, path.read_text().splitlines()):
        fingerprint = str(([v["covered"] for v in row["audit"]["classes"].values()], row["audit"]["scan"]))
        if fingerprint != previous or row.get("generation", 0) != generation:
            last_change, previous, generation = row["elapsed_s"], fingerprint, row.get("generation", 0)
        final = state(row["audit"], row["elapsed_s"], row["elapsed_s"]-last_change, row["ready"], max_wall)
    return final


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        root = Path(name)
        summary, audit = load(root/"summary.json"), load(root/"final-audit.json")
        final = verdict(root/"progress.jsonl", summary["max_wall"])
        coverage = sum(v["covered"] for v in audit["classes"].values())/sum(v["expected"] for v in audit["classes"].values())
        assert final == summary["state"] and coverage == summary["coverage_fraction"]
        print(json.dumps({"cell": root.name, "state": final, "coverage_fraction": coverage}))
