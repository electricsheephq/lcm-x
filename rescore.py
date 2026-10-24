import json
from pathlib import Path
root = Path(__file__).resolve().parent
for path in sorted((root / "arms").glob("*/cells.json")):
    data = json.loads(path.read_text())
    print(path.parent.name, [(c["cell"], c["status"]) for c in data["cells"]])
rows = json.loads((root / "arms/dependency-1186-upstream-main-native/rebuild-measurements.json").read_text())
print("wall_seconds", [r["wall_seconds"] for r in rows])
print("wall_seconds_mean", sum(r["wall_seconds"] for r in rows) / len(rows))
print("allocated_generation_bytes", [r["allocated_generation_bytes"] for r in rows])
print("allocated_delta_bytes", [r["allocated_delta_bytes"] for r in rows])
print("retained_lease_counts", [sum(r["leases"].values()) for r in rows])
