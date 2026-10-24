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

pm = json.loads((root / "arms/dependency-1186-upstream-main-native/cells.json").read_text())
by_cell = {c["cell"]: c for c in pm["cells"]}
assert by_cell["b-scanner-on-refusal"]["status"] == "EXPECTED_NEGATIVE"
assert by_cell["b-scanner-on-refusal"]["scanner_enabled"] is True
assert by_cell["b-scanner-on-refusal"]["update_refused"] is True
b, no_rebuild = by_cell["b"], by_cell["e-core-version-no-rebuild"]
assert b["interactive"] is False and b["status"] == no_rebuild["status"] == "COMPLETE"
assert b["before"] == b["after"] == no_rebuild["before"] == no_rebuild["after"]
assert b["fastembed_imported"] and b["core_enabled"] and b["companion_enabled"]
assert any(c["cell"] == "companion-scan" and c.get("route") == "after-core-update"
           and c["scanner_enabled"] and c["verdict"] == "safe" for c in pm["cells"])
for path in (root / "arms").glob("*/cells.json"):
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell"] == "companion-scan":
            assert cell["scanner_enabled"] and cell["verdict"] == "safe"
assert len(rows) >= 2
assert all(r["wall_seconds"] <= 600 and r["allocated_generation_bytes"] <= 3 * 1024**3 for r in rows)
assert all(len(r["new_generation_names"]) == 1 for r in rows)
assert [sum(r["leases"].values()) for r in rows] == [1, 2]
print("PM_b_and_e_no_rebuild PASS; scanner_on_refusal EXPECTED_NEGATIVE; companions_scanning_ON PASS")

paths = list((root / "arms").glob("*/cells.json"))
assert len(paths) == 4, paths
pm_installer = json.loads((root / "arms/dependency-1186-upstream-main-installer/cells.json").read_text())
profile = next(c for c in pm_installer["cells"] if c["cell"] == "a-profile-installer")
assert profile["status"] == "COMPLETE" and profile["root_unchanged"]
assert profile["core_enabled"] and profile["companion_enabled"] and profile["fastembed_imported"]
for arm in ("upstream-main-native", "upstream-main-installer"):
    folder = root / "arms" / ("dependency-1186-" + arm)
    assert any("Plugin installed but not enabled." in " ".join(p.read_text(errors="replace").split())
               and "Run `hermes plugins enable hermes-lcm-x-local-embeddings` to activate." in " ".join(p.read_text(errors="replace").split())
               for p in folder.glob("*.stdout")), arm
print("r5 PM profile installer/root preservation PASS; actual host hints on both PM routes PASS")
