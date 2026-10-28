"""Re-score the preserved per-admission samples without rebuilding a store."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
samples = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
summary = json.loads((root / "summary.json").read_text())
assert summary["protocol"] == "1880-admission-v2-steady"
for rows in (10_000, 100_000):
    for mode in ("absent", "mismatch"):
        arms = [[r["ms"] for r in samples if r["rows"] == rows and r["mode"] == mode and r["arm"] == arm]
                for arm in (0, 1)]
        assert all(len(arm) == 200 for arm in arms)
        base, candidate = map(statistics.median, arms)
        limit = max(base * 1.10, base + 0.05)
        assert candidate <= limit, (rows, mode, base, candidate, limit)
        print(f"PASS {rows} {mode}: base={base:.6f}ms candidate={candidate:.6f}ms limit={limit:.6f}ms n=200")
    for mode, count in (("held", 200), ("rebuilt-held", 20)):
        values = [r["ms"] for r in samples if r["rows"] == rows and r["mode"] == mode]
        assert len(values) == count
        cell = next(c for c in summary["cells"] if c["rows"] == rows and c["mode"] == mode)
        assert cell["model_calls"] == cell["metadata_writes"] == 0
        if mode == "held":
            assert cell["full_history_scans"] == 0
        p95 = sorted(values)[int((len(values) - 1) * 0.95)]
        assert p95 <= (50 if mode == "held" else 100)
        print(f"PASS {rows} {mode}: p50={statistics.median(values):.6f}ms p95={p95:.6f}ms n={count}")
