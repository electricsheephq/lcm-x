"""Re-score retained alternating pairs without rerunning the benchmark."""
import json
from pathlib import Path
import statistics

rows = [json.loads(line) for line in Path(__file__).with_name("pairs.jsonl").read_text().splitlines()]
for level in ("INFO", "DEBUG"):
    for scope in ("preflight", "post_llm_call_ingest"):
        selected = [row for row in rows if row["level"] == level and row["scope"] == scope]
        assert len(selected) == 200
        a = statistics.median(row["baseline_ms"] for row in selected)
        b = statistics.median(row["candidate_ms"] for row in selected)
        delta = statistics.median(row["candidate_ms"] - row["baseline_ms"] for row in selected)
        passed = b - a <= max(a * .1, .05) and delta <= max(a * .1, .05)
        print(json.dumps(dict(scope=scope, level=level, n=200, baseline_ms=a, candidate_ms=b,
                              paired_delta_ms=delta, passed=passed)))
        assert passed
