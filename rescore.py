"""Recompute public timing cells; full oracle re-score inputs use the recorded T2 tag."""
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
read = lambda name: json.loads((root / name).read_text())
summary = {}
for family, filename, arms in (
    ("stream", "timing-stream.json", ("first", "cancel")),
    ("cloud", "timing-cloud.json", ("before", "after")),
):
    data = read(filename)
    summary[family] = {}
    for arm in arms:
        summary[family][arm] = {}
        for tree, samples in data[arm]["samples_ms"].items():
            quartiles = statistics.quantiles(samples, n=4)
            summary[family][arm][tree] = {
                "p50_ms": statistics.median(samples),
                "iqr_ms": quartiles[2] - quartiles[0],
                "n": len(samples),
            }
        if family == "stream" or arm == "after":
            assert summary[family][arm]["head"]["p50_ms"] <= 2 * summary[family][arm]["ga"]["p50_ms"]
stream = read("timing-stream.json")
assert len({tuple(ids) for ids in stream["selected"].values()}) == 1
assert len(stream["selected"]["head"]) == 32
for comparator in ("specified-base", "pre-marker"):
    summary[comparator] = {}
    for arm, data in read(f"steady-cost-{comparator}.json").items():
        base, head = (statistics.median(data["samples_ms"][tree]) for tree in ("base", "head"))
        bound = max(base * 1.10, base + 0.05)
        assert head <= bound
        summary[comparator][arm] = {"base_p50_ms": base, "head_p50_ms": head, "bound_ms": bound}
print(json.dumps(summary, indent=2))
print("PASS: cloud, stream, same 32 selected chunks, and all steady-cost timing cells")
