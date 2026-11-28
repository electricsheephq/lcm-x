"""Record host interference outside timed windows; percentages never gate a run."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

FIELDS = "user nice system idle iowait irq softirq steal".split()


def snapshot():
    result = {"utc": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic(),
              "loadavg": None, "cpu_count": None, "counters": None, "reason": None}
    for key, read in (("loadavg", lambda: os.getloadavg()), ("cpu_count", os.cpu_count)):
        try:
            result[key] = read()
        except Exception:
            pass
    try:
        if sys.platform != "linux":
            raise ValueError("/proc/stat unavailable off Linux")
        line = Path("/proc/stat").read_text().splitlines()[0].split()
        if line[0] != "cpu" or len(line) < 9:
            raise ValueError("short or missing aggregate cpu line")
        counters = dict(zip(FIELDS, map(int, line[1:9])))
        if min(counters.values()) < 0:
            raise ValueError("negative cpu counter")
        result["counters"] = counters
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def delta(before, after):
    loads = [s["loadavg"][0] if s.get("loadavg") else None for s in (before, after)]
    result = {"elapsed_s": after["monotonic"] - before["monotonic"],
              "load1m_before": loads[0], "load1m_after": loads[1],
              "host_load_peak1m": max((v for v in loads if v is not None), default=None),
              "steal_pct": None, "iowait_pct": None, "busy_pct": None}
    if before.get("counters") is not None and after.get("counters") is not None:
        counts = {k: after["counters"][k] - before["counters"][k] for k in FIELDS}
        total = sum(counts.values())  # Excludes guest counters, already counted in user/nice.
        if total > 0 and min(counts.values()) >= 0:
            for name, value in (("steal", counts["steal"]), ("iowait", counts["iowait"]),
                                ("busy", total - counts["idle"] - counts["iowait"] - counts["steal"])):
                result[name + "_pct"] = 100 * value / total
        else:
            result["reason"] = "zero total or regressed cpu counters"
    else:
        result["reason"] = after.get("reason") or before.get("reason") or "cpu counters unavailable"
    return result


def maxima(runs):
    return {key: max((row["host"][field] for row in runs if row.get("host", {}).get(field) is not None),
                     default=None) for key, field in (("max_steal_pct", "steal_pct"),
                                                      ("host_load_peak1m", "host_load_peak1m"))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "delta"))
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    if len(args.files) != (2 if args.mode == "delta" else 0):
        parser.error("delta needs two snapshot files; snapshot takes none")
    print(json.dumps(snapshot() if args.mode == "snapshot" else
                     delta(*(json.loads(p.read_text()) for p in args.files))))
