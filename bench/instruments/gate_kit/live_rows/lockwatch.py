"""External read-only watcher of SQLite's WAL write lock (busy-bar's F_GETLK probe), transitions only.

Busy-bar's monitor keeps every 0.15 ms sample; a cold start can hold the lock for minutes, so this one keeps only
the transitions (time, holder pid) plus the sample count and the largest sampling gap. A hold is a run of samples
with a holder; gaps shorter than one poll are merged as busy-bar does.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from bench.instruments.busy_bar import monitor  # noqa: E402

POLL_NS = monitor.POLL_NS


def holds(transitions: list[list[int]], begin: int, end: int, pid: int | None = None,
          poll_ns: int = POLL_NS) -> list[tuple[int, int]]:
    """Hold periods clipped to [begin, end]; with `pid`, only holds by that process (or an unknown holder)."""
    found: list[tuple[int, int]] = []
    opened = None
    for at, owner in transitions:
        mine = bool(owner) and (pid is None or owner in (pid, -1))
        if mine and opened is None:
            opened = at
        elif not mine and opened is not None:
            found.append((opened, at))
            opened = None
    if opened is not None:
        found.append((opened, transitions[-1][0]))
    merged: list[tuple[int, int]] = []
    for start, stop in found:
        if merged and start - merged[-1][1] < poll_ns:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return [(max(s, begin), min(e, end)) for s, e in merged if s < end and e > begin]


def longest_s(transitions: list[list[int]], begin: int, end: int, pid: int | None = None) -> float:
    return max(((e - s) / 1e9 for s, e in holds(transitions, begin, end, pid)), default=0.0)


class Watcher:
    def __init__(self, database: Path) -> None:
        monitor.flock_type()
        self.process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), str(database) + "-shm"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
        if not select.select([self.process.stdout], [], [], 20)[0] or self.process.stdout.readline() != "READY\n":
            self.process.kill()
            raise RuntimeError("lock watcher did not start: " + (self.process.communicate()[1] or "")[:300])

    def stop(self) -> dict[str, Any]:
        output, error = self.process.communicate("stop\n", timeout=60)
        if self.process.returncode:
            raise RuntimeError(f"lock watcher failed: {error[:300]}")
        return json.loads(output)


def main() -> None:
    fd = os.open(sys.argv[1], os.O_RDONLY)
    transitions: list[list[int]] = []
    samples, last, largest_gap, state = 0, None, 0, None
    try:
        while True:
            owner = monitor.lock_owner(fd) or 0
            now = time.monotonic_ns()
            samples += 1
            if last is not None:
                largest_gap = max(largest_gap, now - last)
            last = now
            if owner != state:
                transitions.append([now, owner])
                state = owner
            if samples == 1:
                print("READY", flush=True)
            if select.select([sys.stdin], [], [], 0)[0]:
                break
            time.sleep(POLL_NS / 1e9)
        transitions.append([last, state or 0])
        print(json.dumps({"transitions": transitions, "samples": samples, "poll_ns": POLL_NS,
                          "largest_gap_ms": largest_gap / 1e6, "platform": sys.platform}))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
