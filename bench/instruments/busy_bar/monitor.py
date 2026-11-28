"""Observe SQLite's unix-VFS WAL write byte with F_GETLK in a separate process."""

from __future__ import annotations

import ctypes
import fcntl
import json
import os
import platform
import select
import statistics
import subprocess
import sys
import time
from pathlib import Path
from functools import lru_cache
from typing import Any

POLL_NS = 150_000  # 0.15 ms; actual intervals are retained, not assumed.
WAL_WRITE_BYTE = 120  # UNIX_SHM_BASE + WAL_WRITE_LOCK (120 + 0).


@lru_cache(maxsize=None)
def flock_type(platform_name: str = sys.platform) -> type[ctypes.Structure]:
    """Native 64-bit Darwin/Linux layouts; refuse unverified ABIs."""
    if (ctypes.sizeof(ctypes.c_void_p) != 8
            or platform.machine() not in ("arm64", "aarch64", "x86_64")):
        raise RuntimeError("unconfirmed struct flock ABI: requires a 64-bit host")
    if platform_name == "darwin":
        fields = [("start", ctypes.c_int64), ("length", ctypes.c_int64),
                  ("pid", ctypes.c_int32), ("type", ctypes.c_int16), ("whence", ctypes.c_int16)]
    elif platform_name.startswith("linux"):
        fields = [("type", ctypes.c_int16), ("whence", ctypes.c_int16),
                  ("start", ctypes.c_int64), ("length", ctypes.c_int64), ("pid", ctypes.c_int32)]
    else:
        raise RuntimeError(f"unconfirmed struct flock platform: {platform_name}")

    class Flock(ctypes.Structure):
        _fields_ = fields

    expected = (24, [0, 8, 16, 20, 22]) if platform_name == "darwin" else (32, [0, 2, 8, 16, 24])
    if (ctypes.sizeof(Flock), [getattr(Flock, name).offset for name, _ in fields]) != expected:
        raise RuntimeError("unconfirmed struct flock layout")
    return Flock


def lock_owner(fd: int) -> int | None:
    layout = flock_type()
    request = layout()
    request.start, request.length = WAL_WRITE_BYTE, 1
    request.type, request.whence = fcntl.F_WRLCK, os.SEEK_SET
    answer = layout.from_buffer_copy(fcntl.fcntl(fd, fcntl.F_GETLK, bytes(request)))
    if answer.type == fcntl.F_UNLCK:
        return None
    if answer.type != fcntl.F_WRLCK or answer.start != WAL_WRITE_BYTE or answer.length != 1:
        raise RuntimeError("unexpected WAL lock response; struct layout/lock definition unconfirmed")
    return answer.pid or -1  # Held even when the holder's PID is not visible (another PID namespace).


def held_periods(samples: list[list[int]], begin: int, end: int,
                 poll_ns: int = POLL_NS) -> list[tuple[int, int]]:
    """Clip sampled holds to a call and merge gaps strictly shorter than a poll."""
    found: list[tuple[int, int]] = []
    opened = None
    for at, owner in samples:
        if owner and opened is None:
            opened = at
        elif not owner and opened is not None:
            found.append((opened, at))
            opened = None
    if opened is not None:
        found.append((opened, samples[-1][0]))
    merged: list[tuple[int, int]] = []
    for start, stop in found:
        if merged and start - merged[-1][1] < poll_ns:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return [(max(start, begin), min(stop, end)) for start, stop in merged
            if start < end and stop > begin]


def intervals(samples: list[list[int]]) -> dict[str, float]:
    gaps = sorted((b[0] - a[0]) / 1e6 for a, b in zip(samples, samples[1:]))
    if not gaps:
        raise RuntimeError("monitor produced too few samples")
    return {"median_ms": statistics.median(gaps),
            "p99_ms": gaps[min(len(gaps) - 1, (99 * len(gaps) + 99) // 100 - 1)],
            "max_ms": max(gaps)}


class Monitor:
    def __init__(self, database: Path) -> None:
        flock_type()  # Fail before starting a child on an unknown platform.
        self.process = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), str(database) + "-shm"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if not select.select([self.process.stdout], [], [], 10)[0]:
            self.process.kill()
            self.process.communicate()
            raise RuntimeError("WAL monitor readiness timed out")
        ready = self.process.stdout.readline()
        if ready != "READY\n":
            _, error = self.process.communicate(timeout=10)
            raise RuntimeError(f"WAL monitor failed: {error}")

    def stop(self) -> dict[str, Any]:
        output, error = self.process.communicate("stop\n", timeout=10)
        if self.process.returncode:
            raise RuntimeError(f"WAL monitor failed: {error}")
        return json.loads(output)


def main() -> None:
    samples: list[list[int]] = []
    fd = os.open(sys.argv[1], os.O_RDONLY)
    try:
        while True:
            owner = lock_owner(fd)
            samples.append([time.monotonic_ns(), owner or 0])
            if len(samples) == 1:
                print("READY", flush=True)
            if select.select([sys.stdin], [], [], 0)[0]:
                break
            time.sleep(POLL_NS / 1e9)
        print(json.dumps({"samples": samples, "poll_ns": POLL_NS,
                          "platform": sys.platform, "intervals": intervals(samples)}))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
