"""Linux floor surrogate: taskset + verified cgroup v2 4 GiB, never a hardware claim."""
from argparse import Namespace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .catchup import run, wall
from .common import dump, isolated, parser
from .corpus import build

LABEL = "floor surrogate"


def command(arguments):
    return ["taskset", "-c", "0,1", sys.executable, "-B", "-m",
            "bench.instruments.local_catchup.floor", *arguments, "--limited"]


def verify_limits(memory_text, affinity_text, swap_text="max"):
    memory = int(memory_text.strip()) if memory_text.strip().isdigit() else None
    swap = int(swap_text.strip()) if swap_text.strip().isdigit() else None
    cpus = affinity_text.strip()
    if memory != 4*2**30 or swap != 0 or cpus not in ("0-1", "0,1"):
        raise RuntimeError("floor surrogate limits did not take effect")
    return dict(label=LABEL, memory_max_bytes=memory, memory_swap_max_bytes=swap, cpus=cpus, verified=True)


def measured_limits(cgroup):
    relative = next(line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    effective = Path("/sys/fs/cgroup")/relative.lstrip("/")
    if effective.resolve() != cgroup.resolve():
        raise RuntimeError("process did not enter requested cgroup")
    cpus = next(line.split(":", 1)[1].strip() for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Cpus_allowed_list:"))
    return verify_limits((effective/"memory.max").read_text(), cpus, (effective/"memory.swap.max").read_text())


def rate(summary, seconds, rows):
    speed = summary.get("rows_s", 0)
    hours = rows/speed/3600 if speed > 0 else None
    valid = (seconds >= 7200 and summary.get("state") == "INCOMPLETE" and rows >= 250000
             and summary.get("audit", {}).get("complete") is False)
    return dict(label=LABEL, state="INVALID" if not valid else "PASS" if hours is not None and hours <= 72 else "FAIL",
                sustained_rows_s=speed, implied_hours=hours, measured_seconds=seconds, bar_hours=72,
                validity="at least two hours, backlog present throughout, declared corpus size")


def label_results(root):
    for path in root.rglob("*"):
        if path.suffix == ".jsonl":
            rows = [dict(json.loads(line), label=LABEL) for line in path.read_text().splitlines()]
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        elif path.suffix == ".json":
            value = json.loads(path.read_text())
            if isinstance(value, dict):
                dump(path, dict(value, label=LABEL))


def main():
    p = parser(__doc__)
    for name in ("scratch", "out", "model-cache", "cgroup"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--cell", choices=("coverage-rate", "recall"), required=True)
    p.add_argument("--rows", type=int, default=250000)
    p.add_argument("--shape", choices=("short", "mixed"), default="short")
    p.add_argument("--baseline", choices=("empty", "upgrade"), default="empty")
    p.add_argument("--max-minutes", type=int, default=150)
    p.add_argument("--limited", action="store_true")
    a = p.parse_args()
    if sys.platform != "linux" or not 121 <= a.max_minutes <= 330:
        raise ValueError("Linux and max minutes in [121,330] required")
    if a.cell == "coverage-rate" and a.max_minutes < 150:
        raise ValueError("coverage-rate requires at least 150 minutes: build + 2h window + final audit")
    if not a.limited:
        proc = subprocess.Popen(command(sys.argv[1:]), start_new_session=True)
        timed_out = False
        try:
            try:
                code = proc.wait(timeout=a.max_minutes*60+120)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    code = None
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # Reap surviving grandchildren too.
            except ProcessLookupError:
                pass
            proc.wait()
        if timed_out:
            a.out.mkdir(parents=True, exist_ok=True)
            if not (a.out/"summary.json").exists():
                dump(a.out/"summary.json", dict(state="ABORTED", reason="outer_deadline", label=LABEL))
            label_results(a.out)
            raise SystemExit(3)
        if code:
            raise subprocess.CalledProcessError(code, proc.args)
        return
    a.out.mkdir(parents=True, exist_ok=False)
    def deadline(*_):
        raise TimeoutError("floor deadline")
    previous = signal.signal(signal.SIGTERM, deadline)
    alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(a.max_minutes*60)
    try:
        limited(a)
    except (TimeoutError, subprocess.TimeoutExpired):
        dump(a.out/"summary.json", dict(state="ABORTED", reason="floor_deadline", label=LABEL,
            shape=a.shape, baseline=a.baseline))
        raise SystemExit(3)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, previous)
        signal.signal(signal.SIGALRM, alarm)
        label_results(a.out)


def limited(a):
    dump(a.out/"limits.json", measured_limits(a.cgroup))
    if a.cell == "recall":
        subprocess.run([sys.executable, "-B", "-m", "bench.instruments.local_catchup.recall",
            "--lcmx-repo", str(a.lcmx_repo), "--hermes-source", str(a.hermes_source),
            "--scratch", str(a.scratch), "--out", str(a.out/"recall"), "--rows", str(a.rows),
            "--model-cache", str(a.model_cache), "--shape", a.shape, "--baseline", a.baseline,
            "--label", LABEL], check=True, timeout=a.max_minutes*60)
        summary = __import__("json").loads((a.out/"recall/summary.json").read_text())
    else:
        a.scratch.mkdir(parents=True, exist_ok=False)
        isolated(a.scratch/"builder-home", a.model_cache)
        os.environ["HOME"] = str(a.scratch/"builder-home")
        db = a.scratch/"store.db"
        manifest = build(a.lcmx_repo, a.hermes_source, db, a.rows, a.shape, a.baseline)
        dump(a.out/"manifest.json", manifest)
        start = time.monotonic()
        measured = run(Namespace(**{**vars(a), "db": db, "manifest": a.out/"manifest.json",
            "home": a.scratch/"homes", "out": a.out/"catchup", "max_wall": wall("2h"),
            "restart_fraction": 2., "restart_after": float("inf"), "progress_interval": 30,
            "cpu_cap": 1.5, "cpu_sustained": 10, "per_mean": .6, "per_p95": .75,
            "aggregate_mean": 7., "aggregate_p95": 8.75}))
        summary = rate(measured, time.monotonic()-start, a.rows)
    dump(a.out/"summary.json", dict(summary, label=LABEL, shape=a.shape, baseline=a.baseline))


if __name__ == "__main__":
    main()
