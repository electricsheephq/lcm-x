"""One-second process CPU-time sampler; 1.0 CPU = one core, never ps %cpu."""
from collections import defaultdict
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import threading

from .common import append


def percentile(values, p=.95):
    return sorted(values)[max(0, math.ceil(len(values)*p)-1)] if values else None


def rolling(samples, window=60):
    result, selected, pos = [], [], 0
    if not samples:
        return result
    for end in range(math.ceil(samples[0][0]-samples[0][1]+window), math.floor(samples[-1][0])+1):
        while pos < len(samples) and samples[pos][0]-samples[pos][1] < end:
            selected.append(samples[pos])
            pos += 1
        selected = [s for s in selected if s[0] > end-window]
        clipped = clip(selected, end-window, end)
        if sum(dt for _, dt, _ in clipped) >= window-1e-6:
            result.append(sum(dt*c for _, dt, c in clipped)/window)
    return result


def clip(samples, lo, hi):
    return [(min(t, hi), min(t, hi)-max(t-dt, lo), c) for t, dt, c in samples
            if min(t, hi) > max(t-dt, lo)]


def statistics(samples):
    duration = sum(dt for _, dt, _ in samples)
    return {"mean_cores": sum(dt*c for _, dt, c in samples)/duration if duration else None,
            "p95_rolling60_cores": percentile(rolling(samples)),
            "peak1s_cores": max((c for _, _, c in samples), default=None), "seconds": duration}


def process(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        t = p.cpu_times()
        return t.user+t.system, p.memory_info().rss, p.num_threads()
    except psutil.Error as exc:
        raise ProcessLookupError(pid) from exc


def swap():
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        amount, unit = re.search(r"used\s*=\s*([\d.]+)([MG])", out).groups()
        return int(float(amount)*{"M": 2**20, "G": 2**30}[unit])
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return (int(fields["SwapTotal"].split()[0])-int(fields["SwapFree"].split()[0]))*1024


class Supervisor:
    def __init__(self, path, cap=12., sustained=10., per_mean=.6, per_p95=.75,
                 aggregate_mean=7., aggregate_p95=8.75):
        self.path, self.cap, self.sustained = path, cap, sustained
        self.limits = (per_mean, per_p95, aggregate_mean, aggregate_p95)
        self.previous, self.samples, self.rss, self.threads = {}, defaultdict(list), defaultdict(list), defaultdict(list)
        self.excess, self.swap_start = 0., swap()
        self.aggregate, self.windows, self.loads = [], {}, []
        self.breach, self.halt, self.lock = None, threading.Event(), threading.RLock()

    def watch(self, pid):
        with self.lock:
            now = time.monotonic()
            self.windows[pid], self.previous[pid] = [now, None], (now, 0.)

    def start(self):
        __import__("psutil")  # Mandatory before launching workers.
        self.swap_start = swap()
        def loop():
            deadline = time.monotonic()
            while not self.halt.is_set():
                try:
                    with self.lock:
                        self.breach = self.sample(tuple(self.windows)) or self.breach
                except Exception as exc:
                    self.breach = f"sampler_error:{type(exc).__name__}"
                deadline += 1
                self.halt.wait(max(0, deadline-time.monotonic()))
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        self.halt.set()
        self.thread.join()


    def sample(self, pids):
        now, total, intervals, raw = time.monotonic(), 0., [], {}
        for pid in pids:
            try:
                cpu, rss, threads = process(pid)
            except (OSError, ProcessLookupError):
                continue
            if pid in self.previous:
                then, previous_cpu = self.previous[pid]
                dt = now-then
                cores = max(0, cpu-previous_cpu)/dt
                self.samples[pid].append((now, dt, cores))
                self.rss[pid].append((now, dt, rss))
                self.threads[pid].append((now, dt, threads))
                total += cores
                intervals.append(dt)
                raw[str(pid)] = {"cores": cores, "rss": rss, "threads": threads,
                                 "cpu_seconds": cpu, "dt": dt}
            self.previous[pid] = (now, cpu)
        dt = min(intervals, default=0.)
        self.aggregate.append((now, dt, total))
        self.excess = self.excess+dt if total > self.cap else 0.
        used = swap()
        self.loads.append(__import__("os").getloadavg()[0])
        append(self.path, {"at": now, "processes": raw, "aggregate_cores": total,
                          "swap_bytes": used, "load": __import__("os").getloadavg()})
        return "swap_increased" if used > self.swap_start else "sustained_cpu_breach" if self.excess >= self.sustained-1e-3 else None

    def summary(self, state="INCOMPLETE", n=None, since=0, end=None, aggregate_window=None):
        end = time.monotonic() if end is None else end
        windows = {p: (max(since, w[0]), min(w[1] if w[1] is not None else end, end)) for p, w in self.windows.items()}
        active = {p: clip(s, *windows.get(p, (since, end))) for p, s in self.samples.items()}
        lo, hi = max((w[0] for w in windows.values()), default=since), min((w[1] for w in windows.values()), default=end)
        if aggregate_window is not None:
            lo, hi = aggregate_window
        max_dt = max((dt for s in self.samples.values() for _, dt, _ in s), default=0)
        per = {str(p): dict(statistics(s), peak_rss=max((v for _, _, v in clip(self.rss[p], *windows.get(p, (since, end)))), default=None),
                           peak_threads=max((v for _, _, v in clip(self.threads[p], *windows.get(p, (since, end)))), default=None), tail=statistics(clip(self.samples[p], windows.get(p, (since, end))[1], end)))
               for p, s in active.items()}
        agg = statistics(clip(self.aggregate, lo, hi))
        def bar(s, mean, p95):
            if state not in ("COMPLETE", "INCOMPLETE"):
                return "N/A"
            if max_dt > 2:
                return "INSUFFICIENT_CADENCE"
            if s["p95_rolling60_cores"] is None:
                return "INSUFFICIENT_60S"
            return "PASS" if s["mean_cores"] <= mean and s["p95_rolling60_cores"] <= p95 else "FAIL"
        return {"per_process": per, "aggregate": agg,
                "per_process_bars": {p: bar(s, *self.limits[:2]) for p, s in per.items()},
                "aggregate_bar": bar(agg, *self.limits[2:]) if (n or len(windows)) == 14 else "NOT_APPLICABLE", "limits": self.limits,
                "tail": statistics(clip(self.aggregate, hi, end)), "max_sample_dt": max_dt,
                "host_load_peak1m": max(self.loads, default=0),
                "non_quiet_window": max(self.loads, default=0) > (__import__("os").cpu_count() or 1)}


def flags(p):
    for name, default in (("cpu-cap", 12), ("cpu-sustained", 10), ("per-mean", .6),
                          ("per-p95", .75), ("aggregate-mean", 7), ("aggregate-p95", 8.75)):
        p.add_argument("--"+name, type=float, default=default)


def supervisor(a, path):
    return Supervisor(path, a.cpu_cap, a.cpu_sustained, a.per_mean, a.per_p95,
                      a.aggregate_mean, a.aggregate_p95)
