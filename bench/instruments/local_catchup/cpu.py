"""One-second process CPU-time sampler; 1.0 CPU = one core, never ps %cpu."""
from collections import defaultdict, deque
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from .common import append


def percentile(values, p=.95):
    return sorted(values)[max(0, math.ceil(len(values)*p)-1)] if values else None


def rolling(samples, window=60):
    result, selected = [], deque()
    duration, integral = 0., 0.
    for t, dt, cpu in samples:
        selected.append((t, dt, cpu))
        duration, integral = duration+dt, integral+dt*cpu
        while selected and selected[0][0] <= t-window:
            _, old_dt, old_cpu = selected.popleft()
            duration, integral = duration-old_dt, integral-old_dt*old_cpu
        if selected and selected[0][0]-selected[0][1] < t-window:
            end, old_dt, old_cpu = selected.popleft()
            removed = old_dt-(end-(t-window))
            duration, integral = duration-removed, integral-removed*old_cpu
            selected.appendleft((end, old_dt-removed, old_cpu))
        if duration >= window-1e-6:
            result.append(integral/duration)
    return result


def statistics(samples):
    duration = sum(dt for _, dt, _ in samples)
    return {"mean_cores": sum(dt*c for _, dt, c in samples)/duration if duration else None,
            "p95_rolling60_cores": percentile(rolling(samples)),
            "peak1s_cores": max((c for _, _, c in samples), default=None), "seconds": duration}


def process(pid):
    try:
        import psutil
        try:
            p = psutil.Process(pid)
            t = p.cpu_times()
            return t.user+t.system, p.memory_info().rss, p.num_threads()
        except psutil.Error as exc:
            raise ProcessLookupError(pid) from exc
    except ImportError:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "time=,rss="], text=True).split()
        threads = (len(subprocess.check_output(["ps", "-M", "-p", str(pid)], text=True).splitlines())-1
            if sys.platform == "darwin" else int(subprocess.check_output(["ps", "-p", str(pid), "-o", "nlwp="], text=True)))
        parts = out[0].replace("-", ":").split(":")
        seconds = sum(float(v)*m for v, m in zip(reversed(parts), (1, 60, 3600, 86400)))
        return seconds, int(out[1])*1024, threads


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
        self.previous, self.samples, self.rss, self.threads = {}, defaultdict(list), defaultdict(int), defaultdict(int)
        self.excess, self.swap_start = 0., swap()
        self.aggregate = []

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
                total += cores
                intervals.append(dt)
                raw[str(pid)] = {"cores": cores, "rss": rss, "threads": threads,
                                 "cpu_seconds": cpu, "dt": dt}
            self.previous[pid] = (now, cpu)
            self.rss[pid], self.threads[pid] = max(self.rss[pid], rss), max(self.threads[pid], threads)
        dt = min(intervals, default=0.)
        self.aggregate.append((now, dt, total))
        self.excess = self.excess+dt if total > self.cap else 0.
        used = swap()
        append(self.path, {"at": now, "processes": raw, "aggregate_cores": total,
                          "swap_bytes": used, "load": __import__("os").getloadavg()})
        return "swap_increased" if used > self.swap_start else "sustained_cpu_breach" if self.excess >= self.sustained else None

    def summary(self):
        per = {str(p): dict(statistics(s), peak_rss=self.rss[p], peak_threads=self.threads[p])
               for p, s in self.samples.items()}
        agg = statistics(self.aggregate)
        def bar(s, mean, p95):
            if s["p95_rolling60_cores"] is None:
                return "INSUFFICIENT_60S"
            return "PASS" if s["mean_cores"] <= mean and s["p95_rolling60_cores"] <= p95 else "FAIL"
        return {"per_process": per, "aggregate": agg,
                "per_process_bars": {p: bar(s, *self.limits[:2]) for p, s in per.items()},
                "aggregate_bar": bar(agg, *self.limits[2:]), "limits": self.limits}


def flags(p):
    for name, default in (("cpu-cap", 12), ("cpu-sustained", 10), ("per-mean", .6),
                          ("per-p95", .75), ("aggregate-mean", 7), ("aggregate-p95", 8.75)):
        p.add_argument("--"+name, type=float, default=default)


def supervisor(a, path):
    return Supervisor(path, a.cpu_cap, a.cpu_sustained, a.per_mean, a.per_p95,
                      a.aggregate_mean, a.aggregate_p95)
