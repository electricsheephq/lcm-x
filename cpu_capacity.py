"""Process CPU capacity, respecting Linux placement and quota constraints."""
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def effective_cpus() -> float:
    """Minimum positive affinity/quota capacity; retain fractional CPU quotas."""
    capacities = []
    try:
        affinity = len(os.sched_getaffinity(0))
        if affinity > 0:
            capacities.append(float(affinity))
    except (AttributeError, OSError):
        pass

    def read(path):
        try:
            return Path(path).read_text()
        except (OSError, ValueError):
            return ""

    groups = []
    for line in read("/proc/self/cgroup").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            groups.append((parts[1].split(","), parts[2]))
    for line in read("/proc/self/mountinfo").splitlines():
        before, sep, after = line.partition(" - ")
        fields, mount = before.split(), after.split()
        if not sep or len(fields) < 6 or len(mount) < 3:
            continue
        v2 = mount[0] == "cgroup2"
        if not v2 and not (mount[0] == "cgroup" and "cpu" in mount[2].split(",")):
            continue
        # mountinfo escapes whitespace and backslashes in paths.
        def unescape(value):
            for code, char in (("040", " "), ("011", "\t"), ("012", "\n"), ("134", "\\")):
                value = value.replace("\\" + code, char)
            return value
        root, base = Path(unescape(fields[3])), Path(unescape(fields[4]))
        for controllers, group in groups:
            if not ((v2 and controllers == [""]) or (not v2 and "cpu" in controllers)):
                continue
            try:
                relative = Path(group).relative_to(root)
            except ValueError:
                continue
            current = base / relative
            # Every visible ancestor can impose a tighter quota than the leaf.
            while True:
                try:
                    values = read(current / "cpu.max").split() if v2 else [
                        read(current / "cpu.cfs_quota_us"), read(current / "cpu.cfs_period_us")]
                    quota, period = map(int, values)
                    if quota > 0 and period > 0:
                        capacities.append(quota / period)
                except (ValueError, TypeError, OverflowError):
                    pass  # unlimited, missing or malformed; try the parent
                if current == base:
                    break
                current = current.parent
    host = os.cpu_count()
    if host:
        capacities.append(float(host))
    return min(capacities) if capacities else 1.0
