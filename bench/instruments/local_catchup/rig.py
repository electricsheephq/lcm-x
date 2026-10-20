"""Independent per-process stores/homes, shared immutable warm cache, staggered binds."""
from argparse import Namespace
from pathlib import Path
import shutil
import time

from .audit import audit, baseline, connect, progress as read_progress
from .catchup import launch, stop, wall
from .common import append, dump, foreign_files, immutable_cache, load, parser, safe_paths
from .corpus import build, pilot_rows
from .cpu import flags, supervisor


def screen_state(fractions):
    return "INVALID" if any(f >= 1 for f in fractions) else "INCOMPLETE"


def main():
    p = parser(__doc__)
    for name in ("scratch", "out", "model-cache"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--processes", type=int, default=14)
    p.add_argument("--rows", type=int, default=250000)
    p.add_argument("--shape", choices=("short", "mixed"), default="short")
    p.add_argument("--baseline", choices=("empty", "upgrade"), default="empty")
    p.add_argument("--max-wall", type=wall, default=86400)
    p.add_argument("--stagger", type=float, default=2.)
    p.add_argument("--peak-disk-gb", type=float, default=8.)
    p.add_argument("--screen16", action="store_true")
    flags(p)
    a = p.parse_args()
    safe_paths(a.scratch)
    immutable_cache(a.model_cache)
    a.out.mkdir(parents=True, exist_ok=False)
    a.scratch.mkdir(parents=True, exist_ok=False)
    n, duration = (16, 60) if a.screen16 else (a.processes, a.max_wall)
    pilot = build(a.lcmx_repo, a.hermes_source, a.scratch/"pilot.db", pilot_rows(1000, a.rows, a.shape), a.shape, a.baseline)
    estimate = n*3*(pilot["database_bytes"]+len(pilot["expected"]["chunks"])*2048)*a.rows/pilot["rows"]
    dump(a.out/"allocation.json", {"estimate_bytes": estimate, "peak_disk_gb": a.peak_disk_gb})
    if estimate > a.peak_disk_gb*2**30 or shutil.disk_usage(a.scratch).free < max(150, a.peak_disk_gb+20)*2**30:
        raise RuntimeError("rig disk budget stop")
    jobs, manifests, processes = [], [], []
    meter = supervisor(a, a.out/"cpu.jsonl")
    try:
        for i in range(n):
            root = a.scratch/str(i)
            root.mkdir()
            out = a.out/str(i)
            out.mkdir()
            db = root/"store.db"
            manifest = build(a.lcmx_repo, a.hermes_source, db, a.rows, a.shape, a.baseline)
            dump(out/"manifest.json", manifest)
            jobs.append(Namespace(**{**vars(a), "db": db, "home": root/"homes", "out": out,
                                    "manifest": out/"manifest.json"}))
            baseline(db, manifest)
            manifests.append(manifest)
        meter.start()
        for job in jobs:
            processes.append(launch(job, 0))
            meter.watch(processes[-1].pid)
            time.sleep(a.stagger)
        while not all((j.out/"worker-0.json").exists() for j in jobs):
            if meter.breach or any(q.poll() is not None for q in processes):
                raise RuntimeError(meter.breach or "worker exited before bind")
            time.sleep(1)
        started, reason, progress = time.monotonic(), "max_wall", []
        next_progress = 0.
        last_change, fingerprints = [started]*n, [None]*n
        while time.monotonic()-started < duration:
            breach = meter.breach
            if breach or any(q.poll() is not None for q in processes):
                reason = breach or "worker_exit"
                break
            import json
            events = [[json.loads(s) for s in (j.out/"events.jsonl").read_text().splitlines()]
                      if (j.out/"events.jsonl").exists() else [] for j in jobs]
            if any(e.get("violation") or e.get("foreground") or e.get("download") for es in events for e in es):
                reason = "foreground_or_network_stop"
                break
            if time.monotonic() >= next_progress:
                next_progress = time.monotonic()+30
                if foreign := [paths for j, q in zip(jobs, processes) if (paths := foreign_files(q.pid, j.home/"0", j.db, a.model_cache))]:
                    dump(a.out/"cross-profile.json", {"paths": foreign})
                    reason = "cross_profile_access"
                    break
                progress = [read_progress(j.db, m) for j, m in zip(jobs, manifests)]
                append(a.out/"progress.jsonl", {"elapsed_s": time.monotonic()-started, "per_process": progress})
                for i, r in enumerate(progress):
                    if r["complete"] and meter.windows[processes[i].pid][1] is None:
                        meter.windows[processes[i].pid][1] = time.monotonic()
                    fingerprint = str((r["classes"], r["scan"]))
                    if fingerprint != fingerprints[i]:
                        fingerprints[i], last_change[i] = fingerprint, time.monotonic()
                if any(not r["raw_unchanged"] or r["changed_vectors"] or r["integrity"] not in ("ok", "not_checked") for r in progress):
                    reason = "corruption_stop"
                    break
                if any(not r["complete"] and time.monotonic()-last_change[i] >= 600
                       and any(e.get("event") == "prepared" for e in events[i]) for i, r in enumerate(progress)):
                    reason = "stalled_provider_ready"
                    break
                if not a.screen16 and all(r["complete"] for r in progress):
                    reason = "complete"
                    break
            time.sleep(1)
    finally:
        for q in processes:
            stop(q)
        meter.close()
    progress = [audit(j.db, m) for j, m in zip(jobs, manifests)]
    for j in jobs:
        with connect(j.db) as c, __import__("sqlite3").connect(j.db.parent/"final.db") as dest:
            c.backup(dest)
    fractions = [sum(v["covered"] for v in r["classes"].values())/sum(v["expected"] for v in r["classes"].values()) for r in progress]
    state = screen_state(fractions) if a.screen16 and reason == "max_wall" else "COMPLETE" if reason == "complete" else "INCOMPLETE" if reason == "max_wall" else "STALLED" if reason == "stalled_provider_ready" else "ABORTED"
    cpu = meter.summary(state, n, started if a.screen16 else 0)
    worst = {k: max((r[k] or 0 for r in cpu["per_process"].values()), default=None)
             for k in ("mean_cores", "p95_rolling60_cores", "peak1s_cores", "peak_rss")}
    dump(a.out/"summary.json", {"state": state, "screen_only_never_gated": a.screen16, "reason": reason,
         "per_process_progress": progress, "per_process_bind": [load(j.out/"worker-0.json") for j in jobs],
         "worst_process_fraction": min(fractions), "worst_process_cpu_rss": worst, "cpu": cpu})


if __name__ == "__main__":
    main()
