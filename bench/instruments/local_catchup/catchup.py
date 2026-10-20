"""Single automatic bind per interpreter; restart once; audit outside foreground."""
import os
import hashlib
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading

from .audit import audit, baseline, progress
from .common import append, config, dump, foreign_files, guards, immutable_cache, isolated, load, parser, product, safe_paths
from .cpu import flags, supervisor


def state(audit_result, elapsed, idle, ready, max_wall):
    if audit_result["complete"]:
        return "COMPLETE"
    if elapsed >= max_wall:
        return "INCOMPLETE"
    if idle >= 600 and ready:
        return "STALLED"
    return "RUNNING"


def wall(value):
    return float(value[:-1])*{"s": 1, "m": 60, "h": 3600}[value[-1]] if value[-1].isalpha() else float(value)


def worker(a):
    safe_paths(a.home, a.db)
    isolated(a.home, a.model_cache)
    mod = product(a.lcmx_repo, a.hermes_source)
    guards(mod, a.out/"events.jsonl", a.generation)
    cfg = config(mod, a.db)
    dump(a.home/"local-settings.json", {"provider": cfg.embedding_provider,
         "model": cfg.embedding_model, "enabled": cfg.embeddings_enabled, "offline": True})
    engine = mod("engine").LCMEngine(config=cfg, hermes_home=str(a.home))
    started = time.monotonic()
    engine.on_session_start("instrument-bind", platform="cli", context_length=200000)
    bind_ms = (time.monotonic()-started)*1000
    dump(a.out/f"worker-{a.generation}.json", {"pid": os.getpid(), "bind_ms": bind_ms,
         "effective_config": __import__("dataclasses").asdict(cfg), "bind_started": started, "one_bind": True, "generation": a.generation})
    stop = False
    def interrupted(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, interrupted)
    while not stop:
        readiness = mod("embedding_provider").probe_provider_availability(cfg)
        last_pass = mod("embedding_maintenance").incremental_embedding_state(a.db)
        dump(a.out/f"status-{a.generation}.json", {"ready": readiness, "last_pass": last_pass})
        time.sleep(1)
    engine.shutdown()


def launch(a, generation):
    home = a.home/str(generation)
    command = [sys.executable, "-B", "-m", "bench.instruments.local_catchup.catchup", "--worker",
        "--lcmx-repo", str(a.lcmx_repo), "--hermes-source", str(a.hermes_source), "--db", str(a.db),
        "--manifest", str(a.manifest), "--home", str(home), "--model-cache", str(a.model_cache),
        "--out", str(a.out), "--scratch", str(a.scratch), "--generation", str(generation)]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, "HOME": str(home), "CFFIXED_USER_HOME": str(home)})
    def capture():
        with (a.out/f"worker-{generation}.log").open("wb") as log:
            __import__("shutil").copyfileobj(p.stdout, log)
    threading.Thread(target=capture, daemon=True).start()
    return p


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def restart_trigger(a, fraction, elapsed):
    if fraction >= 1:
        return None
    return "fraction" if fraction >= a.restart_fraction else "declared_time_fallback" if elapsed >= (a.restart_after if a.restart_after is not None else a.max_wall*.25) else None


def final_copy(a, generation):
    cell = hashlib.sha256(str(a.out.resolve()).encode()).hexdigest()[:16]
    return a.scratch/f"{a.out.name}-{cell}-g{generation}-final.db"


def restart_audit_failed(result):
    return (not result["raw_unchanged"] or result["changed_vectors"] or result["integrity"] != "ok"
            or any(result.get(k) for k in ("changed_metadata", "unexpected_rows", "unpaired_rows")))


def audit_failed(result):
    return restart_audit_failed(result) or result["residue"]


def run(a):
    safe_paths(a.home, a.db)
    initial_audit = baseline(a.db, load(a.manifest))
    immutable_cache(a.model_cache)
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = load(a.manifest)
    meter = supervisor(a, a.out/"cpu.jsonl")
    started, last_change, next_progress, generation = time.monotonic(), time.monotonic(), 0., 0
    previous, history, restarted, events = None, [], False, []
    meter.start()
    p = launch(a, generation)
    meter.watch(p.pid)
    reason, final = "", "RUNNING"
    try:
        while final == "RUNNING":
            now = time.monotonic()
            breach = meter.breach if p.poll() is None else None
            if breach or p.poll() is not None:
                final, reason = "ABORTED", breach or f"worker_exit:{p.returncode}"
                break
            events = [__import__("json").loads(s) for s in (a.out/"events.jsonl").read_text().splitlines()] if (a.out/"events.jsonl").exists() else []
            if any(e.get("violation") or e.get("foreground") or e.get("download") for e in events):
                final, reason = "ABORTED", "foreground_prepare_backfill_or_network"
                break
            if now >= next_progress:
                result = progress(a.db, manifest)
                if foreign := foreign_files(p.pid, a.home/str(generation), a.db, a.model_cache, out=a.out, scratch=final_copy(a, generation)):
                    dump(a.out/"cross-profile.json", {"paths": foreign})
                    final, reason = "ABORTED", "cross_profile_access"
                    break
                if not result["raw_unchanged"] or result["changed_vectors"] or result["integrity"] not in ("ok", "not_checked"):
                    final, reason = "ABORTED", "row_or_vector_corruption"
                    break
                fingerprint = (sum(x["covered"] for x in result["classes"].values()), str(result["scan"]))
                if fingerprint != previous:
                    last_change, previous = now, fingerprint
                total = sum(x["expected"] for x in result["classes"].values())
                fraction = fingerprint[0]/max(1, total)
                status_path = a.out/f"status-{generation}.json"
                status = load(status_path) if status_path.exists() else {}
                ready = status.get("ready", {}).get("available", False) and any(e.get("event") == "prepared" and e.get("generation") == generation for e in events)
                row = {"elapsed_s": now-started, "coverage_fraction": fraction, "audit": result, "status": status, "ready": ready, "generation": generation}
                append(a.out/"progress.jsonl", row)
                history.append(row)
                final = state(result, now-started, now-last_change, ready, a.max_wall)
                reason = "full_expected_id_audit" if final == "COMPLETE" else "zero_progress_600s_provider_ready" if final == "STALLED" else "max_wall" if final == "INCOMPLETE" else ""
                if result["complete"]:
                    meter.windows[p.pid][1] = now
                trigger = restart_trigger(a, fraction, now-started)
                if final == "RUNNING" and not restarted and trigger:
                    checked = audit(a.db, manifest)
                    dump(a.out/"restart-audit.json", checked)
                    if restart_audit_failed(checked):
                        final, reason = "ABORTED", "restart_corruption"
                        break
                    append(a.out/"restarts.jsonl", {"elapsed_s": now-started, "fraction": fraction,
                        "reason": trigger})
                    meter.windows[p.pid][1] = time.monotonic()
                    stop(p)
                    generation, restarted = 1, True
                    p = launch(a, generation)
                    meter.watch(p.pid)
                    last_change, previous = time.monotonic(), None
                    final = "RUNNING"
                next_progress = now+a.progress_interval
            time.sleep(1)
    finally:
        loop_end = time.monotonic()
        stop(p)
        meter.close()
    events = [__import__("json").loads(s) for s in (a.out/"events.jsonl").read_text().splitlines()] if (a.out/"events.jsonl").exists() else []
    result = audit(a.db, manifest)
    if audit_failed(result):
        final, reason = "ABORTED", "final_integrity_or_residue"
    elif final == "COMPLETE" and not result["complete"]:
        final, reason = "ABORTED", "final_coverage_regressed"
    elapsed = time.monotonic()-started
    covered = sum(x["covered"] for x in result["classes"].values())
    initial = sum(initial_audit["classes"][k]["covered"] for k in result["classes"])
    first = initial_audit
    summary = {"state": final, "reason": reason, "audit": result, "restarted": restarted, "restart": "once" if restarted else "none", "max_wall": a.max_wall,
        "end_to_end_s": elapsed, "coverage_fraction": covered/max(1, sum(x["expected"] for x in result["classes"].values())),
        "new_ids_s": (covered-initial)/elapsed, "cpu": meter.summary(final, 1, end=loop_end), "effective_config": (load(a.out/f"worker-{generation}.json")["effective_config"] if (a.out/f"worker-{generation}.json").exists() else None), "rows": manifest["rows"]}
    summary.update(rows_s=(result["covered_rows"]-first["covered_rows"])/elapsed,
        chunks_s=(result["classes"]["chunks"]["covered"]-first["classes"]["chunks"]["covered"])/elapsed,
        tokens_s=sum(v["covered_tokens"]-first["classes"][k]["covered_tokens"] for k, v in result["classes"].items())/elapsed,
        model_preparation_s=sum(e.get("seconds", 0) for e in events if e.get("event") == "prepared"))
    from .audit import connect
    with connect(a.db) as c, __import__("sqlite3").connect(final_copy(a, generation)) as dest:
        c.backup(dest)
    dump(a.out/"final-audit.json", result)
    dump(a.out/"manifest.json", manifest)
    dump(a.out/"summary.json", summary)
    return summary


def arguments():
    p = parser(__doc__)
    for name in ("db", "manifest", "home", "model-cache", "out", "scratch"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--max-wall", type=wall, default=wall("24h"))
    p.add_argument("--restart-fraction", type=float, default=.4)
    p.add_argument("--restart-after", type=wall, default=None)
    p.add_argument("--progress-interval", type=float, default=30)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--generation", type=int, default=0)
    flags(p)
    return p.parse_args()


if __name__ == "__main__":
    a = arguments()
    worker(a) if a.worker else print(run(a)["state"])
