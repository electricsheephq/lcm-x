"""I2 isolated serial cells, real host trace and shared I1 safety/CPU machinery."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

from .audit import audit, connect, preimage, progress
from .catchup import stop
from .common import REQUEST, append, config, dump, foreign_files, foreground_request, guards, immutable_cache, isolated, load, parser, product, safe_paths
from .corpus import build, pilot_rows
from .cpu import Supervisor

SMOKE = "functional smoke, shared host, not a bar result"


def arguments(description):
    p = parser(description)
    for name in ("scratch", "out", "model-cache"):
        p.add_argument("--"+name, type=Path, required=True)
    for name, default in (("rows", 250000), ("blocks", 3), ("seed", 1586), ("samples", 600), ("queries", 200)):
        p.add_argument("--"+name, type=int, default=default)
    p.add_argument("--shape", choices=("short", "mixed"), default="short")
    p.add_argument("--baseline", choices=("empty", "upgrade"), default="empty")
    p.add_argument("--max-seconds", type=float, default=18000)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--label", default="Mac")
    p.add_argument("--worker", type=Path)
    return p.parse_args()


def prepare(a):
    safe_paths(a.scratch, a.out, a.model_cache)
    immutable_cache(a.model_cache)
    if min(a.blocks, a.samples, a.queries) < 1 or a.rows < 2:
        raise ValueError("positive counts required")
    a.out.mkdir(parents=True, exist_ok=False)
    a.scratch.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(a.scratch).free < (150 if sys.platform == "darwin" else 26)*2**30:
        raise RuntimeError("disk headroom stop")
    a.user_home = str(Path.home())
    # Isolate corpus construction too, before any product import.
    isolated(a.scratch/"builder-home", a.model_cache)
    os.environ["HOME"] = str(a.scratch/"builder-home")
    pilot = build(a.lcmx_repo, a.hermes_source, a.scratch/"pilot.db", pilot_rows(1000, a.rows, a.shape), a.shape, a.baseline)
    estimate = 5*(pilot["database_bytes"]+len(pilot["expected"]["chunks"])*2048)*a.rows/pilot["rows"]
    if estimate > 6*2**30:
        raise RuntimeError("estimated live stores exceed 6 GB; declare a larger runner budget first")
    manifest = build(a.lcmx_repo, a.hermes_source, a.scratch/"baseline.db", a.rows, a.shape, a.baseline)
    dump(a.out/"manifest.json", manifest)
    return manifest


def cell(a, state, block, kind):
    if shutil.disk_usage(a.scratch).free < (150 if sys.platform == "darwin" else 26)*2**30:
        raise RuntimeError("disk headroom stop")
    root = a.out/f"{block}-{state}"
    root.mkdir()
    # Only baseline + one serial cell copy live; rescore uses the JSONL receipts.
    db = a.scratch/"cell.db"
    with connect(a.scratch/"baseline.db") as src, sqlite3.connect(db) as dst:
        src.backup(dst)
    spec = dict(vars(a), state=state, block=block, kind=kind, db=db, root=root,
                home=a.scratch/f"{block}-{state}-home")
    spec = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in spec.items()}
    dump(root/"request.json", spec)
    cmd = [sys.executable, "-B", "-m", f"bench.instruments.local_catchup.{kind}",
           "--worker", str(root/"request.json"), "--lcmx-repo", str(a.lcmx_repo),
           "--hermes-source", str(a.hermes_source), "--scratch", str(a.scratch),
           "--out", str(root), "--model-cache", str(a.model_cache)]
    meter = Supervisor(root/"cpu.jsonl")
    with (root/"worker.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "HOME": spec["home"], "CFFIXED_USER_HOME": spec["home"]})
        meter.watch(proc.pid)
        meter.start()
        started, reason = time.monotonic(), None
        try:
            while proc.poll() is None:
                if meter.breach or time.monotonic()-started > a.max_seconds:
                    reason = meter.breach or "cell_deadline"
                    break
                time.sleep(.2)
        finally:
            stop(proc)
            meter.close()
    reason = meter.breach or reason
    result = load(root/"raw-summary.json") if (root/"raw-summary.json").exists() else {}
    if reason or proc.returncode:
        result.update(state="ABORTED", reason=reason or f"worker_exit:{proc.returncode}")
    for path in root.glob("*.jsonl"):
        labelled = [dict(json.loads(line), label=SMOKE if a.smoke else a.label) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in labelled))
    result.update(label=SMOKE if a.smoke else a.label, cpu=meter.summary("NOT_JUDGED" if a.smoke else "INCOMPLETE"), exit=proc.returncode)
    dump(root/"summary.json", result)
    if result.get("state") == "ABORTED":
        raise RuntimeError(f"cell stopped: {root.name}: {result.get('reason')}")
    return result


def guarded_worker(worker, request):
    root = Path(load(request)["root"])
    events_path = root/"events.jsonl"
    event_offset = events_path.stat().st_size if events_path.exists() else 0
    try:
        worker(request)
    except RuntimeError as exc:
        reasons = {"foreground preparation/backfill/join or network": "foreground_preparation_backfill_join_or_network",
                   "offline subprocess launch": "offline_subprocess_launch",
                   "foreground/download preparation forbidden": "foreground_preparation_forbidden",
                   "foreground backfill forbidden": "foreground_backfill_forbidden",
                   "foreground embedding join forbidden": "foreground_embedding_join_forbidden",
                   "offline instrument denies network": "offline_network_forbidden"}
        if str(exc) not in reasons:
            raise
        events = []
        if events_path.exists():
            with events_path.open("rb") as stream:
                stream.seek(event_offset)
                events = [json.loads(s) for s in stream.read().splitlines()]
        event_reasons = {"network_attempt": "offline_network_forbidden", "subprocess_launch": "offline_subprocess_launch",
                         "foreground_join": "foreground_embedding_join_forbidden"}
        for event in events:
            key = event.get("violation") or event.get("event")
            if event.get("foreground"):
                if key == "prepare":
                    event_reasons[key] = "foreground_preparation_forbidden"
                elif key in ("_chunk_backfill_run", "_embedding_backfill_summary_run"):
                    event_reasons[key] = "foreground_backfill_forbidden"
            if key in event_reasons:
                reason = event_reasons[key]
                break
        else:
            reason = reasons.get(str(exc))
        if reason is None:
            raise
        dump(root/"raw-summary.json", dict(state="INVALID", reason=reason))


def runtime(request):
    from types import SimpleNamespace
    a = SimpleNamespace(**load(request))
    for key in ("db", "root", "home", "scratch", "out", "lcmx_repo", "hermes_source", "model_cache"):
        setattr(a, key, Path(getattr(a, key)))
    isolated(a.home, a.model_cache)
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    mod = product(a.lcmx_repo, a.hermes_source)
    guards(mod, a.root/"events.jsonl")
    original_join = threading.Thread.join
    def join(thread, *args, **kw):
        if REQUEST.get() and threading.current_thread() is threading.main_thread() and any(
                name in thread.name.lower() for name in ("embed", "backfill")):
            append(a.root/"events.jsonl", {"violation": "foreground_join", "thread": thread.name})
            raise RuntimeError("foreground embedding join forbidden")
        return original_join(thread, *args, **kw)
    threading.Thread.join = join
    cfg = config(mod, a.db) if a.state != "OFF" else mod("config").LCMConfig(database_path=str(a.db))
    if cfg.embeddings_enabled != (a.state != "OFF"):
        raise RuntimeError("OFF product defaults no longer disable embeddings")
    cfg = replace(cfg, fresh_tail_count=2, leaf_chunk_tokens=2000, context_threshold=.75,
        threshold_full_sweep_enabled=True, large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False, temporal_rollups_enabled=False,
        l3_truncate_tokens=2, incremental_max_depth=0, summary_spend_max_calls=0)
    # Same deterministic summary route as bgprep; embedding route remains real.
    mod("escalation")._invoke_summary_llm = lambda prompt, *args, **kw: (
        "Earlier turns " + hashlib.sha256(prompt.encode()).hexdigest()[:16] + ".\nExpand for details about: turns")
    start = time.perf_counter()
    engine = mod("engine").LCMEngine(config=cfg, hermes_home=str(a.home))
    engine.on_session_start("i2-trace", platform="eval", conversation_id="i2-trace", context_length=6000)
    bind = (time.perf_counter()-start)*1000
    dump(a.root/"config.json", dict(config=asdict(cfg), bind_ms=bind, ort_threads=mod("embedding_provider")._FASTEMBED_THREADS))
    a._ready = lambda: mod("embedding_provider").probe_provider_availability(cfg).get("available", False)
    return a, mod, engine, bind


def trace(engine, mod, count, seed, observe, check):
    from bench.instruments.compaction_probe.bgprep.replay import workload
    view, batch = [], []
    for i in range(count):
        if i % 64 == 0:
            batch = workload(seed+i//64)
        pair = batch[i % 64]
        view.append(pair[0])
        t0 = time.perf_counter()
        with foreground_request():
            engine.ingest(view)
            due = engine.should_compress_preflight(view)
            t1 = time.perf_counter()
            if due:
                view = engine.compress(view, current_tokens=mod("tokens").count_messages_tokens(view))
            t2 = time.perf_counter()
        observe(dict(turn=i, start=t0, end=t2, ingest_preflight_ms=(t1-t0)*1000,
                     dispatch_ms=(t2-t0)*1000, compression_due=bool(due)))
        view.append(pair[1])
        with foreground_request():
            engine.ingest(view)
        engine.last_prompt_tokens = mod("tokens").count_messages_tokens(view)
        engine.note_turn_complete()
        engine._prepare_leaves_after_turn(view)
        check()


def integrity(a, manifest):
    # New trace rows are expected. Audit the immutable corpus subset and old vectors.
    with connect(a.db) as c:
        c.execute("BEGIN")
        original = list(c.execute("SELECT * FROM messages WHERE store_id<=? ORDER BY store_id", (manifest["rows"],)))
        from .audit import digest
        if digest(original) != manifest["preimage"]["raw_digest"]:
            raise RuntimeError("corpus row corruption")
        now = preimage(c, len(manifest["expected"]["leaves"]))
        if now["provenance"] != manifest["preimage"]["provenance"]:
            raise RuntimeError("corpus provenance corruption")
        changed_metadata = [k for k, v in (manifest.get("observed_metadata", {}) | manifest["preimage"]["metadata"]).items() if now["metadata"].get(k) != v]
        checked = audit(a.db, manifest)
        dump(a.root/"final-integrity.json", dict(corpus_raw_digest=digest(original),
            expected_raw_digest=manifest["preimage"]["raw_digest"],
            unpaired_rows=checked["unpaired_rows"], residue=checked["residue"],
            changed_metadata=changed_metadata,
            vector_digest=hashlib.sha256(json.dumps(now["vectors"], sort_keys=True).encode()).hexdigest()))
        if any(now["vectors"].get(k) != v for k, v in (manifest.get("observed_vectors", {}) | manifest["preimage"]["vectors"]).items()):
            raise RuntimeError("pre-existing vector corruption")
        if changed_metadata:
            raise RuntimeError("pre-existing metadata corruption")
        if checked["unpaired_rows"] or checked["residue"]:
            raise RuntimeError("vector/metadata pairing or in-flight residue")
        if c.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("database corruption")


def safety(a):
    events = [json.loads(s) for s in (a.root/"events.jsonl").read_text().splitlines()] if (a.root/"events.jsonl").exists() else []
    if any(e.get("event") == "subprocess_launch" for e in events):
        raise RuntimeError("offline subprocess launch")
    if any(e.get("violation") or e.get("foreground") for e in events):
        raise RuntimeError("foreground preparation/backfill/join or network")
    foreign = foreign_files(os.getpid(), a.home, a.db, a.model_cache, user_home=Path(a.user_home),
                            out=a.root, protected_roots=(a.scratch,))
    source_reads, unexpected = [], []
    for value in foreign:
        path = Path(value)
        if path.suffix in (".py", ".pyc") and ".hermes" not in path.parts and any(
                path.is_relative_to(root) for root in (a.lcmx_repo, a.hermes_source)):
            source_reads.append(path.name)
        else:
            unexpected.append(hashlib.sha256(value.encode()).hexdigest())
    if source_reads or unexpected:
        append(a.root/"file-access.jsonl", dict(source_reads=source_reads, unexpected_path_hashes=unexpected))
    if unexpected:
        raise RuntimeError("cross-profile access")
    now = time.monotonic()
    if now >= getattr(a, "_next_progress", 0):
        a._next_progress = now+30
        if not hasattr(a, "_manifest"):
            a._manifest = load(a.out/"manifest.json")
        value = progress(a.db, a._manifest, False)
        fingerprint = str((value["classes"], value["scan"]))
        if fingerprint != getattr(a, "_fingerprint", None):
            a._fingerprint, a._changed = fingerprint, now
        append(a.root/"progress.jsonl", dict(at=now, progress=value))
        if not value["complete"] and now-a._changed >= 600 and a._ready():
            raise RuntimeError("zero progress for ten minutes while work remains")
    return {"events": "events.jsonl", "prepare_and_backfill_wrapped": True,
            "embedding_thread_join_wrapped": True, "python_socket_calls_denied":
            "connect, connect_ex, sendto, sendmsg, getaddrinfo, gethostbyname, gethostbyname_ex, gethostbyaddr, getnameinfo",
            "subprocess_launches_recorded": "subprocess.Popen", "subprocess_launches_denied": "subprocess.Popen",
            "native_and_subprocess_network_guarded": False, "violations": 0}


def catchup_activity(root, start, end, coverage_start, coverage_end, require_prepared=False):
    events = [json.loads(s) for s in (root/"events.jsonl").read_text().splitlines()] if (root/"events.jsonl").exists() else []
    prepared = [e for e in events if e.get("event") == "prepared" and e.get("foreground") is False]
    prior = any(e.get("at", float("inf")) <= start for e in prepared)
    during = any(start <= e.get("at", -1) <= end for e in prepared)
    runs = sum(e.get("event") in ("_embedding_backfill_summary_run", "_chunk_backfill_run")
        and e.get("foreground") is False and e.get("embedded", 0) > 0
        and start <= e.get("at", -1) <= e.get("end", -1) <= end for e in events)
    delta = sum(coverage_end.values())-sum(coverage_start.values())
    active = delta > 0 and (not require_prepared or prior or during)
    return dict(coverage_start=coverage_start, coverage_end=coverage_end, coverage_delta=delta,
        model_prepared_at_start=prior, background_prepared=during, active_backfills=runs,
        catchup_activity=active, activity_window=dict(start=start, end=end))


def wait_covered(a, mod, engine, manifest):
    """Explicit covered-store setup; never counted as autonomous catch-up evidence."""
    failures = []
    def fill():
        try:
            from .audit import vector_counts
            checked = audit(a.db, manifest)
            append(a.root/"coverage.jsonl", dict(at=time.monotonic(), phase="covered_setup", audit=checked))
            if checked["complete"]:
                return
            start, changed, previous = time.monotonic(), time.monotonic(), None
            while True:
                counts = vector_counts(a.db, manifest)
                if all(counts[k] >= len(v) for k, v in manifest["expected"].items()):
                    checked = audit(a.db, manifest)
                    append(a.root/"coverage.jsonl", dict(at=time.monotonic(), phase="covered_setup", audit=checked))
                    if not checked["complete"]:
                        raise RuntimeError("covered setup final audit failed")
                    return
                safety(a)
                fingerprint = str(counts)
                if fingerprint != previous:
                    changed, previous = time.monotonic(), fingerprint
                if time.monotonic()-changed >= 600 or time.monotonic()-start > a.max_seconds:
                    raise RuntimeError("covered setup deadline/no-progress stop")
                for name, extra in (("_embedding_backfill_summary_run", {}), ("_chunk_backfill_run", {"policy": "conversational", "incremental": True})):
                    getattr(mod("command"), name)(engine._config, a.db, apply=True, limit=32,
                        retry_uncertain=False, include_next_hint=False, **extra)
                time.sleep(.1)
        except Exception as exc:
            failures.append(type(exc).__name__)
    thread = threading.Thread(target=fill, name="i2-covered-setup")
    thread.start()
    thread.join(a.max_seconds+1)
    if thread.is_alive() or failures:
        raise RuntimeError(f"covered setup failed: {failures}")


def bootstrap(values, statistic, seed=1586, resamples=10000):
    rng = random.Random(seed)
    return sorted(statistic(rng.choices(values, k=len(values))) for _ in range(resamples))
