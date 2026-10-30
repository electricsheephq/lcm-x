#!/usr/bin/env python3
"""D1 LOCAL: pre-embed once, then isolated parallel shards for two pinned trees.

Run from a clean instrument checkout; product trees must also be clean. Model
files must already be staged. This command never fetches the corpus or models.

The prepared corpus must contain all 500 questions; shards must partition those
IDs, and the A/A-prime prepared corpus contains 100 questions. Supply two
ROLE=PATH product bindings. --output and --scratch must be fresh directories.
Default concurrency is three; request floor(8/3)=2 ORT threads per child.
Older product defaults are retained; the receipt reports effective thread use.
Each role's sealed SQLite cache and hash manifest live in --output. Per-question
embedding wall time (including cache reads and queries), hits/misses and
corruptions live in embedding-metrics.jsonl; d1-receipt.json aggregates each arm.
--seed-cache ROLE=PATH starts a role's serial build from a copy of a committed cache.
"""
import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pin(tree):
    if subprocess.check_output(["git", "-C", str(tree), "status", "--porcelain"], timeout=60):
        raise ValueError("D1 requires clean product and instrument trees")
    return subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True, timeout=60).strip()


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seed_record(path):
    """Validate a committed D1 cache read-only; return its receipt record."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"seed cache is not an existing file: {path}")
    if any(Path(f"{path}{suffix}").exists() for suffix in ("-journal", "-wal", "-shm")):
        raise ValueError(f"seed cache has an uncommitted SQLite sidecar: {path}")
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"binding", "entries"} <= tables:
                raise ValueError(f"seed cache lacks the binding/entries tables: {path}")
            bindings = db.execute("SELECT identity FROM binding").fetchall()
            if len(bindings) != 1:
                raise ValueError(f"seed cache must hold exactly one binding row: {path}")
            if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError(f"seed cache failed quick_check: {path}")
            entries = db.execute("SELECT count(*) FROM entries").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"seed cache is not a readable SQLite cache: {path}: {exc}") from exc
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size,
            "entries": entries, "identity": bindings[0][0]}


def worker(product, expected, instrument, *argv):
    if pin(product) != expected or pin(ROOT) != instrument:
        raise ValueError("D1 product/instrument identity changed")
    spec = importlib.util.spec_from_file_location("hermes_lcm", Path(product) / "__init__.py", submodule_search_locations=[product])
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_lcm"] = module
    spec.loader.exec_module(module)
    sys.argv = [str(ROOT / "scripts/lcm_longmemeval.py"), *argv]
    runpy.run_path(sys.argv[0], run_name="__main__")


def run_pool(tasks, jobs, launch):
    # Every child has disjoint roots; collect ALL statuses even after a failure.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(launch, task): i for i, task in enumerate(tasks)}
        results = [None] * len(tasks)
        for future in as_completed(futures):
            results[futures[future]] = future.result()
        return results


def signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except (ProcessLookupError, PermissionError):
        pass  # The worker may exit between wait/timeout and the signal.


def run_child(command, *, timeout, interrupt=None, **kwargs):
    with subprocess.Popen(command, start_new_session=True, **kwargs) as child:
        deadline = time.monotonic() + timeout
        while True:
            if interrupt and interrupt[0]:
                signal_group(child.pid, interrupt[0])
                # Retain the group even if its leader exits before descendants.
                until = time.monotonic() + 2
                while time.monotonic() < until:
                    child.poll()  # Reap the leader before probing its group.
                    try:
                        os.killpg(child.pid, 0)
                    except ProcessLookupError:
                        break
                    except PermissionError:
                        pass
                    time.sleep(0.05)
                signal_group(child.pid, signal.SIGKILL)
                child.wait()
                return {"exit_code": 128 + interrupt[0], "timed_out": False}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                signal_group(child.pid, signal.SIGKILL)
                child.wait()
                return {"exit_code": 124, "timed_out": True}
            try:
                exit_code = child.wait(timeout=min(remaining, 0.1))
                if interrupt and interrupt[0]:
                    continue  # Clean the group even though its leader has exited.
                return {"exit_code": exit_code, "timed_out": False}
            except subprocess.TimeoutExpired:
                continue


def main(argv=None):
    interrupt = [0]

    def interrupted(signum, frame):
        if not interrupt[0]:
            interrupt[0] = signum

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return _main(argv, interrupt)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _main(argv, interrupt):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", action="append", required=True, help="ROLE=clean-product-tree (twice)")
    for name in ("prepared-dir", "shards-dir", "aprime-dir", "output", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--build-workers", type=int, default=1)
    parser.add_argument("--thread-budget", type=int, default=8)
    parser.add_argument("--worker-timeout", type=float, default=14400)
    parser.add_argument("--seed-cache", action="append", default=[], metavar="ROLE=PATH",
                        help="start ROLE's serial build from a copy of this committed cache (repeatable; --build-workers 1)")
    args = parser.parse_args(argv)
    if not 0 < args.worker_timeout < float("inf"):
        parser.error("worker-timeout must be positive and finite")
    if not 1 <= args.jobs <= args.thread_budget <= 8 or len(args.product) != 2:
        parser.error("require two roles and 1 <= jobs <= thread-budget <= 8")
    if args.build_workers < 1 or (args.build_workers > 1 and 2 * args.build_workers > args.thread_budget):
        parser.error("require build-workers >= 1 and 2 * build-workers <= thread-budget for parallel builds")
    products = dict(item.split("=", 1) for item in args.product)
    if len(products) != 2 or any(not role.isidentifier() for role in products):
        parser.error("require two distinct identifier role names")
    products = {role: str(Path(tree).resolve()) for role, tree in products.items()}
    seeds = {}
    for item in args.seed_cache:
        role, _, path = item.partition("=")
        if role not in products or not path:
            parser.error(f"--seed-cache needs ROLE=PATH with a --product role: {item}")
        if role in seeds:
            parser.error(f"--seed-cache allows one seed per role: {role}")
        if args.build_workers != 1:
            parser.error("seeded builds are serial only: partitioned builds change batch membership")
        try:
            seeds[role] = seed_record(path)
        except ValueError as exc:
            parser.error(str(exc))
    instrument = pin(ROOT)
    pins = {role: pin(tree) for role, tree in products.items()}
    shards = sorted(args.shards_dir.glob("shard-*/manifest.json"))
    if len(shards) != 6:
        parser.error("require six prepared shard manifests")
    from benchmarking.longmemeval import load_shard_question_ids
    try:
        prepared = set(load_shard_question_ids(args.prepared_dir / "manifest.json"))
        selected = set(load_shard_question_ids(args.shards_dir))
        aprime = set(load_shard_question_ids(args.aprime_dir / "manifest.json"))
        if len(prepared) != 500 or selected != prepared or len(aprime) != 100 or not aprime <= prepared:
            raise ValueError("require shards to partition the 500 prepared IDs exactly and 100 A-prime IDs to be a subset")
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error("output must be a fresh directory; existing output cannot be resumed")
    args.output.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=False)
    caches = {role: args.output.resolve() / role / "embeddings.sqlite" for role in products}
    threads = args.thread_budget // args.jobs
    env = dict(os.environ, LCM_D1_ORT_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("LCM_D1_BUILD_PARTITION", None)  # Only parallel build workers get a partition.
    common = ["--dataset-label", "m", "--provider", "fastembed", "--model", "BAAI/bge-small-en-v1.5"]
    started = time.monotonic()

    def invoke(role, name, command, mode, part=None):
        if interrupt[0]:
            return {"exit_code": 128 + interrupt[0], "timed_out": False}
        scratch = args.scratch.resolve() / role / name
        scratch.mkdir(parents=True)
        child_env = dict(env, LCM_LONGMEMEVAL_EMBED_CACHE=str(caches[role]),
                         LCM_D1_PRODUCT_SHA=pins[role], LCM_D1_ROLE=role,
                         LCM_D1_THREAD_REPORT=str(args.output.resolve() / f"{role}-{name}-threads.json"),
                         LCM_D1_CACHE_MODE=mode, HERMES_HOME=str(scratch / "home"), TMPDIR=str(scratch))
        if part is not None:
            child_env.update(LCM_LONGMEMEVAL_EMBED_CACHE=str(caches[role].parent / "parts" / f"part-{part}.sqlite"),
                             LCM_D1_BUILD_PARTITION=f"{part}/{args.build_workers}",
                             LCM_D1_ORT_THREADS="1", OMP_NUM_THREADS="1")
        with (args.output / f"{role}-{name}.log").open("w") as log:
            return run_child([sys.executable, str(Path(__file__).resolve()), "--worker", products[role], pins[role], instrument, *command], cwd=products[role], env=child_env, stdout=log, stderr=subprocess.STDOUT, timeout=args.worker_timeout, interrupt=interrupt)

    # Enumerate the product's actual chunk/summary units in each tree, including
    # the placebo corpus. Only this pre-embed phase is allowed to write the cache.
    from benchmarking.d1_cache import merge, seal
    manifests, role_threads, build_threads, preembed = {}, {}, {}, []
    if args.build_workers == 1:
        for role, seed in seeds.items():  # Copy and re-hash every seed before any worker starts.
            caches[role].parent.mkdir()
            shutil.copyfile(seed["path"], caches[role])
            caches[role].chmod(0o644)
            if file_sha256(caches[role]) != seed["sha256"]:
                raise ValueError(f"seed cache copy hash mismatch for {role}")
        for role in products:
            if interrupt[0]:
                break
            caches[role].parent.mkdir(exist_ok=role in seeds)
            for name, prepared in (("preembed", args.prepared_dir), ("preembed-aa", args.aprime_dir)):
                command = ["prewarm-cache", "--prepared-dir", str(prepared.resolve()), "--shards-manifest", str(prepared.resolve()), *common]
                result = invoke(role, name, command, "build")
                preembed.append({"role": role, "arm": name, **result})
                if result["exit_code"]:
                    break
                role_threads[role] = json.loads((args.output / f"{role}-{name}-threads.json").read_text())
            if preembed[-1]["exit_code"]:
                continue
            manifests[role] = seal(caches[role])
    else:
        for role in products:
            (caches[role].parent / "parts").mkdir(parents=True)

        def build(task):
            role, part = task
            rows = []
            for name, prepared in (("preembed", args.prepared_dir), ("preembed-aa", args.aprime_dir)):
                command = ["prewarm-cache", "--prepared-dir", str(prepared.resolve()), "--shards-manifest", str(prepared.resolve()), *common]
                began = time.monotonic()
                result = invoke(role, f"{name}-part-{part}", command, "build", part=part)
                rows.append({"role": role, "part": part, "arm": name, **result, "elapsed_s": time.monotonic() - began})
                if result["exit_code"] or interrupt[0]:
                    break
            return rows

        builds = run_pool([(role, part) for role in products for part in range(args.build_workers)], 2 * args.build_workers, build)
        preembed = [row for rows in builds for row in rows]
        merged_entries, duplicate_keys = {}, {}
        for role in products:
            rows = [row for row in preembed if row["role"] == role]
            if interrupt[0] or any(row["exit_code"] for row in rows):
                continue
            # Build workers run one ORT thread, so their reports stay out of role_threads (the shard setting).
            build_threads[role] = [json.loads((args.output / f"{role}-{row['arm']}-part-{row['part']}-threads.json").read_text()) for row in rows]
            parts = [caches[role].parent / "parts" / f"part-{part}.sqlite" for part in range(args.build_workers)]
            merged_entries[role], duplicate_keys[role] = merge(parts, caches[role])
            manifests[role] = seal(caches[role])
    tasks = [(role, path.parent.name, path.parent) for role in products for path in shards]
    tasks += [(role, name, args.aprime_dir) for role in products for name in ("aa-A", "aa-Aprime")]

    def launch(task):
        role, name, prepared = task
        out = args.output.resolve() / role / name
        status = {"role": role, "arm": name, "exit_code": 1, "parse_error": None,
                  "timed_out": False, "product_sha": pins[role], "cache_manifest_sha256": manifests[role]}
        try:
            command = ["run", "--prepared-dir", str(prepared.resolve()), "--output", str(out), "--allow-external-output", "--dump-candidates", str(out / "candidates.jsonl"), *common]
            status.update(invoke(role, name, command, "read"))
            if status["timed_out"] or interrupt[0]:
                return status
            rows = [json.loads(line) for line in (out / "per_question_checkpoint.jsonl").read_text().splitlines()]
            metrics = [json.loads(line) for line in (out / "embedding-metrics.jsonl").read_text().splitlines()]
            hits, misses = (sum(row[field] for row in metrics) for field in ("hits", "misses"))
            status.update(questions=sum("question_id" in row for row in rows),
                          embedding_wall_ms=sum(row["wall_ms"] for row in metrics),
                          cache_hits=hits, cache_misses=misses,
                          cache_hit_rate=hits / (hits + misses) if hits + misses else None,
                          cache_corruptions=sum(row["corruptions"] for row in metrics))
        except Exception as exc:
            status["parse_error"] = f"{type(exc).__name__}: {exc}"
        return status

    effective = [value["effective_intra_op_threads"] for value in role_threads.values()]
    built = [row["effective_intra_op_threads"] for rows in build_threads.values() for row in rows]
    # One-thread build reports still decide the shard verdict: a capped product stays within threads at the shard
    # setting, and an uncapped one reports the same count there. They do not make the budget fully known.
    shard_ok = args.jobs * max((n for n in effective + built if n is not None), default=threads) <= args.thread_budget
    # The build runs 2 * P workers at once, so its own verdict uses that concurrency.
    build_ok = not built or 2 * args.build_workers * max((n for n in built if n is not None), default=1) <= args.thread_budget
    budget_ok = shard_ok and build_ok
    arms = run_pool([task for task in tasks if task[0] in manifests], args.jobs, launch) if budget_ok and not interrupt[0] else []
    receipt = {"instrument_sha": instrument, "products": pins, "jobs": args.jobs,
               "cache_manifests": manifests, "role_threads": role_threads,
               "thread_budget_ok": budget_ok, "build_thread_budget_ok": build_ok, "shard_thread_budget_ok": shard_ok,
               "thread_budget_fully_known": len(effective) == 2 and None not in effective,
               "worker_timeout": args.worker_timeout, "preembed": preembed, "interrupted": bool(interrupt[0]),
               "fresh_output_guard_already_existed": True,
               "ort_threads_per_shard_cap": threads, "thread_budget": args.thread_budget,
               "wall_seconds": time.monotonic() - started, "arms": arms,
               "not_proven": "Eight-hour acceptance requires the complete quiet-window run; recall equivalence is a separate gate."}
    if seeds:
        receipt["seeds"] = seeds
    if args.build_workers > 1:
        receipt.update(build_workers=args.build_workers, merged_entries=merged_entries, duplicate_keys=duplicate_keys, build_threads=build_threads)
    (args.output / "d1-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 128 + interrupt[0] if interrupt[0] else int(not budget_ok or any(row["exit_code"] for row in preembed) or any(arm["exit_code"] or arm["parse_error"] for arm in arms))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(*sys.argv[2:])
    else:
        raise SystemExit(main())
