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
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
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


def run_child(command, *, timeout, **kwargs):
    with subprocess.Popen(command, start_new_session=True, **kwargs) as child:
        try:
            return {"exit_code": child.wait(timeout=timeout), "timed_out": False}
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            return {"exit_code": 124, "timed_out": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", action="append", required=True, help="ROLE=clean-product-tree (twice)")
    for name in ("prepared-dir", "shards-dir", "aprime-dir", "output", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--thread-budget", type=int, default=8)
    parser.add_argument("--worker-timeout", type=float, default=14400)
    args = parser.parse_args(argv)
    if not 0 < args.worker_timeout < float("inf"):
        parser.error("worker-timeout must be positive and finite")
    if not 1 <= args.jobs <= args.thread_budget <= 8 or len(args.product) != 2:
        parser.error("require two roles and 1 <= jobs <= thread-budget <= 8")
    products = dict(item.split("=", 1) for item in args.product)
    if len(products) != 2 or any(not role.isidentifier() for role in products):
        parser.error("require two distinct identifier role names")
    products = {role: str(Path(tree).resolve()) for role, tree in products.items()}
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
    common = ["--dataset-label", "m", "--provider", "fastembed", "--model", "BAAI/bge-small-en-v1.5"]
    started = time.monotonic()

    def invoke(role, name, command, mode):
        scratch = args.scratch.resolve() / role / name
        scratch.mkdir(parents=True)
        child_env = dict(env, LCM_LONGMEMEVAL_EMBED_CACHE=str(caches[role]),
                         LCM_D1_PRODUCT_SHA=pins[role], LCM_D1_ROLE=role,
                         LCM_D1_THREAD_REPORT=str(args.output.resolve() / f"{role}-{name}-threads.json"),
                         LCM_D1_CACHE_MODE=mode, HERMES_HOME=str(scratch / "home"), TMPDIR=str(scratch))
        with (args.output / f"{role}-{name}.log").open("w") as log:
            return run_child([sys.executable, str(Path(__file__).resolve()), "--worker", products[role], pins[role], instrument, *command], cwd=products[role], env=child_env, stdout=log, stderr=subprocess.STDOUT, timeout=args.worker_timeout)

    # Enumerate the product's actual chunk/summary units in each tree, including
    # the placebo corpus. Only this serial phase is allowed to write the cache.
    from benchmarking.d1_cache import seal
    manifests, role_threads, preembed = {}, {}, []
    for role in products:
        caches[role].parent.mkdir()
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
            if status["timed_out"]:
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
    budget_ok = args.jobs * max((n for n in effective if n is not None), default=threads) <= args.thread_budget
    arms = run_pool([task for task in tasks if task[0] in manifests], args.jobs, launch) if budget_ok else []
    receipt = {"instrument_sha": instrument, "products": pins, "jobs": args.jobs,
               "cache_manifests": manifests, "role_threads": role_threads,
               "thread_budget_ok": budget_ok, "thread_budget_fully_known": len(effective) == 2 and None not in effective,
               "worker_timeout": args.worker_timeout, "preembed": preembed,
               "fresh_output_guard_already_existed": True,
               "ort_threads_per_shard_cap": threads, "thread_budget": args.thread_budget,
               "wall_seconds": time.monotonic() - started, "arms": arms,
               "not_proven": "Eight-hour acceptance requires the complete quiet-window run; recall equivalence is a separate gate."}
    (args.output / "d1-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return int(not budget_ok or any(row["exit_code"] for row in preembed) or any(arm["exit_code"] or arm["parse_error"] for arm in arms))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(*sys.argv[2:])
    else:
        raise SystemExit(main())
