#!/usr/bin/env python3
"""D1 LOCAL: pre-embed once, then isolated parallel shards for two pinned trees.

Run from a clean instrument checkout; product trees must also be clean. Model
files must already be staged. This command never fetches the corpus or models.

The prepared corpus must contain all 500 questions; shards must partition those
IDs, and the A/A-prime prepared corpus contains 100 questions. Supply two
ROLE=PATH product bindings. --output and --scratch must be fresh directories.
Default concurrency is three, with at most floor(8/3)=2 ORT threads per child.
The sealed SQLite cache and its hash manifest live in --output. Per-question
embedding wall time (including cache reads and queries), hits/misses and
corruptions live in the checkpoints; d1-receipt.json aggregates each arm.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pin(tree):
    if subprocess.check_output(["git", "-C", str(tree), "status", "--porcelain"]):
        raise ValueError("D1 requires clean product and instrument trees")
    return subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()


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
        return list(pool.map(launch, tasks))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", action="append", required=True, help="ROLE=clean-product-tree (twice)")
    for name in ("prepared-dir", "shards-dir", "aprime-dir", "output", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--thread-budget", type=int, default=8)
    args = parser.parse_args(argv)
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
    if len(load_shard_question_ids(args.shards_dir)) != 500 or len(load_shard_question_ids(args.aprime_dir)) != 100:
        parser.error("require 500 distinct shard IDs and 100 placebo IDs")
    args.output.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=False)
    cache = args.output.resolve() / "embeddings.sqlite"
    threads = args.thread_budget // args.jobs
    env = dict(os.environ, LCM_LONGMEMEVAL_EMBED_CACHE=str(cache),
               LCM_D1_ORT_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    common = ["--dataset-label", "m", "--provider", "fastembed", "--model", "BAAI/bge-small-en-v1.5"]
    started = time.monotonic()

    def invoke(role, name, command, mode):
        scratch = args.scratch.resolve() / role / name
        scratch.mkdir(parents=True)
        child_env = dict(env, LCM_D1_CACHE_MODE=mode, HERMES_HOME=str(scratch / "home"), TMPDIR=str(scratch))
        with (args.output / f"{role}-{name}.log").open("w") as log:
            return subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", products[role], pins[role], instrument, *command], cwd=products[role], env=child_env, stdout=log, stderr=subprocess.STDOUT).returncode

    # Enumerate the product's actual chunk/summary units in each tree, including
    # the placebo corpus. Only this serial phase is allowed to write the cache.
    for role in products:
        for name, prepared in (("preembed", args.prepared_dir), ("preembed-aa", args.aprime_dir)):
            command = ["prewarm-cache", "--prepared-dir", str(prepared.resolve()), "--shards-manifest", str(prepared.resolve()), *common]
            if invoke(role, name, command, "build"):
                raise SystemExit("pre-embed failed; inspect its isolated log")
    from benchmarking.d1_cache import seal
    manifest_hash = seal(cache)
    tasks = [(role, path.parent.name, path.parent) for role in products for path in shards]
    tasks += [(role, name, args.aprime_dir) for role in products for name in ("aa-A", "aa-Aprime")]

    def launch(task):
        role, name, prepared = task
        out = args.output.resolve() / role / name
        command = ["run", "--prepared-dir", str(prepared.resolve()), "--output", str(out), "--allow-external-output", "--dump-candidates", str(out / "candidates.jsonl"), *common]
        rc = invoke(role, name, command, "read")
        rows = [json.loads(line) for line in (out / "per_question_checkpoint.jsonl").read_text().splitlines()] if (out / "per_question_checkpoint.jsonl").exists() else []
        rows = [row for row in rows if "question_id" in row]
        hits = sum(row["embed_cache"]["hits"] for row in rows)
        misses = sum(row["embed_cache"]["misses"] for row in rows)
        return {"role": role, "arm": name, "exit_code": rc, "questions": len(rows),
                "embedding_wall_ms": sum(row.get("embedding", {}).get("wall_ms", 0) for row in rows),
                "cache_hits": hits, "cache_misses": misses, "cache_hit_rate": hits / (hits + misses) if hits + misses else None,
                "cache_corruptions": sum(row.get("embedding", {}).get("corruptions", 0) for row in rows), "cache_manifest_sha256": manifest_hash}

    arms = run_pool(tasks, args.jobs, launch)
    receipt = {"instrument_sha": instrument, "products": pins, "jobs": args.jobs,
               "ort_threads_per_shard_cap": threads, "thread_budget": args.thread_budget,
               "wall_seconds": time.monotonic() - started, "arms": arms,
               "not_proven": "Eight-hour acceptance requires the complete quiet-window run; recall equivalence is a separate gate."}
    (args.output / "d1-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return int(any(arm["exit_code"] for arm in arms))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(*sys.argv[2:])
    else:
        raise SystemExit(main())
