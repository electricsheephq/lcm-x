#!/usr/bin/env python3
"""Real-FastEmbed D1 byte identity: serial vs call-index P=4, with a key-hash P=4 negative control.

Runs only in the d1-shard-identity workflow. Every build goes through
prewarm_embedding_cache and D1Cache on a deterministic synthetic corpus, with
the model staged beforehand and one ORT thread per process. Hard checks:
call-index equals serial (seal, `caches`, `dispatch`), the plan log equals the
serial dispatch log, a seeded verify re-embed has no mismatch, and read-mode
ranked candidates are equal. The key-hash control is expected to differ; if it
does not, the job warns. This proves the mechanism on CI runners only, not on
the release Mac.
"""
import argparse
import importlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MODEL = "BAAI/bge-small-en-v1.5"
WORKERS = 4
SYLLABLES = ["ka", "lo", "mi", "ren", "tu", "sha", "vo", "ne", "dri", "pa", "zel", "or", "qui", "bes", "tan", "gu"]
PROOF_ENV = ("LCM_D1_BUILD_PARTITION", "LCM_D1_DISPATCH_LOG", "LCM_D1_VERIFY_CALLS", "LCM_D1_VERIFY_AGAINST", "LCM_D1_VERIFY_LOG")


def phrase(rng, count):
    return " ".join("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(1, 3))) for _ in range(count))


def corpus(questions, sessions):
    """About 2 * questions * sessions unique units; mostly short, some mid, rare long ones that force wide padding."""
    import benchmarking.longmemeval as lme

    rng = random.Random(20261031)
    for q in range(questions):
        haystack = []
        for s in range(sessions):
            roll = rng.random()
            count = rng.randint(120, 320) if roll < 0.002 else rng.randint(30, 90) if roll < 0.1 else int(rng.lognormvariate(1.8, 0.9)) + 1
            haystack.append([{"role": rng.choice(["user", "assistant"]), "content": f"{phrase(rng, count)} n{q}x{s}"}])
        yield lme.parse_question({
            "question_id": f"q{q}", "question_type": "single-session-user", "question": phrase(rng, 8), "answer": "a",
            "question_date": "2023-01-01", "haystack_dates": ["2023-01-01"] * sessions,
            "haystack_session_ids": [f"s{q}-{s}" for s in range(sessions)], "haystack_sessions": haystack,
            "answer_session_ids": [f"s{q}-0"],
        })


def keyhash(cache, i, count):
    """RR22's legacy partition: each worker sends only the keys it owns from every call."""
    lookup = cache._lookup

    def owned(keys):
        found = lookup(keys)
        found.update((key, [0.0] * cache.dim) for key in keys if int(key[:16], 16) % count != i)
        return found

    cache._lookup = owned
    return cache


def worker(args):
    import benchmarking.longmemeval as lme

    # The per-call deadline only bounds waiting, never the vectors; four parts beside a plan oversubscribe a 3-core runner.
    provider = lme.resolve_harness_provider("fastembed", MODEL, timeout=3600.0)
    if args.keyhash:
        keyhash(provider, *map(int, args.keyhash.split("/")))
    if not args.read:
        report = lme.prewarm_embedding_cache(corpus(args.questions, args.sessions), provider)
        Path(args.report).write_text(json.dumps({key: report[key] for key in ("unique_request_units", "already_cached", "populated")}))
        return 0
    import numpy as np

    keys = sorted(provider.expected)
    found = provider._lookup(keys)  # Checksums are checked against the sealed manifest.
    if len(found) != len(keys) or provider.corruptions:
        raise SystemExit("read-mode cache rows missing or corrupt")
    matrix = np.array([found[key] for key in keys])
    rng = random.Random(7)
    lines = [json.dumps({"model": MODEL, "units": len(keys)})]
    for n in range(50):
        scores = matrix @ np.array(provider.embed_query(phrase(rng, rng.randint(2, 12))))
        top = sorted(range(len(keys)), key=lambda j: (-scores[j], keys[j]))[:10]
        lines.append(json.dumps({"question_id": f"q{n}", "arms": {"dense": [keys[j] for j in top]}}))
    Path(args.read).write_text("\n".join(lines) + "\n")
    return 0


def run(work, base, specs, elapsed):
    """Launch every (name, env, extra argv) at once; return {name: exit code} and record each wall time."""
    children, began = {}, time.monotonic()
    for name, env, extra in specs:
        home = work / "home" / name
        home.mkdir(parents=True)
        log = (work / f"{name}.log").open("w")
        children[name] = (subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", *base, *extra], env=dict(env, HERMES_HOME=str(home)),
                                           stdout=log, stderr=subprocess.STDOUT), log)
    codes = {}
    while len(codes) < len(children):
        for name, (child, log) in children.items():
            if name in codes or child.poll() is None:
                continue
            codes[name], elapsed[name] = child.returncode, round(time.monotonic() - began, 1)
            log.close()
        time.sleep(1)
    for name in children:
        if codes[name]:
            print(f"::group::{name} log tail\n" + "\n".join((work / f"{name}.log").read_text().splitlines()[-40:]) + "\n::endgroup::")
    return codes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--questions", type=int, default=40)
        command.add_argument("--sessions", type=int, default=100)
    check = commands.choices["check"]
    check.add_argument("--model-cache", type=Path, required=True)
    check.add_argument("--work", type=Path, required=True)
    check.add_argument("--summary", type=Path)
    child = commands.choices["worker"]
    child.add_argument("--report")
    child.add_argument("--keyhash")
    child.add_argument("--read")
    args = parser.parse_args(argv)
    if args.command == "worker":
        return worker(args)
    from benchmarking.d1_cache import merge, seal

    compare = importlib.import_module("bench.tools.d1-shard-check.compare_d1_caches")
    spec = importlib.util.spec_from_file_location("d1_driver", ROOT / "scripts/lcm_d1_local.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    work = args.work.resolve()
    work.mkdir(parents=True)
    sha = os.environ.get("GITHUB_SHA", "")
    env = {key: value for key, value in os.environ.items() if key not in PROOF_ENV}
    # The driver's build env: one ORT thread, pinned hash seed, offline model files.
    env.update(LCM_D1_PRODUCT_SHA=sha if re.fullmatch(r"[0-9a-f]{40}", sha) else "0" * 40, LCM_D1_ROLE="ci", LCM_D1_ORT_THREADS="1",
               OMP_NUM_THREADS="1", PYTHONHASHSEED="0", HF_HUB_OFFLINE="1", LCM_LONGMEMEVAL_FASTEMBED_CACHE=str(args.model_cache.resolve()))
    base = ["--questions", str(args.questions), "--sessions", str(args.sessions)]

    def build(name, argv=(), **extra):
        return (name, dict(env, LCM_D1_CACHE_MODE="build", LCM_LONGMEMEVAL_EMBED_CACHE=str(work / f"{name}.sqlite"),
                           LCM_D1_DISPATCH_LOG=str(work / f"{name}.jsonl"), **extra), ["--report", str(work / f"{name}-report.json"), *argv])

    out, hard, elapsed = {}, {}, {}
    phases = [("serial", [build("serial")]),
              ("call_index", [build(f"ci-{i}", LCM_D1_BUILD_PARTITION=f"{i}/{WORKERS}") for i in range(WORKERS)] + [build("plan", LCM_D1_BUILD_PARTITION="plan")]),
              ("key_hash", [build(f"kh-{i}", ["--keyhash", f"{i}/{WORKERS}"]) for i in range(WORKERS)])]
    for phase, specs in phases:
        codes = run(work, base, specs, elapsed)
        print(f"{phase}: {json.dumps({name: elapsed[name] for name in codes})}", flush=True)
        hard[f"{phase}_workers_exit_0"] = not any(codes.values())
    if all(hard.values()):
        for label, prefix in (("call_index", "ci"), ("key_hash", "kh")):
            out[f"{label}_merge"] = merge([work / f"{prefix}-{i}.sqlite" for i in range(WORKERS)], work / f"{label}.sqlite")
        plan = compare.jsonl(work / "plan.jsonl")
        out["verify_calls"] = driver.verify_sample([row["call"] for row in plan], WORKERS)
        verify = build("verify", LCM_D1_VERIFY_CALLS=json.dumps(out["verify_calls"]), LCM_D1_VERIFY_AGAINST=str(work / "call_index.sqlite"),
                       LCM_D1_VERIFY_LOG=str(work / "verify-calls.jsonl"))
        hard["verify_worker_exit_0"] = not run(work, base, [verify], elapsed)["verify"]
        out["verify"], verified = compare.verify(work / "verify-calls.jsonl") if hard["verify_worker_exit_0"] else ({}, False)
        hard["verify_0_mismatches"] = verified and sorted(row["call"] for row in compare.jsonl(work / "verify-calls.jsonl")) == out["verify_calls"]
        out["seals"] = {name: seal(work / f"{name}.sqlite") for name in ("serial", "call_index", "key_hash")}
        out["plan_units"], out["serial_units"] = sum(row["sent"] for row in plan), json.loads((work / "serial-report.json").read_text())["unique_request_units"]
        out["caches_call_index"], hard["call_index_caches_exit_0"] = compare.caches(work / "serial.sqlite", work / "call_index.sqlite")
        hard["call_index_seal_equal"] = out["caches_call_index"]["seal_equal"]
        out["dispatch_call_index"], hard["call_index_dispatch_exit_0"] = compare.dispatch(work / "serial.jsonl", *[work / f"ci-{i}.jsonl" for i in range(WORKERS)])
        hard["plan_log_equals_serial_log"] = (work / "plan.jsonl").read_bytes() == (work / "serial.jsonl").read_bytes()
        hard["merged_entries_equal_plan_units"] = out["call_index_merge"][0] == out["plan_units"] == out["serial_units"]
        out["caches_key_hash"], key_hash_equal = compare.caches(work / "serial.sqlite", work / "key_hash.sqlite")
        # Read workers get their own names: a build worker already owns each build name's home and log.
        reads = [(f"read-{name}", dict(env, LCM_D1_CACHE_MODE="read", LCM_LONGMEMEVAL_EMBED_CACHE=str(work / f"{name}.sqlite")),
                  ["--read", str(work / f"{name}-candidates.jsonl")]) for name in ("serial", "call_index")]
        hard["read_workers_exit_0"] = not any(run(work, base, reads, elapsed).values())
        if hard["read_workers_exit_0"]:
            out["candidates"], hard["read_candidates_equal"] = compare.candidates(work / "serial-candidates.jsonl", work / "call_index-candidates.jsonl")
        out["negative_control_differs"] = not key_hash_equal
    steps = {"serial build": ["serial"], "plan": ["plan"], "call-index parts": [f"ci-{i}" for i in range(WORKERS)],
             "key-hash parts": [f"kh-{i}" for i in range(WORKERS)], "verify": ["verify"], "candidates": ["read-serial", "read-call_index"]}
    timings = {step: max((elapsed[name] for name in names if name in elapsed), default=None) for step, names in steps.items()}
    out.update(hard=hard, timings_s=timings, worker_wall_s=elapsed, passed=all(hard.values()) and "read_candidates_equal" in hard)
    (work / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if out.get("negative_control_differs") is False:
        print("::warning::key-hash P=4 negative control matched serial: this runner cannot show the RR22 batch-membership effect")
    for name, ok in hard.items():
        if not ok:
            print(f"::error::D1 shard identity check failed: {name}")
    if args.summary:
        rows = "\n".join(f"| {name} | {'pass' if ok else 'FAIL'} |" for name, ok in hard.items())
        control = {True: "differs (expected)", False: "matched serial (WARNING)", None: "not run"}[out.get("negative_control_differs")]
        walls = "\n".join(f"| {step} | {seconds} |" for step, seconds in timings.items())
        with args.summary.open("a") as summary:
            summary.write(f"### D1 shard identity\n\n| check | result |\n|---|---|\n{rows}\n\nKey-hash negative control: {control}\n\n"
                          f"Units: {out.get('serial_units')}\n\n| step | wall s |\n|---|---|\n{walls}\n")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
