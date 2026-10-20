"""Frozen-store, cached paired development screen; never release/confirmation proof."""
import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import delivery, judges, models
from .closeout import card_tool
from .common import digest_file, load_lcm, quiet, sha, write_json
from .dev_cache import Incomplete, MemoryCodex, Responses, source_identity
from .dev_panel import PANELS, encoded, freeze, load
from .stores import Stores


def renderer_identity():
    return {p: digest_file(Path(__file__).with_name(p)) for p in
            ("dev.py", "dev_panel.py", "dev_cache.py", "delivery.py", "judges.py", "prompts.py", "models.py")}


def caps(depth):
    return dict(limit=depth, candidate_pool=4 * depth, rerank_window=50,
                per_session=math.ceil(5 * depth / 25), expanded=math.ceil(8 * depth / 25))


def retrieve(q, store, overrides):
    """Promoted depth-screen cap observation, with the existing FMP single renderer."""
    from hermes_lcm import recall_delivery, tools
    depth = overrides.get("recall_depth", 25)
    if type(depth) is not int or not 1 <= depth <= 100:
        raise ValueError("invalid recall depth")
    expected, observed, payload = caps(depth), {}, {}
    original, dispatch = recall_delivery.deliver_recall, delivery.dispatch

    def observe(ordered, **kw):
        caller = sys._getframe(1).f_locals
        observed.update(limit=kw["limit_cap"], candidate_pool=caller["candidate_limit"],
                        rerank_window=caller["rerank_window"], per_session=kw["per_session_limit"],
                        expanded=kw["expanded_hit_limit"])
        return original(ordered, **kw)

    def recall(name, arguments, engine, offered=None):
        raw = dispatch(name, {**arguments, "limit": depth, "detail": "answer_ready"}, engine, offered)
        payload.update(json.loads(raw))
        return raw

    with store.engine(q) as (engine, _), contextlib.ExitStack() as stack:
        for key, value in overrides.items():
            if key == "recall_depth":
                continue
            if key not in vars(engine._config) or key.startswith(("database", "embedding")):
                raise ValueError("unsupported config override")
            setattr(engine._config, key, value)
        for name, value in (("_LCM_RECALL_LIMIT_CAP", depth),
                ("_LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT", expected["per_session"]),
                ("_LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT", expected["expanded"])):
            stack.enter_context(patch.object(tools, name, value))
        stack.enter_context(patch.object(recall_delivery, "deliver_recall", observe))
        stack.enter_context(patch.object(delivery, "dispatch", recall))
        speakers = json.loads((store.directory(q) / "speakers.json").read_bytes())
        with delivery.question_clock(q.question_date):
            evidence, raw = delivery.single(q, engine, speakers)
    hits = payload.get("hits", [])
    expansion = payload.get("provenance", {}).get("answer_ready", {})
    counts = {sid: sum(h.get("session_id") == sid for h in hits) for sid in {h.get("session_id") for h in hits}}
    valid = (observed == expected and payload.get("limit") == depth and "limit_clamped_from" not in payload
             and "error" not in payload and not payload.get("timeout")
             and payload.get("total_results") == len(hits) and len(hits) <= depth
             and expansion.get("per_session_limit") == expected["per_session"]
             and expansion.get("expanded_hit_limit") == expected["expanded"]
             and expansion.get("expanded_hit_count") == sum("content" in h for h in hits)
             and expansion.get("expanded_hit_count", 0) <= expected["expanded"]
             and max(counts.values(), default=0) <= expected["per_session"])
    if not valid:
        raise Incomplete("effective recall readback mismatch")
    return dict(evidence=evidence, source_sha256=sha(raw.encode()), metadata=dict(
        declared=expected, observed=observed, response_limit=payload["limit"], hit_count=len(hits),
        expanded_count=expansion.get("expanded_hit_count"),
        degraded=bool(payload.get("degraded")), coverage=payload.get("provenance", {}).get("coverage", {}),
        degraded_reason_sha256=sha(str(payload.get("degraded_reason", "")).encode()),
        delivered_tokens_estimated=delivery.tokens(evidence), delivered_sha256=sha(evidence.encode())))


def worker(spec):
    lme = load_lcm(spec["repo"])
    questions, panel, guard = load(spec["panel"], spec["guard"], spec["lme"], spec["locomo"], lme)
    product = source_identity(spec["repo"])
    renderer = renderer_identity()
    receipt = []
    for dataset in ("lme-s", "locomo"):
        group = [q for q in questions if q.dataset == dataset]
        store = Stores(Path(spec["cache"]) / "stores", spec["config"], spec["repo"],
                       panel["dataset_sha256"][dataset], lme)
        # Conservative product dependency closure: a product Python change rebuilds ingestion.
        store.identity.update(lcmx_commit=product, ingestion_source_sha256=product,
            ingestion_adapter={p: digest_file(Path(__file__).with_name(p)) for p in ("stores.py", "datasets.py")})
        store.generation = "dev-" + sha(encoded(store.identity))
        for q in group:
            if shutil.disk_usage(spec["cache"]).free < 60 * 1024 ** 3:
                raise Incomplete("disk reserve")
            directory = store.directory(q)
            if spec["prepare"]:
                if directory.exists() and (not (directory / "build-manifest.json").exists() or
                        json.loads((directory / "build-manifest.json").read_bytes()) != store.expected(q)):
                    raise Incomplete("incomplete store generation; retain and use new cache root")
                state = store.build(q)
                receipt.append(dict(id=q.qid, state=state, generation=store.generation))
                continue
            manifest = directory / "build-manifest.json"
            if not manifest.exists() or json.loads(manifest.read_bytes()) != store.expected(q):
                raise Incomplete("cold store; run --prepare outside timed loop")
            store_id = dict(manifest_sha256=digest_file(manifest), database_sha256=digest_file(directory / "lcm.db"),
                            speakers_sha256=digest_file(directory / "speakers.json"))
            identity = dict(store=store_id, query_sha256=sha(q.question.encode()), date=q.question_date,
                            product=product, config=spec["overrides"], renderer=renderer)
            key = sha(encoded(identity))
            cache = Path(spec["cache"]) / "retrieval" / (key + ".json")
            hit = cache.exists()
            if hit:
                saved = json.loads(cache.read_bytes())
                if saved["identity"] != identity or sha(encoded(saved["value"])) != saved["value_sha256"]:
                    raise Incomplete("retrieval identity/digest mismatch")
                value = saved["value"]
            else:
                value = retrieve(q, store, spec["overrides"])
                if digest_file(directory / "lcm.db") != store_id["database_sha256"]:
                    raise Incomplete("frozen store changed")
                write_json(cache, dict(identity=identity, value=value, value_sha256=sha(encoded(value))))
            receipt.append(dict(id=q.qid, cache_key=key, cache="cached" if hit else "fresh",
                                sources=store_id, retrieval_sha256=value["source_sha256"], **value["metadata"]))
    write_json(Path(spec["receipt"]), dict(product=product, rows=receipt, prepare=spec["prepare"]))


def judge_messages(q, hypothesis):
    if q.dataset == "locomo" and q.is_abstention:
        # Existing FMP/LME abstention rubric; cat5's adversarial_answer is never a gold answer.
        import copy
        abstention = copy.copy(q)
        abstention.dataset, abstention.qtype, abstention.gold = "lme-s", "single-session-user", ""
        return judges.messages(abstention, hypothesis), "lme-s"
    return judges.messages(q, hypothesis), q.dataset


def compare(items, rows):
    if not items:
        return dict(n=0, clusters=0, correct_count_difference=0, delta=None, ci95=None)
    groups = {}
    b, a = [], []
    for item in sorted(items, key=lambda e: e["id"]):
        x, y = [int(rows[(arm, item["id"])]["correct"]) for arm in ("baseline", "candidate")]
        b.append(x)
        a.append(y)
        groups.setdefault(item["cluster"], []).append(y - x)
    # score_heldout_expansion.py at 2b9ea632: seeded, question-weighted paired cluster percentile bootstrap.
    sums, ns = np.array([sum(v) for v in groups.values()]), np.array([len(v) for v in groups.values()])
    draws = np.random.default_rng(1188).integers(0, len(ns), size=(10_000, len(ns)))
    boot = sums[draws].sum(axis=1) / ns[draws].sum(axis=1)
    return dict(n=len(items), clusters=len(ns), baseline_correct=sum(b), candidate_correct=sum(a),
                correct_count_difference=sum(a) - sum(b), delta=(sum(a) - sum(b)) / len(items),
                ci95=np.percentile(boot, [2.5, 97.5]).tolist())


def finish(output, report, rows, started, command):
    report.update(wall_seconds=time.monotonic() - started, claim_class="advisory",
                  reuse_boundary="development only; cached A/A is not independent placebo or confirmation")
    write_json(output / "rows.json", list(rows.values()))
    write_json(output / "report.json", report)
    (output / "report.md").write_text(f"# FMP development screen\n\nVerdict: {report['verdict']}\n\n"
        "Paired estimates and guard intervals: report.json. Effective caps/tokens per ID: rows.json.\n\n"
        "Not proven: independent quality gain, board numbers, guard non-inferiority, release or runtime readiness.\n")
    tool = card_tool()

    def invoke(*args):
        if tool.execute(tool.parser().parse_args([args[0], str(output), *args[1:]])):
            raise Incomplete("result card validation")
    invoke("init", "--project", "lcm-x", "--kind", "benchmark", "--owner-lane", "fmp-dev",
           "--code-dir", str(Path(__file__).resolve().parents[2]), "--command", command,
           "--claim-class", "advisory", "--local-raw", "false")
    (output / "RESULT.md").write_text("# Development timing receipt\n\n| Metric | Receipt |\n|---|---|\n"
        f"| {report['verdict']} / wall seconds | report.json |\n\n"
        "Not proven: citable quality, independent placebo/confirmation, publication, release or runtime readiness.\n"
        "T1: unpublished. T2: unpublished.\n")
    invoke("headline", "--benchmark", "FMP/dev-timing", "--metric", "wall_seconds", "--value",
           str(report["wall_seconds"]), "--n", str(len(rows)), "--source", "report.json")
    invoke("add", "report.json", "report.md", "rows.json", "RESULT.md", "inputs", "--role", "per-item")
    invoke("finalize")
    invoke("check")
    invoke("status", "settled")


def execute(args):
    if args.worker:
        worker(json.loads(args.worker.read_bytes()))
        return 0
    if not args.baseline or not args.lme_data or not args.locomo_data:
        raise ValueError("baseline and both data paths are required")
    lme = load_lcm(args.baseline)
    if args.freeze_panels:
        freeze(args.lme_data, args.locomo_data, lme, args.split_file)
        return 0
    if not args.cache_root or not args.output or not 1 <= args.workers <= 4 or args.deadline_min <= 0 or args.min_gain < 1:
        raise ValueError("cache/output, 1..4 workers and positive deadline required")
    if any(models.parse_spec(s)[0] != "codex" for s in (args.reader, args.judge)):
        raise ValueError("Codex subscription specs required; no metered API")
    if shutil.disk_usage(args.cache_root.parent).free < (150 if args.prepare else 60) * 1024 ** 3:
        raise Incomplete("disk admission")
    if args.output.exists():
        raise ValueError("use a new output directory")
    questions, panel, guard = load(args.panel, args.guard, args.lme_data, args.locomo_data, lme)
    args.output.mkdir(parents=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    inputs = args.output / "inputs"
    inputs.mkdir()
    for name, path in (("panel.json", args.panel), ("guard.json", args.guard)):
        (inputs / name).write_bytes(path.read_bytes())
    started = time.monotonic()
    deadline = started + args.deadline_min * 60
    arms = [("baseline", args.baseline, {}), ("candidate", args.candidate or args.baseline,
             json.loads(args.candidate_config or "{}"))]
    report = dict(verdict="INCOMPLETE", remaining_ids=[], calls={}, panels={"dev": digest_file(args.panel),
                  "guard": digest_file(args.guard)}, minimum_useful_correct_count=args.min_gain, arms={})
    rows = {}
    try:
        retrievals = {}
        for arm, repo, overrides in arms:
            receipt = inputs / (arm + "-retrieval.json")
            spec = dict(repo=str(repo.resolve()), config=args.config, overrides=overrides,
                cache=str(args.cache_root.resolve()), prepare=args.prepare, lme=str(args.lme_data.resolve()),
                locomo=str(args.locomo_data.resolve()), panel=str(args.panel.resolve()), guard=str(args.guard.resolve()),
                receipt=str(receipt.resolve()))
            spec_path = inputs / (arm + "-spec.json")
            write_json(spec_path, spec)
            timeout = None if args.prepare else max(.001, deadline - time.monotonic())
            proc = subprocess.run([sys.executable, "-m", "bench.fmp.dev", "--worker", str(spec_path)],
                                  capture_output=True, timeout=timeout)
            if proc.returncode:
                raise Incomplete("retrieval worker failed")
            data = json.loads(receipt.read_bytes())
            retrievals[arm] = {r["id"]: r for r in data["rows"]}
            report["arms"][arm] = dict(product_sha256=data["product"], overrides=overrides)
        if args.prepare:
            write_json(args.output / "prepare.json", dict(status="COMPLETE", label="cold preparation; outside timed loop"))
            return 0
        quota = models.Quota(args.output / "quota.json")
        backend = MemoryCodex(args.cache_root / "transport", quota)
        backend.deadline = deadline
        cache = Responses(args.cache_root / "responses", backend, deadline, renderer_identity())
        stop = threading.Event()

        def work(q):
            for arm, _, _ in arms:
                if stop.is_set() or time.monotonic() >= deadline:
                    return
                try:
                    meta = retrievals[arm][q.qid]
                    value = json.loads((args.cache_root / "retrieval" / (meta["cache_key"] + ".json")).read_bytes())["value"]
                    messages = [{"role": "system", "content": delivery.INSTRUCTIONS.format(date=q.question_date) + delivery.EVIDENCE},
                                {"role": "user", "content": "History evidence:\n" + value["evidence"] + "\nQuestion: " + q.question}]
                    sources = dict(store=meta["sources"], retrieval=meta["retrieval_sha256"])
                    calls = []
                    answer = cache.call(args.reader, messages, calls, role="reader", rubric="fmp-single",
                        sources=sources, max_tokens=models.READER_MAX_TOKENS, tool_choice="none")
                    jm, dataset = judge_messages(q, answer["content"])
                    judgment = cache.call(args.judge, jm, calls, role="judge", rubric=sha(encoded(jm)), sources=sources,
                                          temperature=None if dataset.startswith("lme") else 0)
                    correct = judges.parse(judgment["content"], dataset)
                    if correct is None:
                        raise Incomplete("invalid judge verdict")
                    rows[(arm, q.qid)] = dict(arm=arm, id=q.qid, correct=correct, retrieval=meta,
                        response_sha256=sha(encoded(answer)), judge_sha256=sha(encoded(judgment)), calls=calls)
                except Exception as exc:
                    report.setdefault("errors", []).append(dict(arm=arm, id=q.qid, error=type(exc).__name__))
                    stop.set()
                    return
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(work, questions))
        finally:
            backend.close()
            report["calls"] = cache.counts
        complete = len(rows) == 2 * len(questions) and time.monotonic() < deadline
        if complete:
            report["primary"] = compare(panel["items"], rows)
            report["guards"] = {name: compare([e for e in guard["items"] if e["dataset"] == ds], rows)
                                for name, ds in (("adversarial", "locomo"), ("abstention", "lme-s"))}
            risk = any(g["delta"] is not None and g["delta"] < 0 for g in report["guards"].values())
            win = report["primary"]["correct_count_difference"] >= args.min_gain and report["primary"]["ci95"][0] > 0
            report["verdict"] = "GUARD-RISK" if risk else "DIRECTIONAL-WIN" if win else "NO-SIGNAL"
            if time.monotonic() >= deadline:
                report["verdict"] = "INCOMPLETE"
    except Exception as exc:
        report["error"] = type(exc).__name__
    report["remaining_ids"] = [dict(arm=arm, id=q.qid) for arm, _, _ in arms for q in questions if (arm, q.qid) not in rows]
    finish(args.output, report, rows, started, "python -m bench.fmp.dev (arguments in inputs/*-spec.json)")
    return 0 if report["verdict"] != "INCOMPLETE" else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "panel", "guard", "cache-root", "output", "split-file", "worker"):
        p.add_argument("--" + name, type=Path, default={"panel": PANELS / "dev100.json", "guard": PANELS / "guard.json"}.get(name))
    p.add_argument("--lme-data", type=Path, default=os.environ.get("FMP_LME_DATA"))
    p.add_argument("--locomo-data", type=Path, default=os.environ.get("FMP_LOCOMO"))
    p.add_argument("--candidate-config", help='JSON overrides, e.g. {"recall_depth":50}')
    p.add_argument("--config", choices=("off", "local"), default="local")
    p.add_argument("--reader", default="codex:gpt-6.1-sol@medium")
    p.add_argument("--judge", default="codex:gpt-6.1-sol@low")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--deadline-min", type=float, default=30)
    p.add_argument("--min-gain", type=int, default=1)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--freeze-panels", action="store_true")
    return p


def main():
    quiet()
    os.umask(0o077)
    args = parser().parse_args()
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return execute(args)
    except Exception as exc:
        if args.worker:
            spec = json.loads(args.worker.read_bytes())
            import traceback
            write_json(Path(spec["receipt"]), dict(status="INCOMPLETE", error=type(exc).__name__,
                frames=[dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                        for f in traceback.extract_tb(exc.__traceback__)]))
        print(json.dumps(dict(status="INCOMPLETE", error=type(exc).__name__)), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
