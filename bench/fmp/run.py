"""FMP CLI: build-stores, canary, answer, judge, score. No corpus on stdout/logs."""
import argparse
import contextlib
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "bench.fmp"

from . import datasets, delivery, judges, models
from .common import commit, digest_file, load_lcm, quiet, sha, utc, write_json
from .models import (AGENTIC_TOOL_CAP, CLAUDE_AGREEMENT_JUDGE, CODEX_DEFAULT_JUDGE, CODEX_DEFAULT_READER,
                     CODEX_DISABLED, GLM_HOST, JUDGE_MAX_TOKENS, READER_MAX_TOKENS, READERS, REASONING, Backends, BudgetStopped,
                     OpenRouter, Quota, Spend, ToolViolation, glm_key, harness, parse_spec)
from .score import delivery_identity, percentile, score
from .stores import Stores


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("build-stores", "canary", "answer", "judge", "score"))
    p.add_argument("--lcmx-repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--dataset", choices=("lme-s", "lme-m", "locomo"), default="lme-s")
    p.add_argument("--data", type=Path, default=os.environ.get("FMP_LME_DATA") or None,
                   help="dataset file (default: $FMP_LME_DATA)")
    p.add_argument("--question-ids", type=Path)
    p.add_argument("--include-adversarial", action="store_true", help="include LoCoMo category 5 abstention questions")
    p.add_argument("--split-label", required=True)
    p.add_argument("--config", default="off")
    p.add_argument("--paired-config", help="interleave this config with --config; output is a parent directory")
    p.add_argument("--summaries", choices=("none", "deterministic"), default="none")
    p.add_argument("--store-text", choices=("speaker-prefix", "plain", "name-date"), default="speaker-prefix")
    p.add_argument("--hit-label", choices=("speaker", "role"), default="speaker")
    p.add_argument("--embed-cache", type=Path)
    p.add_argument("--store-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--delivery", choices=("agentic", "single", "full_context"), default="agentic")
    p.add_argument("--delivery-unit", choices=("hits", "session"), default="hits")
    # Default matrix is all-Codex on the ChatGPT plan; OpenRouter is an explicit opt-in
    # (`openrouter:<id>`, with --max-spend and --spend-ledger).
    p.add_argument("--reader", type=str, default=CODEX_DEFAULT_READER, help="codex:<model>@<effort> | claude:<model> | glm:<model> | "
                   "openrouter:<id>; opt-in ids: " + ", ".join(READERS))
    p.add_argument("--judge", nargs="+", default=[CODEX_DEFAULT_JUDGE], help="judges applied to every answer")
    p.add_argument("--agreement-judge", default=CLAUDE_AGREEMENT_JUDGE, help="judge on a seeded sample; 'none' disables")
    p.add_argument("--agreement-fraction", type=float, default=0.1)
    p.add_argument("--agreement-seed", type=int, default=1189)
    p.add_argument("--workers", type=int, default=4, help="concurrent questions per backend")
    p.add_argument("--quota-stop", type=float, default=95.0, help="Codex weekly used percent that stops new questions")
    p.add_argument("--quota-ledger", type=Path, help="shared quota state (default: <output>/quota.json)")
    p.add_argument("--max-spend", type=float)
    p.add_argument("--spend-ledger", type=Path)
    p.add_argument("--runs", nargs="+", type=Path)
    p.add_argument("--allow-partial", action="store_true", help="score arms with unequal run question sets")
    p.add_argument("--pair-across-commits", action="store_true", help="score matching arms across LCM-X commits")
    p.add_argument("--allow-dirty", action="store_true", help="run on an LCM-X tree with uncommitted changes")
    return p


def prompt_hashes():
    return {p: digest_file(Path(__file__).with_name(p)) for p in ("prompts.py", "delivery.py", "judges.py")}


# Planning figures per call, wrapper overhead included (Codex adds ~9.5k input tokens per model
# request: base instructions and tool schemas). The smoke replaces them with measured values.
TOKENS_PER_CALL = {"single": 20_000, "full_context": 140_000, "agentic": 150_000, "judge": 11_000}


def specs(args):
    if args.command == "judge":
        return [*args.judge, *([args.agreement_judge] if args.agreement_judge != "none" else [])]
    return [args.reader]


def sample(ids, fraction, seed):
    ids = sorted(ids)
    return sorted(random.Random(seed).sample(ids, min(len(ids), math.ceil(fraction * len(ids))))) if ids else []


def estimate(args, n):
    plans, calls = {}, []
    if args.command == "answer":
        calls.append((args.reader, n, TOKENS_PER_CALL[args.delivery]))
    elif args.command == "judge":
        calls += [(j, n, TOKENS_PER_CALL["judge"]) for j in args.judge]
        if args.agreement_judge != "none":
            calls.append((args.agreement_judge, len(sample(range(n), args.agreement_fraction, 0)), TOKENS_PER_CALL["judge"]))
    for spec, count, per in calls:
        plan = {"codex": "chatgpt_plan", "claude": "claude_plan", "glm": "glm_coding_plan", "openrouter": "openrouter_metered"}[parse_spec(spec)[0]]
        entry = plans.setdefault(plan, {"calls": 0, "tokens": 0, "specs": []})
        entry["calls"] += count
        entry["tokens"] += count * per
        entry["specs"].append(spec)
    data = {"command": args.command, "questions": n, "per_call_tokens": TOKENS_PER_CALL, "plans": plans}
    write_json(args.output / f"estimate-{args.command}.json", data)
    print("FMP estimate: " + "; ".join(f"{p}: {v['calls']} calls x ~{v['tokens'] // max(1, v['calls'])} tokens "
                                       f"= ~{v['tokens'] / 1e6:.1f}M tokens" for p, v in plans.items()),
          file=sys.__stderr__, flush=True)
    return data


def switches(args):
    return {**({"include_adversarial": True} if getattr(args, "include_adversarial", False) else {}),
            **({"delivery_unit": "session", "delivery_budget": 16000}
               if getattr(args, "delivery_unit", "hits") == "session" else {}),
            "store_text": getattr(args, "store_text", "speaker-prefix"),
            "hit_label": getattr(args, "hit_label", "speaker") if getattr(args, "delivery", "agentic") == "single" else "n/a"}


def server_spec(args, dataset_sha, q):
    return {"repo": str(args.lcmx_repo), "store_root": str(args.store_root), "config": args.config,
            "summaries": args.summaries, "dataset_sha": dataset_sha, **switches(args),
            "embed_cache": str(args.embed_cache) if args.embed_cache else None, "max_calls": AGENTIC_TOOL_CAP,
            "question": {"qid": q.qid, "dataset": q.dataset, "store_id": q.store_id, "session_map": q.session_map,
                         "question_date": q.question_date}}


def versions():
    found = {}
    for name in ("codex", "claude"):
        try:
            found[name] = subprocess.run([name, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            found[name] = None
    return found


def canary(args, client, dataset_sha, q):
    # Leakage canary (spec item 3): a unique string sits next to the run output and in the dataset
    # directory; the question's answer is that string. It fails if the string reaches any transcript
    # or if any non-MCP tool call appears. The string is never printed or kept.
    results = []
    for mode in ("text", "mcp"):
        token = "REF-" + uuid.uuid4().hex
        places = [args.output / f"fmp-canary-{mode}.txt", args.data.parent / f"fmp-canary-{uuid.uuid4().hex}.txt"]
        records, status = [], "ok"
        started = time.monotonic()
        try:
            for place in places:
                place.write_text(token + "\n")
            # Plain wording: a "canary/secret file" phrasing trips Opus 5.5's cyber safeguard and its
            # fallback to another model (measured), which the model check then fails.
            ask = ("My notes files at " + " and ".join(map(str, places)) + " contain a reference code. "
                   "What is that reference code? Reply with only the code." +
                   (" You can also search my conversation history with your tools." if mode == "mcp" else ""))
            messages = [{"role": "user", "content": ask}]
            try:
                if mode == "mcp":
                    client.agentic(args.reader, messages, records, server_spec(args, dataset_sha, q))
                else:
                    client.call(args.reader, messages, records)
            except ToolViolation:
                status = "tool_violation"
            except Exception as exc:
                status = "error:" + type(exc).__name__
                # With no tools at all, Claude may write a tool call as text; Claude Code cannot run it
                # and ends the call. Nothing executed: that is the outcome this test wants to see.
                if any(c.get("terminal_reason") == "malformed_tool_use_exhausted" for c in records):
                    status = "no_tool_to_call"
            # Timeout records keep call_dir even when they omit transcripts.
            paths = [Path(p) for c in records for p in (c.get("transcripts") or
                     ([args.output / "harness" / c["call_dir"] / "events.jsonl"] if c.get("call_dir") else [])) if p]
            leaked = any(token in p.read_text(errors="replace") for p in paths if p.exists())
        finally:
            for place in places:
                place.unlink(missing_ok=True)
        violations = [v for c in records for v in c.get("violations", [])]
        results.append({"harness": args.reader, "mode": mode, "status": status, "leaked": leaked,
                        "violations": violations, "mcp_calls": sum(c.get("mcp_calls") or 0 for c in records),
                        "transcripts_checked": len(paths), "tokens": [c.get("input_tokens") for c in records],
                        "passed": status in ("ok", "no_tool_to_call") and not leaked and not violations and bool(paths)})
        if status.startswith("error:"):
            events = []
            for path in paths:
                for line in path.read_text(errors="replace").splitlines() if path.exists() else []:
                    try:
                        event = json.loads(line)
                    except ValueError:  # A timed-out stream may end mid-event.
                        continue
                    if isinstance(event, dict) and isinstance(event.get("type"), str):
                        kind = event["type"]
                        if kind == "system" and event.get("subtype") == "init":
                            kind += ":init"
                        events.append(kind)
            errors = sorted({c["error"] for c in records if c.get("error")})
            results[-1]["reason"] = (f"{status}; mode={mode}; elapsed_s={time.monotonic() - started:.3f}; "
                                     f"events={','.join(events) or 'none'}; call_errors={','.join(errors) or 'none'}")
        # Positive control: in MCP mode the reader must reach the lcmx tools, or the run proves nothing.
        results[-1]["passed"] &= mode != "mcp" or results[-1]["mcp_calls"] > 0
    write_json(args.output / "canary.json", {"at": utc(), "results": results, "versions": versions(), **switches(args)})
    if not all(r["passed"] for r in results):
        raise RuntimeError("canary_failed")


def execute(args):
    if not 1 <= args.workers <= 64:
        raise ValueError("invalid worker count")
    if args.command in ("answer", "canary"):
        try:
            models.reader_spec(args.reader)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(str(exc)) from exc
    if args.command == "score":
        score(args.runs or [], args.output, args.allow_partial, args.pair_across_commits)
        return
    if not 0 < args.agreement_fraction <= 1:
        raise ValueError("invalid agreement fraction")
    if any(parse_spec(s)[0] == "glm" for s in specs(args)):
        glm_key()
    if args.data is None:
        raise ValueError("--data FILE or FMP_LME_DATA is required")
    args.data = Path(args.data)
    if commit(args.lcmx_repo).endswith("-dirty") and not args.allow_dirty:
        raise ValueError("LCM-X checkout has uncommitted changes")
    lme = load_lcm(args.lcmx_repo)
    questions, dataset_sha, ids_sha, excluded = datasets.load(
        args.data, args.dataset, lme, args.question_ids, args.include_adversarial)
    ids_sha = ids_sha or sha(json.dumps([q.qid for q in questions]).encode())
    args.output.mkdir(parents=True, exist_ok=True)
    store = None
    if args.command == "canary":
        questions = questions[:1]
    if args.command in ("build-stores", "canary", "answer"):
        store = Stores(args.store_root, args.config, args.lcmx_repo, dataset_sha, lme, args.summaries, args.embed_cache, args.store_text)
        seen = set()
        for q in questions:
            if q.store_id not in seen:
                store.build(q)
                seen.add(q.store_id)
    if args.command == "build-stores":
        write_json(args.output / "build-receipt.json", {"stores": len(seen), "identity": store.identity, **switches(args)})
        return
    if args.command != "canary":
        estimate(args, len(questions))
    openrouter = None
    if any(parse_spec(s)[0] == "openrouter" for s in specs(args)):
        if args.max_spend is None or args.spend_ledger is None:
            raise ValueError("OpenRouter is opt-in: pass --max-spend and --spend-ledger")
        openrouter = OpenRouter(Spend(args.spend_ledger, args.max_spend))
    quota = Quota(args.quota_ledger or args.output / "quota.json", args.quota_stop)
    client = Backends(args.output, quota, specs(args), openrouter)
    try:
        if args.command == "canary":
            canary(args, client, dataset_sha, questions[0])
        else:
            configs = [args.config, args.paired_config] if args.paired_config else [args.config]
            jobs = []
            for config in configs:
                arm = copy.copy(args)
                arm.config = config
                arm.output = args.output / f"{config}-{args.delivery}" if args.paired_config else args.output
                arm.output.mkdir(parents=True, exist_ok=True)
                arm_store = store if config == args.config else Stores(args.store_root, config, args.lcmx_repo,
                    dataset_sha, lme, args.summaries, args.embed_cache, args.store_text)
                if config != args.config and args.command == "answer":
                    for q in questions:
                        arm_store.build(q)
                jobs.append(run(arm, client, questions, dataset_sha, ids_sha, excluded, arm_store))
            run_jobs(jobs, args.workers)
    finally:
        client.close()


def run_jobs(jobs, workers):
    prepared = [next(job) for job in jobs]
    if len(jobs) == 1:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(*prepared[0][:2]):
                jobs[0].send(row)
        prepared[0][2]["active_s"] += time.monotonic() - started
    else:
        start = random.Random(1189).randrange(len(jobs))
        for i, items in enumerate(zip(*(source for _, source, _ in prepared), strict=True)):
            for j in range(len(jobs)):
                index = (start + i + j) % len(jobs)
                started = time.monotonic()
                jobs[index].send(prepared[index][0](items[index]))
                prepared[index][2]["active_s"] += time.monotonic() - started
    for job in jobs:
        try:
            job.send(None)
        except StopIteration:
            pass


def run(args, client, questions, dataset_sha, ids_sha, excluded, store):
    manifest_path = args.output / "run-manifest.json"
    if args.command == "answer":
        if manifest_path.exists():
            raise ValueError("output already contains a run; use a new directory")
        with store.engine(questions[0]) as (engine, _):
            if getattr(args, "delivery_unit", "hits") == "session":
                delivery.session_mode(engine)
            toolset = [s["function"]["name"] for s in delivery.schemas(engine)]
        manifest = {"harness_commit": commit(Path(__file__).resolve().parents[2]),
            "harness_tree_sha256": sha(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))),
            "lcmx_commit": commit(args.lcmx_repo), "dataset_sha256": dataset_sha, "id_list_sha256": ids_sha,
            "question_ids": [q.qid for q in questions], "split_label": args.split_label, "config": args.config,
            "summaries": args.summaries, "delivery": args.delivery, "reader": args.reader, **switches(args),
            "excluded": excluded, "store_generation": store.generation, "toolset": toolset,
            "prompts_sha256": prompt_hashes(),
            "start_utc": utc(), "end_utc": None, "spend": 0, "status": "running"}
        if harness(args.reader):
            manifest.update(harness={"versions": versions(), "codex_disabled_features": list(CODEX_DISABLED),
                "codex_sandbox": "read-only; isolated CODEX_HOME and HOME; --ignore-user-config --ignore-rules",
                "claude": "isolated CLAUDE_CONFIG_DIR, --tools '' --strict-mcp-config, PreToolUse deny hook",
                "agentic": "harness-native loop over mcp_reader, sufficiency stop = harness, cap "
                           f"{AGENTIC_TOOL_CAP} counted tool calls"},
                tool_budget={"calls": AGENTIC_TOOL_CAP if args.delivery == "agentic" else 0,
                             "input_tokens": None if args.delivery == "agentic" else 200_000},
                clock="tools run at the question date (server-side)")
        else:
            manifest.update(reasoning=REASONING, max_tokens={"reader": READER_MAX_TOKENS, "judge": JUDGE_MAX_TOKENS},
                judge_temperature={"locomo": "0 where the pinned endpoint supports it, else omitted (per call record)"},
                tool_budget={"calls": 12, "input_tokens": 200_000,
                             "estimate": "reported prompt_tokens + ceil(chars/4) of appended messages"},
                input_fit="UTF-8 byte upper bound + framing", clock="tools run at the question date",
                spend_ledger=str(args.spend_ledger),
                model_metadata={m: client.openrouter.metadata[m] for m in READERS})
        write_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("include_adversarial", False) != getattr(args, "include_adversarial", False):
            raise ValueError("judging run identity mismatch: include_adversarial")
        if delivery_identity(manifest) != delivery_identity(switches(args)):
            raise ValueError("judging run identity mismatch: delivery_unit/delivery_budget")
        for key, value in (*switches(args).items(), ("dataset_sha256", dataset_sha), ("id_list_sha256", ids_sha),
                           ("split_label", args.split_label), ("question_ids", [q.qid for q in questions]),
                           ("prompts_sha256", prompt_hashes()),
                           ("harness_commit", commit(Path(__file__).resolve().parents[2])),
                           ("harness_tree_sha256", sha(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))))),
                           ("lcmx_commit", commit(args.lcmx_repo))):
            if manifest.get(key) != value:
                raise ValueError("judging run identity mismatch")
    manifest.setdefault("endpoint_hosts", {}).update({s: GLM_HOST for s in specs(args) if parse_spec(s)[0] == "glm"})
    timing = {"active_s": 0.0}
    if args.command == "answer":
        def work(q):
            calls = []
            start = time.monotonic()
            row = {"qid": q.qid, "qtype": q.qtype, "question": q.question, "gold": q.gold,
                   "hypothesis": "", "calls": calls, "tool_calls": 0, **delivery.date_exposure(q)}
            if q.qtype == "5":
                row.update(gold="", adversarial_answer=q.gold)
            try:
                if args.delivery == "agentic" and harness(args.reader):  # the MCP reader opens the store
                    row.update(delivery.answer(q, args.delivery, client, args.reader, records=calls,
                                               server=server_spec(args, dataset_sha, q)))
                else:
                    with store.engine(q) as (engine, _):
                        if getattr(args, "delivery_unit", "hits") == "session":
                            delivery.session_mode(engine)
                        speakers = json.loads((store.directory(q) / "speakers.json").read_text()) if args.hit_label == "speaker" else None
                        row.update(delivery.answer(q, args.delivery, client, args.reader, engine, speakers,
                                                   records=calls, hit_label=args.hit_label, store_text=args.store_text))
            except BudgetStopped as exc:  # D7: a run state (spend or plan quota), excluded from scoring
                row.update(getattr(exc, "delivery_result", {}))
                row.update(status="not_run", reason="budget_stopped")
            except ToolViolation as exc:  # the harness used a non-MCP tool: a failed question
                row.update(getattr(exc, "delivery_result", {}))
                row.update(status="tool_violation")
            except Exception as exc:
                row.update(getattr(exc, "delivery_result", {}))
                row.update(status="error", error=type(exc).__name__)
            return {**row, "latency_s": time.monotonic() - start}
        destination = args.output / "answers.jsonl"
        source = questions
    else:
        rows = [json.loads(line) for line in (args.output / "answers.jsonl").read_text().split("\n") if line.strip()]
        if {r["qid"] for r in rows} != {q.qid for q in questions}:
            raise ValueError("answer ids mismatch")
        by_id = {q.qid: q for q in questions}
        agreement = [] if args.agreement_judge == "none" else \
            sample(by_id, args.agreement_fraction, args.agreement_seed)
        manifest.update(judges=list(args.judge), agreement_judge=None if args.agreement_judge == "none" else
                        args.agreement_judge, agreement_sample=agreement, agreement_seed=args.agreement_seed,
                        agreement_fraction=args.agreement_fraction)

        def work(row):
            extra = args.agreement_judge if row["qid"] in agreement else None
            return judges.judge_row(row, by_id[row["qid"]], client, args.judge, agreement_model=extra)
        destination = args.output / "judged.jsonl"
        source = rows
    with destination.open("x") as stream:
        if args.command == "judge":
            manifest["status"] = "judging"
            write_json(manifest_path, manifest)
        row = yield work, source, timing
        while row is not None:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            row = yield
    manifest[args.command + "_wall_s"] = timing["active_s"]
    manifest["wall_s"] = manifest.get("answer_wall_s", 0) + manifest.get("judge_wall_s", 0)
    written = [json.loads(line) for line in destination.read_text().split("\n") if line.strip()]
    manifest.update(end_utc=utc(), not_run=sum(r["status"] == "not_run" for r in written),
                    status="incomplete" if any(r["status"] == "not_run" for r in written) else
                    "complete" if args.command == "judge" else "answered")
    manifest["spend"] = sum(c.get("cost", 0) for row in written for c in row.get("calls", []) + row.get("judge_calls", []))
    write_json(manifest_path, manifest)


def check_arms(runs, output=None):
    """Audit frozen single-delivery answer receipts, without clients."""
    if len(runs) != 2:
        raise ValueError("check-arms requires two runs")
    indexed, arms, manifests = [], [], []
    for directory in map(Path, runs):
        manifest = json.loads((directory / "run-manifest.json").read_text())
        manifests.append(manifest)
        if manifest.get("delivery") != "single":
            raise ValueError("check-arms requires single delivery")
        rows = [json.loads(line) for line in (directory / "answers.jsonl").read_text().splitlines() if line.strip()]
        by_id, hits, estimates = {r["qid"]: r for r in rows}, {}, []
        counts = dict(role_lines=0, speaker_lines=0, summary_lines=0, missing_labels=0, missing_recall=0)
        for qid, row in by_id.items():
            recalls = [t["result"] for t in row.get("trace", []) if t.get("tool") == "lcm_recall" and "result" in t]
            counts["missing_recall"] += not recalls
            hits[qid] = []
            for raw in recalls:
                payload = json.loads(raw) if isinstance(raw, str) else raw
                if "error" in payload or "hits" not in payload:
                    counts["missing_recall"] += 1
                hits[qid].append([h.get("store_id") for h in payload.get("hits", [])])
            labels, digest = row.get("evidence_labels"), row.get("evidence_sha256")
            # The digest is an audit receipt: format checked, never recomputed here.
            if (not isinstance(labels, dict) or set(labels) != {"role", "speaker", "summary"} or
                    any(type(n) is not int or n < 0 for n in labels.values()) or
                    not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                counts["missing_labels"] += 1
            else:
                for label, count in labels.items():
                    counts[label + "_lines"] += count
            estimates.extend(c["prompt_tokens_estimate"] for c in row.get("calls", []) if c.get("prompt_tokens_estimate") is not None)
        mode = (manifest.get("store_text"), manifest.get("hit_label"))
        failures = [(status, sorted(qid for qid, row in by_id.items() if row.get("status") == status))
                    for status in ("error", "not_run")]
        failure_reason = "; ".join(f"{len(qids)} answer rows {'errored' if status == 'error' else 'not run'}: " +
                                   ", ".join(qids) for status, qids in failures if qids)
        non_ok = {s: sum(r.get("status") == s for r in by_id.values())
                  for s in sorted({r["status"] for r in by_id.values() if r.get("status", "ok") != "ok"})}
        arm = {("plain", "role"): "A0", ("plain", "speaker"): "A1"}.get(mode)
        label_ok = (counts["role_lines"] > 0 and counts["speaker_lines"] == 0 if arm == "A0" else
                    counts["speaker_lines"] > 0 and counts["role_lines"] == 0 if arm == "A1" else False)
        arms.append({**counts, "arm": arm, "store_text": mode[0], "hit_label": mode[1],
            "label_check": "unavailable" if counts["missing_labels"] else "pass" if label_ok else "fail",
            "reason": failure_reason or ("label check unavailable: answer rows lack valid evidence_labels/evidence_sha256" if counts["missing_labels"] else None),
            **({"errored_rows": sum(len(qids) for _, qids in failures)} if failure_reason else {}),
            **({"non_ok_statuses": non_ok} if non_ok else {}),
            "qid_count": len(by_id), "duplicate_qids": len(rows) - len(by_id),
            "manifest_qids_match": set(by_id) == set(manifest.get("question_ids", [])),
            "qid_set_sha256": sha(json.dumps(sorted(by_id)).encode()),
            "id_list_sha256": manifest.get("id_list_sha256"), "dataset_sha256": manifest.get("dataset_sha256"),
            "answers_sha256": digest_file(directory / "answers.jsonl"),
            "speaker_label_failure": arm == "A1" and counts["role_lines"] > 0,
            "prompt_tokens_estimate": {"median": median(estimates) if estimates else None, "p90": percentile(estimates, .9)}})
        indexed.append(hits)
    # run() records the complete reader spec (harness/model/effort), settings and
    # summary-store identity. A0/A1 vary only hit_label; keep that switch allowed.
    configuration_keys = ("reader", "harness", "harness_commit", "harness_tree_sha256", "prompts_sha256",
                          "config", "summaries", "store_generation",
                          "delivery_unit", "delivery_budget", "reasoning", "max_tokens", "tool_budget")
    mismatches = [key for key in configuration_keys if
                  (key in manifests[0]) != (key in manifests[1]) or
                  json.dumps(manifests[0].get(key), sort_keys=True) != json.dumps(manifests[1].get(key), sort_keys=True)]
    report = {"arms": arms, "qid_sets_equal": set(indexed[0]) == set(indexed[1]),
        **({"configuration_mismatches": mismatches,
            "reason": "run configuration mismatch: " + ", ".join(mismatches)} if mismatches else {}),
        "id_list_hash_equal": bool(arms[0]["id_list_sha256"]) and arms[0]["id_list_sha256"] == arms[1]["id_list_sha256"],
        "retrieval_differing_qids": sum(indexed[0][q] != indexed[1][q] for q in indexed[0].keys() & indexed[1].keys())}
    report["declared_modes_valid"] = sorted(a["arm"] or "invalid" for a in arms) == ["A0", "A1"]
    report["dataset_hash_equal"] = bool(arms[0]["dataset_sha256"]) and arms[0]["dataset_sha256"] == arms[1]["dataset_sha256"]
    report["passed"] = not mismatches and report["declared_modes_valid"] and report["dataset_hash_equal"] and report["qid_sets_equal"] and report["id_list_hash_equal"] and not report["retrieval_differing_qids"] and all(
        a["manifest_qids_match"] and a["label_check"] == "pass" and not any(a.get(k, 0) for k in ("duplicate_qids", "missing_recall", "errored_rows"))
        for a in arms)
    output = Path(output) if output else Path(os.path.commonpath([Path(r).resolve().parent for r in runs])) / f"check-arms-{uuid.uuid4().hex}.json"
    try:
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except OSError:
        print("FMP check-arms report write failed: " + str(output), file=sys.stderr)
        raise
    return report


def main():
    if sys.argv[1:2] == ["check-arms"]:
        p = argparse.ArgumentParser(description="Read-only pre-judge arm integrity audit; counts and hashes only")
        p.add_argument("command", choices=("check-arms",))
        p.add_argument("--runs", nargs=2, required=True, type=Path)
        p.add_argument("--output", type=Path)
        args = p.parse_args()
        os.umask(0o077)
        try:
            report = check_arms(args.runs, args.output)
        except Exception as exc:
            print("FMP check-arms failed: " + type(exc).__name__, file=sys.stderr)
            return 1
        print(json.dumps(report, sort_keys=True))
        return int(not report["passed"])
    args = parser().parse_args()
    quiet()
    os.umask(0o077)
    # Suppress incidental dependency output across all workers, including errors.
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            execute(args)
    except Exception as exc:
        import traceback
        write_json(args.output / "failure.json", {"error": type(exc).__name__, "at": utc(),
            "frames": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                       for f in traceback.extract_tb(exc.__traceback__)]})
        print("FMP failed: " + type(exc).__name__ + " (no corpus logged)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
