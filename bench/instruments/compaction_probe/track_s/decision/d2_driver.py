"""D2 comparison driver. --config JSON pins material, builds, summary, reader, policy.

Build keys are comparison/candidate, each {worktree, sha, arm}; both may name the
same build for --aa. --option A requires a recorded funding_refusal. Runs use the
existing run_pmw_cell.sh; only injected test producers avoid real provider calls.
"""
import argparse
import contextlib
import importlib.util
import io
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_score  # noqa: E402

LIMIT_SECONDS = 21600
LIMIT = re.compile(r"HTTP(?: Error)?[ :]+429|usage[- _]limit|rate[- _]limit|too many requests", re.I)


def dump(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def prepare(config, option, aa, *, check_hash=True):
    """Resolve content identities before the first replay; never serialize environment."""
    if check_hash:
        card_tool()  # refuse a missing card tool before any paid replay
    pins = {k: config[k] for k in ("material", "builds", "summary", "reader", "policy")}
    if option == "A" and aa:
        raise ValueError("option A cannot be A/A")
    if set(pins["builds"]) != {"candidate", "comparison"}:
        raise ValueError("builds must name candidate and comparison")
    for build in pins["builds"].values():
        if git(build["worktree"], "rev-parse", "HEAD") != build["sha"] or git(build["worktree"], "status", "--porcelain"):
            raise ValueError("build must be clean at its exact SHA")
    if aa and len({b["sha"] for b in pins["builds"].values()}) != 1:
        raise ValueError("A/A requires identical builds")
    if option == "A" and not config.get("funding_refusal"):
        raise ValueError("option A requires a recorded funding_refusal")
    if any(not pins[k] for k in ("reader", "policy")) or any(not pins["summary"].get(k) for k in ("provider", "model", "effort", "tier")):
        raise ValueError("all route, reader and policy pins required")
    route = pins["summary"]
    lane = {("glm", "glm-5.3"): "glm", ("codex", "gpt-6.1-sol"): "codex-sol"}.get((route["provider"], route["model"]))
    if not lane or route["effort"] != "low" or route["tier"] != "default" or pins["reader"] != "glm-5.3":
        raise ValueError("pins must match a supported producer route")
    if lane != "glm" and not config.get("comparability_ruling"):
        raise ValueError("changed D2 summary route requires a comparability_ruling")
    pins.update(summary_lane=lane, comparability_ruling=config.get("comparability_ruling"),
                instrument_hash_ruling=config.get("instrument_hash_ruling"))
    if {b["arm"] for b in pins["builds"].values()} != {pins["policy"]["arm"]}:
        raise ValueError("both builds must use the pinned policy arm")
    kit = Path(__file__).resolve().parents[1]
    if git(kit, "status", "--porcelain", "--untracked-files=all", "--", "."):
        raise ValueError("instrument checkout must be clean")
    paths = git(kit, "ls-files", "--", ".").splitlines()
    instrument = {p: digest(kit / p) for p in paths}
    material = Path(pins["material"])
    pins["materials"] = {}
    for seed in (1, 2, 3):
        folder = material / f"seed-{seed}"
        manifest = folder / "material.manifest.json"
        for p, sha in json.loads(manifest.read_text())["shas"].items():
            if digest(folder / p) != sha:
                raise ValueError("material manifest mismatch")
        facts = json.loads((folder / "facts.json").read_text())
        pins["materials"][str(seed)] = dict(manifest=digest(manifest), facts=digest(folder / "facts.json"),
                                            shas=json.loads(manifest.read_text())["shas"],
                                            fact_ids=[f["id"] for f in facts])
    pins.update(option=option, aa=aa, instrument=instrument,
                instrument_hash=hashlib.sha256(json.dumps(instrument, sort_keys=True).encode()).hexdigest(),
                kit_sha=git(kit, "rev-parse", "HEAD"), funding_refusal=config.get("funding_refusal"),
                retry="unchanged producer/reader retries pinned by instrument hash; one replacement per slot, two total")
    pins.update(instrument_root=str(kit), expected_instrument_hash=config.get("expected_instrument_hash"))
    if check_hash and (not pins["expected_instrument_hash"] or
                       (pins["instrument_hash"] != pins["expected_instrument_hash"] and not pins["instrument_hash_ruling"])):
        raise ValueError("predeclared instrument hash required; mismatch requires instrument_hash_ruling")
    return pins


def changed(pins):
    if "instrument" not in pins:  # injected fixture producers have no prepared filesystem
        return None
    if git(pins["instrument_root"], "status", "--porcelain", "--untracked-files=all", "--", "."):
        return "instrument_changed"
    for kind, paths in (("instrument", {Path(pins["instrument_root"]) / p: sha for p, sha in pins.get("instrument", {}).items()}),
                        ("material", {Path(pins["material"]) / f"seed-{s}" / p: sha
                                      for s, rec in pins.get("materials", {}).items()
                                      for p, sha in dict(rec.get("shas", {}), **{"material.manifest.json": rec["manifest"], "facts.json": rec["facts"]}).items()})):
        try:
            if any(digest(p) != sha for p, sha in paths.items()):
                return kind + "_changed"
        except OSError:
            return kind + "_changed"
    return None


def real_producer(pins):
    def launch(row, attempt, path):
        build = pins["builds"][row["arm"]]
        env = dict(os.environ, PRODUCT_WT=build["worktree"], PRODUCT_SHA=build["sha"],
                   KIT_DIR=str(Path(__file__).resolve().parents[1]), MATERIAL=pins["material"],
                   OUT=str(path), ARM=build["arm"], SEEDS=str(row["seed"]), PARALLEL="1", PY=sys.executable,
                   SUMMARY_LANE=pins["summary_lane"])
        env = {k: v for k, v in env.items() if k != "FAULT" and not k.startswith("S2_")}
        # A child session contains the producer's own replay process group too.
        with (path / "producer.log").open("x") as log:
            return subprocess.Popen(["bash", str(Path(__file__).with_name("run_pmw_cell.sh")), str(row["k"]), str(attempt)],
                                    env=env, stdout=log, stderr=log, start_new_session=True)
    return launch


def validate_artifacts(pins, row, attempt, path):
    try:
        build, seed = pins["builds"][row["arm"]], row["seed"]
        label = f"d{seed}-r{row['k']}" + (f"-a{attempt}" if attempt != 1 else "")
        if row["k"] != 1 or attempt != 1:
            path = path / ("replay-" + label.split("-", 1)[1])
        run = path / "lcmx-runs" / build["arm"] / f"seed-{seed}" / label
        summ = json.loads((run / "summary.json").read_text())
        if summ["status"] != "DONE" or summ["worktree_head"] != build["sha"] or summ["stop_row_index"] != 304:
            return "invalid", "unfinished or wrong build/stop row", None
        if summ["material_sha256"] != pins["materials"][str(seed)]["manifest"]:
            return "invalid", "material pin mismatch", None
        if summ["lane"] != pins["summary_lane"] or summ["reader"] != "glm" or summ["arm"]["env"].get("LCM_SUMMARY_REASONING_EFFORT") != pins["summary"]["effort"]:
            return "invalid", "summary/reader route pin mismatch", None
        calls = [c for e in summ.get("events", []) for c in e.get("summariser_calls", []) if not c.get("exception")]
        if not calls or any(c.get("lane_model") != pins["summary"]["model"] or
                            any(c[key] != pins["summary"][pin] for key, pin in
                                (("provider", "provider"), ("effort", "effort"), ("lane_effort", "effort"),
                                 ("tier", "tier"), ("service_tier", "tier")) if key in c) for c in calls):
            return "invalid", "served summary route pin mismatch", None
        cell = path / ("cell.json" if row["k"] == attempt == 1 else "cell-" + label.split("-", 1)[1] + ".json")
        if json.loads(cell.read_text()).get("fault") is not None:
            return "invalid", "injected fault", None
        receipt = path / "decision/logs" / f"s2-{build['arm']}-{label}.log.wall"
        if re.findall(r"^exit (\d+) end", receipt.read_text(), re.M) != ["0"]:
            return "invalid", "failed replay receipt", None
        metrics = {}
        for cp in (176, 304):
            data = json.loads((path / f"decision/scores/cp-{cp}/{build['arm']}.seed-{seed}.{label}.json").read_text())
            fk = data["metrics"]["facts_kept"]
            if fk.get("reader_truncated"):
                return "truncated", "reader truncated facts", None
            if fk["complete"] is not True or data.get("reader_errors") or not 0 <= fk["value"] <= 1:
                return "invalid", "incomplete checkpoint", None
            if data["reader_readback"].get("model") != pins["reader"] or data["reader_readback"].get("pin_ok") is False:
                return "invalid", "reader pin mismatch", None
            facts = {f: bool(data["probes"][f].get("success")) for f in pins["materials"][str(seed)]["fact_ids"]}
            if len(facts) != fk["denominator"] or abs(sum(facts.values())/len(facts)-fk["value"]) > 1e-12:
                return "invalid", "fact denominator/share mismatch", None
            metrics[str(cp)] = dict(share=fk["value"], facts=facts)
        return "valid", "complete pinned checkpoints", metrics
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        return "missing", f"unreadable artifacts: {type(exc).__name__}", None


def stop_process(process):
    if isinstance(process, subprocess.Popen):
        # The shell launches replay groups; kill every group in its new session.
        rows = [tuple(map(int, r.split())) for r in subprocess.check_output(["ps", "-eo", "pid=,ppid=,pgid="], text=True).splitlines()]
        owned = {process.pid}
        while any(parent in owned and pid not in owned for pid, parent, _ in rows):
            owned.update(pid for pid, parent, _ in rows if parent in owned)
        groups = {group for pid, _, group in rows if pid in owned}
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        process.terminate()
    process.wait(timeout=10)


class RunSignal(SystemExit):
    """A termination signal routed through the existing BaseException cleanup."""


@contextlib.contextmanager
def run_signals():
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    def interrupt(signum, frame):
        for other in previous:
            signal.signal(other, signal.SIG_IGN)  # cleanup must finish before restoration
        raise RunSignal(signal.Signals(signum).name)
    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            if previous[signum] != signal.SIG_IGN:
                signal.signal(signum, interrupt)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def card_tool():
    path = os.environ.get("FMP_RESULT_CARD_TOOL")
    if not path or not Path(path).is_file():
        raise ValueError("set FMP_RESULT_CARD_TOOL to the installed bench-card.py")
    return path


def result_card(out, result):
    path = card_tool()
    spec = importlib.util.spec_from_file_location("d2_bench_card", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    def call(*args):
        with contextlib.redirect_stdout(io.StringIO()):
            if tool.execute(tool.parser().parse_args([args[0], str(out), *map(str, args[1:])])):
                raise ValueError("D2 result card validation failed")
    command = "python3 bench/instruments/compaction_probe/track_s/decision/d2_score.py --ledger ledger.json --out NEW_DIRECTORY"
    # Preserve the lowercase card even on case-insensitive filesystems.
    text = (out / "result.md").read_text()
    card = (text + "\n| verdict | receipt |\n| --- | --- |\n"
            + f"| {result['raw_verdict']} | result.json |\n\nRe-score: `{command}`\n\n"
            + "Not proven: other materials/routes, release or customer readiness.\n"
            + "T1: unpublished. T2: unpublished; retained for release-manager handoff.\n")
    (out / "RESULT.md").write_text(card)
    call("init", "--project", "lcm-x", "--kind", "acceptance", "--owner-lane", "d2-instrument",
         "--code-dir", git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel"), "--command", command, "--claim-class", "advisory")
    call("add", *sorted(p.name for p in out.iterdir() if p.name != "RESULT-MANIFEST.json"), "--role", "per-item")
    call("rescore", "--command", command, "--input", "ledger.json", "pins.json",
         *(["replays"] if (out / "replays").exists() else []))
    call("status", "settled", "--owner-verdict", "Retain all results; publication is release-manager owned.")
    call("finalize")
    call("check", "--strict")


@run_signals()
def run_comparison(pins, out, *, producer=None, validate=None, clock=time.monotonic,
                   sleep=time.sleep, scorer=d2_score.score, started=None, wall_clock=time.time, wall_started=None):
    start = clock() if started is None else started
    wall_start = wall_clock() if wall_started is None else wall_started
    def elapsed():
        return max(clock()-start, wall_clock()-wall_start)

    out.mkdir()  # exclusive comparison reservation, never reuse a ledger
    dump(out / "pins.json", pins)
    runs = 9 if pins["option"] == "B" else 4
    rows = []
    for seed, k in itertools.product((1, 2, 3), range(1, runs+1)):
        for slot, arm in enumerate(("candidate", "comparison") if k % 2 else ("comparison", "candidate"), 1):
            rows.append(dict(arm=arm, seed=seed, k=k, slot=slot, attempts=[], state="planned",
                             validity_reason="not started", started=None, finished=None,
                             artifact=str(out / "replays" / f"{arm}-s{seed}-r{k}")))
    data = dict(option=pins["option"], aa=pins["aa"], pins=pins, rows=rows, elapsed_s=elapsed(), stop_reason=None, monitor_errors=[], monitor_errors_dropped=0)
    producer = producer or real_producer(pins)
    validate = validate or (lambda r, a, p: validate_artifacts(pins, r, a, p))
    active, replacements = [], 0
    offsets, marked = {}, set()

    def save():
        data["elapsed_s"] = elapsed()
        dump(out / "ledger.json", data)

    def limited():
        hits = [r for r, _, _ in active if id(r) in marked]
        for row, _, path in active:
            for p in path.rglob("*.log"):
                offset, tail = offsets.get(p, (0, ""))
                for retry in range(2):
                    try:
                        with p.open("rb") as log:
                            log.seek(offset)
                            text = tail + log.read().decode(errors="replace")
                            offsets[p] = (log.tell(), text[-64:])
                        if LIMIT.search(text):
                            hits.append(row)
                            marked.add(id(row))
                        break
                    except OSError as exc:
                        if retry:
                            if len(data["monitor_errors"]) < 50:
                                data["monitor_errors"].append(dict(path=str(p), error=type(exc).__name__))
                            else:
                                data["monitor_errors_dropped"] += 1
        return hits

    def monitor():
        hits = limited()
        if hits:
            data["stop_reason"] = "rate_limited"
        if elapsed() >= LIMIT_SECONDS:
            data["stop_reason"] = "elapsed stop"
        return hits

    def stop_active(hits):
        error = None
        for row, process, _ in active:
            try:
                stop_process(process)
            except BaseException as exc:
                if error is None:
                    error = exc
            finally:
                row.update(state="rate_limited" if row in hits or row["state"] == "rate_limited" else "truncated",
                           validity_reason=data["stop_reason"], finished=elapsed())
                row["attempts"][-1].update(state=row["state"], validity_reason=row["validity_reason"], finished=row["finished"])
        active.clear()
        if error is not None:
            raise error

    def launch(row):
        nonlocal replacements
        drift = changed(pins)
        if drift:
            data["stop_reason"] = drift
            return
        attempt = len(row["attempts"])+1
        if attempt > 1:
            if attempt > 2 or replacements >= 2:
                data["stop_reason"] = "replacement budget"
                return
            replacements += 1
            row["attempts"][-1]["state"] = "replaced"
        path = Path(row["artifact"]) / f"attempt-{attempt}"
        path.mkdir(parents=True)
        now = elapsed()
        row.update(state="running", started=now, finished=None)
        row["attempts"].append(dict(id=attempt, state="running", started=now, finished=None,
                                    validity_reason="running", artifact=str(path)))
        save()  # pins and running ledger exist before producer entry
        active.append((row, producer(row, attempt, path), path))

    save()
    try:
        for i in range(0, len(rows), 2):
            pending = rows[i:i+2]
            while pending or active:
                hits = monitor()
                if data["stop_reason"]:
                    stop_active(hits)
                    break
                for row in pending:
                    launch(row)
                    hits = monitor()
                    if data["stop_reason"]:
                        stop_active(hits)
                        break
                pending = []
                if data["stop_reason"]:
                    break
                for row, process, path in list(active):
                    rc = process.poll()
                    if rc is None:
                        continue
                    if rc == 3 or row in limited():
                        data["stop_reason"] = "rate_limited"
                        row.update(state="rate_limited", validity_reason="rate_limited", finished=elapsed())
                        row["attempts"][-1].update(state="rate_limited", validity_reason="rate_limited", finished=row["finished"])
                        active.remove((row, process, path))
                        continue
                    state, reason, metrics = validate(row, len(row["attempts"]), path)
                    row.update(state=state, validity_reason=reason, finished=elapsed())
                    row["attempts"][-1].update(state=state, validity_reason=reason, finished=row["finished"])
                    if metrics is not None:
                        row["metrics"] = metrics
                    active.remove((row, process, path))
                    if state != "valid":
                        pending.append(row)
                save()
                if active:
                    sleep(2)
            if data["stop_reason"]:
                break
    except BaseException as exc:
        data.update(stop_reason=str(exc) if isinstance(exc, RunSignal) else type(exc).__name__,
                    state="interrupted", raw_verdict="INCONCLUSIVE")
        stop_active([])
        raise
    finally:
        for row in rows:
            if row["state"] in ("planned", "running"):
                row.update(state="missing", validity_reason=data["stop_reason"] or "unfinished", finished=elapsed())
        save()
    data["stop_reason"] = data["stop_reason"] or changed(pins)
    try:
        result = scorer(data)
    except BaseException as exc:
        data.update(stop_reason=str(exc) if isinstance(exc, RunSignal) else type(exc).__name__,
                    state="interrupted", raw_verdict="INCONCLUSIVE")
        save()
        raise
    drift = changed(pins)
    if drift:
        data["stop_reason"] = drift
        result.update(raw_verdict="INCONCLUSIVE", reason=drift)
    save()
    if data["elapsed_s"] >= LIMIT_SECONDS:
        data["stop_reason"] = "elapsed stop"
        result.update(raw_verdict="INCONCLUSIVE", reason="elapsed stop including scoring")
        save()
    d2_score.write_result(out, result)
    result_card(out, result)
    return result


if __name__ == "__main__":
    started = time.monotonic()  # includes argument/config parsing and pin preparation
    wall_started = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--option", choices=("A", "B"), default="B")
    ap.add_argument("--aa", action="store_true")
    ap.add_argument("--comparability-ruling")
    ap.add_argument("--instrument-hash-ruling")
    ap.add_argument("--print-instrument-hash", action="store_true")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    if args.comparability_ruling:
        config["comparability_ruling"] = args.comparability_ruling
    if args.instrument_hash_ruling:
        config["instrument_hash_ruling"] = args.instrument_hash_ruling
    if args.print_instrument_hash:
        print(prepare(config, args.option, args.aa, check_hash=False)["instrument_hash"])
        raise SystemExit(0)
    if args.out is None:
        ap.error("--out is required for a run")
    result = run_comparison(prepare(config, args.option, args.aa), args.out, started=started, wall_started=wall_started)
    print(result["raw_verdict"])
    raise SystemExit({"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[result["raw_verdict"]])
