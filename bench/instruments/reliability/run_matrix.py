"""Run a named cell set against named Hermes hosts and lcm-x refs; write results.jsonl, MATRIX.md, ISSUE-MAP.md.

    python bench/instruments/reliability/run_matrix.py --hosts eva-0.21.5,customer-0.21.2 \\
        --plugin-ref origin/main --cells 'baseline/*' --jobs 8 --out <dir>

Stdlib only. Claim class: advisory / code_green_local (see README.md).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from bench.instruments.reliability import cells as C, controls as CT, hosts as H, plugin_tree, report  # noqa: E402
from bench.instruments.reliability.scorers import bars  # noqa: E402

PROBE = Path(__file__).with_name("probe.py")
PHASES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def slug(cid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", cid)


def config_yaml(cell: dict, plugin: dict) -> str:
    return (f"context:\n  engine: {plugin['engine']}\n"
            f"compression:\n  enabled: true\n  threshold: 0.8\n  in_place: {'true' if cell['in_place'] else 'false'}\n"
            "  target_ratio: 0.3\n"
            + (f"lcm:\n  context_threshold: {cell['lcm_env']['LCM_CONTEXT_THRESHOLD']}\n"
               if "LCM_CONTEXT_THRESHOLD" in cell["lcm_env"] else "")
            + f"plugins:\n  enabled: [{plugin['enabled']}]\n  disabled: []\n")


def backup(src: Path, dst: Path) -> None:
    """WAL-safe copy through the sqlite3 backup API; a missing source is an error."""
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
        d.execute("PRAGMA journal_mode=DELETE")  # a standalone copy: readable with mode=ro, no -wal/-shm
    finally:
        d.close()
        s.close()


def copy_dbs(home: Path, dbdir: Path) -> list[str]:
    """Copy lcm.db and state.db for scoring; every failure is returned, and any failure means the cell is not scored."""
    errors = []
    for name in ("lcm.db", "state.db"):
        try:
            backup(home / name, dbdir / name)
        except (sqlite3.Error, OSError) as exc:
            errors.append(f"{name}: {exc!r}"[:300])
    return errors


def scratch_dir(root: str | Path | None) -> Path:
    """A private per-cell dir under ``root`` (default: $TMPDIR) for the cell's Hermes home, HOME/TMPDIR and the DB
    copies the scorers read: multi-GB state.db / lcm.db files that never land in --out unless a keep flag asks."""
    return Path(tempfile.mkdtemp(prefix="lcmx-cell-", dir=root)).resolve()


def keeps_dbs(keep_dbs: str, rec: dict) -> bool:
    """--keep-dbs: none (default) keeps no copy; fail keeps a non-PASS cell's, or a PASS with host-parity licences."""
    if keep_dbs == "all":
        return True
    return keep_dbs == "fail" and (rec.get("verdict") != "PASS" or any(report.licensed(rec)))


def release_scratch(scratch: Path, d: Path, rec: dict, keep_dbs: str, keep_home: bool) -> None:
    """Once the cell is scored (or has failed): move what the keep flags ask for into the cell dir, delete the rest.
    A failed move (cross-filesystem copy, disk full) deletes nothing: a copy the flags asked to keep is never lost."""
    try:
        if keeps_dbs(keep_dbs, rec) and (scratch / "db").is_dir():
            shutil.move(scratch / "db", d / "db")
        if keep_home and (scratch / "hermes-home").is_dir():
            shutil.move(scratch / "hermes-home", d / "hermes-home")
    except OSError as exc:  # shutil.Error included
        raise OSError(f"keeping the cell's copies failed, nothing deleted: see {scratch} and {d}: {exc!r}") from exc
    shutil.rmtree(scratch, ignore_errors=True)  # delete all it can, then say so if anything is left
    if scratch.exists():
        raise OSError(f"the cell scratch dir {scratch} could not be fully deleted")


def verdict_fields(cell: dict, d: Path, last: dict, fired: set, citations: dict, backup_errors: list[str],
                   db_dir: Path | None = None) -> dict:
    """The record's verdict. The scorers only run on a finished probe with complete DB copies (in ``db_dir``)."""
    if backup_errors:  # takes precedence over every other outcome
        return {"verdict": "ERROR", "reason": "database copy failed, not scored: " + "; ".join(backup_errors)}
    if last["exit"] == "unsupported":
        return {"verdict": "UNSUPPORTED", "reason": last.get("reason")}
    if last["exit"] != "done":
        return {"verdict": "ERROR", "reason": last.get("reason") or last}
    if reason := unfired_reason(cell, fired, citations):
        return {"verdict": "UNSUPPORTED", "reason": reason}
    try:
        scored = bars.score(cell, d, db_dir)
    except Exception as exc:  # a scorer failure is a harness ERROR, never a PASS
        return {"verdict": "ERROR", "reason": f"scoring failed: {exc!r}"}
    rec = {k: scored[k] for k in ("verdict", "failed_bars", "numbers", "inconclusive_bars", "applicable_bars")}
    if scored.get("reason"):
        rec["reason"] = scored["reason"]
    return rec


def unfired_reason(cell: dict, fired: set, citations: dict) -> str | None:
    missing = [f["kind"] for f in cell["faults"] if f["kind"] not in fired]
    if not missing:
        return None
    if "crash_between_session_end_and_start" in missing:
        return ("the host rotation path never called on_session_end before on_session_start(boundary_reason="
                f"'compression') (rotation notify at {citations.get('rotation_start')}; on_session_end only in "
                f"{citations.get('session_transition')})")
    return f"fault trigger(s) {missing} never fired at this host sha"


def with_lcm_env(cell: dict, plugin: dict, lcm_env: dict | None) -> dict:
    if lcm_env:  # a global override wins over the cell's tuning and is part of the cell record
        cell = {**cell, "lcm_env": {**cell["lcm_env"], **lcm_env}, "global_lcm_env": lcm_env}
        if "LCM_NATIVE_RECOVERY" in lcm_env:  # decided by the plugin's own parser at this ref, not "== true"
            cell["native_recovery"], cell["native_recovery_parser"] = plugin_tree.parse_bool(
                Path(plugin["tree"]), "LCM_NATIVE_RECOVERY", lcm_env["LCM_NATIVE_RECOVERY"])
    return cell


def checked_cell_dir(out: Path, d: Path) -> Path:
    if (out / "cells").is_symlink():
        raise ValueError(f"{out / 'cells'} is a symlink; refused")
    if (out / "cells").resolve() not in d.resolve().parents or out.resolve() not in d.resolve().parents:
        raise ValueError(f"cell dir {d} is not under {out / 'cells'}")
    return d


def run_cell(cell: dict, host_name: str, host: dict, plugin: dict, out: Path, timeout: int, keep: bool,
             keep_dbs: str = "none", lcm_env: dict | None = None, identity: dict | None = None,
             scratch_root: str | Path | None = None) -> dict:
    cell = with_lcm_env(cell, plugin, lcm_env)
    d = checked_cell_dir(out, out / "cells" / host_name / plugin["sha"][:12] / slug(cell["id"]))
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "cell.json").write_text(json.dumps({**cell, "plugin": plugin, "host": host_name, "host_src": host["src"],
                                             "host_python": host["python"]}, indent=1))
    rec = {"cell": cell["id"], "host": host_name, "host_sha": host["sha"], "plugin_ref": plugin["ref"],
           "plugin_sha": plugin["sha"], "targets": cell["targets"], "dir": str(d), "host_identity": identity}
    if not identity or "error" in identity:  # the host tree is not proven to be the configured sha
        rec.update(verdict="ERROR", reason=f"host identity not verified: {(identity or {}).get('error')}")
        (d / "verdict.json").write_text(json.dumps(rec, indent=1, default=str))
        return rec
    if cell.get("drain"):  # decided before any probe: no host run, no DB to copy
        rec.update(verdict="UNSUPPORTED",
                   reason="drain cells need the R2 observer's per-compaction counters (acp-process only)")
        (d / "verdict.json").write_text(json.dumps(rec, indent=1, default=str))
        return rec
    s = scratch_dir(scratch_root)
    try:
        home = s / "hermes-home"
        (home / "plugins").mkdir(parents=True)
        (s / "home").mkdir()
        (s / "db").mkdir()
        start = cell.get("from_plugin") or plugin  # native-on-off: an older ref runs until the plugin_switch fault
        (home / "plugins" / start["dir"]).symlink_to(start["tree"])
        (home / "config.yaml").write_text(config_yaml(cell, start))
        env = {"HOME": str(s / "home"), "PATH": "/usr/bin:/bin", "HERMES_HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(d / "pycache"),
               "OPENROUTER_API_KEY": "test-key", "TMPDIR": str(s / "home"),
               **cell["lcm_env"], "LCM_NATIVE_RECOVERY": "true" if cell["native_recovery"] else "false"}
        started, start_turn, last, phases_run = time.time(), 1, {}, []
        for phase in PHASES:
            try:
                done = subprocess.run([host["python"], str(PROBE), "--cell", str(d / "cell.json"), "--phase", phase,
                                       "--start-turn", str(start_turn), "--cell-dir", str(d)],
                                      cwd=host["src"], env=env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                last = {"exit": "error", "reason": f"phase {phase} timed out after {timeout}s"}
                break
            (d / f"probe-{phase}.log").write_text(done.stdout[-20000:] + "\n--- stderr ---\n" + done.stderr[-20000:])
            phases_run.append(phase)
            lines = [x for x in done.stdout.splitlines() if x.startswith('{"exit"')]
            last = json.loads(lines[-1]) if lines else {"exit": "error", "reason": f"phase {phase} rc={done.returncode}: "
                                                        + (done.stderr.strip().splitlines() or ["no output"])[-1][:300]}
            if last["exit"] == "plugin_switch":  # the candidate takes over the same home and store, native OFF
                (home / "plugins" / start["dir"]).unlink()
                (home / "plugins" / plugin["dir"]).symlink_to(plugin["tree"])
                (home / "config.yaml").write_text(config_yaml(cell, plugin))
                env["LCM_NATIVE_RECOVERY"] = "false"
            if last["exit"] in ("crash", "clean_exit", "tip_switch", "plugin_switch"):
                start_turn = last["next_turn"]
                continue
            break
        else:
            last = {"exit": "error", "reason": "phase budget exhausted"}
        backup_errors = copy_dbs(home, s / "db")  # consulted only for a finished probe (verdict_fields)
        fired_file = d / "faults-fired.jsonl"
        fired = {json.loads(x)["kind"] for x in fired_file.read_text().splitlines()} if fired_file.exists() else set()
        first_phase = d / "phase-A.json"
        citations = json.loads(first_phase.read_text()).get("citations", {}) if first_phase.exists() else {}
        rec.update(phases=phases_run, wall_s=round(time.time() - started, 1), citations=citations)
        rec.update(verdict_fields(cell, d, last, fired, citations, backup_errors, s / "db"))
        (d / "verdict.json").write_text(json.dumps(rec, indent=1, default=str))
    finally:  # scored or not: the scratch DBs go, unless --keep-dbs / --keep-homes moves them into the cell dir
        release_scratch(s, d, rec, keep_dbs, keep)
    if not keep:
        shutil.rmtree(d / "files", ignore_errors=True)
    return rec


def env_refusal(lcm_env: dict) -> str | None:
    """--lcm-env takes LCM_* tuning only: no path-valued keys (a path could point at a live DB) and no secrets
    (they would be persisted in run.json, cell.json and the reports)."""
    for k in lcm_env:
        if not k.startswith("LCM_"):
            return f"--lcm-env key {k} must start with LCM_"
        if re.search(r"(_PATH|_DIR|_HOME|_FILE)$", k, re.I):
            return f"--lcm-env key {k} is path-valued; refused"
        # "_TOKENS" names a token COUNT (LCM_LEAF_CHUNK_TOKENS, ..._THRESHOLD_TOKENS), not a credential.
        if re.search(r"KEY|TOKEN|SECRET|PASSWORD", re.sub(r"_TOKENS(?=_|$)", "", k, flags=re.I), re.I):
            return f"--lcm-env key {k} is secret-shaped; refused"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts-file")
    ap.add_argument("--control", choices=sorted(CT.CONTROLS), help="run a positive control and check its pattern")
    ap.add_argument("--hosts", help="comma-separated host names, or 'all'")
    ap.add_argument("--plugin-ref", help="comma-separated lcm-x git refs")
    ap.add_argument("--lcm-repo", default=str(REPO))
    ap.add_argument("--cells", help="'all' or comma-separated globs")
    ap.add_argument("--jobs", type=int, default=min(8, max(1, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--timeout", type=int, default=900, help="per-phase timeout, seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-homes", action="store_true", help="debugging: keep each cell's hermes-home in its cell dir")
    ap.add_argument("--keep-dbs", choices=("none", "fail", "all"), default="none",
                    help="debugging: keep the db/ copies of no cell (default), of non-PASS / licensed cells, or of all")
    ap.add_argument("--scratch-root", help="parent of the per-cell scratch dirs (Hermes home, DB copies); default $TMPDIR")
    ap.add_argument("--lcm-env", action="append", default=[], metavar="KEY=VAL",
                    help="LCM_* override applied to every cell (repeatable)")
    ap.add_argument("--transport", choices=("acp-process", "gateway-process", "api-server"),
                    help="R2: run the cells through a real host process (process_cell.py); default: R1 in-process")
    ap.add_argument("--turn-timeout", type=float, default=300.0, help="R2: per ACP request timeout, seconds")
    a = ap.parse_args(argv)
    if a.control:
        ctl = CT.CONTROLS[a.control]
        a.hosts = ",".join(ctl["hosts"]) if ctl["hosts"] != "all" else "all"
        a.plugin_ref, a.cells = ",".join(ctl["refs"]), ",".join(ctl["cells"])
    if not (a.hosts and a.plugin_ref and a.cells):
        ap.error("--hosts, --plugin-ref and --cells are required without --control")
    out = Path(a.out).resolve()
    if str(out) == "/tmp" or str(out).startswith(("/tmp/", "/private/tmp")):
        ap.error("--out must not be under /tmp")
    if H.under_real_hermes(out):
        ap.error("--out must not be under the live ~/.hermes")
    scratch_root = Path(a.scratch_root or tempfile.gettempdir())
    if H.under_real_hermes(scratch_root):
        ap.error("--scratch-root must not be under the live ~/.hermes")
    scratch_root = scratch_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    lcm_env = dict(kv.split("=", 1) for kv in a.lcm_env)
    if bad := env_refusal(lcm_env):
        ap.error(bad)
    hosts = H.load(H.hosts_file(a.hosts_file), None if a.hosts == "all" else a.hosts.split(","))
    identities = {}
    for name, host in hosts.items():
        try:
            identities[name] = H.verify(name, host)
        except (ValueError, OSError) as exc:
            identities[name] = {"error": str(exc)}
    if a.transport:  # R2 (process_cell.py); without --transport the R1 in-process path is unchanged
        from bench.instruments.reliability import process_cell
    selected = C.select(a.cells, extra=process_cell.R2_CELLS if a.transport else ())
    plugins = [plugin_tree.export(Path(a.lcm_repo), ref.strip(), out / "plugins", out) for ref in a.plugin_ref.split(",")]
    for ref in sorted({c["from_ref"] for c in selected if c.get("from_ref")}):  # native-on-off starting refs
        old = plugin_tree.export(Path(a.lcm_repo), ref, out / "plugins", out)
        selected = [{**c, "from_plugin": old} if c.get("from_ref") == ref else c for c in selected]
    if len({p["sha"] for p in plugins}) < len(plugins):
        ap.error(f"--plugin-ref values resolve to the same commit: {[(p['ref'], p['sha'][:12]) for p in plugins]}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"argv": sys.argv, "hosts": hosts, "plugins": plugins, "lcm_env": lcm_env,
                                              "host_identity": identities, "cells": [c["id"] for c in selected],
                                              **({"transport": a.transport} if a.transport else {})}, indent=1))
    jobs = [(c, h, hosts[h], p) for p in plugins for h in hosts for c in selected]
    runner = run_cell
    if a.transport:
        from functools import partial
        runner = partial(process_cell.run_cell_process, transport=a.transport, turn_timeout=a.turn_timeout)
    started, results = time.time(), []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool, open(out / "results.jsonl", "w") as sink:
        futures = {pool.submit(runner, c, h, hd, p, out, a.timeout, a.keep_homes, a.keep_dbs, lcm_env,
                               identities[h], scratch_root=scratch_root): (c, h, p) for c, h, hd, p in jobs}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as exc:  # any job failure becomes one bounded ERROR record, never a lost cell
                c, h, p = futures[fut]
                rec = {"cell": c["id"], "host": h, "host_sha": hosts[h]["sha"], "plugin_ref": p["ref"],
                       "plugin_sha": p["sha"], "targets": c["targets"], "verdict": "ERROR",
                       **({"transport": a.transport} if a.transport else {}),
                       "reason": f"harness job failed: {exc!r}"[:500]}
            results.append(rec)
            sink.write(json.dumps(rec, default=str) + "\n")
            sink.flush()
            print(f"{rec['verdict']:<11} {rec['host']:<16} {rec['plugin_ref']:<12} {rec['cell']}"
                  + (f"  {sorted(rec.get('failed_bars', {}))}" if rec.get("failed_bars") else "")
                  + (f"  {str(rec.get('reason'))[:120]}" if rec.get("reason") else ""), flush=True)
    report.write(out, results, time.time() - started, lcm_env)
    if a.control:
        problems = CT.check(a.control, results, list(hosts))
        (out / "CONTROL.json").write_text(json.dumps({"control": a.control, "holds": not problems, "problems": problems}, indent=1))
        print(f"CONTROL {a.control}: {'HOLDS' if not problems else 'DOES NOT HOLD'} {problems[:5]}", flush=True)
        return 0 if not problems else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
