"""D5 clone-run CLI (bench/specs/RELEASE-READINESS-V1.md, "D5 clone run"): orchestration only.

Order: free space, P1 pins, P2 snapshot, the plan (on a clone), P3-P7 (any failure stops the run before either arm),
the rc arm then the GA arm on APFS clones of the verified snapshot, T0/H8 captures around each rc-arm cold bind with
that bind's replay right after its H8 (then its captures are removed), the bars, then the receipt. Every external
effect is a LocalHost method, so a test can run the whole flow with a fake host and a fake edition. The runner never
opens the pinned snapshot itself, only its clones; P2 hashes its files and checks SQLite on scratch clones.

The rules module (``rules_module`` in the run config, a pinned file in the instrument directory, imported only after
P1) provides ``identity(row)``, ``credits`` (the two host-rewrite credits, each ``(row, host_row) -> bool``) and
``covered(session_id)`` for planning. The edition steps are argv templates in the config: ``constants_argv`` writes
``{out}/constants.json`` (with ``range_new_rows``) and ``replay_argv`` writes ``{out}/receipt.json``.
"""

import argparse
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
from contextlib import closing
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time

from . import bars, capture, gateway, pins, plan as planner, provider as prov, sandbox, snapshot
from ..release_gauntlet import kit_evidence

SCHEMA = "d5-clone-receipt/1"
NEED_FREE = 40 * 10**9
PREFLIGHTS = ("P3", "P4", "P5", "P6", "P7")
DOCTOR_STATUS = {"pass": "ok", "ok": "ok", "warn": "warn", "fail": "fail"}  # Any other status stays, and fails bar 5.
# LCM-X's externalized-payload reference (externalize.py _EXTERNALIZED_REF_RE).
EXTERNALIZED_REF = re.compile(r"\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]")
CRON_MODEL = "d5-cron"
CRON_PAUSE_REASON = "d5 clone run: paused before the run"
# How the probe can exercise each lcm_doctor check on a stopped arm (both trees' tools.py lcm_doctor). Store-wide
# checks read the store, the payload files or the arm's config once; per-session checks read the engine's current
# session, so they run once per plan session, bound by on_session_start. The rest never see the run: they are left
# out of bar 5 and reported (release manager, 2026-11-07), and so is any check not named here.
DOCTOR_STORE_WIDE = frozenset((
    "database_integrity", "schema_core_tables", "messages_fts_integrity", "nodes_fts_integrity", "sqlite_storage",
    "payload_storage", "source_lineage_hygiene", "lifecycle_fragmentation", "sensitive_pattern_handling",
    "config_validation", "embedding_provider_health"))
DOCTOR_PER_SESSION = ("fts_index_sync", "orphaned_dag_nodes", "summary_quality")
DOCTOR_NOT_EXERCISED = {
    "identity_migration": "no plugin registration step in the probe",
    "inactive_process": "reads live processes; the arm's gateway has stopped",
    "ingest_health": "the probe process's own counter", "ignore_pattern_drops": "the probe process's own counter",
    "context_pressure": "the probe process's own counter", "compaction_progress": "the probe process's own counter",
    "token_counter": "the probe process's tokenizer state"}


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rows(db, sql, args=()):
    """A read-only query's rows; a store, table or column that is not there reads as no rows."""
    try:
        with closing(capture._readonly(db)) as con:
            return con.execute(sql, args).fetchall()
    except (OSError, sqlite3.Error, capture.CaptureError):
        return []


def _json(value):
    """A JSON object, else None."""
    try:
        value = json.loads(value)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _sha(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


def marks(home):
    """Where the arm's stores stand before its gateway first starts: what facts() counts from."""
    lcm, host = Path(home) / "lcm.db", Path(home) / "state.db"

    def top(db, sql):
        return (_rows(db, sql) or [(0,)])[0][0] or 0

    return {"store_id": top(lcm, "SELECT max(store_id) FROM messages"), "host_id": top(host, "SELECT max(id) FROM messages"),
            "node_id": top(lcm, "SELECT max(node_id) FROM summary_nodes"),
            "proofs": {k: _sha(v) for k, v in _rows(lcm, "SELECT key, value FROM metadata WHERE key GLOB 'compaction_commit_proof:*'")},
            "rewrites": sorted(k for k, in _rows(lcm, "SELECT key FROM metadata WHERE key GLOB 'host_rewrite_identity:*'"))}


def workload_facts(home, before, cron_id=None):
    """Bar 6's host observations, counted from the stopped arm's own files against its marks (counts only)."""
    lcm, host = Path(home) / "lcm.db", Path(home) / "state.db"
    store_after, host_after = (before.get("store_id", 0),), (before.get("host_id", 0),)
    # A published compaction: a session whose LCM-X commit proof row changed and is not native, with a new summary node.
    nodes = {s for s, in _rows(lcm, "SELECT DISTINCT session_id FROM summary_nodes WHERE node_id > ?", (before.get("node_id", 0),))}
    published = sum(key.partition(":")[2] in nodes and before.get("proofs", {}).get(key) != _sha(value)
                    and (_json(value) or {}).get("native") is False
                    for key, value in _rows(lcm, "SELECT key, value FROM metadata WHERE key GLOB 'compaction_commit_proof:*'"))
    # A whitespace rewrite: a host_rewrite_identity row for a new user row, recording the edge whitespace stripped.
    users = {str(i) for i, in _rows(lcm, "SELECT store_id FROM messages WHERE role = 'user' AND store_id > ?", store_after)}
    rewrites = sum(key not in before.get("rewrites", ()) and key.partition(":")[2] in users
                   and any((_json(value) or {}).get("stripped") or ())
                   for key, value in _rows(lcm, "SELECT key, value FROM metadata WHERE key GLOB 'host_rewrite_identity:*'"))
    # A payload swap, in the shape the pinned payload credit rewards: a new host tool row that is one reference to one
    # payload file, whose content LCM-X stores in a row of the same session, role and tool_call_id.
    swaps = 0
    for s, c, text in _rows(host, "SELECT session_id, tool_call_id, content FROM messages WHERE role = 'tool'"
                                  " AND tool_call_id IS NOT NULL AND id > ?", host_after):
        payload = _swap_payload(home, text)
        swaps += payload is not None and bool(_rows(lcm, "SELECT 1 FROM messages WHERE session_id = ? AND role = 'tool'"
                                                         " AND tool_call_id = ? AND content = ? LIMIT 1", (s, c, payload)))
    fired = 0
    if cron_id:  # A run is the host's cron session for the job (cron/scheduler.py), or its saved output.
        sessions = len(_rows(host, "SELECT id FROM sessions WHERE id GLOB ?", (f"cron_{cron_id}_*",)))
        fired = max(sessions, sum(p.is_file() for p in (Path(home) / "cron" / "output" / cron_id).glob("*.md")))
    return {"cron_fired": fired, "compactions_published": published, "whitespace_rewrites": rewrites,
            "payload_swaps": swaps}


def _swap_payload(home, text):
    """The payload content a host row that is exactly one engine-form reference resolves to, else None (the pinned
    payload credit: one whole marker, a basename ref, one payload file whose decoded content is a str)."""
    marker = EXTERNALIZED_REF.fullmatch(text) if isinstance(text, str) else None
    if marker is None or len(EXTERNALIZED_REF.findall(text)) != 1 or Path(marker[1]).name != marker[1] \
            or "quarantined_assistant_output" in text:
        return None
    path = Path(home) / "lcm-large-outputs" / marker[1]
    try:
        payload = json.loads(path.read_bytes().decode("utf-8")) if path.is_file() else None
    except (OSError, ValueError, UnicodeError):
        return None
    return payload["content"] if isinstance(payload, dict) and isinstance(payload.get("content"), str) else None


def pause_cron_jobs(home, now):
    """Pause every job already in the arm's host job file, as the host's pause_job does (cron/jobs.py), keeping the
    file's shape: a declared copy edit, the same in both arms. Returns the paused job ids (ids, not content)."""
    path = Path(home) / "cron" / "jobs.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    jobs = data if isinstance(data, list) else data.get("jobs", [])
    jobs = [j for j in (jobs.values() if isinstance(jobs, dict) else jobs) if isinstance(j, dict)]
    for job in jobs:
        job.update(enabled=False, state="paused", paused_at=now.isoformat(), paused_reason=CRON_PAUSE_REASON)
    if jobs:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    ids = sorted(str(j.get("id")) for j in jobs)
    return {"job_ids": ids, "count": len(ids)}


def cron_job(cron, now):
    """The plan's job as the host's create_job records it (cron/jobs.py). Its id comes from the plan, so both arms
    create the same job; only the creation and next-run times differ."""
    match = re.fullmatch(r"every (\d+)m", cron["schedule"])
    if not match:
        raise ValueError("only an 'every <n>m' cron schedule is supported")
    minutes, empty = int(match[1]), dict.fromkeys((
        "skill", "base_url", "script", "monitor_script", "monitor_url", "monitor_state", "context_from", "paused_at",
        "paused_reason", "last_run_at", "last_status", "last_error", "last_delivery_error", "last_delivery_unverified",
        "last_failure", "origin", "enabled_toolsets", "workdir"))
    schedule = {"kind": "interval", "minutes": minutes, "display": f"every {minutes}m"}
    return {**empty, "id": _sha(cron["prompt"])[:12], "name": cron["prompt"][:50].strip(), "prompt": cron["prompt"],
            "skills": [], "model": CRON_MODEL, "provider": "d5", "no_agent": False, "schedule": schedule,
            "schedule_display": schedule["display"], "repeat": {"times": cron["repeat"], "completed": 0},
            "enabled": True, "state": "scheduled", "created_at": now.isoformat(),
            "next_run_at": (now + timedelta(minutes=minutes)).isoformat(), "failure_streak": 0, "deliver": "local"}


def _doctor_rows(path):
    """A doctor result's (check, status) rows through the gauntlet's doctor-match parser, else None."""
    try:
        return [(r["check"], DOCTOR_STATUS.get(r["status"], r["status"]))
                for r in kit_evidence.doctor_match(path, path)["candidate_signatures"]]
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def probe_checks(out, sessions):
    """(bar-5 rows, not exercised) from the probe's files in *out*; *sessions* are the plan sessions' sha256 12 keys.
    A kind the probe did not produce is left out, so bar 5 reads it as not shown. Store-wide doctor checks come from
    the unbound run, per-session ones from each session's run as ``<check>@<key>``; a check the probe cannot exercise
    is never a row, only {check, reason}."""
    found, skipped = [], []
    if out is None:
        return found, skipped
    rows = _doctor_rows(out / "doctor.json")
    if rows is not None:
        bound = (_json((out / "sessions.json").read_text(encoding="utf-8")) if (out / "sessions.json").is_file()
                 else None) or {}
        for check, status in rows:
            if check in DOCTOR_STORE_WIDE:
                found.append({"kind": "doctor", "check": check, "status": status})
            elif check not in DOCTOR_PER_SESSION:
                skipped.append({"check": check, "reason": DOCTOR_NOT_EXERCISED.get(check, "not classified for the probe")})
        for key in sessions:
            scoped = _doctor_rows(out / f"doctor-{key}.json") if bound.get(key) is True else None
            found += [{"kind": "doctor", "check": f"{c}@{key}", "status": st}
                      for c, st in scoped or () if c in DOCTOR_PER_SESSION]
            skipped += [{"check": f"{c}@{key}", "reason": "no session binding in this tree"}
                        for c in DOCTOR_PER_SESSION if scoped is None]
    schema = _json((out / "schema.json").read_text(encoding="utf-8")) if (out / "schema.json").is_file() else None
    if isinstance(schema, dict) and set(schema) == {"lcm.db", "state.db"}:
        found += [{"kind": "schema", "check": name, "status": "ok" if isinstance(s, dict) and s.get("error") is False
                   and s.get("missing_tables") == [] and type(s.get("version")) is int and s["version"] == s.get("expected")
                   else "fail"} for name, s in sorted(schema.items())]
    return found, sorted(skipped, key=lambda item: item["check"])


class LocalHost:
    """The real effects. ``config`` is the run declaration; ``run_dir`` holds every clone, capture and log."""

    def __init__(self, config, run_dir):
        self.c, self.run_dir, self._rules = config, Path(run_dir), None

    def _dir(self, *parts):
        path = self.run_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def free_bytes(self):
        return shutil.disk_usage(self._dir()).free

    def pins(self):
        args, kwargs = (self.c["manifest"], self.c["instrument_dir"]), {"interpreter": self.c.get("interpreter")}
        try:
            pins.require(*args, **kwargs)  # SystemExit(2) on any mismatch, before anything is imported.
        except SystemExit:
            return pins.verify(*args, **kwargs)
        manifest = json.loads(Path(self.c["manifest"]).read_text(encoding="utf-8"))
        return {**pins.verify(*args, **kwargs), "manifest_sha256": capture._sha256(self.c["manifest"]),
                "functions_sha256": _sha_json(manifest["functions"])}

    def snapshot(self):
        return snapshot.verify(self.c["snapshot_pin"], self.c["snapshot_dir"], self._dir("scratch"))

    def clone(self, name):
        dest = self._dir(name) / "home"
        capture.clone_tree(self.c["snapshot_dir"], dest)
        return dest

    def rules(self):
        if self._rules is None:
            sys.dont_write_bytecode = True  # Bytecode beside a pinned file would fail the pins.
            path = pins._load_path(self.c["instrument_dir"], self.c["rules_module"])  # Refuses absolute and '..'.
            spec = importlib.util.spec_from_file_location("d5_clone_rules", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._rules = {"identity": module.identity, "credits": tuple(module.credits), "covered": module.covered,
                           "configure_planning": module.configure_planning, "configure_bar1": module.configure_bar1}
        return self._rules

    def configure_planning(self, copy):
        """The rules module's planning setup (the pinned host rebuilds each history), before choose_sessions."""
        self.rules()["configure_planning"](plan_home=copy, host_py=self.c.get("host_py"), pin=self.c.get("host_pin"),
                                           scratch=self.run_dir / "planning")

    def configure_bar1(self, **inputs):
        """The rules module's credit maps over the rows bar 1 gets, immediately before bars.bar1."""
        self.rules()["configure_bar1"](**inputs)

    def plan(self, copy):
        self.configure_planning(copy)
        chosen = planner.choose_sessions(copy / "host-0.db", copy / "store.db", self.rules()["covered"])
        return planner.build_plan(chosen, nonce=self.c["nonce"])

    def rows(self, path):
        out = {}
        for kind, names, order in (("store", ("store.db", "lcm.db"), "store_id"), ("host", ("host-0.db", "state.db"), "id")):
            db = next(Path(path) / n for n in names if (Path(path) / n).is_file())
            with closing(capture._readonly(db)) as con:
                con.row_factory = sqlite3.Row
                out[kind] = [dict(r, _dir=str(path)) for r in con.execute(f"SELECT * FROM messages ORDER BY {order}")]
        return out

    def preflight(self, name, plan):
        if name == "P5":  # Egress: run live under the sandbox profile.
            server = prov.Provider(plan["provider"]).start()
            try:
                return sandbox.positive_control(server.url + "/models")
            finally:
                server.stop()
        path = self.c.get("preflights", {}).get(name)  # The others: a recorded preflight receipt, by sha256.
        if not path:
            return {"verdict": "MISSING"}
        return {"verdict": json.loads(Path(path).read_text(encoding="utf-8")).get("verdict"),
                "sha256": capture._sha256(path)}

    def arm(self, name, home, plan):
        tree = self.c[name]
        for old, new in (("store.db", "lcm.db"), ("host-0.db", "state.db")):
            for suffix in ("", "-wal", "-shm"):
                if (home / (old + suffix)).exists():
                    (home / (old + suffix)).rename(home / (new + suffix))
        server = prov.Provider(plan["provider"]).start()
        routing = gateway.edit_copy_routing(home / "state.db", plan["conversations"], sessions_dir=home / "sessions",
                                            run_dir=self.run_dir, provider_url=server.url)
        gateway.write_config(home, gateway.build_config(
            self.c["settings"], provider_url=server.url, home=home, aux=gateway.aux_tasks(tree["host_src"]),
            context_length=self.c["context_length"], stored_route_providers=routing["stored_route_providers"]))
        env, _ = gateway.build_env(self.c["lcm_env"], home=home, provider_url=server.url, host_bin=tree["host_bin"],
                                   port_file=home.parent / "d5-local.port", lcm_tree=tree["lcm_tree"])
        port = server.server.server_address[1]
        return {"name": name, "home": home, "provider": server, "port": port, "tree": tree, "marks": marks(home),
                "session_ids": [c["session_id"] for c in plan["conversations"]],
                "gateway": gateway.Gateway(tree["host_bin"], home, env, home.parent, provider_port=port),
                "payloads": sum(p.is_file() for p in (home / "lcm-large-outputs").rglob("*")),
                "chats": {c["name"]: c["chat_id"] for c in plan["conversations"]}}

    def start(self, ctx):
        ctx["gateway"].start(self.c.get("start_timeout", 300))

    def stop(self, ctx):
        ctx["gateway"].stop()

    def turn(self, ctx, step):
        with ctx["provider"].lock:
            ctx["provider"].step = step["id"]
        try:
            return ctx["gateway"].turn(ctx["chats"][step["conversation"]], step["id"], step["user_text"]).get(
                "completed") is True
        except OSError:
            return False
        finally:
            with ctx["provider"].lock:
                ctx["provider"].step = None

    def capture(self, ctx, kind, step, extras=None):
        out = self._dir(ctx["name"], "captures") / f"{step['id']}-{kind}"
        home = ctx["home"]
        return out, capture.capture(kind, out, store_db=home / "lcm.db", host_db=home / "state.db",
                                    payload_dir=home / "lcm-large-outputs", extras=extras, home=home)

    def replay(self, ctx, step, t0, h8):
        """The edition on both trees, under the sandbox: (receipt, error, range_new_rows)."""
        out = self._dir(ctx["name"], "replays", step["id"])
        values = {"python": self.c["interpreter"], "instrument_dir": self.c["instrument_dir"], "t0": t0, "h8": h8,
                  "out": out, "rc_tree": self.c["rc"]["lcm_tree"], "ga_tree": self.c["ga"]["lcm_tree"]}
        for key in ("constants_argv", "replay_argv"):
            argv = sandbox.wrap([a.format(**values) for a in self.c[key]], ports=[ctx["port"]], socket_roots=[out])
            env = {"PATH": "/usr/bin:/bin", "HOME": str(out), "PYTHONDONTWRITEBYTECODE": "1"}
            if subprocess.run(["/usr/bin/nice", "-n", "19", *argv], cwd=out, env=env, capture_output=True).returncode:
                return None, True, None
        try:
            constants = json.loads((out / "constants.json").read_text(encoding="utf-8"))
            return json.loads((out / "receipt.json").read_text(encoding="utf-8")), False, constants.get("range_new_rows")
        except (OSError, ValueError):
            return None, True, None

    def remove(self, path):
        capture.remove_capture(path, receipt_written=True)

    def egress(self, ctx):
        return gateway.nonlocal_connections(ctx["gateway"].pids())

    def wait(self, seconds):
        time.sleep(max(0, seconds))

    def logs(self, ctx):
        files = sorted(ctx["home"].parent.glob("gateway-*.out")) + sorted((ctx["home"] / "logs").glob("*.log"))
        return [line for f in files for line in f.read_text(encoding="utf-8", errors="replace").splitlines()]

    def roots(self, ctx):
        return [str(ctx["home"] / "plugins" / "hermes-lcm-x"), str(Path(ctx["tree"]["lcm_tree"]).resolve())]

    def schedule(self, ctx, cron):
        """Add the plan's cron job to the arm home's host job file, as create_job adds one; returns its id's sha256."""
        path = ctx["home"] / "cron" / "jobs.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"jobs": []}
        job = cron_job(cron, datetime.now().astimezone())
        if isinstance(data, list):  # The file's own shape is kept.
            data.append(job)
        else:
            data.update(jobs=[*data.get("jobs", []), job], updated_at=job["created_at"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        ctx["cron_id"] = job["id"]
        return _sha(job["id"])

    def pause_cron(self, ctx):
        return pause_cron_jobs(ctx["home"], datetime.now().astimezone())

    def probe(self, ctx):
        """The doctor and schema probe, by the arm's host interpreter under the sandbox, on a clone of the stopped arm
        home (the engine may write to the store it opens). Returns the probe's output dir, or None if it failed."""
        clone, out = ctx["home"].parent / "doctor" / "home", ctx["home"].parent / "doctor" / "out"  # Per arm clone.
        try:
            out.mkdir(parents=True)
            capture.clone_tree(ctx["home"], clone)
        except (OSError, subprocess.CalledProcessError):
            return None
        env = {**ctx["gateway"].env, "HERMES_HOME": str(clone), "HOME": str(clone / "user-home"), "TMPDIR": str(clone / "tmp"),
               **{k: str(clone / v) for k, v in gateway.FORCED_LCM_PATHS.items()}}
        tree = ctx["tree"]
        argv = sandbox.wrap([str(Path(tree["host_bin"]) / "python"), "-P", "-B",  # Never this dir's modules first.
                             str(Path(__file__).with_name("doctor_probe.py")),
                             str(Path(tree["lcm_tree"]).resolve()), str(clone), str(out),
                             json.dumps(ctx["session_ids"])], ports=[])
        try:
            done = subprocess.run(["/usr/bin/nice", "-n", "19", *argv], cwd=clone, env=env, capture_output=True,
                                  timeout=self.c.get("doctor_timeout", 1800))
        except subprocess.TimeoutExpired:
            return None
        return out if done.returncode == 0 else None

    def checks(self, ctx):
        """quick_check and integrity on both stores, then the doctor's checks and both stores' schema (probe_checks)."""
        out = []
        for name in ("lcm.db", "state.db"):
            with closing(capture._readonly(ctx["home"] / name)) as db:
                for kind, pragma in (("quick_check", "quick_check"), ("integrity", "integrity_check")):
                    ok = db.execute(f"PRAGMA {pragma}").fetchall() == [("ok",)]
                    out.append({"kind": kind, "check": name, "status": "ok" if ok else "fail"})
        rows, ctx["doctor_not_exercised"] = probe_checks(self.probe(ctx), [_sha(s)[:12] for s in ctx["session_ids"]])
        return out + rows

    def facts(self, ctx):
        """Workload observations counted from the arm's own files after it ends (workload_facts)."""
        now = sum(p.is_file() for p in (ctx["home"] / "lcm-large-outputs").rglob("*"))
        return {"payload_files_created": now - ctx["payloads"],
                **workload_facts(ctx["home"], ctx.get("marks") or {}, ctx.get("cron_id"))}

    def stats(self, ctx):
        with ctx["provider"].lock:
            stats = ctx["provider"].stats
            return {**{k: dict(v) for k, v in stats.items() if isinstance(v, dict)},
                    "agent_calls_outside_step": stats["agent_calls_outside_step"], "requests": len(stats["request_log"])}

    def close(self, ctx):
        ctx["provider"].stop()


def run_arm(host, name, home, plan, extras):
    ctx = host.arm(name, home, plan)
    cron_paused = host.pause_cron(ctx)  # Before the first gateway start: no snapshot job fires in the run.
    sid = {c["name"]: c["session_id"] for c in plan["conversations"]}
    binds, restarts, egress, tool_turns, started = [], [], [], 0, time.monotonic()
    planned, not_completed, cron_job_sha256 = 0, [], None
    try:
        for epoch in plan["epochs"]:
            if epoch["restart_before"]:
                host.stop(ctx)
            if epoch["epoch"] == plan["cron"]["created_before_epoch"]:  # Created while the gateway is stopped.
                cron_job_sha256 = host.schedule(ctx, plan["cron"])
            host.start(ctx)
            begun, cold = time.monotonic(), []
            for step in epoch["steps"]:
                bind = name == "rc" and step["kind"] == "cold"
                if bind:
                    t0_dir, t0 = host.capture(ctx, "T0", step, extras)
                planned += 1
                try:
                    ok = host.turn(ctx, step) is True
                except Exception:  # A turn that raises or times out is not completed; it is never swallowed.
                    ok = False
                not_completed += [] if ok else [step["id"]]
                cold += [step["conversation"]] if ok and step["kind"] == "cold" else []
                tool_turns += bool(ok and step["tools"])
                if bind:
                    h8_dir, h8 = host.capture(ctx, "H8", step)
                    receipt, error, range_new = host.replay(ctx, step, t0_dir, h8_dir)
                    binds.append({"bind": step["id"], "session_id": sid[step["conversation"]],
                                  "t0_max": t0["store_max_store_id"], "h8_max": h8["store_max_store_id"],
                                  "dup_store_ids": bars.dup_store_ids(receipt), "receipt": receipt,
                                  "t0_at": datetime.fromisoformat(t0["captured_at"]).timestamp(),
                                  "t0_capture_sha256": t0["capture_hashes_sha256"],
                                  "h8_capture_sha256": h8["capture_hashes_sha256"],
                                  "bar2": bars.bar2_bind(receipt, range_new_rows=range_new, error=error)})
                    if receipt is not None:  # Kept for diagnosis when the replay left no receipt (r2.6).
                        host.remove(t0_dir)
                        host.remove(h8_dir)
                egress.append(host.egress(ctx))
            host.wait(epoch["min_seconds"] - (time.monotonic() - begun))
            if epoch["restart_before"]:
                restarts.append({"epoch": epoch["epoch"], "cold_binds": cold})
    finally:
        host.stop(ctx)
    stats = host.stats(ctx)
    host.close(ctx)
    workload = {"conversation_names": list(sid), "restarts": restarts, "tool_heavy_turns": tool_turns,
                "large_conversation_rows": max(c["lcm_rows"] for c in plan["conversations"]),
                "replay_misses": sum(stats.get("replay_misses", {}).values()), "turns_planned": planned,
                "turns_completed": planned - len(not_completed), "turns_not_completed": not_completed,
                "cron_job_sha256": cron_job_sha256, **host.facts(ctx)}
    checks = host.checks(ctx)
    return {"binds": binds, "workload": workload, "egress": {"samples": len(egress), "non_local": sum(egress)},
            "provider": stats, "logs": host.logs(ctx), "roots": host.roots(ctx), "checks": checks,
            "doctor_not_exercised": ctx.get("doctor_not_exercised", []), "cron_paused": cron_paused,
            "rows": host.rows(home), "wall_s": round(time.monotonic() - started, 3)}


def run(host, *, rc_id, prior=(), instrument_fix=None):
    """The whole run; returns the receipt (counts, ids, sha256s and verdicts only)."""
    started = time.monotonic()
    prior = [p for p in prior if p.get("rc") == rc_id and "verdict" in p]  # A stopped run is not an attempt.
    prior_verdicts = [p["verdict"]["raw"] for p in prior]
    receipt = {"schema": SCHEMA, "rc": rc_id, "attempt": len(prior_verdicts) + 1, "instrument_fix": instrument_fix}
    refusal = bars.may_run(prior_verdicts, instrument_fix)
    receipt["preflights"] = steps = {"free_space": {"verdict": "PASS" if host.free_bytes() >= NEED_FREE else "FAIL"}}
    if refusal is None and steps["free_space"]["verdict"] == "PASS":
        steps["P1"] = receipt["pins"] = host.pins()
        if instrument_fix and steps["P1"].get("verdict") == "PASS" and any(
                p.get("pins", {}).get("functions_sha256") != steps["P1"].get("functions_sha256") for p in prior):  # Spec: a fix never changes the rule code on this rc.
            refusal = "the instrument fix changed rule-function code; that applies from the next rc"
    for name in ("P2", "plan", *PREFLIGHTS):
        if refusal or any(s.get("verdict") != "PASS" for s in steps.values()):
            break
        if name == "P2":
            steps["P2"] = receipt["snapshot"] = host.snapshot()
        elif name == "plan":
            plan = host.plan(host.clone("plan"))
            receipt["plan"] = {"sha256": planner.plan_sha256(plan), "declaration": planner.declaration(plan)}
            steps["plan"] = {"verdict": "PASS"}
        else:
            steps[name] = host.preflight(name, plan)
    stopped = refusal or next((k for k, s in steps.items() if s.get("verdict") != "PASS"), None)
    if stopped or len(steps) != 4 + len(PREFLIGHTS):
        return {**receipt, "stopped_at": stopped, "wall_s": {"total": round(time.monotonic() - started, 3)}}
    rules, base = host.rules(), host.run_dir / "plan" / "home"
    arms = {name: run_arm(host, name, host.clone(name), plan, {"RC3/store.db": base / "store.db"})
            for name in ("rc", "ga")}
    ga_rerun = None  # Bar 5: a failed check in the GA arm re-runs that arm once (release manager, 2026-11-07).
    if bars.failed_checks(arms["ga"]["checks"]):
        ga_rerun = run_arm(host, "ga", host.clone("ga-rerun"), plan, {"RC3/store.db": base / "store.db"})
    snap = host.rows(base)
    tops = {k: max((r[c] for r in snap[k]), default=0) for k, c in (("store", "store_id"), ("host", "id"))}
    new = {a: [r for r in arms[a]["rows"]["store"] if r["store_id"] > tops["store"]] for a in arms}
    binds, hosts = arms["rc"]["binds"], [r for r in arms["rc"]["rows"]["host"] if r["id"] > tops["host"]]
    host.configure_bar1(rc_rows=new["rc"], host_rows=hosts, snapshot_store=base / "store.db",  # the plan clone's
                        window_start=max((b["t0_at"] for b in binds), default=None),
                        bind_receipts=[b["receipt"] for b in binds if isinstance(b["receipt"], dict)])
    results = {
        "bar1": bars.bar1(new["rc"], hosts, snap["store"] + snap["host"], new["ga"], binds,
                          {b["bind"]: b["bar2"]["verdict"] for b in binds}, rules["identity"], rules["credits"],
                          missing_receipts=[b["bind"] for b in binds if not isinstance(b["receipt"], dict)]),
        "bar2": bars.bar2([b["bar2"] for b in binds]),
        "bar3": bars.bar3(snap["store"], arms["rc"]["rows"]["store"], rules["identity"]),
        "bar4": bars.bar4(arms["rc"]["logs"], arms["ga"]["logs"], arms["rc"]["roots"], arms["ga"]["roots"]),
        "bar5": bars.bar5(arms["rc"]["checks"], arms["ga"]["checks"], ga_rerun and ga_rerun["checks"],
                          bool(ga_rerun) and bars.workload_items(ga_rerun["workload"])["turns_completed"]),
        "bar6": bars.bar6({a: arms[a]["workload"] for a in arms})}
    receipt.update(
        arms={a: {**{k: arms[a][k] for k in ("workload", "egress", "provider", "doctor_not_exercised", "cron_paused")},
                  "checks": bars.check_counts(arms[a]["checks"])} for a in arms},
        ga_rerun=ga_rerun and {**{k: ga_rerun[k] for k in ("workload", "egress", "provider", "wall_s",
                                                         "doctor_not_exercised", "cron_paused")},
                               "checks": bars.check_counts(ga_rerun["checks"])},
        binds=[{k: v for k, v in b.items() if k not in ("dup_store_ids", "receipt")} for b in binds], bars=results,
        verdict=bars.verdict(results, prior_verdicts),
        wall_s={**{a: arms[a]["wall_s"] for a in arms}, "total": round(time.monotonic() - started, 3)})
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="run declaration (JSON)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--instrument-dir", required=True)
    parser.add_argument("--rc-id", required=True, help="the rc under test (tag or commit)")
    parser.add_argument("--prior-receipt", action="append", default=[], help="an earlier receipt for this rc")
    parser.add_argument("--instrument-fix", help="the reviewed instrument fix, after an instrument defect")
    args = parser.parse_args(argv)
    config = {**json.loads(Path(args.config).read_text(encoding="utf-8")), "instrument_dir": args.instrument_dir}
    prior = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.prior_receipt]
    receipt = run(LocalHost(config, args.run_dir), rc_id=args.rc_id, prior=prior, instrument_fix=args.instrument_fix)
    with (Path(args.run_dir) / "d5-clone-receipt.json").open("x", encoding="utf-8") as out:
        json.dump(receipt, out, sort_keys=True, indent=2)
    print(json.dumps({"verdict": receipt.get("verdict"), "stopped_at": receipt.get("stopped_at")}))
    return 0 if "verdict" in receipt else 2


if __name__ == "__main__":
    sys.exit(main())
