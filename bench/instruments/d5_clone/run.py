"""D5 clone-run CLI (bench/specs/RELEASE-READINESS-V1.md, "D5 clone run"): orchestration only.

Order: free space, P1 pins, P2 snapshot, the plan (on a clone), P3-P7 (any failure stops the run before either arm),
the rc arm then the GA arm on APFS clones of the verified snapshot, T0/H8 captures around each rc-arm cold bind with
that bind's replay right after its H8 (then its captures are removed), the bars, then the receipt. Every external
effect is a LocalHost method, so a test can run the whole flow with a fake host and a fake edition. The runner never
opens the pinned snapshot itself, only its clones; P2 hashes its files and checks SQLite on scratch clones.

The rules module (``rules_module`` in the run config, a pinned file in the instrument directory, imported only after
P1) provides ``identity(row)``, ``credits`` (the two host-rewrite credits, each ``(row, host_row) -> bool``) and
``covered(session_id)`` for planning. The edition steps are argv templates in the config (placeholders ``{python}``,
``{t0}``, ``{h8}``, ``{out}``, ``{instrument_dir}``, ``{rc_tree}``, ``{ga_tree}`` and ``{session_h12}``):
``constants_argv`` must leave the generator's report, with ``range_new_rows``, at ``{out}/constants.json``, and
``replay_argv`` writes the edition's receipt, named by ``replay_receipt`` (default ``receipt.json``), in ``{out}``.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
from contextlib import closing
import os
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
_ISO_Z = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(:[0-9]{2}(\.[0-9]+)?)?Z")  # The settings time, UTC.
NEED_FREE = 40 * 10**9
PREFLIGHTS = ("P3", "P4", "P5", "P6", "P7")
DOCTOR_STATUS = {"pass": "ok", "ok": "ok", "warn": "warn", "fail": "fail"}  # Any other status stays, and fails bar 5.
# LCM-X's externalized-payload reference (externalize.py _EXTERNALIZED_REF_RE).
EXTERNALIZED_REF = re.compile(r"\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]")
CRON_MODEL = "d5-cron"
# The model names the arm config gives the loopback provider (gateway.build_config); a receipt keeps only these raw.
OWN_MODELS = frozenset(("d5-agent", "d5-aux", "d5-cron", "d5-summary"))
CRON_PAUSE_REASON = "d5 clone run: paused before the run"
COLD_INGEST_TIMEOUT = 120
# How the probe can exercise each lcm_doctor check on a stopped arm (both trees' tools.py lcm_doctor). Store-wide
# checks read the store, the payload files or the arm's config once; per-session checks read the engine's current
# session, so they run once per plan session, bound by on_session_start. The rest never see the run: they are left
# out of bar 5 and reported (release manager, 2026-11-07), and so is any check not named here.
DOCTOR_STORE_WIDE = frozenset((
    "database_integrity", "schema_core_tables", "messages_fts_integrity", "nodes_fts_integrity", "sqlite_storage",
    "payload_storage", "source_lineage_hygiene", "lifecycle_fragmentation", "sensitive_pattern_handling",
    "config_validation", "embedding_provider_health"))
DOCTOR_PER_SESSION = ("fts_index_sync", "orphaned_dag_nodes", "summary_quality")
# The nodes the doctor's summary_quality check flags (tools._summary_quality_stats): source/summary >= 400, or a
# source of 100k tokens or more summarized in under 500.
FLAGGED_NODES = ("SELECT count(*) FROM summary_nodes WHERE session_id = ? AND ((source_token_count >= 100000"
                 " AND token_count < 500) OR (token_count > 0 AND CAST(source_token_count AS REAL) / token_count >= 400))")


def flagged_summary_nodes(store, session_ids):
    """{session id: flagged summary nodes} in *store*, read-only (#2471); None for a store that cannot be read."""
    try:
        with closing(capture._readonly(store)) as db:
            return {sid: db.execute(FLAGGED_NODES, (sid,)).fetchone()[0] for sid in session_ids}
    except (sqlite3.DatabaseError, capture.CaptureError, OSError):
        return None
DOCTOR_PRE_OPEN = ("messages_fts_integrity@pre_open", "nodes_fts_integrity@pre_open")  # Before the engine repairs FTS.
DOCTOR_NOT_EXERCISED = {
    "identity_migration": "no plugin registration step in the probe",
    "inactive_process": "reads live processes; the arm's gateway has stopped",
    "ingest_health": "the probe process's own counter", "ignore_pattern_drops": "the probe process's own counter",
    "context_pressure": "the probe process's own counter", "compaction_progress": "the probe process's own counter",
    "token_counter": "the probe process's tokenizer state"}


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rows(db, sql, args=()):
    """A read-only query's rows; a store, table or column that is not there reads as no rows. A store the read-only
    open refuses (CaptureError) raises: read as no rows, it would zero a mark and count history as the run's."""
    try:
        with closing(capture._readonly(db)) as con:
            return con.execute(sql, args).fetchall()
    except (OSError, sqlite3.Error):
        return []


def _json(value):
    """A JSON object, else None."""
    try:
        value = json.loads(value)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


PREFLIGHT_SCHEMA = "d5-clone-preflight/1"
TREE_BOUND_PREFLIGHTS = ("P4",)  # The spec's preflight that both trees store one externalized output identically.
# What a receipt's config_sha256 leaves out of the declaration, each with its reason. Everything else is bound, so a
# field added later is bound without anyone remembering it (release manager, 2026-11-10: a changed setting is an
# instrument change, and the preflights run again). Only what differs by stage by design, or is bound elsewhere, is
# left out: the preflights stage's P3, P6 and P7 receipts must count in the dry run and the run.
RECEIPT_UNBOUND = {  # Whole declaration fields.
    "snapshot_dir": "the snapshot pin is bound by its own sha256",
    "snapshot_pin": "the snapshot pin is bound by its own sha256",
    "preflights": "the receipts' own paths, which differ by stage",
    "replay_receipt": "a file name that differs by stage; a wrong one makes the replay, and so P6, fail closed",
    "settings_as_of": "a disclosed time, never judged",
    "instrument_dir": "a path; P1 pins its files on every run",
}
RECEIPT_TREE_UNBOUND = {  # Fields inside "rc" and "ga"; the rest of each tree (host_bin, host_src, ...) is bound.
    "commit": "the trees field binds it (R5-5); P3, P6 and P7 run the previous GA in both slots",
    "lcm_tree": "the stage's own extract of that commit, a path",
}
RECEIPT_ARGV_MASKED = {  # The value after these flags in constants_argv and replay_argv; the rest of each argv is bound.
    "--arm": "the stage's own arm file, a path (its settings and context length come from bound fields)",
    "--label": "the stage's run label, a name",
    "--tree-candidate": "masked only when it is the declared rc commit, which the trees field binds",
}


def _masked_argv(argv, rc_commit):
    out = list(argv)
    for i in range(len(out) - 1):
        if out[i] in RECEIPT_ARGV_MASKED and (out[i] != "--tree-candidate" or out[i + 1] == rc_commit):
            out[i + 1] = "<masked>"
    return out


PACKAGE = Path(__file__).resolve().parent


def outside_package(run_dir):
    """*run_dir*, refused when it lies in this package: files written there would change the instrument's own hash
    after a receipt recorded it, and the receipt would then never match (ruling R5-5)."""
    if Path(run_dir).resolve().is_relative_to(PACKAGE):
        raise ValueError("the run dir must be outside the instrument package")
    return run_dir


def instrument_identity(root=None):
    """This instrument's identity (ruling R5-5): the sha256 over every file of this package by relative name, which a
    preflight receipt must match, and the repo commit it runs from ("" outside a checkout), recorded for readers."""
    root, h = Path(root).resolve() if root else PACKAGE, hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"):
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {"instrument_sha256": h.hexdigest(), "instrument_commit": commit}


def config_identity(config):
    """The sha256 of the declaration's canonical JSON with the RECEIPT_* parts left out or masked: what a preflight
    receipt binds as config_sha256 (settings, env, context lengths, timeouts, nonce, the host build, both argv lists)."""
    rc_commit = (config.get("rc") or {}).get("commit") if isinstance(config.get("rc"), dict) else None
    bound = {}
    for key, value in config.items():
        if key in RECEIPT_UNBOUND:
            continue
        if key in ("rc", "ga") and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in RECEIPT_TREE_UNBOUND}
        elif key in ("constants_argv", "replay_argv") and isinstance(value, list):
            value = _masked_argv(value, rc_commit)
        bound[key] = value
    return hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def tree_id(config, tree):
    """A tree's identity as the receipts record it: its declared 40-hex commit (the builder extracts the tree from that
    commit), else None. A path is no identity: another tree can take the same path."""
    commit = config[tree].get("commit")
    return commit if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) else None


def _store_path(home):
    return next((Path(home) / n for n in ("lcm.db", "store.db") if (Path(home) / n).is_file()), Path(home) / "lcm.db")


def _keyed(counts):
    """A {session id: count} map keyed by the ids' sha256 12, as the receipt names sessions; None stays None."""
    return None if counts is None else {_sha(sid)[:12]: n for sid, n in counts.items()}


def _sha(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


def _utc_time(value):
    """*value* when it is a real UTC time written as ISO 8601 with a Z; else None."""
    if not (isinstance(value, str) and _ISO_Z.fullmatch(value)):
        return None
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    return value


def _private_models(counts):
    """{model name: n} for a receipt: the provider's own names (OWN_MODELS) stay, any other name (one the arm's settings or
    stored routes asked for) becomes its sha256 12, so no value from the snapshot is written (#2471, receipts carry
    counts, ids and hashes only)."""
    out = Counter()
    for name, n in (counts or {}).items():
        out[name if name in OWN_MODELS else f"sha256-12:{_sha(name)[:12]}"] += n
    return dict(out)


def _private_arm(provider, routing=None):
    """The receipt's provider stats and routing record, with the snapshot's provider and model names hashed."""
    provider = {**provider, **{k: _private_models(provider.get(k)) for k in ("calls_by_model", "unknown_models")
                               if k in provider}}
    if routing is None:
        return provider, None
    names = routing.get("stored_route_providers") or []
    return provider, {**{k: v for k, v in routing.items() if k != "stored_route_providers"},
                      "stored_route_providers": len(names),
                      "stored_route_provider_sha256_12": sorted(_sha(n)[:12] for n in names)}


def _owned_key(value):
    """The product's ownership reading of a stored conversation_id (db_bootstrap): NULL is '', edges are stripped."""
    return str(value or "").strip()


def rekey_copy_conversations(lcm_db, conversations, *, run_dir):
    """Align only the arm copy's conversation keys with its edited routing, preserving all sessions and rows."""
    path = Path(lcm_db).resolve()
    if not path.is_relative_to(Path(run_dir).resolve()):
        raise ValueError("conversation edits require an ARM COPY inside run_dir")
    changes = []
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        tables = []
        for name, in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            quoted = '"' + name.replace('"', '""') + '"'
            if any(row[1] == "conversation_id" for row in db.execute(f"PRAGMA table_info({quoted})")):
                if name not in ("messages", "lcm_lifecycle_state"):
                    raise ValueError("unexpected conversation_id table: sha256-12:" + _sha(name)[:12])
                tables.append(name)
        resolved, groups = [], {}
        for conversation in conversations:
            keys = db.execute("SELECT DISTINCT conversation_id FROM messages WHERE session_id=?",
                              (conversation["session_id"],)).fetchall()
            if len(keys) != 1:
                raise ValueError("copied session must carry exactly one conversation_id")
            old, new = keys[0][0], gateway.session_key(conversation["chat_id"])
            if not _owned_key(old):  # A legacy blank key is owned by every conversation: never a store-wide move.
                raise ValueError("copied session has a legacy blank conversation_id: " + conversation["name"])
            resolved.append((conversation["name"], old, new))
            group = groups.setdefault(old, {"new": new, "names": []})
            group["names"].append(conversation["name"])
            if group["new"] != new:
                raise ValueError("shared conversation has different destinations: " + ", ".join(group["names"]))
        # Resolve and validate the whole plan against the original state before the first rename.
        owned = {}  # Stored values by the key that owns them, as the product reads them (legacy variants included).
        for table in tables:
            for value, in db.execute(f"SELECT DISTINCT conversation_id FROM {table}"):
                owned.setdefault(_owned_key(value), set()).add(value)
        sources = {}
        for old, group in groups.items():
            sources.setdefault(group["new"], []).append(old)
        for olds in sources.values():
            if len(olds) > 1:
                names = [name for old in olds for name in groups[old]["names"]]
                raise ValueError("separate conversations share one destination: " + ", ".join(names))
        for old, group in groups.items():
            new = group["new"]
            if old == new:
                continue
            if owned.get(_owned_key(old), set()) - {old}:  # A variant the exact rename would leave behind.
                raise ValueError("copied conversation key has legacy variants: " + ", ".join(group["names"]))
            if owned.get(_owned_key(new)):
                raise ValueError("copied conversation destination already has rows")
        for old, group in groups.items():
            new = group["new"]
            if old == new:
                continue
            counts = {}
            for table in tables:
                before = db.execute(f"SELECT count(*) FROM {table} WHERE conversation_id=?", (old,)).fetchone()[0]
                db.execute(f"UPDATE {table} SET conversation_id=? WHERE conversation_id=?", (new, old))
                after = db.execute(f"SELECT count(*) FROM {table} WHERE conversation_id=?", (new,)).fetchone()[0]
                left = db.execute(f"SELECT count(*) FROM {table} WHERE conversation_id=?", (old,)).fetchone()[0]
                counts[table] = {"before": before, "after": after, "old_remaining": left}
            group["tables"] = counts
        for name, old, new in resolved:
            if old == new:
                continue
            entry = {"name": name, "old_h12": _sha(old)[:12], "new_h12": _sha(new)[:12], "tables": groups[old]["tables"]}
            if len(groups[old]["names"]) > 1:
                entry["shared_k1_with"] = [other for other in groups[old]["names"] if other != name]
            changes.append(entry)
    return changes


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
    # (The stored proof has no "published" field: compaction._compress_commit_proof_payload writes "native" only.)
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


def _clone(src, dest):
    """A copy-on-write copy of one capture file or directory (APFS clone on macOS)."""
    if Path(src).is_dir():
        capture.clone_tree(src, dest)
    elif sys.platform == "darwin":
        subprocess.run(["cp", "-c", str(src), str(dest)], check=True)
    else:
        shutil.copy2(src, dest)


def probe_checks(out, sessions):
    """(bar-5 rows, not exercised) from the probe's files in *out*; *sessions* are the plan sessions' sha256 12 keys.
    A kind the probe did not produce is left out, so bar 5 reads it as not shown. Store-wide doctor checks come from
    the unbound run, per-session ones from each session's run as ``<check>@<key>``; a check the probe cannot exercise
    is never a row, only {check, reason}."""
    found, skipped = [], []
    if out is None:
        return found, skipped
    rows = _doctor_rows(out / "doctor.json")
    pre_open = dict(c for c in _doctor_rows(out / "fts-pre-open.json") or () if c[0] in DOCTOR_PRE_OPEN)
    pre_rows = [{"kind": "doctor", "check": c, "status": pre_open[c]} for c in DOCTOR_PRE_OPEN if c in pre_open]
    if rows is None:  # The doctor did not report: only a failure the pre-open check saw counts (#2453).
        found += [r for r in pre_rows if r["status"] not in ("ok", "warn")]
    else:
        bound = (_json((out / "sessions.json").read_text(encoding="utf-8")) if (out / "sessions.json").is_file()
                 else None) or {}
        for check, status in rows:
            if check in DOCTOR_STORE_WIDE:
                found.append({"kind": "doctor", "check": check, "status": status})
            elif check not in DOCTOR_PER_SESSION:
                if check not in DOCTOR_NOT_EXERCISED and status != "ok":
                    found.append({"kind": "doctor", "check": check, "status": status})  # A new check's warn or fail counts.
                else:
                    skipped.append({"check": check,
                                    "reason": DOCTOR_NOT_EXERCISED.get(check, "not classified for the probe")})
        found += pre_rows
        unbound = [] if set(DOCTOR_PRE_OPEN) <= set(pre_open) else ["fts-pre-open"]  # Else only a repaired index was seen.
        for key in sessions:
            scoped = _doctor_rows(out / f"doctor-{key}.json") if bound.get(key) is True else None
            found += [{"kind": "doctor", "check": f"{c}@{key}", "status": st}  # Every reported row, a fail included,
                      for c, st in scoped or () if c in DOCTOR_PER_SESSION]  # even when the report is incomplete.
            reported = {c for c, _ in scoped or ()}
            skipped += [{"check": f"{c}@{key}", "reason": "no session binding in this tree" if scoped is None
                         else "not reported by the session run"} for c in DOCTOR_PER_SESSION if c not in reported]
            unbound += [] if set(DOCTOR_PER_SESSION) <= reported else [key]  # Not looked at in full: never a PASS.
        if unbound and all(r["status"] in ("ok", "warn") for r in found):
            found = []  # A plan session, or the store's FTS before the engine opened it, not looked at: never a PASS.
    schema = _json((out / "schema.json").read_text(encoding="utf-8")) if (out / "schema.json").is_file() else None
    if isinstance(schema, dict) and set(schema) == {"lcm.db", "state.db"}:
        found += [{"kind": "schema", "check": name, "status": "ok" if isinstance(s, dict) and s.get("error") is False
                   and s.get("missing_tables") == [] and type(s.get("version")) is int and s["version"] == s.get("expected")
                   else "fail"} for name, s in sorted(schema.items())]
    return found, sorted(skipped, key=lambda item: item["check"])


class LocalHost:
    """The real effects. ``config`` is the run declaration; ``run_dir`` holds every clone, capture and log."""

    def __init__(self, config, run_dir):
        self.c, self.run_dir, self._rules = config, Path(os.path.abspath(run_dir)), None  # Subprocesses run elsewhere.

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
                "rules_sha256": _sha_json(sorted({(e["name"], e["sha256"]) for f in manifest["files"] for e in (f, *f["loads"])}
                                                 | {(e["name"], e["sha256"]) for e in manifest.get("rule_files", [])}))}
        # A total freeze (release manager, 2026-11-07): every pinned file whole, so no same-rc fix touches the kit.

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
            root = str(Path(self.c["instrument_dir"]).resolve())  # Sibling kit modules, pinned by P1, import.
            sys.path.insert(0, root)
            try:
                spec.loader.exec_module(module)
            finally:
                sys.path.remove(root)
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
        flagged = flagged_summary_nodes(copy / "store.db", [c["session_id"] for c in chosen]) or {}
        return planner.build_plan([dict(c, flagged_summary_nodes=flagged.get(c["session_id"])) for c in chosen],
                                  nonce=self.c["nonce"])

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
        try:
            recorded = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"verdict": "MISSING", "mismatch": ["unreadable"]}
        # Ruling R5-5: a receipt counts only for this run's plan, snapshot pin, instrument and declared configuration
        # (config_sha256, release manager 2026-11-11); P4 also for this run's
        # two trees, and the others, which run the previous GA in both arms by design, for that tree in both slots.
        ga = tree_id(self.c, "ga")
        trees = {"rc": tree_id(self.c, "rc"), "ga": ga} if name in TREE_BOUND_PREFLIGHTS else {"rc": ga, "ga": ga}
        want = {"schema": PREFLIGHT_SCHEMA, "name": name, "plan_sha256": planner.plan_sha256(plan),
                "snapshot_pin_sha256": capture._sha256(self.c["snapshot_pin"]),
                "instrument_sha256": instrument_identity()["instrument_sha256"], "trees": trees,
                "config_sha256": config_identity(self.c)}
        mismatch = sorted(k for k, v in want.items() if not isinstance(recorded, dict) or recorded.get(k) != v
                          or (k == "trees" and None in v.values()))  # An undeclared tree never binds a receipt.
        if mismatch:  # Treated as missing: an earlier snapshot's, plan's, instrument's or tree's receipt.
            return {"verdict": "MISSING", "mismatch": mismatch, "sha256": capture._sha256(path)}
        return {"verdict": recorded.get("verdict"), "sha256": capture._sha256(path), "bound": sorted(want)}

    def arm(self, name, home, plan):
        tree = self.c[name]
        for old, new in (("store.db", "lcm.db"), ("host-0.db", "state.db")):
            for suffix in ("", "-wal", "-shm"):
                if (home / (old + suffix)).exists():
                    (home / (old + suffix)).rename(home / (new + suffix))
        server = prov.Provider(plan["provider"]).start()
        routing = gateway.edit_copy_routing(home / "state.db", plan["conversations"], sessions_dir=home / "sessions",
                                            run_dir=self.run_dir, provider_url=server.url)
        routing["conversation_rekeys"] = rekey_copy_conversations(home / "lcm.db", plan["conversations"],
                                                                 run_dir=self.run_dir)
        gateway.write_config(home, gateway.build_config(
            self.c["settings"], provider_url=server.url, home=home, aux=gateway.aux_tasks(tree["host_src"]),
            context_length=self.c["context_length"], stored_route_providers=routing["stored_route_providers"]))
        env, _ = gateway.build_env(self.c["lcm_env"], home=home, provider_url=server.url, host_bin=tree["host_bin"],
                                   port_file=home.parent / "d5-local.port", lcm_tree=tree["lcm_tree"])
        port = server.server.server_address[1]
        return {"name": name, "home": home, "provider": server, "port": port, "tree": tree, "marks": marks(home),
                "routing": routing,
                "session_ids": [c["session_id"] for c in plan["conversations"]],
                "gateway": gateway.Gateway(tree["host_bin"], home, env, home.parent, provider_port=port),
                "payloads": sum(p.is_file() for p in (home / "lcm-large-outputs").rglob("*")),
                "chats": {c["name"]: c["chat_id"] for c in plan["conversations"]},
                "sessions": {c["name"]: c["session_id"] for c in plan["conversations"]}}

    def start(self, ctx):
        ctx["gateway"].start(self.c.get("start_timeout", 300))

    def stop(self, ctx):
        ctx["gateway"].stop()

    def turn(self, ctx, step):
        with ctx["provider"].lock:
            ctx["provider"].step = step["id"]
        try:
            result = ctx["gateway"].turn(ctx["chats"][step["conversation"]], step["id"], step["user_text"])
            with ctx["provider"].lock:  # An agent that never called the model never ran the turn, whatever the
                ran = ctx["provider"].stats["calls_per_step"][step["id"]] > 0  # host reports (a queued turn).
            return result.get("completed") is True and result.get("outcome") == "success" and ran
        except OSError:
            return False
        finally:
            with ctx["provider"].lock:
                ctx["provider"].step = None

    def capture(self, ctx, kind, step, extras=None):
        out = self._dir(ctx["name"], "captures") / f"{step['id']}-{kind}"
        home = ctx["home"]
        if kind == "H8":  # The edition reads the bind's anchor and history lines from the window's host logs.
            extras = {**(extras or {}), **{f"native-{name}.log": home / "logs" / name
                                           for name in ("agent.log", "gateway.log") if (home / "logs" / name).is_file()}}
        derived = (lambda filled: self.t0_evidence(ctx, filled)) if kind == "T0" else None
        return out, capture.capture(kind, out, store_db=home / "lcm.db", host_db=home / "state.db",
                                    payload_dir=home / "lcm-large-outputs", extras=extras, home=home, derived=derived)

    def wait_cold_ingest(self, ctx, step, t0):
        """Wait for this turn's new user row to be durable before taking the bind's H8 capture."""
        started = time.monotonic()
        deadline = started + self.c.get("cold_ingest_timeout", COLD_INGEST_TIMEOUT)
        # Only cold steps wait today; trimming edge whitespace is defensive if a future step's text is rewritten.
        args = (ctx["sessions"][step["conversation"]], t0["store_max_store_id"] or 0, step["user_text"].strip(" \t\r\n"))
        while True:
            try:
                with closing(capture._readonly(ctx["home"] / "lcm.db", timeout=max(0, min(0.5, deadline - time.monotonic())))) as db:
                    found = db.execute("SELECT 1 FROM messages WHERE session_id=? AND store_id>?"
                                       " AND role='user' AND trim(content, ' ' || char(9, 13, 10))=? LIMIT 1", args).fetchone()
                if found:
                    return True, round(time.monotonic() - started, 1)
            except (OSError, sqlite3.Error, capture.CaptureError):
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, round(time.monotonic() - started, 1)
            time.sleep(min(0.5, remaining))

    def t0_evidence(self, ctx, out):
        """The two files the replay edition reads from T0 besides the stores (p_s6r_clone, "observed bind evidence"):
        ``filters.json``, the two non-secret session-pattern settings from the arm's own env, and ``doctor.log``, the
        arm tree's offline doctor text, run under the sandbox on a clone of the capture's stores outside the capture
        dir. A doctor that fails or times out leaves doctor.log out, so that bind's replay errors (bar 2 INCONCLUSIVE)."""
        env = ctx["gateway"].env
        filters = {name: [v.strip() for v in str(env.get(key) or "").split(",") if v.strip()]
                   for name, key in (("ignore_session_patterns", "LCM_IGNORE_SESSION_PATTERNS"),
                                     ("stateless_session_patterns", "LCM_STATELESS_SESSION_PATTERNS"))}
        files = {"filters.json": (json.dumps(filters) + "\n").encode()}
        work = out.parent / f"{out.name}-doctor"
        clone, text = work / "home", work / "doctor.log"
        try:
            clone.mkdir(parents=True)
            for src, name in (("store.db", "lcm.db"), ("host-0.db", "state.db"), ("lcm-large-outputs", "lcm-large-outputs")):
                _clone(out / src, clone / name)
            for name in ("user-home", "tmp"):
                (clone / name).mkdir()
            tree = ctx["tree"]
            argv = sandbox.wrap([str(Path(tree["host_bin"]) / "python"), "-P", "-B",
                                 str(Path(__file__).with_name("doctor_probe.py")), "--text",
                                 str(Path(tree["lcm_tree"]).resolve()), str(clone), str(text)],
                                ports=[], deny_paths=sandbox.live_homes())
            child = {**env, "HERMES_HOME": str(clone), "HOME": str(clone / "user-home"), "TMPDIR": str(clone / "tmp"),
                     **{k: str(clone / v) for k, v in gateway.FORCED_LCM_PATHS.items()}}
            done = subprocess.run(["/usr/bin/nice", "-n", "19", *argv], cwd=clone, env=child, capture_output=True,
                                  timeout=self.c.get("doctor_timeout", 1800))
            if done.returncode == 0 and text.is_file():
                files["doctor.log"] = text.read_bytes()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        finally:
            shutil.rmtree(work, ignore_errors=True)  # The clone only: the capture itself is never touched here.
        return files

    def replay(self, ctx, step, t0, h8):
        """The edition on both trees, under the sandbox: (receipt, error, range_new_rows)."""
        out = self._dir(ctx["name"], "replays", step["id"])
        python = self.c["interpreter"]
        values = {"python": os.path.abspath(python) if os.sep in python else python, "t0": t0, "h8": h8, "out": out,
                  "instrument_dir": os.path.abspath(self.c["instrument_dir"]),  # The argv runs with cwd=out.
                  "rc_tree": os.path.abspath(self.c["rc"]["lcm_tree"]), "ga_tree": os.path.abspath(self.c["ga"]["lcm_tree"]),
                  "session_h12": hashlib.sha256(ctx["sessions"][step["conversation"]].encode()).hexdigest()[:12]}
        receipt_name = self.c.get("replay_receipt", "receipt.json")  # The edition's receipt, by its declared name.
        if Path(receipt_name).name != receipt_name:
            raise ValueError("replay_receipt must be a bare file name")
        for key in ("constants_argv", "replay_argv"):
            argv = sandbox.wrap([a.format(**values) for a in self.c[key]], ports=[ctx["port"]], socket_roots=[out],
                                deny_paths=sandbox.live_homes())  # The live agent's home is never read.
            env = {"PATH": "/usr/bin:/bin", "HOME": str(out), "PYTHONDONTWRITEBYTECODE": "1"}
            code = subprocess.run(["/usr/bin/nice", "-n", "19", *argv], cwd=out, env=env, capture_output=True).returncode
            if code:  # An error, or (constants exit 1) a NOT SHOWN bind: no replay runs after it. Either way the
                return None, True, _range_new_rows(out)  # report's range_new_rows is kept when it was written.
        try:
            return (json.loads((out / receipt_name).read_text(encoding="utf-8")), False, _range_new_rows(out))
        except (OSError, ValueError):
            return None, True, _range_new_rows(out)

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
                             json.dumps(ctx["session_ids"])], ports=[], deny_paths=sandbox.live_homes())
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
            for kind, pragma in (("quick_check", "quick_check"), ("integrity", "integrity_check")):
                try:
                    with closing(capture._readonly(ctx["home"] / name)) as db:
                        ok = db.execute(f"PRAGMA {pragma}").fetchall() == [("ok",)]
                except (sqlite3.DatabaseError, capture.CaptureError):
                    ok = False  # A store too damaged, or in a state, to open read-only is a failed check, never a crash.
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


def _range_new_rows(out):
    try:
        return json.loads((Path(out) / "constants.json").read_text(encoding="utf-8")).get("range_new_rows")
    except (OSError, ValueError, AttributeError):
        return None


def run_arm(host, name, home, plan, extras):
    ctx = host.arm(name, home, plan)
    cron_paused = host.pause_cron(ctx)  # Before the first gateway start: no snapshot job fires in the run.
    flagged_start = _keyed(flagged_summary_nodes(_store_path(home), [c["session_id"] for c in plan["conversations"]]))
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
                    durable, waited = host.wait_cold_ingest(ctx, step, t0) if ok else (False, None)
                    if durable:
                        h8_dir, h8 = host.capture(ctx, "H8", step)
                        receipt, error, range_new = host.replay(ctx, step, t0_dir, h8_dir)
                    else:
                        h8 = {"store_max_store_id": t0["store_max_store_id"], "capture_hashes_sha256": None}
                        receipt, error, range_new = None, True, None
                    bar2 = bars.bar2_bind(receipt, range_new_rows=range_new, error=error)
                    if not durable:
                        timeout = host.c.get("cold_ingest_timeout", COLD_INGEST_TIMEOUT)
                        bar2["reason"] = (f"cold-turn ingest not durable within {timeout:g} s" if ok
                                          else "cold turn not completed")
                    binds.append({"bind": step["id"], "session_id": sid[step["conversation"]],
                                  "cold_ingest_wait_s": waited,
                                  "t0_max": t0["store_max_store_id"], "h8_max": h8["store_max_store_id"],
                                  "dup_store_ids": bars.dup_store_ids(receipt), "receipt": receipt,
                                  "t0_at": datetime.fromisoformat(t0["captured_at"]).timestamp(),
                                  "t0_capture_sha256": t0["capture_hashes_sha256"],
                                  "h8_capture_sha256": h8["capture_hashes_sha256"],
                                  "bar2": bar2})
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
                "cron_job_sha256": cron_job_sha256}
    try:
        workload.update(host.facts(ctx))
    except (sqlite3.DatabaseError, capture.CaptureError) as error:  # A refused read: bar 6's facts are unavailable
        workload["facts_unavailable"] = type(error).__name__  # (never zero), and the checks below still run (#2453).
    checks = host.checks(ctx)
    try:
        rows = host.rows(home)
    except (sqlite3.DatabaseError, capture.CaptureError, OSError, StopIteration):
        rows = None  # A store gone bad after its checks: they are kept (bar 5), and bars 1 and 3 cannot pass.
    return {"binds": binds, "workload": workload, "egress": {"samples": len(egress), "non_local": sum(egress)},
            "provider": stats, "logs": host.logs(ctx), "roots": host.roots(ctx), "checks": checks,
            "doctor_not_exercised": ctx.get("doctor_not_exercised", []), "cron_paused": cron_paused,
            "routing": ctx.get("routing", {}),
            "flagged_summary_nodes_start": flagged_start,  # Before the first turn (ruling R5-4).
            "flagged_summary_nodes": _keyed(flagged_summary_nodes(
                _store_path(home), [c["session_id"] for c in plan["conversations"]])),
            "rows": rows, "wall_s": round(time.monotonic() - started, 3)}


def inherited_flags(plan, arms, ga_rerun=None):
    """Bar 5's inherited sessions (ruling R5-4): each driven session whose snapshot count of flagged summary nodes is not
    0 (or unknown), keyed by its summary_quality check name, with its counts at the start and at each arm's end."""
    def count(arm, key, h12):
        return ((arm or {}).get(key) or {}).get(h12)

    out = {}
    for c in plan["conversations"]:
        if c.get("flagged_summary_nodes") == 0:
            continue
        h12 = _sha(c["session_id"])[:12]
        n = {"start": c.get("flagged_summary_nodes"),
             **{f"{a}{k}": count(arms.get(a), f"flagged_summary_nodes{key}", h12)
                for a in ("rc", "ga") for k, key in (("_start", "_start"), ("", ""))}}
        if ga_rerun:
            n.update(ga_rerun_start=count(ga_rerun, "flagged_summary_nodes_start", h12),
                     ga_rerun=count(ga_rerun, "flagged_summary_nodes", h12))
        out[f"summary_quality@{h12}"] = n
    return out


def run(host, *, rc_id, prior=(), instrument_fix=None):
    """The whole run; returns the receipt (counts, ids, sha256s and verdicts only)."""
    started = time.monotonic()
    prior = [p for p in prior if p.get("rc") == rc_id and "verdict" in p]  # A stopped run is not an attempt.
    prior_verdicts = [p["verdict"]["raw"] for p in prior]
    receipt = {"schema": SCHEMA, "rc": rc_id, "attempt": len(prior_verdicts) + 1, "instrument_fix": instrument_fix}
    receipt["settings"] = {  # A disclosed fact, never judged (release manager, 2026-11-10); only a UTC time is written.
        "as_of": _utc_time(getattr(host, "c", {}).get("settings_as_of")),
        "note": "both arms run the declared reference-settings snapshot taken at as_of, assumed unchanged since"}
    refusal = bars.may_run(prior_verdicts, instrument_fix)
    receipt["preflights"] = steps = {"free_space": {"verdict": "PASS" if host.free_bytes() >= NEED_FREE else "FAIL"}}
    if refusal is None and steps["free_space"]["verdict"] == "PASS":
        steps["P1"] = receipt["pins"] = host.pins()
        if steps["P1"].get("verdict") == "PASS" and any(
                p.get("pins", {}).get("rules_sha256") != steps["P1"].get("rules_sha256") for p in prior):  # Spec: the rule code never changes on this rc.
            refusal = "the rule code changed since an earlier attempt on this rc; that applies from the next rc"
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
    unreadable = [a for a in arms if arms[a]["rows"] is None]
    for a in unreadable:
        arms[a]["rows"] = {"store": [], "host": []}  # Placeholders only: unreadable rc rows make bars 1 and 3 INCONCLUSIVE below; unreadable GA rows alone keep a bar 1 FAIL.
    tops = {k: max((r[c] for r in snap[k]), default=0) for k, c in (("store", "store_id"), ("host", "id"))}
    new = {a: [r for r in arms[a]["rows"]["store"] if r["store_id"] > tops["store"]] for a in arms}
    binds, hosts = arms["rc"]["binds"], [r for r in arms["rc"]["rows"]["host"] if r["id"] > tops["host"]]
    host.configure_bar1(rc_rows=new["rc"], host_rows=hosts, snapshot_store=base / "store.db",  # the plan clone's
                        window_start=max((b["t0_at"] for b in binds), default=None),
                        bind_receipts=[b["receipt"] for b in binds if isinstance(b["receipt"], dict)])
    results = {
        # GA rows unreadable: GA's class-3 allowance is unknown, so it is set to the rc's own rows (never exceeded);
        # a lost or settled unassigned row still fails, and the class-3 judgment is left INCONCLUSIVE below.
        "bar1": bars.bar1(new["rc"], hosts, snap["store"] + snap["host"],
                          new["rc"] if "ga" in unreadable else new["ga"], binds,
                          {b["bind"]: b["bar2"]["verdict"] for b in binds}, rules["identity"], rules["credits"],
                          missing_receipts=[b["bind"] for b in binds if not isinstance(b["receipt"], dict)]),
        "bar2": bars.bar2([b["bar2"] for b in binds]),
        "bar3": bars.bar3(snap["store"], arms["rc"]["rows"]["store"], rules["identity"]),
        "bar4": {**bars.bar4(arms["rc"]["logs"], arms["ga"]["logs"], arms["rc"]["roots"], arms["ga"]["roots"]),
                 "raw_counts_note": "the host writes each log line to two files (agent.log and gateway.log), so the raw"
                                    " counts are 2x on both arms; the verdict compares the arms and is unaffected"},
        "bar5": {**bars.bar5(arms["rc"]["checks"], arms["ga"]["checks"], ga_rerun and ga_rerun["checks"],
                             bool(ga_rerun) and bars.workload_items(ga_rerun["workload"])["turns_completed"],
                             inherited=inherited_flags(plan, arms, ga_rerun),
                             ga_turns_complete=bars.workload_items(arms["ga"]["workload"])["turns_completed"]),
                 "flagged_summary_nodes": {  # Per driven session: at the start, and in each arm at its end (#2471).
                     "start": _keyed({c["session_id"]: c.get("flagged_summary_nodes") for c in plan["conversations"]}),
                     **{a: arms[a]["flagged_summary_nodes"] for a in arms}}},
        "bar6": bars.bar6({a: arms[a]["workload"] for a in arms})}
    if "rc" in unreadable:  # Bars 1 and 3 judge the rc's rows: without them neither can pass or fail.
        for bar in ("bar1", "bar3"):
            results[bar] = {"verdict": bars.INCONCLUSIVE, "rows_unreadable": unreadable}
    elif unreadable:  # Only GA's rows: bar 3 never reads them, and an rc failure bar 1 finds without GA stands.
        verdict = bars.FAIL if results["bar1"]["verdict"] == bars.FAIL else bars.INCONCLUSIVE
        results["bar1"] = {**results["bar1"], "verdict": verdict, "rows_unreadable": unreadable}
    results["bar1"]["known_risk"] = (  # Release manager, 2026-11-07: recorded, not changed (the K6 rule is frozen).
        "class 3's GA allowance matches rows by content identity across the arms; a tool output that names its arm's"
        " home path differs between the arms there, so such a leftover can show as over GA (a false FAIL, never a"
        " false PASS); the fix would be path parity in the runner")
    private = {a: _private_arm(arms[a]["provider"], arms[a]["routing"]) for a in arms}
    receipt.update(
        arms={a: {**{k: arms[a][k] for k in ("workload", "egress", "doctor_not_exercised", "cron_paused")},
                  "provider": private[a][0], "routing": private[a][1],
                  "checks": bars.check_counts(arms[a]["checks"])} for a in arms},
        ga_rerun=ga_rerun and {**{k: ga_rerun[k] for k in ("workload", "egress", "wall_s",
                                                         "doctor_not_exercised", "cron_paused")},
                               "provider": _private_arm(ga_rerun["provider"])[0],
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
    receipt = run(LocalHost(config, outside_package(args.run_dir)), rc_id=args.rc_id, prior=prior, instrument_fix=args.instrument_fix)
    with (Path(args.run_dir) / "d5-clone-receipt.json").open("x", encoding="utf-8") as out:
        json.dump(receipt, out, sort_keys=True, indent=2)
    print(json.dumps({"verdict": receipt.get("verdict"), "stopped_at": receipt.get("stopped_at")}))
    return 0 if "verdict" in receipt else 2


if __name__ == "__main__":
    sys.exit(main())
