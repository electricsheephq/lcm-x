"""Doctor and schema facts for a clone of one stopped arm home, run by that arm's host interpreter.

argv: <lcm_tree> <home> <out_dir> <session ids, JSON>. It writes ``schema.json`` (both stores' schema version and
required tables, read before anything opens the store), then ``fts-pre-open.json`` (the doctor's two FTS
integrity checks, before the engine opens the store and rebuilds a damaged index), then ``doctor.json`` (the ``lcm_doctor`` tool's result,
through the engine, as the release gauntlet runs it), then, per session, ``doctor-<sha256 12>.json`` with the engine
bound to that session by its public ``on_session_start``, and ``sessions.json`` ({sha256 12: bound}). Opening the engine may write to the store it opens, so the runner points this at a
clone and never at the arm home.

``--text <lcm_tree> <home> <out_file>`` instead writes the tree's own offline doctor text (``_doctor_text``), the
``doctor.log`` a T0 capture carries for the replay edition.
"""

from contextlib import closing
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import sys


def _connect(db):
    return sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)


def captured_host_version(home):
    """Only the snapshot's recorded schema number; never open a database or fall back to the pinned host."""
    try:
        record = json.loads((Path(home) / "capture.json").read_text(encoding="utf-8"))
        for section, key in (("p2", "host_schema_version"), ("host-0.db", "schema_version")):
            part = record.get(section)
            value = part.get(key) if isinstance(part, dict) else None
            if type(value) is int:
                return value
    except (OSError, ValueError, AttributeError):
        pass
    return None


def schema(home, boot, host_sql, host_version):
    """{db name: {version, expected, missing_tables, error}} for the LCM-X store and the host store."""
    out = {}
    with closing(_connect(home / "lcm.db")) as conn:
        health = boot.inspect_lcm_schema_health(conn)
        out["lcm.db"] = {"version": boot.read_existing_schema_version(conn), "expected": boot.SCHEMA_VERSION,
                         "missing_tables": list(health.get("missing_tables") or []), "error": bool(health.get("error"))}
    with closing(_connect(home / "state.db")) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rows = conn.execute("SELECT version FROM schema_version").fetchall() if "schema_version" in tables else []
        required = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", host_sql)
        out["state.db"] = {"version": rows[0][0] if len(rows) == 1 else None, "expected": host_version,
                           "missing_tables": [t for t in required if t not in tables], "error": not required}
    return out


def fts_pre_open(home, boot, store, dag):
    """The doctor's two FTS integrity checks, run on the clone before the engine opens it: opening the engine
    rebuilds a damaged FTS index, so the doctor's own run of these checks can no longer see the damage."""
    checks = []
    with closing(sqlite3.connect(home / "lcm.db")) as conn:  # The clone: the check runs inside a rolled-back savepoint.
        for name, spec in (("messages_fts_integrity", store.build_message_fts_spec),
                           ("nodes_fts_integrity", dag.build_nodes_fts_spec)):
            try:
                status = boot.check_external_content_fts_integrity(conn, spec())["status"]
            except Exception:
                status = "fail"
            checks.append({"check": f"{name}@pre_open", "status": "warn" if status == "unchecked" else status,
                           "detail": "before the engine opened the store"})
    return {"overall": "healthy", "checks": checks}


def _package(tree):
    spec = importlib.util.spec_from_file_location("hermes_lcm_x", tree / "__init__.py",
                                                  submodule_search_locations=[str(tree)])
    sys.modules["hermes_lcm_x"] = importlib.util.module_from_spec(spec)  # The package's modules, not its register().


def text(tree, home, out_file):
    """The tree's offline doctor text, as the earlier T0 capture wrote doctor.log."""
    tree, home = Path(tree), Path(home)
    _package(tree)
    config, engine = (importlib.import_module(f"hermes_lcm_x.{name}") for name in ("config", "engine"))
    try:
        doctor = importlib.import_module("hermes_lcm_x.command_doctor")
    except ImportError:  # An older tree keeps it in command.
        doctor = importlib.import_module("hermes_lcm_x.command")
    lcm = engine.LCMEngine(config=config.LCMConfig.from_env(), hermes_home=str(home))
    Path(out_file).write_text(doctor._doctor_text(lcm) + "\n", encoding="utf-8")


def main(tree, home, out, sessions="[]"):
    tree, home, out = Path(tree), Path(home), Path(out)
    _package(tree)
    try:
        host = importlib.import_module("hermes_state_common")
    except ImportError:  # An older host keeps its schema in hermes_state.
        host = importlib.import_module("hermes_state")
    boot = importlib.import_module("hermes_lcm_x.db_bootstrap")
    (out / "schema.json").write_text(json.dumps(schema(home, boot, host.SCHEMA_SQL, captured_host_version(home))))
    store, dag = (importlib.import_module(f"hermes_lcm_x.{name}") for name in ("store", "dag"))
    (out / "fts-pre-open.json").write_text(json.dumps(fts_pre_open(home, boot, store, dag)))
    config, engine = (importlib.import_module(f"hermes_lcm_x.{name}") for name in ("config", "engine"))
    lcm = engine.LCMEngine(config=config.LCMConfig.from_env(), hermes_home=str(home))
    (out / "doctor.json").write_text(lcm.handle_tool_call("lcm_doctor", {}))
    bound = {}
    for session_id in json.loads(sessions):  # The doctor scopes some checks to the engine's current session.
        key = hashlib.sha256(session_id.encode()).hexdigest()[:12]
        try:
            lcm.on_session_start(session_id)
            bound[key] = lcm.current_session_id == session_id
        except Exception:
            bound[key] = False
        if bound[key]:
            (out / f"doctor-{key}.json").write_text(lcm.handle_tool_call("lcm_doctor", {}))
    (out / "sessions.json").write_text(json.dumps(bound))


if __name__ == "__main__":
    if sys.argv[1] == "--text":
        text(*sys.argv[2:5])
    else:
        main(*sys.argv[1:5])
    sys.stdout.flush()
    os._exit(0)  # The engine's background threads never hold the run.
