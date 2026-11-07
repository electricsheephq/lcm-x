"""Doctor and schema facts for a clone of one stopped arm home, run by that arm's host interpreter.

argv: <lcm_tree> <home> <out_dir>. It writes ``schema.json`` (both stores' schema version and required tables, read
before anything opens the store) and then ``doctor.json`` (the ``lcm_doctor`` tool's result, through the engine, as
the release gauntlet runs it). Opening the engine may write to the store it opens, so the runner points this at a
clone and never at the arm home.
"""

from contextlib import closing
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


def main(tree, home, out):
    tree, home, out = Path(tree), Path(home), Path(out)
    spec = importlib.util.spec_from_file_location("hermes_lcm_x", tree / "__init__.py",
                                                  submodule_search_locations=[str(tree)])
    sys.modules["hermes_lcm_x"] = importlib.util.module_from_spec(spec)  # The package's modules, not its register().
    try:
        host = importlib.import_module("hermes_state_common")
    except ImportError:  # An older host keeps its schema in hermes_state.
        host = importlib.import_module("hermes_state")
    boot = importlib.import_module("hermes_lcm_x.db_bootstrap")
    (out / "schema.json").write_text(json.dumps(schema(home, boot, host.SCHEMA_SQL, host.SCHEMA_VERSION)))
    config, engine = (importlib.import_module(f"hermes_lcm_x.{name}") for name in ("config", "engine"))
    lcm = engine.LCMEngine(config=config.LCMConfig.from_env(), hermes_home=str(home))
    (out / "doctor.json").write_text(lcm.handle_tool_call("lcm_doctor", {}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
    sys.stdout.flush()
    os._exit(0)  # The engine's background threads never hold the run.
