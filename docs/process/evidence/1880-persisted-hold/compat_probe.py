"""Synthetic candidate-write / v0.28.1-read probe; never loads a real profile."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types


def refuse_profile(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        value = os.fsdecode(args[0])
        forbidden = str(Path.home() / ".hermes")
        if value == forbidden or value.startswith(forbidden + "/"):
            raise RuntimeError("forbidden live-profile access")


def schema(conn):
    rows = conn.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["write", "read"])
    parser.add_argument("source", type=Path)
    parser.add_argument("scratch", type=Path)
    args = parser.parse_args()
    sys.addaudithook(refuse_profile)
    os.environ["HERMES_HOME"] = str(args.scratch / "synthetic-home")
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    package = types.ModuleType("hermes_lcm")
    package.__path__ = [str(args.source)]
    sys.modules["hermes_lcm"] = package
    config_mod = importlib.import_module("hermes_lcm.config")
    boot = importlib.import_module("hermes_lcm.db_bootstrap")
    boot.dispatch_background_stem_backfill = lambda *a, **kw: None
    store_mod = importlib.import_module("hermes_lcm.store")
    db = args.scratch / "compat.db"
    config = config_mod.LCMConfig(database_path=str(db), embeddings_enabled=False,
                                  temporal_rollups_enabled=False)
    store = store_mod.MessageStore(db, ingest_protection_config=config,
                                   hermes_home=os.environ["HERMES_HOME"])
    try:
        key = "compaction_no_progress:v1:synthetic-session"
        if args.phase == "write":
            sid = store.append("synthetic-session", {"role": "user", "content": "Synthetic preserved row.",
                                                      "timestamp": 100.0})
            before = schema(store.connection)
            record = dict(version=1, fingerprint="a" * 64, reason="no_progress",
                          recorded_at=1792700000.0, expires_at=1792700600.0)
            store.write_metadata_json([key], json.dumps(record, sort_keys=True))
            assert schema(store.connection) == before
            rows = store.connection.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
            result = dict(schema=before, rows=rows, record=record, store_id=sid)
            (args.scratch / "compat-expected.json").write_text(json.dumps(result))
            print("CANDIDATE_WRITE_OK metadata-only; schema unchanged; synthetic row preserved")
        else:
            expected = json.loads((args.scratch / "compat-expected.json").read_text())
            rows = [list(row) for row in store.connection.execute("SELECT * FROM messages ORDER BY store_id")]
            assert rows == expected["rows"]
            record = json.loads(store.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0])
            assert record == expected["record"]
            assert schema(store.connection) == expected["schema"]
            assert Path(store_mod.__file__).resolve().parent == args.source.resolve()
            print("OLD_READER_OK v0.28.1; schema unchanged; rows and unknown metadata key preserved")
    finally:
        store.close()


if __name__ == "__main__":
    main()
