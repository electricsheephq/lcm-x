"""Tiny offline controls for the opt-in busy-bar storage arms."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from bench.instruments.busy_bar import arms, driver, gate, orchestrate

TREE = Path(__file__).resolve().parents[1]
BASE_SHA = "e0b6d0ae7be3795a84547d77aa55efe923ebb91f"
BASE_GATE_SHA256 = "5dd2a2303cb9dd7bea2477c159ecffd6d6904c136e967e123dab7e84587c123d"  # gate.py at 48081b7c


@pytest.fixture(autouse=True)
def save_evidence(tmp_path, request):
    # pytest recreates basetemp before tmp_path; restore an explicitly nested TMPDIR.
    temporary = os.environ.get("TMPDIR")
    if temporary:
        Path(temporary).mkdir(parents=True, exist_ok=True)
    yield
    destination = os.environ.get("BUSY_BAR_EVIDENCE_DIR")
    if destination:
        root = Path(destination) / request.node.name
        for source in tmp_path.rglob("*"):
            if source.is_file() and source.suffix in (".json", ".jsonl") and "work" not in source.parts:
                target = root / source.relative_to(tmp_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)


def parse(*values):
    parser = argparse.ArgumentParser()
    arms.add_arguments(parser)
    return parser.parse_args(values)


def test_defaults_install_nothing():
    args = parse()
    assert vars(args) == {"wal_autocheckpoint": None, "checkpointer": "off", "background_writers": "none",
                          "stem_backlog": 20000, "record_holds": False}
    connect = sqlite3.connect
    with arms.StorageArms(args) as storage:
        assert not storage.enabled
        assert sqlite3.connect is connect
    assert arms.arguments(args) == []


@pytest.mark.parametrize("flag,value", [
    ("--wal-autocheckpoint", "0"), ("--wal-autocheckpoint", "500"),
    ("--checkpointer", "off"), ("--checkpointer", "passive:1"),
    *(("--background-writers", name) for name in ("none", "stem", "integrity", "embedding", "all")),
    ("--stem-backlog", "1"),
])
def test_flag_values(flag, value):
    assert getattr(parse(flag, value), flag[2:].replace("-", "_")) == (
        int(value) if flag in ("--wal-autocheckpoint", "--stem-backlog") else value)


@pytest.mark.parametrize("flag,value", [
    ("--wal-autocheckpoint", "-1"), ("--wal-autocheckpoint", "1.5"),
    ("--checkpointer", "passive:0"), ("--checkpointer", "passive:-1"),
    ("--checkpointer", "passive:NaN"), ("--checkpointer", "active:1"),
    ("--background-writers", "unknown"), ("--stem-backlog", "0"), ("--stem-backlog", "-1"),
    ("--record-holds", "false"),
])
def test_bad_flag_values(flag, value):
    with pytest.raises(SystemExit) as exc:
        parse(flag, value)
    assert exc.value.code == 2


@pytest.mark.parametrize("module", [driver, orchestrate])
def test_both_clis_parse_arm_flags(module, monkeypatch):
    # Stop at the first post-parse operation, without importing a product tree.
    calls = []

    def stop(args):
        calls.append(args)
        raise RuntimeError("parsed")

    if module is driver:
        monkeypatch.setattr(driver, "isolated_environment", stop)
        positional = ["tree", "ingest", "output", "work"]
    else:
        monkeypatch.setattr(orchestrate, "collect", lambda *a, **kw: stop(kw))
        positional = ["baseline", "candidate", "nonexistent-cm1-output", "--work-root", "nonexistent-cm1-work"]
    monkeypatch.setattr(sys, "argv", ["cli", *positional, "--wal-autocheckpoint", "0",
                                     "--checkpointer", "passive:3", "--background-writers", "all",
                                     "--stem-backlog", "1001", "--record-holds"])
    with pytest.raises(RuntimeError, match="parsed"):
        module.main()
    if module is orchestrate:
        assert calls[0]["arm_arguments"] == ["--wal-autocheckpoint", "0", "--checkpointer", "passive:3",
                                             "--background-writers", "all", "--stem-backlog", "1001",
                                             "--record-holds"]


def run_driver(script, tree, output, work, *flags):
    command = [sys.executable, "-B", str(script), str(tree), "ingest", str(output), str(work),
               "--runs", "2", "--k", "1", "--b-view-rows", "2", *flags]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return gate.read(output), command


@pytest.mark.parametrize("writer,expected", [
    ("stem", {"stem-batch"}), ("integrity", {"integrity-scan"}),
    ("embedding", {"embedding-vector-batch"}),
    ("all", {"stem-batch", "integrity-scan", "embedding-vector-batch"}),
])
def test_writer_arm_smoke(tmp_path, writer, expected):
    output = tmp_path / "arms.jsonl"
    rows, _ = run_driver(Path(driver.__file__), TREE, output, tmp_path / "work",
                         "--background-writers", writer, "--stem-backlog", "1001", "--record-holds")
    for i, row in enumerate(rows):
        details = row["storage_arms"]
        holds = gate.read(tmp_path / f"arms-run-{i}" / "holds.jsonl")
        during = [hold for hold in holds if details["schedule_start"] <= hold["start"] < details["schedule_end"]]
        assert expected <= {hold["writer"] for hold in during}
        assert {"foreground-A", "foreground-B"} <= {hold["writer"] for hold in during}
        assert all(hold["outcome"] == "rollback" for hold in during if hold["writer"] == "integrity-scan")
        assert all(hold["hold_ms"] >= 0 and hold["end"] >= hold["start"] for hold in holds)
        database = str(tmp_path / "work" / f"run-{i}" / "lcm.db")
        assert all(hold["database"] == database for hold in holds)
        assert all(hold["start_basis"] in {"acquired", "call_entry"} for hold in holds)
        assert all(hold["acquired_at"] == hold["start"] for hold in holds if hold["start_basis"] == "acquired")
        assert all("acquired_at" not in hold for hold in holds if hold["start_basis"] == "call_entry")
        pairs = [(a, b) for index, a in enumerate(holds) for b in holds[index + 1:]
                 if a["connection_id"] != b["connection_id"]
                 and max(a["start"], b["start"]) < min(a["end"], b["end"])]
        counts = {"acquired_pairs": sum(a["start_basis"] == b["start_basis"] == "acquired" for a, b in pairs),
                  "call_entry_involved_pairs": sum("call_entry" in (a["start_basis"], b["start_basis"])
                                                   for a, b in pairs), "records": len(holds)}
        (tmp_path / f"overlaps-{i}.json").write_text(json.dumps(counts))
        assert counts["acquired_pairs"] == 0, counts
        with sqlite3.connect(tmp_path / "work" / f"run-{i}" / "lcm.db") as conn:
            if "stem-batch" in expected:
                assert conn.execute("SELECT count(*) FROM messages_fts_stem_docsize").fetchone()[0] >= 1001
            if "embedding-vector-batch" in expected:
                assert conn.execute("SELECT count(*) FROM lcm_embedding_vectors").fetchone()[0] == 64


def test_smoke_nonbusy_holds_have_positive_spans(tmp_path):
    run_driver(Path(driver.__file__), TREE, tmp_path / "arms.jsonl", tmp_path / "work",
               "--background-writers", "all", "--stem-backlog", "1001", "--record-holds")
    holds = [hold for i in range(2) for hold in gate.read(tmp_path / f"arms-run-{i}" / "holds.jsonl")
             if hold["outcome"] != "busy"]
    zero_count = sum(hold["end"] == hold["start"] for hold in holds)
    (tmp_path / "zero-length.json").write_text(json.dumps({"nonbusy_records": len(holds),
                                                         "zero_length_records": zero_count}))
    assert holds
    assert zero_count == 0
    assert all(hold["hold_ms"] > 0 and hold["end"] > hold["start"] for hold in holds)


def test_checkpoint_smoke(tmp_path):
    rows, _ = run_driver(Path(driver.__file__), TREE, tmp_path / "checkpoint.jsonl", tmp_path / "work",
                         "--wal-autocheckpoint", "0", "--checkpointer", "passive:10")
    for i, row in enumerate(rows):
        configured = row["storage_arms"]["configured_connections"]
        assert len(configured) >= 3
        assert all(c == {"wal_autocheckpoint": 0, "synchronous": 2} for c in configured)
        checkpoints = gate.read(tmp_path / f"checkpoint-run-{i}" / "checkpoints.jsonl")
        assert checkpoints and all(c["error"] is None and len(c["result"]) == 3 for c in checkpoints)
        assert not (tmp_path / f"checkpoint-run-{i}" / "holds.jsonl").exists()


def normalize(rows):
    """Exclude timing values only; canonically rename OS IDs, retaining relationships."""
    rows = deepcopy(rows)
    for row in rows:
        row.pop("host", None)  # Host readings are reported outside the original timing windows.
        threads, processes = {}, {}
        for op in row["operations"]:
            for key in ("begin", "end", "wall_ms"):
                op.pop(key)
            threads.setdefault(op["thread"], f"thread-{len(threads)}")
            op["thread"] = threads[op["thread"]]
        for record in row["logs"]:
            record.pop("time")
            record["message"] = re.sub(r"lock_hold_ms=\d+\.\d+", "lock_hold_ms=<timing>", record["message"])
            threads.setdefault(record["thread"], f"thread-{len(threads)}")
            processes.setdefault(record["process"], f"process-{len(processes)}")
            record["thread"] = threads[record["thread"]]
            record["process"] = processes[record["process"]]
        for summary in row["success_wall_ms"].values():
            if isinstance(summary, dict):
                for key in ("p50", "p95", "max"):
                    summary.pop(key)
    return rows


def test_default_unchanged(tmp_path):
    base_tree = Path(os.environ.get("BUSY_BAR_BASE_TREE", TREE))
    if base_tree == TREE:
        # Source-only fallback for ordinary CI; this is not a new worktree.
        snapshot = tmp_path / "base-instrument"
        snapshot.mkdir()
        for name in ("driver.py", "cells.py"):
            source = subprocess.run(["git", "show", f"{BASE_SHA}:bench/instruments/busy_bar/{name}"],
                                    cwd=TREE, capture_output=True, check=True)
            (snapshot / name).write_bytes(source.stdout)
        base_script = snapshot / "driver.py"
    else:
        base_script = base_tree / "bench/instruments/busy_bar/driver.py"
    work = tmp_path / "work"
    before, base_command = run_driver(base_script, base_tree, tmp_path / "base.jsonl", work)
    work.rename(tmp_path / "saved-base-work")
    after, candidate_command = run_driver(Path(driver.__file__), TREE, tmp_path / "candidate.jsonl", work)
    assert normalize(before) == normalize(after)
    assert driver.tree_identity(base_tree) == driver.tree_identity(TREE)
    # Prove the field-level summary too, retaining successful-call counts.
    assert gate.comparison(normalize(before), normalize(before)) == gate.comparison(normalize(after), normalize(after))
    (tmp_path / "comparison.json").write_text(json.dumps({"base_command": base_command,
        "candidate_command": candidate_command, "equal": True, "rows": normalize(after)}, indent=2))


def test_default_run_keeps_original_functions(tmp_path):
    # Exercise main's installation boundary and assert again inside the real run.
    code = '''
import importlib.util, json, os, sqlite3, sys
from pathlib import Path
work, tree = Path(sys.argv[2]), Path(sys.argv[3])
for key in list(os.environ):
    if key.startswith(('LCM_', 'HERMES_')):
        del os.environ[key]
os.environ['HERMES_HOME'] = str(work / 'isolated-home')
# Pin every binding before driver/arms can install anything at import time.
spec = importlib.util.spec_from_file_location('hermes_lcm', tree / '__init__.py',
                                            submodule_search_locations=[str(tree)])
sys.modules['hermes_lcm'] = importlib.util.module_from_spec(spec)
import hermes_lcm.db_bootstrap as db
bindings = [(sqlite3, 'connect', sqlite3.connect)]
for name in ('configure_connection', 'dispatch_background_stem_backfill', '_run_stem_backfill',
             '_dispatch_background_integrity_scan', '_run_background_integrity_scan'):
    bindings.append((db, name, getattr(db, name)))
sys.path.insert(0, sys.argv[1])
import driver
original_load = driver.load_tree
original_sequence = driver.cells.sequence
original_run = driver.run
checks = []
def load(tree):
    result = original_load(tree)
    assert sys.modules['hermes_lcm.db_bootstrap'] is db
    return result
def check(stage):
    assert len(bindings) == 6
    for module, name, original in bindings:
        assert getattr(module, name) is original, f'{stage}: {name} changed on default run'
    checks.append(stage)
def run(*args, **kwargs):
    check('run-entry')
    result = original_run(*args, **kwargs)
    check('run-return')
    assert 'storage_arms' not in result and not result['errors']
    return result
def sequence(*args, **kwargs):
    check('inside-run')
    return original_sequence(*args, **kwargs)
driver.load_tree, driver.run, driver.cells.sequence = load, run, sequence
sys.argv = ['driver', str(tree), 'ingest', str(work / 'default.jsonl'), str(work / 'work'),
            '--runs', '1', '--k', '1', '--b-view-rows', '2']
driver.main()
assert checks == ['run-entry', 'inside-run', 'run-return']
print(json.dumps({'checks': checks, 'original_bindings': [name for _, name, _ in bindings]}))
'''
    completed = subprocess.run([sys.executable, "-B", "-c", code, str(Path(driver.__file__).parent),
                                str(tmp_path), str(TREE)], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    (tmp_path / "default-bindings.json").write_text(completed.stdout)


@pytest.mark.parametrize("teardown_fails", [False, True])
def test_failure_before_first_operation_preserves_error(tmp_path, teardown_fails):
    code = '''
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import driver
work, tree = Path(sys.argv[2]), Path(sys.argv[3])
original_finish = driver.arms.StorageArms.finish
finished = []
class CellFailure(RuntimeError):
    pass
def fail(*args):
    raise CellFailure('before-first-operation')
def finish(self, output):
    # Real prepared stem/integrity workers must be waiting at the start gate.
    workers = [*self.bootstrap._stem_threads.values(), *self.bootstrap._integrity_scan_threads.values()]
    assert any(worker.is_alive() for worker in workers)
    assert not self.gate.is_set()
    result = original_finish(self, output)
    assert self.gate.is_set() and not any(worker.is_alive() for worker in workers)
    assert result['schedule_start'] is None
    finished.append(True)
    if teardown_fails:
        raise ValueError('secondary teardown failure')
    return result
teardown_fails = sys.argv[4] == 'True'
driver.cells.sequence = fail
driver.arms.StorageArms.finish = finish
sys.argv = ['driver', str(tree), 'ingest', str(work / 'failure.jsonl'), str(work / 'work'),
            '--background-writers', 'all', '--stem-backlog', '1001', '--record-holds']
start = time.monotonic()
try:
    driver.main()
except CellFailure as exc:
    assert str(exc) == 'before-first-operation'
else:
    raise AssertionError('cell failure did not propagate')
elapsed = time.monotonic() - start
assert finished == [True] and elapsed < 5, (finished, elapsed)
print(json.dumps({'error': 'CellFailure:before-first-operation', 'elapsed_seconds': elapsed,
                  'waiting_workers_joined': True, 'secondary_teardown_failure': teardown_fails}))
'''
    completed = subprocess.run([sys.executable, "-B", "-c", code, str(Path(driver.__file__).parent),
                                str(tmp_path), str(TREE), str(teardown_fails)],
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    (tmp_path / "early-failure.json").write_text(completed.stdout)


@pytest.mark.parametrize("release", ["commit", "rollback", "close", "exit-commit", "exit-rollback",
                                         "exit-error", "sql-commit", "sql-rollback", "savepoint", "autocommit"])
def test_hold_end_precedes_slow_release(tmp_path, monkeypatch, release):
    import time

    path = tmp_path / "slow-release.db"
    recorder = arms.HoldRecorder(path)
    releases = []
    call_spans = []
    target = None
    other = None

    class SlowRelease(sqlite3.Connection):
        def slow(self, call):
            if self is not target:
                return call()
            start = time.monotonic()
            time.sleep(0.005)
            releases.append(time.monotonic())
            result = call()
            # A second writer owns and releases the lock before the first call returns.
            other.execute("BEGIN IMMEDIATE")
            other.commit()
            call_spans.append((start, time.monotonic()))
            return result

        def commit(self):
            return self.slow(super().commit)

        def rollback(self):
            return self.slow(super().rollback)

        def close(self):
            return self.slow(super().close)

        def __exit__(self, *args):
            result = self.slow(lambda: super(SlowRelease, self).__exit__(*args))
            if release == "exit-error":
                raise sqlite3.OperationalError("delayed exit failure")
            return result

        def execute(self, sql, parameters=()):
            def call():
                return super(SlowRelease, self).execute(sql, parameters)

            return self.slow(call) if sql in ("COMMIT", "ROLLBACK", "RELEASE only", "INSERT INTO t VALUES(1)") else call()

    monkeypatch.setattr(sqlite3, "Connection", SlowRelease)
    factory = recorder.connection_class()
    conn = sqlite3.connect(path, factory=factory, isolation_level=None)
    other = sqlite3.connect(path, factory=factory, isolation_level=None)
    conn.execute("CREATE TABLE t(n)")
    recorder.records.clear()
    if release == "savepoint":
        conn.execute("SAVEPOINT only")
        conn.execute("INSERT INTO t VALUES(2)")
    elif release != "autocommit":
        conn.execute("BEGIN IMMEDIATE")
    target = conn
    if release.startswith("exit-"):
        kind = ValueError if release == "exit-rollback" else None
        if release == "exit-error":
            with pytest.raises(sqlite3.OperationalError, match="delayed exit failure"):
                conn.__exit__(kind, None, None)
        else:
            conn.__exit__(kind, None, None)
    elif release.startswith("sql-"):
        conn.execute(release.removeprefix("sql-").upper())
    elif release in ("savepoint", "autocommit"):
        conn.execute("RELEASE only" if release == "savepoint" else "INSERT INTO t VALUES(1)")
    else:
        getattr(conn, release)()
    target = None
    records = recorder.records
    first = next(row for row in records if row["connection_id"] == conn.connection_id)
    pairs = [(a, b) for i, a in enumerate(records) for b in records[i + 1:]
             if a["connection_id"] != b["connection_id"]
             and max(a["start"], b["start"]) < min(a["end"], b["end"])]
    (tmp_path / "slow-release.json").write_text(json.dumps({"records": records, "release_at": releases,
                                                          "overlap_count": len(pairs)}))
    assert len(records) == 2 and len(releases) == 1
    if release == "autocommit":
        # The write's call-entry span includes its call, even after lock release.
        start, end = call_spans[0]
        assert first["start_basis"] == "call_entry" and first["hold_ms"] > 0
        assert first["start"] <= start < releases[0] < end <= first["end"]
        assert len(pairs) == 1
    else:
        assert first["end"] <= releases[0]
        assert len(pairs) == 0
    if release != "close":
        conn.close()
    other.close()


def test_hold_transaction_paths(tmp_path):
    path = tmp_path / "holds.db"
    recorder = arms.HoldRecorder(path)
    conn = sqlite3.connect(path, factory=recorder.connection_class(), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t(n)")
    recorder.records.clear()
    conn.execute("BEGIN IMMEDIATE")
    conn.cursor().execute("INSERT INTO t VALUES(1)")
    conn.commit()
    conn.execute("BEGIN")  # Deferred BEGIN does not itself hold a write lock.
    conn.execute("SELECT * FROM t").fetchall()
    conn.rollback()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO t VALUES(2)")
    conn.rollback()
    conn.isolation_level = "IMMEDIATE"
    with conn:
        conn.executemany("INSERT INTO t VALUES(?)", [(3,), (4,)])
    with pytest.raises(ValueError), conn:
        conn.execute("INSERT INTO t VALUES(5)")
        raise ValueError("rollback")
    conn.execute("INSERT INTO t VALUES(6)")
    conn.close()
    assert [r["outcome"] for r in recorder.records] == ["commit", "rollback", "commit", "rollback", "rollback"]
    assert [r["start_basis"] for r in recorder.records] == ["acquired", "acquired", *["call_entry"] * 3]


def test_nested_savepoint_keeps_outer_hold(tmp_path):
    path = tmp_path / "nested.db"
    recorder = arms.HoldRecorder(path)
    conn = sqlite3.connect(path, factory=recorder.connection_class(), isolation_level=None)
    conn.executescript("CREATE TABLE t(n); CREATE TRIGGER copy AFTER INSERT ON t WHEN new.n=1 "
                       "BEGIN INSERT INTO t VALUES(2); END;")
    recorder.records.clear()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO t VALUES(1)")
    conn.execute("SAVEPOINT nested")
    conn.execute("INSERT INTO t VALUES(3)")
    conn.execute("ROLLBACK TO nested")
    conn.execute("RELEASE nested")
    assert recorder.records == []
    conn.commit()
    assert len(recorder.records) == 1 and recorder.records[0]["outcome"] == "commit"
    assert conn.execute("SELECT n FROM t ORDER BY n").fetchall() == [(1,), (2,)]
    conn.close()


def test_busy_attempt_and_autocommit_rows(tmp_path):
    path = tmp_path / "busy.db"
    recorder = arms.HoldRecorder(path)
    conn = sqlite3.connect(path, factory=recorder.connection_class(), isolation_level=None)
    conn.execute("CREATE TABLE t(n)")
    recorder.records.clear()
    assert conn.executemany("INSERT INTO t VALUES(?)", [(1,), (2,)]).rowcount == 2
    assert len(recorder.records) == 2
    with sqlite3.connect(path) as other:
        other.execute("BEGIN IMMEDIATE")
        conn.execute("PRAGMA busy_timeout=1")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("BEGIN IMMEDIATE")
    assert recorder.records[-1]["outcome"] == "busy"
    assert recorder.records[-1]["start_basis"] == "call_entry"
    assert "acquired_at" not in recorder.records[-1]
    conn.close()


def test_hold_recorder_excludes_other_databases(tmp_path):
    path = tmp_path / "main.db"
    recorder = arms.HoldRecorder(path)
    for database in (":memory:", tmp_path / "other.db", path):
        conn = sqlite3.connect(database=database, factory=recorder.connection_class())
        conn.execute("CREATE TABLE t(n)")
        conn.close()
    assert len(recorder.records) == 1
    assert recorder.records[0]["database"] == str(path)


def test_unconfigured_connection_pragmas(tmp_path):
    from hermes_lcm.db_bootstrap import configure_connection

    path = tmp_path / "pragmas.db"
    configured = sqlite3.connect(path)
    configure_connection(configured)
    private = sqlite3.connect(path)
    values = {name: private.execute(f"PRAGMA {name}").fetchone()[0]
              for name in ("wal_autocheckpoint", "synchronous")}
    assert values == {"wal_autocheckpoint": 1000, "synchronous": 2}
    assert configured.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 500
    (tmp_path / "pragmas.json").write_text(json.dumps({"sqlite_version": sqlite3.sqlite_version, **values}))
    private.close()
    configured.close()


def test_integrity_insert_blocks_second_writer(tmp_path):
    from hermes_lcm.db_bootstrap import check_external_content_fts_integrity
    from hermes_lcm.store import MessageStore, build_message_fts_spec

    store = MessageStore(tmp_path / "scan.db")
    store._conn.executemany("INSERT INTO messages(session_id,role,content,timestamp) VALUES('seed','user',?,0)",
                     [(f"row {i} " + "alpha beta gamma " * 16,) for i in range(2000)])
    store._conn.commit()
    other = sqlite3.connect(store.db_path, timeout=0.001)
    blocked = []
    scanning = False

    def progress():
        if scanning and not blocked:
            try:
                other.execute("BEGIN IMMEDIATE")
                other.rollback()
            except sqlite3.OperationalError as exc:
                blocked.append({"stage": "during INSERT", "sqlite_errorcode": exc.sqlite_errorcode,
                                "message": str(exc)})
        return 0

    class ScanConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            nonlocal scanning
            scanning = "VALUES('integrity-check'" in sql
            try:
                result = super().execute(sql, parameters)
                if scanning:
                    with pytest.raises(sqlite3.OperationalError) as exc:
                        other.execute("BEGIN IMMEDIATE")
                    blocked.append({"stage": "before rollback", "sqlite_errorcode": exc.value.sqlite_errorcode})
                return result
            finally:
                scanning = False

    conn = sqlite3.connect(store.db_path, factory=ScanConnection)
    conn.set_progress_handler(progress, 100)
    assert check_external_content_fts_integrity(conn, build_message_fts_spec())["status"] == "pass"
    conn.set_progress_handler(None, 0)
    assert blocked and blocked[0]["sqlite_errorcode"] & 0xff == sqlite3.SQLITE_BUSY
    assert {row["stage"] for row in blocked} == {"during INSERT", "before rollback"}
    other.execute("BEGIN IMMEDIATE")  # The rollback releases it.
    other.rollback()
    other.close()
    conn.close()
    store.close()
    (tmp_path / "lock-probe.json").write_text(json.dumps({"during_scan": blocked, "after_rollback": "acquired"}))


def test_default_background_activity(tmp_path):
    # Observe the original paths in an isolated process without any arm flags.
    code = '''
import json, sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from bench.instruments.busy_bar import driver
work = Path(sys.argv[1])
driver.isolated_environment(work)
config, engine = driver.load_tree(Path(sys.argv[2]))
driver.offline_setup(engine)
import hermes_lcm.db_bootstrap as db
counts = Counter()
def observe(name, original):
    def wrapped(*a, **kw):
        counts[name] += 1
        return original(*a, **kw)
    return wrapped
with patch.object(db, '_run_stem_backfill', observe('stem', db._run_stem_backfill)), \\
     patch.object(db, '_run_background_integrity_scan', observe('integrity', db._run_background_integrity_scan)), \\
     patch.object(db, 'check_external_content_fts_integrity', observe('integrity_calls', db.check_external_content_fts_integrity)):
    for cell in driver.cells.CELLS:
        row = driver.run(cell, work / cell, config, engine, driver.Capture(), k=1, b_view_rows=2)
        db.join_background_integrity_scans(5)
        db.join_background_stem_backfills(5)
        assert not row['errors']
print(json.dumps(dict(counts)))
'''
    completed = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path / "work"), str(TREE)],
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    counts = json.loads(completed.stdout)
    assert counts == {}
    (tmp_path / "default-writers.json").write_text(json.dumps({"stem": counts.get("stem", 0),
        "integrity": counts.get("integrity", 0), "integrity_calls": counts.get("integrity_calls", 0),
        "cells": list(driver.cells.CELLS)}))


def test_stem_claim_holds_have_background_owner(tmp_path):
    code = r'''
import json, sys
from pathlib import Path
from unittest.mock import patch
from bench.instruments.busy_bar import driver
work, tree = Path(sys.argv[1]), Path(sys.argv[2])
driver.isolated_environment(work)
driver.load_tree(tree)
from hermes_lcm import db_bootstrap as db
original_dispatch = db.dispatch_background_stem_backfill
original_enter = driver.arms.StorageArms.__enter__
storage, claims = None, []
def enter(self):
    global storage
    storage = self
    return original_enter(self)
def dispatch(*args, **kwargs):
    before = len(storage.recorder.records)
    result = original_dispatch(*args, **kwargs)
    claims.extend(dict(row) for row in storage.recorder.records[before:] if row["start_basis"] == "acquired")
    return result
with patch.object(db, "dispatch_background_stem_backfill", dispatch), \
     patch.object(driver.arms.StorageArms, "__enter__", enter):
    sys.argv = ["driver", str(tree), "ingest", str(work / "claims.jsonl"), str(work / "work"),
                "--runs", "2", "--k", "1", "--b-view-rows", "2", "--window-ms", "0",
                "--background-writers", "stem", "--stem-backlog", "1001", "--record-holds"]
    driver.main()
print(json.dumps(claims))
'''
    completed = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path), str(TREE)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    claims = json.loads(completed.stdout)
    (tmp_path / "claim-probe.json").write_text(completed.stdout)
    rows = gate.read(tmp_path / "claims.jsonl")
    assert claims and {hold["writer"] for hold in claims} == {"stem-claim"}
    for i, row in enumerate(rows):
        holds = gate.read(tmp_path / f"claims-run-{i}" / "holds.jsonl")
        own_claims = [hold for hold in claims if hold["database"] == str(tmp_path / "work" / f"run-{i}" / "lcm.db")]
        assert own_claims
        assert all(hold in holds and hold["start"] < row["storage_arms"]["schedule_start"] for hold in own_claims)


@pytest.mark.parametrize("isolation", [None, "", "DEFERRED", "IMMEDIATE"])
@pytest.mark.parametrize("fails", [False, True])
def test_executescript_dml_matches_native(tmp_path, isolation, fails):
    observations = []
    script = "INSERT INTO t VALUES(1); INSERT INTO t VALUES(observe());"
    if fails:
        script += "INSERT INTO missing VALUES(3);"
    recorder = arms.HoldRecorder(tmp_path / "recorded.db")
    for name, factory in (("native", sqlite3.Connection), ("recorded", recorder.connection_class())):
        path = tmp_path / f"{name}.db"
        conn = sqlite3.connect(path, factory=factory, isolation_level=isolation)
        observer = sqlite3.connect(path)
        conn.execute("CREATE TABLE t(n)")
        recorder.records.clear()
        visible = []

        def observe():
            visible.append(observer.execute("SELECT n FROM t ORDER BY n").fetchall())
            return 2

        conn.create_function("observe", 0, observe)
        if fails:
            with pytest.raises(sqlite3.OperationalError, match="missing"):
                conn.executescript(script)
        else:
            conn.executescript(script)
        observations.append((visible, conn.in_transaction, conn.isolation_level))
        conn.close()
        assert observer.execute("SELECT n FROM t ORDER BY n").fetchall() == [(1,), (2,)]
        observer.close()
    assert observations == [([[(1,)]], False, isolation)] * 2
    assert len(recorder.records) == 2
    assert all(row["outcome"] == "commit" and row["hold_ms"] > 0 for row in recorder.records)

    # A script's own BEGIN remains open; restoring None must not commit it.
    for factory in (sqlite3.Connection, recorder.connection_class()):
        conn = sqlite3.connect(tmp_path / "explicit.db", factory=factory, isolation_level=isolation)
        conn.executescript("CREATE TABLE IF NOT EXISTS t(n); BEGIN IMMEDIATE; INSERT INTO t VALUES(3);")
        assert conn.in_transaction
        assert conn.isolation_level == isolation
        conn.rollback()
        conn.close()


def embedding_retirement_probe(tmp_path, mode, cell="ingest"):
    # Real scheduler/engine retirement in an isolated process; only the pass is controlled.
    code = r'''
import json, sqlite3, sys, threading, time
from pathlib import Path
from unittest.mock import patch
from bench.instruments.busy_bar import driver
work, tree, mode, cell = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
driver.isolated_environment(work)
config, engine = driver.load_tree(tree)
driver.offline_setup(engine, embedding=True)
from hermes_lcm import embedding_maintenance as maintenance, maintenance_scheduler
started, retire_started, completed = threading.Event(), threading.Event(), threading.Event()
active_at_finish, active_after_retirement = [], []
original_shutdown = engine.LCMEngine.shutdown
original_retire = engine.LCMEngine._retire_embedding_maintenance
original_drain = engine.LCMEngine.drain_embedding_maintenance
original_finish = driver.arms.StorageArms.finish
original_pass = maintenance.run_incremental_embedding_pass
scheduler = maintenance_scheduler._EMBEDDING_MAINTENANCE_SCHEDULER
original_scheduler_retire = scheduler.retire_owner
first_path = None

def slow_pass(database, *args, **kwargs):
    global first_path
    if first_path is None:
        first_path = str(database)
        started.set()
        assert retire_started.wait(10)
        time.sleep(4.5)  # Exceeds both engines' native 2 s retirement waits.
        with sqlite3.connect(database) as conn:
            conn.execute("CREATE TABLE busy_bar_late_probe(n)")
            conn.execute("INSERT INTO busy_bar_late_probe VALUES(1)")
        result = original_pass(database, *args, **kwargs)
        completed.set()
        return result
    return original_pass(database, *args, **kwargs)

def retire(self, wait):
    if mode == "slow":
        assert started.wait(10)
        retire_started.set()
    result = original_retire(self, wait)
    active_after_retirement.append(started.is_set() and not completed.is_set())
    return result

def shutdown(self):
    result = original_shutdown(self)
    if mode == "shutdown-raise":
        raise RuntimeError("injected shutdown failure")
    return result

def finish(self, output):
    active_at_finish.append(started.is_set() and not completed.is_set())
    return original_finish(self, output)

def drain(self, timeout=None):
    if mode == "slow":
        assert started.wait(10)
        retire_started.set()
        return original_drain(self, timeout=timeout)
    if mode == "drain-raise":
        raise RuntimeError("injected drain failure")
    return False

def retire_owner(owner, timeout):
    if mode == "retire-swallowed" and timeout == 0:
        class BrokenOwners(dict):
            def pop(self, *args):
                raise ValueError("injected inside real retire_owner")
        with patch.object(scheduler, "_owned_keys", BrokenOwners()):
            return original_scheduler_retire(owner, timeout)
    result = original_scheduler_retire(owner, timeout)
    if mode == "retire-raise":
        raise RuntimeError("injected retirement failure")
    return False if mode == "retire-false" else result

with patch.object(maintenance, "run_incremental_embedding_pass", slow_pass if mode == "slow" else original_pass), \
     patch.object(engine.LCMEngine, "_retire_embedding_maintenance", retire), \
     patch.object(engine.LCMEngine, "shutdown", shutdown), \
     patch.object(driver.arms.StorageArms, "finish", finish), \
     patch.object(scheduler, "retire_owner", retire_owner):
    if mode == "slow" or mode.startswith("drain-"):
        engine.LCMEngine.drain_embedding_maintenance = drain
    sys.argv = ["driver", str(tree), cell, str(work / "probe.jsonl"), str(work / "work"),
                "--runs", "2", "--k", "1", "--b-view-rows", "2", "--window-ms", "0",
                "--background-writers", "embedding", "--record-holds"]
    status = 0
    try:
        driver.main()
    except SystemExit as exc:
        status = exc.code
    if mode == "slow":
        assert completed.wait(10)
print(json.dumps({"status": status, "active_at_finish": active_at_finish, "first_path": first_path,
                  "active_after_retirement": active_after_retirement}))
'''
    completed = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path), str(TREE), mode, cell],
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    (tmp_path / "retirement-probe.json").write_text(completed.stdout)
    return json.loads(completed.stdout), gate.read(tmp_path / "probe.jsonl")


@pytest.mark.parametrize("cell", ["ingest", "restart", "session-end"])
def test_embedding_pass_drained_before_retirement(tmp_path, cell):
    proof, rows = embedding_retirement_probe(tmp_path, "slow", cell)
    assert proof["status"] == 0
    assert proof["active_after_retirement"]
    if cell == "session-end":
        assert any(proof["active_after_retirement"])
        # B may share the pass and drain it during its untimed shutdown.
        assert proof["active_at_finish"][-1] is False
        timed = next(r for r in rows[0]["operations"] if r["op"] == "session-end")
        assert timed["wall_ms"] < 4000
        assert any(d["deferred_owner"] and d["duration_ms"] > 0
                   for d in rows[0]["storage_arm_drains"])
    else:
        assert not any(proof["active_after_retirement"])
        assert proof["active_at_finish"] == [False, False]
    assert len(rows) == 2 and all(not row["errors"] and not row["inconclusive"] for row in rows)
    first = gate.read(tmp_path / "probe-run-0" / "holds.jsonl")
    second = gate.read(tmp_path / "probe-run-1" / "holds.jsonl")
    assert any(hold["writer"] == "embedding" and hold["database"] == proof["first_path"] for hold in first)
    assert all(hold["database"] != proof["first_path"] for hold in second)
    assert max(hold["end"] for hold in first) <= rows[0]["storage_arms"]["schedule_end"]


@pytest.mark.parametrize("mode", ["drain-false", "drain-raise", "retire-false", "retire-raise", "shutdown-raise"])
def test_embedding_retirement_failure_is_inconclusive(tmp_path, mode):
    proof, rows = embedding_retirement_probe(tmp_path, mode)
    assert proof["status"] == 2
    assert len(rows) == 1  # Failed retirement must never reuse the recorder for another run.
    assert rows[0]["inconclusive"]
    errors = rows[0]["storage_arms"]["errors"]
    assert errors and any(mode.split("-")[0] in error for error in errors)
    assert gate.comparison(rows, rows)["verdict"] == "INCONCLUSIVE"
    assert not (tmp_path / "probe-run-1").exists()


def test_nonwaiting_embedding_drain_outside_timed_session_end(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_lcm import engine, maintenance_scheduler

    scheduler = maintenance_scheduler._EMBEDDING_MAINTENANCE_SCHEDULER
    clock, retired = [10.0], []
    old_owner = object()
    def slow_drain(timeout):
        clock[0] += 7
        return True
    def retire_owner(owner, timeout):
        retired.append((owner, timeout))
        return slow_drain(timeout) if timeout else False
    def native_retire(instance, wait):
        scheduler.retire_owner(instance._embedding_maintenance_owner, timeout=2 if wait else 0)
    monkeypatch.setattr(scheduler, "retire_owner", retire_owner)
    monkeypatch.setattr(engine.LCMEngine, "_retire_embedding_maintenance", native_retire)
    monkeypatch.setattr(arms.time, "monotonic", lambda: clock[0])
    instance = SimpleNamespace(_embedding_maintenance_owner=old_owner, drain_embedding_maintenance=slow_drain)
    with arms.StorageArms(parse("--background-writers", "embedding")) as storage:
        storage.schedule_start = clock[0]
        operations = driver.Operations(driver.Capture(), False)
        operations(instance, "A", "session-end", lambda: engine.LCMEngine._retire_embedding_maintenance(instance, False), 4)
        instance._embedding_maintenance_owner = object()  # Public on_session_start rearms a new owner.
        details = storage.finish(tmp_path / "arms")
    assert operations.records[0]["wall_ms"] == 0
    assert storage.embedding_drains == [{"deferred_owner": True, "skipped": False, "duration_ms": 7000}]
    assert retired == [(old_owner, 0), (old_owner, 35)]
    assert "errors" not in details


@pytest.mark.parametrize("failure", ["timeout", "error"])
def test_embedding_failure_skips_later_drains(tmp_path, failure):
    from types import SimpleNamespace
    storage = arms.StorageArms(parse("--background-writers", "embedding"))
    calls = []
    def drain(timeout):
        calls.append(timeout)
        if failure == "error":
            raise ValueError("injected")
        return False
    fake = SimpleNamespace(drain_embedding_maintenance=drain)
    storage.drain_embedding(fake)
    storage.drain_embedding(fake)
    assert calls == [35]
    assert [row["skipped"] for row in storage.embedding_drains] == [False, True]
    assert storage.embedding_errors


def test_gate_py_unchanged_from_base():
    # gate.py must stay byte-identical to main 48081b7c; CI's depth-1 checkout lacks that commit,
    # so the working file is pinned by its hash.
    assert hashlib.sha256(Path(gate.__file__).read_bytes()).hexdigest() == BASE_GATE_SHA256, (
        "gate.py is the busy-bar gate; a change to it re-baselines the busy-bar bar, "
        "so update BASE_GATE_SHA256 only together with that decision.")


@pytest.mark.parametrize("writer", ["embedding", "none"])
def test_drain_records_leave_base_smoke_reason_and_text_identical(tmp_path, writer):
    from types import ModuleType
    from tests.test_busy_bar_instrument import ledger, control, stamp, CANDIDATE_IDENTITY

    rows, _ = run_driver(Path(driver.__file__), TREE, tmp_path / "rows.jsonl", tmp_path / "work",
                         "--background-writers", writer)
    base = ModuleType("base_gate")
    base.__package__ = gate.__package__
    source = Path(gate.__file__).read_bytes()
    exec(compile(source, "base_gate.py", "exec"), base.__dict__)
    roles = [stamp(ledger(0), driver.BASELINE_IDENTITY), stamp(ledger(0), CANDIDATE_IDENTITY),
             stamp(control(1), driver.BASELINE_IDENTITY), stamp(control(1), CANDIDATE_IDENTITY)]
    for group in roles:
        for i, row in enumerate(group):
            row.update({k: v for k, v in rows[i % len(rows)].items()
                        if k in ("storage_arms", "storage_arm_drains")})
    old_roles = deepcopy(roles)
    for group in old_roles:
        for row in group:
            row.pop("storage_arm_drains", None)
            row.get("storage_arms", {}).pop("embedding_drains", None)
    for group, old_group in zip(roles, old_roles):
        assert json.dumps(gate.nonstandard_workload("fixture", group)) == json.dumps(
            base.nonstandard_workload("fixture", old_group))
    before, after = base.evaluate(*old_roles), gate.evaluate(*roles)
    assert before["verdict"] == after["verdict"] == "SMOKE PASS"
    assert gate.report(after).encode() == base.report(before).encode()
    if writer == "embedding":
        assert all(row["storage_arm_drains"] for row in rows)
        assert all("embedding_drains" not in row["storage_arms"] for row in rows)
    else:
        assert all("storage_arms" not in row and "storage_arm_drains" not in row for row in rows)


def test_embedding_swallowed_retire_exception_is_inconclusive(tmp_path):
    proof, rows = embedding_retirement_probe(tmp_path, "retire-swallowed", "session-end")
    assert proof["status"] == 2
    assert len(rows) == 1 and rows[0]["inconclusive"]
    assert any("retire" in error for error in rows[0]["storage_arms"]["errors"])
    assert gate.comparison(rows, rows)["verdict"] == "INCONCLUSIVE"
    assert not (tmp_path / "probe-run-1").exists()
