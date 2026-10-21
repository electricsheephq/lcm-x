"""SPEC I1-fix4: one bounded review round; fake workers and small stores only."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from bench.instruments.local_catchup import audit, catchup, common, corpus, cpu, rescore, rig
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_fix_round3",
                  "bench.instruments.local_catchup.tests.test_corpus_audit"]
REPO = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("failure", ["exit", "breach"])
def test_early_abort_writes_summary_and_refreshes_events(fake_catchup, monkeypatch, failure):
    a, _, _, _, _, meter = fake_catchup
    launch = catchup.launch
    def failed_launch(job, generation):
        p = launch(job, generation)
        if failure == "exit":
            p.poll, p.returncode = lambda: 7, 7
        else:
            meter.breach = "swap_increased"
        return p
    monkeypatch.setattr(catchup, "launch", failed_launch)
    def stop(_):
        common.append(a.out/"events.jsonl", {"event": "prepared", "seconds": 3})
    monkeypatch.setattr(catchup, "stop", stop)
    # Ensure summary rates have nonzero elapsed time even on an immediate exit.
    clock = fake_catchup[4]
    monkeypatch.setattr(meter, "close", lambda: clock.__setitem__(0, clock[0]+1))
    summary = catchup.run(a)
    assert (summary["state"], summary["reason"]) == (
        "ABORTED", "worker_exit:7" if failure == "exit" else "swap_increased")
    assert summary["model_preparation_s"] == 3
    assert common.load(a.out/"summary.json") == summary


def test_hung_bind_aborts_with_named_worker(fake_rig, monkeypatch):
    a, checked, final, _, launched, clock = fake_rig
    a.max_wall, a.stagger = 5, 2
    final["complete"] = False
    monkeypatch.setattr(rig, "read_progress", lambda *args, **kw: checked.copy())
    launch = rig.launch
    def hung_launch(job, generation):
        p = launch(job, generation)
        if job.out.name == "1":
            (job.out/"worker-0.json").unlink()
        return p
    monkeypatch.setattr(rig, "launch", hung_launch)
    def bounded_sleep(seconds):
        clock[0] += seconds
        assert clock[0] <= 6, "pre-bind wait escaped max-wall"
    monkeypatch.setattr(rig.time, "sleep", bounded_sleep)
    stops = []
    monkeypatch.setattr(rig, "stop", lambda p: stops.append((p.pid, clock[0])))
    rig.main()
    summary = common.load(a.out/"summary.json")
    assert (summary["state"], summary["reason"]) == ("ABORTED", "bind_timeout:1")
    assert stops == [(1, 5), (2, 5)]
    assert len(launched) == 2 and summary["per_process_bind"][1] is None


def test_successful_launch_and_bind_consume_wall_budget(fake_rig, monkeypatch):
    a, checked, _, _, _, _ = fake_rig
    a.max_wall, a.stagger, a.restart_after, a.restart_fraction = 5, 2, 100, 1
    monkeypatch.setattr(rig, "read_progress", lambda *args, **kw: checked.copy())
    rig.main()
    summary = common.load(a.out/"summary.json")
    assert summary["cpu"]["end"] == 5


def test_new_id_rate_includes_before_first_poll(fake_catchup, monkeypatch):
    a, initial, final, _, _, _ = fake_catchup
    initial["classes"] = {"chunks": {"covered": 0, "expected": 2, "covered_tokens": 0}}
    final["classes"] = {"chunks": {"covered": 2, "expected": 2, "covered_tokens": 2}}
    monkeypatch.setattr(catchup, "progress", lambda *args: final.copy())
    summary = catchup.run(a)
    assert summary["new_ids_s"]*summary["end_to_end_s"] == 2


@pytest.mark.parametrize("reason", ["sustained_cpu_breach", "swap_increased", "worker_exit:7",
                                    "foreground_prepare_backfill_or_network", "final_integrity_or_residue"])
def test_rescore_preserves_guard_abort_without_progress(tmp_path, reason):
    common.dump(tmp_path/"summary.json", {"state": "ABORTED", "reason": reason})
    assert rescore.verdict(tmp_path/"progress.jsonl", 1200) == "ABORTED"
    assert common.load(tmp_path/"summary.json")["reason"] == reason


def test_cpu_peaks_clip_to_same_active_intervals(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "swap", lambda: 0)
    tick = [0.]
    monkeypatch.setattr(cpu.time, "monotonic", lambda: tick[0])
    values = iter([(1, 900, 90), (2, 20, 2), (3, 30, 3), (4, 800, 80)])
    monkeypatch.setattr(cpu, "process", lambda _: next(values))
    m = cpu.Supervisor(tmp_path/"cpu.jsonl")
    m.watch(1)
    for tick[0] in (1, 2, 3, 4):
        m.sample([1])
    m.windows[1][1] = 2.5
    row = m.summary(since=1, end=4)["per_process"]["1"]
    assert row["seconds"] == 1.5
    assert (row["peak_rss"], row["peak_threads"]) == (30, 3)
    row = m.summary(since=1, end=2)["per_process"]["1"]
    assert (row["peak_rss"], row["peak_threads"]) == (20, 2)


def canonical_store_digest(c):
    schema = audit.digest(c.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name"))
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return audit.digest([("schema", schema)]+[(t, audit.digest(sorted(
        c.execute(f'SELECT * FROM "{t}"'), key=repr))) for t in tables])


def canonical_store_snapshot(c):
    """Keep structure and per-column hashes for readable, content-free failures."""
    objects = {name: {"type": kind, "table": table, "digest": audit.digest([(sql,)])}
               for kind, name, table, sql in c.execute(
                   "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name")}
    tables = {}
    for name, obj in objects.items():
        if obj["type"] != "table":
            continue
        quoted = '"' + name.replace('"', '""') + '"'
        rows = sorted(c.execute(f"SELECT * FROM {quoted}"), key=repr)
        columns = [r[1] for r in c.execute(f"PRAGMA table_info({quoted})")]
        tables[name] = {"rows": len(rows), "digest": audit.digest(rows),
                        "columns": {col: audit.digest([(r[i],) for r in rows])
                                    for i, col in enumerate(columns)}}
    return {"digest": canonical_store_digest(c), "objects": objects, "tables": tables}


def store_changes(expected, actual):
    changes = []
    for section in ("objects", "tables"):
        before, after = expected[section], actual[section]
        for name in sorted(before.keys() | after.keys()):
            if before.get(name) == after.get(name):
                continue
            if name not in before or name not in after:
                change = "added" if name in after else "removed"
                changes.append(f"{section}: {name} {change}")
            elif section == "objects":
                obj = after[name]
                changes.append(f"schema: {obj['type']} {name} on table {obj['table']} changed")
            else:
                old, new = before[name], after[name]
                columns = sorted(col for col in old["columns"].keys() | new["columns"].keys()
                                 if old["columns"].get(col) != new["columns"].get(col))
                changes.append(f"table {name}: rows {old['rows']} -> {new['rows']}; "
                               f"changed columns: {', '.join(columns) or '(row association changed)'}")
    return "\n".join(changes)


@pytest.mark.parametrize("mutation,diagnostic", [
    ("ALTER TABLE messages ADD COLUMN origin TEXT", "changed columns: origin"),
    ("UPDATE messages SET content='private replacement'", "changed columns: content"),
    ("CREATE TRIGGER changed AFTER UPDATE ON messages BEGIN SELECT 1; END", "objects: changed added"),
])
def test_store_changes_names_structure_without_message_content(mutation, diagnostic):
    with sqlite3.connect(":memory:") as c:
        c.execute("CREATE TABLE messages(content TEXT)")
        c.execute("INSERT INTO messages VALUES ('private original')")
        before = canonical_store_snapshot(c)
        c.execute(mutation)
        after = canonical_store_snapshot(c)
    assert before["digest"] != after["digest"]
    changes = store_changes(before, after)
    assert diagnostic in changes
    assert "private" not in changes


def test_corpus_reuses_reader_and_preserves_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path/"home"))
    reads, original = [], corpus.connect
    def connect(db):
        reads.append(db)
        return original(db)
    monkeypatch.setattr(corpus, "connect", connect)
    db = tmp_path/"small.db"
    manifest = corpus.build(REPO, REPO, db, 8)
    # The separate provenance-audit extension adds metadata to the manifest.
    # Everything emitted by the old builder stays identical.
    manifest["preimage"].pop("metadata", None)
    manifest["preimage"].pop("provenance", None)
    manifest.pop("expected_metadata", None)
    assert hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == (
        "fa81d53f9f53a91e9fba3ea6aaad7c9eee09f247ea127fd731ad7cd1d27e5b88")
    # Normalize only setup clocks; compare canonical schema/row bytes (page counters vary).
    with sqlite3.connect(db) as c:
        c.execute("UPDATE lcm_embedding_profile SET registered_at='fixture-clock'")
        c.execute("UPDATE lcm_migration_state SET completed_at=0")
        c.execute("UPDATE metadata SET value='fixture-clock' WHERE key LIKE 'fts_integrity_checked_at:%'")
        c.commit()
        snapshot = canonical_store_snapshot(c)
    # #1748 added gap-fill invalidation triggers, first in 34cc7a53f58f,
    # extended in 04b38508e2bb, f84f849e63cb and 1eb0620a917d. Row bytes did not change.
    golden = json.loads(Path(__file__).with_name("local_catchup_store_golden.json").read_text())
    assert snapshot == golden, "Canonical store changed:\n" + store_changes(golden, snapshot)
    assert len(reads) <= 2, "one read connection for the build plus the preimage read"


@pytest.mark.parametrize("table,column", [("lcm_chunk_meta", "store_id"),
    ("lcm_chunk_meta", "chunk_index"), ("lcm_chunk_meta", "char_start"),
    ("lcm_chunk_meta", "char_end"), ("lcm_chunk_meta", "token_estimate"),
    ("lcm_chunk_meta", "embedded_at"), ("lcm_chunk_meta", "archived"),
    ("lcm_embedding_meta", "source_token_count"), ("lcm_embedding_meta", "embedded_at"),
    ("lcm_embedding_meta", "embedded_kind"), ("lcm_embedding_meta", "archived")])
def test_preimage_metadata_is_immutable(corpus, table, column):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    with audit.connect(db) as c:
        manifest["preimage"] = audit.preimage(c)
    with sqlite3.connect(db) as c:
        c.execute("PRAGMA ignore_check_constraints=ON")
        value = "'other'" if column == "embedded_kind" else f"{column}+1"
        c.execute(f"UPDATE {table} SET {column}={value}")
    result = audit.audit(db, manifest)
    assert catchup.audit_failed(result) and not result["complete"]


@pytest.mark.parametrize("table", ["lcm_chunk_vectors", "lcm_embedding_vectors",
                                    "lcm_chunk_meta", "lcm_embedding_meta"])
@pytest.mark.parametrize("extra", ["foreign_identity", "unexpected_id", "missing_pair"])
def test_final_audit_rejects_unexpected_vector_or_metadata(corpus, table, extra):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    assert not catchup.audit_failed(audit.audit(db, manifest))
    key = "chunk_id" if "chunk" in table else "embedded_id"
    with sqlite3.connect(db) as c:
        if extra == "missing_pair":
            c.execute(f"DELETE FROM {table}")
        else:
            columns = [r[1] for r in c.execute(f"PRAGMA table_info({table})")]
            changed = "identity_hash" if extra == "foreign_identity" else key
            selected = ["'unexpected'" if col == changed else col for col in columns]
            c.execute(f"INSERT INTO {table} SELECT {','.join(selected)} FROM {table} LIMIT 1")
    result = audit.audit(db, manifest)
    assert catchup.audit_failed(result) and not result["complete"]
