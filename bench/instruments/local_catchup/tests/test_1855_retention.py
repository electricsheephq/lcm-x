"""Synthetic final-audit failures retain evidence without changing cell verdicts."""
from contextlib import closing
import copy
import hashlib
from pathlib import Path
import sqlite3
from types import SimpleNamespace as NS

import pytest

from bench.instruments.local_catchup import audit, common, measurement, recall
from bench.instruments.local_catchup.tests.test_1586_fix1 import cover
from bench.instruments.local_catchup.tests.test_corpus_audit import REPO

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


@pytest.mark.parametrize("damage,state,error", [
    ("trace", "ON-covered", "trace row corruption: missing at real-turn index 0"),
    ("vector", "ON-catching-up", "pre-existing vector corruption"),
])
@pytest.mark.usefixtures("fake_store_disk_headroom")
def test_failed_recall_cell_retains_evidence(corpus, tmp_path, monkeypatch, damage, state, error):
    db, manifest, mod = corpus
    cover(db, manifest, mod)
    assert audit.audit(db, manifest)["complete"]
    out, covered = tmp_path/"evidence", tmp_path/"covered"
    out.mkdir()
    covered.mkdir()
    for target in (tmp_path/"baseline.db", covered/"store.db"):
        with closing(sqlite3.connect(db)) as src, closing(sqlite3.connect(target)) as dst:
            src.backup(dst)
    common.dump(out/"manifest.json", manifest)
    common.dump(covered/"manifest.json", manifest)
    common.dump(out/"queries.json", [])
    a = NS(scratch=tmp_path, out=out, covered_store=covered, model_cache=tmp_path,
           lcmx_repo=REPO, hermes_source=REPO, smoke=False, label="fixture", max_seconds=30,
           rows=8, seed=1586)
    engine = NS(_dag=NS(add_node=lambda *a, **kw: None),
                _config=NS(recall_query_timeout_s=1), shutdown=lambda: None)
    monkeypatch.setattr(recall, "runtime", lambda request: (
        NS(**{k: Path(v) if k in ("root", "out", "db", "home") else v
              for k, v in common.load(request).items()}), None, engine, 0))
    class Thread:
        def __init__(self, **kw): pass
        def start(self): pass
        def join(self, *args): pass
        def is_alive(self): return False
    monkeypatch.setattr(recall.threading, "Thread", Thread)
    monkeypatch.setattr(recall.threading.Event, "wait", lambda *args: True)
    monkeypatch.setattr(recall, "wait_covered", lambda *args: None)
    monkeypatch.setattr(recall, "calibration_window", lambda *args: None)
    monkeypatch.setattr(recall, "safety", lambda *args: {})
    trace = [["i2-trace", "conversation", "user", 1., hashlib.sha256(b"trace").hexdigest()]]
    key = f"chunks/{manifest['identities']['chunk']}/1:0"
    writer = None
    def publication(a, *args):
        nonlocal writer
        writer = sqlite3.connect(a.db)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("INSERT INTO metadata(key,value) VALUES('last-committed-write','retained')")
        if damage == "vector":
            writer.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'")
        writer.commit()  # Keep the writer open: the committed WAL must be backed up.
        return (trace if damage == "trace" else []), 1, False
    monkeypatch.setattr(recall, "publication_trace", publication)
    monkeypatch.setattr(measurement, "Supervisor", lambda _: NS(
        breach=None, watch=lambda _: None, start=lambda: None, close=lambda: None,
        summary=lambda *args: {}))
    exceptions = []
    def launch(cmd, **kw):
        try:
            recall.worker(Path(cmd[cmd.index("--worker")+1]))
        except RuntimeError as exc:
            exceptions.append(exc)
        return NS(pid=1, returncode=1, poll=lambda: 1)
    monkeypatch.setattr(measurement.subprocess, "Popen", launch)
    monkeypatch.setattr(measurement, "stop", lambda _: dict(stop_killed=False))
    try:
        with pytest.raises(RuntimeError, match="worker_exit:1"):
            measurement.cell(a, state, 0, "recall")
        root = out/f"0-{state}"
        assert [str(exc) for exc in exceptions] == [error]
        assert common.load(root/"summary.json")["state"] == "ABORTED"
        assert common.load(root/"summary.json")["reason"] == "worker_exit:1"
        assert (root/"failed-cell.db").is_file()
        with closing(sqlite3.connect(root/"failed-cell.db")) as saved:
            assert saved.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            assert saved.execute("SELECT value FROM metadata WHERE key='last-committed-write'").fetchone() == ("retained",)
        inputs = common.load(root/"integrity-inputs.json")
        assert inputs["trace_rows"] == (trace if damage == "trace" else [])
        assert inputs["vectors"] == manifest["observed_vectors"] | manifest["preimage"]["vectors"]
        assert inputs["metadata"] == manifest["observed_metadata"] | manifest["preimage"]["metadata"]
        assert inputs["identities"] == manifest["identities"]
        failure = common.load(root/"integrity-failure.json")
        assert failure["error"] == error
        assert failure["identities"] == manifest["identities"]
        assert failure["model"] == common.MODEL
        assert failure["mismatch"] == (dict(kind="trace", index=0, status="missing", expected=trace[0], actual=None)
            if damage == "trace" else dict(kind="vector", key=key, status="changed",
                expected=inputs["vectors"][key], actual=hashlib.sha256(bytes(1536)).hexdigest()))
    finally:
        if writer:
            writer.close()


def test_healthy_integrity_uploads_no_new_files(corpus, tmp_path):
    db, manifest, _ = corpus
    a = NS(db=db, root=tmp_path)
    measurement.integrity(a, manifest)
    assert not (tmp_path/"integrity-inputs.json").exists()
    assert not (tmp_path/"integrity-failure.json").exists()
    assert not (tmp_path/"failed-cell.db").exists()


def test_pins_are_captured_before_audit_extends_them(corpus, tmp_path, monkeypatch):
    db, manifest, mod = corpus
    cover(db, manifest, mod)
    expected = copy.deepcopy(manifest)
    real_audit = measurement.audit
    def extending(db, current):
        inputs = common.load(tmp_path/"integrity-inputs.json")
        assert inputs["vectors"] == expected.get("observed_vectors", {}) | expected["preimage"]["vectors"]
        assert inputs["metadata"] == expected.get("observed_metadata", {}) | expected["preimage"]["metadata"]
        real_audit(db, current)
        assert current["observed_vectors"]
        with closing(sqlite3.connect(db)) as writer:
            writer.execute("INSERT INTO metadata(key,value) VALUES('later-write','excluded')")
            writer.commit()
        raise RuntimeError("injected final audit failure")
    monkeypatch.setattr(measurement, "audit", extending)
    with pytest.raises(RuntimeError, match="injected final audit failure"):
        measurement.integrity(NS(db=db, root=tmp_path), manifest)
    with closing(sqlite3.connect(tmp_path/"failed-cell.db")) as saved:
        assert saved.execute("SELECT value FROM metadata WHERE key='later-write'").fetchone() is None


def test_backup_failure_retains_locked_db_wal_shm(corpus, tmp_path, monkeypatch):
    db, manifest, _ = corpus
    class BrokenBackup(sqlite3.Connection):
        def backup(self, *args, **kw):
            raise sqlite3.OperationalError("injected backup failure")
    monkeypatch.setattr(measurement, "connect", lambda path: sqlite3.connect(
        path.resolve().as_uri()+"?mode=ro", uri=True, factory=BrokenBackup))
    copy_file = measurement.shutil.copy2
    def locked_copy(source, target):
        with closing(sqlite3.connect(db, timeout=0)) as competing_writer:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                competing_writer.execute("BEGIN IMMEDIATE")
        return copy_file(source, target)
    monkeypatch.setattr(measurement.shutil, "copy2", locked_copy)
    with closing(sqlite3.connect(db)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("UPDATE messages SET content='last-committed-write' WHERE store_id=1")
        writer.commit()
        with pytest.raises(RuntimeError, match="^corpus row corruption$"):
            measurement.integrity(NS(db=db, root=tmp_path), manifest)
        assert all((tmp_path/("failed-cell.db"+suffix)).is_file() for suffix in ("", "-wal", "-shm"))
        failure = common.load(tmp_path/"integrity-failure.json")
        assert failure["backup_error"] == "OperationalError: injected backup failure"
        assert failure["backup"] == "locked-db-wal-shm"
        with closing(sqlite3.connect(tmp_path/"failed-cell.db")) as saved:
            assert saved.execute("PRAGMA quick_check").fetchone() == ("ok",)
            assert saved.execute("SELECT content FROM messages WHERE store_id=1").fetchone() == ("last-committed-write",)


def test_retention_errors_never_replace_integrity_error(corpus, tmp_path, monkeypatch):
    db, manifest, _ = corpus
    with closing(sqlite3.connect(db)) as writer:
        writer.execute("UPDATE messages SET content='changed' WHERE store_id=1")
        writer.commit()
    def unwritable(*args):
        raise OSError("injected evidence write failure")
    monkeypatch.setattr(measurement, "dump", unwritable)
    with pytest.raises(RuntimeError) as raised:
        measurement.integrity(NS(db=db, root=tmp_path), manifest)
    assert str(raised.value) == "corpus row corruption"
    assert raised.value.__notes__ == ["integrity evidence write failed: OSError: injected evidence write failure"]
    assert (tmp_path/"failed-cell.db").is_file()
