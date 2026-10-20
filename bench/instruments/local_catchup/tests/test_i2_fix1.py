import json
import sqlite3
import socket
import sys
from types import SimpleNamespace

import pytest

from bench.instruments.local_catchup import audit, common, floor, measurement, recall, turns
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


@pytest.mark.parametrize("result", ["backfill refused: lease busy", {"embedded": 2}])
def test_backfill_event_preserves_refusal_or_report(tmp_path, monkeypatch, result):
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", socket.socket.sendto)
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        monkeypatch.setattr(socket, name, getattr(socket, name))
    monkeypatch.setattr(common.threading, "current_thread", lambda: object())
    provider = SimpleNamespace(_FASTEMBED_THREADS=2,
        FastembedProvider=SimpleNamespace(_construct=lambda *args, **kw: None))
    command = SimpleNamespace(_chunk_backfill_run=lambda: result,
        _embedding_backfill_summary_run=lambda: result)
    common.guards(lambda name: provider if name == "embedding_provider" else command, tmp_path/"events.jsonl")
    assert command._chunk_backfill_run() == result
    event = json.loads((tmp_path/"events.jsonl").read_text().splitlines()[-1])
    assert event["embedded"] == (2 if isinstance(result, dict) else 0)


def test_inactive_turn_pair_is_invalid_even_at_full_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(turns, "prepare", lambda a: None)
    monkeypatch.setattr(turns, "cell", lambda *args: dict(
        bind_ms=1, p90=dict(ingest_preflight_ms=1, dispatch_ms=1),
        catching_up=True, compression_turns=10, catchup_activity=False))
    turns.main(SimpleNamespace(out=tmp_path, seed=1586, blocks=2, samples=600, smoke=False,
                               label="test", shape="short", baseline="empty"))
    result = common.load(tmp_path/"summary.json")
    assert result["state"] == "INVALID"
    assert result["reason"] == "on_no_catchup_activity"
    assert all(b["state"] == "NOT_JUDGED" for b in result["bars"].values())


@pytest.mark.parametrize("delta,prepared,prior,expected", [
    (0, True, False, False), (1, False, False, False),
    (1, True, False, True), (1, False, True, True)])
def test_turn_activity_requires_progress_and_preparation(tmp_path, delta, prepared, prior, expected):
    if prepared or prior:
        common.append(tmp_path/"events.jsonl", dict(event="prepared", foreground=False, at=1 if prior else 3))
    result = measurement.catchup_activity(tmp_path, 2, 5, {"chunks": 0}, {"chunks": delta}, require_prepared=True)
    assert result["catchup_activity"] is expected
    assert result["coverage_delta"] == delta


@pytest.mark.parametrize("at,foreground,embedded,expected", [
    (1, False, 1, False), (3, True, 1, False), (3, False, 0, False), (3, False, 1, False)])
def test_recall_activity_is_inside_overlap_window(tmp_path, at, foreground, embedded, expected):
    common.append(tmp_path/"events.jsonl", dict(event="_chunk_backfill_run", at=at,
        end=at+.1, foreground=foreground, embedded=embedded))
    activity = measurement.catchup_activity(tmp_path, 2, 5, {"chunks": 0}, {"chunks": 0})
    rows = [dict(start=2, end=5, seconds=.1, health="ok", fts_hits=True, honest_coverage=True)]*200
    result = recall.score(rows, [dict(start=2, end=5)]*10,
                          state="ON-catching-up", activity=activity)
    assert result["state"] == ("PASS" if expected else "INVALID")
    if not expected:
        assert result["reason"] == "on_no_catchup_activity"


def test_recall_activity_accepts_positive_coverage_without_backfill_event(tmp_path):
    result = measurement.catchup_activity(tmp_path, 2, 5, {"chunks": 1}, {"chunks": 2})
    assert result["catchup_activity"] and result["coverage_delta"] == 1


@pytest.mark.parametrize("shape,baseline", [("short", "empty"), ("mixed", "upgrade")])
def test_floor_shape_baseline_and_summary(tmp_path, monkeypatch, shape, baseline):
    out, scratch = tmp_path/"out", tmp_path/"scratch"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["floor", "--lcmx-repo", str(tmp_path), "--hermes-source", str(tmp_path),
        "--scratch", str(scratch), "--out", str(out), "--model-cache", str(tmp_path),
        "--cgroup", str(tmp_path), "--limited", "--cell", "coverage-rate", "--shape", shape, "--baseline", baseline])
    monkeypatch.setattr(floor.atexit, "register", lambda *args: None)
    monkeypatch.setattr(floor, "measured_limits", lambda *args: {"verified": True})
    monkeypatch.setattr(floor, "isolated", lambda *args: None)
    monkeypatch.setenv("HOME", str(tmp_path))
    built = []
    def build(*args):
        built.append(args)
        return {"shape": args[4], "baseline": args[5]}
    monkeypatch.setattr(floor, "build", build)
    monkeypatch.setattr(floor, "run", lambda a: dict(state="INCOMPLETE", rows_s=10))
    clock = iter([0, 7200])
    monkeypatch.setattr(floor.time, "monotonic", lambda: next(clock))
    floor.main()
    summary = common.load(out/"summary.json")
    assert built[0][4:] == (shape, baseline)
    assert (summary["shape"], summary["baseline"], summary["state"]) == (shape, baseline, "PASS")


def test_cheap_counts_ignore_new_trace_and_foreign_identity(corpus, monkeypatch):
    db, manifest, mod = corpus
    queries = []
    original = audit.connect
    def connect(path):
        c = original(path)
        c.set_trace_callback(queries.append)
        return c
    monkeypatch.setattr(audit, "connect", connect)
    assert audit.vector_counts(db, manifest) == {"chunks": 0, "leaves": 0}
    fill(db, manifest, mod)
    with sqlite3.connect(db) as c:
        c.execute("DELETE FROM lcm_chunk_vectors WHERE chunk_id='1:0'")
        c.execute("INSERT INTO lcm_chunk_vectors SELECT '99:0',identity_hash,vec FROM lcm_chunk_vectors LIMIT 1")
        for table, key, replacement in (("lcm_chunk_meta", "chunk_id", "99:0"),
                                        ("lcm_embedding_meta", "embedded_id", "99")):
            columns = [r[1] for r in c.execute(f"PRAGMA table_info({table})")]
            fields = ["'"+replacement+"'" if col == key else "99" if col == "store_id" else col for col in columns]
            c.execute(f"INSERT INTO {table} SELECT {','.join(fields)} FROM {table} LIMIT 1")
        c.execute("INSERT INTO lcm_embedding_vectors SELECT '99',identity_hash,vec FROM lcm_embedding_vectors LIMIT 1")
        # A foreign vector cannot conceal the missing corpus chunk.
        c.execute("INSERT INTO lcm_chunk_vectors SELECT chunk_id,'foreign',vec FROM lcm_chunk_vectors LIMIT 1")
    counts = audit.vector_counts(db, manifest)
    assert counts == {"chunks": len(manifest["expected"]["chunks"])-1, "leaves": 1}
    assert not any("json_each" in sql or "SELECT *" in sql or "quick_check" in sql for sql in queries)


def test_recall_loop_uses_cheap_counts_and_boundary_audits(tmp_path, monkeypatch):
    a = SimpleNamespace(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path,
        state="ON-catching-up", smoke=True, rows=8, queries=1, seed=1586, block=0)
    common.dump(tmp_path/"manifest.json", {"canaries": [{"value": "canary"}],
        "expected": {"chunks": {"1:0": 1}, "leaves": {"1": 1}}})
    common.dump(tmp_path/"queries.json", [{"id": "q", "canary": 0}])
    dag = SimpleNamespace(add_node=lambda *args, **kw: 1, close=lambda: None)
    engine = SimpleNamespace(_config=SimpleNamespace(recall_query_timeout_s=.1),
        _dag=dag, shutdown=lambda: None)
    store = SimpleNamespace(close=lambda: None)
    def mod(name):
        return {"store": SimpleNamespace(MessageStore=lambda *args: store),
            "dag": SimpleNamespace(SummaryDAG=lambda *args: dag),
            "tools": SimpleNamespace(lcm_recall=lambda *args, **kw: json.dumps({"hits": []}))}[name]
    monkeypatch.setattr(recall, "runtime", lambda request: (a, mod, engine, 1))
    audits, counts = [], []
    monkeypatch.setattr(recall, "audit", lambda *args: audits.append(1) or {"complete": False})
    monkeypatch.setattr(recall, "progress", lambda *args: pytest.fail("full expected-ID scan on recall path"), raising=False)
    monkeypatch.setattr(audit, "vector_counts", lambda *args: counts.append(1) or {"chunks": 0, "leaves": 0})
    if hasattr(recall, "vector_counts"):
        monkeypatch.setattr(recall, "vector_counts", audit.vector_counts)
    monkeypatch.setattr(recall, "trace", lambda *args: None)
    monkeypatch.setattr(recall, "integrity", lambda *args: None)
    monkeypatch.setattr(recall, "safety", lambda *args: {})
    recall.worker(None)
    assert len(audits) == 2 and counts
    assert all(r["incomplete"] for r in map(json.loads, (tmp_path/"recalls.jsonl").read_text().splitlines()))
