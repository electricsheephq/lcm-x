"""Round seven: verify the five post-stop-bar claims on the stacked follow-up."""
from argparse import Namespace as NS
from contextlib import ExitStack
import hashlib
import json
import sqlite3
import threading

import pytest

from bench.instruments.local_catchup import audit, common, floor, measurement, turns
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


@pytest.mark.parametrize("caller", ["main", "i2-recall"])
@pytest.mark.parametrize("target", ["embedding-backfill", "lcm-query-embed"])
@pytest.mark.parametrize("foreground", [False, True])
def test_f1_join_guard_tracks_request_on_both_callers(tmp_path, monkeypatch, caller, target, foreground):
    request = tmp_path/"request.json"
    common.dump(request, {**{k: str(tmp_path) for k in
        ("db", "root", "home", "scratch", "out", "lcmx_repo", "hermes_source", "model_cache")}, "state": "ON"})
    monkeypatch.setattr(threading.Thread, "join", threading.Thread.join)
    for name in ("isolated", "guards"):
        monkeypatch.setattr(measurement, name, lambda *a: None)
    monkeypatch.setattr(measurement, "product", lambda *a: None)
    def end(*a):
        raise ValueError("wrapper installed")
    monkeypatch.setattr(measurement, "config", end)
    with pytest.raises(ValueError, match="wrapper installed"):
        measurement.runtime(request)
    joined = threading.Thread(target=lambda: None, name=target)
    joined.start()
    errors = []
    def join():
        token = common.REQUEST.set(foreground)
        try:
            joined.join()
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            common.REQUEST.reset(token)
    if caller == "main":
        join()
    else:
        recall = threading.Thread(target=join, name=caller)
        recall.start()
        recall.join()
    joined.join()
    denied = foreground and target != "lcm-query-embed"
    assert errors == (["foreground embedding join forbidden"] if denied else [])
    assert (tmp_path/"events.jsonl").exists() is denied
    if denied:
        assert common.load(tmp_path/"events.jsonl")["violation"] == "foreground_join"


@pytest.mark.parametrize("outcome", ["drain", "deadline", "corruption"])
def test_f2_full_counts_wait_for_clean_boundary_audit(corpus, tmp_path, monkeypatch, outcome):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    with sqlite3.connect(db) as c:
        mod("command")._ensure_inflight_table(c)
        c.execute("INSERT INTO lcm_embedding_backfill_inflight(embedded_id,identity_hash,state) VALUES('1:0',?,'dispatched')",
                  (manifest["identities"]["chunk"],))
        if outcome == "corruption":
            c.execute("UPDATE messages SET content='corrupt' WHERE store_id=1")
    checked = audit.audit(db, manifest)
    assert not checked["complete"] and checked["residue"]
    assert all(checked["classes"][k]["missing_count"] == 0 for k in manifest["expected"])
    assert audit.vector_counts(db, manifest) == {k: len(v) for k, v in manifest["expected"].items()}
    clock, sleeps = [0.], []
    monkeypatch.setattr(measurement.time, "monotonic", lambda: clock[0])
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
        if outcome == "drain":
            with sqlite3.connect(db) as c:
                c.execute("DELETE FROM lcm_embedding_backfill_inflight")
    monkeypatch.setattr(measurement.time, "sleep", sleep)
    a = NS(db=db, root=tmp_path, max_seconds=2)
    engine = NS(_config=None)
    if outcome == "drain":
        measurement.wait_covered(a, mod, engine, manifest)
        assert audit.audit(db, manifest)["complete"]
    else:
        with pytest.raises(RuntimeError, match="covered setup failed"):
            measurement.wait_covered(a, mod, engine, manifest)
    rows = [json.loads(line) for line in (tmp_path/"coverage.jsonl").read_text().splitlines()]
    if outcome == "corruption":
        assert not sleeps and len(rows) == 2
    else:
        assert sleeps and len(rows) >= 3
        assert rows[-1]["audit"]["complete"] is (outcome == "drain")
        if outcome == "deadline":
            assert 2 <= clock[0] <= 3


def test_f3_turns_preserves_first_real_engine_shutdown_failure(corpus, tmp_path, monkeypatch):
    db, _, mod = corpus
    a = NS(out=tmp_path, root=tmp_path, db=db, home=tmp_path/"engine-home",
           state="OFF", samples=1, seed=1, block=0)
    cfg = mod("config").LCMConfig(database_path=str(db), embeddings_enabled=False)
    engine = mod("engine").LCMEngine(config=cfg, hermes_home=str(a.home))
    common.dump(tmp_path/"manifest.json", {})
    monkeypatch.setattr(turns, "runtime", lambda _: (a, mod, engine, 0))
    monkeypatch.setattr(audit, "progress", lambda *a: dict(complete=False, classes={}))
    monkeypatch.setattr(turns, "trace", lambda *a: None)
    monkeypatch.setattr(turns, "catchup_activity", lambda *a, **kw: {})
    monkeypatch.setattr(turns, "safety", lambda *a: {})
    calls = []
    failure = OSError("first store cleanup failure")
    close = engine._store.close
    def failing_close():
        close()
        calls.append("close")
        raise failure if len(calls) == 1 else OSError("repeated store cleanup failure")
    monkeypatch.setattr(engine._store, "close", failing_close)
    try:
        with pytest.raises(OSError) as stopped:
            turns.worker(None)
        assert stopped.value is failure
        assert calls == ["close"]
        assert not (tmp_path/"raw-summary.json").exists()
    finally:
        # The real shutdown already attempted every closer, including the injected one.
        monkeypatch.setattr(engine._store, "close", close)
        engine.shutdown()


@pytest.mark.parametrize("cell", ["coverage-rate", "recall"])
@pytest.mark.parametrize("state", ["PASS", "FAIL", "INVALID", "ABORTED", "INCOMPLETE"])
def test_f4_floor_nonpass_exits_after_summary_and_labels(tmp_path, monkeypatch, cell, state):
    out = tmp_path/"out"
    monkeypatch.setattr(floor.sys, "platform", "linux")
    monkeypatch.setattr(floor.sys, "argv", ["floor", "--lcmx-repo", str(tmp_path),
        "--hermes-source", str(tmp_path), "--scratch", str(tmp_path/"scratch"),
        "--out", str(out), "--model-cache", str(tmp_path/"cache"), "--cgroup", str(tmp_path),
        "--limited", "--cell", cell])
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(floor, "measured_limits", lambda _: dict(verified=True))
    monkeypatch.setattr(floor, "isolated", lambda *a: None)
    monkeypatch.setattr(floor, "build", lambda *a: {})
    monkeypatch.setattr(floor, "run", lambda a: {})
    monkeypatch.setattr(floor, "rate", lambda *a: dict(state=state))
    def recall_run(*a, **kw):
        (out/"recall").mkdir()
        common.dump(out/"recall/summary.json", dict(state=state))
        common.append(out/"recall/rows.jsonl", dict(sample=1))
    monkeypatch.setattr(floor.subprocess, "run", recall_run)
    code = 0
    try:
        floor.main()
    except SystemExit as stopped:
        code = stopped.code
    assert common.load(out/"summary.json") == dict(
        state=state, label=floor.LABEL, shape="short", baseline="empty")
    assert common.load(out/"limits.json")["label"] == floor.LABEL
    if cell == "recall":
        assert common.load(out/"recall/summary.json")["label"] == floor.LABEL
        assert common.load(out/"recall/rows.jsonl")["label"] == floor.LABEL
    assert code == (0 if state == "PASS" else 1)


def test_f5_scratch_protection_allows_only_cell_owned_handles(tmp_path):
    scratch = tmp_path/"scratch"
    a = NS(root=tmp_path/"evidence", home=scratch/"0-OFF-home", db=scratch/"cell.db",
           model_cache=tmp_path/"cache", user_home=tmp_path/"runner-home", scratch=scratch,
           lcmx_repo=tmp_path/"repo", hermes_source=tmp_path/"host", _next_progress=float("inf"))
    own = [a.home/"state.db", a.db, a.root/"output.json", a.model_cache/"model.bin"]
    foreign = [scratch/"1-ON-home/state.db", scratch/"baseline.db", scratch/"other-artifact.bin"]
    for path in own + foreign:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    with ExitStack() as handles:
        for path in own:
            handles.enter_context(path.open("rb"))
        assert measurement.safety(a)["violations"] == 0
        for path in foreign:
            handles.enter_context(path.open("rb"))
        with pytest.raises(RuntimeError, match="cross-profile access"):
            measurement.safety(a)
    assert set(common.load(a.root/"file-access.jsonl")["unexpected_path_hashes"]) == {
        hashlib.sha256(str(path.resolve()).encode()).hexdigest() for path in foreign}
