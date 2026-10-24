"""Round-3 false-PASS regressions on actual request and instrument paths."""
from argparse import Namespace as NS
from contextlib import nullcontext
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import threading

import pytest

from bench.instruments.local_catchup import audit, common, cpu, measurement, probes, recall, rig, turns
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_fix_round3",
                  "bench.instruments.local_catchup.tests.test_corpus_audit"]


@pytest.mark.parametrize("path", ["recall", "cold"])
def test_request_build_on_real_product_child_is_foreground(corpus, tmp_path, monkeypatch, path):
    db, manifest, mod = corpus
    provider = mod("embedding_provider")
    built = []
    class FakeTextEmbedding:
        def __init__(self, **kw):
            built.append((threading.current_thread().name, getattr(common, "REQUEST", NS(get=lambda: False)).get()))
        def query_embed(self, texts):
            return [[.1]*384 for _ in texts]
        embed = query_embed
    monkeypatch.setattr(provider, "_load_fastembed", lambda: FakeTextEmbedding)
    monkeypatch.setattr(mod("tools"), "probe_provider_availability", lambda config: {"available": True, "detail": "synthetic fastembed backend"})
    monkeypatch.setattr(provider.FastembedProvider, "_construct", provider.FastembedProvider._construct)
    for name in ("_chunk_backfill_run", "_embedding_backfill_summary_run"):
        monkeypatch.setattr(mod("command"), name, getattr(mod("command"), name))
    events = tmp_path/"events.jsonl"
    common.guards(mod, events)
    if path == "recall":
        reader = NS(_config=common.config(mod, db), _hermes_home=str(tmp_path),
                    _store=mod("store").MessageStore(db), _dag=mod("dag").SummaryDAG(db),
                    current_session_id="i2-recall")
        try:
            # The marker, when available, must cross both real profile_bound children.
            with getattr(common, "foreground_request", nullcontext)():
                mod("tools").lcm_recall({"query": "canary"}, engine=reader)
        finally:
            reader._store.close()
            reader._dag.close()
    else:
        # v2 item 3: force a cold foreground probe despite independent background warm-up.
        monkeypatch.setattr(mod("engine").LCMEngine, "_schedule_embedding_maintenance", lambda self: None)
        common.dump(tmp_path/"manifest.json", manifest)
        monkeypatch.setattr(probes, "guards", lambda *args: None)
        monkeypatch.setattr(probes, "isolated", lambda home, cache: home.mkdir(parents=True))
        monkeypatch.setattr(probes, "product", lambda *args: mod)
        a = NS(home=tmp_path/"cold", db=db, out=tmp_path, manifest=tmp_path/"manifest.json",
               lcmx_repo=Path(__file__).resolve().parents[4], hermes_source=tmp_path)
        probes.cold_side(a, True)
    recorded = [json.loads(line) for line in events.read_text().splitlines()]
    assert any(e.get("event") == "prepare" and e["foreground"] for e in recorded)
    assert not any(request for _, request in built)


@pytest.mark.parametrize("coverage,reason,arms,expected", [
    ({"chunk": "full", "summary": "full"}, "summary semantic index catching up (3 rows not yet embedded)", ["chunk"], False),
    ({"chunk": "none", "summary": "full"}, "chunk vectors are unavailable", ["chunk", "summary"], False),
    ({"chunk": "bounded", "summary": "full_approx"}, "chunk arm coverage bounded: scored the 10 most-recent of 99 vectors (older vectors excluded); summary arm coverage full_approx: whole corpus reached via the binary prescreen, but top-k is approximate (stage-1 keeps only the closest survivors before exact rescore)", ["chunk", "summary"], True),
    ({"chunk": "full", "summary": "full"}, "chunk semantic index catching up; summary semantic index catching up", ["chunk", "summary"], True),
])
def test_disclosure_is_bound_to_every_incomplete_arm(coverage, reason, arms, expected):
    assert recall.honest_coverage(dict(degraded=True, degraded_reason=reason,
        provenance=dict(coverage=coverage)), arms) is expected


@pytest.mark.parametrize("state", ["OFF", "ON-covered", "ON-catching-up"])
@pytest.mark.parametrize("failure", ["absent", "failed", "stale", "healthy"])
def test_worker_semantic_arms_cannot_pass_fts_only(tmp_path, monkeypatch, state, failure):
    common.dump(tmp_path/"manifest.json", dict(expected=dict(chunks={"1": 1, "2": 1}, leaves={"1": 1}),
                                               canaries=[dict(value="query")]))
    common.dump(tmp_path/"queries.json", [dict(id=str(i), canary=0, query_class="canary" if i % 2 == 0 else "broad_nl", text="What turn was acknowledged") for i in range(400)])
    a = NS(out=tmp_path, root=tmp_path, db=tmp_path/"db", home=tmp_path,
           state=state, smoke=False, rows=250000, queries=200, seed=1, block=0)
    store = NS(close=lambda: None)
    ids = iter(range(400))
    engine = NS(_config=NS(recall_query_timeout_s=1), _dag=NS(add_node=lambda *a, **kw: next(ids)), shutdown=lambda: None)
    cov = {"chunk": "full", "summary": "full"} if failure == "healthy" else {"chunk": "none", "summary": "none"}
    reason = {"absent": "chunk semantic index catching up; summary semantic index catching up; embedding provider unavailable: boom", "failed": "chunk semantic index catching up; summary semantic index catching up; chunk arm failed: boom; summary arm failed: boom",
              "stale": "embedding identity is stale", "healthy": "chunk semantic index catching up; summary semantic index catching up"}[failure]
    def call(*args, **kw):
        assert getattr(common, "REQUEST", NS(get=lambda: False)).get() or not hasattr(common, "REQUEST")
        engine._dag.add_node(NS(depth=0), before_commit=lambda: None)
        return json.dumps(dict(degraded=True, degraded_reason=reason, hits=[dict(arms=["fts"])],
            provenance=dict(coverage=cov if failure != "absent" else {"fts": "ok"},
                            arms_run=["fts", "chunk", "summary"] if failure == "healthy" else ["fts"])))
    def mod(name):
        return {"store": NS(MessageStore=lambda *args: store), "dag": NS(SummaryDAG=lambda *args: store),
                "tools": NS(lcm_recall=call)}[name]
    # Declaration item 6: isolate the existing health probe from the new FTS observer.
    monkeypatch.setattr(recall, "diagnostics", lambda *a: ({}, None))
    monkeypatch.setattr(recall, "calibration_window", lambda *a: None)  # Separate raw-window regressions in fix2.
    monkeypatch.setattr(recall, "runtime", lambda _: (a, mod, engine, 1))
    reads = iter([dict(chunks=0, leaves=0) if i % 2 == 0 else dict(chunks=1, leaves=0) for i in range(800)])
    monkeypatch.setattr(recall, "vector_counts", lambda *args: dict(chunks=2, leaves=1) if state == "ON-covered" else next(reads))
    monkeypatch.setattr(recall, "audit", lambda *args: dict(classes=dict(
        chunks=dict(covered=1, expected=2, missing_count=1),
        leaves=dict(covered=0, expected=1, missing_count=1))))
    monkeypatch.setattr(recall, "safety", lambda *args: {})
    monkeypatch.setattr(recall, "integrity", lambda *args: None)
    monkeypatch.setattr(recall, "wait_covered", lambda *args: None)
    monkeypatch.setattr(recall, "trace", lambda *args: None)
    event = threading.Event
    def stopped():
        result = event()
        result.set()
        return result
    class Thread:
        def __init__(self, target, name):
            self.target = target
        def start(self):
            self.target()
        def join(self, *args):
            pass
        def is_alive(self):
            return False
    monkeypatch.setattr(threading, "Event", stopped)
    monkeypatch.setattr(threading, "Thread", Thread)
    recall.worker(None)
    result = common.load(tmp_path/"raw-summary.json")
    # Workers retain raw class statistics; main applies the shared block calibration.
    assert all(c["state"] == ("PASS" if state == "OFF" or failure == "healthy" else "FAIL")
               for c in result["classes"].values())
    assert all(c["failures"] == (0 if state == "OFF" or failure == "healthy" else 200) for c in result["classes"].values())


@pytest.mark.parametrize("table,change", [("summary_nodes", "source_ids='[2]'"),
    ("summary_nodes", "summary='rewritten'"), ("summary_node_provenance", "model='changed'")])
def test_integrity_protects_corpus_provenance(corpus, tmp_path, table, change):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO summary_node_provenance VALUES(1,0,'original',1700000000)")
    with audit.connect(db) as c:
        manifest["preimage"] = audit.preimage(c)
    a = NS(db=db, root=tmp_path)
    # Trace growth is legitimate and must be excluded from the corpus digest.
    mod("dag").SummaryDAG(db).add_node(mod("dag").SummaryNode(session_id="trace", summary="trace", source_ids=[], token_count=1))
    measurement.integrity(a, manifest)
    with sqlite3.connect(db) as c:
        c.execute(f"UPDATE {table} SET {change} WHERE node_id=1")
    with pytest.raises(RuntimeError, match="provenance corruption"):
        measurement.integrity(a, manifest)


def test_turns_invalid_has_reason(tmp_path, monkeypatch):
    a = NS(out=tmp_path, seed=1, blocks=2, samples=600, queries=200, smoke=False,
           label="Mac", shape="short", baseline="empty")
    monkeypatch.setattr(turns, "prepare", lambda _: {})
    monkeypatch.setattr(turns, "cell", lambda *args: dict(catching_up=False, catchup_activity=True,
        compression_turns=1, bind_ms=1, p90=dict(ingest_preflight_ms=1, dispatch_ms=1)))
    turns.main(a)
    assert common.load(tmp_path/"summary.json")["reason"] == "on_coverage_completed_during_block"


@pytest.mark.parametrize("original_error", [False, True])
def test_final_sample_failure_stops_workers(fake_rig, monkeypatch, tmp_path, original_error):
    a, _, _, _, launched, _ = fake_rig
    monkeypatch.setattr(cpu, "swap", lambda: 0)
    meter = cpu.Supervisor(tmp_path/"cpu.jsonl")
    meter.thread = NS(join=lambda: None)
    monkeypatch.setattr(meter, "start", lambda: None)
    def fail(*args):
        raise OSError("final sample")
    monkeypatch.setattr(meter, "sample", fail)
    monkeypatch.setattr(rig, "supervisor", lambda *args: meter)
    stops = []
    monkeypatch.setattr(rig, "stop", lambda proc: stops.append(proc.pid) or dict(stop_killed=False))
    if original_error:
        def broken(*args, **kw):
            raise ValueError("original failure")
        monkeypatch.setattr(rig, "read_progress", broken)
        with pytest.raises(ValueError, match="original failure"):
            rig.main()
    else:
        rig.main()
        assert common.load(a.out/"summary.json")["state"] != "COMPLETE"
    assert {p for p in meter.windows} <= set(stops)
    assert launched


@pytest.mark.parametrize("name", ["sendmsg", "gethostbyaddr", "getnameinfo"])
def test_extra_socket_paths_denied(tmp_path, name):
    provider = NS(_FASTEMBED_THREADS=2, FastembedProvider=NS(_construct=lambda *a, **kw: None))
    command = NS(_chunk_backfill_run=lambda: {}, _embedding_backfill_summary_run=lambda: {})
    events = tmp_path/"events.jsonl"
    common.guards(lambda name: provider if name == "embedding_provider" else command, events)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(RuntimeError, match="offline instrument"):
            if name == "sendmsg":
                sock.sendmsg([b"probe"], [], 0, ("127.0.0.1", 9))
            elif name == "gethostbyaddr":
                socket.gethostbyaddr("127.0.0.1")
            else:
                socket.getnameinfo(("127.0.0.1", 9), 0)
    assert common.load(events)["violation"] == "network_attempt"


def test_subprocess_launch_denied_and_recorded_without_arguments(tmp_path):
    provider = NS(_FASTEMBED_THREADS=2, FastembedProvider=NS(_construct=lambda *a, **kw: None))
    command = NS(_chunk_backfill_run=lambda: {}, _embedding_backfill_summary_run=lambda: {})
    events = tmp_path/"events.jsonl"
    common.guards(lambda name: provider if name == "embedding_provider" else command, events)
    with pytest.raises(RuntimeError, match="offline subprocess launch"):
        subprocess.run([sys.executable, "-B", "-c", "pass"], check=True)
    assert common.load(events)["event"] == "subprocess_launch"
    assert common.load(events)["violation"] == "subprocess_launch"
    assert "args" not in common.load(events)


def test_screen_bind_default_is_bounded_and_invalid(fake_rig, monkeypatch):
    a, _, final, _, _, clock = fake_rig
    a.screen16, a.max_wall, a.stagger = True, 86400, 0
    final["complete"] = False
    launch = rig.launch
    defaults = {}
    def argument(name, **kw):
        defaults[name] = kw.get("default")
    parser = rig.parser(None)
    parser.add_argument = argument
    monkeypatch.setattr(rig, "parser", lambda _: parser)
    def launch_unbound(job, generation):
        proc = launch(job, generation)
        (job.out/"worker-0.json").unlink()
        return proc
    monkeypatch.setattr(rig, "launch", launch_unbound)
    rig.main()
    result = common.load(a.out/"summary.json")
    assert result["state"] == "INVALID" and result["reason"].startswith("bind_timeout:")
    assert clock[0] < 1100
    assert defaults["--bind-timeout"] == 900


@pytest.mark.parametrize("kind", ["raw", "provenance", "vector"])
def test_midrun_progress_detects_corruption(corpus, kind):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    with audit.connect(db) as c:
        manifest["preimage"] = audit.preimage(c)
    assert audit.progress(db, manifest, verify=True)["raw_unchanged"]
    with sqlite3.connect(db) as c:
        c.execute({"raw": "UPDATE messages SET content='changed' WHERE store_id=1",
                   "provenance": "UPDATE summary_nodes SET source_ids='[2]' WHERE node_id=1",
                   "vector": "UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'"}[kind])
    result = audit.progress(db, manifest, verify=True)
    assert not result["raw_unchanged"] or result["changed_vectors"]


@pytest.mark.parametrize("table,key", [("lcm_chunk_vectors", "chunk_id='1:0'"), ("lcm_embedding_vectors", "embedded_id='1'")])
def test_published_vector_cannot_change_after_audit(corpus, table, key):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    assert audit.audit(db, manifest)["complete"]
    with sqlite3.connect(db) as c:
        c.execute(f"UPDATE {table} SET vec=zeroblob(1536) WHERE {key}")
    result = audit.audit(db, manifest)
    assert result["changed_vectors"] and not result["complete"]


def test_final_sample_error_cannot_pass_catchup(fake_catchup, monkeypatch):
    a, _, _, _, _, meter = fake_catchup
    from bench.instruments.local_catchup import catchup
    monkeypatch.setattr(meter, "close", lambda: setattr(meter, "breach", "sampler_error:OSError"))
    result = catchup.run(a)
    assert (result["state"], result["reason"]) == ("ABORTED", "sampler_error:OSError")


@pytest.mark.usefixtures("fake_store_disk_headroom")
def test_final_sample_error_cannot_pass_i2_cell(tmp_path, monkeypatch):
    a = NS(scratch=tmp_path/"scratch", out=tmp_path/"out", model_cache=tmp_path,
           lcmx_repo=tmp_path, hermes_source=tmp_path, smoke=True, label="Mac", max_seconds=1)
    a.scratch.mkdir()
    a.out.mkdir()
    sqlite3.connect(a.scratch/"baseline.db").close()
    meter = NS(breach=None, watch=lambda _: None, start=lambda: None, summary=lambda *args: {})
    meter.close = lambda: setattr(meter, "breach", "sampler_error:OSError")
    monkeypatch.setattr(measurement, "Supervisor", lambda *args: meter)
    monkeypatch.setattr(measurement, "stop", lambda *args: dict(stop_killed=False))
    def launch(*args, **kw):
        common.dump(a.out/"0-OFF"/"raw-summary.json", dict(state="PASS"))
        return NS(pid=1, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="sampler_error:OSError"):
        measurement.cell(a, "OFF", 0, "turns")


@pytest.mark.parametrize("reason", ["foreground preparation/backfill/join or network", "unrelated failure"])
def test_worker_guard_rejection_has_invalid_verdict(tmp_path, reason):
    request = tmp_path/"request.json"
    common.dump(request, dict(root=str(tmp_path)))
    def worker(_):
        raise RuntimeError(reason)
    def direct(fn, request):
        fn(request)
    invoke = getattr(measurement, "guarded_worker", direct)
    if reason == "unrelated failure":
        with pytest.raises(RuntimeError, match=reason):
            invoke(worker, request)
    else:
        invoke(worker, request)
        assert common.load(tmp_path/"raw-summary.json") == dict(
            state="INVALID", reason="foreground_preparation_backfill_join_or_network")


def test_turns_propagates_invalid_cell_without_latency_judgment(tmp_path, monkeypatch):
    a = NS(out=tmp_path, seed=1, blocks=1, samples=600, queries=200, smoke=True,
           label="Mac", shape="short", baseline="empty")
    monkeypatch.setattr(turns, "prepare", lambda _: {})
    monkeypatch.setattr(turns, "cell", lambda *args: dict(state="INVALID", reason="guard"))
    turns.main(a)
    result = common.load(tmp_path/"summary.json")
    assert result["state"] == "INVALID" and result["reason"] == "cell_invalid"
    assert "bars" not in result
