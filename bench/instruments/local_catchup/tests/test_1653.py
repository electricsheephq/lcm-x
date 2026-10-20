"""Twelve bounded instrument regressions; synthetic clocks, fresh stores, no network."""
from argparse import Namespace
import json
import os
import socket
import sqlite3

import psutil
import pytest

from bench.instruments.local_catchup import audit, catchup, common, cpu, measurement, probes, recall, rig, turns
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_fix_round3",
                  "bench.instruments.local_catchup.tests.test_corpus_audit"]


def test_01_screen_has_full_post_bind_window(fake_rig, monkeypatch):
    a, checked, final, _, _, clock = fake_rig
    a.screen16, a.stagger = True, 2
    clock[0] = .25  # Full rolling windows must also work off the integer grid.
    final["complete"] = False
    monkeypatch.setattr(rig, "read_progress", lambda *args, **kw: checked.copy())
    meter = rig.supervisor()
    def summary(*args, **kw):
        lo, hi = kw["aggregate_window"]
        samples = [(lo+i, 1, .2) for i in range(1, int(hi-lo)+1)]
        return dict(per_process={str(p): dict(cpu.statistics(samples), peak_rss=1) for p in meter.windows},
                    end=kw["end"], aggregate_window=[lo, hi])
    monkeypatch.setattr(meter, "summary", summary)
    rig.main()
    result = common.load(a.out/"summary.json")
    assert result["cpu"]["aggregate_window"] == [32.25, 92.25]
    assert all(r["p95_rolling60_cores"] == pytest.approx(.2)
               for r in result["cpu"]["per_process"].values())
    assert result["state"] == "INCOMPLETE" and clock[0] > 92


def test_02_restart_does_not_reuse_old_prepared(fake_rig, monkeypatch):
    a, checked, final, _, launched, _ = fake_rig
    a.max_wall = 700
    final["complete"] = False
    monkeypatch.setattr(rig, "read_progress", lambda *args, **kw: checked.copy())
    launch = rig.launch
    def prepared(job, generation):
        p = launch(job, generation)
        if generation == 0:
            common.append(job.out/"events.jsonl", dict(event="prepared", generation=0))
        return p
    monkeypatch.setattr(rig, "launch", prepared)
    rig.main()
    result = common.load(a.out/"summary.json")
    assert [g for _, g in launched] == [0, 0, 1, 1]
    assert (result["state"], result["reason"]) == ("INCOMPLETE", "max_wall")


@pytest.mark.parametrize("table,change", [
    ("summary_nodes", "source_ids='[2]'"),
    ("summary_node_provenance", "model='changed'")])
def test_03_summary_provenance_is_immutable(corpus, table, change):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO summary_node_provenance VALUES(1,0,'original',1700000000)")
    with audit.connect(db) as c:
        manifest["preimage"] = audit.preimage(c)
    with sqlite3.connect(db) as c:
        c.execute(f"UPDATE {table} SET {change} WHERE node_id=1")
    result = audit.audit(db, manifest)
    assert catchup.audit_failed(result) and not result["complete"]


def test_03_lazy_empty_source_table_preserves_provenance(corpus):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    with sqlite3.connect(db) as c:
        mod("db_bootstrap").ensure_temporal_rollup_tables(c)
    assert audit.audit(db, manifest)["complete"]
    # Once a mapping exists, changing it must still invalidate the preimage.
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO lcm_rollup_sources VALUES(1,1)")
    with audit.connect(db) as c:
        manifest["preimage"] = audit.preimage(c)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE lcm_rollup_sources SET node_id=2")
    assert catchup.audit_failed(audit.audit(db, manifest))


@pytest.mark.parametrize("build_failure", [False, True])
def test_04_close_before_start_and_build_exception(fake_rig, monkeypatch, tmp_path, build_failure):
    monkeypatch.setattr(cpu, "swap", lambda: 0)
    meter = cpu.Supervisor(tmp_path/"cpu.jsonl")
    if not build_failure:
        meter.close()
        return
    monkeypatch.setattr(rig, "supervisor", lambda *args: meter)
    build = rig.build
    def broken(*args):
        if args[2].name == "store.db":
            raise ValueError("original build failure")
        return build(*args)
    monkeypatch.setattr(rig, "build", broken)
    with pytest.raises(ValueError, match="original build failure"):
        rig.main()


def test_05_both_cold_sides_perform_first_bind(tmp_path, monkeypatch):
    for name in ("HOME", "CFFIXED_USER_HOME"):
        monkeypatch.setenv(name, os.environ.get(name, str(tmp_path)))
    a = Namespace(db=tmp_path/"base.db", home=tmp_path/"homes", out=tmp_path/"out",
                  manifest=tmp_path/"manifest.json", lcmx_repo=tmp_path, hermes_source=tmp_path)
    with sqlite3.connect(a.db) as c:
        c.execute("CREATE TABLE binds(session TEXT)")
    common.dump(a.manifest, dict(sessions=["session"], canaries=[dict(id="canary")]))
    monkeypatch.setattr(probes, "isolated", lambda home, cache: home.mkdir(parents=True))
    monkeypatch.setattr(probes, "guards", lambda mod, events: events.touch())
    monkeypatch.setattr(probes, "config", lambda mod, db, enabled: (db, enabled))
    binds, processes = [], []
    class Engine:
        def __init__(self, config, hermes_home):
            self.db, self.enabled = config
            self.home = hermes_home
        def on_session_start(self, session, **kw):
            with sqlite3.connect(self.db) as c:
                fresh = c.execute("SELECT count(*) FROM binds").fetchone()[0] == 0
                c.execute("INSERT INTO binds VALUES(?)", (session,))
            binds.append((self.enabled, fresh, self.db, self.home))
        def shutdown(self):
            pass
    monkeypatch.setattr(probes, "product", lambda *args: lambda name: {
        "engine": Namespace(LCMEngine=Engine), "tools": Namespace(lcm_grep=lambda *a, **kw:
            json.dumps(dict(degraded_to_fts=True, total_results=1)))}[name])
    class Process:
        def __init__(self, target, args):
            self.target, self.args, self.exitcode = target, args, 0
        def start(self):
            processes.append(self)
            self.target(*self.args)
        def join(self):
            pass
    # Exercise the real side implementation, substituting only the spawn boundary.
    if hasattr(probes, "multiprocessing"):
        monkeypatch.setattr(probes.multiprocessing, "get_context", lambda mode: Namespace(Process=Process))
    probes.cold(a)
    assert [(enabled, fresh) for enabled, fresh, _, _ in binds] == [(False, True), (True, True)]
    assert len(processes) == 2
    assert binds[0][2:] != binds[1][2:]


@pytest.mark.parametrize("path", ["connect", "connect_ex", "getaddrinfo", "gethostbyname",
                                  "gethostbyname_ex", "sendto"])
def test_06_offline_paths_are_denied_and_recorded(tmp_path, monkeypatch, path):
    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, lambda *args: pytest.fail("network syscall escaped"))
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        monkeypatch.setattr(socket, name, lambda *args: pytest.fail("resolver escaped"))
    provider = Namespace(_FASTEMBED_THREADS=2, FastembedProvider=Namespace(_construct=lambda *a, **kw: None))
    command = Namespace(_chunk_backfill_run=lambda: {}, _embedding_backfill_summary_run=lambda: {})
    events = tmp_path/"events.jsonl"
    common.guards(lambda name: provider if name == "embedding_provider" else command, events)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        call = getattr(sock, path) if hasattr(sock, path) else getattr(socket, path)
        with pytest.raises(RuntimeError, match="offline instrument denies network"):
            call("fixture.invalid")
    assert common.load(events)["violation"] == "network_attempt"


def test_07_missing_cpu_stays_missing(fake_rig, monkeypatch):
    a = fake_rig[0]
    meter = rig.supervisor()
    monkeypatch.setattr(meter, "summary", lambda *args, **kw: dict(per_process={
        "1": dict(mean_cores=.1, p95_rolling60_cores=None, peak1s_cores=.2, peak_rss=10)}))
    rig.main()
    assert common.load(a.out/"summary.json")["worst_process_cpu_rss"]["p95_rolling60_cores"] is None


def test_08_audit_reads_one_snapshot(corpus, monkeypatch):
    db, manifest, mod = corpus
    original = audit.preimage
    def concurrent(c):
        # Commit a complete paired batch after the first vector SELECT, before metadata.
        fired = False
        def between(sql):
            nonlocal fired
            if not fired and sql == "SELECT * FROM lcm_chunk_meta":
                fired = True
                fill(db, manifest, mod)
        c.set_trace_callback(between)
        return original(c)
    monkeypatch.setattr(audit, "preimage", concurrent)
    result = audit.audit(db, manifest)
    assert not result["unpaired_rows"]
    assert result["classes"]["chunks"]["covered"] == 0
    monkeypatch.setattr(audit, "preimage", original)
    assert audit.audit(db, manifest)["complete"]


@pytest.mark.parametrize("table,column", [
    ("lcm_chunk_meta", "store_id"), ("lcm_chunk_meta", "chunk_index"),
    ("lcm_chunk_meta", "char_start"), ("lcm_chunk_meta", "char_end"),
    ("lcm_chunk_meta", "token_estimate"), ("lcm_embedding_meta", "source_token_count")])
def test_09_new_metadata_matches_manifest(corpus, table, column):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    assert audit.audit(db, manifest)["complete"]
    with sqlite3.connect(db) as c:
        c.execute(f"UPDATE {table} SET {column}={column}+1")
    result = audit.audit(db, manifest)
    assert catchup.audit_failed(result) and not result["complete"]


@pytest.mark.parametrize("observer", ["catchup", "rig"])
@pytest.mark.parametrize("error", [psutil.NoSuchProcess, psutil.AccessDenied])
def test_10_open_file_exit_finishes_audit_summary(fake_catchup, monkeypatch, observer, error):
    a, _, _, _, _, _ = fake_catchup
    module = catchup if observer == "catchup" else rig
    if observer == "rig":
        a.scratch = a.scratch/"rig"
    def gone():
        raise error(1)
    monkeypatch.setattr(psutil, "Process", lambda _: Namespace(open_files=gone))
    monkeypatch.setattr(module, "foreign_files", common.foreign_files)
    if observer == "catchup":
        module.run(a)
    else:
        module.main()
    summary = common.load(a.out/"summary.json")
    assert summary["state"] == "ABORTED" and summary["reason"].startswith("worker_exit")
    final = a.out/"final-audit.json" if observer == "catchup" else a.out/"0"/"final-audit.json"
    assert final.exists()


def test_11_trace_only_backfill_does_not_count(tmp_path):
    common.append(tmp_path/"events.jsonl", dict(event="_chunk_backfill_run", at=3,
        end=4, foreground=False, embedded=32))
    result = measurement.catchup_activity(tmp_path, 2, 5, {"chunks": 0}, {"chunks": 0})
    assert result["active_backfills"] == 1 and result["coverage_delta"] == 0
    assert result["catchup_activity"] is False


@pytest.mark.parametrize("module", [turns, recall])
def test_12_mac_summary_has_corpus_qualifiers(tmp_path, monkeypatch, module):
    a = Namespace(out=tmp_path, seed=1586, blocks=1, samples=10, queries=1,
                  smoke=True, label="Mac", shape="mixed", baseline="upgrade")
    monkeypatch.setattr(module, "prepare", lambda a: {"canaries": [dict(id="canary")]})
    monkeypatch.setattr(module, "cell", lambda *args: dict(state="INVALID", bind_ms=1,
        p90=dict(ingest_preflight_ms=1, dispatch_ms=1), catching_up=True,
        catchup_activity=False, compression_turns=1))
    module.main(a)
    result = common.load(tmp_path/"summary.json")
    assert (result.get("shape"), result.get("baseline")) == ("mixed", "upgrade")
