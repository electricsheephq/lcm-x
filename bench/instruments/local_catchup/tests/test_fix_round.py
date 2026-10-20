"""Named regressions for SPEC I1-fix, no model construction."""
from argparse import Namespace
import builtins
import json
import time

import pytest

from bench.instruments.local_catchup import audit, catchup, common, corpus, cpu, probes, rig
from test_corpus_audit import REPO, fill
import test_corpus_audit as helpers


@pytest.fixture
def fixture_corpus(tmp_path, monkeypatch):
    return helpers.corpus.__wrapped__(tmp_path, monkeypatch)


def test_1_cadence_independent_of_observer(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "swap", lambda: 0)
    monkeypatch.setattr(cpu, "process", lambda _: (time.monotonic(), 1, 1))
    m = cpu.Supervisor(tmp_path/"cpu.jsonl")
    m.start()
    m.watch(1)
    time.sleep(2.15)  # Simulates a blocking audit in the observer.
    m.close()
    assert len(m.samples[1]) >= 2
    assert m.summary()["max_sample_dt"] <= 2
    assert len(cpu.rolling([(30, 30, 1), (62, 32, 0)])) == 3
    m.samples[1].append((65, 3, 0))
    assert m.summary()["per_process_bars"]["1"] == "INSUFFICIENT_CADENCE"


def test_2_expected_ids_reachable(fixture_corpus, tmp_path):
    _, _, mod = fixture_corpus
    db = tmp_path/"upgrade.db"
    manifest = corpus.build(REPO, REPO, db, 8, baseline="upgrade")
    with audit.connect(db) as c:
        reachable = {p.chunk_id for i, role, text in c.execute("SELECT store_id,role,content FROM messages")
                     for p in mod("chunking").chunk_message(i, role, text)}
    assert set(manifest["expected"]["chunks"]) <= reachable
    assert manifest["upgrade_long_chunks"] > 32


def test_3_active_windows_and_na(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu, "swap", lambda: 0)
    m = cpu.Supervisor(tmp_path/"cpu.jsonl")
    m.samples[1] = [(i, 1, 1 if i <= 60 else 0) for i in range(1, 121)]
    m.samples[2] = [(i, 1, .5) for i in range(1, 121)]
    m.aggregate = [(i, 1, 1.5) for i in range(1, 121)]
    m.windows = {1: [0, 60], 2: [10, 100]}
    s = m.summary("COMPLETE", 2)
    assert s["per_process"]["1"]["mean_cores"] == 1
    assert s["aggregate"]["seconds"] == 50
    assert s["tail"]["seconds"] == 60
    assert m.summary("STALLED", 2)["per_process_bars"]["1"] == "N/A"
    assert s["aggregate_bar"] == "NOT_APPLICABLE"


def test_4_restart_rule():
    a = Namespace(restart_fraction=.4, restart_after=None, max_wall=100)
    assert catchup.restart_trigger(a, .39, 24) is None
    assert catchup.restart_trigger(a, .4, 20) == "fraction"
    assert catchup.restart_trigger(a, .3, 25) == "declared_time_fallback"
    assert catchup.restart_trigger(a, 1., 90) is None


def test_5_product_defaults(fixture_corpus):
    db, _, mod = fixture_corpus
    cfg, default = common.config(mod, db), mod("config").LCMConfig()
    assert cfg.embedding_max_batch_items == default.embedding_max_batch_items
    assert cfg.temporal_rollups_enabled == default.temporal_rollups_enabled


@pytest.mark.parametrize("which", ["db", "home"])
def test_6_guards_before_any_write(tmp_path, which):
    a = Namespace(out=tmp_path/"out", home=tmp_path/"home", db=tmp_path/"db")
    setattr(a, which, tmp_path/".hermes"/which)
    with pytest.raises(ValueError):
        probes.cold(a)
    assert not a.out.exists() and not (tmp_path/"absent-model-cache").exists()


def test_6_open_file_allowlist(tmp_path, monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, "Process", lambda _: Namespace(open_files=lambda: [Namespace(path=str(tmp_path/"other"))]))
    assert common.foreign_files(1, tmp_path/"home", tmp_path/"store.db", tmp_path/"cache")
    monkeypatch.setattr(psutil, "Process", lambda _: Namespace(open_files=lambda: [Namespace(path=str(tmp_path/"store.db-wal"))]))
    assert not common.foreign_files(1, tmp_path/"home", tmp_path/"store.db", tmp_path/"cache")


def test_7_progress_has_no_hash_or_integrity_scan(fixture_corpus, monkeypatch):
    db, manifest, _ = fixture_corpus
    monkeypatch.setattr(audit, "preimage", lambda _: pytest.fail("progress hashes full store"))
    r = audit.progress(db, manifest)
    assert not r["complete"] and r["classes"]["chunks"]["covered"] == 0


def test_8_reject_reused_store(fixture_corpus):
    db, manifest, mod = fixture_corpus
    audit.baseline(db, manifest)
    fill(db, manifest, mod)
    with pytest.raises(ValueError, match="baseline"):
        audit.baseline(db, manifest)


def test_9_mixed_pilot_covers_rotation():
    rows = corpus.pilot_rows(1000, 250000, "mixed")
    assert {(20, 300, 1200)[i % 3] for i, _ in enumerate(range(0, rows//2, 250))} == {20, 300, 1200}


def test_10_screen_requires_running_backlogs():
    assert rig.screen_state([.5]*16) == "INCOMPLETE"
    assert rig.screen_state([.5]*15+[1.]) == "INVALID"


def test_11_psutil_mandatory_and_rescore(tmp_path, monkeypatch):
    original = builtins.__import__
    def importing(name, *args, **kw):
        if name == "psutil":
            raise ImportError("mandatory")
        return original(name, *args, **kw)
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ImportError):
        cpu.process(1)
    from bench.instruments.local_catchup.rescore import verdict
    for elapsed, expected in [(599, "RUNNING"), (600, "STALLED")]:
        rows = [{"elapsed_s": t, "coverage_fraction": .5, "audit": {"complete": False, "scan": [], "classes": {"chunks": {"covered": 1}}},
                 "ready": True} for t in (0, elapsed)]
        path = tmp_path/"progress.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        assert verdict(path, 1200) == expected
