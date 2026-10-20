"""Failing-first regressions for the three #1662 instrument defects."""
from argparse import Namespace as NS
import sqlite3

import pytest

from bench.instruments.local_catchup import audit, common, measurement, recall, rig, turns
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit",
                  "bench.instruments.local_catchup.tests.test_fix_round2"]
SQL_CONNECT = sqlite3.connect


@pytest.mark.parametrize("corrupt", [True, False])
def test_rig_verifies_transient_vector_and_final_poll(corpus, fake_rig, monkeypatch, corrupt):
    db, manifest, mod = corpus
    a, _, _, _, _, _ = fake_rig
    a.processes, a.rows, a.max_wall, a.restart_fraction = 1, 8, 700, 2
    monkeypatch.setattr(sqlite3, "connect", SQL_CONNECT)
    monkeypatch.setattr(rig, "connect", audit.connect)
    def build(repo, host, target, *args):
        with SQL_CONNECT(db) as src, SQL_CONNECT(target) as dst:
            src.backup(dst)
        fill(target, manifest, mod)
        return manifest
    monkeypatch.setattr(rig, "build", build)
    calls, restored = [], []
    original = None
    def poll(target, m, verify=False):
        nonlocal original
        if len(calls) == 1 and corrupt:
            with SQL_CONNECT(target) as c:
                original = c.execute("SELECT vec FROM lcm_chunk_vectors WHERE chunk_id='1:0'").fetchone()[0]
                c.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'")
        calls.append(verify)
        result = dict(audit.progress(target, m, verify), complete=False)
        if original is not None and verify:
            with SQL_CONNECT(target) as c:
                c.execute("UPDATE lcm_chunk_vectors SET vec=? WHERE chunk_id='1:0'", (original,))
            original = None
        return result
    def final(target, m):
        result = audit.audit(target, m)
        restored.append(result)
        return result
    monkeypatch.setattr(rig, "read_progress", poll)
    monkeypatch.setattr(rig, "audit", final)
    rig.main()
    result = common.load(a.out/"summary.json")
    assert all(r["complete"] and not r["changed_vectors"] for r in restored)
    if corrupt:
        n = result["verify_every_polls"]
        assert len(calls) == n+1 and calls[-2:] == [True, True]
        assert result["reason"] == "row_or_vector_corruption"
        assert result["state"] != "COMPLETE"
    else:
        assert calls[-1] is True  # Last progress read before the final audit.
        n = result["verify_every_polls"]
        assert n > 0 and all(calls[i-1] for i in range(n, len(calls), n))
        assert result["verified_read_cpu_s_total"] >= result["verified_read_cpu_s_max_per_read"] >= 0


GUARDS = [
    ("foreground/download preparation forbidden", {"event": "prepare", "foreground": True}, "foreground_preparation_forbidden"),
    ("foreground backfill forbidden", {"event": "_chunk_backfill_run", "foreground": True}, "foreground_backfill_forbidden"),
    ("foreground embedding join forbidden", {"violation": "foreground_join"}, "foreground_embedding_join_forbidden"),
    ("offline instrument denies network", {"violation": "network_attempt"}, "offline_network_forbidden"),
    ("offline subprocess launch", {"violation": "subprocess_launch"}, "offline_subprocess_launch"),
]


@pytest.mark.parametrize("message,event,reason", GUARDS)
@pytest.mark.parametrize("source", ["message", "event"])
def test_guard_violation_is_invalid(tmp_path, message, event, reason, source):
    request = tmp_path/"request.json"
    common.dump(request, dict(root=str(tmp_path)))
    def worker(_):
        if source == "event":
            common.append(tmp_path/"events.jsonl", event)
        raise RuntimeError(message if source == "message" else "foreground preparation/backfill/join or network")
    measurement.guarded_worker(worker, request)
    assert common.load(tmp_path/"raw-summary.json") == dict(state="INVALID", reason=reason)


@pytest.mark.parametrize("experiment", [turns, recall], ids=["turns", "recall"])
def test_synchronous_provider_bind_is_invalid_and_experiment_continues(tmp_path, monkeypatch, experiment):
    provider = NS(_FASTEMBED_THREADS=2, FastembedProvider=NS(_construct=lambda *a, **kw: None))
    command = NS(_chunk_backfill_run=lambda: {}, _embedding_backfill_summary_run=lambda: {})
    common.guards(lambda name: provider if name == "embedding_provider" else command,
                  tmp_path/"events.jsonl")
    def bind(_):
        provider.FastembedProvider._construct(None, allow_download=False)
    results = []
    def cell(a, state, block, kind):
        root = tmp_path/f"{block}-{state}"
        root.mkdir()
        request = root/"request.json"
        common.dump(request, dict(root=str(root)))
        measurement.guarded_worker(experiment.worker, request)
        result = common.load(root/"raw-summary.json")
        results.append(result)
        return result
    monkeypatch.setattr(experiment, "runtime", bind)
    monkeypatch.setattr(experiment, "cell", cell)
    monkeypatch.setattr(experiment, "prepare", lambda a: dict(canaries=[{}]))
    experiment.main(NS(out=tmp_path, seed=1586, blocks=2, queries=1, smoke=True,
                       shape="short", baseline="empty"))
    assert len(results) == (4 if experiment is turns else 6)
    assert all(r == dict(state="INVALID", reason="foreground_preparation_forbidden") for r in results)
    assert common.load(tmp_path/"summary.json")["state"] == "INVALID"


@pytest.mark.parametrize("old_event", [None, {"violation": "network_attempt", "detail": "préexisting"}])
def test_unknown_runtime_error_still_raises(tmp_path, old_event):
    request = tmp_path/"request.json"
    common.dump(request, dict(root=str(tmp_path)))
    if old_event is not None:
        common.append(tmp_path/"events.jsonl", old_event)
    def worker(_):
        raise RuntimeError("unrelated bug")
    with pytest.raises(RuntimeError, match="unrelated bug"):
        measurement.guarded_worker(worker, request)


@pytest.mark.parametrize("degraded", [False, None, 1, "true", True])
def test_empty_arm_requires_degraded_true(degraded):
    payload = dict(degraded=degraded, degraded_reason="chunk semantic index catching up",
                   provenance=dict(coverage=dict(chunk="none", summary="full"), arms_run=["summary"]))
    assert recall.empty_arms(payload, ["chunk"]) == (["chunk"] if degraded is True else [])
    assert recall.semantic_healthy(payload, ["chunk"]) is (degraded is True)
