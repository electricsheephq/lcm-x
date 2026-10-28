"""Regressions from the LOCAL pacing review: whole messages and live owners."""
import dataclasses
import sqlite3
import threading

import pytest

from hermes_lcm import command, embedding_maintenance as maintenance
from hermes_lcm import engine as engines, tools
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from tests.test_issue_1014_incremental_embedding import FakeProvider
from tests.test_issue_1586_local_pacing import Clock, REAL_PREPARE
from tests.test_issue_1586_local_pacing import local_store as local_store


@pytest.mark.parametrize("slice_s,repeats,batch_size,slow", [
    (0.0, 190, 32, False),  # Unsliced precursor control.
    (1.0, 190, 32, False),  # A message straddles the 32-row selection cap.
    (1.0, 4000, 32, False),  # The first gap message alone exceeds the cap.
    (1.0, 190, 2, True),  # Budget expires between batches of one gap message.
])
def test_gap_slices_embed_every_chunk(local_store, monkeypatch, slice_s, repeats, batch_size, slow):
    engine, _ = local_store
    config = dataclasses.replace(engine._config, embedding_max_batch_items=batch_size)
    db = engine._store.db_path
    for i in range(22 if repeats == 190 else 3):
        body = (f"gap message {i} sentence. " * repeats) if i else "low boundary " * 60
        engine._store.append("seed", {"role": "assistant", "content": body})
    if repeats == 4000:
        # Seed a synthetic pre-existing long message without the ingest quarantine.
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE messages SET content=? WHERE store_id=2", ("gap message 1 sentence. " * repeats,))
    # Establish cursor bounds and complete coverage, then reproduce the review's
    # unstamped legacy gap without altering the stored messages themselves.
    result = command._chunk_backfill_run(config, db, apply=True, limit=10000,
                                         retry_uncertain=False, policy="", incremental=False)
    assert isinstance(result, dict) and not result["remaining"]
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall()
        expected = {c.chunk_id for row in rows for c in command.chunk_message(*row, policy="conversational")}
        middle = [row[0] for row in rows[1:-1]]
        assert len(command.chunk_message(*rows[1], policy="conversational")) > (32 if repeats == 4000 else 2)
        placeholders = ",".join("?" for _ in middle)
        conn.execute(f"DELETE FROM lcm_chunk_vectors WHERE chunk_id IN (SELECT chunk_id FROM lcm_chunk_meta WHERE store_id IN ({placeholders}))", middle)
        conn.execute(f"DELETE FROM lcm_chunk_meta WHERE store_id IN ({placeholders})", middle)
        conn.execute("DELETE FROM metadata WHERE key LIKE 'chunk_policy_revision%' OR key LIKE 'chunk_gapfill_scan%'")
    clock = Clock()
    batches = []

    class SlowProvider(FakeProvider):
        def embed_document_batches(self, texts, **kwargs):
            batches.append(len(texts))
            if slow:
                clock.now += 2.0
            yield from super().embed_document_batches(texts, **kwargs)

    monkeypatch.setattr(command, "time", clock)
    monkeypatch.setattr(command, "resolve_provider", lambda c, **_k: SlowProvider(c.embedding_provider, c.embedding_model))
    for _ in range(40):
        result = command._chunk_backfill_run(config, db, apply=True, limit=32,
                                             retry_uncertain=False, policy="", incremental=True,
                                             maintenance_slice=slice_s)
        assert isinstance(result, dict) and not result["failed"], result
    with sqlite3.connect(db) as conn:
        actual = {row[0] for row in conn.execute("SELECT chunk_id FROM lcm_chunk_meta")}
    assert actual == expected, f"permanently missing: {sorted(expected - actual)}"
    assert batches and max(batches) <= batch_size


@pytest.mark.parametrize("position", ["queued", "deferred", "active", "active-continuation"])
def test_retirement_transfers_chain_to_live_peer(local_store, monkeypatch, position):
    engine, scheduler = local_store
    engine._config = dataclasses.replace(engine._config, embedding_max_batch_items=17)
    peer_config = dataclasses.replace(engine._config, embedding_max_batch_items=32)
    peer = LCMEngine(config=peer_config, hermes_home=str(engine._hermes_home))
    for i in range(200):
        engine._dag.add_node(SummaryNode(session_id="seed", depth=0, summary=f"peer leaf {i}",
                                         created_at=float(i), latest_at=float(i)))
    entered, release = threading.Event(), threading.Event()
    calls = []
    handoff_delays, retired_slice_delays = [], []
    real_pass = engines.run_incremental_embedding_pass
    real_schedule = scheduler.schedule

    def scheduled(key, job, **kwargs):
        if engine._embedding_maintenance_retired and kwargs.get("owner") is peer._embedding_maintenance_owner:
            handoff_delays.append(kwargs.get("delay", 0.0))
        return real_schedule(key, job, **kwargs)

    def measured(db, config, **kwargs):
        calls.append((config.embedding_max_batch_items, engine._embedding_maintenance_retired))
        if position.startswith("active") and len(calls) == 1:
            entered.set()
            assert release.wait(5)
        result = real_pass(db, config, **kwargs)
        if config.embedding_max_batch_items == 17 and engine._embedding_maintenance_retired:
            retired_slice_delays.append(result["delay"])
        return result

    monkeypatch.setattr(engines, "run_incremental_embedding_pass", measured)
    monkeypatch.setattr(scheduler, "schedule", scheduled)
    try:
        if position.startswith("active"):
            with scheduler._condition:
                engine._schedule_embedding_maintenance()
                if position == "active-continuation":
                    peer._schedule_embedding_maintenance()  # Dedup before A starts, with no follow-up slot.
            assert entered.wait(3)
            if position == "active":
                peer._schedule_embedding_maintenance()
            engine._retire_embedding_maintenance(wait=False)
            release.set()
        else:
            with scheduler._condition:
                if position == "deferred":
                    scheduler._max_pending_jobs = 1
                    scheduler.schedule(("occupied", "embeddings"), lambda: None)
                engine._schedule_embedding_maintenance()
                peer._schedule_embedding_maintenance()
                engine._retire_embedding_maintenance(wait=True)
        # A slow SQLite pass also earns the LOCAL multiplier times its wall time as an idle delay.
        # Keep this a handoff/coverage check rather than a shared-runner speed bar.
        if not peer.drain_embedding_maintenance(60):
            with scheduler._condition:
                state = dict(
                    queued_keys=set(scheduler._queued_keys),
                    deferred_jobs=list(scheduler._deferred_jobs),
                    active_keys=set(scheduler._active_keys),
                    follow_up_key=scheduler._follow_up_key,
                    handoff_jobs={key: [id(owner) for owner in owners]
                                  for key, owners in scheduler._handoff_jobs.items()},
                    owned_keys={id(owner): set(keys)
                                for owner, keys in scheduler._owned_keys.items()},
                    not_before=dict(scheduler._not_before),
                    worker_alive=scheduler._worker is not None and scheduler._worker.is_alive(),
                    retired_owner=id(engine._embedding_maintenance_owner),
                    peer_owner=id(peer._embedding_maintenance_owner),
                    retired=engine._embedding_maintenance_retired,
                    peer_retired=peer._embedding_maintenance_retired,
                )
            pytest.fail(f"peer maintenance did not drain in 60s: {state!r}; calls={calls!r}")
        with sqlite3.connect(engine._store.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM lcm_embedding_meta").fetchone()[0] == 200
        assert calls and all(size == 32 for size, retired in calls if retired)
        assert not getattr(scheduler, "_handoff_jobs", {})
        if not position.startswith("active"):
            assert all(size == 32 for size, _ in calls), "retired owner's pass ran"
        if position == "active-continuation":
            assert handoff_delays[0] >= retired_slice_delays[0], "handoff bypassed the slice's idle delay"
    finally:
        release.set()
        peer.shutdown()


def test_retired_maintain_does_not_restore_prepare_callback(local_store, monkeypatch):
    engine, scheduler = local_store
    jobs, passes = [], []
    monkeypatch.setattr(scheduler, "schedule", lambda key, job, **kw: jobs.append(job) or True)
    key = maintenance._local_model_key(engine._config)
    maintenance._PREPARE[key] = ("unavailable", "offline fixture")
    maintenance._PREPARE_RETRY[key] = (5, float("inf"))
    monkeypatch.setattr(engines, "run_incremental_embedding_pass", lambda *a, **kw: passes.append(1))
    with pytest.MonkeyPatch.context() as prepare_patch:
        prepare_patch.setattr(maintenance, "start_local_model_prepare", REAL_PREPARE)
        engine._schedule_embedding_maintenance()
        owner = engine._embedding_maintenance_owner
        assert maintenance._PREPARE_READY[key]
        engine._retire_embedding_maintenance(wait=True)
        assert all(token is not owner for _, token in maintenance._PREPARE_READY[key])
        jobs[0]()  # A popped/stale closure races retirement.
    assert all(token is not owner for _, token in maintenance._PREPARE_READY[key])
    assert not passes


@pytest.mark.parametrize("summary_pending,chunk_pending", [(0, 0), (1, 0), (0, 1)])
def test_terminal_reason_distinguishes_complete_from_stalled(local_store, monkeypatch,
                                                           summary_pending, chunk_pending):
    engine, _ = local_store

    def result(remaining):
        return dict(status="complete", selected=0, embedded=0, privacy_withheld=0,
                    skipped=[], failed=[], error=None, stop_reason=None,
                    remaining=remaining, uncertain=0)

    monkeypatch.setattr(command, "_embedding_backfill_summary_run", lambda *a, **kw: result(summary_pending))
    monkeypatch.setattr(command, "_chunk_backfill_run", lambda *a, **kw: result(chunk_pending))
    state = {}
    for _ in range(4):
        last = maintenance.run_incremental_embedding_pass(engine._store.db_path, engine._config, slice_state=state)
    assert not last["continue"]
    reason = "complete" if summary_pending == chunk_pending == 0 else "no_progress"
    assert last["outcome"].split()[-1] == reason
    assert last["outcome"] in command.handle_lcm_command("status", engine)
    readiness = tools._embedding_readiness(engine, {"available": True, "model": engine._config.embedding_model})
    assert readiness.endswith(reason)
