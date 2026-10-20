"""#1656: summary slices progress and duplicate triggers preserve pacing."""
import dataclasses
from types import SimpleNamespace

import pytest

from hermes_lcm import command, embedding_maintenance as maintenance, engine as engines
from hermes_lcm.dag import SummaryNode
from tests.test_issue_1014_incremental_embedding import FakeProvider
from tests.test_issue_1586_local_pacing import Clock
from tests.test_issue_1586_local_pacing import local_store as local_store


def test_slow_summary_discovery_drains_across_slices(local_store, monkeypatch):
    engine, _ = local_store
    config = dataclasses.replace(engine._config, embedding_max_batch_items=2)
    for i in range(5):
        engine._dag.add_node(SummaryNode(session_id="seed", depth=0,
                                        summary=f"slow summary {i}", created_at=i, latest_at=i))
    clock = Clock()
    pending_rows = command._embedding_pending_rows
    summary_run = command._embedding_backfill_summary_run
    slices = []

    def slow(*args, **kwargs):
        result = pending_rows(*args, **kwargs)
        clock.now += 2.0
        return result

    def measured(*args, **kwargs):
        result = summary_run(*args, **kwargs)
        slices.append(result)
        return result

    monkeypatch.setattr(command, "time", clock)
    monkeypatch.setattr(maintenance, "time", clock)
    monkeypatch.setattr(command, "_embedding_pending_rows", slow)
    monkeypatch.setattr(command, "_embedding_backfill_summary_run", measured)
    state = {}
    for _ in range(40):
        result = maintenance.run_incremental_embedding_pass(engine._store.db_path, config, slice_state=state)
        if not result["continue"]:
            break
    selected = [s for s in slices if s["selected"]]
    assert selected and all(s["embedded"] > 0 and not s["failed"] for s in selected)
    assert [s["embedded"] for s in selected] == [2, 2, 1]
    assert selected[0]["stop_reason"] is None  # The slice selects only one batch.
    assert selected[0]["remaining"] == 3
    assert result["outcome"].endswith(" complete")
    assert selected[-1]["remaining"] == 0


@pytest.mark.parametrize("maintenance_slice", [0.0, 1.0])
@pytest.mark.parametrize("slow_discovery", [False, True])
def test_summary_budget_still_stops_later_batches(local_store, monkeypatch, maintenance_slice, slow_discovery):
    engine, _ = local_store
    config = dataclasses.replace(engine._config, embedding_max_batch_items=2)
    for i in range(3):
        engine._dag.add_node(SummaryNode(session_id="seed", depth=0,
                                        summary=f"budget summary {i}", created_at=i, latest_at=i))
    clock = Clock()
    pending_rows = command._embedding_pending_rows

    def discover(*args, **kwargs):
        result = pending_rows(*args, **kwargs)
        clock.now += 2.0 if slow_discovery else 0.0
        return result

    class SlowProvider(FakeProvider):
        def embed_document_batches(self, texts, **kwargs):
            clock.now += 2.0
            yield from super().embed_document_batches(texts, **kwargs)

    monkeypatch.setattr(command, "time", clock)
    monkeypatch.setattr(command, "_embedding_pending_rows", discover)
    monkeypatch.setattr(command, "_embedding_backfill_budget_s", lambda: 1.0)
    monkeypatch.setattr(command, "resolve_provider", lambda c, **kw: SlowProvider(c.embedding_provider, c.embedding_model))
    result = command._embedding_backfill_summary_run(
        config, engine._store.db_path, apply=True, limit=10,
        retry_uncertain=False, maintenance_slice=maintenance_slice,
    )
    embedded = 0 if slow_discovery and not maintenance_slice else 2
    assert result["embedded"] == embedded
    assert result["remaining"] == 3 - embedded
    assert result["stop_reason"] == "op_budget_exhausted"
    assert result["failed"] == []


@pytest.fixture
def clock_scheduler(monkeypatch):
    clock = Clock()
    scheduler = engines._RollupMaintenanceScheduler(max_pending_jobs=1, kind="incremental embedding")
    # Drive the real worker synchronously; timed condition waits advance only the fake clock.
    scheduler._worker = SimpleNamespace(is_alive=lambda: True)
    waits = []

    def wait(timeout=None):
        assert timeout is not None and timeout > 0
        assert len(waits) < 100
        waits.append(timeout)
        clock.now += timeout

    monkeypatch.setattr(engines, "time", clock)
    monkeypatch.setattr(scheduler._condition, "wait", wait)
    yield scheduler, clock
    scheduler.quiesce(0)


@pytest.mark.parametrize("bank", ["queued", "deferred"])
@pytest.mark.parametrize("initial_delay, duplicate_delay", [(0, 5), (10, 5), (0, 0), (10, 0)])
def test_duplicate_keeps_later_deadline(clock_scheduler, bank, initial_delay, duplicate_delay):
    scheduler, clock = clock_scheduler
    key = ("fixture.db", "summary")
    ran = []
    start = clock.now

    def job():
        ran.append(clock.now)
        scheduler.quiesce(0)

    if bank == "deferred":
        assert scheduler.schedule(("other.db", "summary"), lambda: None)
    assert scheduler.schedule(key, job, delay=initial_delay)
    clock.now += 1
    assert scheduler.schedule(key, lambda: pytest.fail("duplicate replaced the job"), delay=duplicate_delay)
    deadline = max(start + initial_delay, clock.now + duplicate_delay)
    scheduler._run()
    assert ran == [deadline]
    assert not scheduler._not_before
    assert not scheduler._queued_keys and not scheduler._deferred_jobs


def test_duplicate_during_active_follow_up_keeps_delay(clock_scheduler):
    scheduler, clock = clock_scheduler
    key = ("fixture.db", "summary")
    ran = []
    start = clock.now

    def final():
        ran.append(clock.now)
        scheduler.quiesce(0)

    def follow_up():
        assert scheduler._running_follow_up
        assert scheduler.schedule(key, final)
        assert scheduler.schedule(key, final, delay=5)

    assert scheduler.schedule(key, lambda: scheduler.schedule(key, follow_up))
    scheduler._run()
    assert ran == [start + 5]
    assert scheduler._follow_up_key is None
    assert scheduler._follow_up_job is None
    assert not scheduler._not_before


def test_delayed_active_key_replaces_immediate_follow_up(clock_scheduler):
    scheduler, clock = clock_scheduler
    key = ("fixture.db", "summary")
    ran = []
    start = clock.now

    def final():
        ran.append(clock.now)
        scheduler.quiesce(0)

    def active():
        assert scheduler.schedule(key, lambda: pytest.fail("immediate follow-up survived"))
        assert scheduler.schedule(key, final, delay=5)
        assert scheduler._follow_up_key is None

    assert scheduler.schedule(key, active)
    scheduler._run()
    assert ran == [start + 5]
    assert not scheduler._not_before
