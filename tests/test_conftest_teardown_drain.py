"""The per-test teardown drain waits for the same background work as the scheduler's own drain()."""
import pytest

from hermes_lcm import maintenance_scheduler
from tests.conftest import _wait_scheduler_idle


@pytest.mark.issue(1717)
def test_teardown_drain_waits_for_a_deferred_job():
    # No job was scheduled, so no worker thread runs; the deferred job stays pending until cleared.
    scheduler = maintenance_scheduler._RollupMaintenanceScheduler(kind="incremental embedding")
    scheduler._deferred_jobs[("conversation", "store")] = lambda: None
    assert scheduler.drain({("conversation", "store")}, timeout=0.05) is False
    assert _wait_scheduler_idle(scheduler, timeout=0.05) is False
    with scheduler._condition:
        scheduler._deferred_jobs.clear()
    assert _wait_scheduler_idle(scheduler, timeout=0.05) is True
