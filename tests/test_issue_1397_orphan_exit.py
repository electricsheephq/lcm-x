"""Exit shares one deadline with orphaned local encoder slot cleanup."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["released", "budget", "none"])
def test_1397_exit_waits_for_real_fastembed_orphan_with_shared_budget(tmp_path, mode):
    marker = tmp_path / "encoder-finished"
    script = r"""
import atexit
import faulthandler
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

faulthandler.dump_traceback_later(8)  # Diagnostic only; the parent owns the hang watchdog.
print("stage: imports", file=sys.stderr, flush=True)
repo, marker, mode = sys.argv[1:]
sys.path.insert(0, repo)
spec = importlib.util.spec_from_file_location("hermes_lcm", Path(repo) / "__init__.py", submodule_search_locations=[repo])
package = importlib.util.module_from_spec(spec)
sys.modules["hermes_lcm"] = package
from hermes_lcm import engine, command, embedding_maintenance, embedding_provider
from tests.test_issue_1330_retire_embedding_maintenance import _seed_batches

started, release, pass_done, quiescing = (threading.Event() for _ in range(4))
state = {"waits": []}
clock = [1000.0]
engine._EMBEDDING_EXIT_TIMEOUT_S = 0.5
slots = embedding_provider._LOCAL_EMBED_WORKER_SLOTS

def finish():
    available = 0
    while available < embedding_provider._LOCAL_EMBED_MAX_WORKERS and slots.acquire(blocking=False):
        available += 1
    for _ in range(available):
        slots.release()
    state.update(available=available, max_slots=embedding_provider._LOCAL_EMBED_MAX_WORKERS,
                 finished=Path(marker).exists())
    print(json.dumps(state), flush=True)

atexit.register(finish)  # Runs after the scheduler's exit handler.

class Model:
    def embed(self, texts):
        assert threading.current_thread().name == "lcm-fastembed-embedding"
        state["encoder_start"] = time.monotonic()
        started.set()
        if mode != "none":
            assert release.wait(10)  # Hang watchdog only.
        Path(marker).write_text("finished")
        return [[1.0, 0.0] for _ in texts]

# Freeze provider time until the fake encoder starts. Dispatch/setup cannot
# consume the tiny provider budget or prevent the encoder from starting.
def provider_now():
    return time.monotonic() - state["encoder_start"] if started.is_set() else 0.0

embedding_provider.time = SimpleNamespace(monotonic=provider_now)
config = _seed_batches(Path(marker).parent / "orphan.db", "summaries")
print("stage: seeded", file=sys.stderr, flush=True)
provider = embedding_provider.FastembedProvider("test-model", timeout=10 if mode == "none" else 0.01)
provider._model = Model()
command.resolve_provider = lambda *_a, **_k: provider
embedding_provider.probe_provider_availability = lambda _c: {"available": True}

def job():
    embedding_maintenance.run_incremental_embedding_pass(config.database_path, config)
    pass_done.set()
    if mode == "budget":
        assert quiescing.wait(10)
        # Deterministically spend part of the SAME deadline on scheduler work.
        clock[0] += 0.2

scheduler = engine._EMBEDDING_MAINTENANCE_SCHEDULER
assert scheduler.schedule(("orphan-exit", "embeddings"), job)
assert started.wait(10)
assert pass_done.wait(10)
print("stage: orphan ready", file=sys.stderr, flush=True)
if mode != "budget":
    assert scheduler.drain({("orphan-exit", "embeddings")}, timeout=10)

def exit_now():
    if "quiesce_start" not in state:
        state["quiesce_start"] = clock[0]
        quiescing.set()
    return clock[0]

class ObservedSlots:
    def acquire(self, *, timeout):
        state["waits"].append([clock[0], timeout])
        if slots.acquire(blocking=False):
            return True
        state["occupied"] = True
        if mode == "released":
            release.set()
            return slots.acquire(timeout=10)  # Only the watchdog uses real time.
        clock[0] += timeout
        return False

    def release(self):
        slots.release()

real_drain = embedding_provider._drain_local_embedding_workers
def observed_drain(deadline):
    state["drain_deadline"] = deadline
    state["drain_start"] = clock[0]
    result = real_drain(deadline)
    state.update(drain_result=result, drain_end=clock[0])
    return result

real_quiesce = scheduler.quiesce
def observed_quiesce(timeout):
    state["timeout"] = timeout
    engine.time = SimpleNamespace(monotonic=exit_now)
    embedding_provider.time = SimpleNamespace(monotonic=lambda: clock[0])
    embedding_provider._LOCAL_EMBED_WORKER_SLOTS = ObservedSlots()
    return real_quiesce(timeout)

scheduler.quiesce = observed_quiesce
embedding_provider._drain_local_embedding_workers = observed_drain
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve().parents[1]), str(marker), mode],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"exit subprocess hang watchdog: {exc.stderr}", pytrace=False)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads(result.stdout.splitlines()[-1])
    deadline = state["quiesce_start"] + state["timeout"]
    assert state["drain_deadline"] == deadline
    assert state["waits"]
    for now, timeout in state["waits"]:
        assert 0 <= timeout <= max(0, deadline - now), state
    assert state["drain_end"] <= deadline
    if mode == "budget":
        assert state["drain_start"] == pytest.approx(state["quiesce_start"] + 0.2)
        assert not state["drain_result"] and not state["finished"]
        assert state["occupied"]
        assert state["available"] == state["max_slots"] - 1
    else:
        assert state["drain_result"] and state["finished"]
        assert state["available"] == state["max_slots"]
        assert state.get("occupied", False) is (mode == "released")
