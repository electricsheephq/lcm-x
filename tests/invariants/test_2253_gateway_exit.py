"""#2253: a busy session-end tail at a host exit that skips the engine's shutdown (I2).

The Hermes gateway ends each agent's session, closes the agent and leaves through ``os._exit``. Until the host calls the
context engine's ``shutdown()`` at teardown (NousResearch/hermes-agent#87328), the tail a busy session end kept in
memory is neither retried, stored nor reported in that process. The first case pins that named limit (§3.5 M9); the
second shows that the engine's existing drain stores the tail once the host makes the call; the third shows that a
normal interpreter exit drains it through ``atexit`` even when the host never calls ``shutdown()``.
"""

import os
import subprocess
import sys

import pytest

from tests.invariants.test_p1e_decided_tails import contents
from tests.test_issue_2055_ingest_ledger_writes import engines as engines

SUBPROCESS = r'''
import os, pathlib, sys
import pytest
import tests.conftest  # Register the checkout as hermes_lcm, as pytest does in the parent.
from hermes_lcm.store import MessageStore
from tests.harness import pin_char_counter
from tests.test_host_uid_shadow import _engine, _state_db
from tests.invariants.test_p1e_decided_tails import history, queue
root, exit_style = pathlib.Path(sys.argv[1]), sys.argv[2]
patch = pytest.MonkeyPatch()
pin_char_counter(patch)
_state_db(root, [("S", None, None)])
store = MessageStore(root / "lcm.db")
store.ensure_ingest_ledger()
store.close()
engine = _engine(root)
queue(engine, history(True))
print("pending", len(engine._pending_session_tails), file=sys.stderr, flush=True)
if exit_style == "engine_shutdown":
    engine.shutdown()
sys.stderr.flush()
if exit_style == "normal_exit":
    sys.exit(0)
os._exit(0)
'''


@pytest.mark.parametrize("exit_style", ["gateway_today", "engine_shutdown", "normal_exit"])
def test_busy_end_tail_at_hard_exit(engines, tmp_path, exit_style):
    root = tmp_path / "child"
    root.mkdir()
    env = {**os.environ, "LCM_INGEST_LEDGER": "shadow", "LCM_HOST_MESSAGE_UID": "off"}
    result = subprocess.run([sys.executable, "-c", SUBPROCESS, str(root), exit_style],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "pending 1" in result.stderr
    assert "session=S unflushed=2" in result.stderr
    from tests.test_host_uid_shadow import _engine
    fresh = _engine(root)
    try:
        stored = contents(fresh)
    finally:
        fresh.shutdown()
    if exit_style == "gateway_today":
        # No shutdown or atexit drain and no exit report: the tail is missing until a host resume of S.
        assert stored == ["A", "B"]
        assert "after shutdown retry" not in result.stderr
    else:
        assert stored == ["A", "B", "C", "D"]
