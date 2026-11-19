"""Synthetic regressions for the host compaction gate."""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.config import LCMConfig
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_2851_rebind_cursor import _view
from tests.test_real_turn_loop_acp_override import HERMES


@contextmanager
def _open(path):
    config = LCMConfig(database_path=str(path / "lcm.db"), fresh_tail_count=2, leaf_chunk_tokens=400,
                       context_threshold=0.001, threshold_full_sweep_enabled=False)
    engine = engine_module.LCMEngine(config=config, hermes_home=str(path))
    try:
        engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
        yield engine
    finally:
        engine.shutdown()


def _mode(monkeypatch, on):
    if on:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    else:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)


_BUSY = "cooldown:lcm_compaction_in_progress"
_IDLE = (False, False, None)


def _gate_state(engine):
    blocked = type(engine)._automatic_compression_blocked
    return blocked(engine), blocked(engine, ignore_cooldown=True), engine._compression_block_reason()


@contextmanager
def _pass_waiting_in_summary(tmp_path, monkeypatch):
    ready, release = Event(), Event()
    owner_states = []
    _state_db(tmp_path, [("S", None, None)])
    with _open(tmp_path) as engine, ThreadPoolExecutor(max_workers=1) as threads:
        def summarize(**kwargs):
            owner_states.append(_gate_state(engine))
            ready.set()
            assert release.wait(10)
            return "Earlier turns.\nExpand for details about: turns", 1

        monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
        view = _view(6)
        future = threads.submit(engine.compress, view, 24000)
        try:
            assert ready.wait(10), "managed compression did not enter the summarizer"
            assert owner_states == [_IDLE]
            yield engine
        finally:
            release.set()
            assert len(future.result(timeout=10)) < len(view)
            assert _gate_state(engine) == _IDLE


@pytest.mark.parametrize("on", [False, True], ids=["unset", "on"])
def test_other_thread_reports_temporary_block(tmp_path, monkeypatch, on):
    _mode(monkeypatch, on)
    with _pass_waiting_in_summary(tmp_path, monkeypatch) as engine:
        state = _gate_state(engine)
        print(f"other thread gate={state!r}")
        assert state == (True, True, _BUSY)


def _probe_real_host_gate(tmp_path, gate, bypass_cooldown):
    with pytest.MonkeyPatch.context() as patch:
        _mode(patch, False)
        with _pass_waiting_in_summary(tmp_path, patch) as engine:
            agent = SimpleNamespace(context_compressor=engine, _compression_blocked_transient=None)
            return gate(agent, bypass_cooldown), agent._compression_blocked_transient


@pytest.mark.skipif(HERMES is None, reason="no real Hermes runtime available")
@pytest.mark.parametrize("bypass_cooldown", [False, True], ids=["automatic", "recovery"])
def test_real_host_gate_defers_busy_engine(tmp_path, bypass_cooldown):
    python, src = HERMES
    repo = Path(__file__).resolve().parents[2]
    source = """
import json, os, socket, sys
from pathlib import Path
def offline(*args, **kwargs):
    raise OSError("offline host gate probe")
socket.socket.connect = socket.create_connection = socket.getaddrinfo = offline
import agent.conversation_compression as host
sys.path.insert(0, sys.argv[1])
import tests.conftest
from tests.invariants.test_compress_block_contract import _probe_real_host_gate
state = _probe_real_host_gate(Path(os.environ["HERMES_HOME"]), host._automatic_compression_gate_blocks,
                              sys.argv[2] == "True")
print(json.dumps([host.__file__, state]))
"""
    home = tmp_path / "hermes-home"
    home.mkdir()
    completed = subprocess.run(
        [python, "-c", source, str(repo), str(bypass_cooldown)], cwd=src or None,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HERMES_HOME": str(home),
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        capture_output=True, text=True, timeout=45,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    host_file, state = json.loads(completed.stdout.splitlines()[-1])
    if src:
        assert Path(host_file).resolve() == Path(src).resolve() / "agent/conversation_compression.py"
    print(f"real host {host_file}, bypass_cooldown={bypass_cooldown}, gate={state!r}")
    assert state == [_BUSY, _BUSY]


@pytest.mark.parametrize("child", [False, True], ids=["bypassed", "delegated-child"])
def test_bypassed_pass_does_not_report_busy(tmp_path, monkeypatch, child):
    _mode(monkeypatch, False)
    ready, release = Event(), Event()
    view = _view(1)
    with _open(tmp_path) as engine, ThreadPoolExecutor(max_workers=1) as threads:
        def bypassed(*args):
            assert _gate_state(engine) == _IDLE
            ready.set()
            assert release.wait(10)
            return view

        monkeypatch.setattr(engine, "_compress_bypassed", bypassed)

        def compress():
            if child:
                engine._mark_thread_context_stateless("child")
            try:
                assert engine._bypasses_lcm_context_management()
                return engine.compress(view, current_tokens=24000)
            finally:
                if child:
                    engine._clear_thread_context_stateless("child")

        if not child:
            monkeypatch.setattr(engine, "_session_ignored", True)
        future = threads.submit(compress)
        try:
            assert ready.wait(10), "bypassed compression did not reach the fallback"
            assert _gate_state(engine) == _IDLE
        finally:
            release.set()
        assert future.result(timeout=10) is view
        assert _gate_state(engine) == _IDLE
