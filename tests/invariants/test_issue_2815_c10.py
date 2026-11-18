"""Synthetic regressions for the host busy signal and the UID cut window."""

import hashlib
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.engine import _summary_part_text
from hermes_lcm.ingest_protection import _has_lossy_sensitive_redaction
from tests.invariants.test_issue_2815_c9 import _cold, _contents, _mode, _open, _view
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_581_582_store_complete_survival import _rows, _stage, _turn
from tests.test_real_turn_loop_acp_override import HERMES

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
from tests.invariants.test_issue_2815_c10 import _probe_real_host_gate
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


def test_mixed_bound_prefix_and_unbound_legacy_stretch_uses_full_walk(tmp_path, monkeypatch, caplog):
    legacy = [row for i in range(40) for row in _turn(f"legacy-{i}", 0, stamp_user=False)]
    added = [row for i in range(35) for row in _turn(f"new-{i}", 2000.0 + i * 10)]
    caplog.set_level(logging.INFO, logger="hermes_lcm.reconcile")
    seed, stored, cuts = _cold(tmp_path / "on", monkeypatch, on=True, bound=True, legacy=legacy, added=added)
    _, base, base_cuts = _cold(tmp_path / "unset", monkeypatch, on=False, bound=True, legacy=legacy, added=added)
    print(f"mixed cut: seed={len(seed)} on={len(stored)} unset={len(base)} projections={len(cuts)}")
    assert len(seed) == 82
    assert stored == base == seed + _contents(added)
    assert cuts == base_cuts == list(range(152, 81, -1))
    assert "LCM UID cut budget disabled: unbound_host_rows=87" in caplog.text


def test_unbound_tool_pair_with_64_new_rows_still_stores_66(tmp_path, monkeypatch):
    added = _view(16, "new")
    seed, stored, cuts = _cold(tmp_path / "on", monkeypatch, on=True, bound=False, added=added)
    assert len(stored) == 66
    assert stored == seed + _contents(added)
    assert cuts == list(range(66, 1, -1))


def test_bound_reference_shape_keeps_cut_budget(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None)])
    old = _view(2, "old")
    lossy = old[1]
    lossy["_tool_call_uids"] = ["synthetic-tool-call-uid"]
    lossy["tool_calls"][0]["function"]["arguments"] = '{"password":"synthetic-password"}'
    prior = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Earlier version {i}",
         "message_uid": f"updated-{i}", "timestamp": 2000.0 + i}
        for i in range(200)
    ]
    added = [{**row, "content": f"New version {i}"} for i, row in enumerate(prior)]
    _mode(monkeypatch, False)
    with _open(tmp_path) as seed:
        seed._config.sensitive_patterns_enabled = True
        seed.ingest(deepcopy(prior + old))
        _stage(seed, 0, [row["store_id"] for row in _rows(seed)[:2]])
        head = {"role": "user", "content": _summary_part_text(seed._dag.get_node(1))}
        seed._store.append("S", {"role": "assistant", "content": "Newest stored row outside the view"},
                           conversation_id="conv")
        seed_count = len(_rows(seed))
    _mode(monkeypatch, True)
    with _open(tmp_path) as cold:
        cold._config.sensitive_patterns_enabled = True
        view = cold._redact_active_replay_messages(deepcopy([*old[:4], head, head, *old[4:], *added]))
        assert _has_lossy_sensitive_redaction(json.dumps(view[1]["tool_calls"]))
        assert 0 in cold._host_uid_resolve(view, prefix_only=True)
        assert 1 not in cold._host_uid_resolve(view)
        assert cold._host_uid_whole_scaffold(head) and "message_uid" not in head
        assert cold._host_uid_host_uids(row["message_uid"] for row in old + added) == {
            row["message_uid"] for row in old + added
        }
        cuts = []
        project = cold._project_replay_candidate

        def count(run, candidate):
            cuts.append(candidate.cursor)
            return project(run, candidate)

        monkeypatch.setattr(cold, "_project_replay_candidate", count)
        cold.ingest(view)
        columns = ("store_id", "session_id", "conversation_id", "role", "content", "tool_calls", "tool_call_id",
                   "tool_name", "observed_at")
        stored = [[row[key] for key in columns] for row in _rows(cold)]
    digest = hashlib.sha256(json.dumps(stored, sort_keys=True).encode()).hexdigest()
    print(f"bound reference: seed={seed_count} stored={len(stored)} projections={len(cuts)} sha256={digest}")
    assert len(stored) == 413
    assert len(cuts) == 64


def test_cut_window_checks_bindings_and_scaffolds_without_replay_identities(tmp_path, monkeypatch):
    _mode(monkeypatch, True)
    _state_db(tmp_path, [("S", None, None)])
    with _open(tmp_path) as engine:
        check = getattr(engine, "_host_uid_unbound_rows", None)
        if check is None:
            pytest.skip("c9 has no cut-window predicate")
        host = {"role": "user", "content": "Synthetic host row"}
        engine.ingest([host])
        _stage(engine, 0, [row["store_id"] for row in _rows(engine)])
        head = {"role": "user", "content": _summary_part_text(engine._dag.get_node(1))}
        message_identities, scaffolds = [], []
        scaffold_check = engine._host_uid_whole_scaffold

        def scaffold(row):
            scaffolds.append(row)
            return scaffold_check(row)

        monkeypatch.setattr(engine, "_host_uid_whole_scaffold", scaffold)
        monkeypatch.setattr(engine, "_message_replay_identity", lambda *a, **kw: message_identities.append(a))
        unbound = [host, {**host, "message_uid": []}, {**host, "message_uid": "unbound"}]
        assert check([{**host, "message_uid": "bound"}, *unbound, head], {"bound"}) == 3
        assert scaffolds == [*unbound, head]
        print(f"cut-window replay identity calls={len(message_identities)}")
        assert message_identities == []
