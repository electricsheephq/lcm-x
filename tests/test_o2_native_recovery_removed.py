"""O2: the legacy flag cannot select a writer or change preflight decisions."""
import copy
import logging
import sys
from types import SimpleNamespace

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_605_foreground_budget import (
    _engine, _view, _provider, _leaves, _stop_lines,
    _pin_lcm_char_counter, clock as _clock_fixture,  # noqa: F401
)


clock = _clock_fixture

WARNING = (
    "LCM_NATIVE_RECOVERY is no longer supported (removed in v0.25.0) and is ignored; "
    "compaction uses the LCM path"
)


def _forbid_native(monkeypatch):
    calls = []

    class Native:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("host native compressor must never be called")

    monkeypatch.setitem(sys.modules, "agent.context_compressor", SimpleNamespace(ContextCompressor=Native))
    return calls


def test_t1_flag_uses_lcm_path(tmp_path, monkeypatch, clock):
    calls = _forbid_native(monkeypatch)
    provider = _provider(monkeypatch, clock, default=0.01)
    engine = _engine(tmp_path, native_recovery=True)
    engine._compression_cancelled_check = lambda: False
    try:
        result = engine.compress(_view(), current_tokens=engine.threshold_tokens + 1)
        assert provider.calls and _leaves(engine)
        assert engine.last_compression_status == "compacted"
        assert len(result) < len(_view())
        assert calls == []
        proof = engine._compress_commit_proof
        assert proof["native"] is False
        assert "native_summary_index" not in proof
    finally:
        engine.shutdown()


def test_t1b_ten_second_foreground_budget(tmp_path, monkeypatch, clock, caplog):
    calls = _forbid_native(monkeypatch)
    provider = _provider(monkeypatch, clock, default=None)
    engine = _engine(tmp_path, native_recovery=True, foreground_soft_seconds=0, foreground_hard_seconds=10)
    engine._compression_cancelled_check = lambda: False
    try:
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.compress(_view(), current_tokens=engine.threshold_tokens + 1)
        assert calls == []
        # The standard 15s call estimate plus finalize reserve exceeds 10s.
        assert provider.calls == []
        assert all(timeout <= 10 - 5 - started + 0.5 for _model, timeout, started in provider.calls)
        assert clock.offset <= 10
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["stop_reason"] == "time_budget_exhausted"
        assert telemetry["budget_exhausted"] is True
        stops = _stop_lines(caplog)
        assert len(stops) == 1 and float(stops[0][2]) <= 10.5
    finally:
        engine.shutdown()


@pytest.mark.parametrize("state", ["below", "threshold", "cleanup"])
@pytest.mark.parametrize("replay_changed", [False, True])
def test_t2_preflight_flag_is_ignored(tmp_path, monkeypatch, state, replay_changed):
    decisions = []
    for enabled in (False, True):
        home = tmp_path / str(enabled)
        home.mkdir()
        engine = _engine(home, native_recovery=enabled, threshold_full_sweep_enabled=False,
                         leaf_chunk_tokens=1_000_000, survival_fit=False)
        rows = _view(3)
        engine.threshold_tokens = 1 if state == "threshold" else 100_000
        # Exercise both preflight branches: unchanged input and an ingest rewrite.
        replay = copy.deepcopy(rows)
        if replay_changed or state == "cleanup":
            replay[-1]["content"] += " replay adjustment"
        monkeypatch.setattr(engine, "_ingest_messages", lambda _rows: replay)
        monkeypatch.setattr(engine, "_replay_diff_requests_ingest_cleanup", lambda *_: state == "cleanup")
        try:
            gate = engine.should_compress(prompt_tokens=engine.threshold_tokens if state == "threshold" else 1)
            decision = engine.should_compress_preflight(rows)
            decisions.append((gate, decision, engine._preflight_below_threshold_cleanup_only,
                              engine._preflight_automatic_request))
        finally:
            engine.shutdown()
    assert decisions[0] == decisions[1]
    assert decisions[0][1] is (state == "cleanup")


def test_t3_warning_once_per_process(tmp_path, monkeypatch, caplog):
    # Reset only the warning guard, emulating a fresh process without reopening stores.
    monkeypatch.setattr(engine_module, "_NATIVE_RECOVERY_WARNING_LOGGED", False)
    for enabled, expected in [(False, 0), (True, 1), (True, 1), (False, 1)]:
        with caplog.at_level(logging.WARNING):
            engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "warning.db"),
                                               native_recovery=enabled), hermes_home=str(tmp_path))
            engine.shutdown()
        assert sum(record.getMessage() == WARNING for record in caplog.records) == expected


def test_user_replay_externalization_ignores_flag(tmp_path, monkeypatch):
    # #1016: the subject is an externalized user/assistant row, not the 100k floor.
    monkeypatch.setattr("hermes_lcm.ingest_protection._NON_TOOL_EXTERNALIZATION_FLOOR_CHARS", 0)
    for enabled in (False, True):
        home = tmp_path / str(enabled)
        home.mkdir()
        engine = _engine(home, native_recovery=enabled, large_output_externalization_enabled=True,
                         large_output_externalization_threshold_chars=12_000,
                         large_output_externalization_path=str(home / "externalized"))
        try:
            # #1016: the marker sits outside the stub's head/tail preview.
            rows = [{"role": "user", "content": "ordinary retained prose " * 1400 + "retained fact "
                     + "ordinary retained prose " * 1400}]
            replay = engine._ingest_messages(rows)
            assert "retained fact" not in replay[0]["content"]
            assert "retained fact" not in engine._store.get_session_messages("S")[0]["content"]
        finally:
            engine.shutdown()
