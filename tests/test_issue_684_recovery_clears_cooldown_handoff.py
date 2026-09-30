"""#684: a host recovery call (``compress(..., bypass_cooldown=True)``, #608) after a boundary-cooldown preflight
runs the summariser; the cooldown handoff no longer makes it cleanup-only.

The preflight is the real one: a boundary cooldown is active and the replay diff requests an ingest cleanup, so it
sets ``_preflight_cleanup_only_due_to_boundary_cooldown``. Only the cleanup verdict and the replay list the
preflight sees are stubbed; compress() then runs on the real store. Forced overflow, an ordinary automatic call
during the cooldown and the native-recovery handoff (kept by decision, #463/#464) are pinned unchanged."""

from __future__ import annotations

import time

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
ACCEPTED = "Earlier turns.\nExpand for details about: turns"


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


@pytest.fixture
def provider(monkeypatch):
    calls: list[str] = []

    def summarise(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append(model)
        return ACCEPTED

    monkeypatch.setattr(escalation, "_call_llm_for_summary", summarise)
    return calls


def _engine(tmp_path, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "max_assembly_tokens": 100_000, "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _view(turns: int = 6) -> list[dict]:
    rows = [{"role": "system", "content": "system prompt"}]
    for i in range(turns):
        rows += [{"role": "user", "content": f"[T{i}] user turn{PAD}", "timestamp": 10.0 * (i + 1)},
                 {"role": "assistant", "content": f"reply to T{i}{PAD}"}]
    return rows


def _cooldown_preflight(engine, view, monkeypatch) -> None:
    """A boundary cooldown, then a preflight whose replay diff requests an ingest cleanup."""
    engine._last_boundary_skip_time = time.time()
    real_ingest = engine._ingest_messages
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_replay_diff_requests_ingest_cleanup", lambda original, replay: True)
        patch.setattr(engine, "_ingest_messages", lambda messages: [
            *real_ingest(messages)[:-1], {**messages[-1], "_replay_cleanup": True}])
        assert engine.should_compress_preflight(view) is True
    assert engine._compression_boundary_cooldown_active()


def _ceiling(engine) -> int:
    return int(engine.context_length * (1 - engine._config.survival_reserve))


def _nodes(engine) -> list:
    return engine._dag.get_session_nodes("S")


def test_recovery_call_after_a_cooldown_preflight_runs_the_summariser(tmp_path, monkeypatch, provider):
    engine = _engine(tmp_path)
    view = _view()
    try:
        _cooldown_preflight(engine, view, monkeypatch)
        assert engine._preflight_cleanup_only_due_to_boundary_cooldown is True
        host_tokens = engine.threshold_tokens + 1
        assert engine.threshold_tokens <= host_tokens < _ceiling(engine)
        assert not engine._should_force_overflow_recovery(observed_tokens=host_tokens)
        engine.compress(view, current_tokens=host_tokens, bypass_cooldown=True)
        assert provider and _nodes(engine), "a summariser pass ran and a leaf is stored"
        assert engine._last_compression_status == "compacted"
        assert engine._preflight_cleanup_only_due_to_boundary_cooldown is False
    finally:
        engine.shutdown()


def test_an_ordinary_automatic_call_during_the_cooldown_is_still_cleanup_only(tmp_path, monkeypatch, provider):
    engine = _engine(tmp_path)
    view = _view()
    try:
        _cooldown_preflight(engine, view, monkeypatch)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert provider == [] and _nodes(engine) == []
        assert engine._last_compression_status == "sanitized"
    finally:
        engine.shutdown()


def test_forced_overflow_after_a_cooldown_preflight_is_unchanged(tmp_path, monkeypatch, provider):
    engine = _engine(tmp_path)
    view = _view()
    try:
        _cooldown_preflight(engine, view, monkeypatch)
        assert engine._should_force_overflow_recovery(observed_tokens=150_000)
        engine.compress(view, current_tokens=150_000)
        assert provider and _nodes(engine) and engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()


def test_the_native_recovery_handoff_is_kept_by_a_recovery_call(tmp_path, monkeypatch, provider):
    """Decision on #684: native recovery keeps ownership of a below-threshold list; bypass_cooldown does not
    clear ``_native_recovery_preflight_cleanup_only``."""
    engine = _engine(tmp_path, native_recovery=True, context_threshold=0.5)
    view = _view()
    try:
        engine._last_boundary_skip_time = time.time()
        with monkeypatch.context() as patch:
            real_ingest = engine._ingest_messages
            patch.setattr(engine, "_replay_diff_requests_ingest_cleanup", lambda original, replay: True)
            patch.setattr(engine, "_ingest_messages", lambda messages: [
                *real_ingest(messages)[:-1], {**messages[-1], "_replay_cleanup": True}])
            engine.should_compress_preflight(view)
        assert engine._native_recovery_preflight_cleanup_only is True
        assert engine._preflight_cleanup_only_due_to_boundary_cooldown is True
        host_tokens = engine.threshold_tokens - 1
        engine.compress(view, current_tokens=host_tokens, bypass_cooldown=True)
        assert provider == [] and _nodes(engine) == []
        assert engine._last_compression_status == "sanitized"
    finally:
        engine.shutdown()
