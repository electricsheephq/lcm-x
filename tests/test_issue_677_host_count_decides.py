"""#677: below the host threshold the host count decides, on every preflight branch and on direct entry.

The Hermes host order: ``should_compress(host_tokens)`` is False, so the host asks
``should_compress_preflight(messages)``; on True it calls ``compress(messages, current_tokens=host_tokens)`` through
the automatic breaker gate. LCM's own count of the list can be >= T while the host's count is < T (the rough-count
branch); #651 made only the maintenance branch cleanup-only. A caller that skips the gate while the #651 hold runs
(``record_rejected_compaction(); compress(...)``) is cleanup-only too. Summaries are stubbed and counted."""

from __future__ import annotations

import inspect

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

PAD = " alpha beta gamma delta" * 30


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """As in the #651 tests: LCM's character estimate and no host estimator."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


@pytest.fixture
def summaries(monkeypatch):
    calls: list[str] = []

    def summarize(**kwargs):
        calls.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return calls


def _engine(tmp_path, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 100, "fresh_tail_pressure_yield_enabled": False,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _view(turns: int = 6) -> list[dict]:
    rows = [{"role": "system", "content": "system prompt"}]
    for i in range(turns):
        rows += [{"role": "user", "content": f"[T{i}] user turn{PAD}", "timestamp": 10.0 * (i + 1)},
                 {"role": "assistant", "content": f"reply to T{i}{PAD}"}]
    return rows


def _leaves(engine) -> int:
    return len(engine._dag.get_session_nodes(engine._session_id))


def _host_gate(compressor, *, bypass_cooldown: bool = False) -> bool:
    """The host's ``not force and _automatic_compression_gate_blocks`` read (a missing gate never blocks)."""
    blocked = getattr(type(compressor), "_automatic_compression_blocked", None)
    if not callable(blocked):
        return False
    accepts = bypass_cooldown and "ignore_cooldown" in inspect.signature(blocked).parameters
    return bool(blocked(compressor, ignore_cooldown=True) if accepts else blocked(compressor))


def _divergent(tmp_path, **config):
    """LCM rough >= T > host count (the review's numbers: T=35k, host 30k, LCM 40k)."""
    engine = _engine(tmp_path, **config)
    view = _view()
    rough = count_messages_tokens(view)
    engine.threshold_tokens = rough - 200  # LCM's count of the list is over T
    host = engine.threshold_tokens - 300  # the host's (anchored) count is under T
    ceiling = int(engine.context_length * (1 - engine._config.survival_reserve))
    assert host < engine.threshold_tokens <= rough < ceiling
    return engine, view, rough, host


def _assert_cleanup_only(engine, summaries):
    assert summaries == [] and _leaves(engine) == 0
    assert engine._last_compression_status == "sanitized"


# -- the rough-count branch: the host count decides ---------------------------------------------------------

def test_host_order_rough_over_threshold_host_count_under_is_cleanup_only(tmp_path, summaries):
    engine, view, rough, host = _divergent(tmp_path)
    try:
        assert engine.should_compress(host) is False  # the host's threshold call: below T
        assert engine.should_compress_preflight(view) is True  # the rough >= T branch asks
        assert engine._preflight_below_threshold_cleanup_only is False  # not the #651 maintenance branch
        assert _host_gate(engine) is False
        result = engine.compress(view, current_tokens=host)
        _assert_cleanup_only(engine, summaries)
        assert len(result) == len(view)
        assert engine._preflight_automatic_request is False  # consumed
        assert engine._no_progress_hold is None  # cleanup-only is not a threshold pass
    finally:
        engine.shutdown()


def test_control_the_maintenance_branch_stays_cleanup_only(tmp_path, summaries):
    """LCM rough < T under critical budget pressure: the #651 branch, unchanged."""
    engine = _engine(tmp_path, deferred_maintenance_enabled=True, critical_budget_pressure_ratio=0.5)
    view = _view()
    rough = count_messages_tokens(view)
    engine.threshold_tokens = 100_000
    engine.context_length = int(rough * 1.6)
    try:
        assert engine.should_compress(rough) is False
        assert engine.should_compress_preflight(view) is True
        assert engine._preflight_below_threshold_cleanup_only is True
        engine.compress(view, current_tokens=rough)
        _assert_cleanup_only(engine, summaries)
    finally:
        engine.shutdown()


# -- direct entry while the #651 hold runs ------------------------------------------------------------------

@pytest.mark.parametrize("count", ["under_threshold", "over_threshold"])
def test_direct_rejection_then_compress_is_cleanup_only_while_the_hold_runs(tmp_path, summaries, count):
    engine, view, rough, host = _divergent(tmp_path)
    tokens_seen = host if count == "under_threshold" else rough
    try:
        engine.record_rejected_compaction()
        # The Hermes path is closed while the hold runs; a caller that skips the gate is closed here too.
        assert _host_gate(engine) is True
        assert engine.should_compress_preflight(view) is False
        engine.compress(view, current_tokens=tokens_seen)
        _assert_cleanup_only(engine, summaries)
        assert engine._no_progress_hold is not None  # no leaf: the hold still runs
        assert engine._preflight_below_threshold_cleanup_only is False  # the handoff is one-shot
    finally:
        engine.shutdown()


# -- regression guards --------------------------------------------------------------------------------------

def test_host_count_at_threshold_on_the_threshold_path_still_summarises(tmp_path, summaries):
    engine, view, rough, host = _divergent(tmp_path)
    try:
        assert engine.should_compress(rough) is True  # the host's count reached T: the threshold path
        engine.compress(view, current_tokens=rough)
        assert summaries and _leaves(engine) >= 1
        assert engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()


def test_preflight_request_with_host_count_at_threshold_still_summarises(tmp_path, summaries):
    engine, view, rough, host = _divergent(tmp_path)
    try:
        assert engine.should_compress_preflight(view) is True
        engine.compress(view, current_tokens=engine.threshold_tokens)
        assert summaries and _leaves(engine) >= 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("call", ["manual", "provider_overflow", "forced_overflow"])
def test_forced_and_provider_overflow_calls_during_the_hold_still_summarise(tmp_path, summaries, call):
    cap = count_messages_tokens(_view()) + 1_000  # above the view: only the forced cell observes over it
    engine, view, rough, host = _divergent(tmp_path, max_assembly_tokens=cap)
    try:
        engine.record_rejected_compaction()
        assert _host_gate(engine) is True
        if call == "manual":  # /compress
            engine.compress(view, current_tokens=host, force=True)
        elif call == "provider_overflow":  # the host's recovery attempt
            assert _host_gate(engine, bypass_cooldown=True) is False
            engine.compress(view, current_tokens=host, bypass_cooldown=True)
        else:  # over the assembly cap: forced overflow is never held
            engine.compress(view, current_tokens=cap + 1)
        assert summaries and _leaves(engine) >= 1, engine._last_compression_noop_reason
    finally:
        engine.shutdown()


def test_the_request_is_one_shot(tmp_path, summaries):
    engine, view, rough, host = _divergent(tmp_path)
    try:
        assert engine.should_compress_preflight(view) is True
        engine.compress(view, current_tokens=host)
        _assert_cleanup_only(engine, summaries)
        # a later automatic call with no new preflight request runs as before #677
        engine.compress(view, current_tokens=host)
        assert summaries and _leaves(engine) >= 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("host_tokens", [None, 0])
def test_unknown_host_count_keeps_todays_behaviour(tmp_path, summaries, host_tokens):
    engine, view, rough, host = _divergent(tmp_path)
    try:
        assert engine.should_compress_preflight(view) is True
        engine.compress(view, current_tokens=host_tokens)
        assert summaries and _leaves(engine) >= 1
        assert engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()


# -- round 3 (#679 review): lifecycle boundaries and the survival ceiling ------------------------------------

@pytest.mark.parametrize("boundary", ["reset", "rebind"])
def test_a_request_never_crosses_a_reset_or_rebind(tmp_path, summaries, boundary):
    engine, view, rough, host = _divergent(tmp_path)
    threshold = engine.threshold_tokens
    try:
        assert engine.should_compress_preflight(view) is True  # the previous binding's request
        if boundary == "reset":
            engine.on_session_reset()
        else:
            engine.on_session_start("S2", platform="telegram", context_length=200_000, conversation_id="conv2")
        assert engine._preflight_automatic_request is False
        assert engine._preflight_below_threshold_cleanup_only is False
        engine.threshold_tokens = threshold
        # an automatic call below the threshold with no request of its own runs as before #677
        engine.compress(view, current_tokens=host)
        assert summaries and _leaves(engine) >= 1
        assert engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()


def test_the_hold_guard_never_makes_a_call_at_the_survival_ceiling_cleanup_only(tmp_path, summaries):
    """A threshold above the survival ceiling (95,000 over 85,000): the hold never blocks a call at the ceiling,
    so the direct-entry guard leaves it a normal pass."""
    engine = _engine(tmp_path)
    view = _view()
    try:
        engine.context_length = 100_000
        engine.threshold_tokens = 95_000
        ceiling = int(engine.context_length * (1 - engine._config.survival_reserve))
        assert ceiling == 85_000 < engine.threshold_tokens
        engine.record_rejected_compaction()
        assert engine._no_progress_hold_active() is True
        assert engine._no_progress_hold_blocks(ceiling) is False
        engine.compress(view, current_tokens=ceiling)
        assert summaries and _leaves(engine) >= 1, engine._last_compression_noop_reason
        assert engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()
