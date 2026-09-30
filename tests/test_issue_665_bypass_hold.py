"""#665: the host's refusal of a compaction of a bypassed (auxiliary) session never arms the foreground's
no-progress hold. The same refusal on the foreground session still arms it (#651)."""

from __future__ import annotations

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
    """LCM's character estimate and no host estimator, as in the #651 tests."""
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


def _engine(tmp_path) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=100,
                                        fresh_tail_pressure_yield_enabled=False,
                                        database_path=str(tmp_path / "lcm.db")))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _view(turns: int = 6) -> list[dict]:
    rows = [{"role": "system", "content": "system prompt"}]
    for i in range(turns):
        rows += [{"role": "user", "content": f"[T{i}] user turn{PAD}", "timestamp": 10.0 * (i + 1)},
                 {"role": "assistant", "content": f"reply to T{i}{PAD}"}]
    return rows


def test_t5_a_bypassed_session_refusal_does_not_hold_the_foreground(tmp_path, summaries):
    engine = _engine(tmp_path)
    try:
        engine.threshold_tokens = 100
        engine._mark_thread_context_stateless("aux-review")  # an auxiliary side channel on this thread
        assert engine._bypasses_lcm_context_management() is True
        engine.record_rejected_compaction()  # the host refused the auxiliary session's compaction
        engine._clear_thread_context_stateless()  # back to the foreground session
        assert engine._bypasses_lcm_context_management() is False and engine._session_id == "S"
        assert engine._no_progress_hold_active() is False
        assert engine.get_status()["no_progress_hold"] is None
        assert engine._automatic_compression_blocked() is False and engine.should_compress(1_000) is True
        view = _view()
        engine.compress(view, current_tokens=count_messages_tokens(view))
        assert summaries and engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()


def test_t5_contrast_a_foreground_refusal_arms_the_hold(tmp_path, summaries):
    engine = _engine(tmp_path)
    try:
        engine.threshold_tokens = 100
        engine.record_rejected_compaction()
        assert engine._no_progress_hold_active() is True
        assert engine.get_status()["no_progress_hold"]["reason"] == "host_rejected"
        assert engine.should_compress(1_000) is False
    finally:
        engine.shutdown()
