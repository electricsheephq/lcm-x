"""Regression tests for forced/manual compaction spend-guard recovery."""

import logging
import sys
from types import ModuleType

from hermes_lcm.config import LCMConfig
from hermes_lcm.escalation import SummarySpendGuard


def _engine_class():
    """Provide the same minimal host stub used by the repository CI job."""
    if "agent.context_engine" not in sys.modules:
        agent_mod = sys.modules.get("agent") or ModuleType("agent")
        agent_mod.__path__ = []
        context_engine_mod = ModuleType("agent.context_engine")

        class ContextEngine:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.last_prompt_tokens = 0

            def get_status(self):
                return {}

        context_engine_mod.ContextEngine = ContextEngine
        sys.modules["agent"] = agent_mod
        sys.modules["agent.context_engine"] = context_engine_mod

    engine_module = sys.modules.get("hermes_lcm.engine")
    if engine_module is not None and not hasattr(engine_module, "LCMEngine"):
        sys.modules.pop("hermes_lcm.engine", None)
    from hermes_lcm.engine import LCMEngine

    return LCMEngine


def _engine(tmp_path, *, max_assembly_tokens=80):
    LCMEngine = _engine_class()
    config = LCMConfig(
        database_path=str(tmp_path / "spend-guard.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=8,
        max_assembly_tokens=max_assembly_tokens,
        summary_spend_max_calls=1,
        summary_spend_window_seconds=3600,
        summary_spend_backoff_seconds=3600,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine.on_session_start("spend-session", conversation_id="spend-conversation")
    return engine


def _messages():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old user context " * 80},
        {"role": "assistant", "content": "old assistant context " * 80},
        {"role": "user", "content": "fresh tail"},
    ]


def test_backoff_open_reports_only_active_backoff():
    guard = SummarySpendGuard(max_calls=1, backoff_seconds=50)

    assert guard.backoff_open(now=1000) is False
    guard.record_call(now=1000)
    assert guard.backoff_open(now=1000) is True
    assert guard.backoff_open(now=1049) is True
    assert guard.backoff_open(now=1050) is False


def test_forced_overflow_does_not_clear_backoff(tmp_path, monkeypatch, caplog):
    """Automatic forced-overflow recovery must leave an open backoff alone.

    force_overflow is set every turn the prompt exceeds the assembly cap, so
    clearing per turn would defeat the guard in the runaway-loop case it exists
    for (see the NOTE in compaction.py). The emergency still converges through
    the guard's deterministic L3 path.
    """
    engine = _engine(tmp_path)
    calls = []

    def fake_summary(**kwargs):
        calls.append(kwargs["spend_guard"].allows())
        return "forced summary", 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", fake_summary)
    engine._summary_spend_guard.record_call()
    assert engine._summary_spend_guard.backoff_open()

    try:
        with caplog.at_level(logging.INFO, logger="hermes_lcm.engine"):
            engine.compress(_messages())
        still_open = engine._summary_spend_guard.backoff_open()
    finally:
        engine.shutdown()

    assert calls and not any(calls)
    assert still_open is True
    assert "summary spend backoff cleared" not in caplog.text


def test_manual_rotate_apply_clears_backoff(tmp_path, caplog):
    engine = _engine(tmp_path, max_assembly_tokens=0)
    for index in range(4):
        engine._store.append(
            engine._session_id,
            {"role": "user", "content": f"stored-{index}"},
            source="test",
        )
    engine._store._conn.commit()
    engine._summary_spend_guard.record_call()
    assert engine._summary_spend_guard.backoff_open()

    try:
        with caplog.at_level(logging.INFO, logger="hermes_lcm.engine"):
            result = engine.rotate_active_session(apply=True)
    finally:
        engine.shutdown()

    assert result["ok"] is True
    assert engine._summary_spend_guard.backoff_open() is False
    assert "[lcm] summary spend backoff cleared (manual rotate)" in caplog.text


def test_ordinary_compress_does_not_clear_backoff(tmp_path, monkeypatch):
    engine = _engine(tmp_path, max_assembly_tokens=0)
    calls = []

    def fake_summary_call(prompt, max_tokens, model="", timeout=None):
        calls.append(model)
        return "provider must not be called while backing off"

    monkeypatch.setattr(
        "hermes_lcm.escalation._call_llm_for_summary",
        fake_summary_call,
    )
    engine._summary_spend_guard.record_call()
    assert engine._summary_spend_guard.backoff_open()

    try:
        engine.compress(_messages())
        still_open = engine._summary_spend_guard.backoff_open()
    finally:
        engine.shutdown()

    assert calls == []
    assert still_open is True
