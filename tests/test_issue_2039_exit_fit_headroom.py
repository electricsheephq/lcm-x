"""#2039: over-window threshold exits retain headroom without changing the no-cut fallback."""
from __future__ import annotations

import pytest

import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

from tests.test_issue_668_exit_fit import _stored_turns


@pytest.fixture(autouse=True)
def estimators(monkeypatch):
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(survival_fit, "_host_estimate", tokens.count_messages_tokens)


@pytest.fixture
def engine(tmp_path):
    obj = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), fresh_tail_count=4,
                                   leaf_chunk_tokens=1600, context_threshold=0.75,
                                   threshold_full_sweep_enabled=True, embeddings_enabled=False))
    obj.on_session_start("S", platform="telegram", conversation_id="conv", context_length=262_144)
    yield obj
    obj.shutdown()


def test_over_window_exit_fit_leaves_host_threshold_headroom(engine):
    view = _stored_turns(engine, 300, 395)
    measure = engine._survival_measure(view)
    observed = measure + 48_602
    assert engine._survival_fit_budget(view, observed) == 174_220 < measure
    cap = int(engine.threshold_tokens * 0.95)
    assert cap == 186_777
    tail = engine._fresh_tail_start(view)
    stored = engine._store.get_session_messages("S", limit=100_000)

    result = engine._survival_fit(view, view, observed, "exit_fit:noop", request_cap=cap)

    fitted = engine._survival_measure(result)
    assert fitted + 48_602 <= cap
    assert fitted + 92_000 <= 262_144
    assert engine._last_survival_fit["reason"] == "noop"
    assert engine._survival_fit_pending_warning is not None
    assert len(result) < len(view) and result[0]["role"] == "user"
    assert result == view[len(view) - len(result):]  # whole turns only, with no projection
    assert result[-(len(view) - tail):] == view[tail:]
    assert engine._store.get_session_messages("S", limit=100_000) == stored


def test_exit_cap_without_whole_turn_cut_keeps_plain_window_fallback(engine):
    view = _stored_turns(engine, 300, 395, newest_words=100_000)
    measure = engine._survival_measure(view)
    observed = measure + 48_602
    assert engine._survival_fit_budget(view, observed) == 174_220 < measure
    cap = int(engine.threshold_tokens * 0.95)
    exit_budget = engine._survival_fit_budget(view, observed, request_cap=cap)
    assert engine._survival_measure(view[-2:]) > exit_budget

    result = engine._survival_fit(view, view, observed, "exit_fit:noop", request_cap=cap)

    # Base 32d3c3e2: 40 ordinary turns (605 each) + newest (150,013) fit the plain budget.
    # One more ordinary turn does not fit, so today's uncapped recursion keeps rows 518 onward.
    assert result == view[518:]
    assert engine._survival_measure(result) == 174_213
    assert engine._last_survival_fit["reason"] == "noop"
    assert engine._last_survival_fit["reached_budget"] is True
    assert engine._survival_fit_pending_warning is not None
