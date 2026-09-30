"""#650: the survival fit keeps LCM's summary prefix.

On Hermes the list has no system row and LCM's assembled summary is the first row (role user); the fit
used to drop it with the oldest turns. The prefix now stays whenever whole oldest turns can leave
instead and the final list (system slot, prefix, fitted tail, notice) fits the budget; otherwise the old
rule applies and a WARNING says so. Summaries are stubbed; the list is assembled by compress()."""

from __future__ import annotations

import json
import logging
import sys
import types

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
NOTICE = "[LCM survival fit:"
EMERGENCY = "LCM survival fit dropped the summary prefix"


def _rough(messages) -> int:
    """A stand-in for the host's rough request estimator (chars / 4 over content and tool calls)."""
    return sum(4 + (len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or ""))) // 4
               for m in messages)


@pytest.fixture
def host(monkeypatch):
    module = types.ModuleType("agent.model_metadata")
    module.estimate_messages_tokens_rough = _rough
    monkeypatch.setitem(sys.modules, "agent.model_metadata", module)

    def summarize(**kwargs):
        return "Earlier turns." + " summary" * 150 + "\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return _rough


def _engine(tmp_path, window=24_000) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=1600, context_threshold=0.75,
                                        database_path=str(tmp_path / "lcm.db"),
                                        threshold_full_sweep_enabled=True))
    engine.on_session_start("S", platform="telegram", context_length=window, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float, list_users: bool = False) -> list[dict]:
    text = f"[{tag}] user turn{PAD}"
    return [{"role": "user", "content": [{"type": "text", "text": text}] if list_users else text, "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _rows(engine) -> list[dict]:
    return engine._store.get_session_messages(engine._session_id, limit=100_000)


def _hidden_backlog(engine, *, system=False, list_users=False, big_newest=0) -> list[dict]:
    """Every turn stored; the host list no longer shows the oldest 48 turns (a hidden backlog)."""
    head = [{"role": "system", "content": "system prompt"}] if system else []
    turns = [r for i in range(108) for r in _turn(f"T{i:03d}", 10.0 * i, list_users)]
    if big_newest:
        turns += [{"role": "user", "content": "[N] newest", "timestamp": 5000.0},
                  {"role": "assistant", "content": "answer " * big_newest}]
    engine.ingest(head + turns)
    view = head + turns[2 * 48:]
    engine._ingest_cursor = len(view)  # the host's cursor over the list it shows
    return view


def _spy(engine, monkeypatch, *, fit=True) -> dict:
    """The list compress() hands the fit and the fit's budget; ``fit=False`` returns that list unfitted."""
    seen: dict = {}
    real_fit, real_budget = engine._survival_fit, engine._survival_fit_budget

    def spy_fit(messages, result, *args, **kwargs):
        seen["input"] = list(result) if isinstance(result, list) else result
        return real_fit(messages, result, *args, **kwargs) if fit else result

    def spy_budget(*args, **kwargs):
        seen["budget"] = real_budget(*args, **kwargs)
        return seen["budget"]

    monkeypatch.setattr(engine, "_survival_fit", spy_fit)
    monkeypatch.setattr(engine, "_survival_fit_budget", spy_budget)
    return seen


def _left(before, after) -> list[dict]:
    return [m for m in before if all(m is not kept for kept in after)]


def _today(pre, lead, budget, measure) -> list[dict]:
    """The v0.24.6 rule for an all-stored list: the oldest whole turns after the system slot leave, the
    first cut at a user row whose remainder fits (the summary prefix may leave with them)."""
    for index in range(lead + 1, len(pre)):
        if pre[index].get("role") == "user" and measure(pre[:lead] + pre[index:]) <= budget:
            return pre[:lead] + pre[index:]
    raise AssertionError("no whole-turn cut fits")


def _cold_ingest_adds(tmp_path, fitted) -> int:
    cold = _engine(tmp_path)
    try:
        before = len(_rows(cold))
        cold.ingest(fitted)
        return len(_rows(cold)) - before
    finally:
        cold.shutdown()


# -- T1 ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("list_users", [False, True], ids=["carrier-summary", "plain-summary"])
def test_t1_hermes_list_keeps_the_summary_row(tmp_path, host, monkeypatch, list_users):
    """No system row; the summary compress() assembled is the first row (a carrier when the next user row
    is a string, as Hermes merges it). Over budget: the summary stays first, only raw stored rows leave,
    the list fits, and re-ingesting it (same process and cold) stores nothing."""
    engine = _engine(tmp_path)
    try:
        view = _hidden_backlog(engine, list_users=list_users)
        seen = _spy(engine, monkeypatch)
        result = engine.compress(view, current_tokens=host(view))
        pre, budget = seen["input"], seen["budget"]
        assert pre[0]["role"] == "user" and "Summary (d" in str(pre[0]["content"])
        assert engine._survival_generated(pre[0])
        assert (engine._generated_context_carrier_remainder(pre[0]) is not None) is (not list_users)
        assert engine._survival_measure(pre) > budget and engine._last_survival_fit is not None
        assert result[0] is pre[0], str(result[0]["content"])[:60]
        dropped = _left(pre, result)
        store_ids = engine._get_store_id_map_for_messages(pre)
        assert dropped and all(id(m) in store_ids and not engine._survival_generated(m) for m in dropped)
        assert engine._survival_measure(result) <= budget and host(result) <= budget
        assert engine._last_survival_fit["dropped_rows"] == len(dropped)
        assert not any(NOTICE in str(m.get("content")) for m in result)  # no system row: no in-list notice
        stored = len(_rows(engine))
        engine.ingest(result)
        assert len(_rows(engine)) == stored
        engine.on_session_end("S", result)
        assert len(_rows(engine)) == stored
    finally:
        engine.shutdown()
    assert _cold_ingest_adds(tmp_path, result) == 0


# -- T2 ---------------------------------------------------------------------------------------------

def test_t2_system_head_keeps_the_summary_after_it_and_the_notice_in_it(tmp_path, host, monkeypatch):
    engine = _engine(tmp_path)
    try:
        view = _hidden_backlog(engine, system=True)
        seen = _spy(engine, monkeypatch)
        result = engine.compress(view, current_tokens=host(view))
        pre, budget = seen["input"], seen["budget"]
        assert pre[0]["role"] == "system" and engine._survival_generated(pre[1])
        assert engine._survival_measure(pre) > budget and engine._last_survival_fit is not None
        assert result[0]["role"] == "system" and NOTICE in result[0]["content"]
        assert result[1] is pre[1]  # the summary, unedited, right after the system row
        assert not any(NOTICE in str(m.get("content")) for m in result[1:])
        dropped = _left(pre, result)
        store_ids = engine._get_store_id_map_for_messages(pre)
        assert dropped[0] is pre[0]  # the system row is replaced by its copy carrying the notice
        assert all(id(m) in store_ids and not engine._survival_generated(m) for m in dropped[1:])
        assert engine._last_survival_fit["dropped_rows"] == len(dropped) - 1
        assert engine._survival_measure(result) <= budget
    finally:
        engine.shutdown()


# -- T3 / T4: the emergency ---------------------------------------------------------------------------

def _emergency(tmp_path, host, monkeypatch, *, list_users):
    """The assembled list, and a budget the newest turn fits but the summary prefix plus it does not."""
    engine = _engine(tmp_path)
    view = _hidden_backlog(engine, list_users=list_users, big_newest=4000)
    seen = _spy(engine, monkeypatch, fit=False)
    pre = engine.compress(view, current_tokens=host(view))
    assert pre == seen["input"] and engine._survival_generated(pre[0])
    measure = engine._survival_measure
    newest = pre[max(i for i, m in enumerate(pre) if m["role"] == "user"):]
    prefix = measure(pre[:1])
    budget = measure(newest) + prefix // 2
    assert measure(newest) <= budget < measure(pre[:1] + newest)
    del engine._survival_fit, engine._survival_fit_budget  # the spies go; the real fit runs below
    return engine, pre, budget, prefix


def test_t3_prefix_plus_newest_turn_over_budget_falls_back_to_the_old_rule(tmp_path, host, monkeypatch, caplog):
    engine, pre, budget, prefix = _emergency(tmp_path, host, monkeypatch, list_users=True)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine._survival_fit(pre, list(pre), 0, "test", request_cap=budget)
        assert engine._survival_measure(result) <= budget
        assert result == _today(pre, 0, budget, engine._survival_measure)
        assert result[0] is not pre[0] and not engine._survival_generated(result[0])
        warnings = [r.getMessage() for r in caplog.records if EMERGENCY in r.getMessage()]
        assert warnings == [f"{EMERGENCY} (emergency: prefix={prefix} tokens, budget={budget})"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("emergency", [False, True], ids=["kept", "emergency"])
def test_t4_a_carrier_in_the_prefix_is_never_silently_lost(tmp_path, host, monkeypatch, caplog, emergency):
    """The carrier (summary + a real user row) either stays, within budget, or leaves as a counted,
    stored user row with the emergency WARNING."""
    engine, pre, budget, _ = _emergency(tmp_path, host, monkeypatch, list_users=False)
    try:
        remainder = engine._generated_context_carrier_remainder(pre[0])
        assert remainder is not None
        if not emergency:
            budget = engine._survival_measure(pre) - 1  # an old turn can leave instead
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine._survival_fit(pre, list(pre), 0, "test", request_cap=budget)
        assert engine._survival_measure(result) <= budget
        dropped = _left(pre, result)
        assert engine._last_survival_fit["dropped_rows"] == len(dropped)
        assert (EMERGENCY in caplog.text) is emergency
        if emergency:
            assert dropped[0] is pre[0]  # left, and counted above as the user row it carries
            assert any(r["role"] == "user" and r["content"] == remainder for r in _rows(engine))
        else:
            assert result[0] is pre[0]
    finally:
        engine.shutdown()


# -- T5 ---------------------------------------------------------------------------------------------

def test_t5_a_list_under_budget_is_not_fitted(tmp_path, host, monkeypatch):
    engine = _engine(tmp_path, window=200_000)
    try:
        view = _hidden_backlog(engine)
        seen = _spy(engine, monkeypatch)
        result = engine.compress(view, current_tokens=host(view))
        assert engine._survival_measure(seen["input"]) <= seen["budget"]
        assert result == seen["input"] and engine._last_survival_fit is None
        assert engine._survival_generated(result[0])
    finally:
        engine.shutdown()
