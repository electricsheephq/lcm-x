"""#1281: a user-only mark must buy a fit, hard-cap rescue, or recovery."""
from pathlib import Path
from typing import Any

import pytest

# conftest registers the plugin as hermes_lcm at runtime.
import hermes_lcm.escalation as escalation  # type: ignore[import-not-found]
import hermes_lcm.survival_fit as survival_fit  # type: ignore[import-not-found]
import hermes_lcm.tokens as tokens  # type: ignore[import-not-found]
from hermes_lcm.config import LCMConfig  # type: ignore[import-not-found]
from hermes_lcm.engine import LCMEngine  # type: ignore[import-not-found]

from tests.harness import pin_char_counter


@pytest.mark.parametrize("regime,parallel,count,recovery,marked", [
    ("i", False, 505, False, True),
    ("ii", True, 580, False, False),
    ("iii", False, 592, False, True),
    ("608", True, 580, True, True),
])
def test_public_user_mark_regimes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, regime: str,
    parallel: bool, count: int, recovery: bool, marked: bool,
) -> None:
    monkeypatch.setattr(tokens, "_encoder_ready", True)
    monkeypatch.setattr(tokens, "_encoder", None)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)

    def provider(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("synthetic summary route outage")

    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"), temporal_rollups_enabled=False,
        reserve_tokens_floor=0, leaf_chunk_tokens=8000, threshold_full_sweep_enabled=True,
        large_output_externalization_enabled=True, large_output_active_replay_stubbing_enabled=True,
        large_output_externalization_threshold_chars=12000,
        large_output_active_replay_stub_threshold_tokens=10000,
        large_output_active_replay_stub_aged_threshold_tokens=2000)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("synthetic-probe", platform="cli", context_length=32768)
    instruction = (
        "Please examine the records in their original order and compare each observation with the earlier evidence. "
        "Explain which facts support the result and which details still need checking before we choose the next step. "
        "Keep the source records intact so that we can inspect the complete history later. ")
    user_text = (instruction * 4)[:1100]
    prose = ("I will inspect the next record and compare its details with the earlier evidence. "
             "The result should tell us which step to take next and which facts still need checking. "
             "I will keep the source intact and report what the check actually shows. ")
    text = (prose * 5)[:1100]
    calls = [{"id": f"c{i}", "type": "function", "function": {"name": "probe", "arguments": "{}"}}
             for i in range(count)]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_text, "timestamp": 1.0}]
    if parallel:
        messages.append({"role": "assistant", "content": text, "tool_calls": calls, "timestamp": 2.0})
    for i, call in enumerate(calls):
        if not parallel:
            messages.append({"role": "assistant", "content": text, "tool_calls": [call], "timestamp": 2.0 + 2 * i})
        messages.append({"role": "tool", "content": text if parallel else "ok", "tool_call_id": call["id"],
                         "name": "probe", "tool_name": "probe", "timestamp": 3.0 + 2 * i})
    try:
        engine.ingest(messages)
        before = engine._store.get_session_messages(engine.current_session_id, limit=100000)
        current = tokens.count_messages_tokens(messages)
        fitted = engine.compress(messages, current_tokens=current, bypass_cooldown=recovery)
        user = next(row for row in fitted if row["role"] == "user")
        is_marked = bool(survival_fit._PROJECTED_RE.fullmatch(user["content"]))
        measured = engine._survival_measure(fitted)
        budget = engine._survival_fit_budget(messages, current)
        print(f"regime={regime} marked={is_marked} tokens={measured} budget={budget} window={engine.context_length}")
        assert is_marked == marked
        assert budget is not None
        unmarked = [{**row, "content": user_text} if row is user else row for row in fitted]
        unmarked_tokens = engine._survival_measure(unmarked)
        if regime == "i":
            assert measured <= budget < unmarked_tokens <= engine.context_length
        elif regime == "iii":
            assert budget < measured <= engine.context_length < unmarked_tokens
        else:
            assert budget < measured <= unmarked_tokens <= engine.context_length
        if not marked:
            assert user["content"] == user_text
        assert measured <= engine.context_length
        assert engine._store.get_session_messages(engine.current_session_id, limit=100000) == before
        engine.ingest(fitted)
        assert engine._store.get_session_messages(engine.current_session_id, limit=100000) == before
    finally:
        engine.shutdown()


@pytest.mark.parametrize("host,exact_window", [
    pytest.param(False, 28200, id="fallback"),
    pytest.param(True, 56363, id="doubled-host"),
])
@pytest.mark.parametrize("delta,marked", [
    pytest.param(0, False, id="at-cap"),
    pytest.param(-1, True, id="one-over-cap"),
])
def test_public_user_mark_positive_host_overhead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    host: bool, exact_window: int, delta: int, marked: bool,
) -> None:
    pin_char_counter(monkeypatch)  # type: ignore[no-untyped-call]
    if host:
        monkeypatch.setattr(survival_fit, "_host_estimate", lambda rows: 2 * tokens.count_messages_tokens(rows))

    def provider(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("synthetic summary route outage")

    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"), temporal_rollups_enabled=False,
        reserve_tokens_floor=0, leaf_chunk_tokens=8000, threshold_full_sweep_enabled=True,
        large_output_externalization_enabled=True, large_output_active_replay_stubbing_enabled=True,
        large_output_externalization_threshold_chars=12000,
        large_output_active_replay_stub_threshold_tokens=10000,
        large_output_active_replay_stub_aged_threshold_tokens=2000,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("overhead-boundary", platform="cli", context_length=exact_window + delta)
    instruction = (
        "Please examine the records in their original order and compare each observation with the earlier evidence. "
        "Explain which facts support the result and which details still need checking before we choose the next step. "
        "Keep the source records intact so that we can inspect the complete history later. ")
    user_text = (instruction * 4)[:1100]
    prose = ("I will inspect the next record and compare its details with the earlier evidence. "
             "The result should tell us which step to take next and which facts still need checking. "
             "I will keep the source intact and report what the check actually shows. ")
    tool_text = (prose * 5)[:1100]
    calls = [{"id": f"c{i}", "type": "function", "function": {"name": "probe", "arguments": "{}"}}
             for i in range(580)]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_text, "timestamp": 3.0},
        {"role": "assistant", "content": tool_text,
         "tool_calls": calls, "timestamp": 4.0},
    ]
    messages.extend({"role": "tool", "content": tool_text, "tool_name": "probe", "name": "probe",
                     "tool_call_id": f"c{i}", "timestamp": 5.0 + i} for i in range(580))
    try:
        engine.ingest(messages)
        before = engine._store.get_session_messages(engine.current_session_id, limit=100000)
        observed = engine._survival_measure(messages) + 37
        fitted = engine.compress(messages, current_tokens=observed)
        overhead = engine._survival_host_overhead(messages, observed)
        assert overhead == 37
        users = [row for row in fitted if row["role"] == "user"]
        assert len(users) == 1
        user = users[0]
        assert bool(survival_fit._PROJECTED_RE.fullmatch(user["content"])) == marked
        if not marked:
            assert user["content"] == user_text
        unmarked = [{**row, "content": user_text} if row is user else row for row in fitted]
        unmarked_tokens = engine._survival_measure(unmarked)
        hard_cap = engine.context_length - overhead
        print(f"host={host} delta={delta} overhead={overhead} unmarked={unmarked_tokens} "
              f"hard_cap={hard_cap} fitted={engine._survival_measure(fitted)} marked={marked}")
        assert unmarked_tokens == hard_cap - delta
        budget = engine._survival_fit_budget(messages, observed)
        assert budget is not None
        measured = engine._survival_measure(fitted)
        assert budget < measured <= hard_cap
        assert engine._store.get_session_messages(engine.current_session_id, limit=100000) == before
        engine.ingest(fitted)
        assert engine._store.get_session_messages(engine.current_session_id, limit=100000) == before
    finally:
        engine.shutdown()
