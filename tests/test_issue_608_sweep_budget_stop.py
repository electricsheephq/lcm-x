"""#608: a spent threshold-sweep time budget is a stop condition, not an error.

Engine-level: a host list in, the returned list, the public status, the sweep telemetry and the log
lines out. Summaries are stubbed and counted; the sweep clock is a real monotonic clock plus an offset
that a wrapped engine step advances."""

from __future__ import annotations

import inspect
import logging
import re
import time

import pytest

import hermes_lcm.compaction as lcm_compaction
import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
RETRY_LINE = "retrying with smaller oldest chunk"
BUDGET_LINE = "spent its time budget before the first leaf"
CONDENSATION_LINE = "condensation stopped"


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """The token literals in this file were computed with LCM's character estimate and no host estimator;
    pin that counter so a host tree or tiktoken on the path does not change them (#615)."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


class _Clock:
    """time.monotonic() plus an offset the test advances inside an engine step."""

    def __init__(self):
        self._real = time.monotonic
        self.offset = 0.0

    def __call__(self) -> float:
        return self._real() + self.offset


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(lcm_compaction.time, "monotonic", fake)
    return fake


@pytest.fixture
def summaries(monkeypatch):
    calls: list[str] = []

    def summarize(**kwargs):
        calls.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return calls


def _engine(tmp_path, context_length: int = 200_000, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": True, "max_assembly_tokens": 100_000,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=context_length, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _view(turns: int = 6) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[row for i in range(turns) for row in _turn(f"T{i}", 10.0 * (i + 1))]]


def _advance_on_call(monkeypatch, engine, name: str, clock: _Clock, seconds: float, *, call: int | None = 1):
    """Wrap engine step ``name``: its ``call``-th invocation (every one when None) moves the sweep clock."""
    original = getattr(engine, name)
    count = 0

    def wrapped(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if call is None or count == call:
            clock.offset += seconds
        return result

    monkeypatch.setattr(engine, name, wrapped)


def _step_seconds(line: str, step: str, last: bool = False) -> float:
    """#618 item 6: a step's logged seconds. The clock is real plus an offset, so a step logs its offset plus the
    real time it took: tests assert a range from the offset, never an exact figure."""
    match = re.search(rf"\b{step}=(\d+\.\d)s{'$' if last else ''}", line)
    assert match, line
    return float(match.group(1))


def _count(caplog, text: str) -> int:
    return sum(text in record.getMessage() for record in caplog.records)


def _spend_budget_before_first_leaf(engine, view, clock, monkeypatch, caplog):
    """Test 1's state: the first store-id map of pass 0 takes the whole budget."""
    _advance_on_call(monkeypatch, engine, "_get_store_id_map_for_messages", clock, 121.0)
    engine.ingest(view)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        return engine.compress(view, current_tokens=engine.threshold_tokens + 1)


# -- 1. the budget is spent before the first leaf ------------------------------------------------------

def test_budget_spent_in_the_first_map_returns_the_input_unchanged(tmp_path, summaries, clock, monkeypatch, caplog):
    engine = _engine(tmp_path, leaf_chunk_tokens=2000)  # a multi-row chunk: the base logs two retry lines
    view = _view()
    try:
        result = _spend_budget_before_first_leaf(engine, view, clock, monkeypatch, caplog)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert result is view
        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == "threshold sweep time budget spent before the first leaf"
        assert telemetry["status"] == "noop" and telemetry["leaf_passes"] == 0
        assert telemetry["stop_reason"] == "time_budget_exhausted" and telemetry["budget_exhausted"] is True
        assert summaries == []
        assert _count(caplog, RETRY_LINE) == 0
        assert _count(caplog, BUDGET_LINE) == 1
        line = next(r.getMessage() for r in caplog.records if BUDGET_LINE in r.getMessage())
        assert "(budget 120s); steps: " in line and 121.0 <= _step_seconds(line, "anchor_ids") < 151.0  # numbers and step names only
        assert "user turn" not in line and "alpha" not in line
    finally:
        engine.shutdown()


def test_budget_spent_in_the_store_complete_step_stops_before_the_identity_anchor(
        tmp_path, summaries, clock, monkeypatch, caplog):
    """The view is all fresh tail and the owned backlog is hidden: the store-complete step is the one
    that spends the budget, and the pass leaves before the identity-anchor step."""
    engine = _engine(tmp_path)
    old = [*_turn("H1", 100.0), *_turn("H2", 110.0)]
    tail = _turn("T9", 900.0)
    anchor_calls = []
    original_anchor = engine._identity_anchor_summary_input
    monkeypatch.setattr(engine, "_identity_anchor_summary_input",
                        lambda *a, **k: anchor_calls.append(1) or original_anchor(*a, **k))
    _advance_on_call(monkeypatch, engine, "_store_complete_backlog", clock, 121.0)
    try:
        engine.ingest([*old, *tail])
        view = list(tail)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert result is view and engine._last_compression_status == "noop"
        assert telemetry["stop_reason"] == "time_budget_exhausted"
        assert anchor_calls == [] and summaries == []
        assert _count(caplog, BUDGET_LINE) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("method", "call", "step"), [
    ("_get_store_id_map_for_messages", 1, "anchor_ids"),
    ("_committed_replay_drops", 1, "replay_drops"),
    ("_get_store_id_map_for_messages", 2, "store_id_map"),
    ("_identity_anchor_summary_input", 1, "identity_anchor"),
])
def test_pass_leaves_right_after_the_step_that_spends_the_budget(
        tmp_path, summaries, clock, monkeypatch, caplog, method, call, step):
    engine = _engine(tmp_path)
    view = _view()
    _advance_on_call(monkeypatch, engine, method, clock, 121.0, call=call)
    try:
        engine.ingest(view)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert result is view and summaries == [] and engine._last_compression_status == "noop"
        line = next(r.getMessage() for r in caplog.records if BUDGET_LINE in r.getMessage())
        assert 121.0 <= _step_seconds(line, step, last=True) < 151.0  # the last timed step: nothing ran after it
    finally:
        engine.shutdown()


def test_budget_spent_in_a_provider_timeout_names_the_summariser_step(tmp_path, clock, monkeypatch, caplog):
    """The first summariser call times out after 110 s: the smaller-chunk retry finds 10 s left and the
    sweep stops before the first leaf; the WARNING shows where the time went."""
    engine = _engine(tmp_path, leaf_chunk_tokens=2000)
    view = _view()
    calls = []

    def summarize(**kwargs):
        calls.append(1)
        clock.offset += 110.0
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    try:
        engine.ingest(view)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert calls == [1] and result is view and telemetry["stop_reason"] == "time_budget_exhausted"
        assert _count(caplog, RETRY_LINE) == 1 and _count(caplog, BUDGET_LINE) == 1
        line = next(r.getMessage() for r in caplog.records if BUDGET_LINE in r.getMessage())
        assert 110.0 <= _step_seconds(line, "summariser") < 140.0
    finally:
        engine.shutdown()


# -- 2. five seconds left at the pre-call check ------------------------------------------------------------

def test_five_seconds_left_at_the_pre_call_check_makes_no_summariser_call(
        tmp_path, summaries, clock, monkeypatch, caplog):
    engine = _engine(tmp_path)
    view = _view()
    _advance_on_call(monkeypatch, engine, "_identity_anchor_summary_input", clock, 115.0)
    try:
        engine.ingest(view)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert summaries == []
        assert result is view and telemetry["stop_reason"] == "time_budget_exhausted"
        assert _count(caplog, RETRY_LINE) == 0
        with pytest.raises(lcm_engine.SweepBudgetExhausted, match="threshold full sweep time budget exhausted"):
            engine._summarize_leaf_chunk_with_rescue(view[1:5], deadline=clock() + 5.0)
        assert summaries == []
    finally:
        engine.shutdown()


# -- 3. a provider timeout with budget left keeps the smaller-chunk retry -----------------------------------

def test_provider_timeout_with_budget_left_still_retries_a_smaller_chunk(tmp_path, clock, monkeypatch, caplog):
    engine = _engine(tmp_path)
    calls = []

    def summarize(**kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1:
            raise TimeoutError("provider timed out")
        return "Earlier turns.", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    chunk = [row for i in range(3) for row in _turn(f"R{i}", 10.0 * i)]
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            used, _tokens, _text, _level, attempts = engine._summarize_leaf_chunk_with_rescue(
                chunk, deadline=clock() + 100.0)
        assert len(calls) == 2 and attempts == 2 and len(used) < len(chunk)
        assert _count(caplog, RETRY_LINE) == 1
    finally:
        engine.shutdown()


# -- 4. leaves stored, then the budget ends at a pre-call check --------------------------------------------

@pytest.mark.parametrize("left_at_call", [5.0, -5.0], ids=["five-seconds-left", "five-seconds-over"])
def test_budget_ending_after_a_stored_leaf_keeps_the_partial_result(
        tmp_path, clock, monkeypatch, caplog, left_at_call):
    engine = _engine(tmp_path, leaf_chunk_tokens=200)
    view = _view(8)
    calls = []

    def summarize(**kwargs):
        calls.append(1)
        clock.offset += 110.0  # the first leaf takes most of the budget
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    _advance_on_call(monkeypatch, engine, "_identity_anchor_summary_input", clock, 10.0 - left_at_call, call=2)
    try:
        engine.ingest(view)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert calls == [1] and result is not view and len(engine._dag.get_session_nodes("S")) == 1
        assert engine._last_compression_status == "compacted"
        assert telemetry["leaf_passes"] == 1 and telemetry["status"] == "partial"
        assert telemetry["stop_reason"] == "time_budget_exhausted" and telemetry["budget_exhausted"] is True
        assert "leaf_summary_error" not in caplog.text and _count(caplog, "sweep stopped after") == 0
        assert _count(caplog, RETRY_LINE) == 0 and _count(caplog, BUDGET_LINE) == 0
    finally:
        engine.shutdown()


# -- 5. condensation: the budget ends at its pre-call check -------------------------------------------------

@pytest.mark.parametrize("advance", [0.0, 10.0], ids=["five-seconds-left", "deadline-passes-in-group-selection"])
def test_condensation_budget_end_is_a_stop_reason_without_a_warning(
        tmp_path, clock, monkeypatch, caplog, advance):
    engine = _engine(tmp_path, condensation_fanin=2, summary_prefix_target_tokens=100)
    calls = []
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **k: calls.append(1) or ("merged", 1))
    for index in range(2):
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=f"group {index}", token_count=1000,
                                         source_token_count=2000, source_ids=[], source_type="messages",
                                         created_at=index))
    _advance_on_call(monkeypatch, engine, "_select_threshold_sweep_condensation_group", clock, advance)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            passes, reason = engine._run_threshold_sweep_condensation(
                target_tokens=100, pass_budget=5, deadline=clock() + 5.0)
        assert (passes, reason) == (0, "time_budget_exhausted")
        assert calls == [] and _count(caplog, CONDENSATION_LINE) == 0
    finally:
        engine.shutdown()


# -- 6. the hold after a no-leaf stop -----------------------------------------------------------------------

def test_no_leaf_budget_stop_holds_the_threshold_answer_only(tmp_path, summaries, clock, monkeypatch, caplog):
    engine = _engine(tmp_path)
    view = _view()
    try:
        _spend_budget_before_first_leaf(engine, view, clock, monkeypatch, caplog)
        assert engine._sweep_budget_hold_until > time.monotonic()
        assert engine.should_compress(engine.threshold_tokens + 1) is False
        assert engine.should_compress_preflight(view) is False
        assert engine.should_compress(150_000) is True  # over the 100k assembly cap: overflow recovery
        clock.offset += 601.0  # #618: the hold reads the monotonic clock
        assert engine.should_compress(engine.threshold_tokens + 1) is True
        engine._start_sweep_budget_hold()
        assert engine.should_compress(engine.threshold_tokens + 1) is False
        clock.offset = 0.0  # a normal budget: compress() is not gated, and a stored leaf clears the hold
        monkeypatch.setattr(engine, "_get_store_id_map_for_messages",
                            LCMEngine._get_store_id_map_for_messages.__get__(engine))
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert engine._last_compression_status == "compacted" and summaries
        assert engine._sweep_budget_hold_until == 0.0
        assert engine.should_compress(engine.threshold_tokens + 1) is True
    finally:
        engine.shutdown()


# -- 7. the empty anchor slice is not mapped -----------------------------------------------------------------

def test_empty_anchor_slice_is_not_mapped_and_the_pass_result_is_the_same(tmp_path, summaries, monkeypatch):
    """No system prompt: the anchor slice is empty. Skipping its map uses the value the map returns for
    an empty slice, so the pass publishes the same leaf and returns the same list as without the skip."""
    engine = _engine(tmp_path)
    view = _view()[1:]
    mapped = []
    original = engine._get_store_ids_for_messages
    monkeypatch.setattr(engine, "_get_store_ids_for_messages",
                        lambda messages, *a, **k: mapped.append(len(messages)) or original(messages, *a, **k))
    try:
        assert original([]) == []
        engine.ingest(view)
        result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        nodes = engine._dag.get_session_nodes("S")
        assert engine._last_compression_status == "compacted"
        assert [node.source_ids for node in nodes] == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        assert result[-2:] == view[-2:] and len(result) == 3
        assert 0 not in mapped
    finally:
        engine.shutdown()


# -- 8. the hold never applies at or over the survival ceiling ------------------------------------------------

def _hold(engine) -> None:
    engine._start_sweep_budget_hold()  # #618: armed for the bound conversation, on the monotonic clock


def test_hold_ends_at_the_survival_ceiling_in_should_compress(tmp_path):
    """A1: window 100,000 and reserve 0.15 make the ceiling 85,000."""
    engine = _engine(tmp_path, context_length=100_000, max_assembly_tokens=0)
    try:
        _hold(engine)
        assert engine.should_compress(84_999) is False
        assert engine.should_compress(85_000) is True
    finally:
        engine.shutdown()


def test_hold_ends_at_the_ceiling_by_the_last_prompt_size_in_preflight_without_a_replay_diff(tmp_path):
    """A2: the listed messages are far below the ceiling; the host's last prompt decides."""
    engine = _engine(tmp_path, context_length=100_000, max_assembly_tokens=0)
    view = _view()
    try:
        engine.ingest(view)
        _hold(engine)
        engine.last_prompt_tokens = 1_000
        assert engine.should_compress_preflight(view) is False
        engine.last_prompt_tokens = 85_000
        assert engine.should_compress_preflight(view) is True
    finally:
        engine.shutdown()


def test_hold_ends_at_the_ceiling_by_the_last_prompt_size_in_preflight_with_a_replay_diff(tmp_path, monkeypatch):
    """A3: the replay differs from the host list (not a cleanup the host must adopt)."""
    engine = _engine(tmp_path, context_length=100_000, max_assembly_tokens=0)
    view = _view()
    replay_diffs = []
    original = engine._ingest_messages

    def ingest_with_a_diff(messages):
        replay = [dict(message) for message in original(messages)]
        replay[1]["content"] += " (replayed)"
        replay_diffs.append(1)
        return replay

    monkeypatch.setattr(engine, "_ingest_messages", ingest_with_a_diff)
    try:
        _hold(engine)
        engine.last_prompt_tokens = 1_000
        assert engine.should_compress_preflight(view) is False
        engine.last_prompt_tokens = 85_000
        assert engine.should_compress_preflight(view) is True
        assert replay_diffs == [1, 1]
    finally:
        engine.shutdown()


def test_hold_applies_at_any_size_when_no_window_is_known(tmp_path):
    """A4: context_length 0 has no ceiling: today's hold."""
    engine = _engine(tmp_path, context_length=0, max_assembly_tokens=0)
    try:
        engine.context_length = 0
        engine.threshold_tokens = 200
        _hold(engine)
        assert engine.should_compress(10_000_000) is False
    finally:
        engine.shutdown()


def test_a_list_over_the_survival_budget_is_fitted_during_a_hold(tmp_path, summaries, clock, monkeypatch, caplog):
    """A5: every sweep spends its budget in its first map. After the first no-leaf stop, a list over the
    survival budget is still asked for and fitted: status noop, no raise, the result at or under budget."""
    engine = _engine(tmp_path, context_length=6_000)
    _advance_on_call(monkeypatch, engine, "_get_store_id_map_for_messages", clock, 121.0, call=None)
    small, large = _view(4), _view(40)
    budget = int(6_000 * (1 - 0.15))
    try:
        engine.ingest(small)
        engine.compress(small, current_tokens=engine.threshold_tokens + 1)
        assert engine._sweep_budget_hold_active() and engine._last_compression_noop_reason.startswith(
            "threshold sweep time budget spent")
        engine.ingest(large)
        tokens = engine._survival_measure(large)
        assert tokens > budget
        assert engine.should_compress(tokens) is True
        result = engine.compress(large, current_tokens=tokens)
        assert engine._last_compression_status == "noop"
        assert engine.get_status()["threshold_full_sweep"]["stop_reason"] == "time_budget_exhausted"
        assert result is not large and engine._survival_measure(result) <= budget
        assert summaries == []
    finally:
        engine.shutdown()


# -- 9. a recovery attempt comes back under the compaction threshold ------------------------------------------

THRESHOLD = 3_000  # below the request of _view(8) plus 1,000 host tokens (3,896), below the ceiling (5,100)
HOST_TOKENS = 1_000  # system prompt and tool schemas the host adds to its request estimate


def _rejected_state(tmp_path, monkeypatch, clock, *, spend=True, turns=8, threshold=THRESHOLD):
    """Window 6,000 (ceiling 5,100); every sweep spends its budget in its first map when ``spend``."""
    engine = _engine(tmp_path, context_length=6_000)
    engine.threshold_tokens = threshold
    if spend:
        _advance_on_call(monkeypatch, engine, "_get_store_id_map_for_messages", clock, 121.0, call=None)
    view = _view(turns)
    engine.ingest(view)
    return engine, view


def _shorter_by_host_score(engine, result, messages) -> bool:
    """The host's shrink score (compress_scored_by_tokens): fewer messages, or the estimate under 95%."""
    return len(result) < len(messages) or engine._survival_measure(result) < 0.95 * engine._survival_measure(messages)


def _request(engine, messages) -> int:
    return engine._survival_measure(messages) + HOST_TOKENS


def _fit_spy(monkeypatch, engine) -> list:
    caps = []
    original = engine._survival_fit

    def spy(*args, window_cap=None, request_cap=None, **kwargs):
        caps.append((window_cap, request_cap))
        return original(*args, window_cap=window_cap, request_cap=request_cap, **kwargs)

    monkeypatch.setattr(engine, "_survival_fit", spy)
    return caps


def _fit_records(engine) -> int:
    return int((engine._store.read_metadata_json("survival_fit:counter") or {}).get("count") or 0)


def _dropped_rows_are_stored(engine, view, result) -> bool:
    stored = {(r["role"], r["content"]) for r in engine._store.get_session_messages("S", limit=100_000)}
    kept = {(m["role"], m["content"]) for m in result}
    return all((m["role"], m["content"]) in stored for m in view[1:] if (m["role"], m["content"]) not in kept)


def test_rejected_request_after_a_no_leaf_stop_comes_back_shorter(tmp_path, summaries, clock, monkeypatch):
    """C1: below the ceiling, recovery attempt, the request estimate as current_tokens: the result plus the
    host tokens is under 95% of the threshold, the dropped rows are all stored, reason recovery_attempt."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock)
    request = _request(engine, view)
    assert THRESHOLD < request < int(6_000 * 0.85)
    try:
        result = engine.compress(view, current_tokens=request, bypass_cooldown=True)
        assert _shorter_by_host_score(engine, result, view) and summaries == []
        assert engine._survival_measure(result) + HOST_TOKENS <= int(THRESHOLD * 0.95)
        assert _dropped_rows_are_stored(engine, view, result)
        assert engine._last_survival_fit["reason"].startswith("recovery_attempt:")
        assert engine._last_compression_status == "noop"
    finally:
        engine.shutdown()


def test_same_state_without_a_recovery_attempt_uses_the_exit_cap(tmp_path, summaries, clock, monkeypatch):
    """C2 + #668: threshold exit gets headroom (whole turns before the fresh tail) without the recovery cap."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock)
    try:
        result = engine.compress(view, current_tokens=engine._survival_measure(view) + 1_000, bypass_cooldown=False)
        assert len(result) < len(view) and engine._last_survival_fit["reason"] == "exit_fit:noop"
        assert result[-2:] == view[-2:]  # the fresh tail (two rows) stays
        assert engine._survival_measure(result) + HOST_TOKENS <= int(THRESHOLD * 0.95)
    finally:
        engine.shutdown()


def test_a_recovery_attempt_that_shortens_the_list_runs_one_capped_fit(tmp_path, summaries, clock, monkeypatch):
    """C3: a sweep stores leaves and the list is shorter: one fit for the call, with both caps."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock, spend=False)
    caps = _fit_spy(monkeypatch, engine)
    try:
        request = _request(engine, view)
        result = engine.compress(view, current_tokens=request, bypass_cooldown=True)
        assert engine._last_compression_status == "compacted" and summaries
        assert _shorter_by_host_score(engine, result, view)
        assert caps == [(request, int(THRESHOLD * 0.95))]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("current_tokens", [None, 0])
def test_without_a_request_size_the_cap_is_the_measure_of_the_list(tmp_path, summaries, clock, monkeypatch,
                                                                   current_tokens):
    """C4: no positive current_tokens: the window cap is the measure of the list; one fit for the call."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock)
    caps = _fit_spy(monkeypatch, engine)
    try:
        result = engine.compress(view, current_tokens=current_tokens, bypass_cooldown=True)
        assert caps == [(engine._survival_measure(view), int(THRESHOLD * 0.95))]
        assert _shorter_by_host_score(engine, result, view)
        assert engine._survival_measure(result) <= int(THRESHOLD * 0.95)
    finally:
        engine.shutdown()


def test_the_next_ingest_after_a_capped_fit_stores_only_the_new_turn(tmp_path, summaries, clock, monkeypatch):
    """C5: the fitted list plus one new turn: the new turn is stored once, no old row again."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock)

    def rows():
        return sorted((r["role"], r["content"]) for r in engine._store.get_session_messages("S", limit=100_000))
    try:
        result = engine.compress(view, current_tokens=engine._survival_measure(view) + 1_000, bypass_cooldown=True)
        assert len(result) < len(view)
        before = rows()
        new = _turn("N1", 999.0)
        engine.ingest(result + new)
        after = rows()
        added = list(after)
        for row in before:
            added.remove(row)
        assert sorted(added) == sorted((m["role"], m["content"]) for m in new)
    finally:
        engine.shutdown()


def test_the_except_path_of_a_recovery_attempt_returns_the_capped_fit(tmp_path, clock, monkeypatch):
    """C6: _compress_impl raises; recovery attempt; list below the ceiling: the fitted list under the
    threshold share, one fit, no raise."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock, spend=False)
    caps = _fit_spy(monkeypatch, engine)

    def boom(*args, **kwargs):
        raise RuntimeError("injected (ids only)")

    monkeypatch.setattr(engine, "_compress_impl", boom)
    try:
        result = engine.compress(view, current_tokens=_request(engine, view), bypass_cooldown=True)
        assert _shorter_by_host_score(engine, result, view)
        assert engine._survival_measure(result) + HOST_TOKENS <= int(THRESHOLD * 0.95)
        assert engine._last_survival_fit["reason"] == "recovery_attempt:exception:RuntimeError"
        assert engine._last_compression_status == "error" and len(caps) == 1 and _fit_records(engine) == 1
    finally:
        engine.shutdown()


def test_compress_accepts_the_host_recovery_keyword(tmp_path):
    """C7 (in-process half): the host passes bypass_cooldown only when compress() names it."""
    engine = _engine(tmp_path)
    try:
        assert "bypass_cooldown" in inspect.signature(engine.compress).parameters
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("observed", "expected"), [(None, 5100), (1432 + 500, 4600), (100_000, 2100)])
def test_survival_budget_without_a_cap_is_unchanged(tmp_path, observed, expected):
    """C8: window_cap=None gives the numbers pinned at c1312f3c (window 6,000, reserve 0.15, 1,432 tokens)."""
    engine = _engine(tmp_path, context_length=6_000)
    view = [{"role": "system", "content": "system prompt"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"[{i}]{PAD}"} for i in range(8)]
    try:
        assert lcm_engine.count_messages_tokens(view) == 1432
        assert engine._survival_fit_budget(view, observed, window_cap=None) == expected
        assert engine._survival_fit_budget(view, observed) == expected
    finally:
        engine.shutdown()


# -- 10. every result of a recovery attempt fits under the threshold (Amendment 3b) ------------------------------

def _host_shrank(result, messages) -> bool:
    """compress_scored_by_tokens the host's way, with this process's estimate (no host on the path): fewer
    messages, or the estimate under 95% of the estimate before. The real-host variant is in
    test_real_host_bypass_cooldown_wire.py."""
    return len(result) < len(messages) or 0 < lcm_engine.count_messages_tokens(result) < \
        0.95 * lcm_engine.count_messages_tokens(messages)


def _one_notice(result) -> bool:
    return str(result[0]["content"]).count("[LCM survival fit: ") == 1


def _no_leaf_recovery(tmp_path, monkeypatch, clock, caplog, *, turns=8, threshold=THRESHOLD):
    """A recovery attempt whose compaction stores no leaf. With the request at or over the threshold that is
    the threshold sweep spending its budget; below it (or with no threshold) the sweep does not run, and the
    compaction is a stand-in that returns the list unchanged with status noop."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock, turns=turns, threshold=threshold)
    if not 0 < threshold <= _request(engine, view):
        def no_leaf(messages, **kwargs):
            engine._last_compression_status = "noop"
            return messages

        monkeypatch.setattr(engine, "_compress_impl", no_leaf)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        result = engine.compress(view, current_tokens=_request(engine, view), bypass_cooldown=True)
    return engine, view, result


def test_no_leaf_recovery_attempt_below_the_ceiling_fits_under_the_threshold(
        tmp_path, summaries, clock, monkeypatch, caplog):
    """D1: threshold < request < ceiling, no leaf: result plus host tokens at most int(0.95 * threshold); the
    dropped rows are stored; one counter record; reason recovery_attempt."""
    engine, view, result = _no_leaf_recovery(tmp_path, monkeypatch, clock, caplog)
    try:
        assert THRESHOLD < _request(engine, view) < int(6_000 * 0.85) and summaries == []
        assert engine._survival_measure(result) + HOST_TOKENS <= int(THRESHOLD * 0.95)
        assert _dropped_rows_are_stored(engine, view, result)
        assert _fit_records(engine) == 1 and _count(caplog, "survival fit applied") == 1
        assert engine._last_survival_fit["reason"] == "recovery_attempt:noop"
    finally:
        engine.shutdown()


def test_no_leaf_recovery_attempt_over_the_ceiling_fits_under_the_threshold_in_one_fit(
        tmp_path, summaries, clock, monkeypatch, caplog):
    """D2: a list over the survival ceiling, no leaf: the same bound, exactly one counter record and one
    notice for the call (at 7e37bd96 the list came back at the ceiling)."""
    engine, view, result = _no_leaf_recovery(tmp_path, monkeypatch, clock, caplog, turns=40)
    try:
        assert engine._survival_measure(view) > int(6_000 * 0.85)
        assert engine._survival_measure(result) + HOST_TOKENS <= int(THRESHOLD * 0.95)
        assert _dropped_rows_are_stored(engine, view, result)
        assert _fit_records(engine) == 1 and _count(caplog, "survival fit applied") == 1 and _one_notice(result)
        assert engine._last_survival_fit["reason"] == "recovery_attempt:noop"
    finally:
        engine.shutdown()


def test_recovery_attempt_whose_sweep_result_is_inside_the_budget_is_not_fitted(
        tmp_path, summaries, clock, monkeypatch):
    """D3: the sweep stores leaves and its result is inside the budget: returned as it is, no record."""
    engine, view = _rejected_state(tmp_path, monkeypatch, clock, spend=False)
    try:
        result = engine.compress(view, current_tokens=_request(engine, view), bypass_cooldown=True)
        assert engine._last_compression_status == "compacted" and summaries
        assert engine._survival_measure(result) + HOST_TOKENS <= int(THRESHOLD * 0.95)
        assert engine._last_survival_fit is None and _fit_records(engine) == 0
    finally:
        engine.shutdown()


def test_recovery_attempt_whose_sweep_result_is_over_the_budget_is_fitted_and_keeps_the_leaves(
        tmp_path, summaries, clock, monkeypatch, caplog):
    """D4: the sweep stores leaves and its result is still over the budget: fitted to the budget; the
    published leaves stay; the hold is not started."""
    threshold = 1_650  # budget int(0.95 * 1,650) - 1,000 = 567: under the sweep result, over the newest turn
    engine, view = _rejected_state(tmp_path, monkeypatch, clock, spend=False, threshold=threshold)
    caps = _fit_spy(monkeypatch, engine)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=_request(engine, view), bypass_cooldown=True)
        nodes = engine._dag.get_session_nodes("S")
        assert engine._last_compression_status == "compacted" and summaries and nodes
        assert engine._last_survival_fit["reason"] == "recovery_attempt:compacted" and len(caps) == 1
        assert engine._survival_measure(result) + HOST_TOKENS <= int(threshold * 0.95)
        assert _count(caplog, "could not reach budget") == 0 and _fit_records(engine) == 1
        assert [n.node_id for n in engine._dag.get_session_nodes("S")] == [n.node_id for n in nodes]
        assert engine._sweep_budget_hold_until == 0.0
    finally:
        engine.shutdown()


def test_without_a_threshold_the_recovery_budget_is_amendment_3s(tmp_path, summaries, clock, monkeypatch, caplog):
    """D5: threshold_tokens 0: the threshold term is left out (window 6,000, reserve 0.15)."""
    view = [{"role": "system", "content": "system prompt"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"[{i}]{PAD}"} for i in range(8)]
    engine = _engine(tmp_path, context_length=6_000)
    try:  # 1,432 counted tokens, 1,932 observed: window cap 1,932, overhead 500
        assert engine._survival_fit_budget(view, 1932, window_cap=1932, request_cap=None) == 1142
        assert engine._survival_fit_budget(view, 1932, window_cap=1932, request_cap=int(3000 * 0.95)) == 1142
        assert engine._survival_fit_budget(view, 1932, window_cap=1932, request_cap=int(1500 * 0.95)) == 925
    finally:
        engine.shutdown()
    engine, view, result = _no_leaf_recovery(tmp_path / "compress", monkeypatch, clock, caplog, threshold=0)
    try:  # request 3,896: int(3,896 * 0.85) - 1,000
        assert _count(caplog, "budget=2311)") == 1
        assert engine._last_survival_fit["reason"] == "recovery_attempt:noop"
        assert engine._survival_measure(result) <= 2311
    finally:
        engine.shutdown()


def test_recovery_attempt_below_the_threshold_keeps_amendment_3s_bound(
        tmp_path, summaries, clock, monkeypatch, caplog):
    """D6: request below the threshold: result plus host tokens at most 85% of the request."""
    engine, view, result = _no_leaf_recovery(tmp_path, monkeypatch, clock, caplog, threshold=4_500)
    try:
        request = _request(engine, view)
        assert request < 4_500
        assert engine._survival_measure(result) + HOST_TOKENS <= int(request * 0.85)
        assert engine._last_survival_fit["reason"] == "recovery_attempt:noop" and _fit_records(engine) == 1
    finally:
        engine.shutdown()


def test_a_second_recovery_attempt_shrinks_again_and_stops_where_nothing_can_leave(
        tmp_path, summaries, clock, monkeypatch, caplog):
    """D7: on the fitted list, with its smaller request: fewer rows again while stored rows can leave. With a
    budget under the newest turn: the first attempt logs the budget WARNING once; the next returns the
    identical list, no second WARNING, no second record."""
    engine, view, first = _no_leaf_recovery(tmp_path, monkeypatch, clock, caplog)
    try:
        second = engine.compress(first, current_tokens=_request(engine, first), bypass_cooldown=True)
        assert len(second) < len(first) < len(view)
        assert _dropped_rows_are_stored(engine, view, second)
    finally:
        engine.shutdown()
    caplog.clear()
    engine, view, first = _no_leaf_recovery(tmp_path / "floor", monkeypatch, clock, caplog, threshold=1_200)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            again = engine.compress(first, current_tokens=_request(engine, first), bypass_cooldown=True)
        assert len(first) < len(view) and again is first
        assert _count(caplog, "could not reach budget") == 1 and _fit_records(engine) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("turns", "threshold"), [(8, THRESHOLD), (40, THRESHOLD), (8, 4_500)],
                         ids=["D1-below-ceiling", "D2-over-ceiling", "D6-below-threshold"])
def test_recovery_results_pass_the_host_shrink_score(tmp_path, summaries, clock, monkeypatch, caplog, turns,
                                                     threshold):
    """D8 (in-process): the D1, D2 and D6 results pass the host's shrink score, computed the host's way."""
    engine, view, result = _no_leaf_recovery(tmp_path, monkeypatch, clock, caplog, turns=turns, threshold=threshold)
    try:
        assert _host_shrank(result, view)
        assert engine._last_survival_fit["reason"] == "recovery_attempt:noop"
    finally:
        engine.shutdown()


def test_the_next_ingest_after_an_over_ceiling_recovery_fit_stores_only_the_new_turn(
        tmp_path, summaries, clock, monkeypatch, caplog):
    """D9: after D2, the fitted list plus one new turn: the new turn is stored once, no old row again."""
    engine, view, result = _no_leaf_recovery(tmp_path, monkeypatch, clock, caplog, turns=40)

    def rows():
        return sorted((r["role"], r["content"]) for r in engine._store.get_session_messages("S", limit=100_000))
    try:
        assert engine._last_survival_fit["reason"] == "recovery_attempt:noop"
        before = rows()
        new = _turn("N1", 9_999.0)
        engine.ingest(result + new)
        added = list(rows())
        for row in before:
            added.remove(row)
        assert sorted(added) == sorted((m["role"], m["content"]) for m in new)
    finally:
        engine.shutdown()


# -- 11. the recovery budget rule holds for any request estimator (#615) ------------------------------------------

def _dict_text_estimate(messages) -> int:
    """A counter unlike LCM's: every message's whole dict as text, (len + 3) // 4."""
    return sum((len(str(m)) + 3) // 4 for m in messages)


def _real_host_estimate(messages) -> int:
    from agent.model_metadata import estimate_messages_tokens_rough
    return int(estimate_messages_tokens_rough(messages))


@pytest.mark.parametrize("estimate", [_dict_text_estimate, _real_host_estimate], ids=["dict-text", "real-host"])
def test_the_recovery_budget_rule_does_not_depend_on_the_counter(tmp_path, summaries, clock, monkeypatch, caplog,
                                                                  estimate):
    """One marked recovery call (bypass_cooldown=True): the logged budget is min(int(min(window, R) * (1 -
    reserve)), int(threshold * 0.95)) minus the overhead R - estimate(view), and the returned list measured by
    the same estimator is inside it. Every number is computed here from that estimator."""
    if estimate is _real_host_estimate:
        pytest.importorskip("agent.model_metadata")
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: int(estimate(messages)))
    engine, view = _rejected_state(tmp_path, monkeypatch, clock)
    request = estimate(view) + HOST_TOKENS
    if not 0 < THRESHOLD <= request:
        def no_leaf(messages, **kwargs):
            engine._last_compression_status = "noop"
            return messages

        monkeypatch.setattr(engine, "_compress_impl", no_leaf)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=request, bypass_cooldown=True)
        window = min(engine.context_length, request)
        reserve = engine._config.survival_reserve
        overhead = min(window // 2, max(0, request - estimate(view)))
        expected = min(int(window * (1 - reserve)), int(engine.threshold_tokens * 0.95)) - overhead
        logged = [int(m.group(1)) for r in caplog.records
                  if (m := re.search(r"LCM survival fit applied .*budget=(\d+)\)$", r.getMessage()))]
        assert logged == [expected]
        assert estimate(result) <= expected
        assert engine._last_survival_fit["reason"].startswith("recovery_attempt:")
    finally:
        engine.shutdown()
