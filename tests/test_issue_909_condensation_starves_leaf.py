"""#909: pre-leaf condensation reserves a leaf call; stored progress and replay cleanup respect holds.

The provider only advances a fake clock. The real escalation, publication, preflight and compaction run.
"""

import time

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import extract_externalized_ref, load_externalized_payload

PAD = " alpha beta gamma delta" * 30
SUMMARY = "Earlier turns.\nExpand for details about: turns"


@pytest.fixture(autouse=True)
def _char_counter(monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        offset = 0.0

        def __call__(self):
            return 1000.0 + self.offset

    fake = Clock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def build(**overrides):
        settings = dict(database_path=str(tmp_path / "lcm.db"), fresh_tail_count=2, leaf_chunk_tokens=400,
                        context_threshold=0.001, threshold_full_sweep_enabled=True,
                        max_assembly_tokens=100_000, l3_truncate_tokens=2, condensation_fanin=2,
                        foreground_soft_seconds=0)
        settings.update(overrides)
        engine = LCMEngine(config=LCMConfig(**settings), hermes_home=str(tmp_path / "hermes"))
        engine.on_session_start("S", context_length=200_000, conversation_id="conv")
        engines.append(engine)
        return engine

    yield build
    for engine in engines:
        engine.shutdown()


def _view():
    rows = [{"role": "system", "content": "system prompt"}]
    for i in range(12):
        rows += [{"role": "user", "content": f"[T{i}] user turn{PAD}", "timestamp": 10.0 * (i + 1)},
                 {"role": "assistant", "content": f"reply to T{i}{PAD}"}]
    return rows


def _frontier(engine):
    ids = []
    for i in range(2):
        ids.append(engine._dag.add_node(SummaryNode(
            session_id="S", depth=0, summary=f"group {i}{PAD * 4}", token_count=1000,
            source_token_count=2000, source_ids=[], source_type="messages", created_at=float(i + 1))))
    return ids


@pytest.mark.parametrize("unchanged_frontier", [False, True], ids=["shrunk", "unchanged"])
def test_stored_condensation_then_estimate_refused_leaf_is_partial_without_a_hold(
        make_engine, monkeypatch, clock, unchanged_frontier):
    engine = make_engine(foreground_hard_seconds=100)
    _frontier(engine)
    if unchanged_frontier:
        monkeypatch.setattr(engine, "_summary_frontier_tokens", lambda: 2000)
    calls = []

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append(timeout)
        clock.offset += 55.0
        return SUMMARY

    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    view = _view()
    engine.ingest(view)
    engine.compress(view, current_tokens=engine.threshold_tokens + 1)
    telemetry = engine.get_status()["threshold_full_sweep"]
    assert len(calls) == 1
    assert telemetry["pre_leaf_condensation_passes"] == 1
    assert telemetry["leaf_passes"] == 0
    assert telemetry["stop_reason"] == "time_budget_exhausted"
    assert telemetry["status"] == "partial"
    assert not engine._sweep_budget_hold_active()
    assert not engine._no_progress_hold_active()


def test_slow_rejected_condensation_is_budget_cut_and_the_reserved_leaf_is_stored(
        make_engine, monkeypatch, clock):
    engine = make_engine(summary_timeout_ms=300_000)
    source_ids = _frontier(engine)
    calls = []
    condensation_calls = 0
    failures = []
    record_failure = engine._summary_circuit_breaker.record_failure
    monkeypatch.setattr(engine._summary_circuit_breaker, "record_failure",
                        lambda route: failures.append(route) or record_failure(route))

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        nonlocal condensation_calls
        is_leaf = "user turn" in prompt
        seconds = 30.0 if is_leaf or condensation_calls else 70.0
        calls.append((is_leaf, timeout, clock.offset))
        if not is_leaf:
            condensation_calls += 1
        if seconds > timeout:
            clock.offset += timeout
            raise TimeoutError("provider timed out")
        clock.offset += seconds
        return SUMMARY if is_leaf else prompt * 3

    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    view = _view()
    engine.ingest(view)
    engine.compress(view, current_tokens=engine.threshold_tokens + 1)
    telemetry = engine.get_status()["threshold_full_sweep"]
    assert [leaf for leaf, _timeout, _start in calls] == [False, False, True]
    assert [timeout for _leaf, timeout, _start in calls] == pytest.approx([85, 15, 30], abs=0.5)
    assert telemetry["pre_leaf_condensation_stop_reason"] == "time_budget_exhausted"
    assert telemetry["pre_leaf_condensation_passes"] == 0
    assert telemetry["leaf_passes"] == 1
    assert failures == []  # the L2 timeout is a budget cut, not a provider failure
    assert not engine._summary_circuit_breaker._open_until
    assert not engine._sweep_budget_hold_active()
    assert not engine._no_progress_hold_active()
    assert all(engine._dag.get_node(node_id) is not None for node_id in source_ids)
    assert not any(node.depth > 0 for node in engine._dag.get_session_nodes("S"))
    assert clock.offset <= 115.5


def test_less_than_minimum_condensation_time_skips_to_the_leaf(make_engine, monkeypatch, clock):
    engine = make_engine(summary_timeout_ms=300_000)
    _frontier(engine)
    calls = []
    prepare = engine._prepare_retained_user_anchor

    def slow_prework(messages):
        prepare(messages)
        clock.offset += 75

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append(prompt)
        clock.offset += 30
        return SUMMARY

    view = _view()
    engine.ingest(view)
    monkeypatch.setattr(engine, "_prepare_retained_user_anchor", slow_prework)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    engine.compress(view, current_tokens=engine.threshold_tokens + 1)
    telemetry = engine.get_status()["threshold_full_sweep"]
    assert len(calls) == 1 and "user turn" in calls[0]
    assert telemetry["pre_leaf_condensation_passes"] == 0 and telemetry["leaf_passes"] == 1


@pytest.mark.parametrize("hold", ["sweep", "no_progress"])
def test_held_replay_cleanup_is_adopted_without_a_summariser_call(make_engine, monkeypatch, clock, hold):
    engine = make_engine(large_output_externalization_enabled=True,
                         large_output_externalization_threshold_chars=1_000_000,
                         large_output_active_replay_stubbing_enabled=True,
                         large_output_active_replay_stub_threshold_tokens=5)
    calls = []
    monkeypatch.setattr(escalation, "_invoke_summary_llm", lambda *args, **kwargs: calls.append(args) or SUMMARY)
    payload = "durable live tool payload " * 100
    view = [*_view(),
            {"role": "assistant", "content": "running tool", "tool_calls": [
                {"id": "live-call", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "live-call", "content": payload}]
    if hold == "sweep":
        engine._start_sweep_budget_hold()
    else:
        engine._start_no_progress_hold("no_progress")
    assert engine.should_compress_preflight(view) is True
    result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
    assert calls == []
    assert engine._dag.get_session_nodes("S") == []
    assert engine.last_compression_status == "sanitized"
    stub = next(row["content"] for row in result if row.get("tool_call_id") == "live-call")
    ref = extract_externalized_ref(stub)
    assert ref is not None
    recovered = load_externalized_payload(ref, config=engine._config, hermes_home=engine._hermes_home)
    assert recovered is not None and recovered["content"] == payload


def test_sweep_ranks_oldest_group_and_avoids_light_group_stall(make_engine, monkeypatch, clock):
    engine = make_engine()
    ids = []
    for index, (depth, token_count) in enumerate([(0, 1), (0, 1), (0, 10_000), (1, 1_000), (1, 1_000)]):
        ids.append(engine._dag.add_node(SummaryNode(
            session_id="S", depth=depth,
            summary="x" if token_count == 1 else f"group {index}{PAD * 4}",
            token_count=token_count, source_token_count=token_count * 2,
            source_ids=[], source_type="messages" if depth == 0 else "nodes",
            created_at=float(index + 1),
        )))
    calls = []
    monkeypatch.setattr(escalation, "_invoke_summary_llm", lambda *args, **kwargs: calls.append(args) or SUMMARY)
    before = engine._summary_frontier_tokens()
    passes, reason = engine._run_threshold_sweep_condensation(
        target_tokens=before - 1_000, pass_budget=1, deadline=clock() + 100,
    )
    assert (passes, reason) == (1, "summary_prefix_target_reached")
    assert engine._summary_frontier_tokens() < before
    assert len(calls) == 1
    condensed = [node for node in engine._dag.get_session_nodes("S") if node.source_ids]
    assert len(condensed) == 1 and condensed[0].source_ids == ids[-2:]
    assert all(engine._dag.get_node(node_id) is not None for node_id in ids)
