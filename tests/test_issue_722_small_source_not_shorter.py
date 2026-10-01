"""#722: working routes store small, uncompressible sources whole without circuit penalties."""

import logging

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import SummaryCircuitBreaker


LONG_RESULT = "long result " * 800
SOURCE = "abc " * 783  # 784 tokens with the pinned character counter
INFO = "LCM summary result not shorter; small source stored whole"
REJECTED = "LCM summary result rejected"


@pytest.fixture(autouse=True)
def char_counter(monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: tokens._fallback_token_estimate(text) if text else 0)


def provider(monkeypatch, *answers):
    calls = []

    def invoke(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        answer = answers[min(len(calls), len(answers) - 1)]
        calls.append(model)
        return answer

    monkeypatch.setattr(escalation, "_call_llm_for_summary", invoke)
    return calls


def observe_breaker(monkeypatch, breaker):
    events = []
    for name in ("record_rejection", "record_failure", "record_success"):
        original = getattr(breaker, name)

        def record(model, *, _name=name, _original=original, **kwargs):
            events.append((_name, model))
            return _original(model, **kwargs)

        monkeypatch.setattr(breaker, name, record)
    return events


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), fresh_tail_count=2,
        leaf_chunk_tokens=400, dynamic_leaf_chunk_max=1000,
        context_threshold=0.025, threshold_full_sweep_enabled=False,
        max_assembly_tokens=100_000, summary_model="m1", condensation_fanin=2,
    ))
    instance.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    try:
        yield instance
    finally:
        instance.shutdown()


def compact_leaf(engine, monkeypatch, caplog, pad="abc " * 387):
    seen = []
    real = escalation.summarize_with_escalation

    def summarize(*args, **kwargs):
        result = real(*args, **kwargs)
        seen.append((kwargs, result))
        return result

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    view = [
        {"role": "user", "content": pad, "timestamp": 10.0},
        {"role": "assistant", "content": pad},
        {"role": "user", "content": "fresh user", "timestamp": 20.0},
        {"role": "assistant", "content": "fresh reply"},
    ]
    engine.ingest(view)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
    return view, result, seen


def rejected(caplog):
    return [r for r in caplog.records if REJECTED in r.getMessage()]


def test_leaf_stores_784_token_source_whole(engine, monkeypatch, caplog):
    calls = provider(monkeypatch, LONG_RESULT)
    events = observe_breaker(monkeypatch, engine._summary_circuit_breaker)
    _, _, seen = compact_leaf(engine, monkeypatch, caplog)
    kwargs, (summary, level) = seen[0]
    leaf, = engine._dag.get_session_nodes("S")
    assert kwargs["source_tokens"] == 784
    assert summary == leaf.summary == kwargs["text"] and level == 3
    assert engine._last_leaf_level_3_verbatim
    assert engine._dag.get_node_provenance(leaf.node_id) == {"escalation_level": 3, "model": "deterministic"}
    assert calls == ["m1"] and events == [] and not rejected(caplog)
    line = (f"{INFO} (source_tokens=784, result_tokens={tokens.count_tokens(LONG_RESULT)}, model=m1)")
    assert [r.getMessage() for r in caplog.records if INFO in r.getMessage()] == [line]
    assert any("1 level 3 leaves" in r.getMessage() for r in caplog.records)


def test_large_leaf_still_rejects(engine, monkeypatch, caplog):
    engine._config.leaf_chunk_tokens = engine._config.dynamic_leaf_chunk_max = 2000
    calls = provider(monkeypatch, LONG_RESULT)
    events = observe_breaker(monkeypatch, engine._summary_circuit_breaker)
    _, _, seen = compact_leaf(engine, monkeypatch, caplog, pad="abc " * 995)
    assert seen[0][0]["source_tokens"] == 2000
    assert calls == ["m1", "m1"] and events == [("record_rejection", "m1")] * 2
    assert len(rejected(caplog)) == 2
    assert all(r.levelno == logging.WARNING and "reason=not_shorter" in r.getMessage() for r in rejected(caplog))
    assert INFO not in caplog.text


def test_empty_small_result_still_rejects(monkeypatch, caplog):
    calls = provider(monkeypatch, "")
    breaker = SummaryCircuitBreaker()
    events = observe_breaker(monkeypatch, breaker)
    summary, level = escalation.summarize_with_escalation(
        SOURCE, source_tokens=784, token_budget=200, model="m1", circuit_breaker=breaker,
        verbatim_small_source=True)
    assert level == 3 and summary != SOURCE
    assert calls == ["m1", "m1"] and events == [("record_rejection", "m1")] * 2
    assert len(rejected(caplog)) == 2 and all("reason=no_content" in r.getMessage() for r in rejected(caplog))
    assert INFO not in caplog.text


def test_maybe_condense_stores_small_combined_source_whole(engine, monkeypatch, caplog):
    calls = provider(monkeypatch, LONG_RESULT)
    events = observe_breaker(monkeypatch, engine._summary_circuit_breaker)
    children = [SummaryNode(session_id="S", depth=0, summary="abc " * 390, token_count=391,
                            source_token_count=1000, source_ids=[], source_type="messages", created_at=float(i + 1))
                for i in range(2)]
    for child in children:
        engine._dag.add_node(child)
    combined = "\n\n---\n\n".join(child.summary for child in children)
    assert 512 < tokens.count_tokens(combined) <= 1024
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        assert engine._maybe_condense() == 1
    parent, = [node for node in engine._dag.get_session_nodes("S") if node.depth == 1]
    assert parent.summary == combined and parent.source_ids == [child.node_id for child in children]
    assert engine._dag.get_node_provenance(parent.node_id) == {"escalation_level": 3, "model": "deterministic"}
    assert calls == ["m1"] and events == [] and not rejected(caplog)


def test_caller_without_flag_still_rejects(monkeypatch, caplog):
    calls = provider(monkeypatch, LONG_RESULT)
    breaker = SummaryCircuitBreaker()
    events = observe_breaker(monkeypatch, breaker)
    summary, level = escalation.summarize_with_escalation(
        SOURCE, source_tokens=784, token_budget=200, model="m1", circuit_breaker=breaker,
        verbatim_small_source=False)
    assert level == 3 and summary != SOURCE
    assert calls == ["m1", "m1"] and events == [("record_rejection", "m1")] * 2
    assert len(rejected(caplog)) == 2 and INFO not in caplog.text


def test_first_not_shorter_result_stops_fallback_chain(monkeypatch):
    calls = provider(monkeypatch, LONG_RESULT, "short")
    provenance = {}
    assert escalation.summarize_with_escalation(
        SOURCE, source_tokens=784, token_budget=200, model="m1", fallback_models=["m2"],
        provenance=provenance, verbatim_small_source=True) == (SOURCE, 3)
    assert calls == ["m1"] and provenance == {"model": "deterministic"}


def test_second_automatic_compaction_does_not_retry_chunk(engine, monkeypatch, caplog):
    calls = provider(monkeypatch, LONG_RESULT)
    view, _, _ = compact_leaf(engine, monkeypatch, caplog)
    leaf, = engine._dag.get_session_nodes("S")
    first_calls = list(calls)
    engine.compress(view, current_tokens=engine.threshold_tokens + 1)
    assert calls == first_calls == ["m1"]
    assert [node.node_id for node in engine._dag.get_session_nodes("S")] == [leaf.node_id]


def test_six_small_results_leave_circuit_closed(monkeypatch):
    calls = provider(monkeypatch, LONG_RESULT)
    breaker = SummaryCircuitBreaker()
    events = observe_breaker(monkeypatch, breaker)
    results = [escalation.summarize_with_escalation(
        SOURCE, source_tokens=784, token_budget=200, model="m1", circuit_breaker=breaker,
        verbatim_small_source=True) for _ in range(6)]
    assert breaker.allows("m1") and breaker._rejections.get("m1", 0) == 0
    assert events == [] and calls == ["m1"] * 6 and results == [(SOURCE, 3)] * 6


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("result", [SOURCE, LONG_RESULT], ids=["equal", "longer"])
def test_not_shorter_on_either_level_returns_source(monkeypatch, level, result):
    calls = provider(monkeypatch, *([""] * (level - 1)), result)
    provenance = {}
    assert escalation.summarize_with_escalation(
        SOURCE, source_tokens=784, token_budget=200, provenance=provenance,
        verbatim_small_source=True) == (SOURCE, 3)
    assert len(calls) == level and provenance == {"model": "deterministic"}


@pytest.mark.parametrize(("bound", "whole"), [(391, False), (392, True)])
def test_threshold_is_twice_configured_bound(monkeypatch, bound, whole):
    calls = provider(monkeypatch, LONG_RESULT)
    summary, level = escalation.summarize_with_escalation(
        SOURCE, source_tokens=784, token_budget=200, l3_truncate_tokens=bound,
        verbatim_small_source=True)
    assert level == 3 and (summary == SOURCE) is whole
    assert len(calls) == (1 if whole else 2)
