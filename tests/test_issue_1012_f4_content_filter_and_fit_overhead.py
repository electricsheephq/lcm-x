"""#1012 F4: chunk-local filtering and conversation-local survival overhead."""
from __future__ import annotations

import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta epsilon" * 40
SUMMARY = "Earlier ordinary turns.\nExpand for details about: ordinary turns"
FILTER_BODY = {"error": {"code": "1301", "message": "contentFilter"}}


class HTTPError(RuntimeError):
    def __init__(self, status, body):
        super().__init__("summary request refused")
        self.status_code, self.body = status, body


@pytest.fixture(autouse=True)
def _hermetic_counter(monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(survival_fit, "_host_estimate", tokens.count_messages_tokens)


def _provider(monkeypatch, status=400):
    calls = []

    def call_llm(**kwargs):
        policy, transcript = (row["content"] for row in kwargs["messages"])
        calls.append(transcript)
        if "[X]" in transcript:
            raise HTTPError(status, FILTER_BODY if status == 400 else {"error": "upstream error"})
        nonce = re.search(r'<lcm-summary nonce="([0-9a-f]{32})">', policy).group(1)
        content = f'<lcm-summary nonce="{nonce}">\n{SUMMARY}\n</lcm-summary>'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    module = ModuleType("agent.auxiliary_client")
    module.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)
    return calls


def _engine(tmp_path, window=200_000, **overrides):
    config = {"database_path": str(tmp_path / "lcm.db"), "fresh_tail_count": 2,
              "leaf_chunk_tokens": 800, "dynamic_leaf_chunk_enabled": False,
              "context_threshold": 0.001, "threshold_full_sweep_enabled": False,
              "l3_truncate_tokens": 128, "summary_circuit_breaker_failure_threshold": 2,
              **overrides}
    engine = LCMEngine(config=LCMConfig(**config))
    engine.on_session_start("S", platform="telegram", context_length=window, conversation_id="conv")
    return engine


def _view():
    messages = [{"role": "system", "content": "system prompt"}]
    for i, tag in enumerate(("X", "Y", "Z", "tail")):
        messages.extend([
            {"role": "user", "content": f"[{tag}] user{PAD}", "timestamp": float(i + 1)},
            {"role": "assistant", "content": f"reply{PAD}"},
        ])
    return messages


def test_content_filter_stores_l3_then_l1_and_advances_backlog(tmp_path, monkeypatch):
    calls = _provider(monkeypatch)
    summaries = []
    real_summary = engine_module.summarize_with_escalation

    def record_summary(**kwargs):
        result = real_summary(**kwargs)
        summaries.append((kwargs["text"], dict(kwargs["provenance"])))
        return result

    monkeypatch.setattr(engine_module, "summarize_with_escalation", record_summary)
    engine = _engine(tmp_path, threshold_full_sweep_enabled=True)
    try:
        view = _view()
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert engine._summary_circuit_breaker.allows("")
        assert engine._summary_circuit_breaker._failures.get("<task-default>", 0) == 0
        assert engine._summary_circuit_breaker._rejections.get("<task-default>", 0) == 0
        nodes = sorted(engine._dag.get_session_nodes("S"), key=lambda node: node.node_id)
        assert len(nodes) >= 2
        first = nodes[0]
        assert first.summary == escalation._deterministic_truncate(summaries[0][0], 128)
        assert summaries[0][1]["content_filter_error"] is True
        assert engine._dag.get_node_provenance(first.node_id) == {
            "escalation_level": 3, "model": "deterministic"}
        assert nodes[1].summary == SUMMARY
        assert engine._dag.get_node_provenance(nodes[1].node_id)["escalation_level"] == 1
        assert "[Y]" in calls[2] and "[X]" not in calls[2]
        assert min(nodes[1].source_ids) > max(first.source_ids)
        assert engine._last_compacted_store_id >= max(nodes[1].source_ids)
        assert engine._summary_circuit_breaker.allows("")
        assert engine._last_leaf_content_filter_error is False
        assert len(engine._store.get_session_messages("S")) == len(view)
    finally:
        engine.shutdown()


def test_server_errors_still_open_route_after_configured_failures(tmp_path, monkeypatch):
    calls = _provider(monkeypatch, status=500)
    engine = _engine(tmp_path)
    try:
        view = _view()
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert len(calls) == 2
        assert engine._summary_circuit_breaker._failures["<task-default>"] == 2
        assert not engine._summary_circuit_breaker.allows("")
        assert engine._dag.get_session_nodes("S") == []
    finally:
        engine.shutdown()


@pytest.mark.parametrize("status,body,expected", [
    (400, FILTER_BODY, True),
    (400, "content_filter", True),
    (400, "content_policy_violation", True),
    (400, {"code": 1301, "message": "sensitive input"}, True),
    (400, {"error": "invalid_request_error", "message": "invalid parameter"}, False),
    (400, {"code": 1301}, False),
    (400, {"code": 13010, "message": "sensitive input"}, False),
    (500, FILTER_BODY, False),
    (429, "content_filter", False),
])
def test_content_filter_classifier(status, body, expected):
    assert escalation.is_summary_content_filter_error(HTTPError(status, body)) is expected


def test_content_filter_classifier_existing_status_shapes():
    assert escalation.is_summary_content_filter_error(RuntimeError("Error code: 400 - contentFilter"))
    exc = RuntimeError("refused")
    exc.response = SimpleNamespace(status_code=400, text='{"error":"content_filter"}')
    assert escalation.is_summary_content_filter_error(exc)
    assert not escalation.is_summary_content_filter_error(RuntimeError("contentFilter"))
    assert not escalation.is_summary_content_filter_error(TimeoutError("Error code: 400 - contentFilter"))
    assert not escalation.is_summary_content_filter_error(None)


def test_content_filter_records_provenance_without_any_circuit_event(monkeypatch):
    _provider(monkeypatch)
    breaker = escalation.SummaryCircuitBreaker(failure_threshold=2, cooldown_seconds=300)
    for method in ("record_failure", "record_config_error", "record_rejection"):
        monkeypatch.setattr(breaker, method, lambda *args, **kwargs: pytest.fail("filter recorded circuit event"))
    provenance = {}
    text = f"[X]{PAD * 2}"
    summary, level = escalation.summarize_with_escalation(
        text=text, source_tokens=tokens.count_tokens(text), token_budget=200,
        l3_truncate_tokens=128, circuit_breaker=breaker, provenance=provenance)
    assert level == 3 and summary == escalation._deterministic_truncate(text, 128)
    assert provenance["content_filter_error"] is True
    assert breaker.allows("")


def test_fit_remembers_largest_compress_overhead_and_rebind_resets(tmp_path, monkeypatch):
    _provider(monkeypatch)
    window = 65_536
    engine = _engine(tmp_path, window=window, context_threshold=0.95)
    try:
        small = [{"role": "user", "content": "ordinary first turn", "timestamp": 1.0}]
        engine.ingest(small)
        engine.compress(small, current_tokens=tokens.count_messages_tokens(small) + 12_500)
        large = list(small)
        for i in range(20):
            large.extend([
                {"role": "user", "content": "a" * 5400, "timestamp": float(i + 2)},
                {"role": "assistant", "content": "b" * 5400},
            ])
        counted = tokens.count_messages_tokens(large)
        assert counted + 2600 >= int(window * (1 - engine._config.survival_reserve))
        engine.ingest(large)
        budgets = []
        real_budget = engine._survival_fit_budget

        def record_budget(*args, **kwargs):
            budget = real_budget(*args, **kwargs)
            budgets.append(budget)
            return budget

        monkeypatch.setattr(engine, "_survival_fit_budget", record_budget)
        fitted = engine.compress(large, current_tokens=counted + 2600)
        expected = int(window * (1 - engine._config.survival_reserve)) - 12_500
        assert budgets and set(budgets) == {expected}
        assert tokens.count_messages_tokens(fitted) <= expected
        assert len(fitted) < len(large)
        assert len(engine._store.get_session_messages("S")) == len(large)
        engine.on_session_start("S2", platform="telegram", context_length=window, conversation_id="other")
        assert engine._survival_fit_budget(large, counted + 2600) == int(window * 0.85) - 2600
        engine.on_session_start("S2", platform="telegram", context_length=window, conversation_id="conv")
        assert engine._survival_fit_budget(large, counted + 2600) == int(window * 0.85) - 2600
    finally:
        engine.shutdown()


def test_session_end_clears_overhead_and_window_cap_still_applies(tmp_path):
    engine = _engine(tmp_path, window=65_536)
    try:
        view = [{"role": "user", "content": "ordinary turn", "timestamp": 1.0}]
        counted = tokens.count_messages_tokens(view)
        assert engine._survival_fit_budget(view, counted + 50_000, window_cap=20_000) == 7000
        assert engine._survival_fit_budget(view, counted + 2600) == int(65_536 * 0.85) - 32_768
        engine.on_session_end("unrelated", [])
        assert engine._survival_overhead_observation == ("conv", 50_000)
        engine.on_session_end("S", view)
        assert engine._survival_overhead_observation is None
        assert engine._survival_fit_budget(view, counted + 2600) == int(65_536 * 0.85) - 2600
    finally:
        engine.shutdown()
