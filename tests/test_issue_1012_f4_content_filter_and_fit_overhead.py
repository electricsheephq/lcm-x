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
from hermes_lcm.dag import SummaryNode

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
    engine = LCMEngine(config=LCMConfig(**config), hermes_home=str(tmp_path / "home"))
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


def test_session_end_keeps_peak_overhead_and_window_cap_still_applies(tmp_path):
    engine = _engine(tmp_path, window=65_536)
    try:
        view = [{"role": "user", "content": "ordinary turn", "timestamp": 1.0}]
        counted = tokens.count_messages_tokens(view)
        assert engine._survival_fit_budget(view, counted + 50_000, window_cap=20_000) == 7000
        assert engine._survival_fit_budget(view, counted + 2600) == int(65_536 * 0.85) - 32_768
        engine.on_session_end("unrelated", [])
        assert engine._survival_overhead_observation == ("conv", 50_000)
        engine.on_session_end("S", view)
        assert engine._survival_overhead_observation == ("conv", 50_000)
        assert engine._survival_fit_budget(view, counted + 2600) == int(65_536 * 0.85) - 32_768
    finally:
        engine.shutdown()


def test_smaller_window_does_not_erase_peak_for_later_larger_window(tmp_path):
    """Same conversation, 65,536 -> 16,384 -> 65,536: the half-window cap applies per fit, not to the stored peak."""
    engine = _engine(tmp_path, window=65_536)
    try:
        view = [{"role": "user", "content": "ordinary turn", "timestamp": 1.0}]
        counted = tokens.count_messages_tokens(view)
        assert engine._survival_fit_budget(view, counted + 32_768) == int(65_536 * 0.85) - 32_768
        engine.on_session_start("S2", platform="telegram", context_length=16_384, conversation_id="conv")
        assert engine.context_length == 16_384
        assert engine._survival_fit_budget(view, counted + 2600) == int(16_384 * 0.85) - 8192
        engine.on_session_start("S3", platform="telegram", context_length=65_536, conversation_id="conv")
        assert engine.context_length == 65_536
        assert engine._survival_fit_budget(view, counted + 2600) == int(65_536 * 0.85) - 32_768
    finally:
        engine.shutdown()


def test_commit_sequence_preserves_peak_for_next_fit(tmp_path, monkeypatch, caplog):
    """(a) Hermes in-place commit callbacks must retain the conversation peak."""
    _provider(monkeypatch)
    window = 65_536
    engine = _engine(tmp_path, window=window)
    try:
        view = _view()
        counted = tokens.count_messages_tokens(view)
        engine.ingest(view)
        compressed = engine.compress(view, current_tokens=counted + 12_500)
        assert engine._last_compression_status == "compacted"
        assert engine._compress_commit_proof is not None
        with caplog.at_level("INFO"):
            engine.on_session_end("S", view)
        assert "as a compaction commit" in caplog.text
        engine.on_session_start("S", platform="telegram", context_length=window,
                                boundary_reason="compression", old_session_id="S",
                                conversation_id="conv")
        # Keep this an ordinary ceiling fit, below the summarisation threshold.
        engine.threshold_tokens = window
        large = list(compressed)
        for i in range(20):
            large.extend([
                {"role": "user", "content": "a" * 5400, "timestamp": float(i + 10)},
                {"role": "assistant", "content": "b" * 5400},
            ])
        counted = tokens.count_messages_tokens(large)
        assert counted + 2600 >= int(window * (1 - engine._config.survival_reserve))
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
    finally:
        engine.shutdown()


def test_skipped_boundary_same_conversation_preserves_peak(tmp_path):
    """(b) Exercise the mismatched-predecessor boundary's real generic reset."""
    engine = _engine(tmp_path, window=65_536, context_threshold=0.95)
    try:
        view = [{"role": "user", "content": "ordinary turn", "timestamp": 1.0}]
        counted = tokens.count_messages_tokens(view)
        engine.compress(view, current_tokens=counted + 12_500)
        engine.on_session_start("S2", platform="telegram", context_length=65_536,
                                boundary_reason="compression", old_session_id="unrelated",
                                conversation_id="conv")
        assert engine._last_boundary_skip_time > 0
        assert engine._conversation_id == "conv"
        assert engine._survival_fit_budget(view, counted + 2600) == int(65_536 * 0.85) - 12_500
    finally:
        engine.shutdown()


@pytest.mark.parametrize("mode", ["auxiliary", "stateless", "late_auxiliary"])
def test_bypassed_compress_does_not_observe_foreground_overhead(tmp_path, mode):
    """(c) Real calling-agent markers drive thread-local and late classification."""
    engine = _engine(tmp_path, window=65_536, context_threshold=0.95,
                     stateless_session_patterns=["child"] if mode == "stateless" else [])

    class CallingAgent:
        session_id = "child"
        _parent_session_id = "S"
        _memory_write_origin = "assistant_tool"
        _memory_write_context = "foreground"
        log_prefix = ""
        enabled_toolsets = {"terminal", "file", "web"}

        def __init__(self):
            self.engine = engine

        def start(self):
            self.engine.on_session_start(self.session_id, platform="cli", conversation_id="child-conv")

        def compress(self, view, observed):
            result = self.engine.compress(view, current_tokens=observed)
            assert self.engine._bypasses_lcm_context_management()
            return result

    try:
        view = [{"role": "user", "content": "ordinary foreground", "timestamp": 1.0}]
        counted = tokens.count_messages_tokens(view)
        engine.compress(view, current_tokens=counted + 12_500)
        agent = CallingAgent()
        if mode == "auxiliary":
            agent._memory_write_origin = agent._memory_write_context = "background_review"
        agent.start()
        previous = engine._survival_overhead_observation
        if mode == "late_auxiliary":
            agent._memory_write_origin = agent._memory_write_context = "background_review"
        agent.compress(view, counted + 50_000)
        assert engine._survival_overhead_observation == previous
        if mode == "auxiliary":
            assert previous == ("conv", 12_500)
        else:
            # These starts actually bind another conversation, so (d) clears
            # the old peak; the bypassed compress must leave it unobserved.
            assert previous is None
        assert engine._store.get_session_count("child") == 0
    finally:
        engine.shutdown()


def test_different_conversation_rebind_clears_peak(tmp_path):
    """(d) A different conversation must not inherit the old peak."""
    engine = _engine(tmp_path, window=65_536, context_threshold=0.95)
    try:
        view = [{"role": "user", "content": "ordinary turn", "timestamp": 1.0}]
        counted = tokens.count_messages_tokens(view)
        engine.compress(view, current_tokens=counted + 12_500)
        engine.on_session_start("S2", platform="telegram", context_length=65_536, conversation_id="other")
        assert engine._survival_overhead_observation is None
        assert engine._survival_fit_budget(view, counted + 2600) == int(65_536 * 0.85) - 2600
    finally:
        engine.shutdown()


def test_content_filtered_condensation_group_progresses_at_level_3(tmp_path, monkeypatch):
    """A filtered condensation group is written at level 3 instead of staying on the frontier."""
    calls = _provider(monkeypatch)
    engine = _engine(tmp_path, condensation_fanin=2, l3_truncate_tokens=8)
    try:
        for index in range(2):
            text = f"[X] group {index}{PAD}"
            engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=text,
                                             token_count=escalation.count_tokens(text), source_token_count=0,
                                             source_ids=[], source_type="messages", created_at=index))
        assert engine._fit_can_rescue(False)
        assert engine._maybe_condense(force_overflow=False) == 1
        condensed = [node for node in engine._dag.get_session_nodes("S") if node.depth == 1]
        assert len(condensed) == 1
        assert engine._dag.get_node_provenance(condensed[0].node_id)["escalation_level"] == 3
        assert calls and all("[X]" in call for call in calls)
        assert engine._summary_circuit_breaker.allows("")
        assert engine._summary_circuit_breaker._failures.get("<task-default>", 0) == 0
    finally:
        engine.shutdown()


def test_profile_reset_clears_overhead_peak(tmp_path):
    engine = _engine(tmp_path, window=65_536)
    try:
        view = [{"role": "user", "content": "ordinary turn", "timestamp": 1.0}]
        engine._survival_fit_budget(view, tokens.count_messages_tokens(view) + 12_500)
        assert engine._survival_overhead_observation == ("conv", 12_500)
        engine._reset_profile_runtime_state()
        assert engine._survival_overhead_observation is None
    finally:
        engine.shutdown()
