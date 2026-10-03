"""#751: a summary route's context-length refusal reaches the adaptive smaller-chunk leaf rescue.

The real engine, compaction leaf loop, escalation chain and circuit breaker run; only the host
``agent.auxiliary_client.call_llm`` is faked. The fake route accepts a transcript up to ``window`` of the
first one LCM offers and refuses anything larger with a provider context-length error, so no test depends
on which token counter is installed.
"""
from __future__ import annotations

import logging
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

import hermes_lcm.escalation as escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import SummaryCircuitBreaker
from hermes_lcm.tokens import count_tokens

WINDOW = 0.85
PAD = " alpha beta gamma delta epsilon" * 40  # about 300 tokens per message
CONTEXT_LENGTH_ERROR = (
    "Error code: 400 - This model's maximum context length is 8192 tokens. "
    "However, your messages resulted in 12000 tokens. (context_length_exceeded)"
)
SERVER_ERROR = "Error code: 500 - upstream connect error"
SUMMARY = (
    "Older turns T0 onwards: the user and assistant discussed alpha through epsilon; "
    "no decisions changed.\nExpand for details about: older turns"
)
RETRY_LINE = "retrying with smaller oldest chunk after a context-length refusal"
CIRCUIT_LINE = "circuit opened"


def _install_provider(monkeypatch, window=WINDOW, error=CONTEXT_LENGTH_ERROR, refuse_model=None):
    """Fake host call_llm; returns the transcript length of every call."""
    calls: list[int] = []

    def call_llm(**kwargs):
        policy, transcript = kwargs["messages"][0]["content"], kwargs["messages"][1]["content"]
        calls.append(len(transcript))
        refused = (kwargs.get("model") == refuse_model) if refuse_model else len(transcript) > calls[0] * window
        if refused:
            raise RuntimeError(error)
        nonce = re.search(r'<lcm-summary nonce="([0-9a-f]{32})">', policy).group(1)
        reply = f'<lcm-summary nonce="{nonce}">\n{SUMMARY}\n</lcm-summary>'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    module = ModuleType("agent.auxiliary_client")
    module.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)
    return calls


def _engine(tmp_path, sweep=False, **config):
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 2000, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": sweep, "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _view():
    view = [{"role": "system", "content": "system prompt"}]
    for i in range(8):
        view += [{"role": "user", "content": f"[T{i}] user turn{PAD}", "timestamp": 10.0 * (i + 1)},
                 {"role": "assistant", "content": f"reply to T{i}{PAD}"}]
    return view


def _compress(engine, view, caplog):
    engine.ingest(view)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        return engine.compress(view, current_tokens=engine.threshold_tokens + 1)


def _state(engine, calls):
    return calls, engine._last_compression_status, engine._last_compression_noop_reason


@pytest.mark.parametrize("sweep", [False, True], ids=["no-sweep", "sweep"])
def test_t1_context_length_refusal_retries_a_smaller_chunk_and_writes_a_leaf(tmp_path, monkeypatch, caplog, sweep):
    calls = _install_provider(monkeypatch)
    engine = _engine(tmp_path, sweep=sweep)
    view = _view()
    try:
        _compress(engine, view, caplog)
        nodes = engine._dag.get_session_nodes("S", limit=1000)
        assert len(engine._store.get_session_messages("S", limit=1000)) == len(view)
        assert min(calls) <= calls[0] * WINDOW, _state(engine, calls)
        assert nodes and "deterministic truncation" not in nodes[0].summary, _state(engine, calls)
        assert nodes[0].summary.startswith("Older turns T0")
        assert RETRY_LINE in caplog.text
        assert CIRCUIT_LINE not in caplog.text and engine._summary_route_available()
    finally:
        engine.shutdown()


def test_t2_a_chunk_no_attempt_fits_keeps_level_3_and_the_652_stop(tmp_path, monkeypatch, caplog):
    calls = _install_provider(monkeypatch, window=0.0)  # every call refused for size
    engine = _engine(tmp_path)
    view = _view()
    try:
        _compress(engine, view, caplog)  # no exception escapes the rescue
        assert len(calls) == 6, calls  # three attempts, level 1 and level 2 each
        assert calls[0] > calls[2] > calls[4], calls  # each attempt offers a smaller oldest chunk
        assert engine._dag.get_session_nodes("S", limit=1000) == []
        assert engine._last_compression_noop_reason == "summary result rejected"
        assert len(engine._store.get_session_messages("S", limit=1000)) == len(view)
        # the first two attempts record no route failure; the last one counts as before, so the circuit opens
        assert CIRCUIT_LINE in caplog.text and not engine._summary_route_available()
    finally:
        engine.shutdown()


def test_t3_other_provider_errors_still_count_and_get_no_rescue(tmp_path, monkeypatch, caplog):
    calls = _install_provider(monkeypatch, window=0.0, error=SERVER_ERROR)
    engine = _engine(tmp_path)
    try:
        _compress(engine, _view(), caplog)
        assert len(calls) == 2, calls  # level 1 and level 2 of the first chunk, as before
        assert RETRY_LINE not in caplog.text
        assert CIRCUIT_LINE in caplog.text and not engine._summary_route_available()
        assert engine._last_compression_noop_reason == "summary result rejected"
    finally:
        engine.shutdown()


def test_t4_without_a_fit_the_rescue_writes_a_summary_not_a_truncation(tmp_path, monkeypatch, caplog):
    calls = _install_provider(monkeypatch)
    engine = _engine(tmp_path, survival_fit=False)  # #652 writes level 3 here, so base writes a truncation
    try:
        _compress(engine, _view(), caplog)
        nodes = engine._dag.get_session_nodes("S", limit=1000)
        assert nodes and nodes[0].summary.startswith("Older turns T0"), (_state(engine, calls), nodes)
    finally:
        engine.shutdown()


def test_t5_a_fallback_route_that_fits_answers_without_a_rescue(tmp_path, monkeypatch, caplog):
    calls = _install_provider(monkeypatch, refuse_model="small-model")
    engine = _engine(tmp_path, summary_model="small-model", summary_fallback_models=["big-model"])
    try:
        _compress(engine, _view(), caplog)
        nodes = engine._dag.get_session_nodes("S", limit=1000)
        assert len(calls) == 2 and calls[0] == calls[1]  # the same chunk, refused then answered
        assert nodes and nodes[0].summary.startswith("Older turns T0")
        assert RETRY_LINE not in caplog.text
        assert engine._summary_circuit_breaker._failures.get("small-model", 0) == 0
    finally:
        engine.shutdown()


def test_t6_without_the_flag_a_context_length_refusal_still_counts(monkeypatch):
    """Condensation, rollups and level 3 repair do not pass the flag: their behaviour is unchanged."""
    _install_provider(monkeypatch, window=0.0)
    text = "older turn alpha beta gamma " * 400
    for flag in (False, True):
        breaker = SummaryCircuitBreaker(failure_threshold=2, cooldown_seconds=300)
        provenance: dict = {}
        kwargs = {"context_length_rescue": True} if flag else {}
        summary, level = escalation.summarize_with_escalation(
            text=text, source_tokens=count_tokens(text), token_budget=200, circuit_breaker=breaker,
            provenance=provenance, **kwargs)
        assert level == 3
        assert breaker.allows("") is flag
        assert provenance.get("context_length_error", False) is flag


@pytest.mark.parametrize("message, expected", [
    (CONTEXT_LENGTH_ERROR, True),
    ("prompt is too long: 210000 tokens > 200000 maximum", True),
    ("Input too long for requested model", True),
    ("This request exceeds the model's context window", True),
    ("Rate limit reached: too many tokens per minute", False),
    ("Error code: 429 - rate_limit_exceeded", False),
    ("input length and `max_tokens` exceed context limit: 198000 + 8192 > 200000", True),
    ("Error code: 429 - token limit for this window", False),
    ("The input token count exceeds the maximum number of tokens allowed", True),
    ("You exceeded your current quota; token limit reached", False),
    (SERVER_ERROR, False),
    ("Request timed out", False),
])
def test_t7_context_length_classifier(message, expected):
    is_summary_context_length_error = escalation.is_summary_context_length_error
    assert is_summary_context_length_error(RuntimeError(message)) is expected
    assert is_summary_context_length_error(None) is False
    assert is_summary_context_length_error(TimeoutError("context length")) is False


def test_t8_a_429_status_is_never_a_context_length_refusal():
    class _StatusError(RuntimeError):
        status_code = 429

    assert escalation.is_summary_context_length_error(_StatusError("too many tokens in this window")) is False
    assert escalation.is_summary_context_length_error(RuntimeError("too many tokens in this window")) is True
