"""#682: a summary route that cannot serve its model is named once, opens its circuit on the first failure and
shows in lcm_status.

The provider is a fake ``agent.auxiliary_client.call_llm`` that behaves like the Hermes host: it fills
``route_info`` (``_record_route_info``) before the request, then raises the provider's error, whose status is
on ``status_code`` or ``response.status_code`` as the host reads it. No Hermes code is imported."""

from __future__ import annotations

import inspect
import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import SummaryCircuitBreaker

NAMED = "LCM summary route cannot serve the summary model"
PAD = " alpha beta gamma delta" * 30


class _StatusError(Exception):
    """An SDK error (openai / anthropic APIStatusError shape): the status on ``status_code``."""

    def __init__(self, message: str, status_code: int | None):
        super().__init__(message)
        self.status_code = status_code


class _ResponseError(Exception):
    """An error whose status is only on ``response.status_code`` (the host's second read)."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status_code)


ZAI_UNKNOWN_MODEL = _StatusError(
    "Error code: 400 - {'error': {'code': '1211', 'message': 'Unknown Model, please check the model code.'}}", 400)
ANTHROPIC_NOT_FOUND = _StatusError(
    "Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: gpt-5.6-sol'}}", 404)
OPENAI_DOES_NOT_EXIST = _StatusError(
    "The model `gpt-5.6-sol` does not exist or you do not have access to it.", 404)
CODEX_NOT_SUPPORTED = _StatusError(
    "The 'glm-5' model is not supported when using Codex with a ChatGPT account.", 400)


@pytest.mark.parametrize("exc", [
    ZAI_UNKNOWN_MODEL,
    ANTHROPIC_NOT_FOUND,
    OPENAI_DOES_NOT_EXIST,
    CODEX_NOT_SUPPORTED,
    _ResponseError("model_not_found: no such model", 404),
    RuntimeError("Error code: 400 - Unknown Model"),  # string only: judged by its text
    RuntimeError("The model `x` does not exist"),
], ids=["zai-400-unknown-model", "anthropic-404-not-found", "openai-does-not-exist", "codex-not-supported",
        "response-status-404", "string-only-400", "string-only-no-status"])
def test_truth_table_positives(exc):
    assert escalation.is_summary_route_config_error(exc) is True


@pytest.mark.parametrize("exc", [
    _StatusError("This model's maximum context length is 128000 tokens; the model does not exist in that size", 400),
    _StatusError("Rate limit reached for model gpt-x: too many requests", 429),
    _StatusError("model not found", 429),
    TimeoutError("model not found"),
    RuntimeError("Request timed out."),
    _StatusError("Internal server error: model not found upstream", 500),
    _ResponseError("Bad gateway: unknown model", 502),
    _StatusError("Payment required: insufficient credits for model", 402),
    _StatusError("unknown model", 402),
    _StatusError("Bad request: invalid temperature", 400),
    _StatusError("HERMES_MODEL_ADMISSION_CONSUMED: admission already used", 400),
    RuntimeError("Error code: 503 - unknown model"),
    None,
], ids=["context-length-400", "rate-limit-429", "model-token-429", "timeout-type", "timeout-text", "server-500",
        "response-502", "billing-402", "model-token-402", "400-without-model-token", "admission-consumed-400",
        "string-only-503", "none"])
def test_truth_table_negatives(exc):
    assert escalation.is_summary_route_config_error(exc) is False


# -- a Hermes-like host -----------------------------------------------------------------------------------------

def _response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _envelope(messages) -> str:
    """A valid reply under the nonce contract, taken from the request's system message."""
    nonce = messages[0]["content"].split('<lcm-summary nonce="', 1)[1].split('"', 1)[0]
    return f'<lcm-summary nonce="{nonce}">\nEarlier turns were summarised here.\nExpand for details about: turns\n</lcm-summary>'


class _Host:
    """``call_llm`` with the host's keyword-only signature and route_info; the script says fail or succeed."""

    def __init__(self, *script, route=("zai", "gpt-5.6-sol")):
        self.script, self.route, self.calls = list(script), route, []

    def call_llm(self, task=None, *, provider=None, model=None, messages, temperature=None, max_tokens=None,
                 timeout=None, reasoning_config=None, route_info=None):
        self.calls.append({"model": model, "route_info": route_info})
        if route_info is not None:  # the host fills it before the request (_record_route_info)
            route_info["provider"], route_info["model"] = self.route
        answer = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(answer, BaseException):
            raise answer
        return _response(_envelope(messages))


def _install(monkeypatch, call_llm):
    module = ModuleType("agent.auxiliary_client")
    module.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)


class _Clock:
    def __init__(self, monkeypatch):
        import time as real_time
        self.now = real_time.monotonic()
        monkeypatch.setattr(escalation, "time", SimpleNamespace(monotonic=lambda: self.now))

    def advance(self, seconds: float) -> None:
        self.now += seconds


SOURCE = "source text " * 200


def _summarize(breaker, **kwargs):
    return escalation.summarize_with_escalation(
        SOURCE, source_tokens=escalation.count_tokens(SOURCE), token_budget=50, circuit_breaker=breaker, **kwargs)


def _warnings(caplog, text: str | None = None) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING and (text is None or text in r.getMessage())]


# -- the circuit ------------------------------------------------------------------------------------------------

def test_one_config_error_opens_the_circuit_and_level_2_is_not_attempted(monkeypatch, caplog):
    host = _Host(ZAI_UNKNOWN_MODEL)
    _install(monkeypatch, host.call_llm)
    breaker = SummaryCircuitBreaker()
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
        _summary, level = _summarize(breaker)
    assert level == 3 and len(host.calls) == 1  # level 1 only: level 2 finds the route open
    assert not breaker.allows("")
    assert isinstance(host.calls[0]["route_info"], dict) and host.calls[0]["model"] is None
    (line,) = _warnings(caplog)
    assert NAMED in line and "provider=zai model=gpt-5.6-sol" in line
    assert "LCM-X sent no model (summary_model unset)" in line
    assert "auxiliary.compression.provider" in line and "auxiliary.compression.model" in line
    assert "model.provider / model.default" in line


def test_other_failures_keep_the_threshold_of_two(monkeypatch):
    host = _Host(_StatusError("Internal server error", 500))
    _install(monkeypatch, host.call_llm)
    breaker = SummaryCircuitBreaker()
    _summary, level = _summarize(breaker)
    assert level == 3 and len(host.calls) == 2  # level 1 and level 2, as before #682
    assert not breaker.allows("") and not breaker.in_config_episode("")
    assert breaker.route_status([""])["last_error_class"] == "provider_failure"


def test_the_warning_names_the_model_lcm_x_sent(monkeypatch, caplog):
    host = _Host(CODEX_NOT_SUPPORTED, route=("openai-codex", "glm-5"))
    _install(monkeypatch, host.call_llm)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        _summarize(SummaryCircuitBreaker(), model="glm-5")
    (line,) = _warnings(caplog, NAMED)
    assert "provider=openai-codex model=glm-5" in line and "LCM-X sent model 'glm-5'" in line


def test_a_host_that_fills_no_route_is_named_as_the_host_default_route(monkeypatch, caplog):
    def call_llm(task=None, *, messages, temperature=None, max_tokens=None, route_info=None):
        raise ANTHROPIC_NOT_FOUND

    _install(monkeypatch, call_llm)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        _summarize(SummaryCircuitBreaker())
    (line,) = _warnings(caplog, NAMED)
    assert "host default route" in line


def test_exactly_one_warning_across_compactions(tmp_path, monkeypatch, caplog, _pinned_counter):
    host = _Host(ZAI_UNKNOWN_MODEL)
    _install(monkeypatch, host.call_llm)
    clock = _Clock(monkeypatch)
    engine = _engine(tmp_path, sweep=False)  # no sweep deadline: the test clock only moves the cooldown
    try:
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
            for index in range(4):
                view = _view(6 + 2 * index)
                engine.ingest(view)
                engine.compress(view, current_tokens=engine.threshold_tokens + 1, force=True)
                clock.advance(engine._summary_circuit_breaker.cooldown_seconds + 1)
        assert len(host.calls) == 4  # one doomed call per compaction, level 2 never
        assert len(_warnings(caplog, NAMED)) == 1
        repeats = [r for r in caplog.records if NAMED in r.getMessage() and r.levelno == logging.DEBUG]
        assert len(repeats) == 3
        assert [r.getMessage() for r in caplog.records
                if r.name == "hermes_lcm.escalation" and r.levelno >= logging.WARNING] == _warnings(caplog, NAMED)
    finally:
        engine.shutdown()


def test_a_success_ends_the_episode(monkeypatch, caplog):
    host = _Host(ZAI_UNKNOWN_MODEL, ZAI_UNKNOWN_MODEL, "ok", ZAI_UNKNOWN_MODEL)
    _install(monkeypatch, host.call_llm)
    clock = _Clock(monkeypatch)
    breaker = SummaryCircuitBreaker()
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        for expected_level in (3, 3, 1, 3):
            assert _summarize(breaker)[1] == expected_level
            clock.advance(breaker.cooldown_seconds + 1)
            if expected_level == 1:
                assert not breaker.in_config_episode("")
                assert breaker.route_status([""])["last_error_class"] is None
    assert len(host.calls) == 4
    assert len(_warnings(caplog, NAMED)) == 2  # the first episode and the one after the success


# -- a host without the parameter ---------------------------------------------------------------------------------

def test_a_host_without_route_info_gets_no_route_info_and_raises_no_type_error(monkeypatch, caplog):
    seen = []

    def call_llm(task=None, *, provider=None, model=None, messages, temperature=None, max_tokens=None,
                 timeout=None, reasoning_config=None):
        seen.append(model)
        if len(seen) == 1:
            raise ANTHROPIC_NOT_FOUND
        return _response(_envelope(messages))

    _install(monkeypatch, call_llm)
    clock = _Clock(monkeypatch)
    breaker = SummaryCircuitBreaker()
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        assert _summarize(breaker)[1] == 3
        clock.advance(breaker.cooldown_seconds + 1)
        assert _summarize(breaker)[1] == 1
    assert len(seen) == 2 and not any("TypeError" in line or "unexpected keyword" in line for line in _warnings(caplog))
    assert "host default route" in _warnings(caplog, NAMED)[0]


def test_the_signature_is_inspected_once_per_call_llm(monkeypatch):
    host = _Host("ok")
    _install(monkeypatch, host.call_llm)
    inspected = []
    real_signature = inspect.signature

    def counting_signature(obj, *args, **kwargs):
        if getattr(obj, "__func__", None) is _Host.call_llm:
            inspected.append(obj)
        return real_signature(obj, *args, **kwargs)

    monkeypatch.setattr(escalation.inspect, "signature", counting_signature)
    monkeypatch.setattr(escalation, "_route_info_support", {})
    bound = host.call_llm
    sys.modules["agent.auxiliary_client"].call_llm = bound
    for _ in range(3):
        assert escalation._call_llm_for_summary("policy\n\nCONTENT:\ntranscript", 200)
    assert len(inspected) == 1 and all(call["route_info"] == {"provider": "zai", "model": "gpt-5.6-sol"}
                                       for call in host.calls)


# -- lcm_status -------------------------------------------------------------------------------------------------

@pytest.fixture
def _pinned_counter(monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


def _engine(tmp_path, sweep: bool = True) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(
        fresh_tail_count=2, leaf_chunk_tokens=400, context_threshold=0.001, threshold_full_sweep_enabled=sweep,
        max_assembly_tokens=100_000, database_path=str(tmp_path / "lcm.db")))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _view(turns: int) -> list[dict]:
    rows = [{"role": "system", "content": "system prompt"}]
    for i in range(turns):
        rows += [{"role": "user", "content": f"[T{i}] user turn{PAD}", "timestamp": 10.0 * (i + 1)},
                 {"role": "assistant", "content": f"reply to T{i}{PAD}"}]
    return rows


def test_lcm_status_shows_the_summary_route(tmp_path, monkeypatch, _pinned_counter):
    host = _Host(ZAI_UNKNOWN_MODEL)
    _install(monkeypatch, host.call_llm)
    engine = _engine(tmp_path)
    try:
        closed = json.loads(lcm_tools.lcm_status({}, engine=engine))["summary_route"]
        assert closed == {"state": "closed", "seconds_left": 0, "last_error_class": None, "provider": None,
                          "model": None}
        view = _view(6)
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert len(host.calls) == 1
        opened = json.loads(lcm_tools.lcm_status({}, engine=engine))["summary_route"]
        cooldown = engine._summary_circuit_breaker.cooldown_seconds
        assert opened["state"] == "open" and 0 < opened["seconds_left"] <= cooldown
        assert (opened["last_error_class"], opened["provider"], opened["model"]) == (
            "config_error", "zai", "gpt-5.6-sol")
        assert json.loads(lcm_tools.lcm_status({}, engine=engine))["config"]["summary_model"] == "(auxiliary)"
    finally:
        engine.shutdown()
