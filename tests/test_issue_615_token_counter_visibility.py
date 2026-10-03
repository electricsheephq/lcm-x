"""#615: disclose the token counter without starting or waiting on its loader."""

import json
import time

import pytest

from hermes_lcm import tokens as tokens_mod
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture(autouse=True)
def _reset_encoder_state(monkeypatch):
    monkeypatch.setattr(tokens_mod, "_encoder", None)
    monkeypatch.setattr(tokens_mod, "_encoder_ready", False)
    monkeypatch.setattr(tokens_mod, "_encoder_thread", None)
    # Retire fake counts independently of the production generation.
    monkeypatch.setattr(tokens_mod, "_encoder_generation", -615)
    tokens_mod._count_tokens_cached.cache_clear()
    yield
    tokens_mod._count_tokens_cached.cache_clear()


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def make(name="lcm_615"):
        config = LCMConfig(database_path=str(tmp_path / f"{name}.db"))
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / name))
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.shutdown()


class FakeEncoder:
    def encode(self, text):
        return list(text)


def _status():
    fn = getattr(tokens_mod, "token_counter_status", None)
    assert fn is not None, "token_counter_status missing"
    return fn()


def _fallback(monkeypatch):
    monkeypatch.setattr(tokens_mod, "_encoder_ready", True)


def _adopted(monkeypatch):
    monkeypatch.setattr(tokens_mod, "_encoder", FakeEncoder())
    monkeypatch.setattr(tokens_mod, "_encoder_ready", True)


def _doctor(engine):
    payload = json.loads(engine.handle_tool_call("lcm_doctor", {}))
    checks = [check for check in payload["checks"] if check["check"] == "token_counter"]
    assert len(checks) == 1, "expected exactly one token_counter check"
    return payload, checks[0]


def test_status_reports_tiktoken_when_encoder_adopted(monkeypatch):
    _adopted(monkeypatch)
    assert _status() == {
        "counter": "tiktoken", "encoding": "cl100k_base", "state": "ready",
    }


def test_status_reports_char_estimate_when_load_failed(monkeypatch):
    _fallback(monkeypatch)
    assert _status() == {
        "counter": "char_estimate", "encoding": "", "state": "unavailable",
    }


def test_status_reports_char_estimate_while_loading(monkeypatch):
    monkeypatch.setattr(tokens_mod, "_encoder_thread", object())
    assert _status() == {
        "counter": "char_estimate", "encoding": "", "state": "loading",
    }


def test_status_reports_not_loaded_before_first_count():
    assert _status() == {
        "counter": "not_loaded", "encoding": "", "state": "not_started",
    }


def test_status_read_never_starts_or_waits_on_loader(monkeypatch):
    calls = []
    monkeypatch.setattr(tokens_mod, "_load_encoder", lambda: calls.append("load"))
    monkeypatch.setattr(tokens_mod, "_ENCODER_FIRST_WAIT_S", 5.0)
    start = time.monotonic()
    first = _status()
    second = _status()
    elapsed = time.monotonic() - start
    assert first == second == {
        "counter": "not_loaded", "encoding": "", "state": "not_started",
    }
    assert tokens_mod._encoder_thread is None
    assert calls == []
    assert elapsed < 0.5


def test_lcm_doctor_tool_reports_token_counter_as_pass_on_fallback(monkeypatch, make_engine):
    _fallback(monkeypatch)
    engine = make_engine()
    engine._session_id = "test-session"
    engine.context_length = 200000
    payload, check = _doctor(engine)
    assert check["status"] == "pass"
    assert check["detail"] == {
        "counter": "char_estimate", "encoding": "", "state": "unavailable",
    }
    assert payload["overall"] == "healthy"
    assert all(check["status"] == "pass" for check in payload["checks"])
    assert not any(item["check"] == "token_counter" for item in payload["guidance"])


def test_lcm_doctor_tool_reports_tiktoken_detail(monkeypatch, make_engine):
    _adopted(monkeypatch)
    _, check = _doctor(make_engine())
    assert check["status"] == "pass"
    assert check["detail"] == {
        "counter": "tiktoken", "encoding": "cl100k_base", "state": "ready",
    }


def test_lcm_status_tool_payload_names_counter(monkeypatch, make_engine):
    _fallback(monkeypatch)
    engine = make_engine()
    engine.on_session_start("test-session", context_length=100000)
    active = json.loads(engine.handle_tool_call("lcm_status", {}))
    unbound = json.loads(make_engine("unbound").handle_tool_call("lcm_status", {}))
    expected = {"counter": "char_estimate", "encoding": "", "state": "unavailable"}
    assert active.get("token_counter") == expected
    assert unbound["error"] == "No active session"
    assert unbound.get("token_counter") == expected


def test_status_and_doctor_text_name_counter(monkeypatch, make_engine):
    _fallback(monkeypatch)
    engine = make_engine()
    engine._session_id = "test-session"
    engine.context_length = 200000
    status = handle_lcm_command("status", engine)
    doctor = handle_lcm_command("doctor", engine)
    for text in (status, doctor):
        assert "token_counter: char_estimate" in text
        assert "token_counter_encoding: (none)" in text
        assert "token_counter_state: unavailable" in text
    assert "status: ok" in doctor
    assert "triage_guidance:\n- none" in doctor
    issues = doctor.split("issues:", 1)[1].split("observations:", 1)[0]
    actions = doctor.split("recommended_actions:", 1)[1].split("triage_guidance:", 1)[0]
    assert "token_counter" not in issues
    assert "token_counter" not in actions
