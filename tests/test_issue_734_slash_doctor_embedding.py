"""#734: the slash doctor reports the same embedding health as the tool."""

import json

import pytest

from hermes_lcm import tools as lcm_tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.vector_store import VectorStore


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    engines = []

    def make(scenario):
        config = LCMConfig(
            database_path=str(tmp_path / f"{scenario}.db"),
            embeddings_enabled=scenario == "unavailable",
            embedding_provider="fastembed",
            embedding_model="BAAI/bge-small-en-v1.5",
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
        engines.append(engine)
        engine._session_id = "test-session"
        engine.context_length = 200000
        if scenario == "active_profile":
            store = VectorStore(engine._store.db_path)
            try:
                store.register_profile(config.embedding_model, config.embedding_provider, 3)
            finally:
                store.close()
        monkeypatch.setattr(
            lcm_tools,
            "probe_provider_availability",
            lambda _config: {
                "provider": "fastembed",
                "model": config.embedding_model,
                "available": False,
                "probed": False,
                "detail": "fastembed optional dependency is unavailable",
            },
        )
        return engine

    yield make
    for engine in engines:
        engine.shutdown()


def _issues(text):
    return next(line for line in text.splitlines() if line.startswith("issues:"))


def test_t1_enabled_provider_unavailable(make_engine):
    text = handle_lcm_command("doctor", make_engine("unavailable"))

    assert "embedding_provider_health: warn (fastembed optional dependency is unavailable)" in text
    assert "embedding_provider_health" in _issues(text)
    assert "status: issues-found" in text
    assert "/lcm embed warmup" in text
    assert "- embedding_provider_health: inspect warning-only" in text


def test_t2_disabled_with_active_profile(make_engine):
    text = handle_lcm_command("doctor", make_engine("active_profile"))

    assert "embedding_provider_health: warn (embeddings are disabled" in text
    assert "embedding_provider_health" in _issues(text)
    assert "status: issues-found" in text
    assert "embeddings are off" in text
    assert "reinstall" not in text and "pip install" not in text


def test_t3_disabled_without_profile(make_engine):
    text = handle_lcm_command("doctor", make_engine("no_profile"))

    assert (
        "embedding_provider_health: pass semantic embeddings are disabled "
        "(LCM_EMBEDDINGS_ENABLED=false)"
    ) in text
    assert "embedding_provider_health" not in _issues(text)
    assert "status: ok" in text
    assert "triage_guidance:\n- none" in text


def test_t4_check_exception_does_not_break_doctor(make_engine, monkeypatch):
    engine = make_engine("no_profile")
    calls = []

    def raise_check(candidate):
        calls.append(candidate)
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(lcm_tools, "_embedding_provider_health_check", raise_check)
    text = handle_lcm_command("doctor", engine)

    assert "LCM doctor" in text
    assert "embedding_provider_health" not in text
    assert "status: ok" in text
    assert calls == [engine]


@pytest.mark.parametrize("scenario", ["unavailable", "active_profile", "no_profile"])
def test_t5_slash_and_tool_status_parity(make_engine, scenario):
    engine = make_engine(scenario)
    payload = json.loads(engine.handle_tool_call("lcm_doctor", {}))
    check = next(item for item in payload["checks"] if item["check"] == "embedding_provider_health")
    text = handle_lcm_command("doctor", engine)

    line = next(
        (line for line in text.splitlines() if line.startswith("embedding_provider_health:")),
        None,
    )
    assert line is not None, "slash doctor must expose the embedding check"
    assert line.split()[1] == check["status"]
