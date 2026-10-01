"""YAML absolute compaction trigger and environment precedence (#48)."""

import json

import pytest

import hermes_lcm.config as config_module
from hermes_lcm import tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for key in ("LCM_ABSOLUTE_THRESHOLD_TOKENS", "LCM_CONTEXT_THRESHOLD", "LCM_MODEL_THRESHOLDS"):
        monkeypatch.delenv(key, raising=False)
    engines = []

    def make(yaml_text, *, env=None, manual=False, autoraise=False):
        (hermes_home / "config.yaml").write_text(yaml_text)
        if env is not None:
            monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", env)
        config = LCMConfig(context_threshold=0.40) if manual else LCMConfig.from_env()
        config.database_path = str(tmp_path / f"absolute-{len(engines)}.db")
        if autoraise:
            config.codex_gpt55_autoraise_enabled = True
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engines.append(engine)
        return config, engine

    try:
        yield make
    finally:
        for engine in reversed(engines):
            engine.shutdown()


def _update_model(engine, window=100_000, *, codex=False):
    engine.update_model(
        model="gpt-5.5" if codex else "coding-model",
        context_length=window,
        provider="openai-codex" if codex else "custom",
        base_url="https://example.invalid/v1",
        api_key="test-secret",
        api_mode="responses" if codex else "chat",
    )


def _select_parser(monkeypatch, parser):
    if parser == "fallback":
        monkeypatch.setattr(config_module, "yaml", None)
    else:
        monkeypatch.setattr(config_module, "yaml", pytest.importorskip("yaml"))


@pytest.mark.parametrize("parser", ("pyyaml", "fallback"))
def test_yaml_absolute_threshold_pins_across_model_windows(make_engine, monkeypatch, parser):
    _select_parser(monkeypatch, parser)
    config, engine = make_engine("lcm:\n  context_threshold: 0.4\n  context_threshold_tokens: 20000\n")
    for window in (64_000, 128_000, 1_000_000):
        _update_model(engine, window)
        assert engine.threshold_tokens == 20_000
    assert config.config_sources["absolute_threshold_tokens"] == "config_yaml:lcm.context_threshold_tokens"


def test_env_absolute_wins_over_yaml(make_engine):
    config, engine = make_engine("lcm:\n  context_threshold_tokens: 20000\n", env="130000")
    _update_model(engine, 262_144)
    assert engine.threshold_tokens == 130_000
    assert config.config_sources["absolute_threshold_tokens"] == "env:LCM_ABSOLUTE_THRESHOLD_TOKENS"


def test_env_zero_turns_yaml_absolute_off(make_engine):
    _, engine = make_engine(
        "lcm:\n  context_threshold: 0.4\n  context_threshold_tokens: 20000\n", env="0"
    )
    _update_model(engine)
    assert engine.threshold_tokens == 40_000


def test_env_empty_falls_through_to_yaml(make_engine):
    config, engine = make_engine("lcm:\n  context_threshold_tokens: 20000\n", env="")
    _update_model(engine)
    assert engine.threshold_tokens == 20_000
    assert config.config_source_warnings == []


def test_invalid_env_falls_back_to_yaml_with_warning(make_engine):
    config, engine = make_engine("lcm:\n  context_threshold_tokens: 20000\n", env="not-a-number")
    _update_model(engine)
    assert engine.threshold_tokens == 20_000
    assert any("LCM_ABSOLUTE_THRESHOLD_TOKENS" in w for w in config.config_source_warnings)


@pytest.mark.parametrize("parser", ("pyyaml", "fallback"))
@pytest.mark.parametrize("value", ("lots", "true", "-5", "20000.0", '"lots"', '"20000.0"'))
def test_invalid_yaml_absolute_is_ignored_with_warning(make_engine, monkeypatch, parser, value):
    _select_parser(monkeypatch, parser)
    config, engine = make_engine(f"lcm:\n  context_threshold: 0.4\n  context_threshold_tokens: {value}\n")
    _update_model(engine)
    assert engine.threshold_tokens == 40_000
    assert config.config_sources["absolute_threshold_tokens"] == "default"
    assert any("lcm.context_threshold_tokens" in w for w in config.config_source_warnings)


def test_yaml_absolute_zero_means_unset(make_engine):
    config, engine = make_engine("lcm:\n  context_threshold_tokens: 0\n  context_threshold: 0.4\n")
    _update_model(engine)
    assert engine.threshold_tokens == 40_000
    assert config.config_sources["absolute_threshold_tokens"] == "default"
    assert config.config_source_warnings == []


def test_yaml_key_is_no_longer_reported_as_ignored(make_engine):
    config, engine = make_engine("lcm:\n  context_threshold_tokens: 20000\n  leaf_chunk_tokens: 12345\n")
    _update_model(engine)
    engine.on_session_start("doctor-source-session", platform="telegram", context_length=100_000)
    payload = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    config_check = next(c for c in payload["checks"] if c["check"] == "config_validation")
    assert "context_threshold_tokens" not in config.ignored_config_yaml_lcm_keys
    assert "leaf_chunk_tokens" in config.ignored_config_yaml_lcm_keys
    assert not any("lcm.context_threshold_tokens" in w for w in config_check["detail"])


def test_yaml_absolute_suppresses_codex_gpt55_autoraise(make_engine):
    _, engine = make_engine("lcm:\n  context_threshold_tokens: 130000\n", autoraise=True)
    _update_model(engine, 400_000, codex=True)
    assert engine.threshold_tokens == 130_000
    assert engine._config.codex_gpt55_autoraise_enabled is False


def test_clone_for_agent_keeps_yaml_absolute(make_engine):
    _, engine = make_engine("lcm:\n  context_threshold_tokens: 20000\n")
    _update_model(engine, 200_000)
    clone = engine.clone_for_agent()
    try:
        assert clone.threshold_tokens == 20_000
    finally:
        clone.shutdown()


def test_manual_config_does_not_read_yaml_absolute(make_engine):
    _, engine = make_engine("lcm:\n  context_threshold_tokens: 20000\n", manual=True)
    _update_model(engine)
    assert engine.threshold_tokens == 40_000


@pytest.mark.parametrize("yaml_value", (0, 20_000))
@pytest.mark.parametrize("later_env", ("0", "-5", "", "   ", None, "not-a-number", "90000"))
def test_runtime_env_change_remains_authoritative(make_engine, monkeypatch, yaml_value, later_env):
    yaml_text = "lcm:\n  context_threshold: 0.4\n"
    if yaml_value:
        yaml_text += f"  context_threshold_tokens: {yaml_value}\n"
    config, engine = make_engine(yaml_text, env="130000")
    _update_model(engine)
    assert engine.threshold_tokens == 130_000
    if later_env is None:
        monkeypatch.delenv("LCM_ABSOLUTE_THRESHOLD_TOKENS")
    else:
        monkeypatch.setenv("LCM_ABSOLUTE_THRESHOLD_TOKENS", later_env)
    _update_model(engine, 120_000)
    expected = 48_000 if later_env in ("0", "-5") else (yaml_value or 48_000)
    if later_env == "90000":
        expected = 90_000
    assert engine.threshold_tokens == expected
    assert config.absolute_threshold_tokens == yaml_value
    assert config.config_sources["absolute_threshold_tokens"] == "env:LCM_ABSOLUTE_THRESHOLD_TOKENS"
    clone = engine.clone_for_agent()
    try:
        assert clone.threshold_tokens == expected
    finally:
        clone.shutdown()


@pytest.mark.parametrize("parser", ("pyyaml", "fallback"))
@pytest.mark.parametrize("value", ("", "null", "~", '{}', '""'))
def test_empty_yaml_absolute_is_absent_without_warning(make_engine, monkeypatch, parser, value):
    _select_parser(monkeypatch, parser)
    config, engine = make_engine(f"lcm:\n  context_threshold: 0.4\n  context_threshold_tokens: {value}\n")
    _update_model(engine)
    assert engine.threshold_tokens == 40_000
    assert config.absolute_threshold_tokens == 0
    assert config.config_sources["absolute_threshold_tokens"] == "default"
    assert config.config_source_warnings == []
