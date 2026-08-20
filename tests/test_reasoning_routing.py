import logging

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.model_routing import apply_lcm_reasoning_effort


@pytest.mark.parametrize(
    "field,env_name,yaml_key",
    [
        ("summary_reasoning_effort", "LCM_SUMMARY_REASONING_EFFORT", "summary_reasoning_effort"),
        ("expansion_reasoning_effort", "LCM_EXPANSION_REASONING_EFFORT", "expansion_reasoning_effort"),
    ],
)
def test_reasoning_effort_yaml_env_precedence(tmp_path, monkeypatch, field, env_name, yaml_key):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"lcm:\n  {yaml_key}: low\n")
    assert getattr(LCMConfig.from_env(), field) == "low"
    monkeypatch.setenv(env_name, "high")
    config = LCMConfig.from_env()
    assert getattr(config, field) == "high"
    assert config.config_sources[field] == "env"


@pytest.mark.parametrize(
    "field,env_name",
    [
        ("summary_reasoning_effort", "LCM_SUMMARY_REASONING_EFFORT"),
        ("expansion_reasoning_effort", "LCM_EXPANSION_REASONING_EFFORT"),
    ],
)
def test_unset_reasoning_effort_keeps_the_default(tmp_path, monkeypatch, field, env_name):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(env_name, raising=False)

    config = LCMConfig.from_env()

    assert getattr(config, field) == ""
    assert config.config_sources[field] == "default"
    assert config.config_source_warnings == []


@pytest.mark.parametrize(
    "field,env_name,yaml_key",
    [
        ("summary_reasoning_effort", "LCM_SUMMARY_REASONING_EFFORT", "summary_reasoning_effort"),
        ("expansion_reasoning_effort", "LCM_EXPANSION_REASONING_EFFORT", "expansion_reasoning_effort"),
    ],
)
def test_invalid_reasoning_effort_warns_and_falls_back_to_the_default(
    tmp_path, monkeypatch, caplog, field, env_name, yaml_key
):
    """Unsupported values are ignored with a warning, never a failed config load."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv(env_name, "turbo")

    with caplog.at_level(logging.WARNING, logger="hermes_lcm.config"):
        config = LCMConfig.from_env()

    assert getattr(config, field) == ""
    assert config.config_sources[field] == "default"
    assert any(yaml_key in warning and "turbo" in warning for warning in config.config_source_warnings)
    assert any("turbo" in record.getMessage() for record in caplog.records)


def test_invalid_reasoning_effort_in_config_yaml_warns_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_SUMMARY_REASONING_EFFORT", raising=False)
    (tmp_path / "config.yaml").write_text("lcm:\n  summary_reasoning_effort: turbo\n")

    config = LCMConfig.from_env()

    assert config.summary_reasoning_effort == ""
    assert config.config_sources["summary_reasoning_effort"] == "default"
    assert any(
        "summary_reasoning_effort" in warning and "turbo" in warning
        for warning in config.config_source_warnings
    )


def test_direct_invalid_reasoning_effort_is_ignored_not_raised(caplog):
    """A bad direct value must degrade to the task default, not fail a live call."""
    kwargs = {"extra_body": {"existing": True}}

    with caplog.at_level(logging.WARNING, logger="hermes_lcm.model_routing"):
        apply_lcm_reasoning_effort(kwargs, "turbo")

    assert kwargs == {"extra_body": {"existing": True}}
    assert any("turbo" in record.getMessage() for record in caplog.records)


def test_empty_reasoning_effort_preserves_request_defaults():
    kwargs = {"extra_body": {"existing": True}}
    apply_lcm_reasoning_effort(kwargs, "")
    assert kwargs == {"extra_body": {"existing": True}}


@pytest.mark.parametrize(
    "effort,expected",
    [
        ("none", {"enabled": False, "effort": "none"}),
        ("minimal", {"effort": "minimal"}),
        ("xhigh", {"effort": "xhigh"}),
    ],
)
def test_reasoning_effort_request_payload(effort, expected):
    kwargs = {"extra_body": {"existing": True}}
    apply_lcm_reasoning_effort(kwargs, effort)
    assert kwargs == {
        "extra_body": {"existing": True},
        "reasoning_config": expected,
    }
