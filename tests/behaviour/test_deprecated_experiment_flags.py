"""Characterize the loading sources of the two retiring experiments."""

import pytest

from hermes_lcm import config, presets


FLAGS = (
    ("cache_friendly_condensation_enabled", "LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED"),
    ("large_output_transcript_gc_enabled", "LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED"),
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    for key in list(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


@pytest.mark.parametrize("field,key", FLAGS)
@pytest.mark.parametrize("raw,expected", [(None, False), ("false", False), ("true", True)])
def test_env_flag_values(monkeypatch, field, key, raw, expected):
    if raw is not None:
        monkeypatch.setenv(key, raw)
    assert getattr(config.LCMConfig.from_env(), field) is expected


@pytest.mark.parametrize("field,key", FLAGS)
def test_direct_construction_does_not_warn(field, key):
    c = config.LCMConfig(**{field: True})
    assert getattr(c, field) is True
    assert c.config_source_warnings == []


@pytest.mark.parametrize("field,key", FLAGS)
def test_yaml_flag_is_ignored(tmp_path, field, key):
    (tmp_path / "config.yaml").write_text(f"lcm:\n  {field}: true\n")
    c = config.LCMConfig.from_env()
    assert getattr(c, field) is False
    assert c.ignored_config_yaml_lcm_keys == [field]
    assert c.config_source_warnings == []


def test_presets_do_not_enable_retiring_flags():
    for preset in presets.shipped_presets():
        for field, key in FLAGS:
            assert field not in preset.runtime_env
            assert field not in presets._PRESET_FIELDS


def _warning(key):
    return (
        f"deprecated: {key} is set; this experiment is removed in v0.29.0 (#1246). "
        "Until then it behaves as before."
    )


@pytest.mark.parametrize("field,key", FLAGS)
@pytest.mark.parametrize("raw,enabled", [
    (None, False), ("", False), ("false", False), ("0", False), ("no", False),
    ("off", False), ("true", True), ("1", True), ("yes", True), ("on", True),
    (" TRUE ", True),
])
def test_deprecation_warning_table(monkeypatch, field, key, raw, enabled):
    if raw is not None:
        monkeypatch.setenv(key, raw)
    c = config.LCMConfig.from_env()
    assert getattr(c, field) is enabled
    assert c.config_source_warnings == ([_warning(key)] if enabled else [])


def test_both_flags_warn_once_per_loaded_config(monkeypatch):
    for field, key in FLAGS:
        monkeypatch.setenv(key, "true")
    for _ in range(2):
        c = config.LCMConfig.from_env()
        assert c.config_source_warnings == [_warning(key) for field, key in FLAGS]


@pytest.mark.parametrize("field,key", FLAGS)
@pytest.mark.parametrize("raw", [None, "false", "true"])
def test_deprecation_warning_consumers(monkeypatch, tmp_path, caplog, field, key, raw):
    import json

    from hermes_lcm import tools
    from hermes_lcm.command import handle_lcm_command
    from hermes_lcm.engine import LCMEngine

    if raw is not None:
        monkeypatch.setenv(key, raw)
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("LCM_TEMPORAL_ROLLUPS_ENABLED", "false")
    c = config.LCMConfig.from_env()
    c.database_path = str(tmp_path / "lcm.db")
    expected = [_warning(key)] if raw == "true" else []
    engine = LCMEngine(config=c, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("experiment-warning", context_length=200000)
        assert engine.get_status()["config_source_warnings"] == expected
        assert json.loads(tools.lcm_status({}, engine=engine))["config_source_warnings"] == expected
        text = handle_lcm_command("status", engine)
        if expected:
            assert "config_source_warnings: " + expected[0] in text
        else:
            assert "config_source_warnings:" not in text
        doctor = json.loads(tools.lcm_doctor({}, engine=engine))
        check = next(row for row in doctor["checks"] if row["check"] == "config_validation")
        assert check == {
            "check": "config_validation", "status": "warn" if expected else "pass",
            "detail": expected if expected else "all settings within normal ranges",
        }
    finally:
        engine.shutdown()
    assert not any("deprecated:" in record.message for record in caplog.records)
