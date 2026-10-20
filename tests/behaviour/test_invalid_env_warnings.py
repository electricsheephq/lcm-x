"""Characterize fallback values and the existing configuration warning channel."""

import json
import logging

import pytest

from hermes_lcm import config, tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    for key in list(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


VALUES = [
    ("LCM_SURVIVAL_FIT", "survival_fit", raw, value)
    for raw, value in [(None, True), ("", True), (" \t", True), ("flase", True)]
    + [(raw, True) for raw in ("1", "true", "yes", "on", " TRUE ")]
    + [(raw, False) for raw in ("0", "false", "no", "off", " OFF ")]
] + [
    ("LCM_MAX_ASSEMBLY_TOKENS", "max_assembly_tokens", raw, value)
    for raw, value in [(None, 0), ("", 0), (" \t", 0), ("abc", 0), ("24k", 0),
                       ("1.5", 0), (" 24 ", 24), ("-1", -1), ("1_000", 1000)]
] + [
    ("LCM_SURVIVAL_RESERVE", "survival_reserve", raw, value)
    for raw, value in [(None, 0.15), ("", 0.15), (" \t", 0.15), ("abc", 0.15),
                       ("24k", 0.15), (" 0.25 ", 0.25), ("1e-2", 0.01), ("7.5", 7.5)]
]


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("key,field,raw,value", VALUES)
def test_fallback_values_unchanged(monkeypatch, key, field, raw, value):
    if raw is not None:
        monkeypatch.setenv(key, raw)
    assert getattr(config.LCMConfig.from_env(), field) == value


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("extra", [
    {},
    *[{"LCM_RECALL_ARM_WEIGHTS": raw} for raw in
      ("fts=0.5,", "fts=0.5,,summary=1.0", ",", "", " \t")],
    *[{"LCM_MODEL_THRESHOLDS": raw} for raw in
      ("glm-5.2:0.70,", ",", "", " \t")],
    *[{"LCM_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS": raw} for raw in ("", " \t")],
])
def test_valid_env_warning_consumers(monkeypatch, tmp_path, caplog, extra):
    monkeypatch.setenv("LCM_SURVIVAL_FIT", "true")
    monkeypatch.setenv("LCM_MAX_ASSEMBLY_TOKENS", "100000")
    monkeypatch.setenv("LCM_SURVIVAL_RESERVE", "0.15")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("LCM_TEMPORAL_ROLLUPS_ENABLED", "false")
    for key, value in {
        "LCM_CONTEXT_THRESHOLD": "0.75", "LCM_LEAF_CHUNK_TOKENS": "24000",
        "LCM_SUMMARY_TIMEOUT_MS": "120000", "LCM_EMBEDDING_PROVIDER": "voyage",
        "LCM_PREANSWER_EVIDENCE_MODE": "off", "LCM_EMBEDDING_STORAGE_DTYPE": "float32",
        "LCM_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS": "24",
        "LCM_RECALL_ARM_WEIGHTS": "fts=.5", "LCM_MODEL_THRESHOLDS": "model:.75",
    }.items():
        monkeypatch.setenv(key, value)
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    c = config.LCMConfig.from_env()
    c.database_path = str(tmp_path / "lcm.db")
    with caplog.at_level(logging.WARNING):
        engine = LCMEngine(config=c, hermes_home=str(tmp_path))
        try:
            engine.on_session_start("warning-channel", context_length=200000)
            assert engine.get_status()["config_source_warnings"] == []
            assert json.loads(tools.lcm_status({}, engine=engine))["config_source_warnings"] == []
            assert "config_source_warnings:" not in handle_lcm_command("status", engine)
            doctor = json.loads(tools.lcm_doctor({}, engine=engine))
            check = next(row for row in doctor["checks"] if row["check"] == "config_validation")
            assert check == {"check": "config_validation", "status": "pass",
                             "detail": "all settings within normal ranges"}
        finally:
            engine.shutdown()
    assert c.config_source_warnings == []
    assert not any("invalid env" in record.message for record in caplog.records)


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("key,field,raw,value", VALUES)
def test_uniform_warning_table(monkeypatch, key, field, raw, value):
    if raw is not None:
        monkeypatch.setenv(key, raw)
    c = config.LCMConfig.from_env()
    assert getattr(c, field) == value
    invalid = raw in ("flase", "abc", "24k", "1.5")
    expected = [f"invalid env {key}={raw!r} ignored; using {value!r}"] if invalid else []
    if key == "LCM_SURVIVAL_RESERVE" and raw == "7.5":
        assert len(c.config_source_warnings) == 1
        assert "clamped to 0.9" in c.config_source_warnings[0]
    else:
        assert c.config_source_warnings == expected


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("spec", [
    spec for spec in config.ENV_FIELD_SPECS
    if spec.name not in config._SOURCE_TRACKED_ENV_FIELDS and spec.py_type in (int, float, bool)
], ids=lambda spec: spec.env_key)
def test_every_uniform_typed_field_warns(monkeypatch, spec):
    monkeypatch.setenv(spec.env_key, "invalid")
    c = config.LCMConfig.from_env()
    default = getattr(config.LCMConfig(), spec.name)
    assert getattr(c, spec.name) == default
    assert c.config_source_warnings == [f"invalid env {spec.env_key}='invalid' ignored; using {default!r}"]


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("raw", ["x" * 100, "\n\t'" * 30])
def test_warning_raw_repr_is_bounded(monkeypatch, raw):
    monkeypatch.setenv("LCM_MAX_ASSEMBLY_TOKENS", raw)
    assert config.LCMConfig.from_env().config_source_warnings == [
        f"invalid env LCM_MAX_ASSEMBLY_TOKENS={repr(raw)[:40]} ignored; using 0"
    ]


@pytest.mark.usefixtures("isolated_env")
def test_invalid_env_uses_existing_consumers_without_logging(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("LCM_SURVIVAL_FIT", "flase")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("LCM_TEMPORAL_ROLLUPS_ENABLED", "false")
    c = config.LCMConfig.from_env()
    c.database_path = str(tmp_path / "lcm.db")
    expected = ["invalid env LCM_SURVIVAL_FIT='flase' ignored; using True"]
    engine = LCMEngine(config=c, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("warning-channel", context_length=200000)
        assert engine.get_status()["config_source_warnings"] == expected
        assert json.loads(tools.lcm_status({}, engine=engine))["config_source_warnings"] == expected
        assert "config_source_warnings: " + expected[0] in handle_lcm_command("status", engine)
        doctor = json.loads(tools.lcm_doctor({}, engine=engine))
        check = next(row for row in doctor["checks"] if row["check"] == "config_validation")
        assert check == {"check": "config_validation", "status": "warn", "detail": expected}
    finally:
        engine.shutdown()
    assert not any("invalid env" in record.message for record in caplog.records)


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("key,field,raw,value", [
    ("LCM_CONTEXT_THRESHOLD", "context_threshold", "7.5", 7.5),
    ("LCM_SURVIVAL_RESERVE", "survival_reserve", "-3", -3.0),
    ("LCM_LEAF_CHUNK_TOKENS", "leaf_chunk_tokens", "-5", -5),
    ("LCM_SUMMARY_TIMEOUT_MS", "summary_timeout_ms", "0", 0),
    ("LCM_EMBEDDING_PROVIDER", "embedding_provider", "voyge", "voyge"),
    ("LCM_PREANSWER_EVIDENCE_MODE", "preanswer_evidence_mode", "bogus", "bogus"),
    ("LCM_EMBEDDING_STORAGE_DTYPE", "embedding_storage_dtype", "float8", "float8"),
    ("LCM_SUMMARY_MODEL", "summary_model", "", ""),
    ("LCM_SUMMARY_MODEL", "summary_model", " \t", " \t"),
    ("LCM_SURVIVAL_RESERVE", "survival_reserve", "inf", float("inf")),
])
def test_ranges_choices_and_strings_unchanged(monkeypatch, key, field, raw, value):
    monkeypatch.setenv(key, raw)
    c = config.LCMConfig.from_env()
    assert getattr(c, field) == value
    if key == "LCM_SUMMARY_MODEL":
        assert c.config_source_warnings == []
    else:
        assert len(c.config_source_warnings) == 1
        assert key in c.config_source_warnings[0]


@pytest.mark.usefixtures("isolated_env")
@pytest.mark.parametrize("key,raw", [
    ("LCM_SURVIVAL_RESERVE", "-.1"),
    ("LCM_EMBEDDING_STORAGE_DTYPE", "float8"),
])
def test_range_choice_warnings_reach_consumers(monkeypatch, tmp_path, caplog, key, raw):
    monkeypatch.setenv(key, raw)
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("LCM_TEMPORAL_ROLLUPS_ENABLED", "false")
    c = config.LCMConfig.from_env()
    c.database_path = str(tmp_path / "lcm.db")
    expected = list(c.config_source_warnings)
    assert len(expected) == 1
    engine = LCMEngine(config=c, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("warning-channel", context_length=200000)
        status = engine.get_status()
        assert status["config_source_warnings"] == expected
        assert status["config_source_warnings"] is not c.config_source_warnings
        assert json.loads(tools.lcm_status({}, engine=engine))["config_source_warnings"] == expected
        assert "config_source_warnings: " + expected[0] in handle_lcm_command("status", engine)
        doctor = json.loads(tools.lcm_doctor({}, engine=engine))
        check = next(row for row in doctor["checks"] if row["check"] == "config_validation")
        assert check == {"check": "config_validation", "status": "warn", "detail": expected}
    finally:
        engine.shutdown()
    assert not any(key in record.message for record in caplog.records)
