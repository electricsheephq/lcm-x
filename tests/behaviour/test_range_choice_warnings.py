"""Pin values independently of the range/choice/drop diagnostics added later."""

import math

import pytest

from hermes_lcm import config


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    for key in list(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


# key suffix, field, raw, value used, warning before this fix, warning after.
ROWS = [
    ("CONTEXT_THRESHOLD", "context_threshold", raw, value, malformed, warns)
    for raw, value, malformed, warns in [
        (None, .75, False, False), (".5", .5, False, False),
        ("1", 1., False, False), ("0", 0., False, True),
        ("-1", -1., False, True), ("1.01", 1.01, False, True),
        ("nan", float("nan"), False, True), ("inf", float("inf"), False, True),
        ("bad", .75, True, True), ("", .75, True, True),
    ]
] + [
    ("SURVIVAL_RESERVE", "survival_reserve", raw, value, malformed, warns)
    for raw, value, malformed, warns in [
        (None, .15, False, False), (".15", .15, False, False),
        ("0", 0., False, False), (".9", .9, False, False),
        ("-.1", -.1, False, True), (".91", .91, False, True),
        ("nan", float("nan"), False, True), ("inf", float("inf"), False, True),
        ("-inf", float("-inf"), False, True), ("bad", .15, True, True),
        ("", .15, False, False),
    ]
] + [
    (key, field, raw, value, malformed, warns)
    for key, field, default in [
        ("LEAF_CHUNK_TOKENS", "leaf_chunk_tokens", 24000),
        ("SUMMARY_TIMEOUT_MS", "summary_timeout_ms", 120000),
    ]
    for raw, value, malformed, warns in [
        (None, default, False, False), (str(default), default, False, False),
        ("1", 1, False, False), ("0", 0, False, True),
        ("-1", -1, False, True), ("bad", default, True, True),
        ("", default, True, True),
    ]
] + [
    (key, field, raw, default if raw is None else raw, False, warns)
    for key, field, default, values in [
        ("EMBEDDING_PROVIDER", "embedding_provider", "", [
            (None, False), ("", False), (" \t", False), (" VoYaGeAI ", False),
            ("voyage", False), ("ollama", False), ("fastembed", False),
            ("fast-embed", False), ("openai-compatible", False),
            ("openai", False), ("siliconflow", False), ("voyge", True),
        ]),
        ("PREANSWER_EVIDENCE_MODE", "preanswer_evidence_mode", "", [
            (None, False), ("", False), (" \t", False), ("off", False),
            (" LEGACY_SELECTIVE ", False), ("requirements_v1", False), ("bogus", True),
        ]),
        ("EMBEDDING_STORAGE_DTYPE", "embedding_storage_dtype", "float32", [
            (None, False), ("", False), (" \t", False), ("float32", False),
            (" INT8 ", False), ("float8", True),
        ]),
    ]
    for raw, warns in values
] + [
    ("EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS", "empty_lifecycle_gc_max_age_hours", raw,
     value, False, warns)
    for raw, value, warns in [
        (None, 24., False), ("24", 24., False), ("0", 0., False),
        ("-1", -1., False), ("inf", float("inf"), False),
        ("bad", 24., True), ("", 24., False), (" \t", 24., False),
    ]
] + [
    ("RECALL_ARM_WEIGHTS", "recall_arm_weights", raw, value, False, warns)
    for raw, value, warns in [
        (None, {"fts": .5, "summary": 1., "chunk": 1.}, False),
        (" FTS =0,summary=2", {"fts": 0., "summary": 2., "chunk": 1.}, False),
        ("fts=1,fts=2", {"fts": 2., "summary": 1., "chunk": 1.}, False),
        ("unknown=1", {"fts": .5, "summary": 1., "chunk": 1.}, True),
        ("fts=nan", {"fts": .5, "summary": 1., "chunk": 1.}, True),
        ("fts=inf", {"fts": .5, "summary": 1., "chunk": 1.}, True),
        ("fts=-1", {"fts": .5, "summary": 1., "chunk": 1.}, True),
        ("broken,fts=bad,chunk=.2", {"fts": .5, "summary": 1., "chunk": .2}, True),
        ("", {"fts": .5, "summary": 1., "chunk": 1.}, False),
        (" \t", {"fts": .5, "summary": 1., "chunk": 1.}, False),
        (",", {"fts": .5, "summary": 1., "chunk": 1.}, False),
        ("fts=0.5,", {"fts": .5, "summary": 1., "chunk": 1.}, False),
        ("fts=0.5,,summary=1.0", {"fts": .5, "summary": 1., "chunk": 1.}, False),
        ("fts", {"fts": .5, "summary": 1., "chunk": 1.}, True),
        ("bogus=1", {"fts": .5, "summary": 1., "chunk": 1.}, True),
    ]
] + [
    ("MODEL_THRESHOLDS", "model_thresholds", raw, value, False, warns)
    for raw, value, warns in [
        (None, {}, False), ("model:.5", {"model": .5}, False),
        (" model :1", {"model": 1.}, False), ("ns:model:.2", {"ns:model": .2}, False),
        ("model:0", {}, True), ("model:1.1", {}, True), ("model:nan", {}, True),
        ("model:inf", {}, True), ("broken,:.5,model:bad,ok:.3", {"ok": .3}, True),
        ("", {}, False), (" \t", {}, False), (",", {}, False),
        ("glm-5.2:0.70,", {"glm-5.2": .7}, False),
        ("glm-5.2:0.70,,other:.5", {"glm-5.2": .7, "other": .5}, False),
        ("glm-5.2:abc", {}, True),
    ]
]


@pytest.mark.parametrize("key,field,raw,value,before,after", ROWS)
def test_values_unchanged(monkeypatch, key, field, raw, value, before, after):
    if raw is not None:
        monkeypatch.setenv("LCM_" + key, raw)
    actual = getattr(config.LCMConfig.from_env(), field)
    if isinstance(value, float) and math.isnan(value):
        assert math.isnan(actual)
    else:
        assert actual == value


@pytest.mark.parametrize("key,field,raw,value,before,after", ROWS)
def test_warning_presence(monkeypatch, key, field, raw, value, before, after):
    if raw is not None:
        monkeypatch.setenv("LCM_" + key, raw)
    warnings = config.LCMConfig.from_env().config_source_warnings
    assert bool(warnings) is after
    if warnings:
        assert len(warnings) == 1
        assert "LCM_" + key in warnings[0]


def test_parsers_keep_dictionary_contract(caplog):
    assert config._parse_arm_weights("unknown=1,fts=nan,chunk=.2,fts=-1", {"fts": .5, "chunk": 1.}) == {
        "fts": .5, "chunk": .2,
    }
    assert [record.message for record in caplog.records] == [
        "LCM_RECALL_ARM_WEIGHTS: negative weight -1.0 for arm 'fts' is invalid; using default 0.5"
    ]
    assert config._parse_model_thresholds_env("broken,:.2,bad:nan,ok:.5") == {"ok": .5}


@pytest.mark.parametrize("key,raw,effect", [
    ("CONTEXT_THRESHOLD", "1.1", "using 1.1 unchanged"),
    ("SURVIVAL_RESERVE", "-1", "clamped to 0.0"),
    ("SURVIVAL_RESERVE", "1", "clamped to 0.9"),
    ("SURVIVAL_RESERVE", "nan", "clamped to 0.0"),
    ("LEAF_CHUNK_TOKENS", "0", "treated as 1 for leaf sizing"),
    ("SUMMARY_TIMEOUT_MS", "0", "summary calls receive a timeout of <=0 s"),
    ("SUMMARY_TIMEOUT_MS", "-5000", "only pre-compaction extraction is floored at 0.001 s"),
    ("EMBEDDING_PROVIDER", "voyge", "ProviderUnavailable"),
    ("PREANSWER_EVIDENCE_MODE", "bogus", "ignored; using 'off'"),
    ("EMBEDDING_STORAGE_DTYPE", "float8", "ignored; using 'float32'"),
])
def test_warning_names_raw_and_consumer_effect(monkeypatch, key, raw, effect):
    monkeypatch.setenv("LCM_" + key, raw)
    warning, = config.LCMConfig.from_env().config_source_warnings
    assert f"LCM_{key}={raw!r}" in warning
    assert effect in warning


@pytest.mark.parametrize("key,accepted", [
    ("EMBEDDING_PROVIDER", ["voyage", "voyageai", "ollama", "fastembed", "fast-embed",
                            "openai-compatible", "openai", "siliconflow"]),
    ("PREANSWER_EVIDENCE_MODE", ["off", "legacy_selective", "requirements_v1"]),
    ("EMBEDDING_STORAGE_DTYPE", ["float32", "int8"]),
])
def test_choice_warning_lists_accepted_set_and_bounds_raw(monkeypatch, key, accepted):
    raw = "x" * 80
    monkeypatch.setenv("LCM_" + key, raw)
    warning, = config.LCMConfig.from_env().config_source_warnings
    assert f"LCM_{key}={repr(raw)[:40]}" in warning
    assert "x" * 41 not in warning
    for token in accepted:
        assert repr(token) in warning


@pytest.mark.parametrize("key,raw,dropped", [
    ("RECALL_ARM_WEIGHTS", "bad,unknown=1,fts=oops,summary=nan,chunk=-1",
     ["bad", "unknown=1", "fts=oops", "summary=nan", "chunk=-1"]),
    ("MODEL_THRESHOLDS", "bad,:.5,model:bad,zero:0,high:1.1,nan:nan,ok:.5",
     ["bad", ":.5", "model:bad", "zero:0", "high:1.1", "nan:nan"]),
    ("RECALL_ARM_WEIGHTS", ",fts,,bogus=1,", ["fts", "bogus=1"]),
    ("MODEL_THRESHOLDS", ",glm-5.2:abc,,", ["glm-5.2:abc"]),
])
def test_dropped_items_are_listed_once_per_key(monkeypatch, key, raw, dropped):
    monkeypatch.setenv("LCM_" + key, raw)
    warning, = config.LCMConfig.from_env().config_source_warnings
    assert "LCM_" + key in warning
    assert "using" in warning
    assert "''" not in warning
    for item in dropped:
        assert repr(item) in warning


def test_malformed_source_reader_does_not_range_warn_on_host_fallback(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text("lcm:\n  context_threshold: 7.5\n")
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "bad")
    c = config.LCMConfig.from_env()
    assert c.context_threshold == 7.5
    assert c.config_sources["context_threshold"] == "config_yaml:lcm.context_threshold"
    assert c.config_source_warnings == ["invalid env LCM_CONTEXT_THRESHOLD='bad' ignored"]


@pytest.mark.parametrize("key,raw", [
    ("CONTEXT_THRESHOLD", " " * 60 + "1.1"),
    ("SURVIVAL_RESERVE", " " * 60 + "-.1"),
])
def test_range_raw_repr_is_bounded(monkeypatch, key, raw):
    monkeypatch.setenv("LCM_" + key, raw)
    warning, = config.LCMConfig.from_env().config_source_warnings
    assert f"LCM_{key}={repr(raw)[:40]}" in warning
    assert repr(raw) not in warning


@pytest.mark.parametrize("raw", ["", " \t", ",", "fts=0.5,", "fts=0.5,,summary=1.0"])
def test_arm_empty_segments_preserve_values(raw):
    defaults = {"fts": .5, "summary": 1., "chunk": 1.}
    assert config._parse_arm_weights(raw, defaults) == defaults


@pytest.mark.parametrize("raw,expected", [
    ("", {}), (" \t", {}), (",", {}),
    ("glm-5.2:0.70,", {"glm-5.2": .7}),
    ("glm-5.2:0.70,,other:.5", {"glm-5.2": .7, "other": .5}),
])
def test_model_empty_segments_preserve_values(raw, expected):
    assert config._parse_model_thresholds_env(raw) == expected
