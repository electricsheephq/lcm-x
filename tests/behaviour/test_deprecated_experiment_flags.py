"""Pin parsed-and-ignored experiments and the remaining deprecation."""

import pytest

from hermes_lcm import config, presets


FLAGS = (
    ("cache_friendly_condensation_enabled", "LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED"),
    ("deferred_maintenance_enabled", "LCM_DEFERRED_MAINTENANCE_ENABLED"),
    ("large_output_transcript_gc_enabled", "LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED"),
)
REMOVED_FLAGS = (FLAGS[0], FLAGS[2])
DEPRECATED_FLAGS = (FLAGS[1],)
REMOVED_IN = {"LCM_DEFERRED_MAINTENANCE_ENABLED": "v0.31.0"}


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
        f"deprecated: {key} is set; this experiment is removed in {REMOVED_IN[key]} (#1246). "
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
    assert c.config_source_warnings == ([_warning(key)] if enabled and key in REMOVED_IN else [])


def test_each_flag_warns_once_per_loaded_config(monkeypatch):
    for field, key in FLAGS:
        monkeypatch.setenv(key, "true")
    for _ in range(2):
        c = config.LCMConfig.from_env()
        assert c.config_source_warnings == [_warning(key) for field, key in DEPRECATED_FLAGS]


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
    expected = [_warning(key)] if raw == "true" and key in REMOVED_IN else []
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


@pytest.mark.parametrize("field,key", REMOVED_FLAGS)
@pytest.mark.parametrize("raw", [None, "false", "true"])
def test_removed_flag_warns_once_per_process(monkeypatch, tmp_path, caplog, field, key, raw):
    from hermes_lcm import engine as engine_module

    warning_flag = {
        "cache_friendly_condensation_enabled": "_CACHE_FRIENDLY_CONDENSATION_WARNING_LOGGED",
        "large_output_transcript_gc_enabled": "_LARGE_OUTPUT_TRANSCRIPT_GC_WARNING_LOGGED",
    }[field]
    monkeypatch.setattr(engine_module, warning_flag, False)
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.setenv("LCM_TEMPORAL_ROLLUPS_ENABLED", "false")
    if raw is not None:
        monkeypatch.setenv(key, raw)
    for i in range(2):
        c = config.LCMConfig.from_env()
        c.database_path = str(tmp_path / f"warning-{i}.db")
        assert c.config_source_warnings == []
        engine = engine_module.LCMEngine(config=c, hermes_home=str(tmp_path))
        engine.shutdown()
    expected = f"{key} is no longer supported (removed in v0.30.0) and is ignored"
    if field == "large_output_transcript_gc_enabled":
        expected += "; summarized tool rows are no longer rewritten"
    assert [record.message for record in caplog.records if key in record.message] == (
        [expected] if raw == "true" else []
    )


@pytest.mark.parametrize("field,key", REMOVED_FLAGS)
def test_removed_flag_true_matches_unset_compaction(monkeypatch, tmp_path, field, key):
    from types import SimpleNamespace

    from hermes_lcm import engine as engine_module, externalize
    from hermes_lcm.dag import SummaryNode
    from tests.harness import engine_context, pin_char_counter

    pin_char_counter(monkeypatch)
    monkeypatch.setattr(externalize, "time", SimpleNamespace(
        strftime=lambda *_: "20260101_000000", gmtime=lambda: (),
        time_ns=lambda: 1, time=lambda: 1.0,
    ))
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kwargs: (
        "Leaf summary. Expand for details about: oldest raw chunk" if kwargs["depth"] == 0
        else "Condensed summary. Expand for details about: earlier leaves", 1,
    ))
    results = []
    for raw in (None, "true"):
        if raw is not None:
            monkeypatch.setenv(key, raw)
        loaded = config.LCMConfig.from_env()
        assert getattr(loaded, field) is (raw == "true")
        assert loaded.config_source_warnings == []
        with engine_context(
            tmp_path / str(raw), window=200000, session_id="experiment-parity",
            fresh_tail_count=2, leaf_chunk_tokens=100, dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=200, condensation_fanin=2, threshold_full_sweep_enabled=False,
            embeddings_enabled=False, temporal_rollups_enabled=False,
            large_output_externalization_enabled=True, large_output_externalization_threshold_chars=200,
            **{field: getattr(loaded, field)},
        ) as engine:
            engine._dag.add_node(SummaryNode(
                session_id=engine._session_id, depth=0, summary="Earlier leaf " * 30,
                token_count=100, source_token_count=200, source_ids=[1],
                source_type="messages", created_at=0,
            ))
            messages = [{"role": "system", "content": "Stable system."},
                        {"role": "assistant", "content": "Running tool", "tool_calls": [
                            {"id": "call-parity", "type": "function", "function": {
                                "name": "read", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "call-parity", "content": "abcdef" * 2000}]
            messages += [{"role": "user" if i % 2 == 0 else "assistant",
                          "content": f"Message {i}: " + "chunk " * 35} for i in range(6)]
            output = engine.compress(messages)
            nodes = engine._dag.get_session_nodes(engine._session_id)
            assert any(node.depth == 1 for node in nodes)
            rows = engine._store.get_range(engine._session_id)
            assert any(row["role"] == "tool" for row in rows)
            assert all(not row["content"].startswith("[GC'd externalized") for row in rows)
            keys = ("role", "content", "tool_calls", "tool_call_id")
            results.append((
                [{k: row[k] for k in keys if k in row} for row in output],
                [{k: row[k] for k in keys if k in row} for row in rows],
                [(n.depth, n.summary, n.source_ids, n.source_type) for n in nodes],
                engine.get_status()["condensation_suppressed_reason"],
            ))
    assert results[0] == results[1]
