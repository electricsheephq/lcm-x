"""#1013: the product defaults are the deployed configuration.

Each feature that the new defaults turn on (large-output externalization,
active-replay stubbing, temporal rollups, the threshold full sweep) must run on
a default install with no extra setup: a fresh store, payloads under the Hermes
home that holds the LCM database, and rollup maintenance skipping an in-memory
database.
"""

import json
import logging
import sqlite3
from pathlib import Path

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import extract_externalized_ref, load_externalized_payload
from hermes_lcm.tokens import count_messages_tokens


def _default_engine(tmp_path, name="lcm.db"):
    hermes_home = tmp_path / "hermes"
    config = LCMConfig(database_path=str(hermes_home / name))
    return LCMEngine(config=config, hermes_home=str(hermes_home)), hermes_home


def _tool_pair(call_id, payload):
    return [
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": payload},
    ]


def test_defaults_match_the_deployed_configuration():
    config = LCMConfig()

    assert config.context_threshold == 0.75
    assert config.fresh_tail_count == 24
    assert config.fresh_tail_max_tokens == 24_000
    assert config.leaf_chunk_tokens == 8_000
    assert config.threshold_full_sweep_enabled is True
    assert config.large_output_externalization_enabled is True
    assert config.large_output_active_replay_stubbing_enabled is True
    assert config.temporal_rollups_enabled is True
    # Unchanged by #1013.
    assert config.embeddings_enabled is False
    assert config.survival_fit is True
    assert config.survival_reserve == 0.15
    assert config.large_output_externalization_threshold_chars == 12_000
    assert config.large_output_active_replay_stub_threshold_tokens == 10_000


def test_externalization_is_active_by_default_and_writes_under_the_hermes_home(tmp_path):
    engine, hermes_home = _default_engine(tmp_path)
    content = "tool-output:" + ("x" * 13_000)
    try:
        serialized = engine._serialize_messages(
            [{"role": "tool", "tool_call_id": "call_default", "content": content}]
        )
    finally:
        engine.shutdown()

    assert "[Externalized tool output" in serialized
    payload_files = list((hermes_home / "lcm-large-outputs").glob("*.json"))
    assert len(payload_files) == 1
    payload = json.loads(payload_files[0].read_text())
    assert payload["tool_call_id"] == "call_default"
    assert payload["content"] == content


def test_active_replay_stubbing_is_active_by_default_on_a_fresh_store(tmp_path):
    engine, hermes_home = _default_engine(tmp_path)
    engine.on_session_start("defaults-stub", conversation_id="defaults-stub-conv", context_length=200_000)
    payload = "alpha beta gamma delta epsilon " * 2_500  # well over the 10000-token default
    messages = _tool_pair("big-call", payload)
    for index in range(26):  # push the big result out of the default 24-message fresh tail
        messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"})
    try:
        result = engine._assemble_context({"role": "system", "content": "system"}, messages)
        stub = next(m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "big-call")
        ref = extract_externalized_ref(stub["content"])
        recovered = load_externalized_payload(ref, config=engine._config, hermes_home=engine._hermes_home)
    finally:
        engine.shutdown()

    assert stub["content"].startswith("[Externalized tool output:")
    assert recovered is not None
    assert recovered["content"] == payload
    assert Path(hermes_home / "lcm-large-outputs" / ref).is_file()


def test_temporal_rollups_are_active_by_default_on_a_fresh_store(tmp_path):
    engine, hermes_home = _default_engine(tmp_path)
    try:
        engine.on_session_start("defaults-rollup", conversation_id="defaults-rollup-conv", context_length=200_000)
    finally:
        engine.shutdown()

    connection = sqlite3.connect(str(hermes_home / "lcm.db"))
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        connection.close()
    assert "lcm_rollups" in tables


def test_temporal_rollup_maintenance_skips_an_in_memory_database_by_default(caplog):
    engine = LCMEngine(config=LCMConfig(database_path=":memory:"))
    try:
        with caplog.at_level(logging.WARNING):
            engine.on_session_start("defaults-memory", conversation_id="defaults-memory-conv", context_length=200_000)
    finally:
        engine.shutdown()

    assert engine._config.temporal_rollups_enabled is True
    assert "temporal rollup maintenance" in caplog.text
    assert "maintenance skipped" in caplog.text


def test_threshold_full_sweep_is_active_by_default(tmp_path, monkeypatch):
    engine, _hermes_home = _default_engine(tmp_path)
    engine._session_id = "defaults-sweep"
    engine.threshold_tokens = 1
    messages = [{"role": "system", "content": "system"}]
    for index in range(60):
        messages.append({
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"FACT-{index} " + (f"dense-{index} " * 150),
        })
    calls = []

    def fake_leaf(chunk, focus_topic=None, deadline=None):
        del focus_topic, deadline
        source_tokens = count_messages_tokens(chunk)
        calls.append(source_tokens)
        return chunk, source_tokens, "retained " + " ".join(m["content"].split()[0] for m in chunk), 1, 0

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_leaf)
    try:
        engine.compress(messages, current_tokens=count_messages_tokens(messages))
        telemetry = engine.get_status()["threshold_full_sweep"]
    finally:
        engine.shutdown()

    assert len(calls) > 1
    assert all(tokens <= 8_000 for tokens in calls)
    assert telemetry["status"] == "completed"
    assert telemetry["leaf_passes"] == len(calls)
    assert telemetry["budget_exhausted"] is False


# The inherited Hermes compression.threshold is raised to at least the LCM default; the
# explicit LCM settings are not floored.


def _hermes_home_with_config(tmp_path, monkeypatch, text):
    hermes_home = tmp_path / "hermes-config"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(text)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("LCM_CONTEXT_THRESHOLD", raising=False)
    return hermes_home


def test_host_seeded_compression_threshold_is_raised_to_the_lcm_default(tmp_path, monkeypatch):
    # A bare Hermes install seeds compression.threshold 0.50 and no lcm section.
    _hermes_home_with_config(tmp_path, monkeypatch, "compression:\n  enabled: true\n  threshold: 0.50\n")

    c = LCMConfig.from_env()

    assert c.context_threshold == 0.75
    assert c.config_sources["context_threshold"] == "config_yaml:compression.threshold(floored)"
    assert any(
        "compression.threshold=0.5 raised to the LCM default 0.75" in warning
        for warning in c.config_source_warnings
    )


def test_inherited_compression_threshold_above_the_default_is_used_as_is(tmp_path, monkeypatch):
    _hermes_home_with_config(tmp_path, monkeypatch, "compression:\n  enabled: true\n  threshold: 0.85\n")

    c = LCMConfig.from_env()

    assert c.context_threshold == 0.85
    assert c.config_sources["context_threshold"] == "config_yaml:compression.threshold"
    assert not any("raised to the LCM default" in warning for warning in c.config_source_warnings)


def test_lcm_context_threshold_in_config_yaml_is_not_floored(tmp_path, monkeypatch):
    _hermes_home_with_config(
        tmp_path, monkeypatch, "lcm:\n  context_threshold: 0.5\ncompression:\n  enabled: true\n  threshold: 0.50\n"
    )

    c = LCMConfig.from_env()

    assert c.context_threshold == 0.5
    assert c.config_sources["context_threshold"] == "config_yaml:lcm.context_threshold"


def test_lcm_context_threshold_env_is_not_floored(tmp_path, monkeypatch):
    _hermes_home_with_config(tmp_path, monkeypatch, "compression:\n  enabled: true\n  threshold: 0.50\n")
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.4")

    c = LCMConfig.from_env()

    assert c.context_threshold == 0.4
    assert c.config_sources["context_threshold"] == "env:LCM_CONTEXT_THRESHOLD"
    assert not any("raised to the LCM default" in warning for warning in c.config_source_warnings)


def test_disabled_host_compression_falls_back_to_the_default(tmp_path, monkeypatch):
    _hermes_home_with_config(tmp_path, monkeypatch, "compression:\n  enabled: false\n  threshold: 0.5\n")

    c = LCMConfig.from_env()

    assert c.context_threshold == 0.75
    assert c.config_sources["context_threshold"] == "default"


def test_floored_inherited_threshold_still_allows_the_codex_gpt55_autoraise(tmp_path, monkeypatch):
    hermes_home = _hermes_home_with_config(
        tmp_path, monkeypatch, "compression:\n  enabled: true\n  threshold: 0.50\n"
    )
    config = LCMConfig.from_env()
    config.database_path = str(tmp_path / "autoraise.db")
    engine = LCMEngine(config=config, hermes_home=str(hermes_home))
    try:
        threshold, source, notice = engine._runtime_context_threshold(model="gpt-5.5", provider="openai-codex")
    finally:
        engine.shutdown()

    assert (threshold, source) == (0.85, "codex_gpt55_autoraise")
    assert notice == {"from": 0.75, "to": 0.85}
