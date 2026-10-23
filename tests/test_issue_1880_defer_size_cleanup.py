"""#1880 part A: durable size cleanup waits for a real host boundary."""

import json
import logging
from unittest.mock import Mock

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from tests.test_active_tool_stubbing import assembled_tool, tool_pair


@pytest.fixture
def engines(tmp_path):
    opened = []

    def build(**overrides):
        config = LCMConfig(**{
            "database_path": str(tmp_path / f"arm-{len(opened)}.db"),
            "fresh_tail_count": 32,
            "leaf_chunk_tokens": 20_000,
            "large_output_externalization_enabled": True,
            "large_output_externalization_threshold_chars": 1_000_000,
            "large_output_active_replay_stubbing_enabled": True,
            "large_output_active_replay_stub_threshold_tokens": 5,
            **overrides,
        })
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
        engine.on_session_start("size-cleanup", conversation_id="size-cleanup", context_length=200_000)
        engine.threshold_tokens = 100_000
        opened.append(engine)
        return engine

    yield build
    for engine in opened:
        engine.shutdown()


def messages():
    return [{"role": "system", "content": "system"},
            *tool_pair("live", "synthetic live payload " * 100)]


def stored_bytes(engine):
    """Every messages-table column, including timestamps, in store order."""
    rows = engine._store._conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
    return json.dumps([list(row) for row in rows], ensure_ascii=False).encode()


def test_below_threshold_host_gate_defers_tool_cleanup(engines, monkeypatch, caplog):
    engine = engines()
    view = messages()
    compress = Mock(wraps=engine.compress)
    monkeypatch.setattr(engine, "compress", compress)
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm.compaction"):
        requested = engine.should_compress_preflight(view)
        if requested:  # The host only enters its boundary through this gate.
            engine.compress(view, current_tokens=count_messages_tokens(view))
    assert requested is False
    compress.assert_not_called()
    assert not engine._compaction_request.below_threshold_cleanup_only
    assert not engine._compaction_request.cleanup_only_due_to_boundary_cooldown
    assert not engine._compaction_request.automatic_request
    assert "deferred_size_only_cleanup" in caplog.text
    assert len(engine._store.get_session_messages(engine._session_id)) == len(view)


@pytest.mark.parametrize("prefix", [
    "[Externalized LCM ingest payload:",
    "[Externalized payload: kind=raw_payload;",
    "[Externalized tool output:",
])
def test_each_size_only_class_defers(engines, monkeypatch, prefix):
    engine = engines()
    original = [{"role": "user", "content": "original payload"}]
    replay = [{"role": "user", "content": prefix + " synthetic]"}]
    monkeypatch.setattr(engine, "_ingest_messages", lambda _: replay)
    assert engine.should_compress_preflight(original) is False


@pytest.mark.parametrize("pressure", ["original", "replay", "equal", "overflow"])
def test_real_pressure_still_requests_boundary(engines, monkeypatch, pressure):
    engine = engines(max_assembly_tokens=200 if pressure == "overflow" else 100_000)
    view = messages()
    if pressure == "original":
        engine.threshold_tokens = count_messages_tokens(view) - 1
    elif pressure == "equal":
        engine.threshold_tokens = count_messages_tokens(view)
    elif pressure == "replay":
        view = [{"role": "user", "content": "short"}]
        replay = [{"role": "user", "content": "[Externalized tool output: " + "stub " * 100}]
        engine.threshold_tokens = count_messages_tokens(replay)
        monkeypatch.setattr(engine, "_ingest_messages", lambda _: replay)
    else:
        assert engine._should_force_overflow_recovery(messages=view)
    assert engine.should_compress_preflight(view) is True


@pytest.mark.parametrize("kind", ["quarantine", "ignored", "length", "mixed"])
def test_non_size_cleanup_still_requests_boundary(engines, monkeypatch, kind):
    engine = engines()
    original = [{"role": "assistant", "content": "original"}]
    placeholder = ("[LCM active replay placeholder: assistant output quarantined; synthetic]"
                   if kind != "ignored" else "[LCM active replay placeholder: message ignored; synthetic]")
    replay = [{"role": "assistant", "content": placeholder}]
    if kind == "length":
        replay.append({"role": "user", "content": "extra"})
    elif kind == "mixed":
        original.insert(0, {"role": "user", "content": "large"})
        replay.insert(0, {"role": "user", "content": "[Externalized tool output: synthetic]"})
    monkeypatch.setattr(engine, "_ingest_messages", lambda _: replay)
    assert engine.should_compress_preflight(original) is True
    assert engine._compaction_request.below_threshold_cleanup_only


def test_deferred_stub_adopted_at_next_threshold_with_identical_stored_rows(engines, monkeypatch):
    deferred, immediate = engines(), engines()
    monkeypatch.setattr("time.time", lambda: 1_700_000_000.0)
    view = messages()
    assert deferred.should_compress_preflight(view) is False
    before = stored_bytes(deferred)
    # The old arm enters compress immediately, using its unchanged public API.
    immediate._ingest_messages(view)
    immediate_result = immediate.compress(view, current_tokens=1_000)
    assert stored_bytes(immediate) == before
    deferred.threshold_tokens = 1
    assert deferred.should_compress_preflight(view) is True
    result = deferred.compress(view, current_tokens=1_000)
    assert assembled_tool(result, "live")["content"].startswith("[Externalized tool output:")
    assert assembled_tool(result, "live")["content"] == assembled_tool(immediate_result, "live")["content"]
    assert stored_bytes(deferred) == stored_bytes(immediate) == before
    assert deferred._dag.get_session_node_count(deferred._session_id) == 0
