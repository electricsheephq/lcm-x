"""Synthetic, offline origin-reference acceptance and preservation controls."""
import copy
import socket

import pytest

import hermes_lcm.engine as em
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig

ORIGIN = "  Synthetic initial request \t\nUnicode café 日本語\n"
LATEST = "Cancel the initial task. Perform the synthetic replacement instead."
HISTORICAL = "[Session-start user request (historical reference). Later user messages take precedence; do not resume cancelled or replaced work.]"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    tokens._count_tokens_cached.cache_clear()
    monkeypatch.setattr(em, "summarize_with_escalation", lambda **kw: (
        "Synthetic history without request bytes.\nExpand for details about: fixture", 1))
    def forbidden(*args, **kwargs):
        raise AssertionError("NETWORK_FORBIDDEN")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    instance = em.LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "db.sqlite"), fresh_tail_count=4,
        leaf_chunk_tokens=1, large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False, temporal_rollups_enabled=False,
        embeddings_enabled=False, proactive_recall_enabled=False, background_leaf_prep_enabled=False,
    ), hermes_home=str(tmp_path / "home"))
    instance.on_session_start("S", platform="cli", context_length=200000)
    yield instance
    instance.shutdown()
    tokens._count_tokens_cached.cache_clear()


def rounds(start=0, count=8):
    rows = []
    for i in range(start, start + count):
        rows.extend([
            {"role": "assistant", "content": "Synthetic read", "tool_calls": [{
                "id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "Synthetic data " * 100},
        ])
    return rows


def messages(with_system=False):
    return ([{"role": "system", "content": "Synthetic host policy"}] if with_system else []) + [
        {"role": "user", "content": ORIGIN},
        {"role": "assistant", "content": "Synthetic interim reply"},
        {"role": "user", "content": LATEST}, *rounds(),
    ]


def assert_origin(result):
    carriers = [r for r in result if r.get("role") in {"user", "assistant"}]
    count = sum(str(r.get("content", "")).encode().count(ORIGIN.encode()) for r in carriers)
    assert count == 1, "whole origin byte count must be one"
    assert any(HISTORICAL + "\n" + ORIGIN in str(r.get("content", "")) for r in carriers), "historical carrier missing"
    assert any(LATEST in str(r.get("content", "")) for r in result), "latest instruction missing"


@pytest.mark.parametrize("with_system", [False, True])
def test_origin_survives_first_compaction(engine, with_system):
    source = messages(with_system)
    original = copy.deepcopy(source)
    result = engine.compress(source, force=True)
    assert_origin(result)
    stored = engine._durable_real_user_messages()
    assert [r["content"] == s for r, s in zip(stored, (ORIGIN, LATEST))] == [True, True]
    assert source == original


def test_origin_survives_reassembly(engine):
    first = engine.compress(messages(), force=True)
    second = engine.compress([*first, *rounds(8)], force=True)
    assert_origin(second)
    users = engine._durable_real_user_messages(stop_after=100)
    assert len(users) == 2, "generated context must not become a user occurrence"


def test_origin_survives_proven_rotation_resume(engine):
    first = engine.compress(messages(), force=True)
    engine.on_session_start("S2", platform="cli", context_length=200000,
                            boundary_reason="compression", old_session_id="S")
    config, home = engine._config, engine._hermes_home
    engine.shutdown()
    reopened = em.LCMEngine(config=config, hermes_home=home)
    try:
        reopened.on_session_start("S2", platform="cli", context_length=200000)
        result = reopened.compress([*first, *rounds(8)], force=True)
        assert_origin(result)
    finally:
        reopened.shutdown()
