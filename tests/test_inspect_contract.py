"""Pin inspect response bytes, slash-command output and shared patch reach."""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_lcm import command, inspect_tools, tools


@pytest.fixture
def inventory(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Fixed metadata with real SQL reads and no scheduler, clock or model."""
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        CREATE TABLE messages (store_id, session_id, token_estimate);
        INSERT INTO messages VALUES (7, 'session', 12), (8, 'session', 12);
        CREATE TABLE summary_nodes (
            node_id, session_id, depth, token_count, source_token_count,
            source_type, created_at, earliest_at, latest_at, expand_hint, source_ids
        );
        INSERT INTO summary_nodes VALUES
            (3, 'session', 0, 4, 12, 'messages', 100, 90, 95, 'expand', '[5, 6]');
        CREATE TABLE lcm_rollups (
            rollup_id INTEGER PRIMARY KEY, period_kind, period_start, scope, status,
            error, UNIQUE(period_kind, period_start, scope)
        );
        INSERT INTO lcm_rollups VALUES (1, 'day', '2026-07-15', 'session', 'ready', 'café');
        CREATE TABLE lcm_rollup_state (period_kind, scope, last_build_cursor, last_built_at);
        INSERT INTO lcm_rollup_state VALUES ('day', 'session', 7, 100);
    """)
    row: dict[str, Any] = dict(
        store_id=7, session_id="session", source="cli", conversation_id="conversation",
        role="assistant", timestamp=100, token_estimate=12,
        content="hidden [Externalized tool output: tool_call_id=call-1; chars=17; bytes=17; ref=payload.json]",
        tool_calls=None,
    )
    rows = [row, dict(row, store_id=8)]

    def page(session_id: str, *, limit: int) -> list[dict[str, Any]]:
        assert session_id == "session"
        return rows[:limit]

    def metadata(engine: Any, ref: str, session_id: str) -> dict[str, Any]:
        assert ref == "payload.json"
        return dict(readable=True, payload_session_id=session_id, marker="call-time-metadata")

    monkeypatch.setattr(tools, "_inspect_externalized_payload_metadata", metadata)
    monkeypatch.setattr(tools, "_LCM_INSPECT_REF_SCAN_MESSAGE_LIMIT", 1)
    engine = SimpleNamespace(
        current_session_id="session", current_conversation_id="conversation",
        current_session_platform="cli", side_channel_active=False,
        current_session_ignored=False, current_session_stateless=False,
        compression_count=1, last_prompt_tokens=12, threshold_tokens=800,
        _session_id="session", _conversation_id="conversation", _last_compacted_store_id=6,
        _config=SimpleNamespace(
            fresh_tail_count=1, fresh_tail_max_tokens=200, temporal_rollups_enabled=True,
            ignore_session_patterns=["never*"], stateless_session_patterns=["session"],
            ignore_message_patterns=[],
        ),
        _store=SimpleNamespace(connection=connection, get_session_count=lambda _: 2,
                               load_session_page=page),
        _dag=SimpleNamespace(connection=connection, get_session_depth_stats=lambda _: {
            0: dict(count=1, tokens=4, source_tokens=12),
        }),
        _lifecycle=SimpleNamespace(get_by_conversation=lambda _: None, get_by_session=lambda _: None),
        _get_session_fresh_tail=lambda _: ([row], SimpleNamespace(
            tokens=12, token_limited=False, tool_group_extended=False,
        )),
        get_status=lambda: dict(runtime_identity={"kind": "fixture"}, ingest_protection={}),
    )
    yield engine
    connection.close()


def response_bytes(engine: Any) -> dict[str, str]:
    result = {
        "missing": tools.lcm_inspect({}),
        "session": tools.lcm_inspect({}, engine=engine),
        "clamped": tools.lcm_inspect({"limit": 999}, engine=engine),
        "rollups": command._rollups_status_text(engine),
    }
    engine.current_session_id = ""
    result["unbound"] = tools.lcm_inspect({}, engine=engine)
    return result


def test_inspect_response_bytes(inventory: Any) -> None:
    expected = json.loads((Path(__file__).parent / "fixtures" / "inspect_contract.json").read_text())
    assert response_bytes(inventory) == expected


def test_inspect_shared_patch_points_reach_callers(inventory: Any) -> None:
    result = json.loads(tools.lcm_inspect({}, engine=inventory))
    refs = result["externalized_refs"]
    assert refs["scanned_messages"] == 1
    assert refs["scan_truncated"] is True
    assert refs["items"][0]["marker"] == "call-time-metadata"
    assert tools.lcm_inspect is inspect_tools.lcm_inspect


def test_inspect_rollup_failure_preserves_logger(
    inventory: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    inventory._dag.connection.execute("DROP TABLE lcm_rollups")
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm.tools"):
        result = json.loads(tools.lcm_inspect({}, engine=inventory))
    assert result["temporal_rollups"]["query_error"] == "OperationalError: no such table: lcm_rollups"
    assert [(r.name, r.levelname, r.message) for r in caplog.records] == [
        ("hermes_lcm.tools", "DEBUG", "LCM temporal rollup status query failed"),
    ]
    assert caplog.records[0].exc_info is not None
