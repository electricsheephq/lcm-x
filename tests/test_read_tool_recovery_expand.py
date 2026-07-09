from __future__ import annotations

import json

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _marker(content: str) -> str:
    return (
        content[:32]
        + "\n\n"
        + f"[Truncated: tool response was {len(content):,} chars. "
        "Full output could not be saved to sandbox.]"
    )


def _read_call(call_id: str, path: str) -> dict:
    return {
        "role": "assistant",
        "content": "reading the file",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": path})},
            }
        ],
    }


def _engine_with_recovered_turn(tmp_path, *, enabled=True, externalize=False):
    f = tmp_path / "source.txt"
    file_content = "LINE ONE\nLINE TWO\n" * 200
    f.write_text(file_content, encoding="utf-8")
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        read_tool_recovery_enabled=enabled,
        large_output_externalization_enabled=externalize,
        large_output_externalization_threshold_chars=500,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start("chat-1", platform="cli", conversation_id="c1", context_length=200000)
    engine.ingest([
        {"role": "user", "content": "read it"},
        _read_call("call_1", str(f)),
        {"role": "tool", "tool_call_id": "call_1", "content": _marker(file_content)},
    ])
    return engine, f, file_content


def _marker_store_id(engine) -> int:
    rows = engine._store.get_session_messages("chat-1")
    tool_rows = [r for r in rows if r["role"] == "tool"]
    assert len(tool_rows) == 1
    return tool_rows[0]["store_id"]


class TestExpandRecoveredContent:
    def test_expand_surfaces_recovered_content(self, tmp_path):
        engine, _f, file_content = _engine_with_recovered_turn(tmp_path)
        store_id = _marker_store_id(engine)

        result = json.loads(lcm_tools.lcm_expand(
            {"store_id": store_id, "max_tokens": 1_000_000}, engine=engine
        ))
        assert "recovered_tool_content" in result
        rec = result["recovered_tool_content"]
        assert rec["source_type"] == "recovered_read_tool_content"
        assert rec["content"] == file_content
        assert rec["recovered_chars"] == len(file_content)
        # The base expansion is coherent on PR #364; PR #365 adds independently
        # pageable metadata while preserving the durable marker for traceability.
        assert result["content"] == file_content
        assert "[Truncated:" in result["transcript_content"]

    def test_recovered_content_is_pageable(self, tmp_path):
        engine, _f, file_content = _engine_with_recovered_turn(tmp_path)
        store_id = _marker_store_id(engine)

        first = json.loads(lcm_tools.lcm_expand(
            {"store_id": store_id, "max_tokens": 20}, engine=engine
        ))["recovered_tool_content"]
        assert first["has_more"] is True
        assert first["content_offset"] == 0

        second = json.loads(lcm_tools.lcm_expand(
            {"store_id": store_id, "max_tokens": 20,
             "recovered_content_offset": first["next_content_offset"]},
            engine=engine,
        ))["recovered_tool_content"]
        assert second["content_offset"] == first["next_content_offset"]
        # Reassembling the pages reconstructs the file prefix.
        assert (first["content"] + second["content"]) == file_content[: len(first["content"]) + len(second["content"])]

    def test_externalized_recovered_content_is_hydrated_before_paging(self, tmp_path):
        engine, _f, file_content = _engine_with_recovered_turn(tmp_path, externalize=True)
        store_id = _marker_store_id(engine)
        recovered_row = engine._store.get_recovered_tool_content_count("chat-1")
        assert recovered_row == 1

        first_result = json.loads(lcm_tools.lcm_expand(
            {"store_id": store_id, "max_tokens": 20}, engine=engine
        ))
        first = first_result["recovered_tool_content"]
        assert first_result["recovered_chars"] == len(file_content)
        assert first["recovered_chars"] == len(file_content)
        assert first["content"] == file_content[: len(first["content"])]
        assert first["has_more"] is True

        second = json.loads(lcm_tools.lcm_expand(
            {
                "store_id": store_id,
                "max_tokens": 20,
                "recovered_content_offset": first["next_content_offset"],
            },
            engine=engine,
        ))["recovered_tool_content"]
        assert second["content"] == file_content[
            first["next_content_offset"]:
            first["next_content_offset"] + len(second["content"])
        ]

    def test_no_recovery_object_when_nothing_recovered(self, tmp_path):
        engine, _f, _content = _engine_with_recovered_turn(tmp_path, enabled=False)
        store_id = _marker_store_id(engine)
        result = json.loads(lcm_tools.lcm_expand(
            {"store_id": store_id, "max_tokens": 1000}, engine=engine
        ))
        assert "recovered_tool_content" not in result
        assert "[Truncated:" in result["content"]

    def test_plain_message_has_no_recovery_object(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            read_tool_recovery_enabled=True,
        )
        engine = LCMEngine(config=config)
        engine.on_session_start("chat-1", platform="cli", conversation_id="c1", context_length=200000)
        engine.ingest([{"role": "user", "content": "just a normal message"}])
        rows = engine._store.get_session_messages("chat-1")
        store_id = rows[0]["store_id"]
        result = json.loads(lcm_tools.lcm_expand(
            {"store_id": store_id, "max_tokens": 1000}, engine=engine
        ))
        assert "recovered_tool_content" not in result
