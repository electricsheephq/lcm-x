"""Regressions for LCM-X issue #3: real user-turn provenance."""

from __future__ import annotations

import hermes_lcm.engine as lcm_engine
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def _tool_chain(*, count: int = 8, content_size: int = 160) -> list[dict]:
    messages: list[dict] = []
    for index in range(count):
        call_id = f"issue_3_call_{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"working step {index} " + "a" * content_size,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"result {index} " + "b" * content_size,
                },
            ]
        )
    return messages


def _summary(**_kwargs) -> tuple[str, int]:
    return "Earlier tool work.\nExpand for details about: tool chain", 1


def _assert_provider_sequence(messages: list[dict]) -> None:
    non_system = [message for message in messages if message.get("role") != "system"]
    assert non_system and non_system[0].get("role") == "user"
    for previous, current in zip(non_system, non_system[1:]):
        if "tool" not in {previous.get("role"), current.get("role")}:
            assert previous.get("role") != current.get("role")
    for index, message in enumerate(non_system):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        for offset, call in enumerate(message["tool_calls"], start=1):
            assert non_system[index + offset].get("role") == "tool"
            assert non_system[index + offset].get("tool_call_id") == call.get("id")


def test_normal_compaction_retains_only_real_user_and_generated_summary_is_not_user(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-normal.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start("issue-3-normal", platform="cli", context_length=200_000)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    user_prompt = {"role": "user", "content": "the only real user prompt"}

    try:
        result = engine.compress(
            [
                {"role": "system", "content": "stable system prompt"},
                user_prompt,
                *_tool_chain(),
            ]
        )
    finally:
        engine.shutdown()

    assert result[1] == user_prompt
    assert [
        message.get("content")
        for message in result
        if message.get("role") == "user"
    ] == [user_prompt["content"]]
    assert any(
        message.get("role") == "assistant"
        and "Earlier tool work" in str(message.get("content"))
        for message in result
    )
    _assert_provider_sequence(result)


def test_forced_overflow_retains_only_real_user(tmp_path) -> None:
    config = LCMConfig(
        fresh_tail_count=100,
        leaf_chunk_tokens=10_000,
        max_assembly_tokens=80,
        database_path=str(tmp_path / "issue-3-overflow.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start("issue-3-overflow", platform="cli", context_length=200_000)
    user_prompt = {"role": "user", "content": "the only overflow user prompt"}
    messages = [
        {"role": "system", "content": "overflow system prompt"},
        user_prompt,
        *_tool_chain(content_size=240),
    ]

    try:
        result = engine.compress(messages)
    finally:
        engine.shutdown()

    assert result[:2] == messages[:2]
    assert len(result) < len(messages)
    _assert_provider_sequence(result)


@pytest.mark.parametrize(
    "literal",
    [
        "CONTEXT SUMMARY\nLiteral user-authored summary words",
        (
            "[Recent Summary (d0, node 1)]\n"
            "Literal user-authored text\n"
            "[Expand for details: literal request]"
        ),
        (
            "[Current user objective preserved from compacted history]\n"
            "Literal user-authored objective"
        ),
        (
            "[Your active task list was preserved across context compression]\n"
            "- literal user-authored todo"
        ),
    ],
)
def test_user_literal_matching_scaffold_marker_remains_real_across_restart(
    tmp_path,
    monkeypatch,
    literal,
) -> None:
    database_path = str(tmp_path / "issue-3-literal.db")
    session_id = "issue-3-literal"
    conversation_id = "issue-3-literal-conversation"
    user_prompt = {"role": "user", "content": literal}
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=database_path,
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)

    before_restart = LCMEngine(config=config)
    before_restart.on_session_start(
        session_id,
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    active_context = before_restart.compress(
        [
            {"role": "system", "content": "stable system prompt"},
            user_prompt,
            *_tool_chain(),
        ]
    )
    assert active_context[1] == user_prompt
    literal_row = next(
        row
        for row in before_restart._store.get_session_messages(session_id)
        if row["content"] == literal
    )
    assert before_restart._has_real_user_scaffold_provenance(
        literal_row["store_id"]
    )
    before_restart.shutdown()

    after_restart = LCMEngine(config=config)
    after_restart.on_session_start(
        session_id,
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    try:
        result = after_restart.compress(
            [
                *active_context,
                *_tool_chain(count=2),
            ]
        )
        rows = after_restart._store.get_session_messages(session_id)
    finally:
        after_restart.shutdown()

    assert result[1] == user_prompt
    assert [row["content"] for row in rows].count(literal) == 1
    _assert_provider_sequence(result)


def test_no_system_history_retains_sole_real_user(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-no-system.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start("issue-3-no-system", platform="cli", context_length=200_000)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    first_user = {"role": "user", "content": "old gateway request"}

    try:
        result = engine.compress([first_user, *_tool_chain()])
    finally:
        engine.shutdown()

    assert result[0] == first_user
    assert [
        message.get("content")
        for message in result
        if message.get("role") == "user"
    ] == [first_user["content"]]


def test_no_system_history_compacts_stale_first_user_after_later_real_user(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-no-system-later.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-no-system-later",
        platform="cli",
        context_length=200_000,
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    first_user = {"role": "user", "content": "old gateway request"}
    later_user = {"role": "user", "content": "new gateway request"}

    try:
        result = engine.compress(
            [
                first_user,
                {"role": "assistant", "content": "old answer " + "x" * 200},
                later_user,
                {"role": "assistant", "content": "new answer"},
            ]
        )
    finally:
        engine.shutdown()

    assert first_user not in result
    assert later_user in result


def test_later_real_user_disqualifies_initial_anchor(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-later-user.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start("issue-3-later-user", platform="cli", context_length=200_000)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    stale_user = {"role": "user", "content": "stale initial request"}
    later_user = {"role": "user", "content": "new current request"}
    messages = [
        {"role": "system", "content": "system"},
        stale_user,
        {"role": "assistant", "content": "old response " + "x" * 200},
        later_user,
        {
            "role": "assistant",
            "content": "new response",
            "tool_calls": [{"id": "issue_3_later", "type": "function"}],
        },
        {
            "role": "tool",
            "tool_call_id": "issue_3_later",
            "content": "new result",
        },
    ]

    try:
        result = engine.compress(messages)
    finally:
        engine.shutdown()

    assert stale_user not in result
    assert later_user in result


def test_unproven_durable_scaffold_is_not_promoted_to_real_user(tmp_path) -> None:
    config = LCMConfig(database_path=str(tmp_path / "issue-3-old-scaffold.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-old-scaffold",
        platform="cli",
        context_length=200_000,
    )
    pseudo_user = {
        "role": "user",
        "content": (
            "[Recent Summary (d0, node 9)]\n"
            "Generated legacy summary\n"
            "[Expand for details: legacy]"
        ),
    }

    try:
        engine._store.append_batch(engine._session_id, [pseudo_user])
        assert engine._leading_anchor_count(
            [
                {"role": "system", "content": "system"},
                pseudo_user,
                {"role": "assistant", "content": "derived response"},
            ]
        ) == 1
    finally:
        engine.shutdown()


def test_scaffold_provenance_failure_rolls_back_message_batch(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(database_path=str(tmp_path / "issue-3-atomic.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-atomic",
        platform="cli",
        context_length=200_000,
    )
    literal = {
        "role": "user",
        "content": (
            "[Recent Summary (d0, node 1)]\n"
            "Literal user-authored text\n"
            "[Expand for details: literal request]"
        ),
    }

    def fail_metadata(_message, _store_id):
        raise OSError("simulated provenance failure")

    monkeypatch.setattr(
        engine,
        "_real_user_scaffold_metadata_rows",
        fail_metadata,
    )
    try:
        with pytest.raises(OSError, match="simulated provenance failure"):
            engine._ingest_messages([literal])
        assert engine._store.get_session_count(engine._session_id) == 0
    finally:
        engine.shutdown()


def test_structured_assistant_tail_keeps_tool_pair_and_is_not_pure_scaffold(
    tmp_path,
) -> None:
    config = LCMConfig(database_path=str(tmp_path / "issue-3-structured.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start("issue-3-structured", platform="cli", context_length=200_000)
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="prior compacted work",
            token_count=10,
            source_token_count=100,
            source_ids=[],
            source_type="messages",
            created_at=1.0,
            expand_hint="prior work",
        )
    )
    user_prompt = {"role": "user", "content": "the only real user prompt"}
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": "continuing tool work"}],
        "tool_calls": [
            {
                "id": "issue_3_structured",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
    }
    tool = {
        "role": "tool",
        "tool_call_id": "issue_3_structured",
        "content": "structured result",
    }

    try:
        result = engine._assemble_context(
            {"role": "system", "content": "system"},
            [user_prompt, assistant, tool],
            include_lcm_note=False,
            preserve_leading_user=True,
        )
    finally:
        engine.shutdown()

    assert [message["role"] for message in result] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert result[2]["tool_calls"] == assistant["tool_calls"]
    assert result[2]["content"][-1] == assistant["content"][0]
    assert not engine._is_replayed_context_scaffold_message(result[2])
    _assert_provider_sequence(result)


def test_systemless_summary_merge_preserves_assistant_tool_lineage(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-systemless-lineage.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-systemless-lineage",
        platform="cli",
        context_length=200_000,
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    first_user = {"role": "user", "content": "the only gateway request"}
    tool_chain = _tool_chain(count=1)

    try:
        active_context = engine.compress([first_user, *tool_chain])
        stored_rows = engine._store.get_session_messages(engine._session_id)
        assistant_store_id = next(
            row["store_id"]
            for row in stored_rows
            if row["role"] == "assistant"
            and row.get("tool_calls") == tool_chain[0]["tool_calls"]
        )
        tool_store_id = next(
            row["store_id"]
            for row in stored_rows
            if row["role"] == "tool"
            and row.get("tool_call_id") == tool_chain[1]["tool_call_id"]
        )

        engine.compress(
            [
                *active_context,
                {"role": "user", "content": "a later gateway request"},
                {"role": "assistant", "content": "a later answer"},
            ]
        )
        nodes = engine._dag.get_session_nodes(engine._session_id)
    finally:
        engine.shutdown()

    assert any(
        assistant_store_id in node.source_ids
        and tool_store_id in node.source_ids
        for node in nodes
    )


def test_summary_prefixed_by_proactive_memory_remains_scaffolding(
    tmp_path,
) -> None:
    config = LCMConfig(database_path=str(tmp_path / "issue-3-memory-summary.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-memory-summary",
        platform="cli",
        context_length=200_000,
    )
    message = {
        "role": "user",
        "content": (
            "<relevant-memories>\n"
            "prior recalled context\n"
            "</relevant-memories>\n\n"
            "---\n\n"
            "[Recent Summary (d0, node 1)]\n"
            "Earlier work\n"
            "[Expand for details: earlier work]"
        ),
    }

    try:
        assert engine._is_replayed_context_scaffold_message(message)
    finally:
        engine.shutdown()


def test_orphan_tool_is_dropped_before_summary_role_selection(tmp_path) -> None:
    config = LCMConfig(database_path=str(tmp_path / "issue-3-orphan.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start("issue-3-orphan", platform="cli", context_length=200_000)
    engine._dag.add_node(
        SummaryNode(
            session_id=engine._session_id,
            depth=0,
            summary="prior compacted work",
            token_count=10,
            source_token_count=100,
            source_ids=[],
            source_type="messages",
            created_at=1.0,
            expand_hint="prior work",
        )
    )
    user_prompt = {"role": "user", "content": "the only real user prompt"}
    orphan = {
        "role": "tool",
        "tool_call_id": "missing",
        "content": "orphan result",
    }
    assistant = {"role": "assistant", "content": "continue after orphan"}

    try:
        result = engine._assemble_context(
            {"role": "system", "content": "system"},
            [user_prompt, orphan, assistant],
            include_lcm_note=False,
            preserve_leading_user=True,
        )
    finally:
        engine.shutdown()

    assert orphan not in result
    assert [message["role"] for message in result] == [
        "system",
        "user",
        "assistant",
    ]
    assert "prior compacted work" in str(result[-1]["content"])
    assert "continue after orphan" in str(result[-1]["content"])
    _assert_provider_sequence(result)
