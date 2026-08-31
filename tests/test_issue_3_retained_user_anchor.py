"""Retained-user anchor regressions for LCM-X issue #3."""

from __future__ import annotations

import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def _tool_chain(*, count: int = 8, content_size: int = 160) -> list[dict]:
    messages: list[dict] = []
    for index in range(count):
        call_id = f"issue_3_anchor_{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"working step {index} " + "a" * content_size,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": "{}",
                            },
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
    non_system = [
        message for message in messages if message.get("role") != "system"
    ]
    assert non_system and non_system[0].get("role") == "user"
    for previous, current in zip(non_system, non_system[1:]):
        if "tool" not in {previous.get("role"), current.get("role")}:
            assert previous.get("role") != current.get("role")
    for index, message in enumerate(non_system):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        for offset, call in enumerate(message["tool_calls"], start=1):
            assert non_system[index + offset].get("role") == "tool"
            assert non_system[index + offset].get("tool_call_id") == call.get(
                "id"
            )


def test_normal_compaction_retains_sole_real_user_with_lineage(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-normal.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-normal",
        platform="cli",
        context_length=200_000,
    )
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
        prompt_row = next(
            row
            for row in engine._store.get_session_messages(engine._session_id)
            if row["content"] == user_prompt["content"]
        )
        mapped = engine._get_store_id_map_for_messages(result[1:])
        folded_row = next(
            row
            for row in engine._store.get_session_messages(engine._session_id)
            if row.get("tool_calls") == result[2].get("tool_calls")
        )
    finally:
        engine.shutdown()

    assert result[1] == user_prompt
    assert mapped[id(result[1])] == prompt_row["store_id"]
    assert mapped[id(result[2])] == folded_row["store_id"]
    assert result[2].get("tool_calls") == folded_row.get("tool_calls")
    assert [
        message.get("content")
        for message in result
        if message.get("role") == "user"
    ] == [user_prompt["content"]]
    _assert_provider_sequence(result)


def test_forced_overflow_retains_sole_real_user_with_lineage(tmp_path) -> None:
    config = LCMConfig(
        fresh_tail_count=100,
        leaf_chunk_tokens=10_000,
        max_assembly_tokens=80,
        database_path=str(tmp_path / "issue-3-overflow.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-overflow",
        platform="cli",
        context_length=200_000,
    )
    user_prompt = {"role": "user", "content": "the only overflow user prompt"}
    messages = [
        {"role": "system", "content": "overflow system prompt"},
        user_prompt,
        *_tool_chain(content_size=240),
    ]

    try:
        result = engine.compress(messages)
        prompt_row = next(
            row
            for row in engine._store.get_session_messages(engine._session_id)
            if row["content"] == user_prompt["content"]
        )
        mapped = engine._get_store_id_map_for_messages([result[1]])
    finally:
        engine.shutdown()

    assert result[:2] == messages[:2]
    assert len(result) < len(messages)
    assert mapped[id(result[1])] == prompt_row["store_id"]
    _assert_provider_sequence(result)


def test_generated_context_order_and_query_survive_retained_user(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-recall.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-recall",
        platform="cli",
        context_length=200_000,
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    user_prompt = {"role": "user", "content": "query this retained prompt"}
    seen_query_messages: list[dict] = []

    def proactive_recall(
        messages: list[dict],
        summary_role: str,
        _active_node_ids: set,
    ) -> dict:
        seen_query_messages.extend(messages)
        return {
            "role": summary_role,
            "content": "<relevant-memories>\nvolatile recall\n</relevant-memories>",
        }

    monkeypatch.setattr(
        engine,
        "_build_proactive_recall_message",
        proactive_recall,
    )

    try:
        result = engine.compress(
            [
                {"role": "system", "content": "stable system prompt"},
                user_prompt,
                *_tool_chain(),
            ]
        )
        mapped = engine._get_store_id_map_for_messages(result[1:])
    finally:
        engine.shutdown()

    folded_content = result[2]["content"]
    assert seen_query_messages[0] == user_prompt
    assert folded_content.index("[Recent Summary") < folded_content.index(
        "<relevant-memories>"
    )
    assert folded_content.index("<relevant-memories>") < folded_content.index(
        "working step"
    )
    assert id(result[2]) in mapped
    _assert_provider_sequence(result)


def test_fold_lineage_failure_omits_generated_context_but_keeps_tail(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-fold-failure.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-fold-failure",
        platform="cli",
        context_length=200_000,
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    monkeypatch.setattr(engine, "_write_folded_tail_lineage", lambda *_args: False)
    user_prompt = {"role": "user", "content": "keep me on metadata failure"}

    try:
        result = engine.compress(
            [
                {"role": "system", "content": "stable system prompt"},
                user_prompt,
                *_tool_chain(),
            ]
        )
        mapped = engine._get_store_id_map_for_messages(result[1:])
    finally:
        engine.shutdown()

    assert result[1] == user_prompt
    assert result[2]["content"].startswith("working step")
    assert not any(
        "[Recent Summary" in str(message.get("content") or "")
        for message in result
    )
    assert id(result[1]) in mapped
    assert id(result[2]) in mapped
    _assert_provider_sequence(result)


def test_folded_duplicate_maps_to_registered_occurrence_only(tmp_path) -> None:
    config = LCMConfig(
        database_path=str(tmp_path / "issue-3-fold-duplicate.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-fold-duplicate",
        platform="cli",
        context_length=200_000,
    )
    duplicate = {"role": "assistant", "content": "duplicate assistant"}

    try:
        store_ids = engine._store.append_batch(
            engine._session_id,
            [duplicate, duplicate],
        )
        folded = engine._prepend_generated_context_to_message(
            duplicate,
            "[Recent Summary (d0, node 1)]\nsummary\n[Expand for details: duplicate]",
        )
        assert engine._write_folded_tail_lineage(folded, store_ids[1])
        engine._last_compacted_store_id = store_ids[1]
        mapped = engine._get_store_id_map_for_messages([folded])
        ambiguous = engine._get_store_id_map_for_messages(
            [folded.copy(), folded.copy()]
        )
    finally:
        engine.shutdown()

    assert mapped[id(folded)] == store_ids[1]
    assert ambiguous == {}


def test_structured_assistant_tail_keeps_content_and_lineage(tmp_path) -> None:
    config = LCMConfig(
        database_path=str(tmp_path / "issue-3-fold-structured.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-fold-structured",
        platform="cli",
        context_length=200_000,
    )
    source = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "structured assistant text"},
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
        ],
        "metadata": {"provider": "test"},
    }

    try:
        store_id = engine._store.append_batch(engine._session_id, [source])[0]
        folded = engine._prepend_generated_context_to_message(
            source,
            "[Recent Summary (d0, node 1)]\nsummary",
        )
        assert engine._write_folded_tail_lineage(folded, store_id)
        engine._last_compacted_store_id = store_id
        mapped = engine._get_store_id_map_for_messages([folded])
    finally:
        engine.shutdown()

    assert folded["content"][0] == {
        "type": "text",
        "text": "[Recent Summary (d0, node 1)]\nsummary",
    }
    assert folded["content"][1:] == source["content"]
    assert folded["metadata"] == source["metadata"]
    assert mapped[id(folded)] == store_id


def test_dictionary_assistant_tail_keeps_structured_content(tmp_path) -> None:
    config = LCMConfig(
        database_path=str(tmp_path / "issue-3-fold-dictionary.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-fold-dictionary",
        platform="cli",
        context_length=200_000,
    )
    source = {
        "role": "assistant",
        "content": {"type": "text", "text": "dictionary assistant text"},
        "metadata": {"provider": "test"},
    }

    try:
        store_id = engine._store.append_batch(engine._session_id, [source])[0]
        folded = engine._prepend_generated_context_to_message(
            source,
            "[Recent Summary (d0, node 1)]\nsummary",
        )
        assert engine._write_folded_tail_lineage(folded, store_id)
        engine._last_compacted_store_id = store_id
        mapped = engine._get_store_id_map_for_messages([folded])
    finally:
        engine.shutdown()

    assert folded["content"] == [
        {"type": "text", "text": "[Recent Summary (d0, node 1)]\nsummary"},
        source["content"],
    ]
    assert folded["metadata"] == source["metadata"]
    assert mapped[id(folded)] == store_id


def test_folded_source_is_summarized_after_leaving_fresh_tail(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-fold-backlog.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-fold-backlog",
        platform="cli",
        context_length=200_000,
    )
    captured: list[str] = []

    def capture_summary(**kwargs) -> tuple[str, int]:
        captured.append(kwargs["text"])
        return _summary(**kwargs)

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
    user_prompt = {"role": "user", "content": "sole user for folded backlog"}

    try:
        first = engine.compress(
            [
                {"role": "system", "content": "stable system prompt"},
                user_prompt,
                *_tool_chain(),
            ]
        )
        folded_source = next(
            row
            for row in engine._store.get_session_messages(engine._session_id)
            if row.get("tool_calls") == first[2].get("tool_calls")
        )
        captured.clear()
        second = engine.compress(
            [
                *first,
                {
                    "role": "assistant",
                    "content": "later tool work",
                    "tool_calls": [
                        {
                            "id": "issue_3_later",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "issue_3_later",
                    "content": "later tool result",
                },
                {"role": "assistant", "content": "later assistant tail"},
            ]
        )
        summarized_source_ids = {
            source_id
            for node in engine._dag.get_session_nodes(engine._session_id)
            for source_id in node.source_ids
        }
    finally:
        engine.shutdown()

    assert any(folded_source["content"] in text for text in captured)
    assert folded_source["store_id"] in summarized_source_ids
    _assert_provider_sequence(second)


def test_tight_reassembly_keeps_existing_fold_lineage(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-fold-tight.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-fold-tight",
        platform="cli",
        context_length=200_000,
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)

    try:
        first = engine.compress(
            [
                {"role": "system", "content": "stable system prompt"},
                {"role": "user", "content": "sole user for tight reassembly"},
                *_tool_chain(),
            ]
        )
        source_map = engine._get_store_id_map_for_messages(first[1:])
        folded_store_id = source_map[id(first[2])]
        reassembled = engine._assemble_context(
            first[0],
            first[2:],
            assembly_cap_override=count_messages_tokens(first),
            retained_user_message=first[1],
        )
        remapped = engine._get_store_id_map_for_messages(reassembled[1:])
    finally:
        engine.shutdown()

    assert reassembled[2] == first[2]
    assert remapped[id(reassembled[2])] == folded_store_id


def test_folded_tail_lineage_survives_restart(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "issue-3-fold-restart.db"
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(database_path),
    )
    first = LCMEngine(config=config)
    first.on_session_start(
        "issue-3-fold-restart",
        platform="cli",
        context_length=200_000,
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    messages = [
        {"role": "system", "content": "stable system prompt"},
        {"role": "user", "content": "restart-retained user"},
        *_tool_chain(),
    ]

    try:
        result = first.compress(messages)
        expected = first._get_store_id_map_for_messages(result[1:])
        expected_folded_store_id = expected[id(result[2])]
    finally:
        first.shutdown()

    second = LCMEngine(config=config)
    try:
        second.on_session_start(
            "issue-3-fold-restart",
            platform="cli",
            context_length=200_000,
        )
        rows_before_replay = second._store.get_session_count(second._session_id)
        should_compress = second.should_compress_preflight(result)
        rows_after_replay = second._store.get_session_count(second._session_id)
        reconciliation = second._last_ingest_reconciliation
        restarted = second._get_store_id_map_for_messages(result[1:])
    finally:
        second.shutdown()

    assert should_compress is False
    assert rows_after_replay == rows_before_replay, reconciliation
    assert restarted[id(result[2])] == expected_folded_store_id
    assert id(result[1]) in restarted


def test_preflight_second_user_invalidates_prepared_anchor(
    tmp_path,
) -> None:
    config = LCMConfig(
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        context_threshold=0.01,
        database_path=str(tmp_path / "issue-3-preflight-second-user.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-preflight-second-user",
        platform="cli",
        context_length=200_000,
    )
    first = [
        {"role": "system", "content": "stable system prompt"},
        {"role": "user", "content": "first and initially sole user"},
    ]

    try:
        engine.compress(first)
        assert engine._leading_anchor_count(first) == 2
        engine.threshold_tokens = 1
        messages = [
            *first,
            {"role": "user", "content": "second user invalidates the anchor"},
        ]
        should_compress = engine.should_compress_preflight(messages)
        anchor_metadata = engine._store.read_metadata_json(
            engine._retained_user_anchor_metadata_key()
        )
    finally:
        engine.shutdown()

    assert should_compress is True
    assert anchor_metadata == {"store_id": 0, "version": 1}


def test_second_real_user_disqualifies_first_raw_anchor(
    tmp_path,
    monkeypatch,
) -> None:
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "issue-3-second-user.db"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "issue-3-second-user",
        platform="cli",
        context_length=200_000,
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    first_user = {"role": "user", "content": "first prompt becomes stale"}
    second_user = {"role": "user", "content": "second prompt is current"}

    try:
        first_result = engine.compress(
            [
                {"role": "system", "content": "stable system prompt"},
                first_user,
                *_tool_chain(),
            ]
        )
        second_result = engine.compress(
            [
                *first_result,
                second_user,
                *_tool_chain(count=4),
            ]
        )
        anchor_metadata = engine._store.read_metadata_json(
            engine._retained_user_anchor_metadata_key()
        )
    finally:
        engine.shutdown()

    user_contents = [
        str(message.get("content") or "")
        for message in second_result
        if message.get("role") == "user"
    ]
    assert any(second_user["content"] in content for content in user_contents)
    assert first_user["content"] not in user_contents
    assert anchor_metadata == {"store_id": 0, "version": 1}
