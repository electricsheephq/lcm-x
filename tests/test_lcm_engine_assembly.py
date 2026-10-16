"""LCM engine integration tests: assembly."""

import json
import logging
import time


import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine



class TestAssemblyGuardrails:
    def test_max_assembly_tokens_caps_recent_tail(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail.db"),
            max_assembly_tokens=60,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )

        result = instance._assemble_context(
            {"role": "system", "content": "s" * 10},
            [
                {"role": "user", "content": "a" * 20},
                {"role": "assistant", "content": "b" * 20},
                {"role": "user", "content": "c" * 20},
            ],
        )

        assert [msg["content"] for msg in result[1:]] == ["b" * 20, "c" * 20]

    def test_reserve_tokens_floor_caps_recent_tail(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_headroom.db"),
            reserve_tokens_floor=40,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1
        instance.context_length = 100

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )

        result = instance._assemble_context(
            {"role": "system", "content": "s" * 10},
            [
                {"role": "user", "content": "a" * 20},
                {"role": "assistant", "content": "b" * 20},
                {"role": "user", "content": "c" * 20},
            ],
        )

        assert [msg["content"] for msg in result[1:]] == ["b" * 20, "c" * 20]

    def test_max_assembly_tokens_does_not_emit_raw_messages_across_droppable_assistant_gap(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_varied.db"),
            max_assembly_tokens=70,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )

        result = instance._assemble_context(
            {"role": "system", "content": "s" * 10},
            [
                {"role": "user", "content": "a" * 10},
                {"role": "assistant", "content": "b" * 45},
                {"role": "user", "content": "c" * 20},
            ],
        )

        assert [msg["content"] for msg in result[1:]] == ["c" * 20]

    def test_summary_budget_skips_oversized_summary_and_keeps_later_fit_part(self, tmp_path, monkeypatch):
        import importlib
        from hermes_lcm.dag import SummaryNode

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_summary.db"),
            max_assembly_tokens=189,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )

        instance._dag.add_node(SummaryNode(
            session_id="guardrail-session", depth=2,
            summary="A" * 15, token_count=15,
            source_token_count=100, source_ids=[],
            source_type="messages", created_at=time.time(),
        ))
        instance._dag.add_node(SummaryNode(
            session_id="guardrail-session", depth=1,
            summary="B" * 120, token_count=120,
            source_token_count=200, source_ids=[],
            source_type="messages", created_at=time.time(),
        ))
        instance._dag.add_node(SummaryNode(
            session_id="guardrail-session", depth=0,
            summary="C" * 10, token_count=10,
            source_token_count=80, source_ids=[],
            source_type="messages", created_at=time.time(),
        ))

        result = instance._assemble_context(
            {"role": "system", "content": "s" * 10},
            [{"role": "user", "content": "tail" * 10}],
        )

        assert len(result) == 3
        summary_blob = result[1]["content"]
        assert "A" * 15 in summary_blob
        assert "B" * 120 not in summary_blob
        assert "C" * 10 in summary_blob

    def test_max_assembly_tokens_drops_oversized_newest_assistant_and_keeps_user_prompt(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_newest.db"),
            max_assembly_tokens=120,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )

        result = instance._assemble_context(
            {"role": "system", "content": "s" * 10},
            [
                {"role": "user", "content": "a" * 20},
                {"role": "assistant", "content": "b" * 140},
            ],
        )

        contents = [msg["content"] for msg in result[1:]]
        assert any("a" * 20 in content for content in contents)
        assert not any(msg.get("role") == "user" and msg.get("content") == "a" * 20 for msg in result[1:])

    def test_context_anchor_is_budgeted_under_max_assembly_tokens(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=1,
            database_path=str(tmp_path / "lcm_guardrail_anchor_budget.db"),
            max_assembly_tokens=120,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )
        monkeypatch.setattr(lcm_engine_module, "count_tokens", lambda text: len(text))
        monkeypatch.setattr(
            lcm_engine_module,
            "summarize_with_escalation",
            lambda **kwargs: ("summary", 1),
        )

        oversized_anchor = "current objective " * 7
        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "stale"},
            {"role": "user", "content": oversized_anchor},
            {"role": "assistant", "content": "a" * 50},
            {"role": "tool", "tool_call_id": "call_anchor", "content": "t" * 50},
        ]

        result = instance.compress(messages, current_tokens=140)

        assert lcm_engine_module.count_messages_tokens(result) <= 120
        assert oversized_anchor not in [msg.get("content") for msg in result]
        assert not instance.get_status()["overflow_recovery_failed"]

    def test_reserve_tokens_floor_warns_when_misconfigured(self, tmp_path, caplog):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_guardrail_warn.db"),
            reserve_tokens_floor=100,
        )
        instance = LCMEngine(config=config)
        instance.context_length = 100

        with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
            assert instance._effective_assembly_token_cap() is None

        assert "reserve_tokens_floor=100 disables reserve-based assembly cap" in caplog.text

    def test_compress_forces_overflow_recovery_when_context_hits_assembly_cap(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=100,
            database_path=str(tmp_path / "lcm_guardrail_forced.db"),
            max_assembly_tokens=90,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )
        monkeypatch.setattr(lcm_engine_module, "count_tokens", lambda text: len(text))
        monkeypatch.setattr(
            lcm_engine_module,
            "summarize_with_escalation",
            lambda **kwargs: ("summary", 1),
        )

        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 20},
            {"role": "user", "content": "c" * 20},
            {"role": "assistant", "content": "d" * 20},
        ]

        result = instance.compress(messages, current_tokens=90)

        assert len(result) < len(messages)
        assert result[-2:] == messages[-2:]
        assert lcm_engine_module.count_messages_tokens(result) < 90
        assert instance._dag.get_session_nodes("guardrail-session")

    def test_forced_overflow_tail_capping_updates_bookkeeping_without_middle_compaction(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_tail_only.db"),
            max_assembly_tokens=70,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )

        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 40},
            {"role": "assistant", "content": "b" * 40},
        ]

        result = instance.compress(messages, current_tokens=90)

        assert result == messages[:2]
        assert instance.compression_count == 1
        assert instance._ingest_cursor == len(result)
        assert not instance.get_status()["overflow_recovery_failed"]

    def test_forced_overflow_recovery_reserves_provider_overhead(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_overhead.db"),
            max_assembly_tokens=90,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )

        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 30},
            {"role": "assistant", "content": "b" * 40},
        ]

        result = instance.compress(messages, current_tokens=100)

        assert result == messages[:2]
        assert lcm_engine_module.count_messages_tokens(result) < 70

    def test_forced_overflow_recovery_does_not_duplicate_existing_summary_message(self, tmp_path, monkeypatch):
        import importlib
        from hermes_lcm.dag import SummaryNode

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_summary_dup.db"),
            max_assembly_tokens=90,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )

        node = SummaryNode(
            session_id="guardrail-session",
            depth=0,
            summary="sum",
            token_count=3,
            source_token_count=50,
            source_ids=[],
            source_type="messages",
            created_at=time.time(),
            expand_hint="x",
        )
        node_id = instance._dag.add_node(node)
        summary_blob = (
            f"[Recent Summary (d0, node {node_id})]\n"
            f"sum\n"
            f"[Expand for details: x]"
        )
        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "assistant", "content": summary_blob},
            {"role": "user", "content": "tail" * 2},
        ]

        result = instance.compress(messages, current_tokens=90)

        joined = "\n\n".join(msg.get("content", "") for msg in result)
        assert joined.count("[Expand for details:") == 1
        assert not instance.get_status()["overflow_recovery_failed"]

    def test_forced_overflow_recovery_flags_irreducible_single_tail_overflow(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_irreducible.db"),
            max_assembly_tokens=70,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )

        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "assistant", "content": "a" * 20},
            {"role": "user", "content": "b" * 80},
        ]

        result = instance.compress(messages, current_tokens=110)

        assert result == [messages[0], messages[-1]]
        assert instance.get_status()["overflow_recovery_failed"]

    def test_overflow_recovery_failure_flag_resets_after_successful_compression(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=100,
            database_path=str(tmp_path / "lcm_guardrail_flag_reset.db"),
            max_assembly_tokens=70,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )
        monkeypatch.setattr(lcm_engine_module, "count_tokens", lambda text: len(text))
        monkeypatch.setattr(
            lcm_engine_module,
            "summarize_with_escalation",
            lambda **kwargs: ("summary", 1),
        )

        failed_messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "assistant", "content": "a" * 20},
            {"role": "user", "content": "b" * 80},
        ]
        instance.compress(failed_messages, current_tokens=110)
        assert instance.get_status()["overflow_recovery_failed"]

        success_messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 20},
            {"role": "user", "content": "c" * 20},
            {"role": "assistant", "content": "d" * 20},
        ]
        instance.compress(success_messages, current_tokens=90)

        assert not instance.get_status()["overflow_recovery_failed"]

    def test_compress_ignores_stale_last_prompt_tokens_for_overflow_recovery(self, tmp_path, monkeypatch):
        import importlib

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_guardrail_stale_prompt.db"),
            max_assembly_tokens=70,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "guardrail-session"
        instance.compression_count = 1
        instance.last_prompt_tokens = 200

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(msg.get("content", "")),
        )
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(msg.get("content", "")) for msg in messages),
        )

        messages = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 20},
        ]

        result = instance.compress(messages)

        assert result == messages



class TestAssemblyToolPairGuardrail:
    """Regression: active context must return provider-valid tool sequences."""

    def _make_engine(self, tmp_path, db_name="lcm_tool_pairs.db"):
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / db_name),
        )
        instance = LCMEngine(config=config)
        instance._session_id = "tool-pair-test"
        instance.compression_count = 1
        instance.context_length = 200000
        return instance

    def _assert_provider_tool_sequence_valid(self, messages):
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "tool":
                raise AssertionError(f"bare/late tool result at index {i}: {msg!r}")

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                expected_ids = [
                    str((tool_call or {}).get("id") or (tool_call or {}).get("tool_call_id") or "").strip()
                    for tool_call in (msg.get("tool_calls") or [])
                    if isinstance(tool_call, dict)
                ]
                expected_ids = [call_id for call_id in expected_ids if call_id]

                for offset, expected_id in enumerate(expected_ids, start=1):
                    assert i + offset < len(messages), (
                        f"missing direct tool result for {expected_id} after assistant index {i}"
                    )
                    tool_msg = messages[i + offset]
                    assert tool_msg.get("role") == "tool", (
                        f"expected tool result for {expected_id} at index {i + offset}, got {tool_msg!r}"
                    )
                    assert str(tool_msg.get("tool_call_id") or "").strip() == expected_id

                i += 1 + len(expected_ids)
                continue

            i += 1

    def test_assemble_removes_orphan_tool_result(self, tmp_path):
        """When a tool result references a call_id whose assistant tool_call
        was removed (e.g., compacted by LCM), the assembled active context
        must not contain that orphan tool result."""
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_orphan.db"),
        )
        instance = LCMEngine(config=config)
        instance._session_id = "orphan-test"
        instance.compression_count = 1
        instance.context_length = 200000

        sys_msg = {"role": "system", "content": "You are helpful."}
        # Simulate real-world shape: assistant summary replaced the original
        # tool_call, leaving an orphan tool result in the fresh tail.
        tail_messages = [
            {"role": "assistant", "content": "[Session Arc Summary] ..."},
            {"role": "tool", "tool_call_id": "call_orphan_x", "content": "orphan result"},
            {"role": "assistant", "tool_calls": [{"id": "call_ok", "function": {"name": "patch", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_ok", "content": "patch result"},
            {"role": "assistant", "content": "Done."},
        ]

        result = instance._assemble_context(sys_msg, tail_messages)

        # The orphan tool result (call_orphan_x) must be removed
        orphan_ids = [
            m.get("tool_call_id") for m in result
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_orphan_x"
        ]
        assert len(orphan_ids) == 0, f"Orphan tool result still present: {orphan_ids}"

    def test_assemble_inserts_stub_for_missing_tool_result(self, tmp_path):
        """When an assistant tool_call has no matching tool result in the
        assembled context, a stub result must be inserted."""
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_stub.db"),
        )
        instance = LCMEngine(config=config)
        instance._session_id = "stub-test"
        instance.compression_count = 1
        instance.context_length = 200000

        sys_msg = {"role": "system", "content": "You are helpful."}
        tail_messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_no_result", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "assistant", "content": "Continuing..."},
        ]

        result = instance._assemble_context(sys_msg, tail_messages)

        # There must be a stub tool result for call_no_result
        stub_ids = [
            m.get("tool_call_id") for m in result
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_no_result"
        ]
        assert len(stub_ids) >= 1, f"No stub result for assistant tool_call: {stub_ids}"

    def test_assemble_drops_structured_blank_and_thinking_only_assistant_messages(self, tmp_path):
        instance = self._make_engine(tmp_path, "lcm_blank_thinking_cleanup.db")
        sys_msg = {"role": "system", "content": "sys"}
        blank_content = [{"type": "text", "text": ""}]
        thinking_content = [{"type": "thinking", "thinking": "private chain of thought"}]
        visible_content = [{"type": "text", "text": "Visible answer"}]
        tail_messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": blank_content},
            {"role": "assistant", "content": thinking_content},
            {"role": "assistant", "content": visible_content},
        ]

        result = instance._assemble_context(sys_msg, tail_messages)

        assert {"role": "assistant", "content": blank_content} not in result
        assert {"role": "assistant", "content": thinking_content} not in result
        assert {"role": "assistant", "content": visible_content} in result
        self._assert_provider_tool_sequence_valid(result)

    def test_assembly_cap_ignores_dropped_internal_assistant_turns_during_tail_selection(self, tmp_path, monkeypatch):
        import importlib

        lcm_engine_module = importlib.import_module("hermes_lcm.engine")
        monkeypatch.setattr(
            lcm_engine_module,
            "count_message_tokens",
            lambda msg: len(str(msg.get("content", ""))),
        )
        monkeypatch.setattr(
            lcm_engine_module,
            "count_messages_tokens",
            lambda messages: sum(len(str(msg.get("content", ""))) for msg in messages),
        )

        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_cap_precleanup_tail.db"),
            max_assembly_tokens=50,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "cap-precleanup-test"
        instance.compression_count = 1
        instance.context_length = 200000

        sys_msg = {"role": "system", "content": "s" * 5}
        user_msg = {"role": "user", "content": "VISIBLE_USER_OBJECTIVE"}
        noisy_assistant = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "x" * 500}],
        }

        result = instance._assemble_context(
            sys_msg,
            [user_msg, noisy_assistant],
            assembly_cap_override=50,
        )

        assert user_msg in result
        assert noisy_assistant not in result
        self._assert_provider_tool_sequence_valid(result)

    def test_assemble_cleanup_preserves_valid_tool_call_adjacency(self, tmp_path):
        instance = self._make_engine(tmp_path, "lcm_tool_call_cleanup_preserve.db")
        sys_msg = {"role": "system", "content": "sys"}
        tool_call_msg = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "deciding which tool to call"}],
            "tool_calls": [{"id": "call_keep", "function": {"name": "terminal", "arguments": "{}"}}],
        }
        tool_result_msg = {"role": "tool", "tool_call_id": "call_keep", "content": "tool output"}
        tail_messages = [
            {"role": "user", "content": "run it"},
            tool_call_msg,
            tool_result_msg,
            {"role": "assistant", "content": "Done."},
        ]

        result = instance._assemble_context(sys_msg, tail_messages)

        expected_tool_call_msg = dict(tool_call_msg)
        expected_tool_call_msg["content"] = ""
        assert expected_tool_call_msg in result
        assert tool_result_msg in result
        call_index = result.index(expected_tool_call_msg)
        assert result[call_index + 1] == tool_result_msg
        self._assert_provider_tool_sequence_valid(result)

    def test_assemble_cleanup_repairs_tool_sequence_after_dropping_blank_turn(self, tmp_path):
        instance = self._make_engine(tmp_path, "lcm_tool_call_cleanup_repair.db")
        sys_msg = {"role": "system", "content": "sys"}
        tool_call_msg = {
            "role": "assistant",
            "tool_calls": [{"id": "call_repair", "function": {"name": "terminal", "arguments": "{}"}}],
        }
        blank_content = [{"type": "text", "text": ""}]
        real_tool_result = {"role": "tool", "tool_call_id": "call_repair", "content": "real output"}
        tail_messages = [
            {"role": "user", "content": "run it"},
            tool_call_msg,
            {"role": "assistant", "content": blank_content},
            real_tool_result,
            {"role": "assistant", "content": "Done."},
        ]

        result = instance._assemble_context(sys_msg, tail_messages)

        assert {"role": "assistant", "content": blank_content} not in result
        call_index = result.index(tool_call_msg)
        assert result[call_index + 1] == real_tool_result
        self._assert_provider_tool_sequence_valid(result)

    def test_compress_drops_unsafe_assistant_content_without_mutating_store(self, tmp_path, monkeypatch):
        def mock_summary(**kwargs):
            return "Leaf summary.\nExpand for details about: cleanup", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", mock_summary)
        config = LCMConfig(
            fresh_tail_count=4,
            database_path=str(tmp_path / "lcm_compress_cleanup.db"),
            leaf_chunk_tokens=80,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "compress-cleanup-test"
        instance.compression_count = 1
        instance.context_length = 200000
        instance.threshold_tokens = 1

        blank_content = [{"type": "text", "text": ""}]
        thinking_content = [{"type": "reasoning", "text": "internal reasoning only"}]
        visible_content = [{"type": "text", "text": "Visible final answer"}]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old question " + "x" * 200},
            {"role": "assistant", "content": "old answer " + "y" * 200},
            {"role": "user", "content": "current question"},
            {"role": "assistant", "content": blank_content},
            {"role": "assistant", "content": thinking_content},
            {"role": "assistant", "content": visible_content},
        ]

        result = instance.compress(messages)

        assert {"role": "assistant", "content": blank_content} not in result
        assert {"role": "assistant", "content": thinking_content} not in result
        assert {"role": "assistant", "content": visible_content} in result
        assert instance._store.get_session_count("compress-cleanup-test") == len(messages)
        stored_contents = [
            row.get("content")
            for row in instance._store.get_range("compress-cleanup-test", limit=20)
        ]
        assert json.dumps(blank_content, ensure_ascii=False, sort_keys=True) in stored_contents
        assert json.dumps(thinking_content, ensure_ascii=False, sort_keys=True) in stored_contents
        self._assert_provider_tool_sequence_valid(result)

    def test_no_compaction_cleanup_resets_cursor_for_next_turn(self, tmp_path):
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_no_compaction_cursor_cleanup.db"),
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "cursor-cleanup-test"
        instance.context_length = 200000
        instance.threshold_tokens = 190000

        blank_content = [{"type": "text", "text": ""}]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": blank_content},
            {"role": "assistant", "content": "visible answer"},
        ]

        sanitized = instance.compress(messages)
        assert len(sanitized) == len(messages) - 1
        assert instance._ingest_cursor == len(sanitized)
        assert instance._store.get_session_count("cursor-cleanup-test") == len(messages)

        next_messages = sanitized + [{"role": "user", "content": "new follow-up"}]
        instance.compress(next_messages)

        rows = instance._store.get_session_messages("cursor-cleanup-test")
        assert len(rows) == len(messages) + 1
        assert rows[-1]["content"] == "new follow-up"

    def test_rebind_reconciliation_tolerates_sanitized_active_context_cleanup(self, tmp_path):
        db_path = str(tmp_path / "lcm_rebind_sanitized_active_cleanup.db")
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        session_id = "rebind-cleanup-test"
        blank_content = [{"type": "text", "text": ""}]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": blank_content},
            {"role": "assistant", "content": "visible answer"},
        ]

        first = LCMEngine(config=config)
        first.on_session_start(session_id, context_length=200000)
        sanitized = first.compress(messages)

        assert len(sanitized) == 3
        assert first._store.get_session_count(session_id) == 4
        first.shutdown()

        rebound = LCMEngine(config=LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        ))
        rebound.on_session_start(session_id, context_length=200000)
        rebound.compress(sanitized + [{"role": "user", "content": "new follow-up"}])

        rows = rebound._store.get_session_messages(session_id)
        assert len(rows) == 5
        assert [row["role"] for row in rows] == [
            "system",
            "user",
            "assistant",
            "assistant",
            "user",
        ]
        assert [row["content"] for row in rows] == [
            "sys",
            "question",
            json.dumps(blank_content, ensure_ascii=False, sort_keys=True),
            "visible answer",
            "new follow-up",
        ]
        assert rebound._last_ingest_reconciliation["action"] == "advanced cursor"
        assert rebound._last_ingest_reconciliation["cursor"] == len(sanitized)

    def test_no_compaction_cleanup_does_not_return_untracked_tool_stubs_after_rebind(self, tmp_path):
        db_path = str(tmp_path / "lcm_no_compaction_pending_tool_stub.db")
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        session_id = "pending-tool-stub-test"
        pending_call = {
            "id": "call_pending",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "tool_calls": [pending_call]},
        ]

        first = LCMEngine(config=config)
        first.on_session_start(session_id, context_length=200000)
        active_context = first.compress(messages)

        assert active_context == messages
        assert all(msg.get("role") != "tool" for msg in active_context)
        assert first._store.get_session_count(session_id) == 3
        first.shutdown()

        rebound = LCMEngine(config=LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        ))
        rebound.on_session_start(session_id, context_length=200000)
        rebound.compress(active_context + [{"role": "user", "content": "new follow-up"}])

        rows = rebound._store.get_session_messages(session_id)
        assert len(rows) == 4
        assert [row["role"] for row in rows] == ["system", "user", "assistant", "user"]
        assert rows[-1]["content"] == "new follow-up"
        assert rebound._last_ingest_reconciliation["action"] == "advanced cursor"

    def test_active_context_cleanup_strips_internal_parts_from_mixed_assistant_content(self, tmp_path):
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_mixed_internal_cleanup.db"),
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        instance = LCMEngine(config=config)
        instance.on_session_start("mixed-internal-cleanup-test", context_length=200000)
        mixed_content = [
            {"type": "thinking", "text": "secret chain of thought"},
            {"type": "text", "text": "visible final"},
        ]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": mixed_content},
            {"role": "assistant", "content": "<think>hidden</think>string final"},
        ]

        active_context = instance.compress(messages)

        assert active_context[2] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "visible final"}],
        }
        assert active_context[3] == {"role": "assistant", "content": "string final"}
        rows = instance._store.get_session_messages("mixed-internal-cleanup-test")
        assert rows[2]["content"] == json.dumps(mixed_content, ensure_ascii=False, sort_keys=True)
        assert rows[3]["content"] == "<think>hidden</think>string final"

    def test_rebind_reconciliation_tolerates_stripped_active_assistant_content(self, tmp_path):
        db_path = str(tmp_path / "lcm_rebind_stripped_active_cleanup.db")
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        session_id = "rebind-stripped-cleanup-test"
        mixed_content = [
            {"type": "thinking", "text": "secret chain of thought"},
            {"type": "text", "text": "visible final"},
        ]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": mixed_content},
        ]

        first = LCMEngine(config=config)
        first.on_session_start(session_id, context_length=200000)
        active_context = first.compress(messages)

        assert active_context == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [{"type": "text", "text": "visible final"}]},
        ]
        assert first._store.get_session_count(session_id) == 3
        first.shutdown()

        rebound = LCMEngine(config=LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        ))
        rebound.on_session_start(session_id, context_length=200000)
        rebound.compress(active_context + [{"role": "user", "content": "new follow-up"}])

        rows = rebound._store.get_session_messages(session_id)
        assert len(rows) == 4
        assert [row["role"] for row in rows] == ["system", "user", "assistant", "user"]
        assert rows[2]["content"] == json.dumps(mixed_content, ensure_ascii=False, sort_keys=True)
        assert rows[3]["content"] == "new follow-up"
        assert rebound._last_ingest_reconciliation["action"] == "advanced cursor"
        assert rebound._last_ingest_reconciliation["cursor"] == len(active_context)

    def test_active_context_cleanup_strips_internal_content_from_assistant_tool_calls(self, tmp_path):
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_tool_call_internal_cleanup.db"),
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        instance = LCMEngine(config=config)
        session_id = "tool-call-internal-cleanup-test"
        instance.on_session_start(session_id, context_length=200000)
        tool_call = {
            "id": "call_lookup",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<think>hidden</think>", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_lookup", "content": "result"},
        ]

        active_context = instance.compress(messages)

        assert active_context[2] == {"role": "assistant", "content": "", "tool_calls": [tool_call]}
        assert active_context[3] == {"role": "tool", "tool_call_id": "call_lookup", "content": "result"}
        rows = instance._store.get_session_messages(session_id)
        assert rows[2]["content"] == "<think>hidden</think>"
        assert rows[2]["tool_calls"] == [tool_call]

    def test_rebind_reconciliation_tolerates_stripped_assistant_tool_call_content(self, tmp_path):
        db_path = str(tmp_path / "lcm_rebind_tool_call_internal_cleanup.db")
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        session_id = "rebind-tool-call-internal-cleanup-test"
        tool_call = {
            "id": "call_lookup",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<think>hidden</think>", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_lookup", "content": "result"},
        ]

        first = LCMEngine(config=config)
        first.on_session_start(session_id, context_length=200000)
        active_context = first.compress(messages)
        assert active_context[2]["content"] == ""
        assert first._store.get_session_count(session_id) == 4
        first.shutdown()

        rebound = LCMEngine(config=LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        ))
        rebound.on_session_start(session_id, context_length=200000)
        rebound.compress(active_context + [{"role": "user", "content": "new follow-up"}])

        rows = rebound._store.get_session_messages(session_id)
        assert len(rows) == 5
        assert [row["role"] for row in rows] == ["system", "user", "assistant", "tool", "user"]
        assert rows[2]["content"] == "<think>hidden</think>"
        assert rows[4]["content"] == "new follow-up"
        assert rebound._last_ingest_reconciliation["action"] == "advanced cursor"
        assert rebound._last_ingest_reconciliation["cursor"] == len(active_context)

    def test_rebind_reconciliation_keeps_literal_json_string_assistant_content(self, tmp_path):
        db_path = str(tmp_path / "lcm_rebind_literal_json_string_cleanup.db")
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        session_id = "rebind-literal-json-string-test"
        literal_json_text = json.dumps(
            [{"type": "thinking", "text": "this is user-visible literal JSON text"}],
            ensure_ascii=False,
            sort_keys=True,
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": literal_json_text},
        ]

        first = LCMEngine(config=config)
        first.on_session_start(session_id, context_length=200000)
        active_context = first.compress(messages)

        assert active_context == messages
        assert first._store.get_session_count(session_id) == 3
        first.shutdown()

        rebound = LCMEngine(config=LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        ))
        rebound.on_session_start(session_id, context_length=200000)
        rebound.compress(active_context + [{"role": "user", "content": "new follow-up"}])

        rows = rebound._store.get_session_messages(session_id)
        assert len(rows) == 4
        assert [row["role"] for row in rows] == ["system", "user", "assistant", "user"]
        assert [row["content"] for row in rows].count(literal_json_text) == 1
        assert rows[-1]["content"] == "new follow-up"
        assert rebound._last_ingest_reconciliation["action"] == "advanced cursor"
        assert rebound._last_ingest_reconciliation["cursor"] == len(active_context)

    def test_compacted_rebind_keeps_literal_json_string_assistant_content(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "lcm_rebind_compacted_literal_json_string_cleanup.db")
        config = LCMConfig(
            fresh_tail_count=2,
            database_path=db_path,
            leaf_chunk_tokens=1,
            context_threshold=0.95,
        )
        session_id = "rebind-compacted-literal-json-string-test"
        literal_json_text = json.dumps(
            [{"type": "thinking", "text": "this is still visible literal JSON"}],
            ensure_ascii=False,
            sort_keys=True,
        )

        def mock_summary(**kwargs):
            return "Older literal-json replay setup summary", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", mock_summary)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "older question"},
            {"role": "assistant", "content": "older answer"},
            {"role": "user", "content": "fresh question"},
            {"role": "assistant", "content": literal_json_text},
        ]

        first = LCMEngine(config=config)
        first.on_session_start(session_id, context_length=200000)
        active_context = first.compress(messages)
        assert any("Older literal-json replay setup summary" in (msg.get("content") or "") for msg in active_context)
        assert active_context[-2:] == messages[-2:]
        assert first._store.get_session_count(session_id) == len(messages)
        first.shutdown()

        rebound = LCMEngine(config=LCMConfig(
            fresh_tail_count=2,
            database_path=db_path,
            leaf_chunk_tokens=1,
            context_threshold=0.95,
        ))
        rebound.on_session_start(session_id, context_length=200000)
        rebound.compress(active_context + [{"role": "user", "content": "new follow-up"}])

        rows = rebound._store.get_session_messages(session_id)
        assert len(rows) == len(messages) + 1
        assert [row["content"] for row in rows].count(literal_json_text) == 1
        assert rows[-1]["content"] == "new follow-up"
        assert rebound._last_ingest_reconciliation["action"] == "advanced cursor"
        assert rebound._last_ingest_reconciliation["cursor"] == len(active_context)

    def test_source_id_mapping_matches_stripped_assistant_active_context(self, tmp_path):
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_source_id_stripped_cleanup.db"),
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        instance = LCMEngine(config=config)
        session_id = "source-id-stripped-cleanup-test"
        instance.on_session_start(session_id, context_length=200000)
        mixed_content = [
            {"type": "thinking", "text": "secret chain of thought"},
            {"type": "text", "text": "visible final"},
        ]
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": mixed_content},
        ]

        active_context = instance.compress(messages)
        rows = instance._store.get_session_messages(session_id)

        assert active_context[2]["content"] == [{"type": "text", "text": "visible final"}]
        assert rows[2]["content"] == json.dumps(mixed_content, ensure_ascii=False, sort_keys=True)
        assert instance._get_store_ids_for_messages([active_context[2]]) == [rows[2]["store_id"]]

    def test_source_id_mapping_matches_stripped_tool_call_active_context(self, tmp_path):
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_source_id_tool_call_cleanup.db"),
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        instance = LCMEngine(config=config)
        session_id = "source-id-tool-call-cleanup-test"
        instance.on_session_start(session_id, context_length=200000)
        tool_call = {
            "id": "call_lookup",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<think>hidden</think>", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call_lookup", "content": "result"},
        ]

        active_context = instance.compress(messages)
        rows = instance._store.get_session_messages(session_id)

        assert active_context[2]["content"] == ""
        assert rows[2]["content"] == "<think>hidden</think>"
        assert instance._get_store_ids_for_messages([active_context[2]]) == [rows[2]["store_id"]]

    def test_rebind_reconciliation_preserves_visible_suffix_delta_when_sanitized_tail_collapsed(self, tmp_path):
        db_path = str(tmp_path / "lcm_rebind_collapsed_tail_delta.db")
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        )
        session_id = "collapsed-tail-delta-test"
        stored_messages = [
            {"role": "assistant", "content": [{"type": "thinking", "text": "hidden"}]},
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ]

        first = LCMEngine(config=config)
        first.on_session_start(session_id, context_length=200000)
        first.compress(stored_messages)
        assert first._store.get_session_count(session_id) == 3
        first.shutdown()

        rebound = LCMEngine(config=LCMConfig(
            fresh_tail_count=10,
            database_path=db_path,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
        ))
        rebound.on_session_start(session_id, context_length=200000)
        rebound.compress([
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ])

        rows = rebound._store.get_session_messages(session_id)
        assert len(rows) == 5
        assert [row["role"] for row in rows] == ["assistant", "user", "assistant", "user", "assistant"]
        assert [row["content"] for row in rows[-2:]] == ["ping", "pong"]
        assert rebound._last_ingest_reconciliation["action"] == "persisted batch"

    def test_compress_output_is_valid_tool_pair_sequence(self, tmp_path, monkeypatch):
        """Full compress() output must not contain orphan tool results
        and must include stubs for missing results."""
        import importlib
        esc_module = importlib.import_module("hermes_lcm.escalation")
        importlib.import_module("hermes_lcm.engine")

        config = LCMConfig(
            fresh_tail_count=4,
            database_path=str(tmp_path / "lcm_compress_pair.db"),
            leaf_chunk_tokens=200,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "compress-pair-test"
        instance.context_length = 200000
        instance.threshold_tokens = 500

        def mock_summary(**kwargs):
            return "Leaf summary.\nExpand for details about: test", 1

        monkeypatch.setattr(esc_module, "summarize_with_escalation", mock_summary)

        messages = [{"role": "system", "content": "You are helpful."}]
        # Build a conversation where an assistant tool_call gets compacted
        # but its tool result might survive into the fresh tail.
        messages.append({"role": "user", "content": "Q0: " + "x" * 200})
        # This assistant tool_call + result pair will be compacted:
        messages.append({"role": "assistant", "tool_calls": [{"id": "call_compacted", "function": {"name": "terminal", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": "call_compacted", "content": "result that gets compacted"})
        # More filler to push the pair into the raw backlog:
        for i in range(1, 10):
            messages.append({"role": "user", "content": f"Q{i}: " + "y" * 200})
            messages.append({"role": "assistant", "content": f"A{i}: " + "z" * 200})

        result = instance.compress(messages)

        # After compression, no orphan tool results
        assistant_ids = set()
        for m in result:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m.get("tool_calls") or []:
                    cid = tc.get("id") if isinstance(tc, dict) else ""
                    if cid:
                        assistant_ids.add(cid)
        result_ids = set()
        for m in result:
            if m.get("role") == "tool" and m.get("tool_call_id"):
                result_ids.add(m.get("tool_call_id"))
        orphaned = result_ids - assistant_ids
        assert len(orphaned) == 0, f"Orphan tool results after compress: {orphaned}"
        # And no missing results (every assistant call has a result or stub)
        missing = assistant_ids - result_ids
        # Missing results should have stubs — verify they exist
        for cid in missing:
            stub_found = any(
                m.get("role") == "tool" and m.get("tool_call_id") == cid
                for m in result
            )
            assert stub_found, f"Missing stub for tool_call_id {cid}"

    def test_overflow_recovery_fallback_removes_orphan_tool_result(self, tmp_path):
        """Overflow recovery fallback must not return a bare orphan tool result."""
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_overflow_orphan.db"),
        )
        instance = LCMEngine(config=config)
        instance._session_id = "overflow-orphan-test"
        instance.compression_count = 1
        instance.context_length = 200000

        sys_msg = {"role": "system", "content": "sys"}
        tail_messages = [
            {"role": "user", "content": "u" * 200},
            {"role": "tool", "tool_call_id": "call_orphan", "content": "orphan tool result"},
        ]

        result = instance._assemble_overflow_recovery_context(
            sys_msg,
            tail_messages,
            assembly_cap_override=1,
        )

        orphan_ids = [
            m.get("tool_call_id") for m in result
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_orphan"
        ]
        assert len(orphan_ids) == 0, f"Overflow fallback leaked orphan tool result: {orphan_ids}"

    def test_overflow_recovery_fallback_inserts_stub_for_missing_tool_result(self, tmp_path):
        """Overflow recovery fallback must sanitize an assistant tool_call-only tail."""
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_overflow_stub.db"),
        )
        instance = LCMEngine(config=config)
        instance._session_id = "overflow-stub-test"
        instance.compression_count = 1
        instance.context_length = 200000

        sys_msg = {"role": "system", "content": "sys"}
        tail_messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_missing", "function": {"name": "terminal", "arguments": "{}"}}],
            },
        ]

        result = instance._assemble_overflow_recovery_context(
            sys_msg,
            tail_messages,
            assembly_cap_override=1,
        )

        stub_ids = [
            m.get("tool_call_id") for m in result
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_missing"
        ]
        assert len(stub_ids) >= 1, f"Overflow fallback missing stub tool result: {stub_ids}"

    def test_sanitize_tool_pairs_is_idempotent(self, tmp_path):
        """Applying the helper twice must not change the result again."""
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_idempotent.db"),
        )
        instance = LCMEngine(config=config)
        instance._session_id = "idempotent-test"
        instance.compression_count = 1
        instance.context_length = 200000

        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_once", "function": {"name": "terminal", "arguments": "{}"}}],
            },
            {"role": "assistant", "content": "after"},
        ]

        once = instance._sanitize_tool_pairs([dict(m) for m in messages])
        twice = instance._sanitize_tool_pairs([dict(m) for m in once])
        assert once == twice

    def test_sanitize_tool_pairs_keeps_valid_sequence_unchanged(self, tmp_path):
        """A valid tool-call/result sequence must be preserved as-is."""
        config = LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_valid_unchanged.db"),
        )
        instance = LCMEngine(config=config)
        instance._session_id = "valid-unchanged-test"
        instance.compression_count = 1
        instance.context_length = 200000

        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_ok", "function": {"name": "terminal", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_ok", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]

        result = instance._sanitize_tool_pairs([dict(m) for m in messages])
        assert result == messages
        self._assert_provider_tool_sequence_valid(result)

    def test_sanitize_tool_pairs_drops_late_tool_result_after_intervening_message(self, tmp_path):
        instance = self._make_engine(tmp_path, "lcm_late_tool_result.db")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_late", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "user", "content": "intervening turn"},
            {"role": "tool", "tool_call_id": "call_late", "content": "late result"},
        ]

        result = instance._sanitize_tool_pairs([dict(m) for m in messages])

        self._assert_provider_tool_sequence_valid(result)
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_late"
        assert "earlier conversation" in result[1]["content"]
        assert all(msg.get("content") != "late result" for msg in result)

    def test_sanitize_tool_pairs_drops_duplicate_late_result(self, tmp_path):
        instance = self._make_engine(tmp_path, "lcm_duplicate_tool_result.db")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_dup", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_dup", "content": "direct result"},
            {"role": "assistant", "content": "done"},
            {"role": "tool", "tool_call_id": "call_dup", "content": "duplicate late result"},
        ]

        result = instance._sanitize_tool_pairs([dict(m) for m in messages])

        self._assert_provider_tool_sequence_valid(result)
        assert [msg.get("content") for msg in result].count("direct result") == 1
        assert all(msg.get("content") != "duplicate late result" for msg in result)

    def test_sanitize_tool_pairs_keeps_ordered_parallel_results(self, tmp_path):
        instance = self._make_engine(tmp_path, "lcm_parallel_ordered.db")
        messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "call_a", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "call_b", "function": {"name": "terminal", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_a", "content": "A"},
            {"role": "tool", "tool_call_id": "call_b", "content": "B"},
            {"role": "assistant", "content": "done"},
        ]

        result = instance._sanitize_tool_pairs([dict(m) for m in messages])

        assert result == messages
        self._assert_provider_tool_sequence_valid(result)

    def test_sanitize_tool_pairs_replaces_out_of_order_parallel_results_with_stubs(self, tmp_path):
        instance = self._make_engine(tmp_path, "lcm_parallel_out_of_order.db")
        messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "call_a", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "call_b", "function": {"name": "terminal", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_b", "content": "B out of order"},
            {"role": "tool", "tool_call_id": "call_a", "content": "A out of order"},
        ]

        result = instance._sanitize_tool_pairs([dict(m) for m in messages])

        self._assert_provider_tool_sequence_valid(result)
        assert [msg.get("tool_call_id") for msg in result[1:3]] == ["call_a", "call_b"]
        assert result[1]["content"] == "A out of order"
        assert "earlier conversation" in result[2]["content"]
        assert all(msg.get("content") != "B out of order" for msg in result)
