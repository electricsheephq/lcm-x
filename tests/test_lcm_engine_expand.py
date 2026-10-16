"""LCM engine integration tests: expand."""

import json
import time

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.tools as lcm_tools

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens, count_tokens

from .engine_helpers import (
    engine as engine,
)


class TestEngineTools:

    def test_handle_describe_overview(self, engine):
        result = json.loads(engine.handle_tool_call("lcm_describe", {}))
        assert "session_id" in result
        assert "store_message_count" in result

    def test_handle_expand_paginates_message_sources_with_cursor_metadata(self, engine):
        store_ids = [
            engine._store.append(
                "test-session",
                {"role": "user", "content": f"raw page message {idx}"},
            )
            for idx in range(5)
        ]
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Paged raw source summary",
                token_count=10,
                source_token_count=50,
                source_ids=store_ids,
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {"node_id": node_id, "source_offset": 1, "source_limit": 2, "max_tokens": 1000},
            )
        )

        assert [item["store_id"] for item in result["expanded"]] == store_ids[1:3]
        assert [item["source_index"] for item in result["expanded"]] == [1, 2]
        assert result["pagination"] == {
            "source_offset": 1,
            "content_offset": 0,
            "source_limit": 2,
            "returned_sources": 2,
            "total_sources": 5,
            "next_source_offset": 3,
            "next_content_offset": 0,
            "has_more": True,
            "remaining_sources": 2,
        }

    def test_handle_expand_keeps_ingest_placeholder_ref_unsliced_under_tiny_budget(self, engine):
        data_uri = "data:image/png;base64," + ("QUJD" * 80)
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "see image " + data_uri + " please inspect"},
        )
        stored = engine._store.get_session_messages("test-session")[-1]
        assert "see image [Externalized LCM ingest payload:" in stored["content"]
        assert stored["content"].endswith(" please inspect")
        assert "ref=" in stored["content"]
        assert data_uri not in stored["content"]
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Ingest marker recovery handle summary",
                token_count=10,
                source_token_count=10,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id, "max_tokens": 1}))

        item = result["expanded"][0]
        assert item["store_id"] == store_id
        assert item["content"] == stored["content"]
        assert item["content_truncated"] is False
        assert item["next_content_offset"] == 0
        assert "ref=" in item["content"]
        assert result["pagination"]["has_more"] is False

    def test_handle_expand_paginates_long_text_with_embedded_ingest_placeholder(self, engine):
        from hermes_lcm.tokens import count_tokens

        data_uri = "data:image/png;base64," + ("QUJD" * 80)
        content = ("intro " * 140) + data_uri + (" outro" * 140)
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": content},
        )
        stored = engine._store.get_session_messages("test-session")[-1]
        assert "[Externalized LCM ingest payload:" in stored["content"]
        assert "ref=" in stored["content"]
        assert data_uri not in stored["content"]
        assert len(stored["content"]) > 512
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Long embedded ingest marker summary",
                token_count=10,
                source_token_count=count_tokens(stored["content"]),
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        first = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id, "max_tokens": 20}))

        item = first["expanded"][0]
        assert item["store_id"] == store_id
        assert item["content"] != stored["content"]
        assert item["content_truncated"] is True
        assert item["next_content_offset"] > 0
        assert count_tokens(item["content"]) <= 20
        assert first["pagination"]["has_more"] is True
        assert first["pagination"]["next_source_offset"] == 0
        assert first["pagination"]["next_content_offset"] == item["next_content_offset"]

    def test_handle_expand_paginates_oversized_message_content_without_losing_raw_tail(self, engine):
        from hermes_lcm.tokens import count_tokens

        content = "alpha " * 400
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": content},
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Oversized raw message summary",
                token_count=10,
                source_token_count=count_tokens(content),
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        first = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {"node_id": node_id, "max_tokens": 20},
            )
        )

        first_item = first["expanded"][0]
        assert first_item["store_id"] == store_id
        assert first_item["content_offset"] == 0
        assert first_item["content_truncated"] is True
        assert count_tokens(first_item["content"]) <= 20
        assert first["pagination"]["has_more"] is True
        assert first["pagination"]["next_source_offset"] == 0
        assert first["pagination"]["next_content_offset"] > 0

        second = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {
                    "node_id": node_id,
                    "source_offset": first["pagination"]["next_source_offset"],
                    "content_offset": first["pagination"]["next_content_offset"],
                    "max_tokens": 20,
                },
            )
        )

        assert second["expanded"][0]["content_offset"] == first["pagination"]["next_content_offset"]
        assert second["expanded"][0]["content"] == content[first["pagination"]["next_content_offset"]:][:len(second["expanded"][0]["content"])]

    def test_handle_expand_advances_content_cursor_when_budget_cannot_fit_character(self, engine, monkeypatch):
        import hermes_lcm.tokens as token_utils

        def fake_count_tokens(text):
            return 0 if not text else len(text) + 1

        monkeypatch.setattr(token_utils, "count_tokens", fake_count_tokens)
        content = "abcdef"
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": content},
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Tiny budget raw message summary",
                token_count=10,
                source_token_count=10,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        first = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id, "max_tokens": 1}))

        assert first["expanded"][0]["content"] == "a"
        assert first["expanded"][0]["content_offset"] == 0
        assert first["expanded"][0]["next_content_offset"] == 1
        assert first["pagination"]["has_more"] is True
        assert first["pagination"]["next_source_offset"] == 0
        assert first["pagination"]["next_content_offset"] == 1

        second = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {
                    "node_id": node_id,
                    "source_offset": first["pagination"]["next_source_offset"],
                    "content_offset": first["pagination"]["next_content_offset"],
                    "max_tokens": 1,
                },
            )
        )

        assert second["expanded"][0]["content"] == "b"
        assert second["expanded"][0]["content_offset"] == 1
        assert second["pagination"]["next_content_offset"] == 2

    def test_expand_query_routes_configured_reasoning_effort_to_synthesis(self, engine, monkeypatch):
        engine._config.expansion_reasoning_effort = "low"
        engine._store.append(
            "test-session", {"role": "user", "content": "reasoning expansion evidence"}
        )
        captured = {}

        def fake_synthesize(**kwargs):
            captured.update(kwargs)
            return "expanded answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        result = json.loads(engine.handle_tool_call("lcm_expand_query", {
            "prompt": "What is the evidence?",
            "query": "reasoning expansion evidence",
        }))

        assert result["answer"] == "expanded answer"
        assert captured["reasoning_effort"] == "low"

    def test_handle_expand_query_recursively_descends_parent_nodes_to_leaf_messages(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "recursive answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "LEAF RAW SECRET zeta detail"},
        )
        leaf_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="leaf summary mentions broad topic only",
                token_count=10,
                source_token_count=10,
                source_ids=[store_id],
                source_type="messages",
                created_at=1,
            )
        )
        middle_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=1,
                summary="middle summary points at leaf",
                token_count=10,
                source_token_count=20,
                source_ids=[leaf_id],
                source_type="nodes",
                created_at=2,
            )
        )
        parent_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=2,
                summary="parent summary points at middle",
                token_count=10,
                source_token_count=30,
                source_ids=[middle_id],
                source_type="nodes",
                created_at=3,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What exact leaf detail is present?",
                    "node_ids": [parent_id],
                    "max_tokens": 20,
                    "context_max_tokens": 1000,
                },
            )
        )

        assert result["answer"] == "recursive answer"
        serialized_context = json.dumps(captured["context_blocks"])
        assert "LEAF RAW SECRET zeta detail" in serialized_context
        assert any(block.get("type") == "child_messages" for block in captured["context_blocks"])

    def test_handle_expand_query_deep_parent_reaches_leaf_messages(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "deep recursive answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "DEEP LEAF RAW SECRET omega detail"},
        )
        child_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="leaf summary",
                token_count=5,
                source_token_count=5,
                source_ids=[store_id],
                source_type="messages",
                created_at=1,
            )
        )
        for depth in range(1, 7):
            child_id = engine._dag.add_node(
                SummaryNode(
                    session_id="test-session",
                    depth=depth,
                    summary=f"depth {depth} summary",
                    token_count=5,
                    source_token_count=5 * (depth + 1),
                    source_ids=[child_id],
                    source_type="nodes",
                    created_at=depth + 1,
                )
            )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What exact deep leaf detail is present?",
                    "node_ids": [child_id],
                    "max_tokens": 20,
                    "context_max_tokens": 1000,
                },
            )
        )

        serialized_context = json.dumps(captured["context_blocks"])
        assert result["answer"] == "deep recursive answer"
        assert "DEEP LEAF RAW SECRET omega detail" in serialized_context
        assert sum(block.get("type") == "descendant_child_nodes" for block in captured["context_blocks"]) >= 5
        assert any(block.get("type") == "child_messages" for block in captured["context_blocks"])

    def test_expand_query_descendant_collection_handles_zero_token_deep_chain_without_recursion(self, engine):
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "ZERO TOKEN DEEP LEAF evidence"},
        )
        child_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="",
                token_count=0,
                source_token_count=0,
                source_ids=[store_id],
                source_type="messages",
                created_at=1,
            )
        )
        for depth in range(1, 1105):
            child_id = engine._dag.add_node(
                SummaryNode(
                    session_id="test-session",
                    depth=depth,
                    summary="",
                    token_count=0,
                    source_token_count=0,
                    source_ids=[child_id],
                    source_type="nodes",
                    created_at=depth + 1,
                )
            )

        root = engine._dag.get_node(child_id)
        blocks = lcm_tools._collect_context_blocks_for_node(engine, root, max_tokens=32000)

        serialized_context = json.dumps(blocks)
        assert "ZERO TOKEN DEEP LEAF evidence" not in serialized_context
        assert len(blocks) < 1105
        assert lcm_tools._context_content_token_count(blocks) <= 33000
        path_blocks = [block for block in blocks if "source_path" in block]
        assert path_blocks
        assert max(len(block["source_path"]) for block in path_blocks) <= 8
        assert any(block.get("source_path_truncated") is True for block in path_blocks)
        assert any(block.get("source_path_depth", 0) > len(block["source_path"]) for block in path_blocks)

    def test_handle_expand_query_uses_raw_hits_when_summary_search_misses(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "raw bridge answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "PHOENIXRAWONLY appears only in the raw message"},
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="summary deliberately omits the distinctive raw identifier",
                token_count=10,
                source_token_count=10,
                source_ids=[store_id],
                source_type="messages",
                created_at=1,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What mentions PHOENIX?",
                    "query": "PHOENIXRAWONLY",
                    "max_tokens": 20,
                    "context_max_tokens": 1000,
                },
            )
        )

        assert result["answer"] == "raw bridge answer"
        assert result["raw_matches"]
        serialized_context = json.dumps(captured["context_blocks"])
        assert "PHOENIXRAWONLY appears only in the raw message" in serialized_context

    def test_handle_expand_query_keeps_raw_snippets_out_of_synthesis_context(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "raw snippet answer"

        def fake_store_search(query, session_id=None, limit=5):
            return [
                {
                    "store_id": 123,
                    "session_id": session_id or "test-session",
                    "source": "telegram",
                    "role": "user",
                    "timestamp": 1,
                    "content": "A",
                    "snippet": "UNBUDGETED RAW SNIPPET LEAK",
                    "search_rank": 1,
                }
            ]

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        monkeypatch.setattr(engine._store, "search", fake_store_search)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What raw detail?",
                    "query": "anything",
                    "max_tokens": 20,
                    "context_max_tokens": 1,
                },
            )
        )

        serialized_context = json.dumps(captured["context_blocks"])
        assert result["answer"] == "raw snippet answer"
        assert result["raw_matches"][0]["snippet"] == "UNBUDGETED RAW SNIPPET LEAK"
        assert "UNBUDGETED RAW SNIPPET LEAK" not in serialized_context
        raw_block = next(block for block in captured["context_blocks"] if block["type"] == "raw_messages")
        assert "snippet" not in raw_block["messages"][0]

    def test_handle_expand_query_deduped_raw_hit_does_not_leak_snippet(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "deduped raw answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "DEDUPEDRAW message evidence"},
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="summary points at deduped raw evidence",
                token_count=10,
                source_token_count=10,
                source_ids=[store_id],
                source_type="messages",
                created_at=1,
            )
        )

        def fake_store_search(query, session_id=None, limit=5):
            return [
                {
                    "store_id": store_id,
                    "session_id": session_id or "test-session",
                    "source": "telegram",
                    "role": "user",
                    "timestamp": 1,
                    "content": "DEDUPEDRAW message evidence",
                    "snippet": "DEDUPED RAW SNIPPET SHOULD NOT LEAK",
                    "search_rank": 1,
                }
            ]

        monkeypatch.setattr(engine._store, "search", fake_store_search)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What raw detail?",
                    "node_ids": [node_id],
                    "query": "DEDUPEDRAW",
                    "max_tokens": 20,
                    "context_max_tokens": 1000,
                },
            )
        )

        serialized_context = json.dumps(captured["context_blocks"])
        assert result["answer"] == "deduped raw answer"
        assert "DEDUPEDRAW message evidence" in serialized_context
        assert "DEDUPED RAW SNIPPET SHOULD NOT LEAK" not in serialized_context
        assert not any(block.get("type") == "raw_messages" for block in captured["context_blocks"])
        assert result["raw_matches"] == []

    def test_handle_expand_query_keeps_raw_tool_calls_out_of_synthesis_context(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "raw tool call answer"

        def fake_store_search(query, session_id=None, limit=5):
            return [
                {
                    "store_id": 456,
                    "session_id": session_id or "test-session",
                    "source": "telegram",
                    "role": "assistant",
                    "timestamp": 1,
                    "content": "A",
                    "tool_calls": [{"function": {"arguments": "UNBUDGETED TOOL ARGUMENT LEAK" * 20}}],
                    "tool_call_id": "call_123",
                    "tool_name": "expensive_tool",
                    "search_rank": 1,
                }
            ]

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        monkeypatch.setattr(engine._store, "search", fake_store_search)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What raw detail?",
                    "query": "anything",
                    "max_tokens": 20,
                    "context_max_tokens": 1,
                },
            )
        )

        serialized_context = json.dumps(captured["context_blocks"])
        raw_block = next(block for block in captured["context_blocks"] if block["type"] == "raw_messages")
        raw_item = raw_block["messages"][0]
        assert result["answer"] == "raw tool call answer"
        assert "UNBUDGETED TOOL ARGUMENT LEAK" not in serialized_context
        assert "tool_calls" not in raw_item
        assert raw_item["tool_calls_omitted"] is True
        assert raw_item["tool_call_id"] == "call_123"
        assert raw_item["tool_name"] == "expensive_tool"

    def test_handle_expand_query_raw_hit_context_windows_around_match(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "raw tail answer"

        content = "prefix-noise " * 80 + "TAILMATCH exact evidence"

        def fake_store_search(query, session_id=None, limit=5):
            return [
                {
                    "store_id": 789,
                    "session_id": session_id or "test-session",
                    "source": "telegram",
                    "role": "user",
                    "timestamp": 1,
                    "content": content,
                    "snippet": "TAILMATCH exact evidence",
                    "search_rank": 1,
                }
            ]

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        monkeypatch.setattr(engine._store, "search", fake_store_search)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What raw detail?",
                    "query": "TAILMATCH",
                    "max_tokens": 20,
                    "context_max_tokens": 6,
                },
            )
        )

        raw_block = next(block for block in captured["context_blocks"] if block["type"] == "raw_messages")
        raw_item = raw_block["messages"][0]
        assert result["answer"] == "raw tail answer"
        assert "TAILMATCH" in raw_item["content"]
        assert raw_item["content_offset"] == content.index("TAILMATCH")
        assert raw_item["match_window_offset"] == content.index("TAILMATCH")
        assert "prefix-noise" not in raw_item["content"]

    def test_handle_expand_query_raw_hit_match_window_uses_sanitized_query_terms(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "raw sanitized answer"

        content = "match unrelated prefix-noise " * 40 + "tail match exact evidence"

        def fake_store_search(query, session_id=None, limit=5):
            return [
                {
                    "store_id": 790,
                    "session_id": session_id or "test-session",
                    "source": "telegram",
                    "role": "user",
                    "timestamp": 1,
                    "content": content,
                    "snippet": "tail match exact evidence",
                    "search_rank": 1,
                }
            ]

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        monkeypatch.setattr(engine._store, "search", fake_store_search)

        for query in ["tail-match", "tail.match", "tail/match", "tail:match", "tail(match)"]:
            captured.clear()
            result = json.loads(
                engine.handle_tool_call(
                    "lcm_expand_query",
                    {
                        "prompt": "What raw detail?",
                        "query": query,
                        "max_tokens": 20,
                        "context_max_tokens": 6,
                    },
                )
            )

            raw_block = next(block for block in captured["context_blocks"] if block["type"] == "raw_messages")
            raw_item = raw_block["messages"][0]
            assert result["answer"] == "raw sanitized answer"
            assert "tail match" in raw_item["content"]
            assert raw_item["content_offset"] == content.index("tail match")
            assert raw_item["match_window_offset"] == content.index("tail match")
            assert "match unrelated" not in raw_item["content"]

    def test_handle_expand_query_raw_hit_truncation_returns_store_expand_cursor(self, engine, monkeypatch):
        import hermes_lcm.tokens as token_utils

        def fake_count_tokens(text):
            return 0 if not text else len(text) + 1

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            return "raw cursor answer"

        monkeypatch.setattr(token_utils, "count_tokens", fake_count_tokens)
        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "PHOENIXRAWCURSOR has a longer raw detail"},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What mentions PHOENIX?",
                    "query": "PHOENIXRAWCURSOR",
                    "max_tokens": 20,
                    "context_max_tokens": 1,
                },
            )
        )

        raw_page = next(item for item in result["context_pagination"] if item["type"] == "raw_messages")
        assert raw_page["expand_args"] == {"store_id": store_id, "content_offset": 1}

    def test_handle_expand_query_advances_content_cursor_when_context_budget_cannot_fit_character(self, engine, monkeypatch):
        import hermes_lcm.tokens as token_utils

        captured = {}

        def fake_count_tokens(text):
            return 0 if not text else len(text) + 1

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "bounded answer"

        monkeypatch.setattr(token_utils, "count_tokens", fake_count_tokens)
        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "abcdef"},
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="",
                token_count=0,
                source_token_count=10,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What raw detail?",
                    "node_ids": [node_id],
                    "max_tokens": 5,
                    "context_max_tokens": 1,
                },
            )
        )

        message_block = next(block for block in captured["context_blocks"] if block["type"] == "messages")
        assert message_block["messages"][0]["content"] == "a"
        assert message_block["messages"][0]["next_content_offset"] == 1
        assert result["context_truncated"] is True
        assert any(
            item["node_id"] == node_id
            and item.get("pagination", {}).get("next_content_offset") == 1
            for item in result["context_pagination"]
        )

    def test_handle_expand_paginates_child_node_sources(self, engine):
        child_ids = []
        for idx in range(3):
            store_id = engine._store.append(
                "test-session",
                {"role": "user", "content": f"child node raw {idx}"},
            )
            child_ids.append(
                engine._dag.add_node(
                    SummaryNode(
                        session_id="test-session",
                        depth=0,
                        summary=f"child summary {idx}",
                        token_count=10,
                        source_token_count=10,
                        source_ids=[store_id],
                        source_type="messages",
                        created_at=idx,
                    )
                )
            )
        parent_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=1,
                summary="parent summary",
                token_count=10,
                source_token_count=30,
                source_ids=child_ids,
                source_type="nodes",
                created_at=0,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {"node_id": parent_id, "source_offset": 1, "source_limit": 1},
            )
        )

        assert result["source_type"] == "nodes"
        assert [item["node_id"] for item in result["expanded"]] == [child_ids[1]]
        assert result["expanded"][0]["source_index"] == 1
        assert result["pagination"]["has_more"] is True
        assert result["pagination"]["next_source_offset"] == 2

    def test_handle_expand_child_node_pagination_preserves_source_id_order(self, engine):
        newer_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="newer child in source order first",
                token_count=10,
                source_token_count=10,
                source_ids=[],
                source_type="messages",
                created_at=2,
            )
        )
        older_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="older child in source order second",
                token_count=10,
                source_token_count=10,
                source_ids=[],
                source_type="messages",
                created_at=1,
            )
        )
        parent_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=1,
                summary="parent summary",
                token_count=10,
                source_token_count=20,
                source_ids=[newer_id, older_id],
                source_type="nodes",
                created_at=3,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {"node_id": parent_id, "source_offset": 0, "source_limit": 1},
            )
        )

        assert [item["node_id"] for item in result["expanded"]] == [newer_id]
        assert result["expanded"][0]["source_index"] == 0

    def test_handle_expand_child_node_sources_respect_max_tokens(self, engine):
        child_ids = []
        for idx in range(2):
            child_ids.append(
                engine._dag.add_node(
                    SummaryNode(
                        session_id="test-session",
                        depth=0,
                        summary=(f"child {idx} " * 80),
                        token_count=160,
                        source_token_count=160,
                        source_ids=[],
                        source_type="messages",
                        created_at=idx,
                    )
                )
            )
        parent_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=1,
                summary="parent summary",
                token_count=10,
                source_token_count=320,
                source_ids=child_ids,
                source_type="nodes",
                created_at=3,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": parent_id, "max_tokens": 5}))

        assert len(result["expanded"]) == 1
        assert result["expanded"][0]["summary_truncated"] is True
        assert result["pagination"]["has_more"] is True
        assert result["pagination"]["next_source_offset"] == 1

    def test_handle_expand_includes_externalized_metadata_for_large_tool_result_sources(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_expand.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        store_id = engine._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_big", "content": content},
        )
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Externalized tool-output summary",
                token_count=10,
                source_token_count=20,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))

        assert result["source_type"] == "messages"
        assert result["expanded"][0]["store_id"] == store_id
        assert result["expanded"][0]["externalized"]["tool_call_id"] == "call_big"
        assert result["expanded"][0]["externalized"]["content_chars"] == len(content)
        assert result["expanded"][0]["externalized"]["ref"].endswith(".json")

    def test_handle_expand_does_not_attach_other_sessions_externalized_metadata(self, tmp_path):
        shared_home = tmp_path / "hermes"
        config_a = LCMConfig(
            database_path=str(tmp_path / "a.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine_a = LCMEngine(config=config_a, hermes_home=str(shared_home))
        engine_a._session_id = "session-a"

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine_a._serialize_messages([
            {"role": "tool", "tool_call_id": "call_shared", "content": content}
        ])

        config_b = LCMConfig(
            database_path=str(tmp_path / "b.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine_b = LCMEngine(config=config_b, hermes_home=str(shared_home))
        engine_b._session_id = "session-b"
        cur = engine_b._store._conn.execute(
            """INSERT INTO messages
               (session_id, source, role, content, tool_call_id, timestamp, token_estimate, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("session-b", "", "tool", content, "call_shared", 0, 0, 0),
        )
        engine_b._store._conn.commit()
        store_id = int(cur.lastrowid)
        node_id = engine_b._dag.add_node(
            SummaryNode(
                session_id="session-b",
                depth=0,
                summary="Other session summary",
                token_count=10,
                source_token_count=20,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine_b.handle_tool_call("lcm_expand", {"node_id": node_id}))

        assert "externalized" not in result["expanded"][0]

    def test_handle_expand_finds_externalized_metadata_for_sanitized_tool_output(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_sanitized.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        raw_content = (("Chart screenshot notes. " * 80) + "\n\n" + "data:image/png;base64," + ("A" * 5000))
        store_id = engine._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_media", "content": raw_content},
        )
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_media", "content": raw_content}
        ])
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Sanitized externalized tool-output summary",
                token_count=10,
                source_token_count=20,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))

        assert result["expanded"][0]["externalized"]["tool_call_id"] == "call_media"
        assert result["expanded"][0]["externalized"]["session_id"] == "test-session"

    def test_handle_describe_externalized_ref_returns_metadata(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_describe.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name

        result = json.loads(engine.handle_tool_call("lcm_describe", {"externalized_ref": ref}))

        assert result["externalized_ref"] == ref
        assert result["kind"] == "tool_result"
        assert result["tool_call_id"] == "call_big"
        assert result["session_id"] == "test-session"
        assert result["content_chars"] == len(content)
        assert result["content_preview"].startswith("RESULT:")

    def test_handle_expand_externalized_ref_returns_payload_content(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_payload.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name

        result = json.loads(engine.handle_tool_call("lcm_expand", {"externalized_ref": ref}))

        assert result["externalized_ref"] == ref
        assert result["source_type"] == "externalized_payload"
        assert result["content"] == content
        assert result["tool_call_id"] == "call_big"
        assert result["content_truncated"] is False

    def test_handle_expand_externalized_ref_respects_max_tokens(self, tmp_path):
        from hermes_lcm.tokens import count_tokens

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_payload_budget.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name

        result = json.loads(engine.handle_tool_call("lcm_expand", {"externalized_ref": ref, "max_tokens": 10}))

        assert result["externalized_ref"] == ref
        assert result["source_type"] == "externalized_payload"
        assert result["content_truncated"] is True
        assert count_tokens(result["content"]) <= 10
        assert result["tool_call_id"] == "call_big"

    def test_handle_expand_externalized_ref_uses_content_offset_cursor(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_payload_cursor.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name

        first = json.loads(engine.handle_tool_call("lcm_expand", {"externalized_ref": ref, "max_tokens": 10}))
        assert first["has_more"] is True
        assert first["next_content_offset"] > 0

        second = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {
                    "externalized_ref": ref,
                    "content_offset": first["next_content_offset"],
                    "max_tokens": 10,
                },
            )
        )

        assert second["content_offset"] == first["next_content_offset"]
        assert second["content"] == content[first["next_content_offset"]:][:len(second["content"])]

    def test_handle_expand_externalized_ref_advances_content_cursor_when_budget_cannot_fit_character(self, tmp_path, monkeypatch):
        import hermes_lcm.tokens as token_utils

        def fake_count_tokens(text):
            return 0 if not text else len(text) + 1

        monkeypatch.setattr(token_utils, "count_tokens", fake_count_tokens)
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_externalized_payload_tiny_cursor.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=2,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "abcdef"
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_tiny", "content": content}
        ])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name

        first = json.loads(engine.handle_tool_call("lcm_expand", {"externalized_ref": ref, "max_tokens": 1}))

        assert first["content"] == "a"
        assert first["content_offset"] == 0
        assert first["next_content_offset"] == 1
        assert first["has_more"] is True

        second = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {
                    "externalized_ref": ref,
                    "content_offset": first["next_content_offset"],
                    "max_tokens": 1,
                },
            )
        )

        assert second["content"] == "b"
        assert second["content_offset"] == 1
        assert second["next_content_offset"] == 2

    def test_handle_expand_query_uses_independent_context_budget_for_auxiliary_retrieval(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            captured["max_tokens"] = max_tokens
            return "bounded answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        first_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "first filler " * 80},
        )
        second_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "SECOND RAW DETAIL survives auxiliary context expansion"},
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Summary mentioning second raw detail",
                token_count=10,
                source_token_count=200,
                source_ids=[first_store_id, second_store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What detail survived?",
                    "node_ids": [node_id],
                    "max_tokens": 5,
                    "context_max_tokens": 500,
                },
            )
        )

        assert result["answer"] == "bounded answer"
        assert captured["max_tokens"] == 5
        context_json = json.dumps(captured["context_blocks"])
        assert "SECOND RAW DETAIL" in context_json

    def test_handle_expand_query_applies_context_budget_globally_across_nodes(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "bounded answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        first_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "first filler " * 80},
        )
        first_node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="first summary",
                token_count=10,
                source_token_count=200,
                source_ids=[first_store_id],
                source_type="messages",
                created_at=2,
            )
        )
        second_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "SECOND NODE RAW DETAIL should require another page"},
        )
        second_node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="second summary without raw detail",
                token_count=10,
                source_token_count=20,
                source_ids=[second_store_id],
                source_type="messages",
                created_at=1,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What raw details exist?",
                    "node_ids": [first_node_id, second_node_id],
                    "max_tokens": 5,
                    "context_max_tokens": 1,
                },
            )
        )

        context_json = json.dumps(captured["context_blocks"])
        assert "SECOND NODE RAW DETAIL" not in context_json
        assert result["context_truncated"] is True
        assert any(
            item["node_id"] == second_node_id and item.get("pagination", {}).get("has_more")
            for item in result["context_pagination"]
        )

    def test_handle_expand_query_counts_summary_blocks_against_context_budget(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "bounded answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "raw detail should wait behind summary budget"},
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary=("long summary filler " * 80) + "UNBUDGETED SUMMARY TAIL",
                token_count=200,
                source_token_count=20,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What fits?",
                    "node_ids": [node_id],
                    "max_tokens": 5,
                    "context_max_tokens": 5,
                },
            )
        )

        context_json = json.dumps(captured["context_blocks"])
        assert "UNBUDGETED SUMMARY TAIL" not in context_json
        assert "raw detail should wait" not in context_json
        assert captured["context_blocks"][0]["summary_truncated"] is True
        assert result["context_truncated"] is True
        assert any(
            item["node_id"] == node_id and item["type"] == "summary"
            for item in result["context_pagination"]
        )

    def test_handle_expand_query_reports_last_child_summary_truncation_in_context_pagination(self, engine, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "bounded answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        child_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary=("child summary detail " * 80) + "CHILD SUMMARY TAIL",
                token_count=200,
                source_token_count=200,
                source_ids=[],
                source_type="messages",
                created_at=0,
            )
        )
        parent_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=1,
                summary="",
                token_count=0,
                source_token_count=200,
                source_ids=[child_id],
                source_type="nodes",
                created_at=1,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What child details fit?",
                    "node_ids": [parent_id],
                    "max_tokens": 5,
                    "context_max_tokens": 5,
                },
            )
        )

        context_json = json.dumps(captured["context_blocks"])
        assert "CHILD SUMMARY TAIL" not in context_json
        child_block = next(block for block in captured["context_blocks"] if block["type"] == "child_nodes")
        assert child_block["children"][0]["summary_truncated"] is True
        assert result["context_truncated"] is True
        assert any(
            item["type"] == "child_summary"
            and item["node_id"] == parent_id
            and item["child_node_id"] == child_id
            and item["summary_truncated"] is True
            and item["expand_args"] == {"node_id": child_id}
            for item in result["context_pagination"]
        )

    def test_handle_expand_query_externalized_truncation_returns_ref_in_context_pagination(
        self, tmp_path, monkeypatch
    ):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "bounded answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_expand_query_externalized_truncated.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        content = "EXTERNALIZED RAW DETAIL " + ("abcdef" * 1000)
        engine._serialize_messages(
            [{"role": "tool", "tool_call_id": "call_ext", "content": content}]
        )
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name
        placeholder = f"[GC'd externalized tool output: tool_call_id=call_ext; chars={len(content)}; ref={ref}]"
        store_id = engine._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_ext", "content": placeholder},
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="externalized payload summary",
                token_count=10,
                source_token_count=200,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What externalized detail exists?",
                    "node_ids": [node_id],
                    "max_tokens": 5,
                    "context_max_tokens": 20,
                },
            )
        )

        message_block = next(
            block for block in captured["context_blocks"] if block["type"] == "messages"
        )
        assert message_block["messages"][0]["content_source"] == "externalized_payload"
        assert message_block["messages"][0]["content_truncated"] is True
        assert result["context_truncated"] is True
        assert any(
            item["type"] == "messages"
            and item["node_id"] == node_id
            and item["content_source"] == "externalized_payload"
            and item["externalized_ref"] == ref
            and item["pagination"]["has_more"] is True
            and item["expand_args"]
            == {
                "externalized_ref": ref,
                "content_offset": item["pagination"]["next_content_offset"],
            }
            for item in result["context_pagination"]
        )

        captured.clear()
        engine._session_id = "new-session"
        cross_session_result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What externalized detail exists?",
                    "node_ids": [node_id],
                    "session_ids": ["test-session"],
                    "max_tokens": 5,
                    "context_max_tokens": 20,
                },
            )
        )
        cross_session_block = next(
            block for block in captured["context_blocks"] if block["type"] == "messages"
        )
        assert cross_session_block["messages"][0]["content_source"] != (
            "externalized_payload"
        )
        assert all(
            "externalized_ref" not in item.get("expand_args", {})
            for item in cross_session_result["context_pagination"]
        )

    def test_handle_expand_query_counts_externalized_transcript_content_against_context_budget(self, tmp_path, monkeypatch):
        import hermes_lcm.tokens as token_utils

        captured = {}

        def fake_count_tokens(text):
            return len(str(text or ""))

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "bounded answer"

        monkeypatch.setattr(token_utils, "count_tokens", fake_count_tokens)
        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_expand_query_externalized_transcript_budget.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=2,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        content = "PAYLOAD"
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_ext", "content": content}
        ])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name
        transcript_content = (
            f"[GC'd externalized tool output: tool_call_id=call_ext; chars={len(content)}; ref={ref}]"
            + (" transcript filler" * 20)
        )
        first_store_id = engine._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_ext", "content": transcript_content},
        )
        first_node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="",
                token_count=0,
                source_token_count=200,
                source_ids=[first_store_id],
                source_type="messages",
                created_at=1,
            )
        )
        second_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "SECOND_RAW_DETAIL"},
        )
        second_node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="",
                token_count=0,
                source_token_count=20,
                source_ids=[second_store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What details fit?",
                    "node_ids": [first_node_id, second_node_id],
                    "max_tokens": 5,
                    "context_max_tokens": 30,
                },
            )
        )

        context_json = json.dumps(captured["context_blocks"])
        assert "transcript_content" in context_json
        assert "SECOND_RAW_DETAIL" not in context_json
        assert result["context_truncated"] is True
        assert any(
            item["node_id"] == second_node_id
            and item["type"] == "messages"
            and item.get("pagination", {}).get("has_more") is True
            and item.get("expand_args") == {"node_id": second_node_id, "source_offset": 0, "content_offset": 0}
            for item in result["context_pagination"]
        )

    def test_handle_expand_query_hydrates_externalized_payload_content_for_auxiliary_context(self, tmp_path, monkeypatch):
        captured = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            captured["context_blocks"] = context_blocks
            return "bounded answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_expand_query_externalized.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        content = "EXTERNALIZED RAW DETAIL survives for auxiliary retrieval " + ("abcdef" * 100)
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_ext", "content": content}
        ])
        ref = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json")).name
        placeholder = f"[GC'd externalized tool output: tool_call_id=call_ext; chars={len(content)}; ref={ref}]"
        store_id = engine._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_ext", "content": placeholder},
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="externalized payload summary",
                token_count=10,
                source_token_count=200,
                source_ids=[store_id],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What externalized detail exists?",
                    "node_ids": [node_id],
                    "max_tokens": 5,
                    "context_max_tokens": 500,
                },
            )
        )

        context_json = json.dumps(captured["context_blocks"])
        assert "EXTERNALIZED RAW DETAIL" in context_json
        assert "externalized_payload" in context_json
        assert result["context_truncated"] is False

    def test_compress_gc_rewrites_summarized_externalized_tool_results(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc.db"),
            fresh_tail_count=0,
            leaf_chunk_tokens=50,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("Summarized tool result.\nExpand for details about: tool result", 1),
        )

        content = "RESULT:\n" + ("abcdef" * 2000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_gc", "content": content},
        ]

        compressed = engine.compress(messages)

        assert compressed[0]["role"] == "system"
        # The summary follows the system prompt, so it must be role "user" —
        # an assistant summary as the first non-system message is rejected by
        # Anthropic (it extracts the system prompt and requires messages[0]
        # to be "user").
        assert compressed[1]["role"] == "user"
        assert "Recent Summary" in compressed[1]["content"]
        stored_tool = next(row for row in engine._store.get_range("test-session") if row["role"] == "tool")
        assert stored_tool["content"].startswith("[GC'd externalized tool output:")
        assert "ref=" in stored_tool["content"]
        assert content[:100] not in stored_tool["content"]
        assert stored_tool["token_estimate"] == count_message_tokens(
            {"role": "tool", "tool_call_id": "call_gc", "content": stored_tool["content"]}
        )
        assert stored_tool["token_estimate"] < count_message_tokens(
            {"role": "tool", "tool_call_id": "call_gc", "content": content}
        )
        payload_files = list((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json"))
        assert len(payload_files) == 1
        assert json.loads(payload_files[0].read_text())["content"] == content

    def test_handle_expand_still_resolves_externalized_metadata_after_transcript_gc_rewrite(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_expand.db"),
            fresh_tail_count=0,
            leaf_chunk_tokens=50,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("Summarized tool result.\nExpand for details about: tool result", 1),
        )

        content = "RESULT:\n" + ("abcdef" * 2000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_gc", "content": content},
        ]

        engine.compress(messages)
        node_id = engine._dag.get_session_nodes("test-session")[0].node_id
        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))

        assert result["expanded"][0]["content"].startswith("[GC'd externalized tool output:")
        assert result["expanded"][0]["externalized"]["tool_call_id"] == "call_gc"
        assert result["expanded"][0]["externalized"]["ref"].endswith(".json")

    def test_compress_gc_skips_pinned_tool_results(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_pinned.db"),
            fresh_tail_count=0,
            leaf_chunk_tokens=50,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("Summarized tool result.\nExpand for details about: tool result", 1),
        )

        content = "RESULT:\n" + ("abcdef" * 2000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_gc", "content": content},
        ]
        engine._ingest_messages(messages)
        tool_store_id = next(row for row in engine._store.get_range("test-session") if row["role"] == "tool")["store_id"]
        engine._store.pin(tool_store_id)

        engine.compress(messages)

        pinned_content = engine._store.get(tool_store_id)["content"]
        assert pinned_content.startswith("[Externalized tool output:")
        assert not pinned_content.startswith("[GC'd externalized tool output:")
        assert content[:100] not in pinned_content

    def test_gc_helper_does_not_miss_tool_rows_when_chunk_contains_unmatched_synthetic_messages(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_helper.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        content = "RESULT:\n" + ("abcdef" * 2000)
        tool_store_id = engine._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_gc", "content": content},
        )
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_gc", "content": content}
        ])

        engine._maybe_gc_compacted_tool_results(
            [
                {"role": "assistant", "content": "[Recent Summary (d0, node 1)]"},
                {"role": "tool", "tool_call_id": "call_gc", "content": content},
            ],
            [tool_store_id],
        )

        assert engine._store.get(tool_store_id)["content"].startswith("[GC'd externalized tool output:")

    def test_gc_does_not_tombstone_row_with_only_embedded_externalized_ref(self, tmp_path):
        # Regression: a tool row that merely QUOTES an externalized placeholder
        # inside surrounding text (e.g. a recall-tool result) must not have its
        # whole content tombstoned - that would discard the surrounding text,
        # which was never externalized and is unrecoverable.
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_embed.db"),
            large_output_externalization_enabled=True,
            # 1000 (was 200): the #680 stub is ~255 chars, so the quoting row must stay below the threshold.
            large_output_externalization_threshold_chars=1000,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"

        big = "RESULT:\n" + ("abcdef" * 2000)
        engine._ingest_messages([
            {"role": "tool", "tool_call_id": "call_ext", "content": big},
        ])
        placeholder = next(
            row for row in engine._store.get_range("test-session") if row["role"] == "tool"
        )["content"]
        assert placeholder.startswith("[Externalized tool output:")
        assert "ref=" in placeholder

        embedded = f"Here are the search results:\n{placeholder}\nEnd of results."
        embed_store_id = engine._store.append(
            "test-session",
            {"role": "tool", "tool_call_id": "call_recall", "content": embedded},
        )

        engine._maybe_gc_compacted_tool_results(
            [{"role": "tool", "tool_call_id": "call_recall", "content": embedded}],
            [embed_store_id],
        )

        preserved = engine._store.get(embed_store_id)["content"]
        assert preserved == embedded
        assert not preserved.startswith("[GC'd externalized")

    def test_handle_expand_does_not_inline_full_externalized_payload_for_gc_rows(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_budget.db"),
            fresh_tail_count=0,
            leaf_chunk_tokens=50,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine._session_id = "test-session"
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("Summarized tool result.\nExpand for details about: tool result", 1),
        )

        content = "RESULT:\n" + ("abcdef" * 2000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_gc", "content": content},
        ]
        engine.compress(messages)
        node_id = engine._dag.get_session_nodes("test-session")[0].node_id

        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id, "max_tokens": 1}))

        assert result["expanded"][0]["content"].startswith("[GC'd externalized tool output:")
        assert result["expanded"][0]["externalized"]["ref"].endswith(".json")
        assert "content" not in result["expanded"][0]["externalized"]

    def test_handle_unknown_tool(self, engine):
        result = json.loads(engine.handle_tool_call("unknown_tool", {}))
        assert "error" in result

    def test_tool_dispatch_is_bound_to_engine_instance(self, tmp_path):
        config_a = LCMConfig(database_path=str(tmp_path / "a.db"))
        config_b = LCMConfig(database_path=str(tmp_path / "b.db"))

        engine_a = LCMEngine(config=config_a)
        engine_a._session_id = "session-a"
        engine_b = LCMEngine(config=config_b)
        engine_b._session_id = "session-b"

        engine_a._store.append("session-a", {"role": "user", "content": "alpha project"})
        engine_b._store.append("session-b", {"role": "user", "content": "beta project"})

        result_a = json.loads(engine_a.handle_tool_call("lcm_grep", {"query": "alpha"}))
        result_b = json.loads(engine_b.handle_tool_call("lcm_grep", {"query": "beta"}))

        assert result_a["total_results"] == 1
        assert result_b["total_results"] == 1
        assert "alpha" in result_a["results"][0]["snippet"]
        assert "beta" in result_b["results"][0]["snippet"]

    def test_handle_expand_query_requires_prompt(self, engine):
        result = json.loads(engine.handle_tool_call("lcm_expand_query", {"query": "docker"}))
        assert "error" in result
        assert "prompt" in result["error"]

    def test_handle_expand_query_uses_expansion_model(self, engine, monkeypatch):
        engine._config.expansion_model = "expansion-model-x"
        engine._store.append("test-session", {"role": "user", "content": "Discussed docker rollout plan"})
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Docker rollout summary",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )

        seen = {}

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            seen["prompt"] = prompt
            seen["context_blocks"] = context_blocks
            seen["model"] = model
            seen["max_tokens"] = max_tokens
            seen["timeout"] = timeout
            return "Expansion answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"query": "docker", "prompt": "What was the plan?", "max_tokens": 500},
            )
        )

        assert result["answer"] == "Expansion answer"
        assert result["model"] == "expansion-model-x"
        assert result["node_ids"] == [node_id]
        assert seen["model"] == "expansion-model-x"
        assert seen["timeout"] == engine._config.expansion_timeout_ms / 1000
        assert seen["max_tokens"] == 500
        assert seen["prompt"] == "What was the plan?"
        assert seen["context_blocks"]

    def test_handle_expand_query_timeout_returns_explicit_degraded_error(self, engine, monkeypatch):
        engine._config.expansion_timeout_ms = 2500
        engine._store.append("test-session", {"role": "user", "content": "Discussed docker rollout plan"})
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Docker rollout summary",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            raise TimeoutError("expansion timed out")

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"query": "docker", "prompt": "What was the plan?"},
            )
        )

        assert result["degraded"] is True
        assert "timed out" in result["error"]
        assert result["timeout_seconds"] == 2.5
        assert result["node_ids"] == [node_id]
        assert result["matches"]
        assert "answer" not in result

    def test_handle_expand_query_unexpected_synthesis_error_is_not_degraded(self, engine, monkeypatch):
        engine._store.append("test-session", {"role": "user", "content": "Discussed docker rollout plan"})
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Docker rollout summary",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            raise RuntimeError("schema bug")

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)

        with pytest.raises(RuntimeError, match="schema bug"):
            engine.handle_tool_call(
                "lcm_expand_query",
                {"query": "docker", "prompt": "What was the plan?"},
            )

    def test_handle_expand_query_blank_synthesis_is_not_false_success(self, engine, monkeypatch):
        engine._store.append("test-session", {"role": "user", "content": "Discussed docker rollout plan"})
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Docker rollout summary",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )
        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "   ")

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"query": "docker", "prompt": "What was the plan?"},
            )
        )

        assert result["degraded"] is True
        assert "empty answer" in result["error"]
        assert result["node_ids"] == [node_id]
        assert result["matches"]
        assert "answer" not in result

    def test_handle_expand_query_node_ids_timeout_preserves_requested_match(self, engine, monkeypatch):
        engine._store.append("test-session", {"role": "user", "content": "Discussed docker rollout plan"})
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Docker rollout summary",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            raise TimeoutError("expansion timed out")

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"node_ids": [node_id], "prompt": "What was the plan?"},
            )
        )

        assert result["degraded"] is True
        assert "timed out" in result["error"]
        assert result["query"] == ""
        assert result["node_ids"] == [node_id]
        assert result["matches"][0]["node_id"] == node_id
        assert "answer" not in result

    def test_handle_expand_query_hyphenated_operator_query_falls_back_cleanly(self, engine, monkeypatch):
        engine._store.append(
            "test-session",
            {
                "role": "user",
                "content": "hermes-lcm plugin-only external context-engine generic host support no vendoring stays external",
            },
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="hermes-lcm plugin-only external context-engine generic host support no vendoring stays external",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )

        monkeypatch.setattr(
            lcm_tools, "_synthesize_expansion_answer",
            lambda **kwargs: "Recovered through normalized retrieval",
        )
        monkeypatch.setattr(
            engine._store,
            "_search_like",
            lambda *args, **kwargs: pytest.fail("compound query fell back to LIKE"),
        )
        monkeypatch.setattr(
            engine._dag,
            "_search_like",
            lambda *args, **kwargs: pytest.fail("compound query fell back to LIKE"),
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "query": "8416 OR vendored OR vendoring OR plugin-only OR external context-engine OR generic host support OR hermes-lcm stays external OR no vendoring",
                    "prompt": "What were the agreements?",
                    "max_tokens": 500,
                },
            )
        )

        # #168: raw bare OR tokens are literals, so the full conjunction has no match.
        assert result["answer"] == "No matching summaries or raw messages found in the current session."
        assert result["node_ids"] == []
        assert result["matches"] == []
        assert result["raw_matches"] == []

        indexed = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "query": "plugin-only context-engine hermes-lcm stays external",
                    "prompt": "What were the agreements?",
                    "max_tokens": 500,
                },
            )
        )

        assert indexed["answer"] == "Recovered through normalized retrieval"
        assert indexed["node_ids"] == [node_id]
        assert indexed["matches"]

    def test_handle_expand_query_rejects_non_numeric_limits(self, engine):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"query": "docker", "prompt": "What was the plan?", "max_tokens": "invalid"},
            )
        )

        assert result["error"] == "max_tokens must be an integer"

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"query": "docker", "prompt": "What was the plan?", "max_results": "invalid"},
            )
        )

        assert result["error"] == "max_results must be an integer"

    def test_handle_expand_query_rejects_non_numeric_node_ids(self, engine):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"node_ids": ["not-a-number"], "prompt": "What was the plan?"},
            )
        )

        assert result["error"] == "node_ids must contain only integers"

    def test_handle_expand_query_accepts_valid_integer_node_ids(self, engine, monkeypatch):
        engine._store.append("test-session", {"role": "user", "content": "Discussed docker rollout plan"})
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Docker rollout summary",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )

        def fake_synthesize(*, prompt, context_blocks, model, max_tokens, timeout):
            return "Expansion answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {"node_ids": [node_id], "prompt": "What was the plan?"},
            )
        )

        assert "error" not in result, f"unexpected error: {result.get('error')}"
        assert result["answer"] == "Expansion answer"
        assert result["node_ids"] == [node_id]

    def test_describe_and_expand_are_session_scoped(self, engine):
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="session-a",
                depth=0,
                summary="secret summary",
                token_count=10,
                source_token_count=20,
                source_ids=[],
                source_type="messages",
                created_at=0,
            )
        )

        engine._session_id = "session-b"

        describe = json.loads(engine.handle_tool_call("lcm_describe", {"node_id": node_id}))
        expand = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))

        assert "error" in describe
        assert "error" in expand

    def test_describe_overview_includes_sparse_high_depth_nodes(self, engine):
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=2,
                summary="durable summary",
                token_count=100,
                source_token_count=500,
                source_ids=[],
                source_type="messages",
                created_at=0,
            )
        )

        overview = json.loads(engine.handle_tool_call("lcm_describe", {}))
        assert "d2" in overview["depths"]

    def test_handle_describe_and_expand_accept_explicit_cross_session_node(
        self, engine
    ):
        store_id = engine._store.append(
            "old-session",
            {"role": "user", "content": "historical expanded evidence"},
        )
        child_id = engine._dag.add_node(
            SummaryNode(
                session_id="old-session",
                depth=0,
                summary="historical child summary",
                token_count=5,
                source_token_count=5,
                source_ids=[store_id],
                source_type="messages",
                created_at=1,
            )
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="old-session",
                depth=1,
                summary="historical parent summary",
                token_count=5,
                source_token_count=10,
                source_ids=[child_id],
                source_type="nodes",
                created_at=2,
            )
        )

        describe = json.loads(
            engine.handle_tool_call(
                "lcm_describe",
                {"node_id": node_id, "session_id": "old-session"},
            )
        )
        mismatched = json.loads(
            engine.handle_tool_call(
                "lcm_describe",
                {"node_id": node_id, "session_id": "wrong-session"},
            )
        )
        expanded = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {"node_id": node_id, "session_id": "old-session"},
            )
        )

        assert describe["node_id"] == node_id
        assert describe["session_id"] == "old-session"
        assert describe["expand_hint"] == (
            f"lcm_expand(node_id={node_id}, session_id='old-session')"
        )
        assert describe["children"][0]["expand_hint"] == (
            f"lcm_expand(node_id={child_id}, session_id='old-session')"
        )
        assert (
            mismatched["error"] == f"Node {node_id} not found in session wrong-session"
        )
        assert expanded["session_id"] == "old-session"
        assert expanded["expanded"][0]["node_id"] == child_id
        assert expanded["expanded"][0]["expand_hint"] == (
            f"lcm_expand(node_id={child_id}, session_id='old-session')"
        )
        child_expanded = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {"node_id": child_id, "session_id": "old-session"},
            )
        )
        assert (
            child_expanded["expanded"][0]["content"] == "historical expanded evidence"
        )

    def test_handle_expand_query_merges_sessions_by_relevance(
        self, engine, monkeypatch
    ):
        strong_id = engine._dag.add_node(
            SummaryNode(
                session_id="older-session",
                depth=0,
                summary="strong needle match",
                token_count=20,
                source_token_count=20,
                source_ids=[],
                source_type="messages",
                created_at=1,
            )
        )
        weak_id = engine._dag.add_node(
            SummaryNode(
                session_id="newer-session",
                depth=0,
                summary="weak needle match",
                token_count=1,
                source_token_count=1,
                source_ids=[],
                source_type="messages",
                created_at=2,
            )
        )

        def fake_node_search(query, session_id=None, limit=20):
            del query, limit
            node = engine._dag.get_node(
                strong_id if session_id == "older-session" else weak_id
            )
            node.search_rank = -4.0 if session_id == "older-session" else -3.0
            return [node]

        monkeypatch.setattr(engine._dag, "search", fake_node_search)
        monkeypatch.setattr(engine._store, "search", lambda *args, **kwargs: [])

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "Find the strongest match",
                    "query": "needle",
                    "session_ids": ["older-session", "newer-session"],
                    "max_results": 1,
                    "output": "evidence",
                },
            )
        )

        assert result["node_ids"] == [strong_id]

    def test_handle_expand_query_searches_bounded_explicit_sessions(
        self, engine, monkeypatch
    ):
        captured = {}
        old_store_id = engine._store.append(
            "old-session",
            {"role": "user", "content": "NEEDLE historical raw evidence"},
        )
        engine._store.append(
            "other-session",
            {"role": "user", "content": "NEEDLE excluded raw evidence"},
        )
        old_node_id = engine._dag.add_node(
            SummaryNode(
                session_id="old-session",
                depth=0,
                summary="NEEDLE historical summary evidence",
                token_count=5,
                source_token_count=5,
                source_ids=[old_store_id],
                source_type="messages",
                created_at=1,
            )
        )

        def fake_synthesize(**kwargs):
            captured["context_blocks"] = kwargs["context_blocks"]
            return "cross-session answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What is the historical evidence?",
                    "query": "NEEDLE",
                    "session_ids": ["old-session"],
                    "context_max_tokens": 1000,
                },
            )
        )

        assert result["answer"] == "cross-session answer"
        assert result["node_ids"] == [old_node_id]
        assert result["matches"][0]["session_id"] == "old-session"
        serialized_context = json.dumps(captured["context_blocks"])
        assert "NEEDLE historical raw evidence" in serialized_context
        assert "NEEDLE excluded raw evidence" not in serialized_context

    def test_handle_expand_query_raw_matches_carry_session_ownership(
        self, engine, monkeypatch
    ):
        # Cross-session hits that match only raw messages: `matches` is empty
        # because no summary node matches, and the `answer` output returns no
        # context blocks, so `raw_matches` is the caller's only channel for
        # session ownership. Losing it here loses provenance entirely.
        alpha_store_id = engine._store.append(
            "alpha-session",
            {"role": "user", "content": "KRYPTONITE alpha raw evidence"},
        )
        beta_store_id = engine._store.append(
            "beta-session",
            {"role": "user", "content": "KRYPTONITE beta raw evidence"},
        )

        monkeypatch.setattr(
            lcm_tools,
            "_synthesize_expansion_answer",
            lambda **kwargs: "cross-session raw answer",
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "Which sessions mention KRYPTONITE?",
                    "query": "KRYPTONITE",
                    "session_ids": ["alpha-session", "beta-session"],
                    "context_max_tokens": 1000,
                },
            )
        )

        assert result["answer"] == "cross-session raw answer"
        assert result["matches"] == []
        assert {
            match["store_id"]: match["session_id"] for match in result["raw_matches"]
        } == {
            alpha_store_id: "alpha-session",
            beta_store_id: "beta-session",
        }

    def test_handle_expand_query_evidence_raw_matches_carry_session_ownership(
        self, engine
    ):
        gamma_store_id = engine._store.append(
            "gamma-session",
            {"role": "user", "content": "OBSIDIAN gamma raw evidence"},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "Which session mentions OBSIDIAN?",
                    "query": "OBSIDIAN",
                    "session_ids": ["gamma-session"],
                    "output": "evidence",
                    "context_max_tokens": 1000,
                },
            )
        )

        assert [match["session_id"] for match in result["raw_matches"]] == [
            "gamma-session"
        ]
        assert result["raw_matches"][0]["store_id"] == gamma_store_id

    def test_handle_expand_query_evidence_budget_counts_emitted_encoding(self, engine):
        # `lcm_expand_query` emits every payload as json.dumps(payload) with the
        # default ensure_ascii=True, so non-ASCII evidence goes out as \uXXXX
        # escapes (a surrogate pair per astral emoji). Budgeting an
        # ensure_ascii=False serialization under-counts CJK/emoji evidence by a
        # multiple: the emitted payload overruns context_max_tokens while
        # context_tokens claims it is inside budget.
        engine._store.append(
            "cjk-session",
            {
                "role": "user",
                "content": "TANUKI " + "日本語の長い会話記録です。🎉🚀 " * 40,
            },
        )

        def _evidence(context_max_tokens: int) -> dict:
            return json.loads(
                engine.handle_tool_call(
                    "lcm_expand_query",
                    {
                        "prompt": "What mentions TANUKI?",
                        "query": "TANUKI",
                        "session_ids": ["cjk-session"],
                        "output": "evidence",
                        "context_max_tokens": context_max_tokens,
                    },
                )
            )

        probe = _evidence(100_000)
        assert probe["evidence"], "probe must admit the non-ASCII evidence block"
        emitted_tokens = count_tokens(json.dumps(probe["evidence"]))
        unescaped_tokens = count_tokens(
            json.dumps(probe["evidence"], ensure_ascii=False)
        )
        assert emitted_tokens > unescaped_tokens, "fixture must carry non-ASCII"

        # context_tokens describes what the caller actually receives.
        assert probe["context_tokens"] == emitted_tokens

        # A budget the unescaped serialization fits under but the emitted one
        # does not must be enforced against the emitted encoding.
        budget = (emitted_tokens + unescaped_tokens) // 2
        assert unescaped_tokens <= budget < emitted_tokens
        result = _evidence(budget)
        assert count_tokens(json.dumps(result["evidence"])) <= budget
        assert result["context_tokens"] <= budget
        assert result["context_truncated"] is True

    def test_handle_expand_query_rejects_more_than_twenty_session_ids(self, engine):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "Find evidence",
                    "query": "anything",
                    "session_ids": [f"session-{index}" for index in range(21)],
                },
            )
        )

        assert result["error"] == "session_ids accepts at most 20 entries"

    def test_handle_expand_query_evidence_skips_synthesis_and_stays_bounded(
        self, engine, monkeypatch
    ):
        store_id = engine._store.append(
            "old-session",
            {"role": "user", "content": "bounded evidence " * 200},
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="old-session",
                depth=0,
                summary="bounded evidence summary",
                token_count=5,
                source_token_count=400,
                source_ids=[store_id],
                source_type="messages",
                created_at=1,
            )
        )

        def fail_synthesis(**kwargs):
            raise AssertionError("evidence output must not call the synthesis model")

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fail_synthesis)
        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "Return the evidence",
                    "node_ids": [node_id],
                    "session_ids": ["old-session"],
                    "output": "evidence",
                    "context_max_tokens": 200,
                },
            )
        )

        assert result["output"] == "evidence"
        assert "answer" not in result
        assert result["node_ids"] == [node_id]
        serialized_evidence = json.dumps(result["evidence"], ensure_ascii=False)
        assert count_tokens(serialized_evidence) <= 200
        assert result["context_tokens"] == count_tokens(serialized_evidence)
        assert result["evidence"][0]["session_id"] == "old-session"

    def test_handle_expand_query_evidence_drops_metadata_over_budget(self, engine):
        node_ids = [
            engine._dag.add_node(
                SummaryNode(
                    session_id="old-session",
                    depth=0,
                    summary="",
                    token_count=0,
                    source_token_count=0,
                    source_ids=[],
                    source_type="messages",
                    created_at=index,
                )
            )
            for index in range(100)
        ]

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "Return bounded evidence",
                    "node_ids": node_ids,
                    "session_ids": ["old-session"],
                    "output": "evidence",
                    "max_results": 100,
                    "context_max_tokens": 1,
                },
            )
        )

        serialized_evidence = json.dumps(result["evidence"], ensure_ascii=False)
        assert count_tokens(serialized_evidence) <= 1
        assert result["context_tokens"] == count_tokens(serialized_evidence)
        assert result["context_truncated"] is True

    def test_handle_expand_query_evidence_counts_emitted_json_spacing(self, engine):
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="old-session",
                depth=0,
                summary="",
                token_count=0,
                source_token_count=0,
                source_ids=[],
                source_type="messages",
                created_at=1,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "Return bounded evidence",
                    "node_ids": [node_id],
                    "session_ids": ["old-session"],
                    "output": "evidence",
                    "context_max_tokens": 36,
                },
            )
        )

        serialized_evidence = json.dumps(result["evidence"], ensure_ascii=False)
        assert count_tokens(serialized_evidence) <= 36
        assert result["context_tokens"] == count_tokens(serialized_evidence)
        assert result["context_truncated"] is True

    @pytest.mark.parametrize("invalid_output", ["", 0, []])
    def test_handle_expand_query_rejects_explicit_invalid_output(
        self, engine, monkeypatch, invalid_output
    ):
        monkeypatch.setattr(
            lcm_tools,
            "_synthesize_expansion_answer",
            lambda **kwargs: pytest.fail("invalid output must not invoke synthesis"),
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "Return evidence",
                    "query": "anything",
                    "output": invalid_output,
                },
            )
        )

        assert result["error"] == "output must be one of: answer, evidence"

    def test_handle_expand_query_keeps_raw_cursor_when_summaries_exhaust_budget(
        self, engine, monkeypatch
    ):
        # Summaries can consume the whole context budget while distinct raw
        # messages still matched. Dropping the raw arm silently strands those
        # hits: `context_truncated` alone gives the caller no handle to
        # continue into them, so the raw-message cursor must survive even when
        # no raw content fits.
        raw_store_id = engine._store.append(
            "old-session",
            {"role": "user", "content": "SENTINEL raw hit that must stay reachable"},
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="old-session",
                depth=0,
                summary="SENTINEL " + "summary padding " * 200,
                token_count=400,
                source_token_count=400,
                source_ids=[],
                source_type="messages",
                created_at=1,
            )
        )

        monkeypatch.setattr(
            lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "answer"
        )
        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What is SENTINEL?",
                    "query": "SENTINEL",
                    "session_ids": ["old-session"],
                    "context_max_tokens": 12,
                },
            )
        )

        assert result["context_truncated"] is True
        raw_entry = next(
            (
                item
                for item in result["context_pagination"]
                if item.get("type") == "raw_messages"
            ),
            None,
        )
        assert raw_entry is not None
        assert raw_entry["pagination"]["next_store_id"] == raw_store_id
        assert raw_entry["expand_args"] == {"store_id": raw_store_id}

    def test_handle_expand_query_evidence_keeps_raw_cursor_when_block_is_dropped(
        self, engine
    ):
        # Evidence output drops a raw block that will not fit the serialized
        # budget. The omitted hits still need a continuation handle.
        raw_store_id = engine._store.append(
            "old-session",
            {"role": "user", "content": "OBELISK raw hit that must stay reachable"},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What is OBELISK?",
                    "query": "OBELISK",
                    "session_ids": ["old-session"],
                    "output": "evidence",
                    "context_max_tokens": 1,
                },
            )
        )

        assert result["evidence"] == []
        assert result["context_truncated"] is True
        raw_entry = next(
            (
                item
                for item in result["context_pagination"]
                if item.get("type") == "raw_messages"
            ),
            None,
        )
        assert raw_entry is not None
        assert raw_entry["expand_args"] == {"store_id": raw_store_id}

    def test_handle_expand_query_answer_provenance_reports_explicit_sessions(
        self, engine, monkeypatch
    ):
        # A multi-session answer must not describe its retrieval scope as the
        # current session, and its summary locators must replay against the
        # session the node came from rather than the active one.
        historical_node_id = engine._dag.add_node(
            SummaryNode(
                session_id="old-session",
                depth=0,
                summary="ZIRCON historical summary",
                token_count=5,
                source_token_count=5,
                source_ids=[],
                source_type="messages",
                created_at=1,
            )
        )
        current_node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="ZIRCON current summary",
                token_count=5,
                source_token_count=5,
                source_ids=[],
                source_type="messages",
                created_at=2,
            )
        )

        monkeypatch.setattr(
            lcm_tools,
            "_synthesize_expansion_answer",
            lambda **kwargs: "cross-session answer",
        )
        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What is ZIRCON?",
                    "query": "ZIRCON",
                    "session_ids": ["old-session", "test-session"],
                    "context_max_tokens": 2000,
                },
            )
        )

        scope = result["evidence_provenance"]["retrieval_scope"]
        assert scope["kind"] == "explicit_sessions"
        assert scope["session_ids"] == ["old-session", "test-session"]
        items = {
            item["node_id"]: item
            for item in result["evidence_provenance"]["items"]
            if item["source_type"] == "summary"
        }
        assert items[historical_node_id]["session_id"] == "old-session"
        assert items[historical_node_id]["expand_args"] == {
            "node_id": historical_node_id,
            "session_id": "old-session",
        }
        assert items[current_node_id]["session_id"] == "test-session"
        assert items[current_node_id]["expand_args"] == {"node_id": current_node_id}

    def test_handle_expand_query_marks_historical_messages_as_not_current(
        self, engine, monkeypatch
    ):
        # Threading the node's session into child expansion selects the lookup
        # scope. `from_current_session` still describes the ACTIVE session, so
        # archived rows must not be advertised as current.
        store_id = engine._store.append(
            "old-session",
            {"role": "user", "content": "TOPAZ historical raw evidence"},
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="old-session",
                depth=0,
                summary="TOPAZ historical summary",
                token_count=5,
                source_token_count=5,
                source_ids=[store_id],
                source_type="messages",
                created_at=1,
            )
        )

        captured = {}

        def fake_synthesize(**kwargs):
            captured["context_blocks"] = kwargs["context_blocks"]
            return "historical answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", fake_synthesize)
        json.loads(
            engine.handle_tool_call(
                "lcm_expand_query",
                {
                    "prompt": "What is TOPAZ?",
                    "node_ids": [node_id],
                    "session_ids": ["old-session"],
                    "context_max_tokens": 2000,
                },
            )
        )

        message_block = next(
            block
            for block in captured["context_blocks"]
            if block.get("type") == "messages"
        )
        assert message_block["messages"][0]["store_id"] == store_id
        assert message_block["messages"][0]["session_id"] == "old-session"
        assert message_block["messages"][0]["from_current_session"] is False



class TestHandleExpandStoreId:
    """lcm_expand store_id mode for cross-session raw expansion."""

    def test_store_id_returns_raw_message_cross_session(self, engine):
        store_id = engine._store.append(
            "old-session",
            {"role": "user", "content": "cross session content body"},
            source="cli",
        )
        result = json.loads(engine.handle_tool_call("lcm_expand", {"store_id": store_id}))
        assert result["source_type"] == "raw_message"
        assert result["store_id"] == store_id
        assert result["session_id"] == "old-session"
        assert result["from_current_session"] is False
        assert result["role"] == "user"
        assert result["source"] == "cli"
        assert result["content"].startswith("cross session content")
        assert "exact_ref" not in result

        exact = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {"store_id": store_id, "include_exact_ref": True},
            )
        )
        assert exact["exact_ref"] == f"lcm:{store_id}:0-{len(exact['content'])}"

    def test_store_id_paging_via_content_offset(self, engine):
        big_content = "x" * 10000
        store_id = engine._store.append(
            "old-session", {"role": "user", "content": big_content}
        )
        first = json.loads(
            engine.handle_tool_call(
                "lcm_expand", {"store_id": store_id, "max_tokens": 50}
            )
        )
        assert first["content_truncated"] is True
        assert first["next_content_offset"] > 0
        second = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {
                    "store_id": store_id,
                    "max_tokens": 50,
                    "content_offset": first["next_content_offset"],
                },
            )
        )
        assert second["content_offset"] == first["next_content_offset"]
        assert second["content"]
        # Combined slices should not exceed total content length.
        assert first["content_chars"] == second["content_chars"]

        exact_second = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {
                    "store_id": store_id,
                    "max_tokens": 50,
                    "content_offset": first["next_content_offset"],
                    "include_exact_ref": True,
                },
            )
        )
        exact_end = exact_second["content_offset"] + exact_second["content_returned_chars"]
        assert exact_second["exact_ref"] == (
            f"lcm:{store_id}:{exact_second['content_offset']}-{exact_end}"
        )

    def test_store_id_not_found_returns_error(self, engine):
        result = json.loads(
            engine.handle_tool_call("lcm_expand", {"store_id": 999_999_999})
        )
        assert "error" in result
        assert "store_id" in result["error"]

    def test_store_id_not_an_integer_returns_error(self, engine):
        result = json.loads(
            engine.handle_tool_call("lcm_expand", {"store_id": "not-an-int"})
        )
        assert "error" in result
        assert "integer" in result["error"]

    def test_multiple_modes_returns_error(self, engine):
        store_id = engine._store.append(
            "test-session", {"role": "user", "content": "x"}
        )
        result = json.loads(
            engine.handle_tool_call(
                "lcm_expand", {"store_id": store_id, "node_id": 1}
            )
        )
        assert "error" in result
        assert "Provide only one" in result["error"]

    def test_no_modes_returns_error(self, engine):
        result = json.loads(engine.handle_tool_call("lcm_expand", {}))
        assert "error" in result
        assert "node_id" in result["error"] and "store_id" in result["error"]

    def test_node_id_remains_session_scoped(self, engine):
        # Node belongs to a different session — must not be expandable via node_id.
        engine._store.append("old-session", {"role": "user", "content": "old content"})
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="old-session",
                depth=0,
                summary="old summary",
                token_count=5,
                source_token_count=5,
                source_ids=[1],
                source_type="messages",
                created_at=time.time(),
            )
        )
        result = json.loads(
            engine.handle_tool_call("lcm_expand", {"node_id": node_id})
        )
        assert "error" in result
        assert "current session" in result["error"].lower()

    def test_store_id_cross_session_externalized_ref_surfaced_with_note(self, engine):
        # Seed a foreign-session tool message that references an externalized
        # payload. The ref string follows the produced-placeholder shape so
        # extract_externalized_ref will pick it up.
        placeholder = (
            "[Externalized tool output: tool_call_id=call_abc; "
            "chars=1234; bytes=5678; ref=foreign_payload_ref.json]"
        )
        store_id = engine._store.append(
            "old-session",
            {"role": "tool", "content": placeholder, "tool_call_id": "call_abc"},
        )
        result = json.loads(engine.handle_tool_call("lcm_expand", {"store_id": store_id}))
        assert result["source_type"] == "raw_message"
        assert result["from_current_session"] is False
        assert result["externalized_ref"] == "foreign_payload_ref.json"
        # Cross-session payload metadata is intentionally omitted; an explanatory
        # note is surfaced so callers don't treat the bare ref as expandable.
        assert "externalized" not in result
        assert "externalized_note" in result
        assert "session-scoped" in result["externalized_note"].lower()

    def test_grep_then_expand_round_trip_cross_session(self, engine):
        store_id = engine._store.append(
            "old-session", {"role": "user", "content": "phoenix payload across session"}
        )
        grep_result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "phoenix", "session_scope": "all"},
            )
        )
        cross_hits = [
            hit for hit in grep_result["results"]
            if hit["type"] == "message" and hit["session_id"] == "old-session"
        ]
        assert cross_hits, "cross-session grep should surface the seeded message"
        assert cross_hits[0]["store_id"] == store_id

        expand_result = json.loads(
            engine.handle_tool_call(
                "lcm_expand", {"store_id": cross_hits[0]["store_id"]}
            )
        )
        assert expand_result["source_type"] == "raw_message"
        assert "phoenix payload" in expand_result["content"]



class TestHandleLoadSession:
    """Ordered, paged raw transcript loading by explicit session_id."""

    def _seed_old_session(self, engine):
        rows = [
            ("user", "first old-session message", "cli", 100.0),
            ("assistant", "second old-session answer", "cli", 200.0),
            ("tool", "third old-session tool result", "cli", 300.0),
            ("user", "fourth old-session follow-up", "telegram", 400.0),
        ]
        store_ids = []
        for role, content, source, timestamp in rows:
            store_id = engine._store.append(
                "old-session",
                {"role": role, "content": content, "tool_call_id": "call_x" if role == "tool" else None},
                source=source,
            )
            engine._store._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE store_id = ?",
                (timestamp, store_id),
            )
            store_ids.append(store_id)
        engine._store._conn.commit()
        engine._store.append("test-session", {"role": "user", "content": "current session message"})
        return store_ids

    def test_load_session_returns_ordered_bounded_raw_page(self, engine):
        store_ids = self._seed_old_session(engine)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "old-session", "limit": 2},
            )
        )

        assert result["session_id"] == "old-session"
        assert result["limit"] == 2
        assert result["total_messages"] == 4
        assert result["returned_messages"] == 2
        assert result["has_more"] is True
        assert result["next_cursor"] == store_ids[1]
        assert [item["store_id"] for item in result["messages"]] == store_ids[:2]
        assert [item["role"] for item in result["messages"]] == ["user", "assistant"]
        assert result["messages"][0]["content"] == "first old-session message"
        assert result["messages"][0]["content_chars"] == len("first old-session message")
        assert result["messages"][0]["content_truncated"] is False
        assert result["messages"][0]["from_current_session"] is False
        assert "snippet" not in result["messages"][0]
        assert "exact_ref" not in result["messages"][0]

        exact = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {
                    "session_id": "old-session",
                    "limit": 2,
                    "include_exact_ref": True,
                },
            )
        )
        assert exact["messages"][0]["exact_ref"] == (
            f"lcm:{store_ids[0]}:0-{len('first old-session message')}"
        )

    def test_load_session_pages_after_store_id(self, engine):
        store_ids = self._seed_old_session(engine)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "old-session", "after_store_id": store_ids[1], "limit": 10},
            )
        )

        assert result["after_store_id"] == store_ids[1]
        assert result["has_more"] is False
        assert result["next_cursor"] is None
        assert [item["store_id"] for item in result["messages"]] == store_ids[2:]

    def test_load_session_filters_roles_and_time_range(self, engine):
        store_ids = self._seed_old_session(engine)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {
                    "session_id": "old-session",
                    "roles": ["user", "tool"],
                    "time_from": 250.0,
                    "time_to": 450.0,
                    "limit": 10,
                },
            )
        )

        assert result["roles"] == ["user", "tool"]
        assert result["time_from"] == 250.0
        assert result["time_to"] == 450.0
        assert result["total_messages"] == 2
        assert [item["store_id"] for item in result["messages"]] == [store_ids[2], store_ids[3]]
        assert [item["role"] for item in result["messages"]] == ["tool", "user"]

    def test_load_session_bounds_large_message_content(self, engine):
        store_id = engine._store.append(
            "large-session",
            {"role": "assistant", "content": "abcdef"},
            source="cli",
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "large-session", "max_content_chars": 3},
            )
        )

        assert result["max_content_chars"] == 3
        assert result["messages"][0]["store_id"] == store_id
        assert result["messages"][0]["content"] == "abc"
        assert result["messages"][0]["content_chars"] == 6
        assert result["messages"][0]["content_returned_chars"] == 3
        assert result["messages"][0]["content_truncated"] is True
        assert result["messages"][0]["next_content_offset"] == 3

    def test_load_session_clamps_max_content_chars(self, engine):
        store_id = engine._store.append(
            "large-session",
            {"role": "user", "content": "x" * 25_000},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "large-session", "max_content_chars": 50_000},
            )
        )

        assert result["max_content_chars"] == 20_000
        assert result["max_content_chars_clamped_from"] == 50_000
        assert result["messages"][0]["store_id"] == store_id
        assert len(result["messages"][0]["content"]) == 20_000
        assert result["messages"][0]["content_truncated"] is True

    def test_load_session_rejects_missing_session_id_and_invalid_filters(self, engine):
        missing = json.loads(engine.handle_tool_call("lcm_load_session", {}))
        assert "error" in missing and "session_id" in missing["error"]

        bad_roles = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "old-session", "roles": "user"},
            )
        )
        assert "error" in bad_roles and "roles" in bad_roles["error"]

        bad_cursor = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "old-session", "after_store_id": "not-an-id"},
            )
        )
        assert "error" in bad_cursor and "after_store_id" in bad_cursor["error"]

        bad_limit = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "old-session", "limit": "not-a-limit"},
            )
        )
        assert "error" in bad_limit and "limit" in bad_limit["error"]

        bad_max_content_chars = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "old-session", "max_content_chars": "not-a-size"},
            )
        )
        assert "error" in bad_max_content_chars and "max_content_chars" in bad_max_content_chars["error"]

        bad_range = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "old-session", "time_from": 5, "time_to": 4},
            )
        )
        assert "error" in bad_range and "time_to" in bad_range["error"]

        bad_exact_ref = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "old-session", "include_exact_ref": "yes"},
            )
        )
        assert "error" in bad_exact_ref and "include_exact_ref" in bad_exact_ref["error"]

    def test_load_session_clamps_limit_and_never_falls_back_to_current(self, engine):
        self._seed_old_session(engine)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_load_session",
                {"session_id": "missing-session", "limit": 5000},
            )
        )

        assert result["session_id"] == "missing-session"
        assert result["limit"] == 200
        assert result["limit_clamped_from"] == 5000
        assert result["total_messages"] == 0
        assert result["messages"] == []
        assert result["has_more"] is False
