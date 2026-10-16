"""LCM engine integration tests: grep."""

import json
import time
from pathlib import Path

import pytest

import hermes_lcm.tools as lcm_tools

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import externalize_ingest_payload

from .engine_helpers import (
    engine as engine,
    externalized_search_engine as externalized_search_engine,
)


class TestEngineTools:
    def test_handle_grep(self, engine):
        # Add some data
        engine._store.append("test-session", {"role": "user", "content": "deploy docker containers"})
        result = json.loads(engine.handle_tool_call("lcm_grep", {"query": "docker"}))
        assert "results" in result

    def test_handle_grep_unbound_current_session_does_not_search_all_sessions(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "unbound-current-session.db"))
        instance = LCMEngine(config=config)
        assert instance._session_id == ""
        instance._store.append("session-a", {"role": "user", "content": "docker from session a"})
        instance._store.append("session-b", {"role": "user", "content": "docker from session b"})

        result = json.loads(instance.handle_tool_call("lcm_grep", {"query": "docker", "limit": 10}))

        assert result["session_scope"] == "current"
        assert result["total_results"] == 0
        assert result["results"] == []

    def test_handle_grep_reports_sort_mode(self, engine):
        engine._store.append(
            "test-session",
            {"role": "user", "content": "database migration plan database migration plan"},
        )
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": '"database migration plan"', "limit": 1, "sort": "relevance"},
            )
        )
        assert result["sort"] == "relevance"

    def test_handle_grep_role_filter_pushes_into_message_search(self, engine):
        assistant_id = engine._store.append(
            "test-session",
            {"role": "assistant", "content": "docker target assistant"},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1000.0, assistant_id),
        )
        for idx in range(25):
            store_id = engine._store.append(
                "test-session",
                {"role": "user", "content": f"docker noisy user {idx}"},
            )
            engine._store._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE store_id = ?",
                (2000.0 + idx, store_id),
            )
        engine._store._conn.commit()

        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "docker", "role": "assistant", "limit": 1, "sort": "recency"},
        ))

        assert result["role"] == "assistant"
        assert result["total_results"] == 1
        assert result["results"][0]["role"] == "assistant"
        assert "target assistant" in result["results"][0]["snippet"]

    def test_handle_grep_time_filter_pushes_into_message_search(self, engine):
        older_id = engine._store.append(
            "test-session",
            {"role": "assistant", "content": "docker older assistant"},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1000.0, older_id),
        )
        for idx in range(25):
            store_id = engine._store.append(
                "test-session",
                {"role": "user", "content": f"docker newer user {idx}"},
            )
            engine._store._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE store_id = ?",
                (2000.0 + idx, store_id),
            )
        engine._store._conn.commit()

        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "docker", "time_to": 1500.0, "limit": 1, "sort": "recency"},
        ))

        assert result["time_to"] == 1500.0
        assert result["total_results"] == 1
        assert result["results"][0]["timestamp"] == 1000.0
        assert "older assistant" in result["results"][0]["snippet"]

    def test_handle_grep_sanitizes_period_in_unquoted_fts_query(self, engine):
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "release notes for v2.21 and api.v2"},
        )

        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "v2.21", "limit": 1},
        ))

        assert "error" not in result
        assert result["total_results"] == 1
        assert "release notes for" in result["results"][0]["snippet"]
        assert "api" in result["results"][0]["snippet"]

    def test_handle_grep_rejects_naive_iso_time_filter(self, engine):
        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "docker", "time_to": "2026-05-06T10:00:00"},
        ))

        assert "error" in result
        assert "timezone" in result["error"]

    def test_handle_grep_like_fallback_applies_role_filter_before_limit(self, engine):
        assistant_id = engine._store.append(
            "test-session",
            {"role": "assistant", "content": "docker-compose target assistant"},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1000.0, assistant_id),
        )
        for idx in range(25):
            store_id = engine._store.append(
                "test-session",
                {"role": "user", "content": f"docker-compose noisy user {idx}"},
            )
            engine._store._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE store_id = ?",
                (2000.0 + idx, store_id),
            )
        engine._store._conn.commit()

        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "docker-compose", "role": "assistant", "limit": 1, "sort": "recency"},
        ))

        assert result["role"] == "assistant"
        assert result["total_results"] == 1
        assert result["results"][0]["role"] == "assistant"
        assert "target assistant" in result["results"][0]["snippet"]

    def test_handle_grep_like_fallback_role_filter_preserves_recency_before_limit(self, engine):
        for idx in range(220):
            store_id = engine._store.append(
                "test-session",
                {"role": "assistant", "content": f"docker-compose older assistant {idx}"},
            )
            engine._store._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE store_id = ?",
                (1000.0 + idx, store_id),
            )
        newest_id = engine._store.append(
            "test-session",
            {"role": "assistant", "content": "docker-compose newest assistant"},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (5000.0, newest_id),
        )
        engine._store._conn.commit()

        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "docker-compose", "role": "assistant", "limit": 1, "sort": "recency"},
        ))

        assert result["role"] == "assistant"
        assert len(result["results"]) == 1
        assert result["results"][0]["role"] == "assistant"
        assert result["results"][0]["store_id"] == newest_id
        assert "newest assistant" in result["results"][0]["snippet"]

    def test_handle_grep_like_fallback_recency_sorts_tied_cap_by_score(self, engine):
        best_id = engine._store.append(
            "test-session",
            {"role": "assistant", "content": "foo/bar best assistant"},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (5000.0, best_id),
        )
        for idx in range(600):
            store_id = engine._store.append(
                "test-session",
                {"role": "assistant", "content": f"foo low assistant {idx}"},
            )
            engine._store._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE store_id = ?",
                (5000.0, store_id),
            )
        engine._store._conn.commit()

        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "foo/bar", "role": "assistant", "limit": 1, "sort": "recency"},
        ))

        assert result["role"] == "assistant"
        assert len(result["results"]) == 1
        assert result["results"][0]["role"] == "assistant"
        assert result["results"][0]["store_id"] == best_id
        assert "best assistant" in result["results"][0]["snippet"]

    def test_handle_grep_like_fallback_recency_sorts_tied_cap_by_directness(self, engine, monkeypatch):
        best_id = engine._store.append(
            "test-session",
            {"role": "assistant", "content": "foo/bar baz direct assistant"},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (5000.0, best_id),
        )
        first_repeated_id = None
        for idx in range(600):
            store_id = engine._store.append(
                "test-session",
                {"role": "assistant", "content": f"foo/bar baz foo foo foo foo low assistant {idx}"},
            )
            if first_repeated_id is None:
                first_repeated_id = store_id
            engine._store._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE store_id = ?",
                (5000.0, store_id),
            )
        engine._store._conn.commit()
        monkeypatch.setattr(
            engine._store,
            "_search_like",
            lambda *args, **kwargs: pytest.fail("compound query fell back to LIKE"),
        )

        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "foo/bar baz", "role": "assistant", "limit": 1, "sort": "recency"},
        ))

        assert result["role"] == "assistant"
        assert len(result["results"]) == 1
        assert result["results"][0]["role"] == "assistant"
        # #168: compound punctuation is sanitized to terms and stays on FTS.
        assert result["results"][0]["store_id"] == first_repeated_id
        assert "low assistant 0" in result["results"][0]["snippet"]

    def test_handle_grep_like_fallback_recency_extends_tied_cap_for_json_penalty(self, engine):
        best_id = engine._store.append(
            "test-session",
            {"role": "tool", "content": "foo/bar direct tool"},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (5000.0, best_id),
        )
        for idx in range(600):
            store_id = engine._store.append(
                "test-session",
                {"role": "tool", "content": f'{{"message":"foo/bar low tool {idx}"}}'},
            )
            engine._store._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE store_id = ?",
                (5000.0, store_id),
            )
        engine._store._conn.commit()

        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "foo/bar", "role": "tool", "limit": 1, "sort": "recency"},
        ))

        assert result["role"] == "tool"
        assert len(result["results"]) == 1
        assert result["results"][0]["role"] == "tool"
        assert result["results"][0]["store_id"] == best_id
        assert "direct tool" in result["results"][0]["snippet"]

    def test_handle_grep_like_fallback_applies_time_filter_before_limit(self, engine):
        older_id = engine._store.append(
            "test-session",
            {"role": "assistant", "content": "docker-compose older assistant"},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1000.0, older_id),
        )
        for idx in range(25):
            store_id = engine._store.append(
                "test-session",
                {"role": "user", "content": f"docker-compose newer user {idx}"},
            )
            engine._store._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE store_id = ?",
                (2000.0 + idx, store_id),
            )
        engine._store._conn.commit()

        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "docker-compose", "time_to": 1500.0, "limit": 1, "sort": "recency"},
        ))

        assert result["time_to"] == 1500.0
        assert result["total_results"] == 1
        assert result["results"][0]["timestamp"] == 1000.0
        assert "older assistant" in result["results"][0]["snippet"]

    def test_handle_grep_accepts_timezone_aware_iso_time_filter(self, engine):
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "docker timezone aware"},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1_700_000_000.0, store_id),
        )
        engine._store._conn.commit()

        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "docker", "time_from": "2023-11-14T22:13:20Z"},
        ))

        assert result["time_from"] == 1_700_000_000.0
        assert result["total_results"] == 1
        assert result["results"][0]["store_id"] == store_id

    def test_handle_grep_session_scope_all_returns_cross_session_hits(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "docker rollout current session"})
        engine._store.append("old-session", {"role": "user", "content": "docker rollout old session"})

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "all", "limit": 10},
            )
        )

        assert result["session_scope"] == "all"
        assert result["total_results"] == 2
        session_ids = {hit["session_id"] for hit in result["results"]}
        assert session_ids == {"test-session", "old-session"}
        from_current_flags = {
            hit["session_id"]: hit["from_current_session"] for hit in result["results"]
        }
        assert from_current_flags == {"test-session": True, "old-session": False}
        # No ignored_session_scope key when the requested scope is now supported.
        assert "ignored_session_scope" not in result

    def test_handle_grep_truly_unknown_session_scope_stays_current_and_reports(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "docker rollout current session"})
        engine._store.append("old-session", {"role": "user", "content": "docker rollout old session"})

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "everything", "limit": 10},
            )
        )

        assert result["session_scope"] == "current"
        assert result["ignored_session_scope"] == "everything"
        assert "current" in result["scope_note"]
        assert result["total_results"] == 1
        assert result["results"][0]["session_id"] == "test-session"

    def test_handle_grep_source_filter_in_current_session_includes_only_matching_summaries(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "docker logs from discord"}, source="discord")
        engine._store.append("s-telegram", {"role": "user", "content": "docker logs from telegram"}, source="telegram")
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="discord summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[1],
                source_type="messages",
                created_at=time.time(),
            )
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="s-telegram",
                depth=0,
                summary="telegram summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[2],
                source_type="messages",
                created_at=time.time(),
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "current", "source": "discord", "limit": 10},
            )
        )

        assert result["session_scope"] == "current"
        assert result["source"] == "discord"
        assert any(item["type"] == "message" for item in result["results"])
        assert any(item["type"] == "summary" for item in result["results"])
        assert all(item.get("session_id") == "test-session" for item in result["results"])
        assert all(item.get("source", "discord") == "discord" for item in result["results"] if item["type"] == "message")

    def test_handle_grep_source_filter_excludes_unrelated_summaries_in_mixed_source_session(self, engine):
        discord_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "docker logs from discord"},
            source="discord",
        )
        telegram_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "docker logs from telegram"},
            source="telegram",
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="discord summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[discord_store_id],
                source_type="messages",
                created_at=time.time(),
            )
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="telegram summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[telegram_store_id],
                source_type="messages",
                created_at=time.time(),
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "current", "source": "discord", "limit": 10},
            )
        )

        assert any(item["type"] == "message" for item in result["results"])
        assert any(item["type"] == "summary" for item in result["results"])
        assert all(item.get("session_id") == "test-session" for item in result["results"])
        assert all(
            "telegram summary" not in item.get("snippet", "")
            for item in result["results"]
            if item["type"] == "summary"
        )
        assert any(
            "discord summary" in item.get("snippet", "")
            for item in result["results"]
            if item["type"] == "summary"
        )

    def test_handle_grep_source_filter_pages_past_newer_unrelated_summaries(self, engine):
        discord_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "lineage row from discord"},
            source="discord",
        )
        telegram_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "lineage row from telegram"},
            source="telegram",
        )

        for idx in range(220):
            engine._dag.add_node(
                SummaryNode(
                    session_id="test-session",
                    depth=0,
                    summary=f"telegram summary {idx} about docker rollout",
                    token_count=10,
                    source_token_count=10,
                    source_ids=[telegram_store_id],
                    source_type="messages",
                    created_at=1_800_000_000 + idx,
                    latest_at=1_800_000_000 + idx,
                )
            )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="discord summary about docker rollout",
                token_count=10,
                source_token_count=10,
                source_ids=[discord_store_id],
                source_type="messages",
                created_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "current", "source": "discord", "limit": 1, "sort": "recency"},
            )
        )

        assert result["total_results"] == 1
        assert result["results"][0]["type"] == "summary"
        assert "discord summary" in result["results"][0]["snippet"]

    def test_handle_grep_source_filter_like_fallback_pages_past_unrelated_summaries(self, engine):
        discord_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "lineage row from discord"},
            source="discord",
        )
        telegram_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "lineage row from telegram"},
            source="telegram",
        )

        for idx in range(220):
            engine._dag.add_node(
                SummaryNode(
                    session_id="test-session",
                    depth=0,
                    summary=f"telegram summary {idx} about docker:rollout",
                    token_count=10,
                    source_token_count=10,
                    source_ids=[telegram_store_id],
                    source_type="messages",
                    created_at=1_800_000_000 + idx,
                    latest_at=1_800_000_000 + idx,
                )
            )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="discord summary about docker:rollout",
                token_count=10,
                source_token_count=10,
                source_ids=[discord_store_id],
                source_type="messages",
                created_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker:rollout", "session_scope": "current", "source": "discord", "limit": 1, "sort": "recency"},
            )
        )

        assert result["total_results"] == 1
        assert result["results"][0]["type"] == "summary"
        assert "discord summary" in result["results"][0]["snippet"]

    def test_handle_grep_source_filter_like_fallback_sorts_across_matching_candidates(self, engine):
        older_discord_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "older discord lineage"},
            source="discord",
        )
        telegram_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "telegram lineage"},
            source="telegram",
        )
        newer_discord_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "newer discord lineage"},
            source="discord",
        )

        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="older discord summary about docker:rollout",
                token_count=10,
                source_token_count=10,
                source_ids=[older_discord_store_id],
                source_type="messages",
                created_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )
        for idx in range(220):
            engine._dag.add_node(
                SummaryNode(
                    session_id="test-session",
                    depth=0,
                    summary=f"telegram summary {idx} about docker:rollout",
                    token_count=10,
                    source_token_count=10,
                    source_ids=[telegram_store_id],
                    source_type="messages",
                    created_at=1_800_000_000 + idx,
                    latest_at=1_800_000_000 + idx,
                )
            )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="newer discord summary about docker:rollout",
                token_count=10,
                source_token_count=10,
                source_ids=[newer_discord_store_id],
                source_type="messages",
                created_at=1_900_000_000,
                latest_at=1_900_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker:rollout", "session_scope": "current", "source": "discord", "limit": 1, "sort": "recency"},
            )
        )

        assert result["total_results"] == 2
        assert len(result["results"]) == 1
        assert result["results"][0]["type"] == "summary"
        assert "newer discord summary" in result["results"][0]["snippet"]

    def test_handle_grep_unknown_source_filter_matches_unknown_messages_and_summaries(self, engine):
        unknown_store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "docker logs from unknown source"},
            source="unknown",
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="unknown summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[unknown_store_id],
                source_type="messages",
                created_at=time.time(),
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "current", "source": "unknown", "limit": 10},
            )
        )

        assert result["source"] == "unknown"
        assert any(item["type"] == "message" for item in result["results"])
        assert any(item["type"] == "summary" for item in result["results"])
        assert all(item.get("session_id") == "test-session" for item in result["results"])
        assert all(item.get("source", "unknown") == "unknown" for item in result["results"] if item["type"] == "message")
        assert any(
            "unknown summary" in item.get("snippet", "")
            for item in result["results"]
            if item["type"] == "summary"
        )

    def test_handle_grep_unknown_source_filter_matches_whitespace_legacy_summary_lineage(self, engine):
        cursor = engine._store._conn.execute(
            """INSERT INTO messages
               (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("test-session", "\t\n", "user", "docker logs from whitespace legacy source", None, None, None, 1.0, 5, 0),
        )
        legacy_store_id = cursor.lastrowid
        engine._store._conn.commit()
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="whitespace legacy summary about docker logs",
                token_count=10,
                source_token_count=10,
                source_ids=[legacy_store_id],
                source_type="messages",
                created_at=time.time(),
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "current", "source": "unknown", "limit": 10},
            )
        )

        assert result["source"] == "unknown"
        assert any(
            item["type"] == "message" and item.get("source") == "unknown"
            for item in result["results"]
        )
        assert any(
            item["type"] == "summary" and "whitespace legacy summary" in item.get("snippet", "")
            for item in result["results"]
        )

    def test_handle_grep_prefers_conversational_hits_over_tool_output_noise(self, engine):
        engine._store.append(
            "test-session",
            {"role": "user", "content": "vendoring should stay generic host support only"},
        )
        engine._store.append(
            "test-session",
            {"role": "tool", "content": '{"vendoring":"vendoring vendoring vendoring","payload":"generic host support"}'},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["role"] == "user"
        assert result["results"][1]["role"] == "tool"

    def test_handle_grep_relevance_prefers_user_over_newer_assistant_on_similar_match(self, engine):
        engine._store.append(
            "test-session",
            {"role": "user", "content": "external plugin host support should stay generic"},
        )
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "external plugin host support should stay generic"},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "external plugin host support", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["role"] == "user"
        assert result["results"][1]["role"] == "assistant"

    def test_handle_grep_relevance_does_not_let_weaker_user_hit_beat_stronger_assistant_hit(self, engine):
        engine._store.append(
            "test-session",
            {"role": "user", "content": "vendoring blah blah external blah host"},
        )
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "vendoring external host"},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring external host", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "message"
        assert result["results"][0]["role"] == "assistant"
        assert result["results"][1]["type"] == "message"
        assert result["results"][1]["role"] == "user"

    def test_handle_grep_relevance_still_surfaces_preferred_user_hit_from_large_same_rank_pool(self, engine):
        engine._store.append(
            "test-session",
            {"role": "user", "content": "vendoring"},
        )
        for _ in range(150):
            engine._store.append(
                "test-session",
                {"role": "assistant", "content": "vendoring"},
            )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 5, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "message"
        assert result["results"][0]["role"] == "user"

    def test_handle_grep_relevance_prefers_assistant_over_tool_on_similar_match(self, engine):
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "plugin-only support should stay external and generic"},
        )
        engine._store.append(
            "test-session",
            {"role": "tool", "content": "plugin-only support should stay external and generic"},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "plugin-only", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["role"] == "assistant"
        assert result["results"][1]["role"] == "tool"

    def test_handle_grep_relevance_prefers_direct_hit_over_repetition_spam_for_single_term_query(self, engine):
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "query audit notes: vendoring vendoring vendoring vendoring vendoring"},
        )
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "Keep vendoring out of hermes-agent."},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["snippet"].startswith("Keep >>>vendoring<<< out")

    def test_handle_grep_relevance_prefers_direct_summary_hit_over_repetition_spam_summary(self, engine):
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Summary notes: vendoring vendoring vendoring vendoring vendoring",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Keep vendoring out of hermes-agent.",
                token_count=10,
                source_token_count=20,
                source_ids=[2],
                source_type="messages",
                created_at=1_699_999_000,
                earliest_at=1_699_999_000,
                latest_at=1_699_999_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Keep vendoring out of hermes-agent")

    def test_handle_grep_relevance_still_surfaces_direct_summary_when_single_term_matches_many_spammy_candidates(self, engine):
        for idx in range(150):
            engine._dag.add_node(
                SummaryNode(
                    session_id="test-session",
                    depth=0,
                    summary=f"Summary spam {idx}: vendoring vendoring vendoring vendoring vendoring",
                    token_count=10,
                    source_token_count=20,
                    source_ids=[idx + 1],
                    source_type="messages",
                    created_at=1_700_000_000 + idx,
                    earliest_at=1_700_000_000 + idx,
                    latest_at=1_700_000_000 + idx,
                )
            )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Keep vendoring out of hermes-agent.",
                token_count=10,
                source_token_count=20,
                source_ids=[999],
                source_type="messages",
                created_at=1_699_999_000,
                earliest_at=1_699_999_000,
                latest_at=1_699_999_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 5, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Keep vendoring out of hermes-agent")

    def test_handle_grep_relevance_still_surfaces_direct_phrase_summary_when_phrase_matches_many_spammy_candidates(self, engine):
        for idx in range(150):
            engine._dag.add_node(
                SummaryNode(
                    session_id="test-session",
                    depth=0,
                    summary=f"Summary spam {idx}: vendoring external vendoring external vendoring external status",
                    token_count=10,
                    source_token_count=20,
                    source_ids=[3000 + idx],
                    source_type="messages",
                    created_at=1_700_000_000 + idx,
                    earliest_at=1_700_000_000 + idx,
                    latest_at=1_700_000_000 + idx,
                )
            )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Keep vendoring external support plugin-only.",
                token_count=10,
                source_token_count=20,
                source_ids=[9999],
                source_type="messages",
                created_at=1_699_999_000,
                earliest_at=1_699_999_000,
                latest_at=1_699_999_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": '"vendoring external"', "limit": 5, "sort": "relevance"},
            )
        )

        assert any(
            item["type"] == "summary" and item["snippet"].startswith("Keep vendoring external support")
            for item in result["results"]
        )

    def test_handle_grep_relevance_prefers_direct_phrase_summary_over_repeated_phrase_with_varied_filler(self, engine):
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring external rollout checklist vendoring external support matrix vendoring external adapter notes",
                token_count=10,
                source_token_count=20,
                source_ids=[4100],
                source_type="messages",
                created_at=1_700_000_100,
                earliest_at=1_700_000_100,
                latest_at=1_700_000_100,
            )
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Keep vendoring external support plugin-only.",
                token_count=10,
                source_token_count=20,
                source_ids=[4101],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": '"vendoring external"', "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Keep vendoring external support")

    def test_handle_grep_relevance_prefers_direct_phrase_summary_over_repeated_phrase_with_richer_filler(self, engine):
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring external rollout checklist vendoring external support matrix vendoring external adapter integration notes",
                token_count=10,
                source_token_count=20,
                source_ids=[4110],
                source_type="messages",
                created_at=1_700_000_100,
                earliest_at=1_700_000_100,
                latest_at=1_700_000_100,
            )
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="Keep vendoring external support plugin-only.",
                token_count=10,
                source_token_count=20,
                source_ids=[4111],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": '"vendoring external"', "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Keep vendoring external support")

    def test_handle_grep_relevance_unmatched_quote_still_finds_results(self, engine):
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "Keep vendoring out of hermes-agent."},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": '"vendoring', "limit": 5, "sort": "relevance"},
            )
        )

        assert result["total_results"] == 1
        assert result["results"][0]["type"] == "message"
        # #168: the unmatched quote is sanitized and the FTS hit keeps markers.
        assert result["results"][0]["snippet"] == "Keep >>>vendoring<<< out of hermes-agent."

    def test_handle_grep_recency_same_timestamp_pool_matches_store_ordering(self, engine):
        engine._store.append_batch(
            "test-session",
            [
                {
                    "role": "assistant",
                    "content": f"alpha alpha alpha beta beta gamma gamma gamma spam {idx}",
                }
                for idx in range(120)
            ] + [
                {
                    "role": "assistant",
                    "content": "keep alpha beta gamma concise",
                }
            ],
        )

        store_results = engine._store.search("alpha beta gamma", session_id="test-session", limit=5, sort="recency")
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "alpha beta gamma", "limit": 5, "sort": "recency"},
            )
        )

        assert [item["type"] for item in result["results"]] == ["message"] * len(result["results"])
        assert [item["store_id"] for item in result["results"]] == [hit["store_id"] for hit in store_results]

    def test_handle_grep_hybrid_summary_only_matches_dag_order_for_future_timestamps(self, engine):
        now = time.time()
        future = now + (60 * 24 * 3600)
        future_node = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring",
                token_count=10,
                source_token_count=20,
                source_ids=[8001],
                source_type="messages",
                created_at=future,
                earliest_at=future,
                latest_at=future,
            )
        )
        current_node = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring",
                token_count=10,
                source_token_count=20,
                source_ids=[8002],
                source_type="messages",
                created_at=now,
                earliest_at=now,
                latest_at=now,
            )
        )

        dag_results = engine._dag.search("vendoring", session_id="test-session", limit=2, sort="hybrid")
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "hybrid"},
            )
        )

        assert [node.node_id for node in dag_results] == [future_node, current_node]
        assert [item["node_id"] for item in result["results"]] == [future_node, current_node]

    def test_handle_grep_hybrid_message_only_clamps_future_timestamps_consistently(self, engine):
        now = time.time()
        future = now + (60 * 24 * 3600)
        current_ids = [
            engine._store.append(
                "test-session",
                {"role": "assistant", "content": "vendoring external"},
            )
            for _ in range(20)
        ]
        future_id = engine._store.append(
            "test-session",
            {"role": "assistant", "content": "vendoring external"},
        )
        for current_id in current_ids:
            engine._store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?", (now, current_id))
        engine._store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?", (future, future_id))
        engine._store._conn.commit()

        store_results = engine._store.search("vendoring external", session_id="test-session", limit=1, sort="hybrid")
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring external", "limit": 1, "sort": "hybrid"},
            )
        )

        assert [hit["store_id"] for hit in store_results] == [future_id]
        assert [item["store_id"] for item in result["results"]] == [future_id]

    def test_handle_grep_relevance_prefers_much_better_summary_over_vague_user_hit(self, engine):
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "vendoring? maybe?"},
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=1,
                summary="Summary: keep hermes-lcm external and never vendor it into hermes-agent. generic host support only.",
                token_count=20,
                source_token_count=40,
                source_ids=[store_id],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "never vendor hermes-agent", "limit": 2, "sort": "relevance"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Summary: keep hermes-lcm external")
        # #168: the indexed conjunction excludes the vague partial message hit.
        assert result["total_results"] == 1
        assert [item["type"] for item in result["results"]] == ["summary"]

    def test_handle_grep_hybrid_prefers_much_better_summary_over_vague_recent_user_hit(self, engine):
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "vendoring? maybe?"},
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=1,
                summary="Summary: keep hermes-lcm external and never vendor it into hermes-agent. generic host support only.",
                token_count=20,
                source_token_count=40,
                source_ids=[store_id],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "never vendor hermes-agent", "limit": 2, "sort": "hybrid"},
            )
        )

        assert result["results"][0]["type"] == "summary"
        assert result["results"][0]["snippet"].startswith("Summary: keep hermes-lcm external")
        # #168: hybrid inherits the FTS arm's conjunctive filtering.
        assert result["total_results"] == 1
        assert [item["type"] for item in result["results"]] == ["summary"]

    def test_handle_grep_hybrid_does_not_let_weak_summary_beat_stronger_message_hit(self, engine):
        engine._store.append(
            "test-session",
            {"role": "assistant", "content": "Keep vendoring out of hermes-agent."},
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring vendoring vendoring vendoring vendoring",
                token_count=10,
                source_token_count=20,
                source_ids=[1],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "hybrid"},
            )
        )

        assert result["results"][0]["type"] == "message"
        assert result["results"][0]["role"] == "assistant"
        assert result["results"][1]["type"] == "summary"

    def test_handle_grep_recency_preserves_message_ordering_for_same_timestamp_hits(self, engine):
        ids = engine._store.append_batch(
            "test-session",
            [
                {"role": "user", "content": "vendoring"},
                {"role": "assistant", "content": "vendoring vendoring vendoring vendoring vendoring"},
            ],
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id IN (?, ?)",
            (1_700_000_000, ids[0], ids[1]),
        )
        engine._store._conn.commit()

        store_hits = engine._store.search("vendoring", session_id="test-session", limit=2, sort="recency")
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "recency"},
            )
        )

        assert [hit["role"] for hit in store_hits] == ["user", "assistant"]
        assert [item["role"] for item in result["results"]] == ["user", "assistant"]

    def test_handle_grep_recency_prefers_message_over_weaker_summary_at_same_timestamp(self, engine):
        store_id = engine._store.append(
            "test-session",
            {"role": "user", "content": "Keep vendoring external support clean."},
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1_700_000_000, store_id),
        )
        engine._store._conn.commit()
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="vendoring vendoring vendoring",
                token_count=10,
                source_token_count=20,
                source_ids=[store_id],
                source_type="messages",
                created_at=1_700_000_000,
                earliest_at=1_700_000_000,
                latest_at=1_700_000_000,
            )
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "vendoring", "limit": 2, "sort": "recency"},
            )
        )

        assert result["results"][0]["type"] == "message"
        assert result["results"][0]["role"] == "user"
        assert result["results"][1]["type"] == "summary"



class TestHandleGrepCrossSession:
    """Cross-session search via session_scope=all|session and the new filters."""

    def _seed_two_sessions(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "docker plan current"})
        engine._store.append("test-session", {"role": "assistant", "content": "docker plan current reply"})
        engine._store.append("old-session", {"role": "user", "content": "docker plan old"}, source="discord")

    def test_session_scope_all_returns_cross_session_messages(self, engine):
        self._seed_two_sessions(engine)
        result = json.loads(
            engine.handle_tool_call("lcm_grep", {"query": "docker", "session_scope": "all"})
        )
        assert result["session_scope"] == "all"
        sessions_seen = {hit["session_id"] for hit in result["results"]}
        assert sessions_seen == {"test-session", "old-session"}
        for hit in result["results"]:
            assert "from_current_session" in hit
            assert hit["from_current_session"] == (hit["session_id"] == "test-session")
            assert "timestamp" in hit
            assert hit["timestamp"] >= 0

    def test_excludes_explicit_and_current_sessions_from_full_text_candidates(self, engine):
        self._seed_two_sessions(engine)
        engine._store.append(
            "kept-session",
            {"role": "user", "content": "docker plan retained"},
        )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "docker",
                    "session_scope": "all",
                    "exclude_current_session": True,
                    "exclude_session_ids": ["old-session"],
                },
            )
        )

        assert [hit["session_id"] for hit in result["results"]] == ["kept-session"]

    def test_excludes_sessions_before_the_full_text_candidate_cap(self, engine):
        engine._store.append(
            "kept-session",
            {"role": "user", "content": "docker"},
        )
        for _ in range(20):
            engine._store.append(
                "crowding-session",
                {"role": "user", "content": "docker"},
            )

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "docker",
                    "session_scope": "all",
                    "exclude_session_ids": ["crowding-session"],
                    "limit": 1,
                },
            )
        )

        assert [hit["session_id"] for hit in result["results"]] == ["kept-session"]

    @pytest.mark.parametrize("mode", ["semantic", "hybrid"])
    def test_excludes_current_session_from_semantic_candidates(
        self, engine, monkeypatch, mode
    ):
        from hermes_lcm.vector_store import VectorStore

        engine._config.embeddings_enabled = True
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="semantic docker plan",
                token_count=10,
                source_token_count=10,
                source_ids=[],
                source_type="messages",
                created_at=1.0,
            )
        )

        class Provider:
            provider_id = "mock"
            model_id = "mock-model"

            def embed_query(self, _query):
                return [1.0, 0.0]

        monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: Provider())
        vector_store = VectorStore(engine._store.db_path, config=engine._config)
        try:
            vector_store.register_profile("mock-model", "mock", 2)
            identity = vector_store.capture_identity("mock-model", provider="mock")
            vector_store.record_embedding(
                str(node_id),
                "summary",
                "mock-model",
                [1.0, 0.0],
                identity=identity,
            )
        finally:
            vector_store.close()

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "docker",
                    "mode": mode,
                    "exclude_current_session": True,
                },
            )
        )

        assert result["total_results"] == 0
        assert result["results"] == []
        assert result["degraded_to_fts"] is False

    def test_excluding_the_only_explicit_scope_returns_an_empty_bounded_result(self, engine):
        self._seed_two_sessions(engine)

        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "docker",
                    "session_scope": "session",
                    "session_id": "old-session",
                    "exclude_session_ids": ["old-session"],
                    "limit": 5,
                },
            )
        )

        assert result["limit"] == 5
        assert result["total_results"] == 0
        assert result["results"] == []

    def test_rejects_non_list_session_exclusions(self, engine):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "exclude_session_ids": "old-session"},
            )
        )

        assert result == {"error": "exclude_session_ids must be an array of strings"}

    @pytest.mark.parametrize("invalid_id", ["", "   ", 7, None])
    def test_rejects_invalid_session_exclusion_entries(self, engine, invalid_id):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "exclude_session_ids": [invalid_id]},
            )
        )

        assert result == {
            "error": "exclude_session_ids must contain only non-empty strings"
        }

    def test_rejects_non_boolean_current_session_exclusion(self, engine):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "exclude_current_session": "false"},
            )
        )

        assert result == {"error": "exclude_current_session must be a boolean"}

    def test_session_scope_session_restricts_to_explicit_id(self, engine):
        self._seed_two_sessions(engine)
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "session", "session_id": "old-session"},
            )
        )
        assert result["session_scope"] == "session"
        assert result["session_id"] == "old-session"
        assert result["total_results"] == 1
        assert result["results"][0]["session_id"] == "old-session"
        assert result["results"][0]["from_current_session"] is False

    def test_session_scope_session_without_session_id_returns_error(self, engine):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "session"},
            )
        )
        assert "error" in result
        assert "session_id" in result["error"]

    def test_session_scope_current_with_session_id_returns_error(self, engine):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "current", "session_id": "old-session"},
            )
        )
        assert "error" in result
        assert "session_id is only valid with session_scope=session" in result["error"]

    def test_session_scope_all_with_session_id_returns_error(self, engine):
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "all", "session_id": "old-session"},
            )
        )
        assert "error" in result

    def test_limit_clamped_at_hard_cap(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "docker once"})
        result = json.loads(
            engine.handle_tool_call("lcm_grep", {"query": "docker", "limit": 5000})
        )
        assert result["limit"] == 200
        assert result["limit_clamped_from"] == 5000

    def test_limit_zero_returns_error(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "docker plan"})
        result = json.loads(
            engine.handle_tool_call("lcm_grep", {"query": "docker", "limit": 0})
        )
        assert "error" in result
        assert "limit" in result["error"]

    def test_limit_negative_returns_error(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "docker plan"})
        result = json.loads(
            engine.handle_tool_call("lcm_grep", {"query": "docker", "limit": -5})
        )
        assert "error" in result

    def test_empty_engine_session_with_unknown_scope_does_not_leak(self, engine):
        # Regression: unknown session_scope previously fell through to engine._session_id
        # and returned multi-session rows when the engine was unbound. The fix in #104
        # makes empty session_id a literal scoped filter at the data layer; the unknown-
        # scope fallback now routes through current-session and naturally returns zero
        # results instead of leaking. Mirrors the maintainer's repro from PR #102 review.
        engine._session_id = ""
        engine._store.append("session-a", {"role": "user", "content": "docker from a"})
        engine._store.append("session-b", {"role": "user", "content": "docker from b"})
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "bogus", "limit": 10},
            )
        )
        assert result["session_scope"] == "current"
        assert result["ignored_session_scope"] == "bogus"
        assert result["total_results"] == 0
        assert result["results"] == []

    def test_cross_session_scope_returns_only_message_hits(self, engine):
        # Cross-session scope intentionally restricts to raw-message hits.
        # Summary nodes from foreign sessions are excluded entirely (deferred
        # until a real cross-session DAG-expansion contract exists).
        engine._store.append("old-session", {"role": "user", "content": "docker old message"})
        engine._dag.add_node(
            SummaryNode(
                session_id="old-session",
                depth=0,
                summary="docker old summary",
                token_count=5,
                source_token_count=5,
                source_ids=[1],
                source_type="messages",
                created_at=time.time(),
            )
        )
        result = json.loads(
            engine.handle_tool_call("lcm_grep", {"query": "docker", "session_scope": "all"})
        )
        types_seen = {hit["type"] for hit in result["results"]}
        assert "message" in types_seen
        assert "summary" not in types_seen
        # No summary hits means no cross_session_expand_supported marker is needed.
        for hit in result["results"]:
            assert "cross_session_expand_supported" not in hit

    def test_current_scope_still_returns_summary_hits(self, engine):
        # Regression: removing cross-session summary hits must not affect
        # current-session DAG search behavior.
        engine._store.append("test-session", {"role": "user", "content": "docker current"})
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="docker current summary",
                token_count=5,
                source_token_count=5,
                source_ids=[1],
                source_type="messages",
                created_at=time.time(),
            )
        )
        result = json.loads(engine.handle_tool_call("lcm_grep", {"query": "docker"}))
        types_seen = {hit["type"] for hit in result["results"]}
        assert "summary" in types_seen

    def test_source_filter_combined_with_scope_all(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "docker via cli"}, source="cli")
        engine._store.append("old-session", {"role": "user", "content": "docker via discord"}, source="discord")
        result = json.loads(
            engine.handle_tool_call(
                "lcm_grep",
                {"query": "docker", "session_scope": "all", "source": "discord"},
            )
        )
        assert result["session_scope"] == "all"
        assert result["source"] == "discord"
        assert result["total_results"] == 1
        assert result["results"][0]["session_id"] == "old-session"
        assert result["results"][0]["source"] == "discord"

    def test_default_scope_preserves_historical_behavior(self, engine):
        # Omitting session_scope must behave identically to current.
        engine._store.append("test-session", {"role": "user", "content": "docker default"})
        engine._store.append("old-session", {"role": "user", "content": "docker old"})
        result = json.loads(engine.handle_tool_call("lcm_grep", {"query": "docker"}))
        assert result["session_scope"] == "current"
        sessions_seen = {hit["session_id"] for hit in result["results"]}
        assert sessions_seen == {"test-session"}



class TestHandleGrepExternalizedPayloads:
    def _externalize(self, engine, content, tool_call_id="call-search"):
        before = set(Path(engine._hermes_home, "lcm-large-outputs").glob("*.json"))
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        ])
        created = set(Path(engine._hermes_home, "lcm-large-outputs").glob("*.json")) - before
        return next(path.name for path in created)

    def _write_payload_with_created_at(self, engine, ref, created_at):
        storage = Path(engine._hermes_home, "lcm-large-outputs")
        storage.mkdir(parents=True, exist_ok=True)
        (storage / ref).write_text(
            '{"kind":"tool_result","role":"tool","session_id":"test-session",'
            '"tool_call_id":"call-created-at","content_chars":14,"content_bytes":14,'
            f'"created_at":{created_at},"content":"needle payload"}}',
            encoding="utf-8",
        )
        return ref

    def _write_payload_with_content_size(self, engine, ref, size_field):
        storage = Path(engine._hermes_home, "lcm-large-outputs")
        storage.mkdir(parents=True, exist_ok=True)
        content = "needle payload"
        sizes = {
            "content_chars": str(len(content)),
            "content_bytes": str(len(content.encode("utf-8"))),
        }
        sizes[size_field] = ("9" * 400) + ".0"
        (storage / ref).write_text(
            '{"kind":"tool_result","role":"tool","session_id":"test-session",'
            '"tool_call_id":"call-content-size",'
            f'"content_chars":{sizes["content_chars"]},'
            f'"content_bytes":{sizes["content_bytes"]},'
            f'"created_at":1.0,"content":{json.dumps(content)}}}',
            encoding="utf-8",
        )
        return ref

    def _write_payload_with_nested_content(self, engine, ref):
        storage = Path(engine._hermes_home, "lcm-large-outputs")
        storage.mkdir(parents=True, exist_ok=True)
        content = "real payload target"
        payload = {
            "metadata": {
                "session_id": "foreign-session",
                "content": "nested decoy target",
            },
            "kind": "tool_result",
            "role": "tool",
            "session_id": "test-session",
            "tool_call_id": "call-nested-content",
            "content": content,
            "content_chars": len(content),
            "content_bytes": len(content.encode("utf-8")),
            "created_at": 1.0,
        }
        (storage / ref).write_text(json.dumps(payload), encoding="utf-8")
        return ref, content
    def test_default_history_scope_does_not_scan_sidecars(self, externalized_search_engine):
        self._externalize(externalized_search_engine, "private external needle " * 20)

        result = json.loads(externalized_search_engine.handle_tool_call("lcm_grep", {"query": "needle"}))

        assert result["content_scope"] == "history"
        assert result["total_results"] == 0
        assert "externalized_scan" not in result

    def test_externalized_scope_returns_bounded_recoverable_match(self, externalized_search_engine):
        content = "first line\nsecond needle line\n" + ("tail " * 100)
        ref = self._externalize(externalized_search_engine, content)

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep", {"query": "needle", "content_scope": "externalized"}
            )
        )

        assert result["total_results"] == 1
        hit = result["results"][0]
        assert hit["type"] == "externalized"
        assert hit["ref"] == ref
        assert hit["tool_call_id"] == "call-search"
        assert hit["line"] == 2
        assert hit["byte_position"] == len("first line\nsecond ".encode("utf-8"))
        assert hit["original_content_bytes"] == len(content.encode("utf-8"))
        assert hit["scan_truncated"] is False
        recovered = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_expand", {"externalized_ref": ref, "max_tokens": 100_000}
            )
        )
        assert recovered["content"] == content

    @pytest.mark.parametrize(
        ("label", "created_at"),
        [
            ("oversized-integer", "9" * 401),
            ("non-finite-decimal", ("9" * 400) + ".0"),
        ],
    )
    def test_explicit_ref_search_ignores_unrepresentable_created_at(
        self,
        externalized_search_engine,
        label,
        created_at,
    ):
        ref = self._write_payload_with_created_at(
            externalized_search_engine,
            f"explicit-{label}.json",
            created_at,
        )
        valid_ref = self._write_payload_with_created_at(
            externalized_search_engine,
            f"explicit-valid-{label}.json",
            "1.0",
        )

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "needle",
                    "content_scope": "externalized",
                    "externalized_refs": [ref, valid_ref],
                    "sort": "recency",
                },
            )
        )

        assert [item["ref"] for item in result["results"]] == [valid_ref, ref]
        assert result["externalized_scan"]["candidate_files"] == 2
        assert result["externalized_scan"]["scanned_files"] == 2
        assert result["externalized_scan"]["matched_files"] == 2

    @pytest.mark.parametrize(
        ("label", "created_at"),
        [
            ("oversized-integer", "9" * 401),
            ("non-finite-decimal", ("9" * 400) + ".0"),
        ],
    )
    def test_auto_discovery_ignores_unrepresentable_created_at(
        self,
        externalized_search_engine,
        label,
        created_at,
    ):
        ref = self._write_payload_with_created_at(
            externalized_search_engine,
            f"auto-{label}.json",
            created_at,
        )
        valid_ref = self._write_payload_with_created_at(
            externalized_search_engine,
            f"auto-valid-{label}.json",
            "1.0",
        )

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "needle",
                    "content_scope": "externalized",
                    "sort": "recency",
                },
            )
        )

        assert [item["ref"] for item in result["results"]] == [valid_ref, ref]
        assert result["externalized_scan"]["candidate_files"] == 2
        assert result["externalized_scan"]["scanned_files"] == 2
        assert result["externalized_scan"]["matched_files"] == 2

    @pytest.mark.parametrize("explicit_refs", [False, True], ids=["auto", "explicit"])
    @pytest.mark.parametrize("size_field", ["content_bytes", "content_chars"])
    def test_externalized_search_ignores_non_finite_content_size(
        self,
        externalized_search_engine,
        explicit_refs,
        size_field,
    ):
        ref = self._write_payload_with_content_size(
            externalized_search_engine,
            f"non-finite-{size_field}.json",
            size_field,
        )
        args: dict[str, object] = {"query": "needle", "content_scope": "externalized"}
        if explicit_refs:
            args["externalized_refs"] = [ref]

        result = json.loads(externalized_search_engine.handle_tool_call("lcm_grep", args))

        assert [item["ref"] for item in result["results"]] == [ref]
        assert result["results"][0][f"original_{size_field}"] is None
        assert result["externalized_scan"]["scanned_files"] == 1
        assert result["externalized_scan"]["matched_files"] == 1

    @pytest.mark.parametrize("explicit_refs", [False, True], ids=["auto", "explicit"])
    def test_externalized_search_uses_top_level_content_and_remains_expandable(
        self,
        externalized_search_engine,
        explicit_refs,
    ):
        ref, content = self._write_payload_with_nested_content(
            externalized_search_engine,
            f"nested-content-{'explicit' if explicit_refs else 'auto'}.json",
        )
        args: dict[str, object] = {"query": "real payload", "content_scope": "externalized"}
        if explicit_refs:
            args["externalized_refs"] = [ref]

        matched = json.loads(externalized_search_engine.handle_tool_call("lcm_grep", args))
        decoy = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    **args,
                    "query": "nested decoy",
                },
            )
        )
        expanded = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_expand",
                {"externalized_ref": ref, "max_tokens": 100_000},
            )
        )

        assert [item["ref"] for item in matched["results"]] == [ref]
        assert matched["results"][0]["snippet"] == content
        assert decoy["total_results"] == 0
        assert expanded["content"] == content
    def test_both_scope_combines_history_and_payload_hits(self, externalized_search_engine):
        externalized_search_engine._store.append(
            "test-session", {"role": "user", "content": "combined needle in history"}
        )
        self._externalize(externalized_search_engine, "combined needle in payload " * 20)

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep", {"query": "needle", "content_scope": "both"}
            )
        )

        assert {item["type"] for item in result["results"]} == {"message", "externalized"}

    @pytest.mark.parametrize("sort", ["relevance", "hybrid"])
    def test_both_scope_preserves_message_only_ordering(self, externalized_search_engine, sort):
        for role, content in (
            ("user", "needle alpha"),
            ("assistant", "needle needle beta"),
            ("tool", "needle gamma"),
        ):
            externalized_search_engine._store.append(
                "test-session",
                {"role": role, "content": content},
            )

        history = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {"query": "needle", "content_scope": "history", "sort": sort},
            )
        )
        self._externalize(externalized_search_engine, "needle in payload " * 20)
        combined = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {"query": "needle", "content_scope": "both", "sort": sort},
            )
        )

        combined_messages = [item for item in combined["results"] if item["type"] == "message"]
        assert json.dumps(combined_messages, ensure_ascii=False) == json.dumps(
            history["results"],
            ensure_ascii=False,
        )

    @pytest.mark.parametrize("sort", ["relevance", "hybrid", "recency"])
    def test_combined_sort_tiers_externalized_native_rank(self, sort):
        message = {
            "type": "message",
            "role": "user",
            "_sort_rank": 50.0,
            "_sort_ts": 1.0,
        }
        externalized = {
            "type": "externalized",
            "_sort_rank": 0,
            "_sort_ts": 2.0,
        }

        ordered = sorted(
            [externalized, message],
            key=lambda item: lcm_tools._combined_result_sort_key(item, sort),
        )

        assert ordered == [message, externalized]

    def test_auto_discovery_bounds_directory_iteration_before_sorting(
        self,
        externalized_search_engine,
        monkeypatch,
    ):
        storage = Path(externalized_search_engine._hermes_home, "lcm-large-outputs")
        real_iterdir = Path.iterdir
        entries_consumed = 0
        refs_probed = []

        def many_payloads(path):
            nonlocal entries_consumed
            if path != storage:
                yield from real_iterdir(path)
                return
            for index in range(10_000):
                entries_consumed += 1
                suffix = ".json" if index % 2 else ".tmp"
                yield storage / f"payload-{index:05d}{suffix}"

        def reject_probe(_engine, ref, _session_id, **_kwargs):
            refs_probed.append(ref)
            return {"readable": False, "error": "missing"}

        monkeypatch.setattr(Path, "iterdir", many_payloads)
        monkeypatch.setattr(lcm_tools, "_inspect_externalized_payload_metadata", reject_probe)

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {"query": "needle", "content_scope": "externalized"},
            )
        )

        assert entries_consumed == lcm_tools._LCM_GREP_EXTERNALIZED_DISCOVERY_CAP + 1
        assert len(refs_probed) == lcm_tools._LCM_GREP_EXTERNALIZED_DISCOVERY_CAP // 2
        assert result["externalized_scan"]["discovery_files"] == len(refs_probed)
        assert result["externalized_scan"]["discovery_truncated"] is True
        assert result["externalized_scan"]["candidate_files"] == 0
    def test_auto_discovery_filters_session_before_candidate_cap(self, externalized_search_engine):
        owned_refs = {
            self._externalize(
                externalized_search_engine,
                f"owned needle {index} " * 20,
                f"call-owned-{index}",
            )
            for index in range(2)
        }
        storage = Path(externalized_search_engine._hermes_home, "lcm-large-outputs")
        foreign_payload = {
            "kind": "tool_result",
            "role": "tool",
            "session_id": "foreign-session",
            "tool_call_id": "call-foreign",
            "content": "foreign needle",
            "content_chars": len("foreign needle"),
            "content_bytes": len("foreign needle".encode()),
            "created_at": 1.0,
        }
        for index in range(257):
            (storage / f"zzzz-foreign-{index:03d}.json").write_text(
                json.dumps(foreign_payload),
                encoding="utf-8",
            )
        first_256 = sorted(
            (path.name for path in storage.glob("*.json")),
            reverse=True,
        )[:256]
        assert owned_refs.isdisjoint(first_256)

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {"query": "needle", "content_scope": "externalized"},
            )
        )

        assert {item["ref"] for item in result["results"]} == owned_refs
        assert result["externalized_scan"]["candidate_files"] == len(owned_refs)
        assert result["externalized_scan"]["rejected_session_mismatch"] == 257

    def test_auto_discovery_accepts_owned_payload_with_long_precontent_metadata(
        self,
        externalized_search_engine,
    ):
        written = externalize_ingest_payload(
            "needle payload",
            role="user",
            session_id=externalized_search_engine.current_session_id,
            field_path="x" * 20_000,
            config=externalized_search_engine._config,
            hermes_home=externalized_search_engine._hermes_home,
        )
        assert written is not None
        ref = written["path"].name
        metadata = lcm_tools._inspect_externalized_payload_metadata(
            externalized_search_engine,
            ref,
            externalized_search_engine.current_session_id,
        )
        assert metadata == {"readable": False, "error": "metadata_prefix_truncated"}

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {"query": "needle", "content_scope": "externalized"},
            )
        )

        assert [item["ref"] for item in result["results"]] == [ref]

    def test_auto_discovery_keeps_malformed_symlink_and_foreign_payloads_fail_closed(
        self,
        externalized_search_engine,
    ):
        valid_ref = self._externalize(externalized_search_engine, "valid needle " * 20)
        storage = Path(externalized_search_engine._hermes_home, "lcm-large-outputs")
        (storage / "malformed.json").write_text(
            'not-json "session_id": "test-session", "content": "needle"',
            encoding="utf-8",
        )
        (storage / "foreign.json").write_text(
            json.dumps({"session_id": "foreign-session", "content": "needle"}),
            encoding="utf-8",
        )
        (storage / "linked.json").symlink_to(storage / valid_ref)

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {"query": "needle", "content_scope": "externalized"},
            )
        )

        assert [item["ref"] for item in result["results"]] == [valid_ref]
        assert result["externalized_scan"]["rejected_invalid_or_unreadable"] == 1
        assert result["externalized_scan"]["rejected_session_mismatch"] == 1
        assert result["externalized_scan"]["rejected_symlink"] == 1
    def test_explicit_refs_filter_and_rejects_cross_session_payload(self, externalized_search_engine):
        current_ref = self._externalize(externalized_search_engine, "current needle " * 20, "call-current")
        externalized_search_engine._session_id = "other-session"
        other_ref = self._externalize(externalized_search_engine, "other needle " * 20, "call-other")
        externalized_search_engine._session_id = "test-session"

        filtered = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "needle",
                    "content_scope": "externalized",
                    "externalized_refs": [current_ref],
                },
            )
        )
        rejected = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "needle",
                    "content_scope": "externalized",
                    "externalized_refs": [other_ref],
                },
            )
        )

        assert [item["ref"] for item in filtered["results"]] == [current_ref]
        assert "not owned by the active session" in rejected["error"]

    def test_explicit_refs_reject_invalid_and_symlink_refs(self, externalized_search_engine):
        invalid = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "needle",
                    "content_scope": "externalized",
                    "externalized_refs": ["../escape.json"],
                },
            )
        )
        ref = self._externalize(externalized_search_engine, "needle target " * 20)
        storage = Path(externalized_search_engine._hermes_home, "lcm-large-outputs")
        link = storage / "linked.json"
        link.symlink_to(storage / ref)
        symlink = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "needle",
                    "content_scope": "externalized",
                    "externalized_refs": [link.name],
                },
            )
        )

        assert "Invalid externalized ref" in invalid["error"]
        assert "symlink" in symlink["error"]

    @pytest.mark.parametrize(
        "payload_text",
        [
            '{"session_id":"test-session","content":"needle"',
            '{"session_id":"test-session","content":"needle"} trailing',
        ],
        ids=["truncated-tail", "trailing-garbage"],
    )
    def test_explicit_refs_reject_malformed_payloads_that_expand_cannot_load(
        self,
        externalized_search_engine,
        payload_text,
    ):
        storage = Path(externalized_search_engine._hermes_home, "lcm-large-outputs")
        storage.mkdir(parents=True, exist_ok=True)
        ref = "malformed-explicit.json"
        (storage / ref).write_text(payload_text, encoding="utf-8")

        searched = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "needle",
                    "content_scope": "externalized",
                    "externalized_refs": [ref],
                },
            )
        )
        expanded = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_expand",
                {"externalized_ref": ref},
            )
        )

        assert "error" in searched
        assert "not readable" in searched["error"]
        assert "not found" in expanded["error"]

    @pytest.mark.parametrize("sort", ["relevance", "hybrid"])
    def test_explicit_payload_limit_applies_after_native_byte_position_ordering(
        self,
        externalized_search_engine,
        sort,
    ):
        late_match_ref = self._externalize(
            externalized_search_engine,
            ("prefix " * 40) + "needle late",
            "call-late-match",
        )
        early_match_ref = self._externalize(
            externalized_search_engine,
            "needle early " + ("tail " * 40),
            "call-early-match",
        )

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "needle",
                    "content_scope": "externalized",
                    "externalized_refs": [late_match_ref, early_match_ref],
                    "sort": sort,
                    "limit": 1,
                },
            )
        )

        assert result["results"][0]["ref"] == early_match_ref
        assert result["results"][0]["byte_position"] == 0
        assert result["externalized_scan"]["matched_files"] == 2

    def test_externalized_storage_containment_failure_returns_structured_error(
        self,
        externalized_search_engine,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setenv("LCM_HERMES_BASE_DIR", str(tmp_path / "different-base"))

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {"query": "needle", "content_scope": "externalized"},
            )
        )

        assert "error" in result
        assert "Externalized payload storage is unavailable" in result["error"]

    def test_scan_never_reaches_content_after_512000_encoded_bytes(
        self,
        externalized_search_engine,
        monkeypatch,
    ):
        content = ("a" * 520_000) + " unreachable-needle"
        ref = self._externalize(externalized_search_engine, content)
        monkeypatch.setattr(
            lcm_tools,
            "load_externalized_payload",
            lambda *args, **kwargs: pytest.fail("grep validation must not deserialize full payloads"),
        )

        missed = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "unreachable-needle",
                    "content_scope": "externalized",
                    "externalized_refs": [ref],
                },
            )
        )
        bounded_hit = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "aaaa",
                    "content_scope": "externalized",
                    "externalized_refs": [ref],
                },
            )
        )

        assert missed["total_results"] == 0
        assert bounded_hit["results"][0]["scan_truncated"] is True
        assert bounded_hit["results"][0]["content_scanned_bytes"] == 512_000

    def test_explicit_ref_tail_validation_reads_only_a_bounded_window(
        self,
        externalized_search_engine,
        monkeypatch,
    ):
        ref = self._externalize(externalized_search_engine, ("a" * 2_000_000) + "\x00needle")
        payload_path = Path(externalized_search_engine._hermes_home, "lcm-large-outputs", ref)
        metadata_prefix, content_key_seen, prefix_truncated = (
            lcm_tools._read_externalized_payload_metadata_prefix(
                payload_path,
                max_read_bytes=lcm_tools._LCM_GREP_EXTERNALIZED_METADATA_READ_BYTES,
            )
        )
        assert content_key_seen is True
        assert prefix_truncated is False

        real_open = Path.open
        bytes_read = []

        class TrackedHandle:
            def __init__(self, handle):
                self._handle = handle

            def __enter__(self):
                self._handle.__enter__()
                return self

            def __exit__(self, *args):
                return self._handle.__exit__(*args)

            def seek(self, *args):
                return self._handle.seek(*args)

            def read(self, size=-1):
                assert 0 <= size <= lcm_tools._LCM_GREP_EXTERNALIZED_DOCUMENT_TAIL_BYTES + 1
                data = self._handle.read(size)
                bytes_read.append(len(data))
                return data

        def tracked_open(path, *args, **kwargs):
            handle = real_open(path, *args, **kwargs)
            return TrackedHandle(handle) if path == payload_path else handle

        monkeypatch.setattr(Path, "open", tracked_open)

        payload = lcm_tools._validate_externalized_payload_json_tail(payload_path, metadata_prefix)

        assert payload is not None
        assert payload["session_id"] == "test-session"
        assert sum(bytes_read) <= lcm_tools._LCM_GREP_EXTERNALIZED_DOCUMENT_TAIL_BYTES
    def test_externalized_scope_rejects_cross_session_search(self, externalized_search_engine):
        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {"query": "needle", "content_scope": "externalized", "session_scope": "all"},
            )
        )

        assert "session_scope=current only" in result["error"]

    @pytest.mark.parametrize("content_scope", ["externalized", "both"])
    @pytest.mark.parametrize(
        ("filter_name", "filter_args"),
        [
            ("role", {"role": "user"}),
            ("time_from", {"time_from": 0}),
            ("time_to", {"time_to": 4_102_444_800}),
            ("source", {"source": "discord"}),
            ("conversation_id", {"conversation_id": "lane-123"}),
        ],
    )
    def test_raw_message_filters_omit_externalized_sidecar_hits(
        self,
        externalized_search_engine,
        content_scope,
        filter_name,
        filter_args,
    ):
        raw_store_id = externalized_search_engine._store.append(
            "test-session",
            {"role": "user", "content": "history needle from user"},
            source="discord",
            conversation_id="lane-123",
        )
        self._externalize(externalized_search_engine, "tool payload needle " * 20)

        result = json.loads(
            externalized_search_engine.handle_tool_call(
                "lcm_grep",
                {
                    "query": "needle",
                    "content_scope": content_scope,
                    **filter_args,
                },
            )
        )

        expected_store_ids = [raw_store_id] if content_scope == "both" else []
        assert [item["store_id"] for item in result["results"]] == expected_store_ids, filter_name
        assert result["externalized_results_omitted"] is True
        assert "externalized_scan" not in result
