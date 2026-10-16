"""LCM engine integration tests: filtering."""

import hashlib
import json


import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import externalize_ingest_payload
from hermes_lcm.tokens import count_messages_tokens

from .engine_helpers import (
    TestMessageFiltering as _MessageFilteringHelpers,
)


class TestMessageFiltering:
    _timeout_capable_regex_engine = _MessageFilteringHelpers.__dict__["_timeout_capable_regex_engine"]
    _make_engine = _MessageFilteringHelpers.__dict__["_make_engine"]

    def test_no_patterns_means_no_filtering(self, tmp_path):
        engine = self._make_engine(tmp_path, "lcm_msg_unset.db")
        messages = [
            {"role": "user", "content": "Cronjob Response: heartbeat"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "normal text"},
        ]
        engine._ingest_messages(messages)
        assert engine._store.get_session_count("user-123") == 3
        assert engine._ignored_message_count == 0

    def test_anchored_prefix_drops_matching_message(self, tmp_path):
        engine = self._make_engine(
            tmp_path, "lcm_msg_anchor.db",
            ignore_message_patterns=["^Cronjob Response:"],
            ignore_message_patterns_source="env",
        )
        messages = [
            {"role": "user", "content": "Cronjob Response: hermie heartbeat\n(job_id: abc)"},
            {"role": "user", "content": "can you check the database for me?"},
            {"role": "assistant", "content": "Sure, looking now."},
        ]
        active_replay = engine._ingest_messages(messages)

        assert "LCM active replay placeholder: message ignored" in str(active_replay[0].get("content", ""))
        assert "Cronjob Response:" not in str(active_replay[0].get("content", ""))

        stored = engine._store.get_session_messages("user-123")
        stored_contents = [row["content"] for row in stored]
        assert len(stored) == 2
        assert "Cronjob Response:" not in "\n".join(stored_contents)
        assert "can you check the database for me?" in stored_contents
        assert engine._ignored_message_count == 1

    def test_ignored_messages_do_not_feed_compaction_summaries(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_compaction.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )

        def echo_summary(**kwargs):
            return kwargs["text"] + "\nExpand for details about: ignored leak", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", echo_summary)
        messages = [
            {"role": "user", "content": "SECRET ignored backlog must not summarize " + "x" * 200},
            {"role": "user", "content": "fresh visible request"},
        ]

        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
        nodes = engine._dag.get_session_nodes("user-123")
        stored_contents = "\n".join(row["content"] for row in engine._store.get_session_messages("user-123"))

        assert "SECRET" not in stored_contents
        assert "SECRET" not in "\n".join(str(msg.get("content", "")) for msg in result)
        assert nodes == []
        assert engine._ignored_message_count == 1

    def test_ignored_backlog_is_filtered_before_auto_focus_derivation(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_focus.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            captured["focus_topic"] = kwargs["focus_topic"]
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        messages = [
            {"role": "user", "content": "SECRET ignored backlog must not become focus " + "x" * 200},
            {"role": "user", "content": "visible backlog objective " + "y" * 200},
            {"role": "assistant", "content": "fresh tail response"},
        ]

        engine.compress(messages, current_tokens=count_messages_tokens(messages))

        assert "visible backlog objective" in captured["text"]
        assert "SECRET" not in captured["text"]
        assert "visible backlog objective" in captured["focus_topic"]
        assert "SECRET" not in captured["focus_topic"]

    def test_ignored_rows_after_replayed_scaffolds_are_not_summarized(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_after_scaffold.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        messages = [
            {
                "role": "user",
                "content": "[Recent Summary (d0, node 1)]\nold scaffold\n[Expand for details: old scaffold]",
            },
            {"role": "user", "content": "SECRET ignored row after scaffold " + "x" * 200},
            {"role": "user", "content": "visible backlog objective " + "y" * 200},
            {"role": "assistant", "content": "fresh tail response"},
        ]

        engine.compress(messages, current_tokens=count_messages_tokens(messages))

        assert "visible backlog objective" in captured["text"]
        assert "SECRET" not in captured["text"]

    def test_ignored_backlog_is_not_preserved_as_objective_anchor(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_preserved_anchor.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10_000,
            ignore_message_patterns=["SECRET"],
        )
        messages = [
            {"role": "user", "content": "normal visible request"},
            {"role": "user", "content": "SECRET ignored objective must not be preserved"},
            {"role": "assistant", "content": "fresh tail response"},
        ]

        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
        result_text = "\n".join(str(msg.get("content", "")) for msg in result)

        assert "normal visible request" in result_text
        assert "fresh tail response" in result_text
        assert "SECRET" not in result_text
        assert "Current user objective preserved" not in result_text

    def test_preserved_objective_scaffold_does_not_survive_ignored_backlog_filtering(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_preserved_scaffold.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10_000,
            ignore_message_patterns=["SECRET"],
        )
        messages = [
            {
                "role": "user",
                "content": "[Current user objective preserved from compacted history]\ncarry this objective forward",
            },
            {"role": "user", "content": "SECRET ignored objective must not be preserved"},
            {"role": "user", "content": "fresh visible request"},
        ]

        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
        result_text = "\n".join(str(msg.get("content", "")) for msg in result)

        assert "carry this objective forward" not in result_text
        assert "fresh visible request" in result_text
        assert "SECRET" not in result_text

    def test_original_ignore_decision_survives_sensitive_active_redaction(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_sensitive_redaction.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        messages = [
            {"role": "user", "content": "api_key=sk-ignore...cdef ignored before active replay redaction " + "x" * 200},
            {"role": "user", "content": "visible backlog objective " + "y" * 200},
            {"role": "assistant", "content": "fresh tail response"},
        ]

        engine.compress(messages, current_tokens=count_messages_tokens(messages))

        assert "visible backlog objective" in captured["text"]
        assert "api_key" not in captured["text"]
        assert "sk-ignore" not in captured["text"]
        assert engine._ignored_message_count == 1

    def test_original_ignore_decision_survives_redacted_replay_next_turn(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_sensitive_replay.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        captured_texts: list[str] = []

        def capture_summary(**kwargs):
            captured_texts.append(kwargs["text"])
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        first_result = engine.compress(
            [
                {"role": "user", "content": "visible backlog objective " + "y" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
            ],
            current_tokens=10_000,
        )
        first_result_text = "\n".join(str(msg.get("content", "")) for msg in first_result)

        assert "LCM active replay placeholder: message ignored" in first_result_text
        assert "sk-ignore" not in first_result_text

        engine.compress(
            first_result + [{"role": "assistant", "content": "next fresh assistant turn"}],
            current_tokens=10_000,
        )

        assert len(captured_texts) == 1
        assert "visible backlog objective" in captured_texts[0]
        assert all("sk-ignore" not in text for text in captured_texts)
        assert all("LCM active replay placeholder: message ignored" not in text for text in captured_texts)

    def test_already_ingested_ignored_prefix_keeps_placeholder_when_new_turn_appends(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_cached_prefix_appended_turn.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        ignored = {"role": "user", "content": "api_key=sk-ignore...cdef ignored active turn"}

        first_replay = engine._ingest_messages([ignored])
        assert "LCM active replay placeholder: message ignored" in str(first_replay[0].get("content", ""))
        assert "sk-ignore" not in str(first_replay[0].get("content", ""))

        appended_replay = engine._ingest_messages(
            [ignored, {"role": "user", "content": "visible appended turn"}]
        )

        assert "LCM active replay placeholder: message ignored" in str(appended_replay[0].get("content", ""))
        assert "sk-ignore" not in str(appended_replay[0].get("content", ""))
        assert appended_replay[1]["content"] == "visible appended turn"

    def test_extraction_uses_same_ignored_dependency_filtered_view_as_summary(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_extraction_filtered_view.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
            extraction_enabled=True,
        )
        captured: dict[str, str] = {}

        def capture_extraction(**kwargs):
            captured["serialized"] = kwargs["serialized_messages"]

        def capture_summary(**kwargs):
            captured["summary_text"] = kwargs["text"]
            return "visible summary\n[Expand for details: visible]", 1

        monkeypatch.setattr(lcm_engine, "extract_before_compaction", capture_extraction)
        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        messages = [
            {"role": "user", "content": "SECRET ignored backlog must not extract " + "x" * 200},
            {"role": "assistant", "content": "dependent assistant reply must not extract"},
            {"role": "user", "content": "visible backlog should extract " + "y" * 200},
            {"role": "assistant", "content": "fresh tail"},
        ]

        engine.compress(messages, current_tokens=count_messages_tokens(messages))

        assert "visible backlog should extract" in captured["serialized"]
        assert "visible backlog should extract" in captured["summary_text"]
        assert "SECRET" not in captured["serialized"]
        assert "dependent assistant reply" not in captured["serialized"]
        assert "SECRET" not in captured["summary_text"]
        assert "dependent assistant reply" not in captured["summary_text"]

    def test_source_ids_exclude_historical_rows_ignored_by_current_filter(self, tmp_path, monkeypatch):
        db_path = tmp_path / "lcm_msg_ignore_source_ids_exclude_historical.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
            )
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        ignored_store_id = first._store.append(
            "session",
            {"role": "user", "content": "SECRET historical row must not be a source"},
        )
        visible_store_id = first._store.append(
            "session",
            {"role": "user", "content": "visible historical backlog " + "v" * 200},
        )
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=["SECRET"],
            )
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)

            def summary(**kwargs):
                assert "SECRET" not in kwargs["text"]
                assert "visible historical backlog" in kwargs["text"]
                return "visible summary\n[Expand for details: visible]", 1

            monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary)
            messages = [
                {"role": "user", "content": "SECRET historical row must not be a source"},
                {"role": "user", "content": "visible historical backlog " + "v" * 200},
                {"role": "assistant", "content": "fresh tail"},
            ]
            second.compress(messages, current_tokens=count_messages_tokens(messages))

            nodes = second._dag.get_session_nodes("session")
            assert nodes
            assert ignored_store_id not in nodes[0].source_ids
            assert visible_store_id in nodes[0].source_ids
        finally:
            second.shutdown()

    def test_source_ids_exclude_stored_externalized_rows_ignored_by_current_filter(self, tmp_path, monkeypatch):
        # #1016: the subject is an externalized user/assistant row, not the 100k floor.
        monkeypatch.setattr("hermes_lcm.ingest_protection._NON_TOOL_EXTERNALIZATION_FLOOR_CHARS", 0)
        db_path = tmp_path / "lcm_msg_ignore_externalized_source_ids_exclude.db"
        hermes_home = tmp_path / "hermes-externalized-ignore"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=50,
            ),
            hermes_home=str(hermes_home),
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        ignored_store_id = first._store.append(
            "session",
            {"role": "user", "content": "x" * 900 + " SECRET_PAYLOAD_MARKER externalized row " + "x" * 900},  # #1016: marker outside the stub's head/tail preview
        )
        stored_externalized_row = first._store.get(ignored_store_id)
        assert stored_externalized_row is not None
        assert "Externalized payload:" in stored_externalized_row["content"]
        assert "SECRET_PAYLOAD_MARKER" not in stored_externalized_row["content"]
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=["SECRET_PAYLOAD_MARKER"],
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=10_000,
            ),
            hermes_home=str(hermes_home),
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            assert second._latest_user_context_anchor([stored_externalized_row], []) is None
            captured: dict[str, str] = {}

            def summary(**kwargs):
                captured["text"] = kwargs["text"]
                return "visible summary\n[Expand for details: visible]", 1

            monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary)
            messages = [
                stored_externalized_row,
                {"role": "user", "content": "visible historical backlog " + "v" * 200},
                {"role": "assistant", "content": "fresh tail"},
            ]
            second.compress(messages, current_tokens=count_messages_tokens(messages))

            assert "visible historical backlog" in captured["text"]
            assert "SECRET_PAYLOAD_MARKER" not in captured["text"]
            assert "Externalized LCM ingest payload" not in captured["text"]
            nodes = second._dag.get_session_nodes("session")
            assert nodes
            assert ignored_store_id not in nodes[0].source_ids
        finally:
            second.shutdown()

    def test_preflight_filters_stored_externalized_rows_ignored_by_current_filter(self, tmp_path, monkeypatch):
        # #1016: the subject is an externalized user/assistant row, not the 100k floor.
        monkeypatch.setattr("hermes_lcm.ingest_protection._NON_TOOL_EXTERNALIZATION_FLOOR_CHARS", 0)
        db_path = tmp_path / "lcm_msg_ignore_externalized_preflight.db"
        hermes_home = tmp_path / "hermes-externalized-preflight"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10_000,
                context_threshold=0.01,
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=50,
            ),
            hermes_home=str(hermes_home),
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        ignored_store_id = first._store.append(
            "session",
            {"role": "user", "content": "x" * 900 + " SECRET_PAYLOAD_MARKER externalized row " + "x" * 900},  # #1016: marker outside the stub's head/tail preview
        )
        stored_externalized_row = first._store.get(ignored_store_id)
        assert stored_externalized_row is not None
        assert "Externalized payload:" in stored_externalized_row["content"]
        assert "SECRET_PAYLOAD_MARKER" not in stored_externalized_row["content"]
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10_000,
                context_threshold=0.01,
                ignore_message_patterns=["SECRET_PAYLOAD_MARKER"],
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=10_000,
            ),
            hermes_home=str(hermes_home),
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            messages = [
                stored_externalized_row,
                {"role": "assistant", "content": "fresh tail response " + "f" * 200},
            ]

            assert second.should_compress_preflight(messages) is True
            rows = second._store.get_session_messages("session")
            assert [row["store_id"] for row in rows] == [ignored_store_id, ignored_store_id + 1]
            assert rows[1]["content"].startswith("fresh tail response")

            result = second.compress(messages, current_tokens=count_messages_tokens(messages))
            result_text = "\n".join(str(msg.get("content", "")) for msg in result)
            assert "Externalized payload:" not in result_text
            assert "SECRET_PAYLOAD_MARKER" not in result_text
        finally:
            second.shutdown()

    def test_user_copied_externalized_placeholder_after_ignored_externalized_row_is_not_filtered(
        self, tmp_path, monkeypatch
    ):
        # #1055: force externalization for this fixture independently of the bound window.
        monkeypatch.setattr("hermes_lcm.ingest_protection._non_tool_text_over_floor", lambda text, config, **_: bool(text))
        db_path = tmp_path / "lcm_msg_ignore_externalized_literal_copy.db"
        hermes_home = tmp_path / "hermes-externalized-literal-copy"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=50,
            ),
            hermes_home=str(hermes_home),
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        ignored_store_id = first._store.append(
            "session",
            {"role": "user", "content": "SECRET_PAYLOAD_MARKER externalized row " + "x" * 200},
        )
        stored_externalized_row = first._store.get(ignored_store_id)
        assert stored_externalized_row is not None
        placeholder_literal = stored_externalized_row["content"]
        assert "Externalized payload:" in placeholder_literal
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=["SECRET_PAYLOAD_MARKER"],
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=10_000,
            ),
            hermes_home=str(hermes_home),
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            captured: dict[str, str] = {}

            def summary(**kwargs):
                captured["text"] = kwargs["text"]
                return "literal externalized placeholder summary\n[Expand for details: literal externalized placeholder]", 1

            monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary)
            messages = [
                {"role": "user", "content": placeholder_literal},
                {"role": "user", "content": "visible backlog objective " + "v" * 200},
                {"role": "assistant", "content": "fresh tail response"},
            ]
            second.compress(messages, current_tokens=count_messages_tokens(messages))

            rows = second._store.get_session_messages("session")
            assert rows[1]["content"].startswith("visible backlog objective")
            assert "Externalized payload:" not in captured["text"]
            assert "visible backlog objective" in captured["text"]
            assert "SECRET_PAYLOAD_MARKER" not in captured["text"]
            nodes = second._dag.get_session_nodes("session")
            assert nodes
            assert ignored_store_id not in nodes[0].source_ids
        finally:
            second.shutdown()

    def test_prior_externalized_placeholder_scan_pages_past_default_limit(self, tmp_path, monkeypatch):
        # #1055: force externalization for this fixture independently of the bound window.
        monkeypatch.setattr("hermes_lcm.ingest_protection._non_tool_text_over_floor", lambda text, config, **_: bool(text))
        db_path = tmp_path / "lcm_msg_ignore_externalized_prior_scan_pages.db"
        hermes_home = tmp_path / "hermes-externalized-prior-scan-pages"
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=50,
            ),
            hermes_home=str(hermes_home),
        )
        try:
            engine.on_session_start("session", platform="telegram", context_length=1000)
            for idx in range(10_000):
                engine._store.append(
                    "session",
                    {"role": "assistant", "content": f"filler row {idx}"},
                )
            prior_store_id = engine._store.append(
                "session",
                {"role": "user", "content": "SECRET_PAYLOAD_MARKER externalized row " + "x" * 200},
            )
            prior_row = engine._store.get(prior_store_id)
            assert prior_row is not None
            assert "Externalized payload:" in prior_row["content"]
            literal_copy = {"role": "user", "content": prior_row["content"]}

            assert engine._has_prior_raw_externalized_placeholder_row(
                prior_store_id + 1,
                literal_copy,
            ) is True
        finally:
            engine.shutdown()

    def test_duplicate_stored_externalized_rows_ignored_by_current_filter_are_all_filtered(
        self, tmp_path, monkeypatch
    ):
        # #1055: force externalization for this fixture independently of the bound window.
        monkeypatch.setattr("hermes_lcm.ingest_protection._non_tool_text_over_floor", lambda text, config, **_: bool(text))
        db_path = tmp_path / "lcm_msg_ignore_duplicate_externalized_replay.db"
        hermes_home = tmp_path / "hermes-duplicate-externalized-replay"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=50,
            ),
            hermes_home=str(hermes_home),
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        payload = "SECRET_PAYLOAD_MARKER duplicated externalized row " + "x" * 200
        first._store.append("session", {"role": "user", "content": payload})
        first._store.append("session", {"role": "user", "content": payload})
        rows = first._store.get_session_messages("session")
        assert len(rows) == 2
        assert rows[0]["content"] == rows[1]["content"]
        assert "Externalized payload:" in rows[0]["content"]
        placeholder_literal = rows[0]["content"]
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=["SECRET_PAYLOAD_MARKER"],
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=10_000,
            ),
            hermes_home=str(hermes_home),
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            captured: dict[str, str] = {}

            def summary(**kwargs):
                captured["text"] = kwargs["text"]
                return "visible summary\n[Expand for details: visible summary]", 1

            monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary)
            messages = [
                {"role": "user", "content": placeholder_literal},
                {"role": "user", "content": placeholder_literal},
                {"role": "user", "content": "visible backlog objective " + "v" * 200},
                {"role": "assistant", "content": "fresh tail response"},
            ]
            second.compress(messages, current_tokens=count_messages_tokens(messages))

            stored_after = second._store.get_session_messages("session")
            externalized_rows = [row for row in stored_after if row["content"].startswith("[Externalized payload:")]
            assert [row["store_id"] for row in externalized_rows] == [rows[0]["store_id"], rows[1]["store_id"]]
            assert "visible backlog objective" in captured["text"]
            assert "Externalized payload:" not in captured["text"]
            assert "SECRET_PAYLOAD_MARKER" not in captured["text"]
            nodes = second._dag.get_session_nodes("session")
            assert nodes
            assert rows[0]["store_id"] not in nodes[0].source_ids
            assert rows[1]["store_id"] not in nodes[0].source_ids
        finally:
            second.shutdown()

    def test_tool_call_externalized_payload_hidden_text_is_ignored_by_current_filter(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "lcm_msg_ignore_tool_call_externalized_payload.db"
        hermes_home = tmp_path / "hermes-tool-call-externalized-ignore"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                large_output_externalization_path=str(tmp_path / "externalized"),
            ),
            hermes_home=str(hermes_home),
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        externalized = externalize_ingest_payload(
            "DROP_TOOL_CALL_SECRET hidden argument " + "x" * 200,
            role="assistant",
            session_id="session",
            field_path="tool_calls[0].function.arguments",
            config=first._config,
            hermes_home=str(hermes_home),
        )
        assert externalized is not None
        ignored_store_id = first._store.append(
            "session",
            {
                "role": "assistant",
                "content": "visible assistant tool call",
                "tool_calls": [
                    {
                        "id": "call_secret",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": externalized["placeholder"]},
                    }
                ],
            },
        )
        stored_row = first._store.get(ignored_store_id)
        assert stored_row is not None
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=["DROP_TOOL_CALL_SECRET"],
                large_output_externalization_path=str(tmp_path / "externalized"),
            ),
            hermes_home=str(hermes_home),
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            captured: dict[str, str] = {}

            def summary(**kwargs):
                captured["text"] = kwargs["text"]
                return "visible summary\n[Expand for details: visible summary]", 1

            monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary)
            messages = [
                stored_row,
                {"role": "user", "content": "visible backlog objective " + "v" * 200},
                {"role": "assistant", "content": "fresh tail response"},
            ]
            second.compress(messages, current_tokens=count_messages_tokens(messages))

            assert "visible backlog objective" in captured["text"]
            assert "visible assistant tool call" not in captured["text"]
            assert "DROP_TOOL_CALL_SECRET" not in captured["text"]
            nodes = second._dag.get_session_nodes("session")
            assert nodes
            assert ignored_store_id not in nodes[0].source_ids
        finally:
            second.shutdown()

    def test_ignored_tool_call_externalized_payload_active_replay_drops_tool_refs(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_tool_call_externalized_active_replay.db"
        hermes_home = tmp_path / "hermes-tool-call-externalized-active-replay"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10_000,
                large_output_externalization_path=str(tmp_path / "externalized"),
            ),
            hermes_home=str(hermes_home),
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        externalized = externalize_ingest_payload(
            "DROP_TOOL_CALL_SECRET hidden argument " + "x" * 200,
            role="assistant",
            session_id="session",
            field_path="tool_calls[0].function.arguments",
            config=first._config,
            hermes_home=str(hermes_home),
        )
        assert externalized is not None
        ignored_store_id = first._store.append(
            "session",
            {
                "role": "assistant",
                "content": "visible assistant tool call",
                "tool_calls": [
                    {
                        "id": "call_secret",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": externalized["placeholder"]},
                    }
                ],
            },
        )
        stored_row = first._store.get(ignored_store_id)
        assert stored_row is not None
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10_000,
                ignore_message_patterns=["DROP_TOOL_CALL_SECRET"],
                large_output_externalization_path=str(tmp_path / "externalized"),
            ),
            hermes_home=str(hermes_home),
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            result = second.compress(
                [
                    stored_row,
                    {"role": "tool", "tool_call_id": "call_secret", "content": "tool result"},
                    {"role": "user", "content": "fresh visible request"},
                ],
                current_tokens=10_000,
            )
            result_text = json.dumps(result, ensure_ascii=False)
            assert "Externalized LCM ingest payload:" not in result_text
            assert "DROP_TOOL_CALL_SECRET" not in result_text
            assert all(not msg.get("tool_calls") for msg in result)
        finally:
            second.shutdown()

    def test_tool_call_multiple_externalized_payload_parts_are_matched_individually(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_tool_call_externalized_parts.db"
        hermes_home = tmp_path / "hermes-tool-call-externalized-parts"
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                large_output_externalization_path=str(tmp_path / "externalized"),
                ignore_message_patterns=[r"^DROP_TOOL_CALL_SECRET"],
            ),
            hermes_home=str(hermes_home),
        )
        try:
            engine.on_session_start("session", platform="telegram", context_length=1000)
            first_payload = externalize_ingest_payload(
                "benign hidden argument " + "b" * 100,
                role="assistant",
                session_id="session",
                field_path="tool_calls[0].function.arguments.first",
                config=engine._config,
                hermes_home=str(hermes_home),
            )
            second_payload = externalize_ingest_payload(
                "DROP_TOOL_CALL_SECRET second hidden argument " + "x" * 100,
                role="assistant",
                session_id="session",
                field_path="tool_calls[0].function.arguments.second",
                config=engine._config,
                hermes_home=str(hermes_home),
            )
            assert first_payload is not None and second_payload is not None
            row = {
                "session_id": "session",
                "role": "assistant",
                "content": "visible assistant tool call",
                "tool_calls": [
                    {
                        "id": "call_secret",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": json.dumps(
                                {
                                    "first": first_payload["placeholder"],
                                    "second": second_payload["placeholder"],
                                }
                            ),
                        },
                    }
                ],
            }
            assert engine._matches_ignore_message_patterns(row, stored_row=True) is True
        finally:
            engine.shutdown()

    def test_active_externalized_stub_without_store_id_is_filtered_after_ignore_added(self, tmp_path, monkeypatch):
        # #1055: force externalization for this fixture independently of the bound window.
        monkeypatch.setattr("hermes_lcm.ingest_protection._non_tool_text_over_floor", lambda text, config, **_: bool(text))
        db_path = tmp_path / "lcm_msg_ignore_externalized_active_stub.db"
        hermes_home = tmp_path / "hermes-externalized-active-stub"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10_000,
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=50,
            ),
            hermes_home=str(hermes_home),
        )
        try:
            first.on_session_start("session", platform="telegram", context_length=1000)
            active = first.compress(
                [{"role": "user", "content": "SECRET_PAYLOAD_MARKER externalized row " + "x" * 200}],
                current_tokens=10_000,
            )
            active_stub = {"role": "user", "content": active[0]["content"]}
            ignored_store_id = first._store.get_session_messages("session")[0]["store_id"]
            assert "Externalized payload:" in active_stub["content"]
        finally:
            first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10_000,
                ignore_message_patterns=["SECRET_PAYLOAD_MARKER"],
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=10_000,
            ),
            hermes_home=str(hermes_home),
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            messages = [active_stub, {"role": "assistant", "content": "fresh tail response"}]
            result = second.compress(messages, current_tokens=10_000)
            result_text = "\n".join(str(msg.get("content", "")) for msg in result)
            assert "Externalized payload:" not in result_text
            rows = second._store.get_session_messages("session")
            assert [row["store_id"] for row in rows] == [ignored_store_id, ignored_store_id + 1]
        finally:
            second.shutdown()

    def test_active_externalized_stub_followed_by_user_is_not_re_stored_after_ignore_added(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_externalized_active_stub_then_user.db"
        hermes_home = tmp_path / "hermes-externalized-active-stub-then-user"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10_000,
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=50,
            ),
            hermes_home=str(hermes_home),
        )
        try:
            first.on_session_start("session", platform="telegram", context_length=1000)
            active = first.compress(
                [{"role": "user", "content": "SECRET_PAYLOAD_MARKER externalized row " + "x" * 200}],
                current_tokens=10_000,
            )
            active_stub = {"role": "user", "content": active[0]["content"]}
            ignored_store_id = first._store.get_session_messages("session")[0]["store_id"]
        finally:
            first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10_000,
                ignore_message_patterns=["SECRET_PAYLOAD_MARKER"],
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=10_000,
            ),
            hermes_home=str(hermes_home),
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            messages = [active_stub, {"role": "user", "content": "fresh visible request"}]
            result = second.compress(messages, current_tokens=10_000)
            result_text = "\n".join(str(msg.get("content", "")) for msg in result)
            assert "Externalized payload:" not in result_text
            rows = second._store.get_session_messages("session")
            assert [row["store_id"] for row in rows] == [ignored_store_id, ignored_store_id + 1]
            assert rows[1]["content"] == "fresh visible request"
        finally:
            second.shutdown()

    def test_copied_ingest_externalized_placeholder_literal_keeps_source_lineage(self, tmp_path, monkeypatch):
        db_path = tmp_path / "lcm_msg_ignore_ingest_externalized_literal_copy.db"
        hermes_home = tmp_path / "hermes-ingest-externalized-literal-copy"
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                large_output_externalization_path=str(tmp_path / "externalized"),
            ),
            hermes_home=str(hermes_home),
        )
        try:
            engine.on_session_start("session", platform="telegram", context_length=1000)
            externalized = externalize_ingest_payload(
                "DROP: copied ingest placeholder payload " + "x" * 200,
                role="user",
                session_id="session",
                field_path="content",
                config=engine._config,
                hermes_home=str(hermes_home),
            )
            assert externalized is not None
            placeholder_literal = externalized["placeholder"]
            captured: dict[str, str] = {}

            def summary(**kwargs):
                captured["text"] = kwargs["text"]
                return "literal ingest placeholder summary\n[Expand for details: literal ingest placeholder]", 1

            monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary)
            messages = [
                {"role": "user", "content": placeholder_literal},
                {"role": "user", "content": "visible backlog objective " + "v" * 200},
                {"role": "assistant", "content": "fresh tail response"},
            ]
            engine.compress(messages, current_tokens=count_messages_tokens(messages))

            rows = engine._store.get_session_messages("session")
            assert rows[0]["content"] == placeholder_literal
            assert placeholder_literal in captured["text"]
            assert "visible backlog objective" in captured["text"]
            nodes = engine._dag.get_session_nodes("session")
            assert nodes
            assert rows[0]["store_id"] in nodes[0].source_ids
        finally:
            engine.shutdown()

    def test_known_ignored_eof_dependent_reply_is_removed_from_noop_active_context(
        self, tmp_path, monkeypatch
    ):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_eof_dependent_system_anchor.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        dependent_reply = "dependent reply from ignored turn"

        def summary(**kwargs):
            return "visible summary\n[Expand for details: visible]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary)
        first_result = engine.compress(
            [
                {"role": "system", "content": "system anchor"},
                {"role": "user", "content": "visible backlog before ignored turn " + "v" * 200},
                {"role": "user", "content": "SECRET ignored turn"},
                {"role": "assistant", "content": dependent_reply},
            ],
            current_tokens=10_000,
        )
        assert dependent_reply in "\n".join(str(msg.get("content", "")) for msg in first_result)
        assert engine._load_generated_ignored_dependent_reply_hashes()

        result = engine.compress(first_result, current_tokens=count_messages_tokens(first_result))
        result_text = "\n".join(str(msg.get("content", "")) for msg in result)
        rows_text = "\n".join(row["content"] for row in engine._store.get_session_messages("user-123"))

        assert "system anchor" in result_text
        assert "SECRET" not in result_text
        assert dependent_reply not in result_text
        assert dependent_reply in rows_text

    def test_singleton_copied_externalized_placeholder_after_ignored_row_is_lossless(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_externalized_singleton_literal_copy.db"
        hermes_home = tmp_path / "hermes-externalized-singleton-literal-copy"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=50,
            ),
            hermes_home=str(hermes_home),
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        ignored_store_id = first._store.append(
            "session",
            {"role": "user", "content": "SECRET_PAYLOAD_MARKER externalized row " + "x" * 200},
        )
        stored_externalized_row = first._store.get(ignored_store_id)
        assert stored_externalized_row is not None
        placeholder_literal = stored_externalized_row["content"]
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=["SECRET_PAYLOAD_MARKER"],
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=10_000,
            ),
            hermes_home=str(hermes_home),
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            result = second.compress(
                [{"role": "user", "content": placeholder_literal}],
                current_tokens=10_000,
            )

            rows = second._store.get_session_messages("session")
            result_text = "\n".join(str(msg.get("content", "")) for msg in result)
            assert [row["store_id"] for row in rows] == [ignored_store_id]
            assert "Externalized payload:" not in result_text
            assert "SECRET_PAYLOAD_MARKER" not in result_text
        finally:
            second.shutdown()

    def test_known_ignored_eof_dependent_reply_after_visible_user_without_system_is_removed(
        self, tmp_path, monkeypatch
    ):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_eof_dependent_no_system_visible_prefix.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        dependent_reply = "dependent reply from ignored turn"

        def summary(**kwargs):
            return "visible summary\n[Expand for details: visible]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary)
        first_result = engine.compress(
            [
                {"role": "user", "content": "visible backlog before ignored turn " + "v" * 200},
                {"role": "user", "content": "SECRET ignored turn"},
                {"role": "assistant", "content": dependent_reply},
            ],
            current_tokens=10_000,
        )
        assert dependent_reply in "\n".join(str(msg.get("content", "")) for msg in first_result)
        assert engine._load_generated_ignored_dependent_reply_hashes()

        result = engine.compress(first_result, current_tokens=count_messages_tokens(first_result))
        result_text = "\n".join(str(msg.get("content", "")) for msg in result)
        rows_text = "\n".join(row["content"] for row in engine._store.get_session_messages("user-123"))

        assert "visible summary" in result_text
        assert "SECRET" not in result_text
        assert dependent_reply not in result_text
        assert dependent_reply in rows_text

    def test_dependent_assistant_reply_to_ignored_backlog_is_not_summarized(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_reply.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        dependent_message = {"role": "assistant", "content": "dependent assistant reply to ignored turn " + "d" * 500}
        messages = [
            {"role": "user", "content": "SECRET ignored user turn"},
            dependent_message,
            {"role": "user", "content": "visible backlog objective " + "y" * 200},
            {"role": "assistant", "content": "fresh tail response"},
        ]

        engine.compress(messages, current_tokens=count_messages_tokens(messages))

        assert "visible backlog objective" in captured["text"]
        assert "dependent assistant reply" not in captured["text"]
        assert "SECRET" not in captured["text"]
        rows = engine._store.get_session_messages("user-123")
        dependent_ids = [row["store_id"] for row in rows if "dependent assistant reply" in row["content"]]
        nodes = engine._dag.get_session_nodes("user-123")
        assert dependent_ids
        assert all(dependent_ids[0] not in node.source_ids for node in nodes)
        assert engine._last_compacted_store_id >= dependent_ids[0]

    def test_dependent_assistant_reply_to_ignored_system_backlog_is_not_summarized(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_system_dependent_reply.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        messages = [
            {"role": "system", "content": "public system prompt"},
            {"role": "user", "content": "visible backlog objective " + "y" * 200},
            {"role": "system", "content": "SECRET ignored system instruction"},
            {"role": "assistant", "content": "assistant reply derived from ignored system secret " + "d" * 500},
            {"role": "user", "content": "fresh tail request"},
        ]

        engine.compress(messages, current_tokens=count_messages_tokens(messages))

        assert "visible backlog objective" in captured["text"]
        assert "assistant reply derived from ignored system" not in captured["text"]
        assert "SECRET" not in captured["text"]

    def test_trailing_dependent_reply_is_consumed_with_selected_visible_chunk(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_trailing_dependent_reply.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        dependent_message = {"role": "assistant", "content": "trailing dependent assistant reply " + "d" * 500}
        messages = [
            {"role": "user", "content": "visible backlog objective " + "y" * 200},
            {"role": "user", "content": "SECRET ignored user turn"},
            dependent_message,
            {"role": "assistant", "content": "fresh tail response"},
        ]

        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
        result_text = "\n".join(str(msg.get("content", "")) for msg in result)

        assert "visible backlog objective" in captured["text"]
        assert "trailing dependent assistant reply" not in captured["text"]
        assert "SECRET" not in captured["text"]
        assert "trailing dependent assistant reply" not in result_text
        rows = engine._store.get_session_messages("user-123")
        dependent_ids = [row["store_id"] for row in rows if "trailing dependent assistant reply" in row["content"]]
        nodes = engine._dag.get_session_nodes("user-123")
        assert dependent_ids
        assert all(dependent_ids[0] not in node.source_ids for node in nodes)
        assert engine._last_compacted_store_id >= dependent_ids[0]

    def test_dependent_reply_marker_does_not_match_later_identical_reply(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_identical_later_reply.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        repeated_reply = "same assistant reply text"
        first_result = engine.compress(
            [
                {"role": "user", "content": "SECRET ignored user turn"},
                {"role": "assistant", "content": repeated_reply},
                {"role": "user", "content": "fresh tail request"},
            ],
            current_tokens=10_000,
        )
        assert engine._load_generated_ignored_dependent_reply_hashes()

        later_same_reply = {"role": "assistant", "content": repeated_reply}
        engine._ingest_messages(
            first_result
            + [
                {"role": "user", "content": "legitimate visible prompt"},
                later_same_reply,
            ]
        )
        dependent_store_id = next(
            row["store_id"]
            for row in engine._store.get_session_messages("user-123")
            if row["role"] == "assistant" and row["content"] == repeated_reply
        )
        engine._last_compacted_store_id = 0
        engine._current_compress_store_ids_by_message_id = {}
        assert engine._get_store_ids_for_messages([later_same_reply]) == [dependent_store_id]

        assert not engine._is_generated_ignored_dependent_reply(later_same_reply, repeated_reply)

    def test_dependent_reply_marker_does_not_filter_later_identical_reply_during_compress(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_identical_later_compress.db",
            fresh_tail_count=2,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        repeated_reply = "same assistant reply text"
        captured_texts: list[str] = []

        def capture_summary(**kwargs):
            captured_texts.append(kwargs["text"])
            return "visible summary\n[Expand for details: visible summary]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        first_result = engine.compress(
            [
                {"role": "user", "content": "visible backlog before ignored turn " + "v" * 200},
                {"role": "user", "content": "SECRET ignored user turn"},
                {"role": "assistant", "content": repeated_reply},
                {"role": "user", "content": "fresh tail request"},
            ],
            current_tokens=10_000,
        )
        assert engine._load_generated_ignored_dependent_reply_hashes()
        first_summary_texts = list(captured_texts)
        assert all(repeated_reply not in text for text in first_summary_texts)

        engine.compress(
            first_result
            + [
                {"role": "user", "content": "legitimate visible prompt"},
                {"role": "assistant", "content": repeated_reply},
                {"role": "user", "content": "new fresh request"},
                {"role": "assistant", "content": "new fresh response"},
            ],
            current_tokens=10_000,
        )

        later_summary_texts = captured_texts[len(first_summary_texts):]
        assert any(
            "legitimate visible prompt" in text and repeated_reply in text
            for text in later_summary_texts
        )

    def test_content_only_dependent_marker_survives_same_engine_compression_rollover(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_same_engine_rollover.db",
            fresh_tail_count=2,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        dependent_reply = "dependent reply carried in same engine rollover"

        def first_summary(**kwargs):
            return "visible summary\n[Expand for details: visible]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", first_summary)
        engine.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-dependent-same-engine-rollover",
        )
        first_result = engine.compress(
            [
                {"role": "user", "content": "visible backlog before ignored turn " + "v" * 200},
                {"role": "user", "content": "SECRET ignored turn before carried reply"},
                {"role": "assistant", "content": dependent_reply},
                {"role": "user", "content": "fresh tail request"},
            ],
            current_tokens=10_000, force=True,  # manual seed; synthetic pressure is not a host estimate
        )
        assert dependent_reply in "\n".join(str(msg.get("content", "")) for msg in first_result)
        assert engine._load_generated_ignored_dependent_reply_hashes()

        captured_texts: list[str] = []

        def capture_summary(**kwargs):
            captured_texts.append(kwargs["text"])
            return "new visible summary\n[Expand for details: new visible]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        engine.on_session_start(
            "new-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-dependent-same-engine-rollover",
            boundary_reason="compression",
            old_session_id="old-session",
        )
        engine.compress(
            first_result
            + [
                {"role": "user", "content": "new visible backlog " + "z" * 200},
                {"role": "assistant", "content": "new fresh tail response"},
            ],
            current_tokens=10_000, force=True,  # manual seed; synthetic pressure is not a host estimate
        )

        assert captured_texts
        assert all(dependent_reply not in text for text in captured_texts)
        assert all("SECRET" not in text for text in captured_texts)

    def test_duplicate_dependent_content_markers_survive_rollover_copy(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_duplicate_dependent_rollover_copy.db",
            fresh_tail_count=2,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        try:
            engine.on_session_start(
                "old-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-duplicate-dependent-rollover",
            )
            dependent_reply = "duplicate dependent reply carried over rollover"
            content_digest = engine._ignored_dependent_reply_content_fingerprint(
                {"role": "assistant", "content": dependent_reply},
                dependent_reply,
            )
            assert content_digest is not None
            first_store_id = engine._store.append(
                "old-session",
                {"role": "assistant", "content": dependent_reply},
                token_estimate=1,
                source="telegram",
                conversation_id="conv-duplicate-dependent-rollover",
            )
            second_store_id = engine._store.append(
                "old-session",
                {"role": "assistant", "content": dependent_reply},
                token_estimate=1,
                source="telegram",
                conversation_id="conv-duplicate-dependent-rollover",
            )
            source_keys = engine._session_scoped_hash_metadata_keys(
                "ignored_dependent_reply_hashes",
                "old-session",
            )
            engine._write_generated_ignored_dependent_reply_records(
                [
                    {
                        "store": hashlib.sha256(
                            f"old-session\0{first_store_id}".encode("utf-8")
                        ).hexdigest()[:16],
                        "content": content_digest,
                    },
                    {
                        "store": hashlib.sha256(
                            f"old-session\0{second_store_id}".encode("utf-8")
                        ).hexdigest()[:16],
                        "content": content_digest,
                    },
                ],
                source_keys,
            )

            engine._copy_generated_ignore_hashes_to_session(
                "old-session",
                "new-session",
                copy_dependent_content=True,
                source_frontier_store_id=0,
            )

            target_keys = engine._session_scoped_hash_metadata_keys(
                "ignored_dependent_reply_hashes",
                "new-session",
            )
            copied_records = engine._load_generated_ignored_dependent_reply_records(target_keys)
            assert copied_records == [
                {"content": content_digest},
                {"content": content_digest},
            ]

            engine._session_id = "new-session"
            live_records = engine._load_generated_ignored_dependent_reply_records()
            msg = {"role": "assistant", "content": dependent_reply}
            assert engine._matches_preexisting_generated_ignored_dependent_reply(
                msg,
                dependent_reply,
                live_records,
            )
            assert engine._matches_preexisting_generated_ignored_dependent_reply(
                msg,
                dependent_reply,
                live_records,
            )
            assert not engine._matches_preexisting_generated_ignored_dependent_reply(
                msg,
                dependent_reply,
                live_records,
            )
        finally:
            engine.shutdown()

    def test_dependent_reply_marker_survives_compression_rollover_reingest(self, tmp_path, monkeypatch):
        db_path = tmp_path / "lcm_msg_ignore_dependent_rollover.db"
        dependent_reply = "dependent reply carried across compression rollover"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=2,
                leaf_chunk_tokens=10,
                ignore_message_patterns=["SECRET"],
            )
        )
        first.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-dependent-rollover",
        )

        def summary(**kwargs):
            return "visible summary\n[Expand for details: visible]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary)
        first_result = first.compress(
            [
                {"role": "user", "content": "visible backlog before ignored turn " + "v" * 200},
                {"role": "user", "content": "SECRET ignored turn before carried reply"},
                {"role": "assistant", "content": dependent_reply},
                {"role": "user", "content": "fresh tail request"},
            ],
            current_tokens=10_000, force=True,  # manual seed; synthetic pressure is not a host estimate
        )
        assert dependent_reply in "\n".join(str(msg.get("content", "")) for msg in first_result)
        assert first._load_generated_ignored_dependent_reply_hashes()
        first.shutdown()

        captured_texts: list[str] = []

        def capture_summary(**kwargs):
            captured_texts.append(kwargs["text"])
            return "new visible summary\n[Expand for details: new visible]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-dependent-rollover",
                boundary_reason="compression",
                old_session_id="old-session",
            )
            second.compress(
                first_result + [{"role": "assistant", "content": "new fresh tail response"}],
                current_tokens=10_000, force=True,  # manual seed; synthetic pressure is not a host estimate
            )

            assert captured_texts
            assert all(dependent_reply not in text for text in captured_texts)
        finally:
            second.shutdown()

    def test_ignored_plain_assistant_filters_following_assistant_continuation(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_plain_assistant_continuation.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        captured_texts: list[str] = []

        def capture_summary(**kwargs):
            captured_texts.append(kwargs["text"])
            return "visible summary\n[Expand for details: visible]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        engine.compress(
            [
                {"role": "assistant", "content": "SECRET ignored assistant output"},
                {"role": "assistant", "content": "assistant continuation derived from SECRET"},
                {"role": "user", "content": "visible backlog " + "v" * 200},
                {"role": "assistant", "content": "fresh tail response"},
            ],
            current_tokens=10_000,
        )

        assert captured_texts
        assert all("SECRET" not in text for text in captured_texts)
        assert all("assistant continuation derived" not in text for text in captured_texts)
        assert any("visible backlog" in text for text in captured_texts)

    def test_generated_dependent_reply_filters_following_assistant_continuation(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_generated_continuation.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        dependent_reply = "dependent assistant reply after ignored turn"
        continuation = "assistant continuation derived from ignored turn"
        captured_texts: list[str] = []

        def capture_summary(**kwargs):
            captured_texts.append(kwargs["text"])
            return "visible summary\n[Expand for details: visible summary]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        first_result = engine.compress(
            [
                {"role": "user", "content": "visible backlog before ignored turn " + "v" * 200},
                {"role": "user", "content": "SECRET ignored user turn"},
                {"role": "assistant", "content": dependent_reply},
            ],
            current_tokens=10_000,
        )
        assert dependent_reply in "\n".join(str(msg.get("content", "")) for msg in first_result)
        assert engine._load_generated_ignored_dependent_reply_hashes()

        engine.compress(
            first_result
            + [
                {"role": "assistant", "content": continuation},
                {"role": "user", "content": "new visible user boundary"},
                {"role": "assistant", "content": "new fresh response"},
            ],
            current_tokens=10_000,
        )

        assert captured_texts
        assert all(dependent_reply not in text for text in captured_texts)
        assert all(continuation not in text for text in captured_texts)

    def test_generated_dependent_reply_marks_following_fresh_tail_continuation(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_generated_tail_continuation.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        dependent_reply = "dependent assistant reply carried in active tail"
        continuation = "fresh tail continuation derived from ignored turn"
        captured_texts: list[str] = []

        def capture_summary(**kwargs):
            captured_texts.append(kwargs["text"])
            return "visible summary\n[Expand for details: visible summary]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        first_result = engine.compress(
            [
                {"role": "user", "content": "visible backlog before ignored turn " + "v" * 200},
                {"role": "user", "content": "SECRET ignored user turn"},
                {"role": "assistant", "content": dependent_reply},
            ],
            current_tokens=10_000,
        )
        second_result = engine.compress(
            first_result + [{"role": "assistant", "content": continuation}],
            current_tokens=10_000,
        )
        engine.compress(
            second_result
            + [
                {"role": "user", "content": "new visible boundary"},
                {"role": "assistant", "content": "new fresh response"},
            ],
            current_tokens=10_000,
        )

        assert captured_texts
        assert all(dependent_reply not in text for text in captured_texts)
        assert all(continuation not in text for text in captured_texts)

    def test_new_session_carry_over_does_not_poison_same_text_future_reply(self, tmp_path, monkeypatch):
        db_path = tmp_path / "lcm_msg_ignore_dependent_new_session_no_content_poison.db"
        repeated_reply = "OK"
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=["SECRET"],
                threshold_full_sweep_enabled=False,  # #1013: pin the pre-#1013 single-pass compaction

                # #1013 part B: protection tests cover the off path (owner scope ban)
                large_output_externalization_enabled=False,
                large_output_active_replay_stubbing_enabled=False,
                temporal_rollups_enabled=False,
            )
        )
        engine.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-normal-carry-over",
        )
        captured_texts: list[str] = []

        def capture_summary(**kwargs):
            captured_texts.append(kwargs["text"])
            return "visible summary\n[Expand for details: visible summary]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        try:
            engine.compress(
                [
                    {"role": "user", "content": "visible backlog before ignored turn " + "v" * 200},
                    {"role": "user", "content": "SECRET ignored user turn"},
                    {"role": "assistant", "content": repeated_reply},
                ],
                current_tokens=10_000,
            )
            assert engine._load_generated_ignored_dependent_reply_hashes()
            first_summary_texts = list(captured_texts)
            assert all(repeated_reply not in text for text in first_summary_texts)

            engine.carry_over_new_session_context("old-session", "new-session")
            engine.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-normal-carry-over",
            )
            engine.compress(
                [
                    {"role": "user", "content": "legitimate visible prompt " + "z" * 200},
                    {"role": "assistant", "content": repeated_reply},
                    {"role": "user", "content": "fresh request"},
                ],
                current_tokens=10_000,
            )

            later_summary_texts = captured_texts[len(first_summary_texts):]
            assert any(
                "legitimate visible prompt" in text and repeated_reply in text
                for text in later_summary_texts
            )
        finally:
            engine.shutdown()

    def test_dependent_reply_to_ignored_turn_stays_filtered_after_fresh_tail_ages(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_fresh_tail_marker.db",
            fresh_tail_count=2,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        captured_texts: list[str] = []

        def capture_summary(**kwargs):
            captured_texts.append(kwargs["text"])
            return "visible summary\n[Expand for details: visible summary]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        first_result = engine.compress(
            [
                {"role": "user", "content": "visible backlog before secret " + "v" * 200},
                {"role": "user", "content": "SECRET ignored turn before fresh tail"},
                {"role": "assistant", "content": "dependent assistant reply that must not summarize later"},
                {"role": "user", "content": "fresh tail request"},
            ],
            current_tokens=10_000, force=True,  # manual seed; synthetic pressure is not a host estimate
        )

        engine.compress(
            first_result + [{"role": "assistant", "content": "new fresh assistant response"}],
            current_tokens=10_000, force=True,  # manual seed; synthetic pressure is not a host estimate
        )

        assert captured_texts
        assert all("SECRET" not in text for text in captured_texts)
        assert all("dependent assistant reply" not in text for text in captured_texts)
        rows = engine._store.get_session_messages("user-123")
        dependent_ids = [
            row["store_id"]
            for row in rows
            if "dependent assistant reply that must not summarize later" in row["content"]
        ]
        nodes = engine._dag.get_session_nodes("user-123")
        assert dependent_ids
        assert all(dependent_ids[0] not in node.source_ids for node in nodes)
        assert engine._last_compacted_store_id >= dependent_ids[0]

    def test_dependent_reply_in_tail_is_marked_before_anchor_break(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_tail_anchor_break.db",
            fresh_tail_count=3,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        captured_texts: list[str] = []

        def capture_summary(**kwargs):
            captured_texts.append(kwargs["text"])
            return "visible summary\n[Expand for details: visible summary]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        first_result = engine.compress(
            [
                {"role": "system", "content": "stable system anchor"},
                {"role": "user", "content": "SECRET ignored compactable turn"},
                {"role": "assistant", "content": "dependent tail reply after ignored turn"},
                {"role": "user", "content": "fresh tail request"},
                {"role": "assistant", "content": "fresh tail response"},
            ],
            current_tokens=10_000,
        )

        # The first pass drops the ignored compactable turn, then fresh_tail_start
        # collapses back to the system anchor. The dependent assistant reply must
        # still be remembered before that no-op break, otherwise a later pass can
        # summarize it once it ages out of the fresh tail.
        engine.compress(
            first_result
            + [
                {"role": "user", "content": "later visible backlog " + "z" * 200},
                {"role": "assistant", "content": "later visible response"},
            ],
            current_tokens=10_000,
        )

        assert captured_texts
        assert all("SECRET" not in text for text in captured_texts)
        assert all("dependent tail reply" not in text for text in captured_texts)

    def test_preflight_skips_when_only_ignored_backlog_is_eligible(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_preflight_only_ignored_backlog.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        messages = [
            {"role": "user", "content": "SECRET ignored backlog " + "x" * 4000},
            {"role": "user", "content": "fresh request"},
        ]

        assert count_messages_tokens(messages) >= engine.threshold_tokens
        assert engine.should_compress_preflight(messages) is True
        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
        assert all("SECRET" not in str(msg.get("content", "")) for msg in result)
        assert all(
            "LCM active replay placeholder: message ignored" not in str(msg.get("content", ""))
            for msg in result
        )

    def test_preflight_uses_replay_view_when_ignored_backlog_masks_tiny_visible_chunk(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_preflight_ignored_masks_tiny_visible.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=500,
            ignore_message_patterns=["SECRET"],
        )
        messages = [
            {"role": "user", "content": "SECRET ignored backlog " + "x" * 4000},
            {"role": "user", "content": "visible tiny"},
            {"role": "user", "content": "fresh request"},
        ]

        assert count_messages_tokens(messages) >= engine.threshold_tokens
        assert engine.should_compress_preflight(messages) is True
        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
        assert all("SECRET" not in str(msg.get("content", "")) for msg in result)
        assert any("visible tiny" in str(msg.get("content", "")) for msg in result)

    def test_preflight_filters_generated_placeholder_backlog_without_active_patterns_after_rollover(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_preflight_placeholder_without_patterns.db"),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-preflight-placeholder-without-patterns",
            )
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            engine._remember_generated_ignored_placeholder_hash(digest)
            messages = [
                {"role": "user", "content": placeholder},
                {"role": "assistant", "content": "fresh tail"},
            ]

            eligible, reason = engine._leaf_compaction_candidate_status(messages)

            assert eligible is False
            assert "no eligible raw backlog" in reason
            assert engine.should_compress_preflight(messages) is False
        finally:
            engine.shutdown()

    def test_preflight_does_not_persist_generated_placeholder_after_same_session_restart_without_patterns(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_preflight_placeholder_same_session_restart.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                context_threshold=0.35,
            )
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        first._store.append(
            "session",
            {"role": "user", "content": "visible backlog before restart " + "v" * 120},
        )
        placeholder = first._ignored_active_replay_placeholder("api_key=sk-ignore...cdef ignored same-session restart")
        digest = first._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        first._remember_generated_ignored_placeholder_hash(digest)
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                context_threshold=0.35,
            )
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            messages = [
                {"role": "user", "content": placeholder},
                {"role": "assistant", "content": "large fresh tail " + "x" * 700},
            ]
            assert second._leaf_compaction_candidate_status(messages)[0] is False

            assert second.should_compress_preflight(messages) is False

            rows = second._store.get_session_messages("session")
            assert any(row["content"] == placeholder for row in rows)
        finally:
            second.shutdown()

    def test_preflight_ambiguous_generated_placeholder_still_requests_externalization_cleanup(self, tmp_path, monkeypatch):
        # #1055: force externalization for this fixture independently of the bound window.
        monkeypatch.setattr("hermes_lcm.ingest_protection._non_tool_text_over_floor", lambda text, config, **_: bool(text))
        db_path = tmp_path / "lcm_msg_ignore_preflight_placeholder_externalize_cleanup.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                context_threshold=0.35,
            )
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        first._store.append("session", {"role": "user", "content": "old row"})
        placeholder = first._ignored_active_replay_placeholder("api_key=sk-ignore...cdef externalize cleanup")
        digest = first._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        first._remember_generated_ignored_placeholder_hash(digest)
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                context_threshold=0.35,
                large_output_externalization_enabled=True,
                large_output_externalization_threshold_chars=40,
                large_output_externalization_path=str(tmp_path / "externalized"),
            )
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            messages = [
                {"role": "user", "content": placeholder},
                {"role": "user", "content": "oversized raw payload " + "x" * 200},
            ]

            assert second.should_compress_preflight(messages) is True
        finally:
            second.shutdown()

    def test_preflight_requests_cleanup_for_sensitive_tool_call_redaction(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_preflight_sensitive_tool_call_cleanup.db",
            fresh_tail_count=10,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "calling lookup",
                "tool_calls": [
                    {
                        "id": "call_lookup",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"api_key":"sk-sensitive...cdef"}',
                        },
                    }
                ],
            },
        ]

        assert engine._leaf_compaction_candidate_status(messages)[0] is False
        assert engine.should_compress_preflight(messages) is True

    def test_preflight_requests_cleanup_for_sensitive_content_redaction(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_preflight_sensitive_content_cleanup.db",
            fresh_tail_count=10,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        sensitive_text = "api_key=" + "sk-sensitive...cdef"
        messages = [
            {"role": "user", "content": sensitive_text + " should be redacted"},
            {"role": "assistant", "content": "ack"},
        ]

        assert engine._leaf_compaction_candidate_status(messages)[0] is False
        assert engine.should_compress_preflight(messages) is True

    def test_preflight_requests_cleanup_for_sensitive_structured_content_redaction(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_preflight_sensitive_structured_content_cleanup.db",
            fresh_tail_count=10,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        sensitive_url_param = "api_key=" + "sk-sen...cdef"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.test/image.png?" + sensitive_url_param
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "ack"},
        ]

        assert engine._leaf_compaction_candidate_status(messages)[0] is False
        assert engine.should_compress_preflight(messages) is True

    def test_preflight_requests_cleanup_for_sensitive_structured_key_redaction(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_preflight_sensitive_structured_key_cleanup.db",
            fresh_tail_count=10,
            leaf_chunk_tokens=10_000,
            context_threshold=0.95,
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        sensitive_content_key = "api_key=" + "sk-key...cdef"
        sensitive_tool_key = "api_key=" + "sk-toolkey...cdef"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "visible"},
                    {sensitive_content_key: 1},
                ],
            },
            {
                "role": "assistant",
                "content": "calling lookup",
                "tool_calls": [
                    {
                        "id": "call_lookup",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {sensitive_tool_key: 1},
                        },
                    }
                ],
            },
        ]

        assert engine._leaf_compaction_candidate_status(messages)[0] is False
        assert engine.should_compress_preflight(messages) is True

    def test_preflight_preserves_user_literal_placeholder_plus_followup_after_same_session_restart(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_preflight_literal_placeholder_followup.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                context_threshold=0.35,
            )
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        first._store.append("session", {"role": "user", "content": "old row"})
        placeholder = first._ignored_active_replay_placeholder("api_key=sk-ignore...cdef user literal")
        digest = first._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        first._remember_generated_ignored_placeholder_hash(digest)
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                context_threshold=0.35,
            )
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            followup = "actual next user turn"
            messages = [
                {"role": "user", "content": placeholder},
                {"role": "user", "content": followup},
            ]

            second.should_compress_preflight(messages)

            rows = second._store.get_session_messages("session")
            row_texts = [row["content"] for row in rows]
            assert placeholder in row_texts
            assert followup in row_texts
        finally:
            second.shutdown()

    def test_preflight_preserves_ignored_placeholder_for_later_compress(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_preflight_placeholder.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        messages = [
            {"role": "user", "content": "visible backlog objective " + "y" * 200},
            {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
        ]

        assert engine.should_compress_preflight(messages) is True
        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
        result_text = "\n".join(str(msg.get("content", "")) for msg in result)

        assert "LCM active replay placeholder: message ignored" in result_text
        assert "sk-ignore" not in result_text

    def test_ignored_assistant_tool_call_placeholder_uses_redacted_tool_calls(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_assistant_tool_placeholder.db",
            fresh_tail_count=10,
            leaf_chunk_tokens=10_000,
            ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        messages = [
            {
                "role": "assistant",
                "content": "api_key=sk-ignore...cdef ignored tool-call turn",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"api_key":"sk-ignore...cdef"}',
                        },
                    }
                ],
            },
            {"role": "user", "content": "fresh request"},
        ]
        replay_messages = [
            {
                "role": "assistant",
                "content": "redacted ignored tool-call turn",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"api_key":"***"}',
                        },
                    }
                ],
            },
            {"role": "user", "content": "fresh request"},
        ]

        result = engine._apply_ignored_active_replay_placeholders(
            messages,
            replay_messages,
            ignored_messages=[True, False],
        )

        assert result[0]["role"] == "assistant"
        assert "LCM active replay placeholder: message ignored" in str(result[0].get("content", ""))
        assert result[0].get("tool_calls") is None
        assert "sk-ignore" not in json.dumps(result, ensure_ascii=False)
        assert "api_key" not in str(result[0].get("content", ""))

    def test_empty_ingest_clears_compression_boundary_replay_flag(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_empty_boundary_ingest.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
        )
        messages = [{"role": "user", "content": "carried context"}]
        engine._ingest_cursor = len(messages)
        engine._compression_boundary_ingest_pending = True

        engine._ingest_messages(messages)

        assert engine._compression_boundary_ingest_pending is False

    def test_content_only_dependent_marker_is_consumed_after_match(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_content_marker_consumed.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
        )
        message = {"role": "assistant", "content": "OK"}
        digest = engine._ignored_dependent_reply_content_fingerprint(message, "OK")
        assert digest
        engine._write_generated_ignored_dependent_reply_records([{"content": digest}])

        assert engine._is_generated_ignored_dependent_reply(message, "OK") is True
        assert engine._is_generated_ignored_dependent_reply(message, "OK") is False

    def test_ignored_plain_assistant_placeholder_preserves_assistant_role(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_plain_assistant_placeholder.db",
            fresh_tail_count=10,
            leaf_chunk_tokens=10_000,
            ignore_message_patterns=[r"SECRET"],
        )
        messages = [
            {"role": "assistant", "content": "SECRET ignored assistant response"},
            {"role": "user", "content": "fresh request"},
        ]
        result = engine._apply_ignored_active_replay_placeholders(
            messages,
            [
                {"role": "assistant", "content": "redacted assistant response"},
                {"role": "user", "content": "fresh request"},
            ],
            ignored_messages=[True, False],
        )

        assert result[0]["role"] == "assistant"
        assert "LCM active replay placeholder: message ignored" in str(result[0].get("content", ""))
        assert "SECRET" not in str(result[0].get("content", ""))

    def test_ignored_system_placeholder_preserves_system_role(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_system_placeholder.db",
            fresh_tail_count=10,
            leaf_chunk_tokens=10_000,
            ignore_message_patterns=[r"SYSTEM_SECRET"],
        )
        messages = [
            {"role": "system", "content": "SYSTEM_SECRET ignored system prompt"},
            {"role": "user", "content": "fresh request"},
        ]

        result = engine._apply_ignored_active_replay_placeholders(
            messages,
            [
                {"role": "system", "content": "redacted system prompt"},
                {"role": "user", "content": "fresh request"},
            ],
            scan_start=0,
        )

        assert result[0]["role"] == "system"
        assert "LCM active replay placeholder: message ignored" in str(result[0].get("content", ""))
        assert "SYSTEM_SECRET" not in str(result[0].get("content", ""))

    def test_ignored_tool_result_placeholder_preserves_tool_role_and_call_id(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_tool_result_placeholder.db",
            fresh_tail_count=10,
            leaf_chunk_tokens=10_000,
            ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        messages = [
            {"role": "assistant", "content": "calling tool", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "api_key=sk-ignore...cdef ignored tool result"},
            {"role": "user", "content": "fresh request"},
        ]

        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))

        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_1"
        assert "LCM active replay placeholder: message ignored" in str(result[1].get("content", ""))
        assert "api_key" not in str(result[1].get("content", ""))

    def test_dependent_tool_result_to_ignored_assistant_tool_call_is_not_summarized(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_tool.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["IGNORE_TOOL_CALL"],
        )
        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        messages = [
            {
                "role": "assistant",
                "content": "IGNORE_TOOL_CALL assistant turn",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "dependent tool result"},
            {"role": "user", "content": "visible backlog objective " + "y" * 200},
            {"role": "assistant", "content": "fresh tail response"},
        ]

        engine.compress(messages, current_tokens=count_messages_tokens(messages))

        assert "visible backlog objective" in captured["text"]
        assert "dependent tool result" not in captured["text"]
        assert "IGNORE_TOOL_CALL" not in captured["text"]

    def test_dependent_assistant_reply_to_ignored_tool_result_is_not_summarized(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_dependent_tool_reply.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["PRIVATE_TOOL_RESULT"],
        )
        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "PRIVATE_TOOL_RESULT noisy/private output"},
            {"role": "assistant", "content": "assistant answer derived from private tool result"},
            {"role": "user", "content": "visible backlog objective " + "y" * 200},
            {"role": "assistant", "content": "fresh tail response"},
        ]

        engine.compress(messages, current_tokens=count_messages_tokens(messages))

        assert "visible backlog objective" in captured["text"]
        assert "assistant answer derived" not in captured["text"]
        assert "PRIVATE_TOOL_RESULT" not in captured["text"]

    def test_user_authored_ignored_placeholder_text_remains_lossless(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_placeholder_literal.db",
            ignore_message_patterns=["^Cronjob Response:"],
        )
        placeholder = (
            "[LCM active replay placeholder: message ignored; kind=ignored_message; "
            "scope=ignored_message_pattern; field=content; chars=10; bytes=10; "
            "sha256=0123456789abcdef]"
        )

        engine._ingest_messages([{"role": "user", "content": placeholder}])
        stored = engine._store.get_session_messages("user-123")

        assert [row["content"] for row in stored] == [placeholder]
        assert engine._ignored_message_count == 0

    def test_generated_placeholder_hashes_evict_by_recency_not_lexical_order(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_placeholder_hash_recency.db",
        )
        for idx in range(512):
            engine._remember_generated_ignored_placeholder_hash(f"{idx + 1:016x}")
        recent_low_digest = "0000000000000000"

        engine._remember_generated_ignored_placeholder_hash(recent_low_digest)
        loaded = engine._load_generated_ignored_placeholder_hash_list()

        assert recent_low_digest in loaded
        assert loaded[-1] == recent_low_digest
        assert "0000000000000001" not in loaded

    def test_user_authored_quarantine_placeholder_text_remains_lossless_without_filters(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_quarantine_placeholder_literal.db",
        )
        placeholder = (
            "[LCM active replay placeholder: assistant output quarantined; "
            "kind=quarantined_assistant_output; reason=high_repetition; "
            "scope=ignored_message_pattern; field=content; chars=10; bytes=10; "
            "sha256=0123456789abcdef]"
        )

        engine._ingest_messages([{"role": "assistant", "content": placeholder}])
        stored = engine._store.get_session_messages("user-123")

        assert [row["content"] for row in stored] == [placeholder]
        assert engine._ignored_message_count == 0

    def test_user_authored_quarantine_placeholder_text_can_be_summarized_losslessly(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_quarantine_placeholder_literal_compaction.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        captured: dict[str, str] = {}
        placeholder = (
            "[LCM active replay placeholder: assistant output quarantined; "
            "kind=quarantined_assistant_output; reason=high_repetition; "
            "scope=ignored_message_pattern; field=content; chars=10; bytes=10; "
            "sha256=0123456789abcdef]"
        )

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "literal quarantine placeholder summary\n[Expand for details: literal quarantine placeholder]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        engine.compress(
            [
                {"role": "assistant", "content": placeholder},
                {"role": "user", "content": "visible backlog objective " + "y" * 200},
                {"role": "assistant", "content": "fresh tail response"},
            ],
            current_tokens=10_000,
        )

        assert "assistant output quarantined" in captured["text"]
        assert "visible backlog objective" in captured["text"]

    def test_user_authored_ignored_placeholder_text_can_be_summarized_losslessly(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_placeholder_literal_compaction.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["^Cronjob Response:"],

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        captured: dict[str, str] = {}
        placeholder = (
            "[LCM active replay placeholder: message ignored; kind=ignored_message; "
            "scope=ignored_message_pattern; field=content; chars=10; bytes=10; "
            "sha256=0123456789abcdef]"
        )

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "literal placeholder summary\n[Expand for details: literal placeholder]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        engine.compress(
            [
                {"role": "user", "content": placeholder},
                {"role": "user", "content": "visible backlog objective " + "y" * 200},
                {"role": "assistant", "content": "fresh tail response"},
            ],
            current_tokens=10_000,
        )

        assert "LCM active replay placeholder: message ignored" in captured["text"]
        assert "visible backlog objective" in captured["text"]

    def test_ignored_only_backlog_does_not_leave_assistant_first_context(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_assistant_first.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10_000,
            ignore_message_patterns=["SECRET"],
        )
        messages = [
            {"role": "user", "content": "SECRET ignored cron/private turn"},
            {"role": "assistant", "content": "assistant response to ignored turn"},
        ]

        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
        result_text = "\n".join(str(msg.get("content", "")) for msg in result)

        assert "SECRET" not in result_text
        assert not result or result[0].get("role") != "assistant"

    def test_ignored_fresh_tail_does_not_feed_auto_focus(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_fresh_tail_focus.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            captured["focus_topic"] = kwargs["focus_topic"]
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        messages = [
            {"role": "user", "content": "visible backlog objective " + "y" * 200},
            {"role": "user", "content": "SECRET ignored fresh tail must not become focus"},
        ]

        engine.compress(messages, current_tokens=count_messages_tokens(messages))

        assert "visible backlog objective" in captured["text"]
        assert "SECRET" not in captured["text"]
        assert "visible backlog objective" in captured["focus_topic"]
        assert "SECRET" not in captured["focus_topic"]

    def test_triple_bracket_wrapper_variant_dropped(self, tmp_path):
        engine = self._make_engine(
            tmp_path, "lcm_msg_triple.db",
            ignore_message_patterns=["^>>>Cronjob Response<<<:"],
        )
        messages = [
            {"role": "user", "content": ">>>Cronjob Response<<<: heartbeat"},
            {"role": "user", "content": "regular question"},
        ]
        engine._ingest_messages(messages)
        assert engine._store.get_session_count("user-123") == 1
        assert engine._ignored_message_count == 1

    def test_inline_flag_pattern_drops_both_wrapper_variants(self, tmp_path):
        engine = self._make_engine(
            tmp_path, "lcm_msg_inline.db",
            ignore_message_patterns=[r"(?is)^\s*(>>>\s*)?Cronjob Response"],
        )
        messages = [
            {"role": "user", "content": "Cronjob Response: heartbeat"},
            {"role": "user", "content": "  >>> Cronjob Response: heartbeat"},
            {"role": "user", "content": "non-matching content"},
        ]
        engine._ingest_messages(messages)
        assert engine._store.get_session_count("user-123") == 1
        assert engine._ignored_message_count == 2

    def test_active_pattern_does_not_regress_normal_messages(self, tmp_path):
        engine = self._make_engine(
            tmp_path, "lcm_msg_normal.db",
            ignore_message_patterns=["^Cronjob Response:"],
        )
        messages = [
            {"role": "user", "content": "Can you check the database for me?"},
            {"role": "assistant", "content": "Looking now."},
        ]
        engine._ingest_messages(messages)
        stored_contents = [
            row["content"] for row in engine._store.get_session_messages("user-123")
        ]
        assert stored_contents == [
            "Can you check the database for me?",
            "Looking now.",
        ]
        assert engine._ignored_message_count == 0

    def test_anchored_pattern_matches_multimodal_text_part_content(self, tmp_path):
        engine = self._make_engine(
            tmp_path, "lcm_msg_multimodal_anchored.db",
            ignore_message_patterns=["^Cronjob Response:"],
        )
        multimodal = {
            "role": "user",
            "content": [{"type": "text", "text": "Cronjob Response: heartbeat"}],
        }
        engine._ingest_messages([multimodal])
        assert engine._store.get_session_count("user-123") == 0
        assert engine._ignored_message_count == 1

    def test_anchored_pattern_matches_structured_text_value_parts(self, tmp_path):
        engine = self._make_engine(
            tmp_path, "lcm_msg_multimodal_text_value.db",
            ignore_message_patterns=["^Cronjob Response:"],
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "output_text",
                        "text": {"value": "Cronjob Response: nested text"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "content": "Cronjob Response: content field"}
                ],
            },
        ]
        engine._ingest_messages(messages)
        assert engine._store.get_session_count("user-123") == 0
        assert engine._ignored_message_count == 2

    def test_structured_content_without_text_parts_falls_back_to_normalized_json(self, tmp_path):
        engine = self._make_engine(
            tmp_path, "lcm_msg_multimodal_json_fallback.db",
            ignore_message_patterns=["file_123"],
        )
        multimodal = {
            "role": "user",
            "content": [{"type": "input_file", "file_id": "file_123"}],
        }
        engine._ingest_messages([multimodal])
        assert engine._store.get_session_count("user-123") == 0
        assert engine._ignored_message_count == 1

    def test_unanchored_pattern_matches_multimodal_content(self, tmp_path):
        engine = self._make_engine(
            tmp_path, "lcm_msg_multimodal_substr.db",
            ignore_message_patterns=["Cronjob Response:"],
        )
        multimodal = {
            "role": "user",
            "content": [{"type": "text", "text": "Cronjob Response: heartbeat"}],
        }
        engine._ingest_messages([multimodal])
        assert engine._store.get_session_count("user-123") == 0
        assert engine._ignored_message_count == 1

    def test_filter_is_role_agnostic(self, tmp_path):
        engine = self._make_engine(
            tmp_path, "lcm_msg_roles.db",
            ignore_message_patterns=["^Cronjob Response:"],
        )
        messages = [
            {"role": "tool", "content": "Cronjob Response: tool-emitted"},
            {"role": "assistant", "content": "Cronjob Response: assistant-quoted"},
            {"role": "user", "content": "user-normal"},
        ]
        engine._ingest_messages(messages)
        assert engine._store.get_session_count("user-123") == 1
        assert engine._ignored_message_count == 2

    def test_invalid_regex_warned_and_surviving_pattern_still_filters(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="hermes_lcm.message_patterns"):
            engine = self._make_engine(
                tmp_path, "lcm_msg_invalid.db",
                ignore_message_patterns=["[unclosed", "^Cronjob Response:"],
            )

        assert "skipping invalid regex" in caplog.text
        assert "[unclosed" in caplog.text

        engine._ingest_messages([
            {"role": "user", "content": "Cronjob Response: heartbeat"},
            {"role": "user", "content": "normal text"},
        ])
        assert engine._store.get_session_count("user-123") == 1
        assert engine._ignored_message_count == 1

    def test_session_filter_and_message_filter_coexist(self, tmp_path):
        # Session-level filter blocks all writes for a matched session,
        # taking precedence over per-message filtering.
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_both_filters.db"),
            ignore_session_patterns=["cron:*"],
            ignore_message_patterns=["^Cronjob Response:"],
        )
        engine = LCMEngine(config=config)
        engine.on_session_start("cron_123", platform="cron", context_length=1000)
        engine._ingest_messages([
            {"role": "user", "content": "Cronjob Response: heartbeat"},
            {"role": "user", "content": "anything"},
        ])
        assert engine._store.get_session_count("cron_123") == 0
        # Counter does not increment for ignored sessions: ingest short-circuits before the message filter runs.
        assert engine._ignored_message_count == 0

        # On a normal-platform session, only the message filter applies.
        engine.on_session_start("user-123", platform="telegram", context_length=1000)
        engine._ingest_messages([
            {"role": "user", "content": "Cronjob Response: heartbeat"},
            {"role": "user", "content": "regular conversation"},
        ])
        assert engine._store.get_session_count("user-123") == 1
        assert engine._ignored_message_count == 1

    def test_missing_regex_dependency_leaves_messages_unfiltered(self, tmp_path, monkeypatch, caplog):
        from hermes_lcm import message_patterns as message_patterns_mod

        monkeypatch.setattr(message_patterns_mod, "_regex_engine", None)
        monkeypatch.setattr(message_patterns_mod, "_MISSING_REGEX_WARNING_EMITTED", False)

        with caplog.at_level("WARNING", logger="hermes_lcm.message_patterns"):
            engine = self._make_engine(
                tmp_path,
                "lcm_msg_no_regex.db",
                ignore_message_patterns=[r"(a+)+$"],
                ignore_message_patterns_source="env",
            )

        engine._ingest_messages([
            {"role": "user", "content": "a" * 30 + "!"},
        ])

        assert engine._store.get_session_count("user-123") == 1
        assert engine._ignored_message_count == 0
        assert "regex" in caplog.text
        assert "disabled" in caplog.text

    def test_status_surfaces_message_pattern_keys(self, tmp_path):
        engine = self._make_engine(
            tmp_path, "lcm_msg_status.db",
            ignore_message_patterns=["^Cronjob Response:"],
            ignore_message_patterns_source="env",
        )
        engine._ingest_messages([
            {"role": "user", "content": "Cronjob Response: heartbeat"},
        ])
        status = engine.get_status()
        assert status["ignore_message_patterns"] == ["^Cronjob Response:"]
        assert status["ignore_message_patterns_source"] == "env"
        assert status["ignored_message_count"] == 1
        # Existing session-pattern keys are still surfaced unchanged.
        assert status["ignore_session_patterns"] == []
        assert status["ignore_session_patterns_source"] == "default"

    def test_diagnostic_log_emits_once_per_engine(self, tmp_path, caplog):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_msg_log_once.db"),
            ignore_message_patterns=["^Cronjob Response:"],
            ignore_message_patterns_source="env",
        )
        engine = LCMEngine(config=config)

        with caplog.at_level("INFO", logger="hermes_lcm.engine"):
            engine.on_session_start("user-123", platform="telegram", context_length=1000)
            engine.on_session_start("user-456", platform="telegram", context_length=1000)

        assert caplog.text.count("LCM ignore_message_patterns from env: ^Cronjob Response:") == 1

    def test_stateless_session_skips_message_filter_entirely(self, tmp_path):
        # Stateless sessions short-circuit ingest before the message filter runs,
        # mirroring the ignored-session contract. The counter must not increment
        # for a stateless session even when patterns would otherwise match.
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_stateless_msg.db"),
            stateless_session_patterns=["telegram:*"],
            ignore_message_patterns=["^Cronjob Response:"],
        )
        engine = LCMEngine(config=config)
        engine.on_session_start("debug", platform="telegram", context_length=1000)
        engine._ingest_messages([
            {"role": "user", "content": "Cronjob Response: heartbeat"},
            {"role": "user", "content": "anything"},
        ])
        assert engine._store.get_session_count("debug") == 0
        assert engine._ignored_message_count == 0

    def test_cursor_advances_when_filter_drops_entire_batch(self, tmp_path):
        # When every message in a batch matches a filter pattern, _ingest_cursor
        # must still advance to len(messages). Otherwise a second call with the
        # same list would re-evaluate every message and double-increment the
        # counter. Regression guard for the all-filtered early-return path.
        engine = self._make_engine(
            tmp_path, "lcm_msg_cursor_all_filtered.db",
            ignore_message_patterns=["^Cronjob Response:"],
        )
        messages = [
            {"role": "user", "content": "Cronjob Response: alpha"},
            {"role": "user", "content": "Cronjob Response: beta"},
        ]
        engine._ingest_messages(messages)
        assert engine._store.get_session_count("user-123") == 0
        assert engine._ignored_message_count == 2
        assert engine._ingest_cursor == len(messages)

        # Second call with the same list must not re-process the messages.
        engine._ingest_messages(messages)
        assert engine._ignored_message_count == 2
        assert engine._ingest_cursor == len(messages)

    def test_restart_reconciliation_skips_ignored_messages_when_matching_store_tail(self, tmp_path):
        db_path = tmp_path / "lcm_msg_restart_tail.db"
        config = LCMConfig(
            database_path=str(db_path),
            ignore_message_patterns=["^Cronjob Response:"],
        )

        before_restart = LCMEngine(config=config)
        before_restart.on_session_start(
            "user-123",
            platform="telegram",
            context_length=1000,
            conversation_id="chat-1",
        )
        active_context = [
            {"role": "user", "content": "first real message"},
            {"role": "user", "content": "Cronjob Response: heartbeat"},
            {"role": "assistant", "content": "real answer"},
        ]
        before_restart._ingest_messages(active_context)
        before_restart._store.close()
        before_restart._dag.close()
        before_restart._lifecycle.close()

        after_restart = LCMEngine(config=config)
        after_restart.on_session_start(
            "user-123",
            platform="telegram",
            context_length=1000,
            conversation_id="chat-1",
        )
        after_restart._ingest_messages(active_context)

        rows = after_restart._store.get_session_messages("user-123")
        assert [row["content"] for row in rows] == ["first real message", "real answer"]
        assert after_restart._ingest_cursor == len(active_context)

    def test_restart_reconciliation_skips_historical_ignored_rows_when_filter_enabled_later(self, tmp_path):
        db_path = tmp_path / "lcm_msg_restart_later_filter.db"
        before_config = LCMConfig(database_path=str(db_path))

        before_filter = LCMEngine(config=before_config)
        before_filter.on_session_start(
            "user-123",
            platform="telegram",
            context_length=1000,
            conversation_id="chat-1",
        )
        active_context = [
            {"role": "user", "content": "first real message"},
            {"role": "user", "content": "Cronjob Response: heartbeat"},
            {"role": "assistant", "content": "real answer"},
        ]
        before_filter._ingest_messages(active_context)
        before_filter._store.close()
        before_filter._dag.close()
        before_filter._lifecycle.close()

        after_config = LCMConfig(
            database_path=str(db_path),
            ignore_message_patterns=["^Cronjob Response:"],
        )
        after_filter_restart = LCMEngine(config=after_config)
        after_filter_restart.on_session_start(
            "user-123",
            platform="telegram",
            context_length=1000,
            conversation_id="chat-1",
        )
        after_filter_restart._ingest_messages(active_context)

        rows = after_filter_restart._store.get_session_messages("user-123")
        assert [row["content"] for row in rows] == [
            "first real message",
            "Cronjob Response: heartbeat",
            "real answer",
        ]
        assert after_filter_restart._ingest_cursor == len(active_context)

    def test_restart_reconciliation_matches_legacy_stored_json_with_text_first_filter(self, tmp_path):
        db_path = tmp_path / "lcm_msg_legacy_multimodal_reconcile.db"
        session_id = "legacy-structured-session"
        active_context = [
            {"role": "user", "content": "normal before ignored tail"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Cronjob Response: heartbeat"}],
            },
        ]

        before_restart = LCMEngine(config=LCMConfig(database_path=str(db_path)))
        before_restart.on_session_start(
            session_id,
            platform="telegram",
            conversation_id="legacy-structured-conversation",
            context_length=1000,
        )
        before_restart._ingest_messages(active_context)
        before_restart._store.close()
        before_restart._dag.close()
        before_restart._lifecycle.close()

        after_restart = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                ignore_message_patterns=["^Cronjob Response:"],
            )
        )
        after_restart.on_session_start(
            session_id,
            platform="telegram",
            conversation_id="legacy-structured-conversation",
            context_length=1000,
        )
        after_restart._ingest_messages(active_context)

        rows = after_restart._store.get_session_messages(session_id)
        assert [row["content"] for row in rows] == [
            "normal before ignored tail",
            '[{"text": "Cronjob Response: heartbeat", "type": "text"}]',
        ]
        assert after_restart._ingest_cursor == len(active_context)
