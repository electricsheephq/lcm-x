"""LCM engine integration tests: filter replay."""


import pytest

import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

from .engine_helpers import (
    TestMessageFiltering as _MessageFilteringHelpers,
)


class TestMessageFiltering:
    _timeout_capable_regex_engine = _MessageFilteringHelpers.__dict__["_timeout_capable_regex_engine"]
    _make_engine = _MessageFilteringHelpers.__dict__["_make_engine"]

    def test_generated_ignored_active_replay_placeholder_filtered_without_active_patterns(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_placeholder_no_patterns.db",
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
        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "visible backlog summary\n[Expand for details: visible backlog]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        first_result = engine.compress(
            [
                {"role": "user", "content": "first visible backlog " + "y" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
            ],
            current_tokens=10_000,
        )
        assert "LCM active replay placeholder: message ignored" in "\n".join(
            str(msg.get("content", "")) for msg in first_result
        )

        engine.shutdown()
        second = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_placeholder_no_patterns.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,

            # #1013 part B: protection tests cover the off path (owner scope ban)
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False,
        )
        captured.clear()
        second.compress(
            first_result
            + [
                {"role": "user", "content": "second visible backlog " + "z" * 200},
                {"role": "assistant", "content": "fresh tail response"},
            ],
            current_tokens=10_000,
        )

        assert "second visible backlog" in captured["text"]
        assert "LCM active replay placeholder: message ignored" not in captured["text"]
        assert all(
            "LCM active replay placeholder: message ignored" not in str(row["content"])
            for row in second._store.get_session_messages("user-123")
        )

    def test_known_ignored_placeholder_replay_is_not_stored_after_restart(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_placeholder_restart_replay.db"
        first = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_placeholder_restart_replay.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        first_result = first.compress(
            [
                {"role": "user", "content": "visible backlog " + "v" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
            ],
            current_tokens=10_000,
        )
        first.shutdown()
        assert "LCM active replay placeholder: message ignored" in "\n".join(
            str(msg.get("content", "")) for msg in first_result
        )

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start("user-123", platform="telegram", context_length=1000)
            second._ingest_messages(first_result)

            assert all(
                "LCM active replay placeholder: message ignored" not in str(row["content"])
                for row in second._store.get_session_messages("user-123")
            )
        finally:
            second.shutdown()

    def test_restart_reconciliation_preserves_literal_placeholder_with_known_digest(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_restart_literal_known_digest.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        placeholder = first._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
        digest = first._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        first._store.append("session", {"role": "user", "content": "old row"})
        first._remember_generated_ignored_placeholder_hash(digest)
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            second._ingest_messages([{"role": "user", "content": placeholder}])

            rows = second._store.get_session_messages("session")
            assert [row["content"] for row in rows] == ["old row", placeholder]
        finally:
            second.shutdown()

    def test_compacted_restart_preserves_first_delta_literal_placeholder_with_known_digest(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_compacted_restart_literal_first_delta.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        placeholder = first._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
        digest = first._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        old_store_id = first._store.append("session", {"role": "user", "content": "old compacted row"})
        first._last_compacted_store_id = old_store_id
        first._persist_frontier_marker()
        first._remember_generated_ignored_placeholder_hash(digest)
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            assert second._last_compacted_store_id == old_store_id
            second._ingest_messages([{"role": "user", "content": placeholder}])

            rows = second._store.get_session_messages("session")
            assert [row["content"] for row in rows] == ["old compacted row", placeholder]
            assert second._last_ingest_reconciliation["reason"] == "persisted ambiguous delta"
        finally:
            second.shutdown()

    def test_restart_reconciliation_keeps_stored_literal_placeholder_in_tail_match(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_restart_literal_tail_match.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        placeholder = first._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
        digest = first._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        first._store.append("session", {"role": "user", "content": "old row"})
        first._store.append("session", {"role": "user", "content": placeholder})
        first._remember_generated_ignored_placeholder_hash(digest)
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            second._ingest_messages(
                [
                    {"role": "user", "content": "old row"},
                    {"role": "user", "content": placeholder},
                ]
            )

            rows = second._store.get_session_messages("session")
            assert [row["content"] for row in rows] == ["old row", placeholder]
            assert second._ingest_cursor == 2
        finally:
            second.shutdown()

    def test_user_quoted_generated_placeholder_is_stored_losslessly(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_user_quoted_placeholder.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        first_result = engine.compress(
            [
                {"role": "user", "content": "visible backlog objective " + "y" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
            ],
            current_tokens=10_000, force=True,  # manual seed; synthetic pressure is not a host estimate
        )
        placeholder = next(
            str(msg.get("content", ""))
            for msg in first_result
            if "LCM active replay placeholder: message ignored" in str(msg.get("content", ""))
        )

        engine._ingest_messages(first_result + [{"role": "user", "content": placeholder}])

        rows = engine._store.get_session_messages("user-123")
        assert any(row["content"] == placeholder for row in rows)

    def test_stored_placeholder_quote_does_not_declassify_generated_placeholder(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_placeholder_quote_does_not_declassify.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
        )
        placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef ignored active turn")
        digest = engine._active_replay_placeholder_digest(placeholder)
        assert digest
        engine._remember_generated_ignored_placeholder_hash(digest)

        quoted_placeholder = {"role": "user", "content": placeholder}
        engine._ingest_messages([quoted_placeholder])
        rows = engine._store.get_session_messages("user-123")
        assert any(row["content"] == placeholder for row in rows)

        generated_placeholder = {"role": "user", "content": placeholder}
        assert engine._is_ignored_active_replay_placeholder(generated_placeholder, placeholder) is True

        store_map = engine._get_store_id_map_for_messages([generated_placeholder, quoted_placeholder])
        assert id(generated_placeholder) not in store_map
        assert id(quoted_placeholder) in store_map

        generated_placeholder_after_quote = {"role": "user", "content": placeholder}
        engine._generated_ignored_active_replay_placeholder_message_ids.add(
            id(generated_placeholder_after_quote)
        )
        store_map = engine._get_store_id_map_for_messages(
            [quoted_placeholder, generated_placeholder_after_quote]
        )
        assert id(quoted_placeholder) in store_map
        assert id(generated_placeholder_after_quote) not in store_map

        engine._current_compress_store_ids_by_message_id = {id(quoted_placeholder): 1}
        assert engine._is_ignored_active_replay_placeholder(quoted_placeholder, placeholder) is False

    def test_preflight_keeps_stored_placeholder_literal_candidate(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_placeholder_preflight_literal.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            ignore_message_patterns=[r"NEVER_MATCH_THIS_PATTERN"],
        )
        placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef ignored active turn")
        digest = engine._active_replay_placeholder_digest(placeholder)
        assert digest
        engine._remember_generated_ignored_placeholder_hash(digest)
        quoted_placeholder = {"role": "user", "content": placeholder}
        engine._store.append("user-123", quoted_placeholder)

        eligible, reason = engine._leaf_compaction_candidate_status(
            [quoted_placeholder, {"role": "assistant", "content": "fresh tail"}]
        )

        assert eligible is True
        assert reason == "eligible raw backlog outside fresh tail"

    def test_new_session_carry_over_does_not_poison_quoted_placeholder_backlog(self, tmp_path, monkeypatch):
        db_path = tmp_path / "lcm_msg_ignore_placeholder_new_session_no_hash_poison.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
                sensitive_patterns_enabled=True,
                sensitive_patterns=["api_key"],
                threshold_full_sweep_enabled=False,  # #1013: pin the pre-#1013 single-pass compaction

                # #1013 part B: protection tests cover the off path (owner scope ban)
                large_output_externalization_enabled=False,
                large_output_active_replay_stubbing_enabled=False,
                temporal_rollups_enabled=False,
            )
        )
        first.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-placeholder-normal-carry",
        )
        first_result = first.compress(
            [
                {"role": "user", "content": "visible backlog before placeholder " + "v" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
            ],
            current_tokens=10_000,
        )
        placeholder = next(
            str(msg.get("content", ""))
            for msg in first_result
            if "LCM active replay placeholder: message ignored" in str(msg.get("content", ""))
        )
        first.carry_over_new_session_context("old-session", "new-session")
        first.shutdown()

        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "visible summary\n[Expand for details: visible summary]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                threshold_full_sweep_enabled=False,  # #1013: pin the pre-#1013 single-pass compaction

                # #1013 part B: protection tests cover the off path (owner scope ban)
                large_output_externalization_enabled=False,
                large_output_active_replay_stubbing_enabled=False,
                temporal_rollups_enabled=False,
            )
        )
        try:
            second.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-placeholder-normal-carry",
            )
            second.compress(
                [
                    {"role": "user", "content": placeholder},
                    {"role": "user", "content": "visible new backlog " + "z" * 200},
                    {"role": "assistant", "content": "fresh tail response"},
                ],
                current_tokens=10_000,
            )

            assert "LCM active replay placeholder: message ignored" in captured["text"]
            assert any(row["content"] == placeholder for row in second._store.get_session_messages("new-session"))
        finally:
            second.shutdown()

    def test_rollover_session_normal_new_does_not_poison_literal_placeholder_backlog(self, tmp_path, monkeypatch):
        db_path = tmp_path / "lcm_msg_ignore_placeholder_rollover_session_normal_new.db"
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
                sensitive_patterns_enabled=True,
                sensitive_patterns=["api_key"],
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
            conversation_id="conv-placeholder-rollover-normal-new",
        )
        first_result = engine.compress(
            [
                {"role": "user", "content": "visible backlog before normal new " + "v" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
            ],
            current_tokens=10_000,
        )
        placeholder = next(
            str(msg.get("content", ""))
            for msg in first_result
            if "LCM active replay placeholder: message ignored" in str(msg.get("content", ""))
        )
        moved = engine.rollover_session(
            "old-session",
            "new-session",
            previous_messages=first_result,
            platform="telegram",
            context_length=1000,
        )
        assert moved >= 0
        assert engine._session_id == "new-session"

        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "literal placeholder summary\n[Expand for details: literal placeholder]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        # #628: the unstubbed first compaction had no summariser and opened the circuit, which now pauses
        # the next compaction; this pass stubs the summariser, so close the circuit first.
        engine._summary_circuit_breaker.record_success(engine._config.summary_model)
        engine._config.ignore_message_patterns = []
        engine._compiled_ignore_message_patterns = []
        engine.compress(
            [
                {"role": "user", "content": placeholder},
                {"role": "user", "content": "visible normal new backlog " + "z" * 200},
                {"role": "assistant", "content": "fresh tail"},
            ],
            current_tokens=10_000,
        )

        assert "LCM active replay placeholder: message ignored" in captured["text"]
        assert any(row["content"] == placeholder for row in engine._store.get_session_messages("new-session"))
        engine.shutdown()

    def test_ignored_placeholder_hash_survives_compression_rollover_restart(self, tmp_path, monkeypatch):
        db_path = tmp_path / "lcm_msg_ignore_placeholder_rollover.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
                sensitive_patterns_enabled=True,
                sensitive_patterns=["api_key"],
                threshold_full_sweep_enabled=False,  # #1013: pin the pre-#1013 single-pass compaction

                # #1013 part B: protection tests cover the off path (owner scope ban)
                large_output_externalization_enabled=False,
                large_output_active_replay_stubbing_enabled=False,
                temporal_rollups_enabled=False,
            )
        )
        first.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-placeholder-rollover",
        )
        first_result = first.compress(
            [
                {"role": "user", "content": "visible backlog before rollover " + "v" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
            ],
            current_tokens=10_000,
        )
        first.shutdown()

        assert "LCM active replay placeholder: message ignored" in "\n".join(
            str(msg.get("content", "")) for msg in first_result
        )

        captured: dict[str, str] = {}

        def capture_summary(**kwargs):
            captured["text"] = kwargs["text"]
            return "visible new summary\n[Expand for details: visible new]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", capture_summary)
        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                threshold_full_sweep_enabled=False,  # #1013: pin the pre-#1013 single-pass compaction

                # #1013 part B: protection tests cover the off path (owner scope ban)
                large_output_externalization_enabled=False,
                large_output_active_replay_stubbing_enabled=False,
                temporal_rollups_enabled=False,
            )
        )
        try:
            second.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-placeholder-rollover",
                boundary_reason="compression",
                old_session_id="old-session",
            )
            second.compress(
                first_result
                + [
                    {"role": "user", "content": "visible new backlog " + "z" * 200},
                    {"role": "assistant", "content": "fresh tail response"},
                ],
                current_tokens=10_000,
            )

            assert "visible new backlog" in captured["text"]
            assert "LCM active replay placeholder: message ignored" not in captured["text"]
            assert all(
                "LCM active replay placeholder: message ignored" not in str(row["content"])
                for row in second._store.get_session_messages("new-session")
            )
        finally:
            second.shutdown()

    def test_carried_ignored_placeholder_in_fresh_tail_is_not_stored_after_rollover(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_placeholder_rollover_fresh_tail.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
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
        )
        first.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-placeholder-rollover-fresh-tail",
        )
        first_result = first.compress(
            [
                {"role": "user", "content": "visible backlog before rollover " + "v" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
            ],
            current_tokens=10_000,
        )
        first.shutdown()

        assert len(first_result) <= 10
        assert "LCM active replay placeholder: message ignored" in "\n".join(
            str(msg.get("content", "")) for msg in first_result
        )

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,

                # #1013 part B: protection tests cover the off path (owner scope ban)
                large_output_externalization_enabled=False,
                large_output_active_replay_stubbing_enabled=False,
                temporal_rollups_enabled=False,
            )
        )
        try:
            second.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-placeholder-rollover-fresh-tail",
                boundary_reason="compression",
                old_session_id="old-session",
            )
            second._ingest_messages(first_result)

            assert all(
                "LCM active replay placeholder: message ignored" not in str(row["content"])
                for row in second._store.get_session_messages("new-session")
            )
        finally:
            second.shutdown()

    @pytest.mark.parametrize("externalized", [
        False,
        pytest.param(True, marks=pytest.mark.xfail(
            strict=True, raises=AssertionError,
            reason="#1053: with externalization + active-replay stubbing on, the literal is stored twice")),
    ], ids=["externalization-off", "externalization-on"])
    def test_new_placeholder_literal_after_rollover_is_stored_losslessly(self, tmp_path, externalized):
        db_path = tmp_path / "lcm_msg_ignore_placeholder_rollover_new_literal.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=1,
                leaf_chunk_tokens=10,
                ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
                sensitive_patterns_enabled=True,
                sensitive_patterns=["api_key"],
                large_output_externalization_enabled=externalized,
                large_output_active_replay_stubbing_enabled=externalized,
            )
        )
        first.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-placeholder-rollover-new-literal",
        )
        first_result = first.compress(
            [
                {"role": "user", "content": "visible backlog before rollover " + "v" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
            ],
            current_tokens=10_000,
        )
        first.shutdown()
        placeholder = next(
            str(msg.get("content", ""))
            for msg in first_result
            if "LCM active replay placeholder: message ignored" in str(msg.get("content", ""))
        )

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-placeholder-rollover-new-literal",
                boundary_reason="compression",
                old_session_id="old-session",
            )
            second._ingest_messages(first_result + [{"role": "user", "content": placeholder}])

            rows = second._store.get_session_messages("new-session")
            assert [row["content"] for row in rows].count(placeholder) == 1
        finally:
            second.shutdown()

    def test_source_stored_placeholder_literal_after_frontier_is_preserved_after_rollover_restart(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_placeholder_source_literal_after_frontier.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
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
        )
        first.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-placeholder-source-literal-after-frontier",
        )
        first_result = first.compress(
            [
                {"role": "user", "content": "visible backlog before rollover " + "v" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef ignored fresh tail"},
            ],
            current_tokens=10_000,
        )
        placeholder = next(
            str(msg.get("content", ""))
            for msg in first_result
            if "LCM active replay placeholder: message ignored" in str(msg.get("content", ""))
        )
        first._store.append("old-session", {"role": "user", "content": placeholder})
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,

                # #1013 part B: protection tests cover the off path (owner scope ban)
                large_output_externalization_enabled=False,
                large_output_active_replay_stubbing_enabled=False,
                temporal_rollups_enabled=False,
            )
        )
        try:
            second.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-placeholder-source-literal-after-frontier",
                boundary_reason="compression",
                old_session_id="old-session",
            )
            second._ingest_messages(first_result + [{"role": "user", "content": placeholder}])

            rows = second._store.get_session_messages("new-session")
            assert [row["content"] for row in rows].count(placeholder) == 1
        finally:
            second.shutdown()

    def test_stored_placeholder_after_frontier_keeps_rollover_literal(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_placeholder_budget_stored_after_frontier.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        first.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-placeholder-budget-stored-after-frontier",
        )
        placeholder = first._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
        digest = first._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        first._remember_generated_ignored_placeholder_hash(digest)
        first._write_generated_ignored_placeholder_hash_counts({digest: 1})
        first._write_generated_ignored_placeholder_hash_ordinals({digest: {1}})
        first._store.append("old-session", {"role": "user", "content": placeholder})
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-placeholder-budget-stored-after-frontier",
                boundary_reason="compression",
                old_session_id="old-session",
            )
            second._ingest_messages(
                [
                    {
                        "role": "user",
                        "content": "[Current user objective preserved from compacted history]\ncarry objective",
                    },
                    {"role": "user", "content": placeholder},
                ]
            )

            rows = second._store.get_session_messages("new-session")
            assert [row["content"] for row in rows].count(placeholder) == 1
        finally:
            second.shutdown()

    def test_cached_generated_placeholder_copy_does_not_steal_stored_literal_mapping(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_cached_generated_copy.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start(
                "session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-cached-generated-copy",
            )
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            literal = {"role": "user", "content": placeholder}
            generated = {"role": "user", "content": placeholder}
            literal_store_id = engine._store.append("session", literal)
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._generated_ignored_active_replay_placeholder_message_ids.add(id(generated))

            engine._remember_active_replay_messages(
                [literal, generated],
                [literal, generated],
            )
            cached_literal, cached_generated = engine._last_active_replay_messages
            ids_by_message_id = engine._get_store_id_map_for_messages(
                engine._last_active_replay_messages
            )

            assert ids_by_message_id.get(id(cached_literal)) == literal_store_id
            assert id(cached_generated) not in ids_by_message_id
            assert engine._active_replay_generated_placeholder_digest_budget() == {digest: 1}

            engine._ingest_cursor = 2
            active = [literal, generated, {"role": "user", "content": "new turn"}]
            engine._ingest_messages(active)
            assert engine._active_replay_generated_placeholder_digest_budget() == {digest: 1}
        finally:
            engine.shutdown()

    def test_copying_replay_to_insert_new_ignored_placeholder_preserves_existing_generated_ids(self, tmp_path):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_copy_existing_generated_id.db",
            ignore_message_patterns=["SECRET"],
        )
        placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
        digest = engine._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        literal = {"role": "user", "content": placeholder}
        generated = {"role": "user", "content": placeholder}
        raw_ignored = {"role": "user", "content": "SECRET later ignored"}
        engine._store.append("user-123", literal)
        engine._remember_generated_ignored_placeholder_hash(digest)
        engine._generated_ignored_active_replay_placeholder_message_ids.add(id(generated))

        active_replay = engine._apply_ignored_active_replay_placeholders(
            [literal, generated, raw_ignored],
            [literal, generated, raw_ignored],
        )
        copied_literal, copied_generated, copied_new_placeholder = active_replay
        ids_by_message_id = engine._get_store_id_map_for_messages(active_replay)

        assert copied_new_placeholder["content"].startswith("[LCM active replay placeholder: message ignored;")
        assert id(copied_generated) in engine._generated_ignored_active_replay_placeholder_message_ids
        assert ids_by_message_id.get(id(copied_literal)) is not None
        assert id(copied_generated) not in ids_by_message_id

    def test_zero_row_restart_replays_generated_placeholder_from_count_metadata(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_zero_row_restart_placeholder.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        placeholder = first._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
        digest = first._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        first._remember_generated_ignored_placeholder_hash(digest)
        first._write_generated_ignored_placeholder_hash_counts({digest: 1})
        first._write_generated_ignored_placeholder_hash_ordinals({digest: {1}})
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            second._ingest_messages([{"role": "user", "content": placeholder}])

            assert second._store.get_session_messages("session")[0]["content"] == placeholder
            assert second._ingest_cursor == 1
        finally:
            second.shutdown()

    def test_zero_row_suffix_literal_uses_full_replay_ordinals(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_zero_row_suffix_literal.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start("session", platform="telegram", context_length=1000)
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._write_generated_ignored_placeholder_hash_counts({digest: 1})
            engine._write_generated_ignored_placeholder_hash_ordinals({digest: {1}})
            engine._ingest_cursor = 1

            engine._ingest_messages(
                [
                    {"role": "user", "content": placeholder},
                    {"role": "user", "content": placeholder},
                ]
            )

            rows = engine._store.get_session_messages("session")
            assert [row["content"] for row in rows] == [placeholder]
        finally:
            engine.shutdown()

    def test_session_rebind_clears_abandoned_boundary_placeholder_budget(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_abandoned_boundary_budget.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start(
                "old-session",
                platform="telegram",
                context_length=1000,
                conversation_id="old-conv",
            )
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._compression_boundary_ingest_pending = True
            engine._compression_boundary_active_placeholder_digest_budget = {digest: 1}
            engine._compression_boundary_active_placeholder_digest_ordinals = {digest: {1}}
            engine._compression_boundary_stored_placeholder_digest_counts = {digest: 1}

            engine.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="new-conv",
            )
            assert engine._compression_boundary_ingest_pending is False
            assert engine._compression_boundary_active_placeholder_digest_budget == {}
            assert engine._compression_boundary_active_placeholder_digest_ordinals == {}
            assert engine._compression_boundary_stored_placeholder_digest_counts == {}

            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._write_generated_ignored_placeholder_hash_counts({digest: 2})
            engine._write_generated_ignored_placeholder_hash_ordinals({digest: {1, 2}})
            engine._ingest_messages(
                [
                    {"role": "user", "content": placeholder},
                    {"role": "user", "content": placeholder},
                ]
            )

            rows = engine._store.get_session_messages("new-session")
            assert rows == []
        finally:
            engine.shutdown()

    def test_boundary_budget_uses_generated_provenance_when_literal_precedes_generated(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_boundary_literal_before_generated.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start("new-session", platform="telegram", context_length=1000)
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            literal = {"role": "user", "content": placeholder}
            generated = {"role": "user", "content": placeholder}
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._generated_ignored_active_replay_placeholder_message_ids.add(id(generated))
            engine._compression_boundary_ingest_pending = True
            engine._compression_boundary_active_placeholder_digest_budget = {digest: 1}

            engine._ingest_messages(
                [
                    literal,
                    {"role": "user", "content": "visible separator"},
                    generated,
                ]
            )

            rows = engine._store.get_session_messages("new-session")
            assert [row["content"] for row in rows] == [placeholder, "visible separator"]
        finally:
            engine.shutdown()

    def test_boundary_budget_uses_ordinals_after_restart_when_literal_precedes_generated(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_boundary_ordinals_literal_before_generated.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start("new-session", platform="telegram", context_length=1000)
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._compression_boundary_ingest_pending = True
            engine._compression_boundary_active_placeholder_digest_budget = {digest: 1}
            engine._compression_boundary_active_placeholder_digest_ordinals = {digest: {2}}

            engine._ingest_messages(
                [
                    {"role": "user", "content": placeholder},
                    {"role": "user", "content": "visible separator"},
                    {"role": "user", "content": placeholder},
                ]
            )

            rows = engine._store.get_session_messages("new-session")
            assert [row["content"] for row in rows] == [placeholder, "visible separator"]
        finally:
            engine.shutdown()

    def test_boundary_count_only_budget_preserves_ambiguous_placeholder_literals(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_boundary_count_only_ambiguous.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start("new-session", platform="telegram", context_length=1000)
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._compression_boundary_ingest_pending = True
            engine._compression_boundary_active_placeholder_digest_budget = {digest: 1}
            engine._compression_boundary_active_placeholder_digest_ordinals = {}

            engine._ingest_messages(
                [
                    {"role": "user", "content": placeholder},
                    {"role": "user", "content": "visible separator"},
                    {"role": "user", "content": placeholder},
                ]
            )

            rows = engine._store.get_session_messages("new-session")
            assert [row["content"] for row in rows] == [
                placeholder,
                "visible separator",
                placeholder,
            ]
        finally:
            engine.shutdown()

    def test_empty_boundary_placeholder_budget_preserves_literal_placeholder(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_empty_boundary_budget_literal.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-empty-boundary-budget-literal",
            )
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._compression_boundary_ingest_pending = True
            engine._compression_boundary_active_placeholder_digest_budget = {}

            engine._ingest_messages(
                [
                    {"role": "user", "content": placeholder},
                    {"role": "user", "content": "visible separator"},
                ]
            )

            rows = engine._store.get_session_messages("new-session")
            assert [row["content"] for row in rows] == [placeholder, "visible separator"]
        finally:
            engine.shutdown()

    def test_boundary_literal_placeholder_after_normal_turn_is_not_consumed_by_ordinal_metadata(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_boundary_literal_after_normal.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start("new-session", platform="telegram", context_length=1000)
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._compression_boundary_ingest_pending = True
            engine._compression_boundary_active_placeholder_digest_budget = {digest: 1}
            engine._compression_boundary_active_placeholder_digest_ordinals = {digest: {1}}

            engine._ingest_messages(
                [
                    {"role": "user", "content": "actual next user turn"},
                    {"role": "user", "content": placeholder},
                ]
            )

            rows = engine._store.get_session_messages("new-session")
            assert [row["content"] for row in rows] == ["actual next user turn", placeholder]
        finally:
            engine.shutdown()

    def test_empty_session_literal_placeholder_after_normal_turn_is_not_consumed_by_ordinal_metadata(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_empty_session_literal_after_normal.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        first.on_session_start("session", platform="telegram", context_length=1000)
        placeholder = first._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
        digest = first._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        first._remember_generated_ignored_placeholder_hash(digest)
        first._write_generated_ignored_placeholder_hash_counts({digest: 1})
        first._write_generated_ignored_placeholder_hash_ordinals({digest: {1}})
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start("session", platform="telegram", context_length=1000)
            second._ingest_messages(
                [
                    {"role": "user", "content": "actual next user turn"},
                    {"role": "user", "content": placeholder},
                ]
            )

            rows = second._store.get_session_messages("session")
            assert [row["content"] for row in rows] == ["actual next user turn", placeholder]
        finally:
            second.shutdown()

    def test_boundary_singleton_literal_placeholder_is_not_consumed_by_ordinal_metadata(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_boundary_singleton_literal.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-boundary-singleton-literal",
            )
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._compression_boundary_ingest_pending = True
            engine._compression_boundary_active_placeholder_digest_budget = {digest: 1}
            engine._compression_boundary_active_placeholder_digest_ordinals = {digest: {1}}

            engine._ingest_messages([{"role": "user", "content": placeholder}])

            rows = engine._store.get_session_messages("new-session")
            assert [row["content"] for row in rows] == [placeholder]
        finally:
            engine.shutdown()

    def test_cleared_boundary_frontier_does_not_drop_literal_placeholder(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_cleared_boundary_literal.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-cleared-boundary-literal",
            )
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._last_compacted_store_id = 1
            engine._compression_boundary_ingest_pending = False
            engine._compression_boundary_active_placeholder_digest_budget = {}

            engine._ingest_messages([{"role": "user", "content": placeholder}])

            rows = engine._store.get_session_messages("new-session")
            assert [row["content"] for row in rows] == [placeholder]
        finally:
            engine.shutdown()

    def test_compaction_clears_generated_placeholder_count_when_placeholder_not_returned(self, tmp_path, monkeypatch):
        engine = self._make_engine(
            tmp_path,
            "lcm_msg_ignore_compaction_clears_placeholder_count.db",
            fresh_tail_count=1,
            leaf_chunk_tokens=10,
            ignore_message_patterns=["SECRET"],
        )
        ignored_text = "SECRET ignored compactable turn"
        placeholder = engine._ignored_active_replay_placeholder(ignored_text)

        def summary(**kwargs):
            return "visible summary\n[Expand for details: visible]", 1

        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary)
        result = engine.compress(
            [
                {"role": "user", "content": "visible backlog before ignored " + "v" * 200},
                {"role": "user", "content": ignored_text},
                {"role": "user", "content": "fresh tail"},
            ],
            current_tokens=10_000, force=True,  # manual seed; synthetic pressure is not a host estimate
        )

        assert all("LCM active replay placeholder: message ignored" not in str(msg.get("content", "")) for msg in result)
        assert engine._load_generated_ignored_placeholder_hash_counts() == {}

        engine.on_session_start(
            "new-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-cleared-count-literal",
            boundary_reason="compression",
            old_session_id="user-123",
        )
        engine._ingest_messages(result + [{"role": "user", "content": placeholder}])
        rows = engine._store.get_session_messages("new-session")
        assert any(row["content"] == placeholder for row in rows)

    def test_empty_ingest_clears_pending_boundary_placeholder_budget(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_empty_ingest_clears_budget.db"),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            engine.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-empty-ingest-clears-budget",
            )
            placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
            digest = engine._active_replay_placeholder_digest(placeholder)
            assert digest is not None
            engine._remember_generated_ignored_placeholder_hash(digest)
            engine._write_generated_ignored_placeholder_hash_counts({digest: 1})
            engine._compression_boundary_ingest_pending = True
            engine._compression_boundary_active_placeholder_digest_budget = {digest: 1}

            engine._ingest_messages([])

            assert engine._compression_boundary_ingest_pending is False
            assert engine._compression_boundary_active_placeholder_digest_budget == {}
        finally:
            engine.shutdown()

    def test_duplicate_carried_ignored_placeholders_after_rollover_are_not_stored(self, tmp_path):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "lcm_msg_ignore_placeholder_rollover_duplicate.db"),
                fresh_tail_count=2,
                leaf_chunk_tokens=10,
                ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
                sensitive_patterns_enabled=True,
                sensitive_patterns=["api_key"],
            )
        )
        try:
            engine.on_session_start(
                "old-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-placeholder-rollover-duplicate",
            )
            first_result = engine.compress(
                [
                    {"role": "user", "content": "visible backlog before rollover " + "v" * 200},
                    {"role": "user", "content": "api_key=sk-ignore...cdef duplicate ignored"},
                    {"role": "user", "content": "api_key=sk-ignore...cdef duplicate ignored"},
                ],
                current_tokens=10_000, force=True,  # manual seed; synthetic pressure is not a host estimate
            )
            assert sum(
                "LCM active replay placeholder: message ignored" in str(msg.get("content", ""))
                for msg in first_result
            ) == 2

            engine.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-placeholder-rollover-duplicate",
                boundary_reason="compression",
                old_session_id="old-session",
            )
            engine._ingest_messages(first_result)

            assert all(
                "LCM active replay placeholder: message ignored" not in str(row["content"])
                for row in engine._store.get_session_messages("new-session")
            )
        finally:
            engine.shutdown()

    def test_duplicate_carried_ignored_placeholders_after_rollover_restart_are_not_stored(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_placeholder_rollover_duplicate_restart.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=2,
                leaf_chunk_tokens=10,
                ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
                sensitive_patterns_enabled=True,
                sensitive_patterns=["api_key"],
            )
        )
        first.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-placeholder-rollover-duplicate-restart",
        )
        first_result = first.compress(
            [
                {"role": "user", "content": "visible backlog before rollover " + "v" * 200},
                {"role": "user", "content": "api_key=sk-ignore...cdef duplicate ignored"},
                {"role": "user", "content": "api_key=sk-ignore...cdef duplicate ignored"},
            ],
            current_tokens=10_000, force=True,  # manual seed; synthetic pressure is not a host estimate
        )
        first.shutdown()
        assert sum(
            "LCM active replay placeholder: message ignored" in str(msg.get("content", ""))
            for msg in first_result
        ) == 2

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=2,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-placeholder-rollover-duplicate-restart",
                boundary_reason="compression",
                old_session_id="old-session",
            )
            second._ingest_messages(first_result)

            assert all(
                "LCM active replay placeholder: message ignored" not in str(row["content"])
                for row in second._store.get_session_messages("new-session")
            )
        finally:
            second.shutdown()

    def test_duplicate_generated_placeholder_count_survives_separate_ingests_before_rollover_restart(self, tmp_path):
        db_path = tmp_path / "lcm_msg_ignore_placeholder_separate_ingest_count.db"
        first = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
                ignore_message_patterns=[r"api_key=sk-ignore\.\.\.cdef"],
                sensitive_patterns_enabled=True,
                sensitive_patterns=["api_key"],
            )
        )
        first.on_session_start(
            "old-session",
            platform="telegram",
            context_length=1000,
            conversation_id="conv-placeholder-separate-ingest-count",
        )
        first_result = first._ingest_messages(
            [{"role": "user", "content": "api_key=sk-ignore...cdef duplicate ignored"}]
        )
        first_result = first._ingest_messages(
            first_result
            + [{"role": "user", "content": "api_key=sk-ignore...cdef duplicate ignored"}]
        )
        assert sum(
            "LCM active replay placeholder: message ignored" in str(msg.get("content", ""))
            for msg in first_result
        ) == 2
        first.shutdown()

        second = LCMEngine(
            config=LCMConfig(
                database_path=str(db_path),
                fresh_tail_count=10,
                leaf_chunk_tokens=10,
            )
        )
        try:
            second.on_session_start(
                "new-session",
                platform="telegram",
                context_length=1000,
                conversation_id="conv-placeholder-separate-ingest-count",
                boundary_reason="compression",
                old_session_id="old-session",
            )
            second._ingest_messages(first_result)

            assert all(
                "LCM active replay placeholder: message ignored" not in str(row["content"])
                for row in second._store.get_session_messages("new-session")
            )
        finally:
            second.shutdown()
