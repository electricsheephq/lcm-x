"""LCM engine integration tests: contract."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.tools as lcm_tools

from agent.context_engine import ContextEngine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

from .engine_helpers import (
    engine as engine,
    _FakeTimeoutRegexEngine,
)


class TestEngineABC:
    def test_is_context_engine(self, engine):
        assert isinstance(engine, ContextEngine)

    def test_name(self, engine):
        assert engine.name == "lcm-x"

    def test_tool_schemas(self, engine):
        schemas = engine.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "lcm_grep" in names
        assert "lcm_recall" in names
        assert "lcm_describe" in names
        assert "lcm_expand" in names
        assert "lcm_load_session" in names
        assert "lcm_status" in names
        assert "lcm_inspect" in names
        assert "lcm_doctor" in names
        assert "lcm_expand_query" in names

        grep_schema = next(s for s in schemas if s["name"] == "lcm_grep")
        grep_props = grep_schema["parameters"]["properties"]
        assert "session_scope" in grep_props
        assert grep_props["session_scope"]["enum"] == ["current", "all", "session"]
        assert "session_id" in grep_props
        assert "session_scope='session'" in grep_props["session_id"]["description"]
        assert "role" in grep_props
        assert grep_props["role"]["enum"] == [
            "system",
            "user",
            "assistant",
            "tool",
            "unknown",
        ]
        assert "time_from" in grep_props
        assert "time_to" in grep_props
        assert "source" in grep_props
        assert "descendant source lineage" in grep_props["source"]["description"]
        assert "unknown" in grep_props["source"]["description"]
        assert "conversation_id" in grep_props
        assert "Discord" in grep_props["conversation_id"]["description"]
        assert grep_props["exclude_current_session"]["default"] is False
        assert grep_props["exclude_session_ids"]["default"] == []
        recall_schema = next(s for s in schemas if s["name"] == "lcm_recall")
        recall_props = recall_schema["parameters"]["properties"]
        assert recall_props["exclude_current_session"]["default"] is False
        assert recall_props["exclude_session_ids"]["default"] == []
        # The default scope still steers callers to the active session.
        description_lower = grep_schema["description"].lower()
        assert (
            "current-session" in description_lower
            or "current session" in description_lower
            or "active session" in description_lower
        )
        assert "session_search" in grep_schema["description"]
        # The schema now documents the broader scopes — assert by enumerating them in the
        # session_scope description rather than enforcing the legacy current-only wording.
        scope_description = grep_props["session_scope"]["description"]
        assert (
            "all" in scope_description
            and "session" in scope_description
            and "current" in scope_description
        )
        assert "session_search" in scope_description
        # Cross-session search is positioned as plugin-local archive recovery, not memory.
        assert (
            "archive" in grep_schema["description"].lower()
            or "plugin-local" in grep_schema["description"].lower()
        )

        describe_schema = next(s for s in schemas if s["name"] == "lcm_describe")
        expand_schema = next(s for s in schemas if s["name"] == "lcm_expand")
        expand_query_schema = next(
            s for s in schemas if s["name"] == "lcm_expand_query"
        )

        # lcm_describe gained an optional session_id, but its DEFAULT scope is still the
        # current session and it still cannot discover session ids, so both contract
        # strings main pins must survive the feature.
        assert "current session" in describe_schema["description"].lower()
        assert "session_search" in describe_schema["description"]
        describe_props = describe_schema["parameters"]["properties"]
        assert describe_props["session_id"]["type"] == "string"
        # lcm_expand picked up a third mode (store_id); its description must surface that.
        assert "store_id" in expand_schema["description"]
        assert "session_search" in expand_schema["description"]
        expand_props = expand_schema["parameters"]["properties"]
        assert "source_offset" in expand_props
        assert "source_limit" in expand_props
        assert "content_offset" in expand_props
        assert "include_exact_ref" in expand_props
        assert expand_props["include_exact_ref"]["default"] is False
        assert "store_id" in expand_props
        assert expand_props["session_id"]["type"] == "string"
        assert (
            "across sessions" in expand_props["store_id"]["description"].lower()
            or "cross-session" in expand_props["store_id"]["description"].lower()
        )
        assert "pagination" in expand_props["source_offset"]["description"].lower()
        load_schema = next(s for s in schemas if s["name"] == "lcm_load_session")
        load_props = load_schema["parameters"]["properties"]
        assert load_schema["parameters"]["required"] == ["session_id"]
        assert "ordered raw-message transcript" in load_schema["description"]
        assert "after_store_id" in load_props
        assert "max_content_chars" in load_props
        assert "roles" in load_props
        assert "time_from" in load_props
        assert "time_to" in load_props
        assert "include_exact_ref" in load_props
        assert load_props["include_exact_ref"]["default"] is False
        # lcm_expand_query is no longer current-session-only and no longer routes the
        # model to session_search: it performs bounded cross-session retrieval itself
        # over <=20 explicit session_ids. Those two description asserts are therefore
        # forced out by the feature (architect sign-off, 2026-08-20). The default-scope
        # contract they also carried is re-pinned positively below.
        assert "active session" in expand_query_schema["description"].lower()
        assert "session_ids" in expand_query_schema["description"]
        expand_query_props = expand_query_schema["parameters"]["properties"]
        assert "context_max_tokens" in expand_query_props
        assert (
            "fresh context budget"
            in expand_query_props["context_max_tokens"]["description"]
        )
        assert expand_query_props["session_ids"]["minItems"] == 1
        assert expand_query_props["session_ids"]["maxItems"] == 20
        assert expand_query_props["output"]["enum"] == ["answer", "evidence"]
        assert expand_query_props["output"]["default"] == "answer"

    def test_readme_documents_session_scope_contract(self):
        readme = Path(__file__).resolve().parents[1].joinpath("README.md").read_text()
        # cross-session opt-in is now documented as bounded archive recovery
        assert "session_scope='all'" in readme
        assert "session_scope='session'" in readme
        assert "current-session recall" in readme
        assert "session_search" in readme
        # The reframed positioning steers callers away from a memory-system
        # reading and toward bounded archive recovery over rows already in lcm.db.
        assert "archive" in readme.lower() or "externally backfilled" in readme.lower()
        # No implied importer language: anchor the use case on rows already in
        # lcm.db, not on an official OpenClaw/lossless-claw importer.
        assert "imported from OpenClaw" not in readme
        assert "imported from lossless-claw" not in readme
        assert "Lossless raw recovery contract" in readme
        assert "lcm_load_session" in readme
        assert "after_store_id" in readme
        assert "source_offset" in readme
        assert "content_offset" in readme
        assert "LCM_EXPANSION_CONTEXT_TOKENS" in readme

    def test_should_compress(self, engine):
        assert not engine.should_compress(1000)
        assert engine.should_compress(engine.threshold_tokens)

    def test_should_compress_when_explicit_assembly_cap_is_hit(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_should_compress_cap.db"),
            max_assembly_tokens=90,
        )
        instance = LCMEngine(config=config)
        instance.context_length = 200000
        instance.threshold_tokens = int(200000 * config.context_threshold)

        assert instance.should_compress(90)

    def test_preflight_does_not_request_compaction_when_only_fresh_tail_is_over_threshold(self, tmp_path):
        # Yield disabled: this test pins the classic contract that preflight
        # never advertises a pass that would no-op. With the pressure yield
        # enabled the same state legitimately compacts instead (see
        # test_fresh_tail_pressure_yield.py).
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_preflight_fresh_tail.db"),
            fresh_tail_count=4,
            leaf_chunk_tokens=100,
            fresh_tail_pressure_yield_enabled=False,
            threshold_full_sweep_enabled=False,  # #1013: pin the pre-#1013 single-pass compaction
        )
        instance = LCMEngine(config=config)
        instance._session_id = "test-session"
        instance.context_length = 1000
        instance.threshold_tokens = 100
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "tiny old backlog"},
            {"role": "assistant", "content": "tiny old answer"},
            {"role": "user", "content": "fresh " + "x" * 500},
            {"role": "assistant", "content": "fresh " + "y" * 500},
            {"role": "user", "content": "fresh " + "z" * 500},
        ]
        try:
            assert count_messages_tokens(messages) >= instance.threshold_tokens
            assert not instance.should_compress_preflight(messages)
            assert instance.last_compression_status == "noop"
            assert instance.last_compression_was_noop is True
            assert "below leaf chunk threshold" in instance.last_compression_noop_reason
        finally:
            instance.shutdown()

    def test_positive_preflight_clears_prior_noop_status(self, tmp_path):
        # Yield disabled so the first preflight still lands in the classic
        # noop state this test transitions out of (see
        # test_fresh_tail_pressure_yield.py for yield-on coverage).
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_preflight_clears_noop.db"),
            fresh_tail_count=4,
            leaf_chunk_tokens=100,
            fresh_tail_pressure_yield_enabled=False,
            threshold_full_sweep_enabled=False,  # #1013: pin the pre-#1013 single-pass compaction
        )
        instance = LCMEngine(config=config)
        instance._session_id = "test-session"
        instance.context_length = 1000
        instance.threshold_tokens = 100
        noop_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "tiny old backlog"},
            {"role": "assistant", "content": "tiny old answer"},
            {"role": "user", "content": "fresh " + "x" * 500},
            {"role": "assistant", "content": "fresh " + "y" * 500},
            {"role": "user", "content": "fresh " + "z" * 500},
        ]
        # Use repeated word-like segments rather than a single-character run:
        # both cl100k_base and the char fallback must keep this backlog eligible.
        eligible_messages = [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": "old backlog " + "compressible backlog segment " * 100,
            },
            {"role": "assistant", "content": "old answer " + "y" * 600},
            {"role": "user", "content": "fresh " + "a" * 200},
            {"role": "assistant", "content": "fresh " + "b" * 200},
            {"role": "user", "content": "fresh " + "c" * 200},
        ]

        try:
            assert instance.should_compress_preflight(noop_messages) is False
            assert instance.last_compression_was_noop is True
            assert instance.last_compression_noop_reason

            assert instance.should_compress_preflight(eligible_messages) is True
            assert instance.last_compression_status == "pending"
            assert instance.last_compression_was_noop is False
            assert instance.last_compression_noop_reason == ""
        finally:
            instance.shutdown()

    def test_preflight_defers_noncritical_under_threshold_raw_backlog_after_ingest(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_preflight_deferred_noncritical.db"),
            fresh_tail_count=4,
            leaf_chunk_tokens=20,
            deferred_maintenance_enabled=True,
            critical_budget_pressure_ratio=0.90,
        )
        instance = LCMEngine(config=config)
        instance.on_session_start("test-session", platform="cli", context_length=10_000)
        instance.context_length = 10_000
        instance.threshold_tokens = 5_000
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old backlog " + "x" * 200},
            {"role": "assistant", "content": "old answer " + "y" * 200},
            {"role": "user", "content": "fresh " + "a" * 50},
            {"role": "assistant", "content": "fresh " + "b" * 50},
            {"role": "user", "content": "fresh " + "c" * 50},
        ]
        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("summarizer called")),
        )
        try:
            assert count_messages_tokens(messages) < instance.threshold_tokens
            assert instance.should_compress_preflight(messages) is False
            assert instance._store.count_session_load_messages(instance.current_session_id) == len(messages)
            state = instance._lifecycle.get_by_conversation(instance._conversation_id)
            assert state is not None
            assert state.debt_kind == "raw_backlog"
            assert state.debt_size_estimate > 0
        finally:
            instance.shutdown()

    def test_preflight_required_sibling_branches_remain_synchronous(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_preflight_required_siblings.db"),
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            deferred_maintenance_enabled=True,
            critical_budget_pressure_ratio=0.50,
        )
        instance = LCMEngine(config=config)
        instance.on_session_start("required-session", platform="cli", context_length=1_000)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old " + "x" * 500},
            {"role": "assistant", "content": "old " + "y" * 500},
            {"role": "user", "content": "fresh"},
        ]
        try:
            instance.threshold_tokens = 1
            assert instance.should_compress_preflight(messages) is True

            instance.threshold_tokens = 100_000
            instance.context_length = 50
            assert instance.should_compress_preflight(messages) is True

            monkeypatch.setattr(instance, "_ingest_messages", lambda _messages: _messages[:-1])
            instance.context_length = 100_000
            assert instance.should_compress_preflight(messages) is True
        finally:
            instance.shutdown()

    def test_preflight_requests_compaction_for_deferred_maintenance_under_critical_pressure(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_preflight_deferred_critical.db"),
            fresh_tail_count=4,
            leaf_chunk_tokens=100,
            deferred_maintenance_enabled=True,
            critical_budget_pressure_ratio=0.50,
        )
        instance = LCMEngine(config=config)
        instance._bind_lifecycle_state("test-session")
        instance.context_length = 200
        instance.threshold_tokens = 10_000
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "tiny old backlog"},
            {"role": "assistant", "content": "tiny old answer"},
            {"role": "user", "content": "fresh " + "x" * 500},
            {"role": "assistant", "content": "fresh " + "y" * 500},
            {"role": "user", "content": "fresh " + "z" * 500},
        ]
        try:
            rough = count_messages_tokens(messages)
            assert rough < instance.threshold_tokens
            assert instance._critical_budget_pressure_reached(
                observed_tokens=rough,
                messages=messages,
            )
            eligible, reason = instance._leaf_compaction_candidate_status(messages)
            assert not eligible
            assert "below leaf chunk threshold" in reason
            instance._lifecycle.record_debt(
                instance._conversation_id,
                kind="raw_backlog",
                size_estimate=instance._raw_backlog_tokens(messages),
            )
            assert instance._should_run_deferred_maintenance(messages, observed_tokens=rough)
            assert instance.should_compress_preflight(messages)
        finally:
            instance.shutdown()

    def test_preflight_requests_compaction_when_old_backlog_has_leaf_chunk(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_preflight_leaf_chunk.db"),
            fresh_tail_count=4,
            leaf_chunk_tokens=20,
        )
        instance = LCMEngine(config=config)
        instance._session_id = "test-session"
        instance.context_length = 1000
        instance.threshold_tokens = 100
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old backlog " + "x" * 200},
            {"role": "assistant", "content": "old answer " + "y" * 200},
            {"role": "user", "content": "fresh " + "a" * 200},
            {"role": "assistant", "content": "fresh " + "b" * 200},
            {"role": "user", "content": "fresh " + "c" * 200},
        ]
        try:
            assert count_messages_tokens(messages) >= instance.threshold_tokens
            assert instance.should_compress_preflight(messages)
        finally:
            instance.shutdown()

    @staticmethod
    def _oversized_bypass_messages():
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old bypass backlog " + "x" * 400},
            {"role": "assistant", "content": "old bypass answer " + "y" * 400},
            {"role": "user", "content": "fresh bypass tail " + "z" * 400},
        ]

    @pytest.mark.parametrize(
        ("session_id", "config_kwargs"),
        [
            ("ignored:session", {"ignore_session_patterns": ["ignored:*"]}),
            ("stateless:session", {"stateless_session_patterns": ["stateless:*"]}),
        ],
    )
    def test_bypassed_sessions_request_host_compaction_without_lcm_writes(
        self,
        tmp_path,
        session_id,
        config_kwargs,
    ):
        config = LCMConfig(
            database_path=str(tmp_path / f"{session_id.replace(':', '-')}.db"),
            **config_kwargs,
        )
        instance = LCMEngine(config=config)
        messages = self._oversized_bypass_messages()
        try:
            instance.on_session_start(session_id, platform="cli", context_length=1_000)
            instance.threshold_tokens = 20

            assert count_messages_tokens(messages) >= instance.threshold_tokens
            instance.ingest(messages)
            instance.on_session_end(session_id, messages)

            assert instance._store.get_session_count(session_id) == 0
            assert instance._dag.get_session_node_count(session_id) == 0
            assert instance.should_compress_preflight(messages)
            assert instance.should_compress(instance.threshold_tokens + 1)
        finally:
            instance.shutdown()

    @pytest.mark.parametrize(
        ("session_id", "config_kwargs"),
        [
            ("ignored:session", {"ignore_session_patterns": ["ignored:*"]}),
            ("stateless:session", {"stateless_session_patterns": ["stateless:*"]}),
        ],
    )
    def test_bypassed_compression_boundary_child_stays_out_of_lcm_storage(
        self,
        tmp_path,
        session_id,
        config_kwargs,
    ):
        child_session_id = "compressed-child"
        grandchild_session_id = "compressed-grandchild"
        config = LCMConfig(
            database_path=str(tmp_path / f"boundary-{session_id.replace(':', '-')}.db"),
            **config_kwargs,
        )
        instance = LCMEngine(config=config)
        messages = self._oversized_bypass_messages()
        child_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "child turn " + "x" * 400},
        ]
        try:
            instance.on_session_start(session_id, platform="cli", context_length=1_000)
            instance.threshold_tokens = 20
            assert instance.should_compress_preflight(messages)

            instance.on_session_start(
                child_session_id,
                boundary_reason="compression",
                old_session_id=session_id,
                platform="cli",
                context_length=1_000,
            )
            instance.ingest(child_messages)
            instance.on_session_end(child_session_id, child_messages)
            instance.threshold_tokens = 20

            assert instance._store.get_session_count(session_id) == 0
            assert instance._store.get_session_count(child_session_id) == 0
            assert instance._dag.get_session_node_count(session_id) == 0
            assert instance._dag.get_session_node_count(child_session_id) == 0
            assert instance.should_compress_preflight(child_messages)

            instance.on_session_start(
                grandchild_session_id,
                boundary_reason="compression",
                old_session_id=child_session_id,
                platform="cli",
                context_length=1_000,
            )
            instance.ingest(child_messages)
            instance.threshold_tokens = 20

            assert instance._store.get_session_count(grandchild_session_id) == 0
            assert instance._dag.get_session_node_count(grandchild_session_id) == 0
            assert instance.should_compress_preflight(child_messages)
        finally:
            instance.shutdown()

    def test_bypassed_compression_boundary_child_without_platform_keeps_lineage(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "boundary-missing-platform.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)
        child_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "child without platform must stay bypassed"},
        ]
        try:
            instance.on_session_start("ignored-source", platform="cron", context_length=1_000)
            assert instance._bypasses_lcm_context_management()

            instance.on_session_start(
                "compressed-child",
                boundary_reason="compression",
                old_session_id="ignored-source",
                context_length=1_000,
            )

            assert instance._session_id == "compressed-child"
            assert instance._session_platform == ""
            assert not instance._session_ignored
            assert instance._session_stateless
            assert instance._bypasses_lcm_context_management()
            assert instance._has_lcm_bypass_lineage_session("compressed-child", platform="")

            instance.ingest(child_messages)
            instance.on_session_end("compressed-child", child_messages)

            assert instance._store.get_session_count("ignored-source") == 0
            assert instance._store.get_session_count("compressed-child") == 0
            assert instance._dag.get_session_node_count("ignored-source") == 0
            assert instance._dag.get_session_node_count("compressed-child") == 0
        finally:
            instance.shutdown()

    @pytest.mark.parametrize(
        ("session_id", "config_kwargs"),
        [
            ("ignored:session", {"ignore_session_patterns": ["ignored:*"]}),
            ("stateless:session", {"stateless_session_patterns": ["stateless:*"]}),
        ],
    )
    def test_bypassed_compression_boundary_after_rebind_stays_out_of_lcm_storage(
        self,
        tmp_path,
        session_id,
        config_kwargs,
    ):
        child_session_id = "compressed-after-rebind"
        config = LCMConfig(
            database_path=str(tmp_path / f"boundary-rebind-{session_id.replace(':', '-')}.db"),
            **config_kwargs,
        )
        instance = LCMEngine(config=config)
        child_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "child after rebind " + "x" * 400},
        ]
        try:
            instance.on_session_start(session_id, platform="cli", context_length=1_000)
            instance.on_session_start("foreground:normal", platform="cli", context_length=1_000)

            instance.on_session_start(
                child_session_id,
                boundary_reason="compression",
                old_session_id=session_id,
                platform="cli",
                context_length=1_000,
            )
            instance.ingest(child_messages)
            instance.on_session_end(child_session_id, child_messages)

            assert instance._store.get_session_count(session_id) == 0
            assert instance._store.get_session_count(child_session_id) == 0
            assert instance._dag.get_session_node_count(session_id) == 0
            assert instance._dag.get_session_node_count(child_session_id) == 0
        finally:
            instance.shutdown()

    def test_platform_drift_does_not_turn_normal_boundary_child_stateless(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "normal-boundary-platform-drift.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)
        child_messages = [{"role": "user", "content": "normal child should persist"}]
        try:
            instance.on_session_start("normal:source", platform="cli", context_length=1_000)
            instance.ingest([{"role": "user", "content": "normal source persists"}])
            instance.on_session_start("cron:tick", platform="cron", context_length=1_000)

            instance.on_session_start(
                "normal:child",
                boundary_reason="compression",
                old_session_id="normal:source",
                platform="cron",
                context_length=1_000,
            )
            instance.ingest(child_messages)

            assert instance._store.get_session_count("normal:child") == 1
            assert not instance._has_lcm_bypass_lineage_session("normal:child")
        finally:
            instance.shutdown()

    def test_bypass_lineage_does_not_poison_reused_normal_session_id(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "reused-normal-session-id.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored must not persist"}])
            instance.on_session_end("reused-id", [{"role": "user", "content": "ignored end"}])

            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([{"role": "user", "content": "normal should persist"}])

            assert instance._store.get_session_count("reused-id") == 1
            assert instance._has_lcm_bypass_lineage_session("reused-id")
            assert not instance._has_lcm_bypass_lineage_session("reused-id", platform="cli")
        finally:
            instance.shutdown()

    def test_bypass_lineage_does_not_poison_reused_normal_session_id_without_end(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "reused-normal-session-id-no-end.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored must not persist"}])

            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([{"role": "user", "content": "normal should persist"}])

            assert instance._store.get_session_count("reused-id") == 1
            assert not instance._has_lcm_bypass_lineage_session("reused-id", platform="cli")
        finally:
            instance.shutdown()

    def test_same_id_reused_unmatched_suffix_only_end_flushes_current_normal(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "reused-unmatched-suffix-normal-end.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored prefix must not persist"}])
            assert instance._store.get_session_count("reused-id") == 0
            assert instance._dag.get_session_node_count("reused-id") == 0
            assert instance._has_lcm_bypass_lineage_session("reused-id")

            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            foreground_session_id = instance.current_session_id
            foreground_platform = instance.current_session_platform
            foreground_conversation_id = instance.current_conversation_id
            assert foreground_session_id == "reused-id"
            assert foreground_platform == "cli"
            assert not instance.current_session_ignored
            assert not instance.current_session_stateless
            assert not instance._has_lcm_bypass_lineage_session("reused-id", platform="cli")

            instance.on_session_end(
                "reused-id",
                [{"role": "assistant", "content": "unmatched suffix-only normal final"}],
            )

            rows = instance._store.get_range("reused-id")
            assert [(row["role"], row["content"]) for row in rows] == [
                ("assistant", "unmatched suffix-only normal final"),
            ]
            assert instance._dag.get_session_node_count("reused-id") == 0
            assert instance.current_session_id == foreground_session_id
            assert instance.current_session_platform == foreground_platform
            assert instance.current_conversation_id == foreground_conversation_id
            state = instance._lifecycle.get_by_conversation(foreground_conversation_id)
            assert state is not None
            assert state.current_session_id is None
            assert state.last_finalized_session_id == "reused-id"
            assert not instance.current_session_ignored
            assert not instance.current_session_stateless
        finally:
            instance.shutdown()

    def test_bypass_lineage_does_not_poison_reused_normal_compression_boundary(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "reused-normal-boundary.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored must not persist"}])

            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([{"role": "user", "content": "normal should persist"}])
            instance.on_session_start(
                "normal-child",
                boundary_reason="compression",
                old_session_id="reused-id",
                platform="cli",
                context_length=1_000,
            )

            assert not instance._session_stateless
            assert not instance._bypasses_lcm_context_management()
            assert not instance._has_lcm_bypass_lineage_session("normal-child", platform="cli")
        finally:
            instance.shutdown()

    def test_later_bypass_rebind_keeps_boundary_child_out_of_lcm(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "prior-normal-boundary-after-bypass-rebind.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([{"role": "user", "content": "normal should persist"}])
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored must not persist"}])
            instance.on_session_start("foreground", platform="cli", context_length=1_000)

            instance.on_session_start(
                "normal-looking-child",
                boundary_reason="compression",
                old_session_id="reused-id",
                platform="cli",
                context_length=1_000,
            )
            instance.ingest([{"role": "user", "content": "boundary child must stay out of LCM"}])

            assert instance._session_stateless
            assert instance._bypasses_lcm_context_management()
            assert instance._has_lcm_bypass_lineage_session("normal-looking-child", platform="cli")
            assert instance._store.get_session_count("reused-id") == 1
            assert instance._store.get_session_count("normal-looking-child") == 0
        finally:
            instance.shutdown()

    def test_platform_only_bypass_boundary_after_source_end_stays_out_of_lcm(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "platform-only-boundary-after-end.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("old-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored source"}])
            instance.on_session_end("old-id", [{"role": "user", "content": "ignored source end"}])

            instance.on_session_start("foreground", platform="cli", context_length=1_000)
            instance.on_session_start(
                "child-id",
                boundary_reason="compression",
                old_session_id="old-id",
                platform="cli",
                context_length=1_000,
            )
            instance.ingest([{"role": "user", "content": "child must stay out of LCM"}])

            assert instance._store.get_session_count("old-id") == 0
            assert instance._store.get_session_count("child-id") == 0
            assert instance._has_lcm_bypass_lineage_session("child-id", platform="cli")
        finally:
            instance.shutdown()

    def test_ended_bypass_boundary_child_can_reuse_id_as_normal_session(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "bypass-child-reused-normal.db"),
            ignore_session_patterns=["ignored:*"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("ignored:old", platform="cli", context_length=1_000)
            instance.on_session_start(
                "child-id",
                boundary_reason="compression",
                old_session_id="ignored:old",
                platform="cli",
                context_length=1_000,
            )
            instance.on_session_end("child-id", [{"role": "user", "content": "bypass child end"}])

            instance.on_session_start("child-id", platform="cli", context_length=1_000)
            instance.ingest([{"role": "user", "content": "normal child-id reuse"}])

            assert not instance._session_stateless
            assert not instance._bypasses_lcm_context_management()
            assert instance._store.get_session_count("child-id") == 1
        finally:
            instance.shutdown()

    def test_thread_context_stateless_session_end_does_not_flush_raw_messages(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "thread-stateless-session-end.db"))
        instance = LCMEngine(config=config)
        messages = [{"role": "user", "content": "must not persist while thread stateless"}]
        try:
            instance.on_session_start("foreground:session", platform="cli", context_length=1_000)
            instance._mark_thread_context_stateless("auxiliary:session")

            instance.on_session_end("foreground:session", messages)

            assert instance._store.get_session_count("foreground:session") == 0
            assert instance._store.get_session_count("auxiliary:session") == 0
        finally:
            instance.shutdown()

    def test_thread_context_stateless_requests_host_compaction_without_lcm_writes(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "thread-stateless.db"))
        instance = LCMEngine(config=config)
        messages = self._oversized_bypass_messages()
        try:
            instance.on_session_start("foreground:session", platform="cli", context_length=1_000)
            instance.threshold_tokens = 20
            instance._mark_thread_context_stateless("auxiliary:session")

            assert count_messages_tokens(messages) >= instance.threshold_tokens
            instance.ingest(messages)

            assert instance._store.get_session_count("foreground:session") == 0
            assert instance._dag.get_session_node_count("foreground:session") == 0
            assert instance.should_compress_preflight(messages)
            assert instance.should_compress(instance.threshold_tokens + 1)
        finally:
            instance.shutdown()

    def test_thread_context_stateless_no_arg_should_compress_uses_auxiliary_usage(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "thread-stateless-no-arg-usage.db"))
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("foreground:session", platform="cli", context_length=1_000)
            instance.threshold_tokens = 20
            instance._mark_thread_context_stateless("auxiliary:session")

            instance.update_from_response({"prompt_tokens": 25, "completion_tokens": 1, "total_tokens": 26})

            assert instance.last_prompt_tokens == 0
            assert instance.should_compress()
            assert instance._store.get_session_count("foreground:session") == 0
            assert instance._store.get_session_count("auxiliary:session") == 0

            instance._clear_thread_context_stateless("auxiliary:session")
            assert not instance.should_compress()
        finally:
            instance.shutdown()

    def test_thread_context_stateless_compress_uses_auxiliary_usage_when_no_arg(
        self,
        tmp_path,
        monkeypatch,
    ):
        calls = []

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0

            def on_session_start(self, bound_session_id, **kwargs):
                pass

            def update_model(self, **kwargs):
                pass

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                calls.append((list(messages), current_tokens, focus_topic, force))
                self.compression_count += 1
                return [{"role": "system", "content": "native compacted"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(database_path=str(tmp_path / "thread-stateless-no-arg-compress.db"))
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [{"role": "user", "content": "short"}]
        try:
            instance.on_session_start("foreground:session", platform="cli", context_length=1_000)
            instance.threshold_tokens = 20
            instance._mark_thread_context_stateless("auxiliary:session")
            instance.update_from_response({"prompt_tokens": 25, "completion_tokens": 1, "total_tokens": 26})

            assert count_messages_tokens(messages) < instance.threshold_tokens
            assert instance.should_compress()
            result = instance.compress(messages)

            assert result == [{"role": "system", "content": "native compacted"}]
            assert calls == [(messages, 25, None, False)]
            assert instance._store.get_session_count("foreground:session") == 0
            assert instance._store.get_session_count("auxiliary:session") == 0
        finally:
            instance.shutdown()

    def test_thread_context_stateless_compress_without_auxiliary_usage_keeps_tokens_unknown(
        self,
        tmp_path,
        monkeypatch,
    ):
        calls = []

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0

            def on_session_start(self, bound_session_id, **kwargs):
                pass

            def update_model(self, **kwargs):
                pass

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                calls.append((list(messages), current_tokens, focus_topic, force))
                self.compression_count += 1
                return [{"role": "system", "content": "native compacted"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(database_path=str(tmp_path / "thread-stateless-no-usage-compress.db"))
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [{"role": "user", "content": "first auxiliary payload " + ("x " * 200)}]
        try:
            instance.on_session_start("foreground:session", platform="cli", context_length=1_000)
            instance.threshold_tokens = 20
            instance._mark_thread_context_stateless("auxiliary:session")

            assert not instance.should_compress()
            result = instance.compress(messages)

            assert result == [{"role": "system", "content": "native compacted"}]
            assert calls == [(messages, None, None, False)]
            assert instance._store.get_session_count("foreground:session") == 0
            assert instance._store.get_session_count("auxiliary:session") == 0
        finally:
            instance.shutdown()

    def test_thread_context_stateless_session_end_clears_auxiliary_usage(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "thread-stateless-end-clears-usage.db"))
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("foreground:session", platform="cli", context_length=1_000)
            instance.threshold_tokens = 20
            instance._mark_thread_context_stateless("auxiliary:session")
            instance.update_from_response({"prompt_tokens": 25, "completion_tokens": 1, "total_tokens": 26})
            assert instance.should_compress()

            instance.on_session_end("auxiliary:session", [{"role": "user", "content": "done"}])
            instance._mark_thread_context_stateless("auxiliary:session")

            assert not instance.should_compress()
            assert instance._store.get_session_count("foreground:session") == 0
            assert instance._store.get_session_count("auxiliary:session") == 0
        finally:
            instance.shutdown()

    def test_thread_context_stateless_generationless_reregister_clears_usage(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "thread-stateless-reregister-clears-usage.db"))
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("foreground:session", platform="cli", context_length=1_000)
            instance.threshold_tokens = 20
            instance._mark_thread_context_stateless("auxiliary:session")
            instance.update_from_response({"prompt_tokens": 25, "completion_tokens": 1, "total_tokens": 26})
            assert instance.should_compress()

            instance._mark_thread_context_stateless("auxiliary:session")

            assert instance._auxiliary_last_prompt_tokens == {}
            assert not instance.should_compress()
            assert instance._store.get_session_count("foreground:session") == 0
            assert instance._store.get_session_count("auxiliary:session") == 0
        finally:
            instance.shutdown()

    def test_thread_context_stateless_adopting_generation_clears_generationless_usage(
        self,
        tmp_path,
        monkeypatch,
    ):
        config = LCMConfig(database_path=str(tmp_path / "thread-stateless-adopt-generation-clears.db"))
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("foreground:session", platform="cli", context_length=1_000)
            instance.threshold_tokens = 20
            instance._mark_thread_context_stateless("auxiliary:session")
            instance.update_from_response({"prompt_tokens": 25, "completion_tokens": 1, "total_tokens": 26})
            assert instance.should_compress()
            assert instance._auxiliary_session_generations == {}

            monkeypatch.setattr(
                instance,
                "_in_process_auxiliary_caller_generation",
                lambda session_id, **kwargs: 123 if session_id == "auxiliary:session" else 0,
            )
            instance._mark_thread_context_stateless("auxiliary:session")

            assert instance._auxiliary_session_generations == {"auxiliary:session": 123}
            assert instance._auxiliary_last_prompt_tokens == {}
            assert not instance.should_compress()
            assert instance._store.get_session_count("foreground:session") == 0
            assert instance._store.get_session_count("auxiliary:session") == 0
        finally:
            instance.shutdown()

    @pytest.mark.parametrize(
        ("session_id", "config_kwargs"),
        [
            ("ignored:session", {"ignore_session_patterns": ["ignored:*"]}),
            ("stateless:session", {"stateless_session_patterns": ["stateless:*"]}),
        ],
    )
    def test_bypassed_sessions_delegate_compress_to_host_native_compressor(
        self,
        tmp_path,
        monkeypatch,
        session_id,
        config_kwargs,
    ):
        calls = []
        compacted = [
            {"role": "system", "content": "native summary"},
            {"role": "user", "content": "fresh tail"},
        ]

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.compression_count = 0
                self._last_compress_aborted = False
                self._last_summary_error = None
                self._last_summary_auth_failure = False
                self._last_summary_network_failure = False
                self._last_summary_dropped_count = 0
                self._last_summary_fallback_used = False
                self._last_aux_model_failure_error = None
                self._last_aux_model_failure_model = None

            def on_session_start(self, bound_session_id, **kwargs):
                calls.append(("start", bound_session_id, kwargs))

            def update_model(self, **kwargs):
                calls.append(("update_model", kwargs))

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                calls.append(("compress", list(messages), current_tokens, focus_topic, force))
                self.compression_count += 1
                return list(compacted)

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / f"delegate-{session_id.replace(':', '-')}.db"),
            **config_kwargs,
        )
        instance = LCMEngine(config=config)
        messages = self._oversized_bypass_messages()
        try:
            instance.on_session_start(
                session_id,
                platform="cli",
                model="test-model",
                provider="test-provider",
                base_url="https://example.invalid/v1",
                api_key="test-key",
                api_mode="chat_completions",
                context_length=1_000,
            )
            instance.threshold_tokens = 20

            result = instance.compress(messages, current_tokens=123, focus_topic="focus")

            assert result == compacted
            assert ("compress", messages, 123, "focus", False) in calls
            assert instance.compression_count == 1
            assert instance._store.get_session_count(session_id) == 0
            assert instance._dag.get_session_node_count(session_id) == 0
        finally:
            instance.shutdown()

    def test_bypassed_sessions_honor_assembly_cap_below_threshold(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "ignored-assembly-cap.db"),
            ignore_session_patterns=["ignored:*"],
            max_assembly_tokens=90,
        )
        instance = LCMEngine(config=config)
        messages = self._oversized_bypass_messages()
        try:
            instance.on_session_start("ignored:cap", platform="cli", context_length=10_000)
            instance.threshold_tokens = 1_000

            assert count_messages_tokens(messages) < instance.threshold_tokens
            assert count_messages_tokens(messages) >= config.max_assembly_tokens
            assert instance.should_compress_preflight(messages)
            assert instance.should_compress(config.max_assembly_tokens)
        finally:
            instance.shutdown()

    def test_thread_context_fallback_uses_effective_context_length_and_auxiliary_session(
        self,
        tmp_path,
        monkeypatch,
    ):
        calls = []
        init_kwargs = []

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                init_kwargs.append(kwargs)
                self.compression_count = 0

            def on_session_start(self, bound_session_id, **kwargs):
                calls.append(("start", bound_session_id, kwargs))

            def update_model(self, **kwargs):
                calls.append(("update_model", kwargs))

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                self.compression_count += 1
                return [{"role": "system", "content": "native compacted"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(database_path=str(tmp_path / "thread-effective-context.db"))
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start(
                "foreground:session",
                platform="cli",
                model="gpt-5.5",
                provider="openai-codex",
                context_length=500_000,
            )
            instance.threshold_tokens = 20
            instance._mark_thread_context_stateless("auxiliary:session")

            result = instance.compress(self._oversized_bypass_messages(), current_tokens=123)

            assert result == [{"role": "system", "content": "native compacted"}]
            assert init_kwargs[0]["config_context_length"] == 272_000
            assert any(call[0] == "start" and call[1] == "auxiliary:session" for call in calls)
            assert any(
                call[0] == "update_model" and call[1].get("context_length") == 272_000
                for call in calls
            )
        finally:
            instance.shutdown()

    def test_bypassed_native_update_model_failure_still_compacts(self, tmp_path, monkeypatch):
        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0

            def update_model(self, **kwargs):
                raise RuntimeError("metadata sync failed")

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                self.compression_count += 1
                return [{"role": "system", "content": "native compacted despite sync failure"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-update-model-failure.db"),
            ignore_session_patterns=["ignored:*"],
        )
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        try:
            instance.on_session_start("ignored:sync-failure", platform="cli", context_length=2_000)
            instance.threshold_tokens = 500

            result = instance.compress(
                [{"role": "user", "content": "oversized " + "x" * 1_000}],
                current_tokens=600,
            )

            assert result == [
                {"role": "system", "content": "native compacted despite sync failure"}
            ]
            assert instance._store.get_session_count("ignored:sync-failure") == 0
            assert instance._dag.get_session_node_count("ignored:sync-failure") == 0
        finally:
            instance.shutdown()

    def test_unavailable_native_fallback_sanitizes_tool_pairs(self, tmp_path, monkeypatch):
        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "fallback-tool-pairs.db"),
            ignore_session_patterns=["ignored:*"],
            fresh_tail_count=2,
        )
        instance = LCMEngine(config=config)
        messages = [
            {"role": "system", "content": "system"},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [{"id": "call_head", "type": "function"}],
            },
            {"role": "user", "content": "large middle " + "x" * 400},
            {"role": "tool", "tool_call_id": "orphan_tail", "content": "orphan tool result"},
            {"role": "user", "content": "fresh tail"},
        ]
        try:
            instance.on_session_start("ignored:tools", platform="cli", context_length=1_000)

            result = instance.compress(messages, force=True)

            assert any(msg.get("tool_call_id") == "call_head" for msg in result)
            assert not any(msg.get("tool_call_id") == "orphan_tail" for msg in result)
        finally:
            instance.shutdown()

    def test_bypassed_native_fallback_receives_redacted_messages(self, tmp_path, monkeypatch):
        captured_messages = []

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                captured_messages.extend(messages)
                self.compression_count += 1
                return list(messages)

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-redacted-native.db"),
            ignore_session_patterns=["ignored:*"],
        )
        config.sensitive_patterns_enabled = True
        config.sensitive_patterns = ["password_assignment"]
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "password: supersecret1234 " + "x" * 400},
            {"role": "assistant", "content": "ack"},
        ]
        try:
            instance.on_session_start("ignored:redact", platform="cli", context_length=1_000)
            instance.threshold_tokens = 20

            instance.compress(messages, current_tokens=200)

            assert captured_messages
            serialized = json.dumps(captured_messages)
            assert "supersecret1234" not in serialized
        finally:
            instance.shutdown()

    def test_bypassed_native_result_is_sanitized_under_cap(self, tmp_path, monkeypatch):
        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                self.compression_count += 1
                return [
                    {
                        "role": "assistant",
                        "content": "calling tool",
                        "tool_calls": [{"id": "call_1", "type": "function"}],
                    },
                    {"role": "user", "content": "interrupt before tool result"},
                    {"role": "tool", "tool_call_id": "orphan", "content": "late result"},
                ]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-native-sanitize-under-cap.db"),
            ignore_session_patterns=["ignored:*"],
        )
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [{"role": "user", "content": "oversized " + "x" * 1_000}]
        try:
            instance.on_session_start("ignored:native-invalid", platform="cli", context_length=2_000)
            instance.threshold_tokens = 500

            result = instance.compress(messages, current_tokens=600)

            assert not any(msg.get("tool_call_id") == "orphan" for msg in result)
            assert any(msg.get("tool_call_id") == "call_1" for msg in result)
        finally:
            instance.shutdown()

    def test_bypassed_native_fallback_resets_between_sessions(self, tmp_path, monkeypatch):
        instances = []

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.ended = False
                self.reset = False
                instances.append(self)

            def on_session_end(self, session_id, messages):
                self.ended = session_id

            def on_session_reset(self):
                self.reset = True

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                self.compression_count += 1
                return [{"role": "user", "content": "native compacted"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-native-reset.db"),
            ignore_session_patterns=["ignored:*"],
        )
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [{"role": "user", "content": "oversized " + "x" * 1_000}]
        try:
            instance.on_session_start("ignored:first", platform="cli", context_length=2_000)
            instance.threshold_tokens = 500
            instance.compress(messages, current_tokens=600)

            instance.on_session_start("ignored:second", platform="cli", context_length=2_000)
            instance.threshold_tokens = 500
            instance.compress(messages, current_tokens=600)

            assert len(instances) == 2
            assert instances[0].ended == "ignored:first"
            assert instances[0].reset is True
        finally:
            instance.shutdown()

    def test_bypassed_native_fallback_resets_on_session_end_even_for_same_session_id(self, tmp_path, monkeypatch):
        instances = []

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.ended = False
                self.reset = False
                instances.append(self)

            def on_session_end(self, session_id, messages):
                self.ended = session_id

            def on_session_reset(self):
                self.reset = True

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                self.compression_count += 1
                return [{"role": "user", "content": "native compacted"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-native-reset-same-id.db"),
            ignore_session_patterns=["ignored:*"],
        )
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [{"role": "user", "content": "oversized " + "x" * 1_000}]
        try:
            instance.on_session_start("ignored:reused", platform="cli", context_length=2_000)
            instance.threshold_tokens = 500
            instance.compress(messages, current_tokens=600)
            instance.on_session_end("ignored:reused", messages)

            instance.on_session_start("ignored:reused", platform="cli", context_length=2_000)
            instance.threshold_tokens = 500
            instance.compress(messages, current_tokens=600)

            assert len(instances) == 2
            assert instances[0].ended == "ignored:reused"
            assert instances[0].reset is True
        finally:
            instance.shutdown()

    def test_auxiliary_native_fallback_resets_on_session_end(self, tmp_path):
        class FakeContextCompressor:
            def __init__(self):
                self.ended = []
                self.reset = False

            def on_session_end(self, session_id, messages):
                self.ended.append((session_id, messages))

            def on_session_reset(self):
                self.reset = True

        config = LCMConfig(database_path=str(tmp_path / "auxiliary-native-reset-on-end.db"))
        instance = LCMEngine(config=config)
        messages = [{"role": "user", "content": "auxiliary final messages"}]
        compressor = FakeContextCompressor()
        try:
            instance.on_session_start("foreground-session", platform="cli", context_length=2_000)
            instance._mark_thread_context_stateless("auxiliary-session")
            instance._host_fallback_compressor = compressor
            instance._host_fallback_session_id = "auxiliary-session"

            instance.on_session_end("auxiliary-session", messages)

            assert compressor.ended == [("auxiliary-session", messages)]
            assert compressor.reset is True
            assert instance._host_fallback_compressor is None
            assert instance._host_fallback_session_id == ""
            assert not instance._thread_context_has_auxiliary_session("auxiliary-session")
        finally:
            instance.shutdown()

    def test_bypassed_native_fallback_resets_when_same_session_id_rebinds_platform(self, tmp_path, monkeypatch):
        instances = []

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.ended = False
                self.reset = False
                instances.append(self)

            def on_session_end(self, session_id, messages):
                self.ended = session_id

            def on_session_reset(self):
                self.reset = True

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                self.compression_count += 1
                return [{"role": "user", "content": "native compacted"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-native-reset-rebind-platform.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [{"role": "user", "content": "oversized " + "x" * 1_000}]
        try:
            instance.on_session_start("reused-id", platform="cron", context_length=2_000)
            instance.threshold_tokens = 500
            instance.compress(messages, current_tokens=600)

            instance.on_session_start("reused-id", platform="cli", context_length=2_000)
            instance.ingest([{"role": "user", "content": "normal"}])
            instance.on_session_start("reused-id", platform="cron", context_length=2_000)
            instance.threshold_tokens = 500
            instance.compress(messages, current_tokens=600)

            assert len(instances) == 2
            assert instances[0].ended == "reused-id"
            assert instances[0].reset is True
        finally:
            instance.shutdown()

    def test_bypassed_native_fallback_resets_on_session_reset(self, tmp_path, monkeypatch):
        instances = []

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.reset = False
                instances.append(self)

            def on_session_reset(self):
                self.reset = True

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                self.compression_count += 1
                return [{"role": "user", "content": "native compacted"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-native-session-reset.db"),
            ignore_session_patterns=["ignored:*"],
        )
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [{"role": "user", "content": "oversized " + "x" * 1_000}]
        try:
            instance.on_session_start("ignored:reset", platform="cli", context_length=2_000)
            instance.threshold_tokens = 500
            instance.compress(messages, current_tokens=600)
            instance.on_session_reset()
            instance.on_session_start("ignored:reset", platform="cli", context_length=2_000)
            instance.threshold_tokens = 500
            instance.compress(messages, current_tokens=600)

            assert len(instances) == 2
            assert instances[0].reset is True
        finally:
            instance.shutdown()

    def test_late_bypassed_session_end_does_not_reset_active_fallback_compressor(self, tmp_path, monkeypatch):
        instances = []

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.ended = []
                self.reset = False
                instances.append(self)

            def on_session_end(self, session_id, messages):
                self.ended.append(session_id)

            def on_session_reset(self):
                self.reset = True

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                self.compression_count += 1
                return [{"role": "user", "content": "native compacted"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-late-end-active-fallback.db"),
            ignore_session_patterns=["ignored:*"],
        )
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [{"role": "user", "content": "oversized " + "x" * 1_000}]
        try:
            instance.on_session_start("ignored:old", platform="cli", context_length=2_000)
            instance.on_session_end("ignored:old", [])
            instance.on_session_start("ignored:active", platform="cli", context_length=2_000)
            instance.threshold_tokens = 500
            instance.compress(messages, current_tokens=600)

            instance.on_session_end("ignored:old", [])

            assert len(instances) == 1
            assert instances[0].ended == []
            assert instances[0].reset is False
            assert instance._host_fallback_compressor is instances[0]
        finally:
            instance.shutdown()

    def test_late_reused_lineage_suffix_only_session_end_stays_stateless(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "late-reused-normal-end.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored"}])
            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([{"role": "user", "content": "normal"}])
            instance.on_session_start("foreground", platform="cli", context_length=1_000)

            instance.on_session_end("reused-id", [{"role": "assistant", "content": "ambiguous late final"}])

            assert instance._store.get_session_count("reused-id") == 1
            assert instance._store.get_session_count("foreground") == 0
        finally:
            instance.shutdown()

    def test_late_reused_bypass_session_end_does_not_write_to_foreground(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "late-reused-bypass-end.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([{"role": "user", "content": "normal"}])
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored"}])
            instance.on_session_start("foreground", platform="cli", context_length=1_000)

            instance.on_session_end("reused-id", [{"role": "assistant", "content": "late ignored final"}])

            assert instance._store.get_session_count("reused-id") == 1
            assert instance._store.get_session_count("foreground") == 0
        finally:
            instance.shutdown()

    def test_late_same_id_bypass_session_end_does_not_append_to_reused_normal_session(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "late-same-id-bypass-end.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored start"}])
            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([{"role": "user", "content": "normal start"}])

            instance.on_session_end(
                "reused-id",
                [
                    {"role": "user", "content": "ignored start"},
                    {"role": "assistant", "content": "late bypass final"},
                ],
            )

            rows = instance._store.get_range("reused-id")
            assert [(row["role"], row["content"]) for row in rows] == [("user", "normal start")]
        finally:
            instance.shutdown()

    def test_late_same_id_normal_session_end_flushes_when_prefix_matches_current_store(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "late-same-id-normal-end.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored start"}])
            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([{"role": "user", "content": "normal start"}])

            instance.on_session_end(
                "reused-id",
                [
                    {"role": "user", "content": "normal start"},
                    {"role": "assistant", "content": "normal final"},
                ],
            )

            rows = instance._store.get_range("reused-id")
            assert [(row["role"], row["content"]) for row in rows] == [
                ("user", "normal start"),
                ("assistant", "normal final"),
            ]
        finally:
            instance.shutdown()

    def test_same_id_current_normal_first_flush_after_bypass_lineage_persists(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "same-id-normal-first-flush-after-bypass.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored start"}])
            instance.on_session_end("reused-id", [{"role": "user", "content": "ignored end"}])

            instance.on_session_start(
                "reused-id",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=1_000,
            )

            instance.on_session_end(
                "reused-id",
                [
                    {"role": "user", "content": "normal start"},
                    {"role": "assistant", "content": "normal final"},
                ],
            )

            rows = instance._store.get_range("reused-id")
            assert [(row["role"], row["content"], row["conversation_id"]) for row in rows] == [
                ("user", "normal start", "normal-conversation"),
                ("assistant", "normal final", "normal-conversation"),
            ]
            state = instance._lifecycle.get_by_conversation("normal-conversation")
            assert state is not None
            assert state.current_session_id is None
            assert state.last_finalized_session_id == "reused-id"
        finally:
            instance.shutdown()

    def test_off_current_normal_first_flush_after_bypass_lineage_persists_without_bypass_prefix(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "off-current-normal-first-flush-after-bypass.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored start"}])
            instance.on_session_end("reused-id", [{"role": "user", "content": "ignored end"}])

            instance.on_session_start(
                "reused-id",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=1_000,
            )
            instance.on_session_start(
                "foreground",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=1_000,
            )

            instance.on_session_end(
                "reused-id",
                [
                    {"role": "user", "content": "normal start"},
                    {"role": "assistant", "content": "normal final"},
                ],
            )

            rows = instance._store.get_range("reused-id")
            assert [(row["role"], row["content"], row["conversation_id"]) for row in rows] == [
                ("user", "normal start", "normal-conversation"),
                ("assistant", "normal final", "normal-conversation"),
            ]
            assert instance._store.get_session_count("foreground") == 0
            state = instance._lifecycle.get_by_conversation("normal-conversation")
            assert state is not None
            assert state.current_session_id is None
            assert state.last_finalized_session_id == "reused-id"
        finally:
            instance.shutdown()

    def test_late_same_id_bypass_prefix_end_skips_when_current_normal_store_empty(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "late-same-id-bypass-prefix-empty-store.db"),
            ignore_session_patterns=["cron"],
        )
        instance = LCMEngine(config=config)
        try:
            opener = {"role": "user", "content": "shared opener"}

            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([opener])
            assert instance._store.get_session_count("reused-id") == 0
            assert instance._dag.get_session_nodes("reused-id") == []

            instance.on_session_start("reused-id", platform="cli", context_length=1_000)

            instance.on_session_end(
                "reused-id",
                [
                    opener,
                    {"role": "assistant", "content": "late ignored final"},
                ],
            )

            assert instance._store.get_range("reused-id") == []
            assert instance._dag.get_session_nodes("reused-id") == []
        finally:
            instance.shutdown()

    def test_late_off_current_normal_session_end_dedupes_protected_prefix(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "late-off-current-protected-prefix.db"),
            ignore_session_patterns=["cron"],
            sensitive_patterns_enabled=True,
            sensitive_patterns=["password_assignment"],
        )
        instance = LCMEngine(config=config)
        try:
            raw_opener = {"role": "user", "content": "password = supersecret123 normal opener"}

            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored opener"}])

            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([raw_opener])
            stored_before = instance._store.get_range("reused-id")
            assert len(stored_before) == 1
            assert "supersecret123" not in stored_before[0]["content"]
            assert "LCM sensitive redaction" in stored_before[0]["content"]

            instance.on_session_start("foreground", platform="cli", context_length=1_000)

            instance.on_session_end(
                "reused-id",
                [
                    raw_opener,
                    {"role": "assistant", "content": "normal final"},
                ],
            )

            rows = instance._store.get_range("reused-id")
            assert len(rows) == 2
            assert rows[0]["content"] == stored_before[0]["content"]
            assert rows[1]["content"] == "normal final"
            assert all("supersecret123" not in (row["content"] or "") for row in rows)
            assert instance._store.get_session_count("foreground") == 0
        finally:
            instance.shutdown()

    def test_late_off_current_normal_session_end_filters_ignored_suffix(self, tmp_path, monkeypatch):
        from hermes_lcm import message_patterns as message_patterns_mod

        monkeypatch.setattr(message_patterns_mod, "_regex_engine", _FakeTimeoutRegexEngine)
        config = LCMConfig(
            database_path=str(tmp_path / "late-off-current-ignored-suffix.db"),
            ignore_session_patterns=["cron"],
            ignore_message_patterns=["DROP_LATE_SUFFIX"],
        )
        instance = LCMEngine(config=config)
        try:
            normal_opener = {"role": "user", "content": "normal opener"}

            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored opener"}])

            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([normal_opener])

            instance.on_session_start("foreground", platform="cli", context_length=1_000)
            instance.on_session_end(
                "reused-id",
                [
                    normal_opener,
                    {"role": "assistant", "content": "normal final"},
                    {"role": "user", "content": "DROP_LATE_SUFFIX must stay out"},
                ],
            )

            rows = instance._store.get_range("reused-id")
            assert [(row["role"], row["content"]) for row in rows] == [
                ("user", "normal opener"),
                ("assistant", "normal final"),
            ]
            assert instance._store.get_session_count("foreground") == 0
            assert instance._ignored_message_count == 1
        finally:
            instance.shutdown()

    def test_late_off_current_normal_session_end_dedupes_externalized_prefix(self, tmp_path, monkeypatch):
        # #1016: the subject is an externalized user/assistant row, not the 100k floor.
        monkeypatch.setattr("hermes_lcm.ingest_protection._NON_TOOL_EXTERNALIZATION_FLOOR_CHARS", 0)
        config = LCMConfig(
            database_path=str(tmp_path / "late-off-current-externalized-prefix.db"),
            ignore_session_patterns=["cron"],
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=50,
            large_output_externalization_path=str(tmp_path / "externalized"),
        )
        instance = LCMEngine(config=config)
        try:
            raw_opener = {"role": "user", "content": "x" * 900 + " externalized opener " + "x" * 900}  # #1016: marker outside the stub's head/tail preview

            instance.on_session_start("reused-id", platform="cron", context_length=1_000)
            instance.ingest([{"role": "user", "content": "ignored opener"}])

            instance.on_session_start("reused-id", platform="cli", context_length=1_000)
            instance.ingest([raw_opener])
            stored_before = instance._store.get_range("reused-id")
            assert len(stored_before) == 1
            assert stored_before[0]["content"].startswith("[Externalized payload:")
            assert "externalized opener" not in stored_before[0]["content"]

            instance.on_session_start("foreground", platform="cli", context_length=1_000)

            instance.on_session_end(
                "reused-id",
                [
                    raw_opener,
                    {"role": "assistant", "content": "normal final"},
                ],
            )

            rows = instance._store.get_range("reused-id")
            assert len(rows) == 2
            assert rows[0]["content"] == stored_before[0]["content"]
            assert rows[1]["content"] == "normal final"
            assert instance._store.get_session_count("foreground") == 0
        finally:
            instance.shutdown()

    def test_bypass_tail_trim_makes_progress_when_first_message_has_tool_call(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "bypass-trim-tool-call.db"))
        instance = LCMEngine(config=config)
        messages = [
            {
                "role": "assistant",
                "content": "large assistant " + "x" * 1_000,
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
            {"role": "user", "content": "fresh tail"},
            {"role": "user", "content": "fresh tail 2"},
        ]
        try:
            result = instance._trim_bypass_compacted_to_cap(messages, target_tokens=80)

            assert count_messages_tokens(result) <= 80
        finally:
            instance.shutdown()

    def test_bypass_tail_trim_preserves_live_user_when_dropping_oversized_tool_call_pair(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "bypass-trim-tool-call-pair.db"))
        instance = LCMEngine(config=config)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "latest request"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "big", "arguments": "x" * 10_000},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        try:
            result = instance._trim_bypass_compacted_to_cap(messages, target_tokens=80)

            assert count_messages_tokens(result) <= 80
            assert [(msg.get("role"), msg.get("content")) for msg in result] == [
                ("system", "sys"),
                ("user", "latest request"),
            ]
        finally:
            instance.shutdown()

    def test_bypass_tail_trim_reduces_one_or_two_oversized_messages_under_cap(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "bypass-trim-low-cap.db"))
        instance = LCMEngine(config=config)
        cases = [
            [{"role": "user", "content": "x" * 10_000}],
            [
                {"role": "user", "content": "x" * 10_000},
                {"role": "assistant", "content": "y" * 10_000},
            ],
        ]
        try:
            for messages in cases:
                result = instance._trim_bypass_compacted_to_cap(messages, target_tokens=40)
                assert count_messages_tokens(result) <= 40
        finally:
            instance.shutdown()

    def test_bypass_tail_trim_reduces_structured_text_content_under_cap(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "bypass-trim-structured-content.db"))
        instance = LCMEngine(config=config)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "x" * 10_000},
                    {"type": "input_text", "text": {"value": "y" * 10_000}},
                ],
            }
        ]
        try:
            result = instance._trim_bypass_compacted_to_cap(messages, target_tokens=80)
            assert count_messages_tokens(result) <= 80
        finally:
            instance.shutdown()

    def test_bypassed_compress_uses_message_tokens_when_current_tokens_unknown(self, tmp_path, monkeypatch):
        native_called = False

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                nonlocal native_called
                native_called = True
                self.compression_count += 1
                return [{"role": "user", "content": "native compacted"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-current-tokens-zero.db"),
            ignore_session_patterns=["ignored:*"],
        )
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [{"role": "user", "content": "oversized " + "x" * 2_000}]
        try:
            instance.on_session_start("ignored:zero-tokens", platform="cli", context_length=4_000)
            instance.threshold_tokens = 100
            result = instance.compress(messages, current_tokens=0)

            assert native_called
            assert result == [{"role": "user", "content": "native compacted"}]
        finally:
            instance.shutdown()

    def test_bypassed_native_result_falls_back_when_still_over_cap(self, tmp_path, monkeypatch):
        native_called = False

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self._last_compress_aborted = False
                self._last_summary_error = None

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                nonlocal native_called
                native_called = True
                self.compression_count += 1
                self._last_compress_aborted = True
                self._last_summary_error = "native aborted"
                return list(messages)

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-over-cap-native.db"),
            ignore_session_patterns=["ignored:*"],
            max_assembly_tokens=200,
        )
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "older " + "x" * 2_000},
            {"role": "assistant", "content": "middle " + "y" * 2_000},
            {"role": "user", "content": "fresh " + "z" * 2_000},
        ]
        try:
            instance.on_session_start("ignored:over-cap", platform="cli", context_length=10_000)
            instance.threshold_tokens = 1_000

            result = instance.compress(messages, current_tokens=250)

            assert native_called
            assert count_messages_tokens(result) <= config.max_assembly_tokens
            assert instance._last_compress_aborted is False
        finally:
            instance.shutdown()

    def test_bypassed_native_exception_uses_deterministic_fallback(self, tmp_path, monkeypatch):
        native_called = False

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self._last_compress_aborted = True

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                nonlocal native_called
                native_called = True
                raise RuntimeError("native compressor failed")

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        config = LCMConfig(
            database_path=str(tmp_path / "bypass-native-exception.db"),
            ignore_session_patterns=["ignored:*"],
            max_assembly_tokens=200,
        )
        instance = LCMEngine(config=config)
        instance.model = "test-model"
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "older " + "x" * 2_000},
            {"role": "assistant", "content": "middle " + "y" * 2_000},
            {"role": "user", "content": "fresh " + "z" * 2_000},
        ]
        try:
            instance.on_session_start("ignored:native-error", platform="cli", context_length=10_000)
            instance.threshold_tokens = 1_000

            result = instance.compress(messages, current_tokens=250)

            assert native_called
            assert count_messages_tokens(result) <= config.max_assembly_tokens
            assert instance._last_compress_aborted is False
            assert instance._store.get_session_count("ignored:native-error") == 0
            assert instance._dag.get_session_node_count("ignored:native-error") == 0
        finally:
            instance.shutdown()

    def test_update_from_response(self, engine):
        engine.update_from_response({
            "prompt_tokens": 5000,
            "completion_tokens": 200,
            "total_tokens": 5200,
        })
        assert engine.last_prompt_tokens == 5000

    def test_session_reset(self, engine):
        engine.compression_count = 5
        engine.update_from_response({
            "prompt_tokens": 1050,
            "completion_tokens": 120,
            "total_tokens": 1170,
            "input_tokens": 600,
            "output_tokens": 120,
            "cache_read_tokens": 400,
            "cache_write_tokens": 50,
            "reasoning_tokens": 30,
        })
        engine.on_session_reset()
        assert engine.compression_count == 0
        assert engine.last_prompt_tokens == 0
        assert engine.last_input_tokens == 0
        assert engine.last_output_tokens == 0
        assert engine.last_cache_read_tokens == 0
        assert engine.last_cache_write_tokens == 0
        assert engine.last_reasoning_tokens == 0
        assert engine.cache_metrics_available is False



class TestSessionFiltering:
    def test_on_session_start_marks_ignored_session_and_reports_status(self, tmp_path, caplog):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_ignore.db"),
            ignore_session_patterns=["cron:*"],
            ignore_session_patterns_source="env",
        )
        instance = LCMEngine(config=config)

        with caplog.at_level("INFO", logger="hermes_lcm.engine"):
            instance.on_session_start("cron_123", platform="cron", context_length=1000)

        status = instance.get_status()
        assert status["session_ignored"] is True
        assert status["session_stateless"] is False
        assert status["ignore_session_patterns"] == ["cron:*"]
        assert status["ignore_session_patterns_source"] == "env"
        assert "LCM ignore_session_patterns from env: cron:*" in caplog.text
        assert "matched ignore_session_patterns" in caplog.text

    def test_filter_config_diagnostics_log_only_once_per_engine_instance(self, tmp_path, caplog):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_ignore_once.db"),
            ignore_session_patterns=["cron:*"],
            ignore_session_patterns_source="env",
        )
        instance = LCMEngine(config=config)

        with caplog.at_level("INFO", logger="hermes_lcm.engine"):
            instance.on_session_start("cron_123", platform="cron", context_length=1000)
            instance.on_session_start("cron_456", platform="cron", context_length=1000)

        assert caplog.text.count("LCM ignore_session_patterns from env: cron:*") == 1
        assert caplog.text.count("matched ignore_session_patterns") == 2

    def test_on_session_start_marks_stateless_session_and_reports_status(self, tmp_path, caplog):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_stateless.db"),
            stateless_session_patterns=["telegram:*"],
            stateless_session_patterns_source="env",
        )
        instance = LCMEngine(config=config)

        with caplog.at_level("INFO", logger="hermes_lcm.engine"):
            instance.on_session_start("debug", platform="telegram", context_length=1000)

        status = instance.get_status()
        assert status["session_ignored"] is False
        assert status["session_stateless"] is True
        assert status["stateless_session_patterns"] == ["telegram:*"]
        assert status["stateless_session_patterns_source"] == "env"
        assert "LCM stateless_session_patterns from env: telegram:*" in caplog.text
        assert "matched stateless_session_patterns" in caplog.text

    def test_ignored_session_does_not_write_to_store_or_compact(self, tmp_path):
        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=1,
            database_path=str(tmp_path / "lcm_ignore_behavior.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)
        instance.on_session_start("cron_123", platform="cron", context_length=1000)

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "again"},
        ]

        result = instance.compress(messages)

        assert result == messages
        assert instance._store.get_session_count("cron_123") == 0
        assert instance._dag.get_session_nodes("cron_123") == []
        assert instance.compression_count == 0

    def test_stateless_session_does_not_write_to_store_or_compact(self, tmp_path):
        config = LCMConfig(
            fresh_tail_count=2,
            leaf_chunk_tokens=1,
            database_path=str(tmp_path / "lcm_stateless_behavior.db"),
            stateless_session_patterns=["telegram:*"],
        )
        instance = LCMEngine(config=config)
        instance.on_session_start("debug", platform="telegram", context_length=1000)

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "again"},
        ]

        result = instance.compress(messages)

        assert result == messages
        assert instance._store.get_session_count("debug") == 0
        assert instance._dag.get_session_nodes("debug") == []
        assert instance.compression_count == 0

    def test_ignored_session_does_not_rebind_foreground_view(self, tmp_path):
        """A cron-style ignored session arriving while a foreground session is
        bound must not steal the engine's foreground "current session" view.
        The engine continues to rebind ``_session_id`` so cron's own compress
        / handle_tool_call calls correctly short-circuit on
        ``_session_ignored=True``, but ``current_session_id`` (the property
        every LCM tool reads) keeps pointing at the foreground binding.
        """
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_no_rebind_ignored.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)

        instance.on_session_start(
            "20260506_201605_75a4c6",
            platform="telegram",
            conversation_id="telegram-conversation",
            context_length=200_000,
        )
        instance._store.append(
            "20260506_201605_75a4c6",
            {"role": "user", "content": "hi from telegram"},
            token_estimate=4,
            source="telegram",
        )

        instance.on_session_start(
            "cron_eee06bdbb09b_20260506_210051",
            platform="cron",
            conversation_id="cron-conversation",
            context_length=200_000,
        )

        # Bound state continues to follow the side channel so cron's own
        # compress and handle_tool_call calls correctly short-circuit on
        # _session_ignored=True. This is the pre-fix behavior and must be
        # preserved -- without it, cron messages would ingest into the
        # foreground store via _ingest_messages.
        assert instance._session_id == "cron_eee06bdbb09b_20260506_210051"
        assert instance._session_ignored is True

        # Foreground view stays stable across the rebind. Tools that read
        # current_session_id (lcm_status, lcm_grep default scope, etc.)
        # continue to see the operator's real conversation.
        assert instance._foreground_session_id == "20260506_201605_75a4c6"
        assert instance.current_session_id == "20260506_201605_75a4c6"
        assert instance.current_session_platform == "telegram"
        assert instance.current_conversation_id == "telegram-conversation"

    def test_ignored_session_compress_does_not_leak_into_foreground_store(self, tmp_path):
        """Regression: the foreground view must not come at the cost of
        leaking the side channel's transcript into the foreground store. With
        the bound binding still pointing at the cron session,
        ``_session_ignored=True`` correctly gates ingest so cron's compress
        and should_compress_preflight calls leave the foreground row count
        untouched.
        """
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_no_leak_ignored.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)
        instance.on_session_start(
            "telegram-foreground",
            platform="telegram",
            context_length=200_000,
        )
        instance._store.append(
            "telegram-foreground",
            {"role": "user", "content": "telegram-1"},
            token_estimate=2,
            source="telegram",
        )
        instance.on_session_start(
            "cron_xxx",
            platform="cron",
            context_length=200_000,
        )

        cron_messages = [
            {"role": "system", "content": "sys-cron"},
            {"role": "user", "content": "cron-1"},
            {"role": "assistant", "content": "cron-2"},
            {"role": "user", "content": "cron-3"},
        ]
        instance.should_compress_preflight(cron_messages)
        instance.compress(cron_messages)

        assert instance._store.get_session_count("telegram-foreground") == 1
        assert instance._store.get_session_count("cron_xxx") == 0

    def test_lcm_status_stays_on_foreground_after_cron_tick(self, tmp_path):
        """End-to-end: lcm_status must still report the Telegram session id
        and its row counts after a cron-style ignored session has rebound the
        engine. The bound side channel surfaces only via the diagnostic
        ``side_channel_in_flight`` / ``bound_session_id`` keys.
        """
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_status_after_cron.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)
        instance.on_session_start(
            "20260506_201605_75a4c6",
            platform="telegram",
            conversation_id="telegram-conversation",
            context_length=200_000,
        )
        instance._store.append(
            "20260506_201605_75a4c6",
            {"role": "user", "content": "telegram message"},
            token_estimate=3,
            source="telegram",
        )

        instance.on_session_start(
            "cron_eee06bdbb09b_20260506_210051",
            platform="cron",
            conversation_id="cron-conversation",
            context_length=200_000,
        )

        payload = json.loads(lcm_tools.lcm_status({}, engine=instance))

        assert payload["session_id"] == "20260506_201605_75a4c6"
        assert payload["store"]["messages"] == 1
        assert payload["source_lineage"]["messages_total"] == 1
        assert payload["runtime_identity"]["session_id"] == "20260506_201605_75a4c6"
        assert payload["runtime_identity"]["session_platform"] == "telegram"
        assert payload["runtime_identity"]["conversation_id"] == "telegram-conversation"
        assert payload["runtime_identity"]["bound_session_id"] == "cron_eee06bdbb09b_20260506_210051"
        assert payload["runtime_identity"]["bound_session_platform"] == "cron"
        assert payload["runtime_identity"]["bound_conversation_id"] == "cron-conversation"
        assert payload["lifecycle"]["conversation_id"] == "telegram-conversation"
        assert payload["lifecycle"]["current_session_id"] == "20260506_201605_75a4c6"
        assert payload["session_filters"]["ignored"] is False
        assert payload["session_filters"]["stateless"] is False
        assert payload["session_filters"]["side_channel_active"] is True
        assert (
            payload["session_filters"]["side_channel_session_id"]
            == "cron_eee06bdbb09b_20260506_210051"
        )

    def test_lcm_status_reports_bound_session_when_no_foreground_yet(self, tmp_path):
        """A fresh engine that only ever binds an ignored session must still
        report something usable via lcm_status, with the bound session's
        ignore flag intact so operators can see why the row count is zero.
        """
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_first_bind_ignored.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)

        instance.on_session_start(
            "cron_first_run_20260506_210051",
            platform="cron",
            context_length=200_000,
        )

        assert instance._session_id == "cron_first_run_20260506_210051"
        assert instance._session_ignored is True
        assert instance._foreground_session_id == ""
        assert instance.current_session_id == "cron_first_run_20260506_210051"

        payload = json.loads(lcm_tools.lcm_status({}, engine=instance))
        assert payload["session_id"] == "cron_first_run_20260506_210051"
        assert payload["session_filters"]["ignored"] is True
        assert payload["session_filters"]["side_channel_active"] is False
        assert "side_channel_session_id" not in payload["session_filters"]

    def test_stateless_session_does_not_rebind_foreground_view(self, tmp_path):
        """The same protection applies to stateless side-channel sessions: a
        ``debug:*``-style session must not clobber the foreground
        ``current_session_id`` view, even though it does claim ``_session_id``
        for its own lifecycle gating.
        """
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_no_rebind_stateless.db"),
            stateless_session_patterns=["debug:*"],
        )
        instance = LCMEngine(config=config)

        instance.on_session_start(
            "telegram-foreground",
            platform="telegram",
            context_length=200_000,
        )

        instance.on_session_start(
            "debug:probe-1",
            platform="debug",
            context_length=200_000,
        )

        assert instance._session_id == "debug:probe-1"
        assert instance._session_stateless is True
        assert instance._foreground_session_id == "telegram-foreground"
        assert instance.current_session_id == "telegram-foreground"
        assert instance.current_session_platform == "telegram"

    def test_lcm_command_status_text_is_consistent_with_lcm_status_during_cron_tick(self, tmp_path):
        """The /lcm command's _status_text and the lcm_status tool must agree
        on session_id, session_ignored, and session_stateless during a cron
        tick. Without this, an operator reading /lcm status sees session_id
        for the foreground but ignored=true for the side channel.
        """
        from hermes_lcm.command import _status_text

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_command_consistency.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)
        instance.on_session_start(
            "telegram-foreground",
            platform="telegram",
            conversation_id="telegram-conversation",
            context_length=200_000,
        )
        instance._store.append(
            "telegram-foreground",
            {"role": "user", "content": "telegram row"},
            token_estimate=2,
            source="telegram",
        )
        instance.on_session_start(
            "cron_xxx",
            platform="cron",
            conversation_id="cron-conversation",
            context_length=200_000,
        )

        text = _status_text(instance)
        assert "session_id: telegram-foreground" in text
        assert "conversation_id: telegram-conversation" in text
        assert "lifecycle_current_session_id: telegram-foreground" in text
        assert "source_messages_total: 1" in text
        assert "store_messages: 1" in text
        assert "session_ignored: no" in text
        assert "session_stateless: no" in text
        assert "side_channel_active: yes" in text

    def test_lcm_doctor_retention_targets_foreground_during_cron_tick(self, tmp_path):
        """The /lcm doctor retention surface must report on the foreground
        session, not the cron-style side channel that briefly owns
        engine._session_id. The SQL filter and the row aggregation both need
        to follow current_session_id.
        """
        from hermes_lcm.command import _scan_retention_candidates

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_retention_during_cron.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)
        instance.on_session_start("telegram-foreground", platform="telegram", context_length=200_000)
        instance._store.append(
            "telegram-foreground",
            {"role": "user", "content": "telegram retention test"},
            token_estimate=3,
            source="telegram",
        )
        instance.on_session_start("cron_xxx", platform="cron", context_length=200_000)

        scan = _scan_retention_candidates(instance)
        assert scan["error"] is None
        # Foreground row surfaces; cron's empty session is not the scan
        # target. protected is False because the bound id is cron, not the
        # row we are reporting on.
        assert scan["sessions_analyzed"] == 1
        assert scan["sessions"][0]["session_id"] == "telegram-foreground"
        assert scan["sessions"][0]["protected"] is False

    def test_foreground_view_advances_when_a_real_foreground_arrives(self, tmp_path):
        """``_foreground_session_id`` advances forward through real foreground
        bindings (e.g., compression rollovers) but is never set to an ignored
        or stateless session id. This is the rebind path the bug fix must not
        regress.
        """
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_foreground_advances.db"),
            ignore_session_patterns=["cron:*"],
        )
        instance = LCMEngine(config=config)

        instance.on_session_start(
            "telegram-1",
            platform="telegram",
            context_length=200_000,
        )
        assert instance._foreground_session_id == "telegram-1"

        instance.on_session_start(
            "cron_xxx",
            platform="cron",
            context_length=200_000,
        )
        assert instance._foreground_session_id == "telegram-1"

        instance.on_session_start(
            "telegram-2",
            platform="telegram",
            context_length=200_000,
        )
        assert instance._foreground_session_id == "telegram-2"
        assert instance.current_session_id == "telegram-2"
