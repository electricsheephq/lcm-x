"""Shared fixtures and unchanged helpers for LCM engine integration tests."""

import re
import time
from pathlib import Path

import pytest


from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig()
    config.fresh_tail_count = 4  # small for testing
    config.leaf_chunk_tokens = 100  # low threshold for testing
    config.database_path = str(tmp_path / "lcm_test.db")
    e = LCMEngine(config=config)
    e._session_id = "test-session"
    e.context_length = 200000
    e.threshold_tokens = int(200000 * config.context_threshold)
    try:
        yield e
    finally:
        e.shutdown()



@pytest.fixture
def externalized_search_engine(tmp_path):
    home = tmp_path / "hermes-search"
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_externalized_search.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
    )
    instance = LCMEngine(config=config, hermes_home=str(home))
    instance._session_id = "test-session"
    try:
        yield instance
    finally:
        instance.shutdown()



def _real_lcm_summary(engine, session_id, summary, expand_hint, depth=0):
    """Render an LCM summary part backed by a real DAG node. Reconciliation treats
    only DAG-verified summaries as scaffold; unverified summary text is content (#486)."""
    node_id = engine._dag.add_node(SummaryNode(
        session_id=session_id, depth=depth, summary=summary, token_count=5,
        source_token_count=5, source_ids=[], source_type="messages",
        created_at=time.time(), expand_hint=expand_hint,
    ))
    label = {0: "Recent", 1: "Session Arc", 2: "Durable"}.get(depth, f"Depth-{depth}")
    return f"[{label} Summary (d{depth}, node {node_id})]\n{summary}\n[Expand for details: {expand_hint}]"



class _FakeTimeoutPattern:
    def __init__(self, pattern):
        self.pattern = pattern
        self._compiled = re.compile(pattern)

    def search(self, text, *, timeout=None):
        assert timeout is not None
        return self._compiled.search(text)



class _FakeTimeoutRegexEngine:
    error = re.error

    @staticmethod
    def compile(pattern):
        return _FakeTimeoutPattern(pattern)



class TestMessageFiltering:
    @pytest.fixture(autouse=True)
    def _timeout_capable_regex_engine(self, monkeypatch):
        from hermes_lcm import message_patterns as message_patterns_mod

        monkeypatch.setattr(message_patterns_mod, "_regex_engine", _FakeTimeoutRegexEngine)

    def _make_engine(self, tmp_path, db_name, **config_kwargs):
        config_kwargs.setdefault("threshold_full_sweep_enabled", False)  # #1013: pre-#1013 single-pass compaction
        config = LCMConfig(
            database_path=str(tmp_path / db_name),
            **config_kwargs,
        )
        engine = LCMEngine(config=config)
        engine.on_session_start("user-123", platform="telegram", context_length=1000)
        return engine



class TestSessionRollover:
    class HostAgentFrame:
        def __init__(self, session_id: str, parent_session_id: str, hermes_home: Path):
            self.session_id = session_id
            self._parent_session_id = parent_session_id
            self._hermes_home = hermes_home
            self.enabled_toolsets = ["memory", "skills"]
            self.log_prefix = "[subagent-test] "
            self._subagent_id = session_id
            self._delegate_depth = 1

        def on_session_start(self, engine: LCMEngine, **kwargs) -> None:
            engine.on_session_start(
                self.session_id,
                hermes_home=str(self._hermes_home),
                platform=kwargs.pop("platform", "telegram"),
                context_length=kwargs.pop("context_length", 200000),
                **kwargs,
            )

        def update_model(self, engine: LCMEngine, context_length: int) -> None:
            engine.update_model("child-model", context_length)

        def should_compress_preflight(self, engine: LCMEngine, messages):
            return engine.should_compress_preflight(messages)

        def should_compress(self, engine: LCMEngine):
            return engine.should_compress()

        def compress(self, engine: LCMEngine, messages, **kwargs):
            return engine.compress(messages, **kwargs)

        def update_from_response(self, engine: LCMEngine, usage: dict) -> None:
            engine.update_from_response(usage)

        def thread_context_session_id(self, engine: LCMEngine) -> str:
            return engine._thread_context_session_id()

        def thread_context_stateless(self, engine: LCMEngine) -> bool:
            return engine._thread_context_stateless()

        def on_session_end(self, engine: LCMEngine, messages) -> None:
            engine.on_session_end(self.session_id, messages)

        def on_compression_boundary(self, engine: LCMEngine, new_session_id: str, **kwargs) -> None:
            engine.on_session_start(
                new_session_id,
                hermes_home=str(self._hermes_home),
                boundary_reason="compression",
                old_session_id=self.session_id,
                platform=kwargs.pop("platform", "telegram"),
                context_length=kwargs.pop("context_length", 200000),
                **kwargs,
            )

        def on_compression_boundary_from(
            self,
            engine: LCMEngine,
            old_session_id: str,
            **kwargs,
        ) -> None:
            engine.on_session_start(
                self.session_id,
                hermes_home=str(self._hermes_home),
                boundary_reason="compression",
                old_session_id=old_session_id,
                platform=kwargs.pop("platform", "telegram"),
                context_length=kwargs.pop("context_length", 200000),
                **kwargs,
            )

    def _start_host_child(
        self,
        engine: LCMEngine,
        hermes_home: Path,
        session_id: str,
        parent_session_id: str,
        **kwargs,
    ) -> HostAgentFrame:
        frame = self.HostAgentFrame(session_id, parent_session_id, hermes_home)
        frame.on_session_start(
            engine,
            **kwargs,
        )
        return frame
