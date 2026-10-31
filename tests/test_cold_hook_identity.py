"""Exercise the real hook, registry, store and publication; stub only summaries."""
import ast
import logging
from pathlib import Path

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm import plugin_identity
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.engine_registry import (
    ActiveEngineUseStatus,
    use_active_lcm_engine,
    use_cold_lcm_engine,
    use_lcm_engine,
)


def _hook_for(engine):
    # Extract the production hook without registering a plugin against a live
    # profile. Its registry, binding and ingest operations are not mocked.
    source = Path(engine_module.__file__).with_name("__init__.py")
    tree = ast.parse(source.read_text())
    nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_engine_bound_session_id", "_on_post_llm_call")
    ]
    namespace = dict(
        engine=engine,
        active=True,
        _identity=plugin_identity,
        logger=logging.getLogger(__name__),
        ActiveEngineUseStatus=ActiveEngineUseStatus,
        use_active_lcm_engine=use_active_lcm_engine,
        use_cold_lcm_engine=use_cold_lcm_engine,
        use_lcm_engine=use_lcm_engine,
    )
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_on_post_llm_call"]


def _make_engine(tmp_path):
    return LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "fixture.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        incremental_max_depth=0,
        temporal_rollups_enabled=False,
        embeddings_enabled=False,
    ))


HISTORY = [
    {"role": "assistant", "content": "Project cobalt uses SQLite for authoritative history."},
    {"role": "assistant", "content": "The deployment checkpoint is approved."},
    {"role": "user", "content": "Keep the newest turn in context."},
]


@pytest.mark.parametrize("missing_identity", [None, ""])
def test_unknown_cold_identity_defers_then_publishes_and_expands(
    tmp_path, monkeypatch, caplog, missing_identity,
):
    engine = _make_engine(tmp_path)
    monkeypatch.setattr(
        engine_module, "summarize_with_escalation",
        lambda **kwargs: ("Cobalt checkpoint approved.", 1),
    )
    try:
        payload = dict(
            session_id="cold-regression", platform="discord",
            conversation_history=HISTORY,
        )
        if missing_identity is not None:
            payload["conversation_id"] = missing_identity
        with caplog.at_level(logging.WARNING):
            _hook_for(engine)(**payload)
        # No provisional session-as-conversation binding is safe on a cold hook.
        assert engine._store.get_range("cold-regression") == []
        assert not engine._session_id
        assert "deferred cold ingest" in caplog.text
        engine.on_session_start(
            "cold-regression", platform="discord",
            conversation_id="gateway-regression", context_length=200000,
        )
        # Retry the actual host post-turn hook after normal session startup.
        # The host still omits identity; the resident registry binding proves it.
        hook = _hook_for(engine)
        hook(**payload)
        rows = engine._store.get_range("cold-regression")
        first_ids = [row["store_id"] for row in rows]
        assert [row["content"] for row in rows] == [m["content"] for m in HISTORY]
        hook(**payload)
        assert [row["store_id"] for row in engine._store.get_range("cold-regression")] == first_ids
        # Explicit identity must find the same engine and not duplicate history.
        hook(**dict(payload, conversation_id="gateway-regression"))
        assert [row["store_id"] for row in engine._store.get_range("cold-regression")] == first_ids
        rows = engine._store.get_range("cold-regression")
        assert len(rows) == len(HISTORY)
        assert all(row["conversation_id"] == "gateway-regression" for row in rows)
        engine.compress(HISTORY)
        assert engine.last_compression_status == "compacted"
        nodes = engine._dag.get_session_nodes("cold-regression")
        assert len(nodes) == 1
        assert nodes[0].source_ids == first_ids[:2]
        assert len(nodes[0].source_ids) == len(set(nodes[0].source_ids))
        assert [row["store_id"] for row in engine._store.get_range("cold-regression")] == first_ids
        lifecycle = engine._lifecycle.get_by_conversation("gateway-regression")
        assert lifecycle.current_frontier_store_id == 2
        expanded = engine.handle_tool_call("lcm_expand", {"node_id": nodes[0].node_id})
        assert HISTORY[0]["content"] in expanded
        assert HISTORY[1]["content"] in expanded
    finally:
        engine.shutdown()


@pytest.mark.parametrize("mode", ["explicit", "gateway_key", "resident", "host"])
def test_proven_identity_still_ingests(tmp_path, mode):
    engine = _make_engine(tmp_path)
    try:
        payload = dict(
            session_id="proven-regression", platform="discord",
            conversation_history=HISTORY,
        )
        if mode == "explicit":
            payload["conversation_id"] = "proven-gateway"
        elif mode == "gateway_key":
            payload["gateway_session_key"] = "proven-gateway"
        else:
            engine.on_session_start(
                "proven-regression", platform="discord",
                conversation_id="proven-gateway",
            )
            if mode == "host":
                payload["context_compressor"] = engine
        _hook_for(engine)(**payload)
        rows = engine._store.get_range("proven-regression")
        assert len(rows) == len(HISTORY)
        assert all(row["conversation_id"] == "proven-gateway" for row in rows)
    finally:
        engine.shutdown()
