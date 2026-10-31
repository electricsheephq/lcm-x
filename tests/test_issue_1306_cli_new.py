"""CLI reset-only rebinding must preserve foreground ownership and stored turns."""

import ast
import contextlib
from copy import deepcopy
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import pytest

from hermes_lcm import escalation, tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.engine_registry import resolve_active_lcm_engine


@pytest.fixture
def engine(tmp_path, monkeypatch, request):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    tokens._count_tokens_cached.cache_clear()
    monkeypatch.setattr(escalation, "_call_llm_for_summary",
                        lambda *args, **kwargs: "Earlier turns retained. Expand for details about: turns")
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
                                         fresh_tail_count=2, leaf_chunk_tokens=64,
                                         **getattr(request, "param", {})),
                         hermes_home=str(tmp_path / "home"))
    instance.on_session_start("old", platform="cli", context_length=8192)
    yield instance
    instance.shutdown()


def turn(history, text):
    history.extend([{"role": "user", "content": text},
                    {"role": "assistant", "content": f"answer {text}"}])


@pytest.mark.parametrize("gateway", [False, True], ids=["cli", "gateway"])
def test_new_stores_all_turns_under_new_session(engine, gateway, caplog):
    caplog.set_level(logging.DEBUG, logger="hermes_lcm")
    before = []
    for text in ("before-one", "before-two"):
        turn(before, text)
        engine.ingest(before)
    engine.on_session_end("old", before)
    if gateway:
        engine.on_session_start("new", platform="gateway")
    else:
        engine.on_session_reset()
        engine.bind_session_state(None, "new")
    assert engine.bound_session_id == engine.current_session_id == "new"
    assert engine._lifecycle.get_by_conversation("new").current_session_id == "new"
    assert engine._ingest_cursor == 0
    assert engine.compression_count == 0
    assert resolve_active_lcm_engine(session_id="new") is engine
    after = []
    for text in ("alpha", "beta"):
        turn(after, text)
        engine.ingest(after)
    compressed = engine.compress(after, current_tokens=engine.threshold_tokens + 1, force=True)
    assert engine._last_compression_status == "compacted"
    engine.on_session_end("new", after)
    engine.on_session_start("new", platform="cli", boundary_reason="compression", old_session_id="new")
    turn(compressed, "gamma")
    engine.ingest(compressed)
    rows = engine._store.connection.execute(
        "SELECT session_id, role, content FROM messages ORDER BY store_id").fetchall()
    expected = [("old", row["role"], row["content"]) for row in before]
    for text in ("alpha", "beta", "gamma"):
        expected.extend([("new", "user", text), ("new", "assistant", f"answer {text}")])
    assert rows == expected, (
        f"ingest cursor={engine._ingest_cursor}; "
        f"commit proof held={engine._compress_commit_proof is not None}; "
        f"consulted={(engine._compress_commit_proof or {}).get('consulted')!r}; "
        f"_ingest_cursor_needs_reconcile={engine._ingest_cursor_needs_reconcile}\n"
        f"bound_session_id={engine.bound_session_id!r}; "
        f"current_session_id={engine.current_session_id!r}; "
        f"ignored={engine._session_ignored}; stateless={engine._session_stateless}; "
        f"thread_stateless={engine._thread_context_stateless()}\n"
        f"_last_ingest_reconciliation={engine._last_ingest_reconciliation!r}\n"
        "Last 40 hermes_lcm log records (all levels):\n" + "\n".join(
            f"{record.levelname} {record.name}:{record.lineno} {record.getMessage()}"
            for record in [
                record for record in caplog.records
                if (record.name == "hermes_lcm" or record.name.startswith("hermes_lcm."))
            ][-40:]
        )
    )
    # Duplicate reset-time notifications cannot reset a live cursor/counter.
    cursor, count = engine._ingest_cursor, engine.compression_count
    engine.bind_session_state(None, "new")
    assert (engine._ingest_cursor, engine.compression_count) == (cursor, count)


@pytest.mark.parametrize("candidate", ["old", "unrelated"])
@pytest.mark.parametrize("stored", [False, True])
def test_live_foreground_is_not_rebound(engine, candidate, stored):
    if stored:
        engine.ingest([{"role": "user", "content": "stored"}])
    before = binding_snapshot(engine)
    engine.bind_session_state(None, candidate)
    assert binding_snapshot(engine) == before


def test_auxiliary_end_does_not_enable_foreground_rebind(engine):
    engine._register_auxiliary_session("child")
    engine.on_session_end("child", [])
    engine.bind_session_state(None, "new")
    assert engine.current_session_id == "old"


def test_finalized_foreground_rejects_auxiliary_id(engine):
    engine._register_auxiliary_session("child")
    engine._clear_thread_context_stateless()
    engine.on_session_end("old", [])
    engine.bind_session_state(None, "child")
    assert engine.bound_session_id == "old"


def test_finalized_foreground_rejects_unregistered_child_frame(engine):
    engine.on_session_end("old", [])

    class Child:
        session_id = "child"
        _parent_session_id = "old"
        _subagent_id = "child-worker"

        def bind(self):
            engine.bind_session_state(None, self.session_id)

    Child().bind()
    assert engine.bound_session_id == "old"


def test_real_hermes_bare_reset_supplies_new_id(engine):
    """Execute the installed host's two reset methods without booting services.

    Only their AST is loaded: no Hermes source writes, agent construction,
    credentials, provider requests, or imports with host-side boot effects.
    """
    source = os.environ.get("LCM_REAL_HERMES_SRC")
    if not source:
        pytest.skip("set LCM_REAL_HERMES_SRC for the real reset contract")
    path = Path(source) / "run_agent.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AIAgent")
    methods = {"reset_session_state", "_transition_context_engine_session"}
    agent.body = [node for node in agent.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    agent.bases, agent.decorator_list = [], []
    hook = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_call_engine_hook")
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.Module(body=[future, hook, agent], type_ignores=[])
    namespace = {"logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    host = namespace["AIAgent"]()
    host.context_compressor, host.session_id = engine, "new"
    engine.on_session_end("old", [])
    host.reset_session_state()
    assert engine.bound_session_id == "new"
    engine.ingest([{"role": "user", "content": "alpha"}])
    assert engine._store.get_session_count("new") == 1


def binding_snapshot(engine):
    return (
        engine.bound_session_id, engine.current_session_id, engine._conversation_id,
        engine._foreground_session_id, engine._ingest_cursor, engine.compression_count,
        engine._pending_reset_session_id, engine._stable_use_owner_thread,
        engine._store.connection.execute("SELECT * FROM messages ORDER BY store_id").fetchall(),
        engine._store.connection.execute(
            "SELECT * FROM lcm_lifecycle_state ORDER BY conversation_id").fetchall(),
    )


@pytest.mark.parametrize("kwargs", [{"session_id": ""}, {"session_id": None}, {}],
                         ids=["blank", "none", "missing"])
@pytest.mark.parametrize("finalized", [False, True], ids=["live", "ended"])
def test_background_review_detach_raises_without_state_change(engine, kwargs, finalized):
    history = []
    turn(history, "parent")
    engine.ingest(history)
    if finalized:
        engine.on_session_end("old", history)
    before = binding_snapshot(engine)
    attributes = engine.__dict__.copy()
    with pytest.raises(NotImplementedError, match="cannot detach.*keep compression off"):
        engine.bind_session_state(session_db=None, **kwargs)
    assert binding_snapshot(engine) == before
    assert engine.__dict__ == attributes


@pytest.mark.parametrize("binding", ["fresh", "same", "different"])
def test_agent_construction_bind_is_noop(engine, tmp_path, binding, monkeypatch):
    instance = engine
    if binding == "fresh":
        instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "fresh.db")),
                             hermes_home=str(tmp_path / "fresh-home"))
    else:
        history = []
        turn(history, "parent")
        instance.ingest(history)
    try:
        before = binding_snapshot(instance)
        monkeypatch.setattr(instance, "_on_session_start_unlocked",
                            lambda *a, **kw: pytest.fail("construction must not start a session"))
        # agent_init.py supplies these kwargs under suppress(Exception). Deliberately
        # do not suppress here: the three supported construction cases must not raise.
        instance.bind_session_state(session_db=object(),
                                    session_id="old" if binding == "same" else "constructed")
        assert binding_snapshot(instance) == before
    finally:
        if instance is not engine:
            instance.shutdown()


def host_namespace(relative_path, names):
    source = os.environ.get("LCM_REAL_HERMES_SRC")
    if not source:
        pytest.skip("set LCM_REAL_HERMES_SRC for the real host contract")
    path = Path(source) / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "AIAgent" and "AIAgent" in names:
            node.body = [method for method in node.body if isinstance(method, ast.FunctionDef)
                         and method.name in {"reset_session_state", "_transition_context_engine_session"}]
            node.bases, node.decorator_list = [], []
            selected.append(node)
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + selected,
                        type_ignores=[])
    namespace = {"logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def test_real_hermes_construction_after_finalization_uses_fresh_engine(engine, monkeypatch):
    """Construction clones even a finalized plugin instance before offering a bind."""
    history = []
    turn(history, "resident turn")
    engine.on_session_end("old", history)
    before = binding_snapshot(engine)
    namespace = host_namespace("agent/agent_init.py", {"_select_context_engine"})
    namespace.update(suppress=contextlib.suppress,
                     _ra=lambda: SimpleNamespace(logger=logging.getLogger(__name__)))
    # Exercise the host's general-plugin selection branch, with the finalized
    # resident as its candidate. Named loading creates a new instance in register().
    monkeypatch.setitem(sys.modules, "plugins.context_engine", SimpleNamespace(
        load_context_engine=lambda name: None))
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", SimpleNamespace(
        get_plugin_context_engine=lambda: engine))
    constructed = namespace["_select_context_engine"]({"context": {"engine": engine.name}})
    assert constructed is not engine
    try:
        assert constructed.bound_session_id == ""
        constructed.bind_session_state(session_db=object(), session_id="unrelated")
        assert constructed.bound_session_id == ""
        assert binding_snapshot(engine) == before
        constructed.on_session_start("unrelated", platform="cli")
        constructed.ingest([{"role": "user", "content": "unrelated turn"}])
        assert constructed._store.get_session_count("unrelated") == 1
        assert engine.bound_session_id == engine.current_session_id == "old"
        assert engine._store.get_session_count("old") == len(history)
        assert engine._lifecycle.get_by_conversation("old").current_session_id is None
    finally:
        constructed.shutdown()


def real_reset(engine, target):
    namespace = host_namespace("run_agent.py", {"AIAgent", "_call_engine_hook"})
    host = namespace["AIAgent"]()
    host.context_compressor, host.session_id = engine, target
    host.reset_session_state()


def test_real_background_review_keeps_compression_disabled(engine, caplog):
    detach = host_namespace("agent/background_review.py", {"_detach_fork_compression"})[
        "_detach_fork_compression"]
    fork = SimpleNamespace(context_compressor=engine, compression_enabled=False)
    before = binding_snapshot(engine)
    with caplog.at_level(logging.WARNING):
        detach(fork)
    assert fork.compression_enabled is False
    assert fork.compression_in_place is True
    assert not getattr(fork, "_review_defer_compaction_before_first_response", False)
    assert "background-review compressor detachment failed; keeping compression DISABLED" in caplog.text
    assert binding_snapshot(engine) == before


@pytest.mark.parametrize("mode", ["resume", "branch"])
def test_finalized_reset_target_preserves_rows_and_ownership(engine, mode):
    older = []
    turn(older, "older-turn")
    engine.on_session_end("old", [])
    engine.on_session_start("older", platform="cli")
    engine.ingest(older)
    engine.on_session_end("older", older)
    engine.on_session_start("old", platform="cli")
    parent = []
    turn(parent, "parent-turn")
    engine.ingest(parent)
    engine.on_session_end("old", parent)
    rows_before = engine._store.connection.execute(
        "SELECT * FROM messages ORDER BY store_id").fetchall()
    target = "older" if mode == "resume" else "branch"
    history = deepcopy(older if mode == "resume" else parent)
    real_reset(engine, target)
    assert engine.bound_session_id == engine.current_session_id == target
    assert engine._conversation_id == engine._foreground_session_id == target
    assert engine._lifecycle.get_by_conversation(target).current_session_id == target
    assert resolve_active_lcm_engine(session_id=target) is engine
    assert engine._ingest_cursor_needs_reconcile is (mode == "resume")
    # Replaying an older session reconciles its cursor; a fresh branch stores its
    # copied history once under its own id, without changing either old session.
    engine.ingest(history)
    engine.ingest(history)
    turn(history, "continued")
    engine.ingest(history)
    engine.ingest(history)
    rows_after = engine._store.connection.execute(
        "SELECT * FROM messages ORDER BY store_id").fetchall()
    assert rows_after[:len(rows_before)] == rows_before
    rows = engine._store.connection.execute(
        "SELECT session_id, role, content FROM messages ORDER BY store_id").fetchall()
    expected = [("older", m["role"], m["content"]) for m in older]
    expected += [("old", m["role"], m["content"]) for m in parent]
    additions = history[len(older):] if mode == "resume" else history
    expected += [(target, m["role"], m["content"]) for m in additions]
    assert rows == expected


@pytest.mark.parametrize("target", ["older", "branch"])
def test_real_reset_without_engine_end_preserves_live_guard(engine, target):
    # Current CLI /resume and /branch end only the host DB row. Unlike /new,
    # neither calls engine.on_session_end. R1's live-session guard must remain.
    engine.ingest([{"role": "user", "content": "stored"}])
    real_reset(engine, target)
    assert engine.bound_session_id == "old"
    assert engine._lifecycle.get_by_conversation("old").current_session_id == "old"


def test_real_cli_consecutive_new_stores_under_second_id(engine, monkeypatch):
    """Run the installed CLI's /new and boundary flush, then its real agent reset."""
    source = os.environ.get("LCM_REAL_HERMES_SRC")
    if not source:
        pytest.skip("set LCM_REAL_HERMES_SRC for the real CLI contract")
    path = Path(source) / "hermes_cli/cli_session_mixin.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    cli = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "CLISessionMixin")
    cli.body = [node for node in cli.body if isinstance(node, ast.FunctionDef)
                and node.name in {"new_session", "_launch_session_boundary_memory_flush"}]
    cli.bases, cli.decorator_list = [], []
    ids = iter(("first-new", "second-new"))
    namespace = {"contextlib": contextlib, "uuid": uuid, "new_session_id": lambda now: next(ids),
                 "_reset_model_to_config_default": lambda *a: None}
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [cli],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    # Isolate UI/model/plugin services; lifecycle dispatch remains real host code.
    monkeypatch.setitem(sys.modules, "cli", SimpleNamespace(
        logger=logging.getLogger(__name__), CLI_CONFIG={"agent": {}}, datetime=datetime,
        _parse_reasoning_config=lambda value: value,
        _parse_service_tier_config=lambda value: value, _sync_process_session_id=lambda sid: None))
    monkeypatch.setitem(sys.modules, "hermes_cli.cli_model_switch_mixin", SimpleNamespace(
        _resolve_cli_reasoning=lambda cli: None))
    host = host_namespace("run_agent.py", {"AIAgent", "_call_engine_hook"})["AIAgent"]()
    host.context_compressor, host.session_id = engine, "old"
    instance = namespace["CLISessionMixin"]()
    instance.agent, instance.session_id, instance._session_db = host, "old", None
    instance.reasoning_config = {}
    instance._notify_session_boundary = lambda event: None
    instance.conversation_history = []
    turn(instance.conversation_history, "before-one")
    instance.new_session(silent=True)
    first = instance.session_id
    assert first != "old" and engine.bound_session_id == first
    instance.new_session(silent=True)  # Empty history: the real CLI skips engine end.
    second = instance.session_id
    assert second not in {"old", first}
    for text in ("alpha", "beta"):
        turn(instance.conversation_history, text)
        engine.ingest(instance.conversation_history)
    rows = engine._store.connection.execute(
        "SELECT session_id, role, content FROM messages ORDER BY store_id").fetchall()
    expected = [("old", "user", "before-one"), ("old", "assistant", "answer before-one")]
    expected += [(second, m["role"], m["content"]) for m in instance.conversation_history]
    assert rows == expected  # No alpha/beta under old or the first new id; old rows stay intact.
    assert engine._store.get_session_count(first) == 0
    assert engine.bound_session_id == engine.current_session_id == second
    ended = engine._lifecycle.get_by_conversation(first)
    assert ended.current_session_id is None
    assert ended.last_finalized_session_id == first
    assert engine._lifecycle.get_by_conversation(second).current_session_id == second
    assert resolve_active_lcm_engine(session_id=second) is engine


@pytest.mark.parametrize("evidence", ["failed-ingest", "summary", "frontier", "finalized", "maintenance"])
def test_reset_with_nonempty_evidence_keeps_live_binding(engine, monkeypatch, evidence):
    if evidence == "failed-ingest":
        def fail_ingest(*a, **kw):
            raise RuntimeError("ingest has not committed")
        monkeypatch.setattr(engine, "_ingest_messages", fail_ingest)
        engine.ingest([{"role": "user", "content": "pending"}])
    elif evidence == "summary":
        engine._dag.add_node(SummaryNode(session_id="old", depth=2, summary="retained summary"))
    elif evidence == "frontier":
        engine._lifecycle.advance_frontier("old", "old", 1)
    elif evidence == "finalized":
        engine.on_session_end("old", [])
        engine.on_session_start("old", platform="cli")
    else:
        engine._lifecycle.record_maintenance_attempt("old")
    engine.on_session_reset()
    assert engine._store.get_session_count("old") == 0
    before = binding_snapshot(engine)
    engine.bind_session_state(None, "new")
    assert binding_snapshot(engine) == before


@pytest.mark.parametrize("binding", ["fresh", "same"])
def test_gateway_run_turn_binding_is_noop(tmp_path, monkeypatch, binding):
    # gateway/run_turn.py's detached hygiene attempt also offers this positional
    # bind. Fresh construction and the gateway's explicit start retain ownership.
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "gateway.db")),
                         hermes_home=str(tmp_path / "gateway-home"))
    try:
        if binding == "same":
            instance.on_session_start("gateway", platform="gateway")
        before = binding_snapshot(instance)
        monkeypatch.setattr(instance, "_on_session_start_unlocked",
                            lambda *a, **kw: pytest.fail("gateway bind must not start a session"))
        instance.bind_session_state(object(), "gateway")
        assert binding_snapshot(instance) == before
    finally:
        instance.shutdown()


@pytest.mark.parametrize("engine", [
    {"ignore_session_patterns": ["filtered-*"]},
    {"stateless_session_patterns": ["filtered-*"]},
], indirect=True, ids=["ignored", "stateless"])
@pytest.mark.parametrize("reverse", [False, True], ids=["enter-filter", "leave-filter"])
def test_reset_only_applies_target_session_filters(engine, reverse):
    before = []
    turn(before, "durable old turn")
    engine.on_session_end("old", before)
    rows_before = engine._store.connection.execute(
        "SELECT * FROM messages ORDER BY store_id").fetchall()
    real_reset(engine, "filtered-new")
    bypassed = [{"role": "user", "content": "filtered turn must not be stored"}]
    engine.ingest(bypassed)
    if reverse:
        engine.on_session_end("filtered-new", bypassed)
        real_reset(engine, "normal-new")
        assert engine.bound_session_id == engine.current_session_id == "normal-new"
        assert not engine._session_ignored and not engine._session_stateless
        engine.ingest([{"role": "user", "content": "normal turn must be stored"}])
        assert engine._store.get_session_count("normal-new") == 1
    else:
        assert engine.bound_session_id == engine.current_session_id == "filtered-new"
        assert engine._session_ignored is bool(engine._config.ignore_session_patterns)
        assert engine._session_stateless is bool(engine._config.stateless_session_patterns)
    assert engine._store.get_session_count("filtered-new") == 0
    rows_after = engine._store.connection.execute(
        "SELECT * FROM messages WHERE session_id = 'old' ORDER BY store_id").fetchall()
    assert rows_after == rows_before


@pytest.mark.parametrize("node_owner", ["rotated", "conversation-root", None],
                         ids=["bound-session", "conversation-root", "empty-control"])
def test_reset_only_checks_bound_session_dag(engine, node_owner):
    engine.on_session_end("old", [])
    engine.on_session_start("rotated", platform="cli", conversation_id="conversation-root")
    if node_owner:
        engine._dag.add_node(SummaryNode(session_id=node_owner, depth=2,
                                        summary="carried summary must remain reachable"))
    engine.on_session_reset()
    assert engine._session_id != engine._conversation_id
    assert engine._store.get_session_count("rotated") == 0
    before = binding_snapshot(engine)
    engine.bind_session_state(None, "new")
    if node_owner:
        assert binding_snapshot(engine) == before
        assert engine._dag.get_session_node_count(node_owner) == 1
    else:
        assert engine.bound_session_id == engine.current_session_id == "new"
