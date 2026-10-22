"""Stopped callbacks must prove ownership after acquiring the engine lease."""
from threading import Event, Thread
from unittest.mock import Mock

import pytest
from agent.context_engine import ContextEngine

from tests.test_issue_622_slow_load import _Ctx, _Manager, _install_host, _load_plugin


def rows(tag):
    return [{"role": "user", "content": f"{tag} question", "timestamp": 1000.0},
            {"role": "assistant", "content": f"{tag} answer", "timestamp": 1001.0}]


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    monkeypatch.setattr(ContextEngine, "on_turn_complete", lambda *a, **k: None, raising=False)
    _install_host(monkeypatch, tmp_path)
    module = _load_plugin("hermes_lcm_1623_ownership")
    ctx = _Ctx(_Manager())
    module.register(ctx)
    engine = ctx.offered.clone_for_agent()
    yield ctx, engine
    engine.shutdown()
    ctx.offered.shutdown()


def completed(ctx, engine, session, history):
    ctx.hooks["post_llm_call"][0](session_id=session, platform="cli",
                                  conversation_history=history, context_compressor=engine)


@pytest.mark.parametrize("switch", ["resume", "branch", "compression-tip"])
def test_switched_stopped_history_waits_for_correct_session_catchup(plugin, switch):
    # Replays the pinned CLI reset/bind and turn_facade_lease tip-adoption calls.
    ctx, engine = plugin
    engine.on_session_start("A", platform="cli")
    original = rows("original")
    completed(ctx, engine, "A", original)
    before = engine._store.get_session_messages("A")
    if switch != "compression-tip":
        engine.on_session_reset()
        engine.bind_session_state(session_db=None, session_id="X")
    history = (original if switch == "branch" else rows("resumed")) + rows("stopped")
    engine.on_turn_complete(history, interrupted=True, session_id="X")
    assert engine._store.get_session_messages("A") == before
    assert engine._store.get_session_count("X") == 0
    final = history + rows("next")
    completed(ctx, engine, "X", final)
    completed(ctx, engine, "X", final)
    assert engine._store.get_session_messages("A") == before
    assert [m["content"] for m in engine._store.get_session_messages("X")] == [m["content"] for m in final]


@pytest.mark.parametrize("proof_gap", ["cold", "pending-reset", "reconcile", "no-cache", "changed-prefix", "short-prefix"])
def test_unproven_ownership_skips_without_binding_or_noting(plugin, monkeypatch, proof_gap):
    ctx, engine = plugin
    engine.on_session_start("A", platform="cli")
    history = rows("warm")
    if proof_gap != "cold":
        completed(ctx, engine, "A", history)
    if proof_gap == "pending-reset":
        engine._pending_reset_session_id = "A"
    elif proof_gap == "reconcile":
        engine._ingest_cursor_needs_reconcile = True
    elif proof_gap == "no-cache":
        engine._last_active_replay_source_identities = None
    elif proof_gap == "changed-prefix":
        history = rows("different")
    elif proof_gap == "short-prefix":
        history = history[:1]
    before = engine._store.get_session_messages("A")
    note = Mock()
    monkeypatch.setattr(engine, "note_turn_complete", note)
    engine.on_turn_complete(history, interrupted=True, session_id="forbidden-meta-binding")
    assert engine._session_id == "A"
    assert engine._store.get_session_messages("A") == before
    note.assert_not_called()


@pytest.mark.parametrize("same_prefix", [False, True], ids=["different-prefix", "inherited-prefix"])
def test_new_rebind_before_callback_lease_rechecks_new_binding(plugin, monkeypatch, same_prefix):
    ctx, engine = plugin
    engine.on_session_start("A", platform="cli")
    old = rows("old")
    completed(ctx, engine, "A", old)
    original_run = engine._run_stably
    waiting, release = Event(), Event()

    def delayed_run(operation, *, timeout):
        waiting.set()
        assert release.wait(5)
        return original_run(operation, timeout=timeout)

    monkeypatch.setattr(engine, "_run_stably", delayed_run)
    callback = Thread(target=engine.on_turn_complete,
                      args=(old + rows("stopped-old"),), kwargs={"interrupted": True})
    callback.start()
    try:
        assert waiting.wait(5)
        # CLI /new finalizes, resets, and offers the new binding while the
        # callback is waiting. A warm new prefix also must not admit old rows.
        engine.on_session_end("A", old)
        engine.on_session_reset()
        engine.bind_session_state(session_db=None, session_id="N")
        new = old if same_prefix else rows("new")
        engine.ingest(new)
    finally:
        release.set()
        callback.join(5)
    assert not callback.is_alive()
    assert engine._session_id == "N"
    assert [m["content"] for m in engine._store.get_session_messages("N")] == [m["content"] for m in new]
    assert [m["content"] for m in engine._store.get_session_messages("A")] == [m["content"] for m in old]


@pytest.mark.parametrize("flush", ["next-turn", "session-end"])
def test_post_finalize_failed_boundary_is_stored_once(plugin, flush):
    ctx, engine = plugin
    engine.on_session_start("A", platform="cli")
    warm = rows("warm")
    completed(ctx, engine, "A", warm)
    stopped = warm + [rows("stopped")[0]]
    engine.on_turn_complete(stopped, interrupted=True)
    # The pinned host closes a user-only failed turn after finalize_turn.
    boundary = {"role": "assistant", "content": "Operation interrupted.", "timestamp": 1002.0}
    closed = stopped + [boundary]
    # A user-only stop remains mutable until the host closes or retries it.
    assert engine._store.get_session_count("A") == len(warm)
    final = closed + rows("next") if flush == "next-turn" else closed
    if flush == "next-turn":
        completed(ctx, engine, "A", final)
        completed(ctx, engine, "A", final)
    else:
        engine.on_session_end("A", final)
        engine.on_session_end("A", final)
    stored = engine._store.get_session_messages("A")
    assert [m["content"] for m in stored] == [m["content"] for m in final]
