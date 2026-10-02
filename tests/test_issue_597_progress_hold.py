"""#597: turn-paced progress refusals and bounded hidden-backlog observability."""

import logging
from types import SimpleNamespace

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.store_complete as store_complete
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.dag import SummaryNode
from tests.test_issue_622_slow_load import _Ctx, _Manager, _install_host, _load_plugin
from tests.test_issue_651_below_threshold_cleanup_only import _engine, _turn, _view


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: (
        "Earlier turns.\nExpand for details about: turns", 1))


@pytest.fixture
def engine(tmp_path):
    instance = _engine(tmp_path)
    instance.threshold_tokens = 100
    try:
        yield instance
    finally:
        instance.shutdown()


def test_progress_refusal_ends_at_turn_end(engine, caplog):
    with caplog.at_level(logging.INFO):
        engine.compress(_view(), current_tokens=1_000)
        assert engine._last_compress_leaves[0] == "conv" and engine._last_compress_leaves[1] >= 1
        engine.record_rejected_compaction()
        assert engine._no_progress_hold_status()["reason"] == "host_rejected_progress"
        assert engine._compression_block_reason() == "cooldown:lcm_host_rejected_progress"
        assert engine._automatic_compression_blocked()
        assert engine.note_turn_complete() is None
        assert not engine._automatic_compression_blocked()
        engine.note_turn_complete()
    assert "held until turn end (cap 600s): host_rejected_progress" in caplog.text
    assert sum("hold ended at turn end" in r.getMessage() for r in caplog.records) == 1


def test_zero_leaf_refusal_stays_held(engine):
    engine.compress(_turn("fresh", 10.0), current_tokens=1_000)
    assert engine._last_compress_leaves == ("conv", 0)
    engine.record_rejected_compaction()
    assert engine._no_progress_hold_status()["reason"] == "host_rejected"
    engine.note_turn_complete()
    assert engine._automatic_compression_blocked()


def test_progress_in_another_conversation_does_not_shorten_hold(engine):
    engine.compress(_view(), current_tokens=1_000)
    assert engine._last_compress_leaves[1] >= 1
    engine.on_session_start("S2", platform="telegram", context_length=200_000, conversation_id="B")
    engine.record_rejected_compaction()
    assert engine._no_progress_hold_status()["reason"] == "host_rejected"
    engine.note_turn_complete()
    assert engine._automatic_compression_blocked()


def test_turn_end_of_another_conversation_keeps_the_hold(engine):
    engine.compress(_view(), current_tokens=1_000)
    engine.record_rejected_compaction()
    assert engine._no_progress_hold_status()["reason"] == "host_rejected_progress"
    engine._conversation_id = "B"
    engine.note_turn_complete()
    assert engine._automatic_compression_blocked()
    engine._conversation_id = "conv"
    engine.note_turn_complete()
    assert not engine._automatic_compression_blocked()


def test_bypassed_turn_end_keeps_the_hold(engine):
    engine.compress(_view(), current_tokens=1_000)
    engine.record_rejected_compaction()
    engine._mark_thread_context_stateless("aux-597")
    try:
        engine.note_turn_complete()
        assert engine._no_progress_hold is not None
    finally:
        engine._clear_thread_context_stateless()
    engine.note_turn_complete()
    assert engine._no_progress_hold is None


def test_progress_hold_has_600_second_backstop(engine, monkeypatch):
    engine.compress(_view(), current_tokens=1_000)
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(lcm_engine, "time", SimpleNamespace(monotonic=lambda: clock.now))
    engine.record_rejected_compaction()
    assert engine._no_progress_hold == (610.0, "host_rejected_progress")
    clock.now = 609.999
    assert engine._automatic_compression_blocked()
    clock.now = 610.0
    assert not engine._automatic_compression_blocked() and engine._no_progress_hold is None


def test_bypassed_refusal_still_arms_nothing(engine):
    engine.compress(_view(), current_tokens=1_000)
    engine._mark_thread_context_stateless("aux-597")
    try:
        engine.record_rejected_compaction()
        assert engine._no_progress_hold is None
    finally:
        engine._clear_thread_context_stateless()


@pytest.mark.parametrize("reason", ["host_rejected", "no_progress"])
def test_turn_end_leaves_other_holds_untouched(engine, reason):
    engine._start_no_progress_hold(reason)
    engine._start_sweep_budget_hold()
    held, sweep = engine._no_progress_hold, engine._sweep_budget_hold_until
    engine.note_turn_complete()
    assert engine._no_progress_hold == held and engine._sweep_budget_hold_until == sweep


def test_latest_call_resets_observations_and_records_exception_leaves(engine, monkeypatch):
    engine._last_compress_leaves = ("old", 9)
    engine._last_hidden_backlog = store_complete.HiddenBacklog(9, True)

    def fail(messages, **kwargs):
        assert engine._last_compress_leaves is None and engine._last_hidden_backlog is None
        engine._foreground_budget.leaves = 1
        raise RuntimeError("after leaf commit")

    monkeypatch.setattr(engine, "_compress_impl", fail)
    with pytest.raises(RuntimeError, match="after leaf commit"):
        engine.compress(_turn("fresh", 10.0), current_tokens=1_000)
    assert engine._last_compress_leaves == ("conv", 1)
    engine.record_rejected_compaction()
    assert engine._no_progress_hold_status()["reason"] == "host_rejected_progress"


def test_turn_notification_never_raises_on_logging_error(engine, monkeypatch):
    engine._start_no_progress_hold("host_rejected_progress")
    monkeypatch.setattr(lcm_engine.logger, "info", lambda *a: (_ for _ in ()).throw(RuntimeError("log")))
    engine.note_turn_complete()
    assert engine._no_progress_hold is None


@pytest.mark.parametrize("path", ["resident", "host", "cold"])
@pytest.mark.parametrize("broken,ingest_fails", [(False, False), (True, False), (False, True)],
                         ids=["ok", "note-fails", "ingest-fails"])
def test_registered_hook_notifies_ingesting_engine_after_ingest(tmp_path, monkeypatch, caplog, path, broken,
                                                                ingest_fails):
    _install_host(monkeypatch, tmp_path)
    module = _load_plugin(f"hermes_lcm_597_{path}_{broken}")
    ctx = _Ctx(_Manager())
    module.register(ctx)
    prototype = ctx.offered
    active = prototype.clone_for_agent() if path != "cold" else prototype
    if path == "resident":
        active.on_session_start("S597", platform="telegram", conversation_id="C597")
    events, history = [], _turn("hook", 10.0)
    ingest = active.ingest

    def spy_ingest(messages):
        events.append("ingest")
        result = ingest(messages)
        if ingest_fails:
            raise RuntimeError("ingest failed")
        return result

    def note():
        assert active._store.get_session_count("S597") == len(history)
        events.append("note")
        if broken:
            raise RuntimeError("notification failed")

    monkeypatch.setattr(active, "ingest", spy_ingest)
    monkeypatch.setattr(active, "note_turn_complete", note)
    if active is not prototype:
        monkeypatch.setattr(prototype, "note_turn_complete", lambda: pytest.fail("wrong engine notified"))
    kwargs = {"context_compressor": active} if path == "host" else {}
    try:
        with caplog.at_level(logging.DEBUG):
            ctx.hooks["post_llm_call"][0](session_id="S597", conversation_id="C597", platform="telegram",
                                        conversation_history=history, **kwargs)
        assert events == ["ingest", "note"]
        assert active._store.get_session_count("S597") == len(history)
        assert ("turn-complete notification error" in caplog.text) is broken
        assert ("post_llm_call ingest error" in caplog.text) is ingest_fails
    finally:
        if active is not prototype:
            active.shutdown()
        prototype.shutdown()


def _hidden(engine, count=6, covered=False):
    ids = [engine._store.append("S", {"role": "user", "content": f"hidden {i}", "timestamp": float(i + 1)},
                                conversation_id=engine._conversation_id) for i in range(count)]
    if covered:
        engine._dag.add_node(SummaryNode(session_id="OTHER", summary="covered rows", token_count=1,
                                         source_token_count=1, source_ids=ids))
    return ids


@pytest.mark.parametrize("limit,expected", [(10, store_complete.HiddenBacklog(6, False)),
                                           (2, store_complete.HiddenBacklog(2, True))])
def test_backlog_count_and_truthiness_preserve_map_side_effect(engine, monkeypatch, limit, expected):
    _hidden(engine)
    monkeypatch.setattr(store_complete, "_SCAN_LIMIT", limit)
    engine._current_compress_store_ids_by_message_id = {999: 999}
    result = engine._store_complete_backlog([], 0)
    assert result == expected and bool(result)
    assert engine._last_hidden_backlog is result
    assert engine._hidden_backlog_label() == ("2+" if limit == 2 else 6)
    assert engine._current_compress_store_ids_by_message_id == {}
    with pytest.raises(AttributeError):
        result.rows = 99


def test_truncated_covered_scan_is_unknown_drained_and_warns_once(engine, monkeypatch, caplog):
    ids = _hidden(engine, covered=True)
    monkeypatch.setattr(store_complete, "_SCAN_LIMIT", 1)
    engine._current_compress_store_ids_by_message_id = {999: 999}
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            result = engine._store_complete_backlog([], 0)
            assert result == (0, True) and not result
            assert engine._hidden_backlog_label() == "unknown"
    warnings = [r.getMessage() for r in caplog.records if "backlog beyond" in r.getMessage()]
    assert len(warnings) == 1
    assert "conversation=conv frontier=0" in warnings[0] and f"last_store_id={ids[3]}" in warnings[0]
    assert "unknown; pass scheduled as drained" in warnings[0] and "hidden 0" not in warnings[0]
    assert engine._current_compress_store_ids_by_message_id == {999: 999}
    engine._conversation_id = "B"
    _hidden(engine, covered=True)
    engine._store_complete_backlog([], 0)  # an independent warning budget per conversation
    assert engine._hidden_backlog_unknown_warned == {"conv", "B"}


def test_backlog_early_return_is_known_empty(engine, monkeypatch):
    monkeypatch.setattr(engine, "_conversation_id", "")
    assert engine._store_complete_backlog([], 0) == (0, False)
    assert engine._hidden_backlog_label() == 0


@pytest.mark.parametrize("hidden", [False, True], ids=["noop", "completed"])
def test_sweep_status_and_info_line_include_latest_hidden_count(engine, hidden, caplog):
    engine._config.threshold_full_sweep_enabled = True
    if hidden:
        _hidden(engine)
    view = _turn("tail", 100.0)
    engine.ingest(view)
    with caplog.at_level(logging.INFO):
        engine.compress(view, current_tokens=1_000)
    assert engine._last_hidden_backlog is not None
    label = engine._hidden_backlog_label()
    assert engine._last_threshold_full_sweep["hidden_rows"] == label
    prefix = "LCM compaction #" if hidden else "LCM compression no-op:"
    assert any(r.getMessage().startswith(prefix) and f", hidden_rows={label}" in r.getMessage()
               for r in caplog.records)


def test_unevaluated_sweep_omits_hidden_count(engine, monkeypatch, caplog):
    engine._config.threshold_full_sweep_enabled = True
    engine._last_hidden_backlog = store_complete.HiddenBacklog(99, True)
    monkeypatch.setattr(engine, "_stub_first_exit", lambda messages, *a: messages)
    with caplog.at_level(logging.INFO):
        engine.compress(_view(), current_tokens=1_000)
    assert engine._last_hidden_backlog is None and engine._hidden_backlog_label() is None
    assert "hidden_rows" not in engine._last_threshold_full_sweep
    assert engine._last_threshold_full_sweep["stop_reason"] == "stub_first_exit"
    assert not any("hidden_rows=" in r.getMessage() for r in caplog.records)
