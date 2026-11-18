"""Compatibility pins for the UID opt-in and turn-path guards."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from hermes_lcm import compaction, config
from hermes_lcm.host_uid import _uid_recognition_pass
from tests.invariants.test_issue_2815_c4 import _closing
from tests.invariants.test_issue_2815_uid_restart import _open
from tests.test_compaction_telemetry import _hot_usage


def test_uid_recognition_disabled_for_bare_owner(monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    assert config.uid_recognition_enabled(object()) is False


def test_uid_pass_allows_bare_owner(monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    owner = object()
    assert _uid_recognition_pass(lambda self: self)(owner) is owner


def test_same_thread_nesting_keeps_other_threads_out_until_outer_returns(tmp_path, monkeypatch):
    outer, nested, other = [], [], []
    with _closing(_open(tmp_path)) as engine, ThreadPoolExecutor(max_workers=1) as worker:
        def owned(messages, *_args):
            if messages is outer:
                assert engine.compress(nested, current_tokens=0) is not nested
                assert worker.submit(engine.compress, other, 0).result(timeout=1) is other
                assert engine._compress_reentry_refusals == 1
            return list(messages)

        monkeypatch.setattr(engine, "_compress_owned", owned)
        assert engine.compress(outer, current_tokens=0) is not outer
        assert worker.submit(engine.compress, other, 0).result(timeout=1) is not other
        assert engine._compress_reentry_refusals == 1


def test_turn_telemetry_accepts_store_with_only_legacy_updater(tmp_path, monkeypatch):
    class LegacyStore:
        read_compaction_telemetry = Mock(return_value=None)
        increment_compaction_telemetry = Mock(return_value=None)

    with _closing(_open(tmp_path)) as engine, monkeypatch.context() as patch:
        store = LegacyStore()
        patch.setattr(engine, "_store", store)
        engine._compaction_telemetry_counter_rebaseline_pending = True
        engine._compaction_telemetry_turn_reset_pending = True

        engine.update_from_response(_hot_usage(prompt=400))

        store.increment_compaction_telemetry.assert_called_once()
        args, kwargs = store.increment_compaction_telemetry.call_args
        conversation_id, delta, record = args
        assert kwargs == {}
        assert (conversation_id, delta) == (engine._conversation_id, 0)
        assert record["last_observed_prompt_tokens"] == 400
        assert not engine._compaction_telemetry_counter_rebaseline_pending
        assert not engine._compaction_telemetry_turn_reset_pending


def test_admission_reads_hold_after_retained_stop_check(tmp_path, monkeypatch):
    with _closing(_open(tmp_path)) as engine:
        def clear_hold():
            engine._no_progress_hold = None
            return False

        engine._no_progress_hold = (10**12, "hidden_only")
        monkeypatch.setattr(engine, "_retained_stop_hold_active", clear_hold)
        monkeypatch.setattr(engine, "_hold_fit_only_applies", lambda _tokens: False)
        blocks = Mock(return_value=True)
        monkeypatch.setattr(engine, "_no_progress_hold_blocks", blocks)
        monkeypatch.setattr(engine, "_compress_impl", lambda messages, **_kwargs: messages)

        engine.compress([], current_tokens=100)

        blocks.assert_not_called()


def test_hung_route_reads_strike_after_rejected_result_callback(tmp_path, monkeypatch):
    with _closing(_open(tmp_path)) as engine:
        engine._hung_summary_strike = (engine._hold_conversation_key(), "primary")
        state = compaction._CompressionState(sweep_stop_reason="summary_result_rejected")
        state.budget = engine._new_foreground_budget()
        state.budget.hung_route = "primary"
        start_hold = Mock()
        monkeypatch.setattr(engine, "_start_sweep_budget_hold", start_hold)
        monkeypatch.setattr(engine, "_hung_summary_route_has_fallback", lambda _budget: False)

        engine._settle_leaf_stop(state, lambda: setattr(engine, "_hung_summary_strike", None))

        start_hold.assert_called_once_with(turn_paced=True)
