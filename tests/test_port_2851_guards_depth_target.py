"""Rebinding during a depth pass must skip the predecessor's staged fallback."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import Mock

import pytest

from hermes_lcm.tokens import count_messages_tokens
from tests.test_issue_2851_rebind_cursor import WAIT, _bind, _stored
from tests.test_issue_628_summary_circuit import _provider
from tests.test_issue_671_stub_first_exit import exit_engine, make_engine as make_engine, tool_view


@pytest.mark.parametrize("checkpoint", ["fallback", "leaf-error"])
def test_rebind_during_depth_pass(make_engine, monkeypatch, checkpoint):
    engine = exit_engine(make_engine, temporal_rollups_enabled=False, l3_truncate_tokens=2)
    engine._config.depth_target_ratio = 120 / engine.context_length
    view = tool_view()
    fallback, compress_impl = engine._use_stub_first_fallback, engine._compress_impl
    rebound, fallback_observations, impl_returns = [], [], []

    def successor_state():
        return (engine._ingest_cursor, _stored(engine, "B"), deepcopy(engine._last_stub_first_exit),
                deepcopy(engine._last_threshold_full_sweep), engine._dag.get_session_nodes("B"))

    def rebind(state):
        assert state.depth_target_tokens > 0 and state.leaf_passes == 0
        assert callable(state.stub_first_fallback)
        pool.submit(_bind, engine, "B", "convB").result(timeout=WAIT)
        rebound.append(successor_state())

    def at_fallback(state):
        rebind(state)
        staged = state.stub_first_fallback = Mock(wraps=state.stub_first_fallback)
        result = fallback(state)
        fallback_observations.append((result, staged.call_count))
        return result

    def fail_first_leaf(state, leaf, sweep_step_done):
        assert leaf.summary_input_chunk
        rebind(state)
        raise RuntimeError("synthetic first-leaf summary failure")

    def record_impl_return(messages, *args, **kwargs):
        result = compress_impl(messages, *args, **kwargs)
        impl_returns.append(result is messages)
        return result

    with ThreadPoolExecutor(max_workers=1) as pool:
        if checkpoint == "fallback":
            _provider(monkeypatch, None)
            monkeypatch.setattr(engine, "_use_stub_first_fallback", at_fallback)
        else:
            monkeypatch.setattr(engine, "_summarize_selected_leaf", fail_first_leaf)
        monkeypatch.setattr(engine, "_compress_impl", record_impl_return)
        assert engine.compress(view, current_tokens=count_messages_tokens(view)) == view

    assert engine._session_id == "B" and rebound == [successor_state()]
    assert impl_returns == [True]
    if checkpoint == "fallback":
        assert fallback_observations == [(None, 0)], "the stale staged fallback was invoked"
