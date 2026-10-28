"""B5 x #1162: a bypassed child pass never moves the bound conversation's zero-leaf streak."""

import logging

import pytest

from tests.harness import assistant, engine_context, pin_char_counter, user


@pytest.mark.parametrize("case", ["empty", "below", "forced"])
def test_child_compress_leaves_parent_zero_leaf_streak(tmp_path, monkeypatch, caplog, case):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200_000) as engine:
        monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
        key = engine._hold_conversation_key()
        parent = {key: {"passes": 2, "last_reason": "parent pass"}}
        engine.__dict__["_zero_leaf_streaks"] = {key: dict(parent[key])}
        # Shape only: a bypassed pass never selects a leaf, so blocked= is unreachable here (not an isolation proof).
        engine._store_complete_blocked = (7, "user", "raw")
        engine._mark_thread_context_stateless("child")
        view = [] if case == "empty" else [user("child request"), assistant("child reply")]
        try:
            with caplog.at_level(logging.INFO, logger="hermes_lcm"):
                engine.compress(view, force=case == "forced")
        finally:
            engine._clear_thread_context_stateless("child")
        assert engine.__dict__["_zero_leaf_streaks"] == parent
        stops = [r.getMessage() for r in caplog.records if r.getMessage().startswith("LCM compaction stop:")]
        assert len(stops) == 1 and " blocked=" not in stops[0]
