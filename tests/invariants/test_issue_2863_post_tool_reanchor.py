"""I1: report post-tool compaction progress only when Hermes can re-anchor the turn."""

import sys
from types import ModuleType

import pytest

from hermes_lcm import compaction
from tests.harness import engine_context, pin_char_counter


@pytest.mark.parametrize("capability", ["capable", "incapable", "unreadable"])
def test_long_turn_recovery_requires_post_tool_reanchor(tmp_path, monkeypatch, capability):
    def reanchors(*, current_turn_user_idx):
        pass

    def old_preflight():
        pass

    preflight = ModuleType("agent.turn_preflight")
    preflight.compress_after_tool_results = reanchors if capability == "capable" else old_preflight
    if capability == "unreadable":
        preflight.compress_after_tool_results.__signature__ = "unreadable"
    monkeypatch.setitem(sys.modules, "agent.turn_preflight", preflight)
    pin_char_counter(monkeypatch)

    @compaction._reports_progress
    def shorten(self, messages, *args, **kwargs):
        return [{**messages[0], "content": "short"}]

    monkeypatch.setattr(compaction.CompactionMixin, "compress", shorten)
    compaction._host_has_post_tool_reanchor.cache_clear()
    try:
        with engine_context(tmp_path, window=32768, embeddings_enabled=False) as engine:
            capable = capability == "capable"
            assert hasattr(engine, "_last_compression_made_progress") is capable
            # Subsequent hooks use the capability cached on initialization.
            monkeypatch.delitem(sys.modules, "agent.turn_preflight")
            messages = [{"role": "user", "content": "A long current objective. " * 100}]
            assert engine.compress(messages, current_tokens=1000) == [{"role": "user", "content": "short"}]
            assert getattr(engine, "_last_compression_made_progress", False) is capable
            engine._verify_compaction_cleared_threshold = True
            engine.awaiting_real_usage_after_compression = True
            engine.update_from_response({})
            assert engine._verify_compaction_cleared_threshold is (not capable)
            assert engine.awaiting_real_usage_after_compression is (not capable)
    finally:
        compaction._host_has_post_tool_reanchor.cache_clear()
