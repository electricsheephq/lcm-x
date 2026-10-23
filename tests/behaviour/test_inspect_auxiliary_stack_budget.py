"""Keep auxiliary status selection reachable through the inspect facade."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

harness = importlib.import_module("tests.harness")
tools = importlib.import_module("hermes_lcm.tools")
inspect_tools = importlib.import_module("hermes_lcm.inspect_tools")

# Synthetic upper bound from FRAMEDEPTH S18's 22 outside frames on main,
# not a measured host depth: retain 22 intervening frames plus the caller.
OUTSIDE_FRAMES = 22


def test_inspect_handler_adds_no_wrapper_frame() -> None:
    assert tools.lcm_inspect is inspect_tools.lcm_inspect


def _host_frames(engine: Any, remaining: int) -> str:
    if remaining > 1:
        return _host_frames(engine, remaining - 1)
    return str(engine.handle_tool_call("lcm_inspect", {}))


class _AuxiliaryAgent:
    session_id = "review-session"
    _parent_session_id = "S"
    _memory_write_origin = "background_review"

    def inspect(self, engine: Any) -> dict[str, Any]:
        return dict(json.loads(_host_frames(engine, OUTSIDE_FRAMES)))


def test_inspect_preserves_auxiliary_status_at_host_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.pin_char_counter(monkeypatch)
    with harness.engine_context(tmp_path, window=200_000) as engine:
        # Seed distinct diagnostics without a registered/thread-local session
        # marker: only the real frame walker can select this auxiliary state.
        engine._thread_context._last_compression_status = "noop"
        engine._thread_context._last_compression_noop_reason = "auxiliary status probe"
        status = _AuxiliaryAgent().inspect(engine)["compaction"]["last"]
        assert status["status"] == "noop"
        assert status["noop_reason"] == "auxiliary status probe"
