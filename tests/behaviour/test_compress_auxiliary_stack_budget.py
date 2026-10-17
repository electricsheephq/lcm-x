"""Keep late auxiliary detection reachable through the compression entry helper."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

harness = importlib.import_module("tests.harness")

# REALDEPTH C-r34.10-0.21.5-1405-delegate-G-1 has the maximum six host
# frames between plugin entry and the auxiliary frame (seven including it).
# Add one for public compress(), skipped by entering _compress_impl below;
# keep the auxiliary frame itself separate and the 32-frame budget unchanged.
OUTSIDE_FRAMES = 6 + 1


def _host_frames(engine: Any, view: list[dict[str, Any]], remaining: int) -> Any:
    if remaining > 1:
        return _host_frames(engine, view, remaining - 1)
    result = engine._compress_impl(view)
    assert engine._in_process_auxiliary_caller_generation("review-session") > 0
    assert engine._thread_context_session_id() == "review-session"
    assert engine.last_compression_noop_reason == "LCM bypassed below threshold: auxiliary thread context"
    return result


class _AuxiliaryAgent:
    session_id = "review-session"
    _parent_session_id = "foreground-session"
    _memory_write_origin = "background_review"

    def compress(self, engine: Any, view: list[dict[str, Any]]) -> Any:
        return _host_frames(engine, view, OUTSIDE_FRAMES)


@pytest.mark.parametrize("bound_session", ["review-session", "foreground-session"])
def test_begin_compression_preserves_late_auxiliary_at_host_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound_session: str,
) -> None:
    harness.pin_char_counter(monkeypatch)
    with harness.engine_context(tmp_path, window=200_000, session_id=bound_session) as engine:
        view = [harness.user("auxiliary replay must stay out of durable history")]
        # compress() normally catches an already-marked caller before this
        # helper. Enter the real inner path to exercise its late-marker recheck
        # and its subsequent _current_auxiliary_prompt_tokens generation walk.
        # The review binding requires late generation-based registration;
        # the foreground binding matches the session-id walk via the parent.
        result = _AuxiliaryAgent().compress(engine, view)
        assert result == view
        assert engine._thread_context_has_auxiliary_session("review-session") == (
            bound_session == "review-session"
        )
        assert engine._store.get_session_count("review-session") == 0
        assert engine._store.get_session_count("foreground-session") == 0
