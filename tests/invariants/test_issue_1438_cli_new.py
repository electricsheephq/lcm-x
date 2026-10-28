"""I1/I12: a filtered first binding must not strand later CLI /new turns."""

import os
from pathlib import Path

import pytest

from tests.harness import engine_context
from tests.test_issue_1306_cli_new import real_reset


@pytest.mark.parametrize("pattern", ["ignore_session_patterns", "stateless_session_patterns"])
@pytest.mark.parametrize("history_present", [False, True], ids=["empty", "with-history"])
@pytest.mark.parametrize("target", ["normal-new", "filtered-new"])
def test_new_from_filtered_first_session(
    tmp_path: Path, pattern: str, history_present: bool, target: str,
) -> None:
    with engine_context(
        tmp_path, window=8192, session_id="filtered-first", **{pattern: ["filtered-*"]},
    ) as engine:
        history = [{"role": "user", "content": "filtered original turn"}] if history_present else []
        engine.ingest(history)
        # CLI /new only ends nonempty history, then resets without a start hook.
        if history:
            engine.on_session_end("filtered-first", history)
        if os.environ.get("LCM_REAL_HERMES_SRC"):
            real_reset(engine, target)  # type: ignore[no-untyped-call]
        else:
            engine.on_session_reset()
            engine.bind_session_state(None, target)
        after = [{"role": "user", "content": "new question"},
                 {"role": "assistant", "content": "new answer"}]
        engine.ingest(after)
        engine.ingest(after)
        rows = engine._store.connection.execute(
            "SELECT session_id, role, content FROM messages ORDER BY store_id").fetchall()
        print(f"{pattern} history={history_present} target={target} rows={len(rows)}")
        expected = [] if target.startswith("filtered-") else [
            (target, row["role"], row["content"]) for row in after]
        assert rows == expected
        assert engine.bound_session_id == engine.current_session_id == target
        assert engine._foreground_session_id == target
        ended = engine._lifecycle.get_by_conversation("filtered-first")
        assert ended.current_session_id is None
        assert ended.last_finalized_session_id == "filtered-first"
        assert bool(engine._session_ignored or engine._session_stateless) == target.startswith("filtered-")


@pytest.mark.parametrize("pattern", ["ignore_session_patterns", "stateless_session_patterns"])
def test_filtered_first_bind_without_reset_is_noop(tmp_path: Path, pattern: str) -> None:
    with engine_context(
        tmp_path, window=8192, session_id="filtered-first", **{pattern: ["filtered-*"]},
    ) as engine:
        engine.bind_session_state(None, "normal-new")
        assert engine.bound_session_id == "filtered-first"
        assert engine._foreground_session_id == ""


@pytest.mark.parametrize("pattern", ["ignore_session_patterns", "stateless_session_patterns"])
def test_filtered_side_channel_cannot_replace_foreground(tmp_path: Path, pattern: str) -> None:
    with engine_context(
        tmp_path, window=8192, session_id="normal-first", **{pattern: ["filtered-*"]},
    ) as engine:
        original = [{"role": "user", "content": "durable foreground turn"}]
        engine.ingest(original)
        engine.on_session_start("filtered-side", platform="cli")
        engine.on_session_reset()
        engine.bind_session_state(None, "normal-new")
        assert engine.bound_session_id == "filtered-side"
        assert engine._foreground_session_id == "normal-first"
        assert engine._store.get_session_count("normal-first") == 1
        assert engine._store.get_session_count("normal-new") == 0
