"""I12: native fallback publication cannot revive an ended child (#1476)."""
from pathlib import Path
from typing import Any

import pytest

from tests.harness import engine_context


@pytest.mark.parametrize("end_first", [False, True])
def test_child_setter_publication_liveness(tmp_path: Path, end_first: bool) -> None:
    ended: list[str] = []
    resets: list[bool] = []

    class Native:
        def on_session_end(self, sid: str, messages: list[dict[str, Any]]) -> None:
            assert not engine._auxiliary_session_lock._is_owned()
            assert messages == []
            ended.append(sid)

        def on_session_reset(self) -> None:
            assert not engine._auxiliary_session_lock._is_owned()
            resets.append(True)

    class HostChild:
        _subagent_id = "synthetic-child"
        _parent_session_id = "parent"
        session_id = "child"

        def publish(self, engine: Any) -> dict[str, Any]:
            engine._mark_thread_context_stateless("child")
            state: dict[str, Any] = vars(engine._thread_context)
            if end_first:
                engine.on_session_end("child", [])
            engine._host_fallback_compressor = Native()
            engine._host_fallback_session_id = "child"
            engine._clear_thread_context_stateless("child")
            return state

    with engine_context(tmp_path, window=200_000, session_id="parent") as engine:
        parent = (engine.__dict__["_host_fallback_compressor"], engine.__dict__["_host_fallback_session_id"])
        state = HostChild().publish(engine)
        if not end_first:
            assert state["_host_fallback_compressor"] is not None
            engine.on_session_end("child", [])
        assert ended == ["child"]
        assert resets == [True]
        assert state["_host_fallback_compressor"] is None
        assert state["_host_fallback_session_id"] == ""
        assert engine._child_host_fallback_states == {}
        assert (engine.__dict__["_host_fallback_compressor"], engine.__dict__["_host_fallback_session_id"]) == parent
        engine.on_session_end("child", [])
        assert ended == ["child"]
        assert resets == [True]
