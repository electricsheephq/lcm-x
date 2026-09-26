"""#91: forced-overflow recovery must never return an empty transcript.

Construction mirrors
tests/test_lcm_engine.py::test_overflow_recovery_fallback_removes_orphan_tool_result.
The path needs a non-default assembly cap (LCM_MAX_ASSEMBLY_TOKENS or
LCM_RESERVE_TOKENS_FLOOR); the tests pass it as ``assembly_cap_override``.
"""
import logging

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

OBJECTIVE = (
    "[Current user objective preserved from compacted history]\n"
    "KEEP_OBJECTIVE: continue the active plan."
)
OVERSIZED = "oversized assistant/tool chatter " * 400


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(
        config=LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_issue_91.db"),
        )
    )
    instance._session_id = "issue-91"
    instance.compression_count = 1
    instance.context_length = 200_000
    try:
        yield instance
    finally:
        instance.shutdown()


def _recover(engine, tail):
    result = engine._assemble_overflow_recovery_context(
        None, tail, assembly_cap_override=120
    )
    return engine._finalize_forced_overflow_result(
        tail, result, assembly_cap_override=120
    )


@pytest.mark.parametrize("head_role", ["assistant", "user"])
def test_no_system_orphan_tool_tail_overflow_recovery_not_empty(engine, head_role):
    tail = [
        {
            "role": head_role,
            "content": OBJECTIVE
            if head_role == "assistant"
            else "KEEP_OBJECTIVE: continue the active plan.",
        },
        {"role": "assistant", "content": OVERSIZED},
        {"role": "tool", "tool_call_id": "orphan-call", "content": "latest tool status"},
    ]

    final = _recover(engine, tail)

    assert final, "forced-overflow recovery returned an empty transcript"
    assert any(m.get("role") != "tool" for m in final)
    assert not any(m.get("tool_call_id") == "orphan-call" for m in final)
    assert engine._last_compression_status == "overflow_recovery"
    assert engine._ingest_cursor == len(final)
    if head_role == "user":
        # Green on main before the guard; kept as a regression.
        assert any("KEEP_OBJECTIVE" in str(m.get("content")) for m in final)
    else:
        # Fallback keeps the LAST non-tool row of the raw tail, even oversized.
        assert final == [{"role": "assistant", "content": OVERSIZED}]


def test_all_tool_tail_overflow_recovery_returns_raw_tail(engine, caplog):
    tail = [
        {"role": "tool", "tool_call_id": "orphan-a", "content": "status a"},
        {"role": "tool", "tool_call_id": "orphan-b", "content": "status b"},
    ]

    with caplog.at_level(logging.WARNING):
        final = _recover(engine, tail)

    assert final == tail
    shape_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "no non-tool row" in r.getMessage()
    ]
    assert len(shape_warnings) == 1
