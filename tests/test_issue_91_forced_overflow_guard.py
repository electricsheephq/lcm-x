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
from tests.test_issue_529_overflow_followups import assert_provider_shape

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


def _recover(engine, tail, system_msg=None):
    result = engine._assemble_overflow_recovery_context(
        system_msg, tail, assembly_cap_override=120
    )
    return engine._finalize_forced_overflow_result(
        ([system_msg] if system_msg else []) + tail, result, assembly_cap_override=120
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
    # The head is the only non-tool row under the cap, so the objective survives.
    assert any("KEEP_OBJECTIVE" in str(m.get("content")) for m in final)
    assert not any(OVERSIZED in str(m.get("content")) for m in final)
    assert engine._last_compression_status == "overflow_recovery"
    assert engine._ingest_cursor == len(final)
    assert engine._last_overflow_recovery_failed is False


def test_newest_non_tool_row_that_fits_is_chosen_over_older_head(engine):
    # Round 3: a preserved objective that fits outranks a newer assistant
    # status that also fits (the name is kept from round 2 for traceability).
    objective = {"role": "assistant", "content": OBJECTIVE}
    tail = [
        objective,
        {"role": "assistant", "content": OVERSIZED},
        {"role": "assistant", "content": "NEWEST_FITS: short status reply."},
        {"role": "tool", "tool_call_id": "orphan-call", "content": "latest tool status"},
    ]

    final = _recover(engine, tail)

    assert final == [objective]
    assert engine._last_compression_status == "overflow_recovery"
    assert engine._ingest_cursor == 1
    assert engine._last_overflow_recovery_failed is False


def test_older_user_row_that_fits_wins_over_newer_assistant_that_fits(engine):
    user = {"role": "user", "content": "USER_OBJECTIVE: finish the migration."}
    tail = [
        user,
        {"role": "assistant", "content": OVERSIZED},
        {"role": "assistant", "content": "NEWEST_FITS: short status reply."},
        {"role": "tool", "tool_call_id": "orphan-call", "content": "latest tool status"},
    ]

    final = _recover(engine, tail)

    # Normal assembly already keeps a real user row (as a preserved-objective
    # scaffold) before the #91 fallback is reached; pin that it stays kept.
    assert any("USER_OBJECTIVE" in str(m.get("content")) for m in final)
    assert engine._last_overflow_recovery_failed is False
    assert_provider_shape(final)


def test_newer_oversized_user_request_wins_over_older_smaller_assistant(engine):
    request = {"role": "user", "content": "USER_REQUEST: rebuild the index " * 300}
    tail = [
        {"role": "assistant", "content": "older over-cap chatter " * 100},
        request,
        {"role": "tool", "tool_call_id": "orphan-call", "content": "latest tool status"},
    ]

    final = _recover(engine, tail)

    assert final == [request]
    assert_provider_shape(final)
    assert engine._last_compression_status == "overflow_recovery"
    # Over the cap, reported as such by the finalizer.
    assert engine._last_overflow_recovery_failed is True


def _tool_call(call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
    }


def test_tool_call_pair_that_fits_keeps_the_real_result(engine):
    call = _tool_call("call-real")
    result = {"role": "tool", "tool_call_id": "call-real", "content": "REAL_RESULT ok"}
    tail = [call, result]

    final = _recover(engine, tail)

    assert [m.get("role") for m in final] == ["assistant", "tool"]
    assert final[1]["tool_call_id"] == "call-real"
    assert final[1]["content"] == "REAL_RESULT ok"
    assert engine._last_overflow_recovery_failed is False


def test_tool_call_pair_over_cap_keeps_the_call_with_a_stub(engine):
    call = _tool_call("call-big")
    result = {"role": "tool", "tool_call_id": "call-big", "content": "REAL_RESULT " * 400}
    tail = [call, result]

    final = _recover(engine, tail)

    assert [m.get("role") for m in final] == ["assistant", "tool"]
    assert final[0]["tool_calls"][0]["id"] == "call-big"
    # The tool row answers the kept call (no orphan) and is a stub, not the result.
    assert final[1]["tool_call_id"] == "call-big"
    assert "REAL_RESULT" not in str(final[1]["content"])
    assert engine._last_overflow_recovery_failed is False


def test_smallest_non_tool_row_is_chosen_when_none_fits(engine):
    smaller = {"role": "assistant", "content": "older over-cap chatter " * 100}
    larger = {"role": "assistant", "content": "newest over-cap chatter " * 300}
    tail = [
        smaller,
        larger,
        {"role": "tool", "tool_call_id": "orphan-call", "content": "latest tool status"},
    ]

    final = _recover(engine, tail)

    assert final == [smaller]
    assert engine._last_compression_status == "overflow_recovery"
    assert engine._ingest_cursor == 1
    # Still over the cap, but the least-over-cap choice.
    assert engine._last_overflow_recovery_failed is True


@pytest.mark.parametrize("system_msg", [None, {"role": "system", "content": "You are an agent."}])
def test_all_tool_tail_overflow_recovery_emits_placeholder(engine, caplog, system_msg):
    tail = [
        {"role": "tool", "tool_call_id": "orphan-a", "content": "status a"},
        {"role": "tool", "tool_call_id": "orphan-b", "content": "status b"},
    ]

    with caplog.at_level(logging.WARNING):
        final = _recover(engine, tail, system_msg)

    # Never empty, never bare orphan tool rows: one non-tool recovery row.
    assert final, "overflow recovery returned an empty transcript"
    # User role: a system-only transcript is hoisted into the top-level
    # system field by the host's Anthropic conversion and arrives as
    # messages=[] (the same constraint that makes DAG summaries user-role);
    # #529: a system anchor alone must not escape as that empty transcript.
    assert [m.get("role") for m in final] == (["system"] if system_msg else []) + ["user"]
    assert all(m.get("role") != "tool" for m in final)
    assert "overflow recovery" in final[-1]["content"]
    assert_provider_shape(final)
    shape_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "no non-tool row" in r.getMessage()
    ]
    assert len(shape_warnings) == 1
