"""#697: forced recovery budgets every result and degrades rich stubs safely."""

import copy
import json
import logging

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine, count_messages_tokens
from hermes_lcm.externalize import is_externalized_placeholder, maybe_externalize_tool_output


SYSTEM = {"role": "system", "content": "system"}
USER = {"role": "user", "content": "run tools"}


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=10,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=100,
        ),
        hermes_home=str(tmp_path / "hermes"),
    )
    instance.on_session_start("session-a", conversation_id="conversation", context_length=200_000)
    yield instance
    instance.shutdown()


def _tail(call_ids=("call_a",)):
    return [
        copy.deepcopy(USER),
        {
            "role": "assistant",
            "content": "running tools",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                for call_id in call_ids
            ],
        },
        {"role": "tool", "tool_call_id": call_ids[0], "content": "oversized payload " * 1_000},
    ]


def _rich_stub(engine, tail):
    result = tail[-1]
    externalized = maybe_externalize_tool_output(
        result["content"],
        tool_call_id=result["tool_call_id"],
        session_id=engine.current_session_id,
        config=engine._config,
        hermes_home=engine._hermes_home,
        tool_name="read_file",
    )
    assert externalized is not None
    stub = engine._over_cap_tool_result_stub(result, "read_file")
    assert is_externalized_placeholder(stub["content"])
    return stub


def test_t1_plain_stub_keeps_call_when_rich_stub_exceeds_cap(engine):
    tail = _tail()
    original = copy.deepcopy(tail)
    rich = _rich_stub(engine, tail)
    plain = engine._missing_tool_result_stub("call_a")
    cap = count_messages_tokens([SYSTEM, tail[1], plain])
    assert count_messages_tokens([SYSTEM, tail[1], rich]) > cap

    out = engine._assemble_overflow_recovery_context(SYSTEM, tail, assembly_cap_override=cap)

    assert out == [SYSTEM, tail[1], plain]
    assert count_messages_tokens(out) <= cap
    assert tail == original


def test_t2_every_missing_result_is_budgeted_before_keeping_call(engine):
    # No payload file: the oversized result takes the plain-stub path.
    tail = _tail(("call_a", "call_b"))
    plain_a = engine._missing_tool_result_stub("call_a")
    plain_b = engine._missing_tool_result_stub("call_b")
    cap = count_messages_tokens([SYSTEM, tail[1], plain_a])
    assert count_messages_tokens([SYSTEM, tail[1], plain_a, plain_b]) > cap

    out = engine._assemble_overflow_recovery_context(SYSTEM, tail, assembly_cap_override=cap)
    engine._finalize_forced_overflow_result([SYSTEM, *tail], out, assembly_cap_override=cap)

    assert count_messages_tokens(out) <= cap
    assert engine._last_overflow_recovery_failed is False
    # Once both stubs fit, preserve the call and pair it with both results.
    both_cap = count_messages_tokens([SYSTEM, tail[1], plain_a, plain_b])
    assert engine._assemble_overflow_recovery_context(
        SYSTEM, tail, assembly_cap_override=both_cap
    ) == [SYSTEM, tail[1], plain_a, plain_b]


def test_t3_rich_stub_is_preserved_when_it_fits(engine):
    tail = _tail()
    rich = _rich_stub(engine, tail)
    cap = count_messages_tokens([SYSTEM, tail[1], rich])

    out = engine._assemble_overflow_recovery_context(SYSTEM, tail, assembly_cap_override=cap)

    assert out == [SYSTEM, tail[1], rich]
    assert count_messages_tokens(out) <= cap


def test_t4_lineage_write_failure_warns_once_and_rotation_continues(engine, monkeypatch, caplog):
    write = engine._store.write_metadata_json
    attempts = []

    def fail_lineage(keys, *args, **kwargs):
        if keys == [engine._rotation_predecessor_metadata_key("session-b")]:
            attempts.append(keys)
            raise OSError("private payload must not enter logs")
        return write(keys, *args, **kwargs)

    monkeypatch.setattr(engine._store, "write_metadata_json", fail_lineage)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
        engine.on_session_start(
            "session-b",
            conversation_id="conversation",
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="session-a",
        )

    assert engine.current_session_id == "session-b"
    assert len(attempts) == 1
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.levelno == logging.WARNING
    assert "session-a" in warning.getMessage() and "session-b" in warning.getMessage()
    assert "OSError" in warning.getMessage()
    assert "private payload" not in warning.getMessage()
    assert "\n" not in warning.getMessage()
    assert warning.exc_info is None


def test_t5_non_forced_assembly_matches_base_message_bytes(engine):
    tail = _tail(("call_a", "call_b"))
    cap = count_messages_tokens([SYSTEM, tail[1], engine._missing_tool_result_stub("call_a")])

    out = engine._assemble_context(
        SYSTEM, tail, assembly_cap_override=cap, include_lcm_note=False,
        stub_over_cap_tool_results=False,
    )

    # Literal base output, verified by the red-base run before changing engine.py.
    assert json.dumps(out, ensure_ascii=False).encode() == (
        b'[{"role": "system", "content": "system"}, {"role": "user", '
        b'"content": "[Current user objective preserved from compacted history]\\nrun tools"}]'
    )
