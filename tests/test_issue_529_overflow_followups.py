"""#529: forced-overflow recovery follow-ups after #528.

Provider fact (see test_assemble_context_summary_role_is_user_after_system_anchor):
the host's Anthropic conversion hoists system rows into the top-level system
field, so the first non-system row of a returned transcript must be a user row,
and a transcript holding only system rows reaches the provider as messages=[].
"""
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import (
    _OVERFLOW_RECOVERY_OVERCAP_NOTE,
    _OVERFLOW_RECOVERY_PLACEHOLDER,
    LCMEngine,
    count_messages_tokens,
)

CAP = 120
SYSTEM = {"role": "system", "content": "You are an agent."}
OVERSIZED = "oversized assistant/tool chatter " * 400
ORPHAN = {"role": "tool", "tool_call_id": "orphan-call", "content": "latest tool status"}


def assert_provider_shape(result):
    non_system = [m for m in result if m.get("role") != "system"]
    assert len(non_system) >= 1, f"provider would receive messages=[]: {result!r}"
    assert non_system[0].get("role") != "tool", f"tool row leads the transcript: {result!r}"
    assert non_system[0].get("role") == "user", f"first non-system row is not user: {result!r}"


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(
        config=LCMConfig(
            fresh_tail_count=10,
            database_path=str(tmp_path / "lcm_issue_529.db"),
        )
    )
    instance._session_id = "issue-529"
    instance.compression_count = 1
    instance.context_length = 200_000
    try:
        yield instance
    finally:
        instance.shutdown()


def _recover(engine, tail, system_msg=None):
    result = engine._assemble_overflow_recovery_context(
        system_msg, tail, assembly_cap_override=CAP
    )
    return engine._finalize_forced_overflow_result(
        ([system_msg] if system_msg else []) + tail, result, assembly_cap_override=CAP
    )


def _note(newest):
    return {
        "role": "user",
        "content": _OVERFLOW_RECOVERY_OVERCAP_NOTE.format(
            tokens=count_messages_tokens([newest]), cap=CAP
        ),
    }


# -- item 2: an over-cap newest user turn is announced, never silently replaced


def test_newest_over_cap_user_turn_adds_note_after_older_fitting_row(engine):
    older = {"role": "user", "content": "OLDER_ASK: summarize the report."}
    newest = {"role": "user", "content": "NEWEST_ASK: rebuild the index " * 300}
    tail = [older, {"role": "assistant", "content": OVERSIZED}, newest, ORPHAN]

    final = _recover(engine, tail)

    assert final == [older, _note(newest)]
    assert final[1]["content"].startswith("[LCM overflow recovery] Your latest message (")
    assert_provider_shape(final)
    assert engine._last_overflow_recovery_failed is False


def test_note_beats_stale_intent_when_older_row_and_note_exceed_cap(engine):
    older = {"role": "user", "content": "older ask " * 40}  # fits alone, not with the note
    newest = {"role": "user", "content": "NEWEST_ASK: rebuild the index " * 300}
    tail = [older, {"role": "assistant", "content": OVERSIZED}, newest, ORPHAN]
    assert count_messages_tokens([older]) <= CAP < count_messages_tokens([older, _note(newest)])

    final = _recover(engine, tail)

    assert final == [_note(newest)]
    assert_provider_shape(final)
    assert engine._last_overflow_recovery_failed is False


def test_note_follows_system_and_retained_prefix(engine):
    older = {"role": "user", "content": "OLDER_ASK: summarize the report."}
    newest = {"role": "user", "content": "NEWEST_ASK: rebuild the index " * 300}
    tail = [older, {"role": "assistant", "content": OVERSIZED}, newest, ORPHAN]

    final = _recover(engine, tail, system_msg=SYSTEM)

    assert final == [SYSTEM, older, _note(newest)]
    assert_provider_shape(final)


def test_no_note_when_newest_user_row_is_kept(engine):
    newest = {"role": "user", "content": "NEWEST_ASK: short."}
    tail = [
        {"role": "user", "content": "OLDER_ASK: summarize."},
        {"role": "assistant", "content": OVERSIZED},
        newest,
        ORPHAN,
    ]

    final = _recover(engine, tail)

    assert any("NEWEST_ASK" in str(m.get("content")) for m in final)
    assert not any("Your latest message" in str(m.get("content")) for m in final)
    assert_provider_shape(final)


def test_no_note_when_skipped_newer_row_is_assistant(engine):
    older = {"role": "user", "content": "OLDER_ASK: summarize the report."}
    tail = [
        older,
        {"role": "assistant", "content": OVERSIZED},
        {"role": "assistant", "content": "newest over-cap reply " * 300},
        ORPHAN,
    ]

    final = _recover(engine, tail)

    assert any("OLDER_ASK" in str(m.get("content")) for m in final)
    assert not any("Your latest message" in str(m.get("content")) for m in final)
    assert_provider_shape(final)


# -- item 4: a system anchor is not a non-tool row the provider can answer


@pytest.mark.parametrize(
    "tail",
    [
        [ORPHAN],
        [{"role": "user", "content": "NEWEST_ASK: rebuild the index " * 300}, ORPHAN],
        [{"role": "assistant", "content": OVERSIZED}, ORPHAN],
    ],
    ids=["all-tool", "over-cap-user", "over-cap-assistant"],
)
def test_system_anchor_alone_is_never_returned(engine, tail):
    final = _recover(engine, tail, system_msg=SYSTEM)

    assert final[0] == SYSTEM
    assert [m for m in final if m.get("role") != "system"], final


# -- item 3: the two generated rows are scaffold across a restart; nothing else is


def _make_engine(tmp_path, session_id="issue-529-replay"):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine._session_id = session_id
    return engine


NOTE_TEXT = _OVERFLOW_RECOVERY_OVERCAP_NOTE.format(tokens=1234, cap=CAP)


@pytest.mark.parametrize("text", [_OVERFLOW_RECOVERY_PLACEHOLDER, NOTE_TEXT, "  \n" + NOTE_TEXT])
def test_generated_overflow_rows_are_verified_scaffold(engine, text):
    row = {"role": "user", "content": text}
    assert engine._is_replayed_context_scaffold_message(row) is True
    assert engine._is_verified_replay_scaffold_message(row) is True


@pytest.mark.parametrize(
    "row",
    [
        {"role": "user", "content": "I saw this: " + _OVERFLOW_RECOVERY_PLACEHOLDER},
        {"role": "user", "content": _OVERFLOW_RECOVERY_PLACEHOLDER + " Why?"},
        {"role": "user", "content": _OVERFLOW_RECOVERY_PLACEHOLDER.replace("dropped", "kept")},
        {"role": "user", "content": _OVERFLOW_RECOVERY_OVERCAP_NOTE.format(tokens="many", cap=CAP)},
        {"role": "user", "content": _OVERFLOW_RECOVERY_OVERCAP_NOTE.format(tokens=-5, cap=CAP)},
        {"role": "user", "content": NOTE_TEXT + "\nplease retry"},
        {"role": "user", "content": "[LCM overflow recovery] please recall my last message"},
        {"role": "assistant", "content": _OVERFLOW_RECOVERY_PLACEHOLDER},
        {"role": "tool", "tool_call_id": "t", "content": NOTE_TEXT},
    ],
    ids=["quoted", "appended", "edited", "word-slot", "negative-slot", "note-plus-text",
         "prefix-only", "assistant-role", "tool-role"],
)
def test_forged_or_edited_overflow_rows_stay_real_content(engine, row):
    assert engine._is_replayed_context_scaffold_message(row) is False
    assert engine._is_verified_replay_scaffold_message(row) is False


@pytest.mark.parametrize("generated", [_OVERFLOW_RECOVERY_PLACEHOLDER, NOTE_TEXT])
def test_cold_restart_does_not_store_generated_overflow_row(tmp_path, generated):
    first = _make_engine(tmp_path)
    try:
        first._ingest_messages(
            [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "First user request"},
                {"role": "assistant", "content": "First assistant reply"},
            ]
        )
    finally:
        first.shutdown()

    replay = _make_engine(tmp_path)
    try:
        replay._ingest_cursor_needs_reconcile = True
        replay._ingest_messages(
            [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "First user request"},
                {"role": "assistant", "content": "First assistant reply"},
                {"role": "user", "content": generated},
                {"role": "user", "content": "Brand new question after restart"},
            ]
        )
        contents = [r["content"] for r in replay._store.get_session_messages("issue-529-replay")]
    finally:
        replay.shutdown()

    assert contents.count(generated) == 0
    assert contents.count("Brand new question after restart") == 1
    assert contents.count("First user request") == 1


def test_cold_restart_stores_quoted_placeholder_as_real_content(tmp_path):
    quoted = "Earlier the agent said: " + _OVERFLOW_RECOVERY_PLACEHOLDER
    first = _make_engine(tmp_path)
    try:
        first._ingest_messages(
            [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "First user request"},
                {"role": "assistant", "content": "First assistant reply"},
            ]
        )
    finally:
        first.shutdown()

    replay = _make_engine(tmp_path)
    try:
        replay._ingest_cursor_needs_reconcile = True
        replay._ingest_messages(
            [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "First user request"},
                {"role": "assistant", "content": "First assistant reply"},
                {"role": "user", "content": quoted},
            ]
        )
        contents = [r["content"] for r in replay._store.get_session_messages("issue-529-replay")]
    finally:
        replay.shutdown()

    assert contents.count(quoted) == 1
