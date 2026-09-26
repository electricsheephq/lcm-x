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


@pytest.mark.parametrize("system_words", [25, 1], ids=["note-alone", "system-plus-note"])
def test_note_only_fallback_stays_within_cap(engine, system_words):
    # The older row fits with the prefix, the older row + note does not, and
    # neither may the prefix + note: drop the prefix rather than exceed the cap.
    # (A retained user row never reaches this loop: the fallback's early
    # return already keeps it as a non-system, non-tool row.)
    system_msg = {"role": "system", "content": "anchor text " * system_words}
    older = {"role": "user", "content": "older ask " * (1 if system_words > 1 else 30)}
    newest = {"role": "user", "content": "NEWEST_ASK: rebuild the index " * 300}
    tail = [older, {"role": "assistant", "content": OVERSIZED}, newest, ORPHAN]
    note = _note(newest)
    assert count_messages_tokens([system_msg, older]) <= CAP < count_messages_tokens([system_msg, older, note])
    expected = [note] if count_messages_tokens([system_msg, note]) > CAP else [system_msg, note]
    assert expected == ([note] if system_words > 1 else [system_msg, note])

    result = engine._assemble_overflow_recovery_context(system_msg, tail, assembly_cap_override=CAP)

    assert count_messages_tokens(result) <= CAP, result
    assert result == expected
    assert_provider_shape(result)


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


# -- item 3 was reverted (fix round 1): matching the generated text by content
# would drop a real user turn that byte-equals it (loss); provenance-based
# detection is #534. A replayed generated row is stored (duplicate direction).


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


@pytest.mark.parametrize("generated", [_OVERFLOW_RECOVERY_PLACEHOLDER, NOTE_TEXT])
def test_cold_restart_stores_replayed_generated_row_as_real_user_row(tmp_path, generated):
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

    assert contents.count(generated) == 1
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
