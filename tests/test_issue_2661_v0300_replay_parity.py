"""#2661: a replay over a v0.29.2 bare spill-over marker stores no fewer rows than v0.30.0 (I1, I6).

#2584's legacy bare-marker prefix binding (main, not in v0.30.0) skipped a re-sent user occurrence when a later turn
reused the call id, and skipped every row of another conversation of the same session. The counts below are
v0.30.0's, measured on the same shapes.
"""

import pytest

from tests.test_issue_2521_spillover_upgrade_replay import (  # noqa: F401
    CALL, T0, _host_history, _read_store, _run_phase, _v0292_store, isolated_tempdir,
)


def _again(content):
    return (("user", "Dump again."), ("assistant", "", {"tool_calls": [{"id": CALL, "type": "function",
             "function": {"name": "dump", "arguments": "{}"}}]}), ("tool", content, {"tool_call_id": CALL}),
            ("assistant", "Saved again."))


def _counts(rows, conversation=None):
    rows = [r for r in rows if conversation is None or r["conversation_id"] == conversation]
    return sum(r["role"] == "user" for r in rows), len(rows)


@pytest.mark.parametrize("later, expected", [(4, (3, 9)), (0, (3, 11))])
def test_reused_call_id_keeps_a_re_sent_user_occurrence(tmp_path, later, expected):
    """Shape 1: the history plus a turn reusing the call id (a plain-text result), replayed, then replayed again with
    one user row re-sent with a later stamp. v0.30.0 keeps the re-sent message as its own occurrence."""
    original, marker, _ = _v0292_store(tmp_path)
    history = _host_history(marker, _again(marker))
    _run_phase(tmp_path, history)
    history[later]["timestamp"] = history[later]["timestamp"] + 50.5
    _run_phase(tmp_path, history)
    rows = _read_store(tmp_path)
    assert rows[:len(original)] == original
    assert _counts(rows) == expected, f"users, rows = {_counts(rows)}"


@pytest.mark.parametrize("which", [0, 3])
def test_another_conversation_of_the_session_keeps_its_rows(tmp_path, which):
    """Shape 2: the stored history replayed under another conversation id of the same session (one row re-stamped).
    v0.30.0 stores that conversation's rows; a second identical replay adds nothing."""
    original, marker, _ = _v0292_store(tmp_path)
    history = _host_history(marker)
    history[which]["timestamp"] = T0 + 100.5
    for _ in range(2):
        _run_phase(tmp_path, history, conversation="other-conv")
        rows = _read_store(tmp_path)
        assert rows[:len(original)] == original
        assert _counts(rows) == (2, 8), f"users, rows = {_counts(rows)}"
        assert _counts(rows, "other-conv") == (1, 4), f"other-conv users, rows = {_counts(rows, 'other-conv')}"


@pytest.mark.parametrize("mode", ["off", "shadow"])
@pytest.mark.parametrize("change,expected", [("two-users", (4, 12)), ("user-and-assistant", (3, 12)),
                                             ("uid-later", (3, 11))])
def test_later_stamps_or_an_unproven_uid_keep_user_occurrences(tmp_path, monkeypatch, mode, change, expected):
    """#2661 review: with a reused call id, two changed stamps, a changed user and assistant stamp, or a fresh UID on
    the later user row keep every user occurrence, as v0.30.0 does (measured: the same counts)."""
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", mode)
    _, marker, _ = _v0292_store(tmp_path)
    history = _host_history(marker, _again(marker))
    _run_phase(tmp_path, history)
    assert len(_read_store(tmp_path)) == 8
    history[0]["timestamp"] += 50.5
    if change == "two-users":
        history[4]["timestamp"] += 50.5
    elif change == "user-and-assistant":
        history[3]["timestamp"] += 50.5
    else:
        history[0]["message_uid"] = "fresh-user"
    _run_phase(tmp_path, history)
    assert _counts(_read_store(tmp_path)) == expected
