"""#2521: v0.29.2 bare spill-over markers remain one-to-one replay identities after an upgrade (I1, I6).

The upgrade tests start from ``tests/fixtures/v0292_spillover_store.sql``: a ``sqlite3 .dump`` of the store the
real v0.29.2 engine (261d41ea) wrote for one stamped tool turn whose result is a host spill-over marker. v0.29.2
cannot recover ``cache/spillover``, so it stored the marker bare and wrote no v0.30 replay ledger. The tests
replay the same host history plus one new turn through the public engine calls, a new engine per phase.

The ``candidate_written`` tests cover a store that the candidate itself wrote with spill-over recovery patched
off. That store also carries the candidate's own replay ledger, so it models candidate-to-candidate behaviour,
not the upgrade.
"""

import json
import logging
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from copy import deepcopy
from pathlib import Path

import pytest

from hermes_lcm import ingest_persisted
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_10_persisted_output_spillover import _messages, _write_output


FIXTURE = Path(__file__).parent / "fixtures" / "v0292_spillover_store.sql"
# The same turn written by v0.29.2 with a host timestamp on the user row only.
FIXTURE_USER_STAMPS = Path(__file__).parent / "fixtures" / "v0292_spillover_store_user_stamps.sql"
SESSION, CONVERSATION, CALL = "u1-session", "u1-conv", "call_spill"
T0 = 1762700000.0
# The spill file v0.29.2's marker points at: 5,427 characters, 7,227 bytes.
SPILL_TEXT = "SPILLOVER_RECOVERY_NEEDLE:\n" + ("é文abcdef\n" * 600)
ROW_COLUMNS = "store_id, session_id, conversation_id, role, content, tool_call_id, tool_calls, timestamp"


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    directory = tmp_path / "tmp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))


def _home(tmp_path):
    return tmp_path / "hermes"


def _engine(tmp_path):
    return LCMEngine(
        config=LCMConfig(
            database_path=str(_home(tmp_path) / "lcm.db"),
            large_output_externalization_path=str(_home(tmp_path) / "externalized"),
        ),
        hermes_home=str(_home(tmp_path)),
    )


def _run_phase(tmp_path, history, session=SESSION, conversation=CONVERSATION):
    engine = _engine(tmp_path)
    try:
        engine.on_session_start(session, platform="cli", context_length=128000, conversation_id=conversation)
        engine.ingest(deepcopy(history))
        return dict(engine._last_ingest_reconciliation)
    finally:
        engine.shutdown()


def _read_store(tmp_path):
    uri = (_home(tmp_path) / "lcm.db").as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT {ROW_COLUMNS} FROM messages ORDER BY store_id")]


def _tool_rows(rows):
    return Counter(row["tool_call_id"] for row in rows if row["role"] == "tool")


def _call_rows(rows):
    return Counter(call for row in rows if row["role"] == "assistant" and row["tool_calls"]
                   for call in (CALL, "call_twin") if f'"{call}"' in row["tool_calls"])


# --- the v0.29.2 store -------------------------------------------------------------------------------------------

def _v0292_store(tmp_path, fixture=FIXTURE):
    """Load the dump written by the real v0.29.2 engine; write the spill file its marker points at."""
    home = _home(tmp_path)
    spill = home / "cache" / "spillover" / f"{CALL}.txt"
    spill.parent.mkdir(parents=True)
    spill.write_bytes(SPILL_TEXT.encode("utf-8"))
    dump = fixture.read_text(encoding="utf-8").replace("@HERMES_HOME@", str(home))
    with closing(sqlite3.connect(home / "lcm.db")) as connection:
        connection.executescript(dump)
    # The dump declares the FTS tables through writable_schema, so they load on the next connection. Re-index them
    # over the substituted path, as v0.29.2 indexed the marker it stored.
    with closing(sqlite3.connect(home / "lcm.db")) as connection:
        for table in ("messages_fts", "messages_fts_stem"):
            connection.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
            connection.execute(f"INSERT INTO {table}({table}) VALUES('integrity-check')")
        connection.commit()
    rows = _read_store(tmp_path)
    marker = rows[2]["content"]
    assert [row["role"] for row in rows] == ["user", "assistant", "tool", "assistant"]
    assert marker.startswith("<persisted-output>") and f"Full output saved to: {spill}\n" in marker
    assert "[LCM persisted-output" not in marker  # Stored bare: no identity or generation metadata.
    return rows, marker, spill


def _host_history(marker, extra=(), stamps="all"):
    """The host transcript the v0.29.2 store was written from, plus any later rows."""
    def row(i, role, content, **fields):
        if stamps == "all" or role == "user":
            fields["timestamp"] = T0 + i
        return {"role": role, "content": content, **fields}
    first = [
        row(1, "user", "Please dump the big report."),
        row(2, "assistant", "", tool_calls=[{"id": CALL, "type": "function",
                                              "function": {"name": "dump", "arguments": "{}"}}]),
        row(3, "tool", marker, tool_call_id=CALL),
        row(4, "assistant", "The report is saved; the needle is at the top."),
    ]
    return first + [row(5 + i, *message[:2], **(message[2] if len(message) > 2 else {}))
                    for i, message in enumerate(extra)]


NEW_TURN = (("user", "Thanks. What was the first line?"), ("assistant", "SPILLOVER_RECOVERY_NEEDLE:"))


def _replay(tmp_path, history, original, caplog):
    with caplog.at_level(logging.INFO):
        reconciliation = _run_phase(tmp_path, history)
    rows = _read_store(tmp_path)
    assert rows[:len(original)] == original  # No old row changed or disappeared.
    assert "skipped stale no-overlap snapshot" not in caplog.text
    assert reconciliation.get("reason") != "skipped stale no-overlap snapshot"
    contents = {row["content"] for row in rows}
    assert all(message["content"] in contents for message in history)  # Raw host content survives.
    return rows, reconciliation


@pytest.mark.parametrize("spill_file", ["present", "gone"])
def test_v0292_store_upgrade_replay_adds_only_new_turn(tmp_path, caplog, spill_file):
    original, marker, spill = _v0292_store(tmp_path)
    if spill_file == "gone":
        moved = tmp_path / "moved-out-of-home" / spill.name
        moved.parent.mkdir()
        spill.rename(moved)
    history = _host_history(marker, NEW_TURN)

    rows, reconciliation = _replay(tmp_path, history, original, caplog)

    counts = f"stored rows={len(rows)}, tool rows={dict(_tool_rows(rows))}, reconciliation={reconciliation}"
    assert [row["content"] for row in rows[len(original):]] == [content for _, content in NEW_TURN], counts
    assert len(rows) == 6, counts
    assert _tool_rows(rows) == {CALL: 1}, counts
    assert _call_rows(rows) == {CALL: 1}, counts


def test_v0292_users_only_stamped_store_upgrade_replay_stores_each_message_once(tmp_path, caplog):
    original, marker, _ = _v0292_store(tmp_path, FIXTURE_USER_STAMPS)
    history = _host_history(marker, NEW_TURN, stamps="user")

    rows, _ = _replay(tmp_path, history, original, caplog)

    # One stored row per host message, in order. v0.29.2's own replay of this history stores 9 rows: its 3 extra
    # rows repeat messages already stored, so none of them is a new message.
    counts = f"stored rows={len(rows)}, tool rows={dict(_tool_rows(rows))}"
    assert [(row["role"], row["content"]) for row in rows] == [(m["role"], m["content"]) for m in history], counts
    assert _tool_rows(rows) == {CALL: 1}, counts
    assert _call_rows(rows) == {CALL: 1}, counts


# --- #2525 review follow-ups: shapes that miss guard (c) and keep the old duplicate path ----------------------------

def _parts_history(marker):
    """The same host transcript, with the tool result sent as content parts instead of a string (the shape Hermes
    gives a multimodal tool result whose oversized text part it spilled to a marker)."""
    history = _host_history(marker, NEW_TURN)
    history[2]["content"] = [{"type": "text", "text": marker}]
    return history


def _rebound_session_replay(tmp_path):
    """The v0.29.2 history replayed under a new session id in the same conversation."""
    original, marker, _ = _v0292_store(tmp_path)
    history = _host_history(marker, NEW_TURN)
    _run_phase(tmp_path, history, session="u1-session-rebound")
    return original, history, _read_store(tmp_path)


def test_v0292_store_parts_shaped_tool_result_loses_nothing(tmp_path):
    """Guard (c) compares the raw incoming content with the stored marker. A parts-shaped tool result misses it and
    takes the old duplicate path: no row is lost, the call row's copy is linked, and the tool row's copy is stored
    again, unlinked, with the marker text (a surplus row, not a loss)."""
    original, marker, _ = _v0292_store(tmp_path)

    _run_phase(tmp_path, _parts_history(marker))

    rows = _read_store(tmp_path)
    counts = f"stored rows={len(rows)}, tool rows={dict(_tool_rows(rows))}"
    assert rows[:len(original)] == original, counts
    assert [row["content"] for row in rows[-2:]] == [content for _, content in NEW_TURN], counts
    assert _tool_rows(rows) == {CALL: 2} and _call_rows(rows) == {CALL: 2}, counts
    tool_copy = [row for row in rows[len(original):] if row["role"] == "tool"]
    # The copy stores the parts as JSON; its text part is the marker.
    assert len(tool_copy) == 1 and json.loads(tool_copy[0]["content"]) == [{"type": "text", "text": marker}], counts
    with closing(sqlite3.connect(_home(tmp_path) / "lcm.db")) as connection:
        links = connection.execute("SELECT store_id, kind FROM message_relations ORDER BY store_id").fetchall()
    call_copy = [row["store_id"] for row in rows[len(original):] if row["role"] == "assistant" and row["tool_calls"]]
    assert links == [(call_copy[0], "possible_replay_of")], links


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "#2525 review follow-up 1: guard (c) compares raw content, so a parts-shaped tool result re-stores the call "
    "pair. Hermes sends a multimodal tool result as parts, so the comparison is a v0.30.1 design question."))
def test_v0292_store_parts_shaped_tool_result_stores_each_message_once(tmp_path):
    original, marker, _ = _v0292_store(tmp_path)

    _run_phase(tmp_path, _parts_history(marker))

    rows = _read_store(tmp_path)
    assert len(rows) == len(original) + len(NEW_TURN), f"stored rows={len(rows)}"


def test_v0292_store_rebound_session_replay_loses_nothing(tmp_path):
    """The stored-row lookup is session-scoped, like the #773 matcher. A rebound session id takes the old duplicate
    path: the replayed history is stored once more under the new session id (the #1674 class), and nothing is
    lost."""
    original, history, rows = _rebound_session_replay(tmp_path)

    rebound = [row for row in rows if row["session_id"] == "u1-session-rebound"]
    assert rows[:len(original)] == original
    assert [(row["role"], row["content"]) for row in rebound][-2:] == list(NEW_TURN)
    assert len(rebound) == len(history)
    assert _tool_rows(rows) == {CALL: 2} and _call_rows(rows) == {CALL: 2}


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "#2525 review follow-up 2: a rebound session id re-stores the v0.29.2 history (#1674 class; the conversation "
    "alias work in #2461 is the candidate fix)."))
def test_v0292_store_rebound_session_replay_stores_each_message_once(tmp_path):
    original, _history, rows = _rebound_session_replay(tmp_path)

    assert len(rows) == len(original) + len(NEW_TURN), f"stored rows={len(rows)}"


def test_v0292_store_second_replay_after_upgrade_adds_nothing(tmp_path, caplog):
    original, marker, _ = _v0292_store(tmp_path)
    history = _host_history(marker, NEW_TURN)
    upgraded, _ = _replay(tmp_path, history, original, caplog)

    rows, _ = _replay(tmp_path, history, original, caplog)

    assert rows == upgraded, f"stored rows={len(rows)}, tool rows={dict(_tool_rows(rows))}"


def test_v0292_store_new_call_with_identical_marker_is_stored(tmp_path, caplog):
    original, marker, _ = _v0292_store(tmp_path)
    # Same tool, byte-identical marker text (same spill file), a NEW tool_call_id: a new result, not a replay.
    twin_call = [{"id": "call_twin", "type": "function", "function": {"name": "dump", "arguments": "{}"}}]
    history = _host_history(marker, (
        ("user", "Dump it again."),
        ("assistant", "", {"tool_calls": twin_call}),
        ("tool", marker, {"tool_call_id": "call_twin"}),
        ("assistant", "Saved again."),
    ))

    rows, _ = _replay(tmp_path, history, original, caplog)

    counts = f"stored rows={len(rows)}, tool rows={dict(_tool_rows(rows))}"
    assert len(rows) == 8, counts
    assert _tool_rows(rows) == {CALL: 1, "call_twin": 1}, counts
    assert _call_rows(rows) == {CALL: 1, "call_twin": 1}, counts


# --- candidate-written bare stores (candidate-to-candidate behaviour) --------------------------------------------

def _candidate_written_bare_store(tmp_path, monkeypatch, history):
    # The candidate with spill-over recovery patched off; it also writes its own replay ledger.
    with monkeypatch.context() as patch:
        patch.setattr(ingest_persisted, "_safe_hermes_spillover_file", lambda *_: None)
        _run_phase(tmp_path, history, "spillover-upgrade", "spillover-conversation")
    rows = _read_store(tmp_path)
    assert [row["content"] for row in rows] == [message["content"] for message in history]
    assert all("[LCM persisted-output" not in (row["content"] or "") for row in rows)
    return rows


def _candidate_replay(tmp_path, history, original, caplog):
    with caplog.at_level(logging.INFO):
        reconciliation = _run_phase(tmp_path, history, "spillover-upgrade", "spillover-conversation")
    rows = _read_store(tmp_path)
    assert rows[:len(original)] == original
    assert "skipped stale no-overlap snapshot" not in caplog.text
    contents = {row["content"] for row in rows}
    assert all(message["content"] in contents for message in history)
    return rows, reconciliation


def test_candidate_written_unstamped_reply_replay_adds_only_new_turn(tmp_path, monkeypatch, caplog):
    path = _home(tmp_path) / "cache" / "spillover" / "call_spill.txt"
    marker = _write_output(path)
    history = [
        {"role": "user", "content": "Read the synthetic output.", "timestamp": 100.0},
        *_messages(marker),
        {"role": "assistant", "content": "Finished reading the spillover."},  # No timestamp.
    ]
    original = _candidate_written_bare_store(tmp_path, monkeypatch, history)
    new_turn = [
        {"role": "user", "content": "The next request.", "timestamp": 105.0},
        {"role": "assistant", "content": "The next reply.", "timestamp": 106.0},
    ]

    rows, reconciliation = _candidate_replay(tmp_path, [*history, *new_turn], original, caplog)

    assert [row["content"] for row in rows[len(original):]] == [row["content"] for row in new_turn], (
        f"stored rows={len(rows)}, tool rows={dict(_tool_rows(rows))}, reconciliation={reconciliation}"
    )
    assert _tool_rows(rows) == {"call_spill": 1}
    assert reconciliation.get("reason") != "persisted ambiguous delta"


def test_candidate_written_identical_markers_keep_distinct_call_ids(tmp_path, monkeypatch, caplog):
    marker = _write_output(_home(tmp_path) / "cache" / "spillover" / "shared.txt")
    history = [*_messages(marker, "call_a"), *_messages(marker, "call_b")]
    original = _candidate_written_bare_store(tmp_path, monkeypatch, history)

    rows, _ = _candidate_replay(tmp_path, history, original, caplog)
    assert rows == original
    rows, _ = _candidate_replay(tmp_path, [*history, *_messages(marker, "call_c")], original, caplog)
    assert len(rows) == len(original) + 2
    assert _tool_rows(rows) == {"call_a": 1, "call_b": 1, "call_c": 1}


@pytest.mark.parametrize("stored_copies", [1, 2])
def test_candidate_written_repeated_call_id_consumes_each_stored_copy_once(
    tmp_path, monkeypatch, caplog, stored_copies,
):
    marker = _write_output(_home(tmp_path) / "cache" / "spillover" / "shared.txt")
    pair = _messages(marker)
    original = _candidate_written_bare_store(tmp_path, monkeypatch, pair * stored_copies)

    rows, _ = _candidate_replay(tmp_path, pair * 2, original, caplog)

    assert len(rows) == 4
    assert sum(row["role"] == "tool" for row in rows) == 2
