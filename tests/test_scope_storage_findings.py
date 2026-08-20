"""Storage-layer findings: an unreadable decision, a lying report, a mutating
rejection, and two table lists that could drift apart.

Each of these ends the same way -- an operator, or `resolve_startup_teams_state`,
reads a reassuring answer about a store that is not in the state it claims.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm import scope_storage
from hermes_lcm.access_policy import TEAMS_ENABLED_ATTR
from hermes_lcm.scope_storage import (
    ACCESS_SCOPE_COLUMN,
    SCOPE_BEARING_TABLES,
    ScopeBackfillIncompleteError,
    backfill_scopes,
    doctor_status_for,
    persist_teams_enabled,
    read_persisted_teams_enabled,
    read_persisted_teams_marker,
    resolve_startup_teams_state,
    verify_scope_storage,
)


_SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE messages(
    store_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    content      TEXT,
    access_scope TEXT
);
CREATE TABLE lcm_chunk_meta(
    chunk_id      TEXT,
    store_id      INTEGER,
    identity_hash TEXT,
    access_scope  TEXT
);
CREATE TABLE lcm_chunk_vectors(
    chunk_id      TEXT,
    identity_hash TEXT,
    vector        BLOB,
    access_scope  TEXT
);
"""


@pytest.fixture()
def store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "lcm.db")
    conn.executescript(_SCHEMA)
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _resolver(session_id: str) -> str | None:
    return f"owner:{session_id}"


# --- the Teams state marker ------------------------------------------------


@pytest.mark.parametrize("value", ["tru", "", "  ", "enabled?", "2", "maybe"])
def test_an_unrecognized_marker_is_not_authority_to_disable(
    store: sqlite3.Connection, value: str
) -> None:
    """`value in {"1","true","yes","on"}` made every corrupt or truncated
    marker read as an explicit `false`, and `resolve_startup_teams_state`
    returned `disabled` BEFORE checking for owner stamps -- selecting the
    permissive policy for a store that may be fully scoped."""
    store.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?)",
        (scope_storage.TEAMS_ENABLED_METADATA_KEY, value),
    )
    store.commit()

    assert read_persisted_teams_marker(store) == "malformed"
    enabled, reason = resolve_startup_teams_state(store)
    assert enabled is True
    assert reason == "malformed-marker"


def test_a_malformed_marker_makes_the_doctor_red(store: sqlite3.Connection) -> None:
    """An unresolved enable decision is what someone runs the doctor to find."""
    store.execute(
        "INSERT INTO metadata(key, value) VALUES(?, 'tru')",
        (scope_storage.TEAMS_ENABLED_METADATA_KEY,),
    )
    store.commit()

    result = verify_scope_storage(store, teams_enabled=False)

    assert result["status"] == "malformed-marker"
    assert doctor_status_for(result) == "fail"


@pytest.mark.parametrize(
    "value, expected", [("true", True), ("1", True), ("false", False), ("off", False)]
)
def test_a_well_formed_marker_still_reads_as_written(
    store: sqlite3.Connection, value: str, expected: bool
) -> None:
    """POSITIVE CONTROL. Reading everything as malformed also passes the above."""
    store.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?)",
        (scope_storage.TEAMS_ENABLED_METADATA_KEY, value),
    )
    store.commit()

    assert read_persisted_teams_enabled(store) is expected
    assert resolve_startup_teams_state(store)[0] is expected


def test_an_absent_marker_is_still_absent(store: sqlite3.Connection) -> None:
    assert read_persisted_teams_marker(store) == "absent"
    assert read_persisted_teams_enabled(store) is None


def test_a_round_tripped_decision_is_never_malformed(
    store: sqlite3.Connection,
) -> None:
    """Whatever `persist_teams_enabled` writes, the reader must recognise."""
    for decision in (True, False):
        persist_teams_enabled(store, decision)
        assert read_persisted_teams_marker(store) != "malformed"
        assert read_persisted_teams_enabled(store) is decision


# --- backfill validates before it mutates ----------------------------------


def test_a_rejected_backfill_leaves_the_schema_untouched(
    tmp_path,
) -> None:
    """`ensure_scope_columns` ran BEFORE the resolver check, so a call the
    function goes on to reject had already performed eleven ALTERs and written
    the `scope_v1` marker. The caller sees a ValueError and reasonably assumes
    nothing happened -- but a later `ensure_scope_columns` short-circuits on
    that marker and skips the verification sweep."""
    conn = sqlite3.connect(tmp_path / "fresh.db")
    try:
        conn.executescript(
            "CREATE TABLE messages(store_id INTEGER PRIMARY KEY, session_id TEXT);"
        )
        conn.commit()

        with pytest.raises(ValueError, match="owner_for_session is required"):
            backfill_scopes(conn)

        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        assert ACCESS_SCOPE_COLUMN not in columns
        marker = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table'"
            " AND name='lcm_migration_state'"
        ).fetchone()
        assert marker is None, "a rejected call wrote the migration marker"
    finally:
        conn.close()


def test_a_rejected_batch_size_also_leaves_the_schema_untouched(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "fresh.db")
    try:
        conn.executescript(
            "CREATE TABLE messages(store_id INTEGER PRIMARY KEY, session_id TEXT);"
        )
        conn.commit()

        with pytest.raises(ValueError, match="batch_size"):
            backfill_scopes(conn, _resolver, batch_size=0)

        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        assert ACCESS_SCOPE_COLUMN not in columns
    finally:
        conn.close()


# --- the report tells the truth --------------------------------------------


def test_an_orphaned_derived_row_fails_the_backfill(store: sqlite3.Connection) -> None:
    """No foreign key prevents an orphan chunk row, and `_backfill_joined_table`
    simply does not select one. So it returned normally, `failures` stayed
    empty, and the report declared the migration COMPLETE with NULL scopes on
    disk -- while `verify_scope_storage` calls the same store `fail`."""
    store.execute("INSERT INTO messages(session_id, content) VALUES('s1', 'x')")
    store.execute(
        "INSERT INTO lcm_chunk_meta(chunk_id, store_id, identity_hash)"
        " VALUES('c1', 9999, 'h1')"
    )
    store.commit()

    with pytest.raises(ScopeBackfillIncompleteError) as excinfo:
        backfill_scopes(store, _resolver)

    report = excinfo.value.report
    assert report["complete"] is False
    assert report["unstamped_remaining"] == {"lcm_chunk_meta": 1}
    assert "lcm_chunk_meta" in report["failures"]
    # And the two verdicts now agree.
    assert verify_scope_storage(store)["status"] == "fail"


def test_a_fully_stamped_store_still_reports_complete(
    store: sqlite3.Connection,
) -> None:
    """POSITIVE CONTROL: failing every backfill also passes the test above."""
    store.execute("INSERT INTO messages(session_id, content) VALUES('s1', 'x')")
    store.execute(
        "INSERT INTO lcm_chunk_meta(chunk_id, store_id, identity_hash)"
        " VALUES('c1', 1, 'h1')"
    )
    store.commit()

    report = backfill_scopes(store, _resolver)

    assert report["complete"] is True
    assert report["unstamped_remaining"] == {}
    assert verify_scope_storage(store)["status"] == "verified"


def test_the_counters_report_rows_written_not_rows_selected(
    store: sqlite3.Connection,
) -> None:
    """The UPDATE carries `AND access_scope IS NULL`, so a row stamped between
    the SELECT and the UPDATE is selected, counted and NOT written. The count
    is what an operator reads to confirm an enable, so an inflated one is a
    report they cannot trust."""
    for session in ("s1", "s2", "s3"):
        store.execute(
            "INSERT INTO messages(session_id, content) VALUES(?, 'x')", (session,)
        )
    store.commit()

    report = backfill_scopes(store, _resolver, batch_size=64)

    assert report["updated"]["messages"] == 3
    assert report["total_updated"] == 3


def test_a_duplicated_join_does_not_inflate_the_count(
    store: sqlite3.Connection,
) -> None:
    """`lcm_chunk_meta` has no unique key on (chunk_id, identity_hash), so the
    vectors join can return several source rows for ONE target rowid. Each
    duplicate was counted, and every UPDATE past the first no-opped against the
    `IS NULL` guard -- an operator reading `total_updated` sees more work than
    the database did."""
    store.execute("INSERT INTO messages(session_id, content) VALUES('s1', 'x')")
    store.commit()
    store_id = store.execute("SELECT store_id FROM messages").fetchone()[0]
    for _ in range(3):
        store.execute(
            "INSERT INTO lcm_chunk_meta(chunk_id, store_id, identity_hash)"
            " VALUES('c1', ?, 'h1')",
            (store_id,),
        )
    store.execute(
        "INSERT INTO lcm_chunk_vectors(chunk_id, identity_hash, vector)"
        " VALUES('c1', 'h1', X'00')"
    )
    store.commit()

    report = backfill_scopes(store, _resolver)

    assert report["updated"]["lcm_chunk_meta"] == 3
    assert report["updated"]["lcm_chunk_vectors"] == 1, (
        "three source rows for one target row were counted as three writes"
    )
    assert report["complete"] is True


# --- the two lists that must not drift -------------------------------------


def test_the_migration_and_the_verification_share_one_table_list() -> None:
    """A table in only one of them is either silently unverified -- unstamped
    rows never produce a `fail` -- or reported defective forever, because the
    migration never touches it. Neither is loud at the point of the edit."""
    assert scope_storage._ACCESS_SCOPE_TABLES is SCOPE_BEARING_TABLES


def test_the_teams_flag_attribute_is_the_one_the_policy_seam_reads() -> None:
    """Two independent definitions of the attribute name meant a rename left
    `mark_teams_enabled()` setting something `policy_for_engine` never reads --
    a permissive policy on scoped data, with nothing failing."""

    class _Carrier:
        pass

    carrier = _Carrier()
    scope_storage.mark_teams_enabled(carrier)

    assert getattr(carrier, TEAMS_ENABLED_ATTR) is True
    assert scope_storage._TEAMS_ENABLED_ATTRIBUTE == TEAMS_ENABLED_ATTR
    assert scope_storage.teams_enabled(carrier) is True
