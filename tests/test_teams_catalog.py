"""The Teams catalog: created only on explicit enable and fail-closed to old code.

Two properties carry this phase. A store that never enables Teams must end up
with no Teams tables at all -- that is what keeps a single-user install
untouched by any of this. A newer store that does carry Teams tables must remain
unmodified when an older build cannot verify that family exactly.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.teams import catalog


@pytest.fixture()
def store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "lcm.db")
    db_bootstrap.configure_connection(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_a_store_that_never_enables_teams_has_no_catalog(
    store: sqlite3.Connection,
) -> None:
    assert catalog.teams_catalog_exists(store) is False
    tables = {
        row[0]
        for row in store.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert not any(name.startswith("lcm_teams") for name in tables)


def test_checking_for_the_catalog_creates_nothing(store: sqlite3.Connection) -> None:
    """Runs on inspection paths; must not materialise what it reports on."""
    before = {
        row[0]
        for row in store.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    catalog.teams_catalog_exists(store)
    catalog.verify_teams_catalog(store)
    after = {
        row[0]
        for row in store.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert before == after


def test_ensure_creates_the_whole_family(store: sqlite3.Connection) -> None:
    created = catalog.ensure_teams_catalog(store)

    assert set(created) == set(catalog.TEAMS_TABLES)
    assert catalog.teams_catalog_exists(store) is True
    assert catalog.verify_teams_catalog(store) == []


def test_ensure_is_idempotent(store: sqlite3.Connection) -> None:
    catalog.ensure_teams_catalog(store)
    catalog.ensure_teams_catalog(store)
    assert catalog.verify_teams_catalog(store) == []


def test_a_missing_table_is_reported_as_a_defect(store: sqlite3.Connection) -> None:
    catalog.ensure_teams_catalog(store)
    store.execute("DROP TABLE lcm_teams_memberships")
    store.commit()

    assert catalog.verify_teams_catalog(store) == [
        "missing table:lcm_teams_memberships"
    ]


def test_every_table_carries_the_teams_family_prefix() -> None:
    """The catalog stays inside its one reserved table-name family.

    The prefix invariant is a property of this module and does not imply that
    older bootstrap code may repair or downgrade an unknown catalog shape.
    """
    assert all(name.startswith("lcm_teams") for name in catalog.TEAMS_TABLES)


def test_the_family_prefix_is_not_in_the_interim_repair_allowlist() -> None:
    """Old code must not downgrade a Teams schema it cannot verify exactly."""
    assert "lcm_teams" not in db_bootstrap._KNOWN_FEATURE_TABLE_PREFIXES


def test_an_unknown_tenant_reads_as_zero_rather_than_raising(
    store: sqlite3.Connection,
) -> None:
    """A context belonging to no tenant fails on its principal, more precisely."""
    catalog.ensure_teams_catalog(store)

    revisions = catalog.read_revisions(store, "tenant-nobody")

    assert revisions == catalog.CatalogRevisions(0, 0, 0)


def test_bumping_a_revision_makes_a_previously_issued_context_stale(
    store: sqlite3.Connection,
) -> None:
    """Revocation IS a revision bump; this is the operation that expires."""
    catalog.ensure_teams_catalog(store)
    before = catalog.read_revisions(store, "tenant-1")

    catalog.bump_revision(store, "tenant-1", "revocation_epoch")
    after = catalog.read_revisions(store, "tenant-1")

    assert before.revocation_epoch == 0
    assert after.revocation_epoch == 1
    # The other counters are untouched -- revoking is not a policy change.
    assert after.policy_revision == before.policy_revision
    assert after.membership_revision == before.membership_revision


def test_each_revision_counter_moves_independently(store: sqlite3.Connection) -> None:
    catalog.ensure_teams_catalog(store)
    for field in ("policy_revision", "membership_revision", "revocation_epoch"):
        assert catalog.bump_revision(store, "tenant-1", field) == 1

    assert catalog.read_revisions(store, "tenant-1") == catalog.CatalogRevisions(
        1, 1, 1
    )


def test_an_unknown_revision_field_is_refused(store: sqlite3.Connection) -> None:
    """The field name reaches an f-string, so it is checked rather than trusted."""
    catalog.ensure_teams_catalog(store)

    with pytest.raises(ValueError, match="unknown revision field"):
        catalog.bump_revision(store, "tenant-1", "revocation_epoch = 0 --")


def test_unknown_teams_table_refuses_interim_stamp_remediation(tmp_path) -> None:
    from hermes_lcm.dag import SummaryDAG
    from hermes_lcm.store import MessageStore

    db_path = tmp_path / "future-teams.db"
    message_store = MessageStore(db_path)
    dag = SummaryDAG(db_path)
    dag.close()
    message_store.close()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE lcm_teams_requests_future (request_id TEXT)")
        future_version = db_bootstrap.SCHEMA_VERSION + 1
        db_bootstrap.set_schema_version(conn, future_version)
        conn.commit()

        assert (
            db_bootstrap.classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
        )
        result = db_bootstrap.remediate_interim_schema_stamp(conn, apply=True)
        assert result["status"] == "refused"
        assert result["applied"] is False
        assert db_bootstrap.read_existing_schema_version(conn) == future_version
        assert conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='lcm_teams_requests_future'"
        ).fetchone() is not None
    finally:
        conn.close()
