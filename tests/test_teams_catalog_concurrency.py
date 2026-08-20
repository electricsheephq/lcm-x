"""Check-then-write is not a check, once a second worker exists.

Two catalog accessors read a row, decided from what they read, and then wrote
in a SEPARATE statement. A control plane runs more than one reconciler, so the
window between the two is real:

* ``set_revisions`` refused a decrease against the value it had just read. A
  worker that read 5 while another committed 10 still wrote 6 -- and a counter
  that moves BACKWARDS un-revokes, because every context invalidated by the
  later bump matches the rolled-back value again.
* ``provision_principal`` refused a cross-tenant collision against the row it
  had just read. Two workers provisioning the same previously-absent id for
  DIFFERENT tenants both saw "absent", and the loser's upsert then updated the
  winner's row, bumped its own tenant's revision, and returned a ``Principal``
  claiming a tenant the row does not have.

Both are fixed in the WRITE, not by a longer read: the monotonicity and the
tenant are predicates on the conflict update itself, so a write that lost the
race changes nothing and is reported as the refusal it is.

The interleaving here is deterministic rather than threaded: the wrapper
commits the competing value on a SECOND connection at exactly the point the
accessor stops reading and starts writing, which is the window itself.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.teams import catalog


NOW = 1_700_000_000.0


class _RacedConnection:
    """Delegate to a real connection, running ``hook`` before one statement.

    The hook fires once, immediately BEFORE the first statement whose SQL
    contains ``marker`` -- i.e. after the accessor has finished reading and
    decided, and before its write lands.
    """

    def __init__(self, real: sqlite3.Connection, marker: str, hook) -> None:
        self._real = real
        self._marker = marker
        self._hook = hook
        self._fired = False

    def execute(self, sql, *args, **kwargs):
        if not self._fired and self._marker in sql:
            self._fired = True
            self._hook()
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture()
def db(tmp_path):
    """A FILE-backed store: two connections have to see one database."""

    path = tmp_path / "teams.db"
    connection = sqlite3.connect(path)
    catalog.ensure_teams_catalog(connection)
    connection.commit()
    try:
        yield path, connection
    finally:
        connection.close()


def test_a_lost_revision_race_does_not_roll_the_counters_back(db) -> None:
    """Worker A reads 5, worker B commits 10, worker A must not write 6."""

    path, conn = db
    catalog.set_revisions(conn, "t1", catalog.CatalogRevisions(5, 5, 5))

    other = sqlite3.connect(path)

    def commit_a_later_revision() -> None:
        catalog.set_revisions(other, "t1", catalog.CatalogRevisions(10, 10, 10))

    raced = _RacedConnection(conn, "INSERT INTO lcm_teams_revisions(", commit_a_later_revision)
    try:
        with pytest.raises(ValueError):
            catalog.set_revisions(raced, "t1", catalog.CatalogRevisions(6, 6, 6))
    finally:
        other.close()

    stored = catalog.read_revisions(conn, "t1")
    assert (
        stored.policy_revision,
        stored.membership_revision,
        stored.revocation_epoch,
    ) == (10, 10, 10), "the losing worker rolled a revoking counter backwards"


def test_a_lost_provisioning_race_does_not_steal_the_row(db) -> None:
    """Both workers see the id as absent; the loser must not own the winner."""

    path, conn = db
    other = sqlite3.connect(path)

    def provision_for_the_other_tenant() -> None:
        catalog.provision_principal(
            other, principal_id="shared-id", tenant_id="t2", now=NOW
        )

    raced = _RacedConnection(
        conn, "INSERT INTO lcm_teams_principals(", provision_for_the_other_tenant
    )
    try:
        with pytest.raises(LookupError):
            catalog.provision_principal(
                raced, principal_id="shared-id", tenant_id="t1", now=NOW + 1
            )
    finally:
        other.close()

    stored = catalog.read_principal(conn, "shared-id")
    assert stored is not None
    assert stored.tenant_id == "t2", "the losing worker took over the winner's row"
    # And it did not invalidate its OWN tenant's contexts for a write that
    # never landed: t1 has no principal, so it has no revision row either.
    assert catalog.read_revisions(conn, "t1").membership_revision == 0
