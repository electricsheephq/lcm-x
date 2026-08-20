"""The tenant is a PREDICATE, and a revocation is one transaction.

Two classes of finding on the catalog accessors, both of the same shape: a
value arrived as an argument, drove a side effect, and was then dropped before
the statement that actually read or wrote the row.

* ``tenant_id`` chose which revision counter to bump and was absent from every
  WHERE clause, so a membership could be granted across two tenants, a foreign
  principal suspended, and the wrong tenant's contexts invalidated while the
  cross-tenant grant stayed live.
* the mutation and the counter bump that invalidates the contexts depending on
  it were separate transactions, so a failed bump left the mutation committed
  with the old counters still current -- every already-issued context still
  validating against a membership that is gone.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.teams import catalog


NOW = 1_700_000_000.0


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    catalog.ensure_teams_catalog(connection)
    try:
        yield connection
    finally:
        connection.close()


def _two_tenants(conn: sqlite3.Connection) -> None:
    catalog.provision_principal(conn, principal_id="alice", tenant_id="t1", now=NOW)
    catalog.provision_principal(conn, principal_id="mallory", tenant_id="t2", now=NOW)
    catalog.create_collection(
        conn, collection_id="t1-own", tenant_id="t1", kind="own", now=NOW
    )
    catalog.create_collection(
        conn, collection_id="t2-secret", tenant_id="t2", kind="shared", now=NOW
    )


# --- tenant ownership on grants -------------------------------------------


def test_a_grant_across_two_tenants_is_refused(conn: sqlite3.Connection) -> None:
    """The membership row names only a principal and a collection, so nothing
    in the schema stopped this and `authorized_collections` returned it."""
    _two_tenants(conn)

    with pytest.raises(LookupError):
        catalog.grant_membership(
            conn,
            principal_id="alice",
            collection_id="t2-secret",
            grants=["read"],
            tenant_id="t1",
            now=NOW,
        )

    assert catalog.authorized_collections(conn, "alice") == ()


def test_a_grant_naming_a_foreign_principal_is_refused(
    conn: sqlite3.Connection,
) -> None:
    _two_tenants(conn)

    with pytest.raises(LookupError):
        catalog.grant_membership(
            conn,
            principal_id="mallory",
            collection_id="t1-own",
            grants=["read"],
            tenant_id="t1",
            now=NOW,
        )

    assert catalog.authorized_collections(conn, "mallory") == ()


def test_a_refused_grant_does_not_invalidate_the_tenants_contexts(
    conn: sqlite3.Connection,
) -> None:
    """The second half of the finding: the bump ran even when the write was
    nonsense, so a bad reconcile locked out principals nothing had changed."""
    _two_tenants(conn)
    before = catalog.read_revisions(conn, "t1")

    with pytest.raises(LookupError):
        catalog.grant_membership(
            conn,
            principal_id="alice",
            collection_id="t2-secret",
            grants=["read"],
            tenant_id="t1",
            now=NOW,
        )

    assert catalog.read_revisions(conn, "t1") == before


def test_a_grant_within_one_tenant_still_works(conn: sqlite3.Connection) -> None:
    """POSITIVE CONTROL. Refusing every grant also passes the three above."""
    _two_tenants(conn)

    catalog.grant_membership(
        conn,
        principal_id="alice",
        collection_id="t1-own",
        grants=["read"],
        tenant_id="t1",
        now=NOW,
    )

    assert catalog.authorized_collections(conn, "alice") == ("t1-own",)


# --- idempotent grants -----------------------------------------------------


def test_an_unchanged_grant_does_not_advance_the_membership_revision(
    conn: sqlite3.Connection,
) -> None:
    """A desired-state reconciler replays its grants. The unconditional bump
    made every replay invalidate every context issued for the tenant."""
    _two_tenants(conn)
    catalog.grant_membership(
        conn, principal_id="alice", collection_id="t1-own",
        grants=["read", "write"], tenant_id="t1", now=NOW,
    )
    settled = catalog.read_revisions(conn, "t1").membership_revision

    for _ in range(3):
        catalog.grant_membership(
            conn, principal_id="alice", collection_id="t1-own",
            # Same set, different order and spacing: normalization decides.
            grants=[" write ", "read"], tenant_id="t1", now=NOW + 5,
        )

    assert catalog.read_revisions(conn, "t1").membership_revision == settled


def test_a_changed_grant_does_advance_it(conn: sqlite3.Connection) -> None:
    """POSITIVE CONTROL: a real change must still invalidate stale contexts."""
    _two_tenants(conn)
    catalog.grant_membership(
        conn, principal_id="alice", collection_id="t1-own",
        grants=["read"], tenant_id="t1", now=NOW,
    )
    settled = catalog.read_revisions(conn, "t1").membership_revision

    catalog.grant_membership(
        conn, principal_id="alice", collection_id="t1-own",
        grants=["read", "write"], tenant_id="t1", now=NOW + 5,
    )

    assert catalog.read_revisions(conn, "t1").membership_revision > settled


# --- suspension and provisioning ------------------------------------------


def test_suspending_a_foreign_principal_is_refused(conn: sqlite3.Connection) -> None:
    _two_tenants(conn)

    with pytest.raises(LookupError):
        catalog.suspend_principal(
            conn, principal_id="mallory", tenant_id="t1", now=NOW + 1
        )

    assert catalog.read_principal(conn, "mallory").status == "active"


def test_a_refused_suspension_does_not_bump_the_epoch(
    conn: sqlite3.Connection,
) -> None:
    """It used to invalidate the CALLER's contexts for a write that never
    landed -- one tenant able to revoke its own fleet by naming a stranger."""
    _two_tenants(conn)
    before = catalog.read_revisions(conn, "t1").revocation_epoch

    with pytest.raises(LookupError):
        catalog.suspend_principal(
            conn, principal_id="mallory", tenant_id="t1", now=NOW + 1
        )

    assert catalog.read_revisions(conn, "t1").revocation_epoch == before


def test_re_provisioning_another_tenants_principal_is_refused(
    conn: sqlite3.Connection,
) -> None:
    """`principal_id` is the primary key, so the upsert left the stored tenant
    alone, reactivated the row, and reported success in the caller's name."""
    _two_tenants(conn)
    catalog.suspend_principal(
        conn, principal_id="mallory", tenant_id="t2", now=NOW + 1
    )

    with pytest.raises(LookupError):
        catalog.provision_principal(
            conn, principal_id="mallory", tenant_id="t1", now=NOW + 2
        )

    assert catalog.read_principal(conn, "mallory").status == "suspended"
    assert catalog.read_principal(conn, "mallory").tenant_id == "t2"


# --- atomic revocation -----------------------------------------------------


def test_revoking_a_foreign_tenants_membership_removes_nothing(
    conn: sqlite3.Connection,
) -> None:
    _two_tenants(conn)
    catalog.grant_membership(
        conn, principal_id="mallory", collection_id="t2-secret",
        grants=["read"], tenant_id="t2", now=NOW,
    )

    assert (
        catalog.revoke_membership(
            conn, principal_id="mallory", collection_id="t2-secret", tenant_id="t1"
        )
        is False
    )
    assert catalog.authorized_collections(conn, "mallory") == ("t2-secret",)


def test_a_failed_epoch_bump_leaves_the_membership_in_place(
    conn: sqlite3.Connection,
) -> None:
    """The atomicity property, made observable.

    The deletion used to commit BEFORE either bump, so a bump that failed --
    or any observer between the statements -- saw a membership that was gone
    while both counters were still current, and every context issued before it
    kept validating. One transaction means no such observer exists.
    """
    _two_tenants(conn)
    catalog.grant_membership(
        conn, principal_id="alice", collection_id="t1-own",
        grants=["read"], tenant_id="t1", now=NOW,
    )
    conn.execute("DROP TABLE lcm_teams_revisions")
    conn.commit()

    with pytest.raises(sqlite3.Error):
        catalog.revoke_membership(
            conn, principal_id="alice", collection_id="t1-own", tenant_id="t1"
        )

    conn.rollback()
    assert catalog.read_memberships(conn, "alice"), (
        "the deletion committed without the invalidation that makes it binding"
    )


def test_a_successful_revocation_still_removes_and_invalidates(
    conn: sqlite3.Connection,
) -> None:
    """POSITIVE CONTROL for the transaction change."""
    _two_tenants(conn)
    catalog.grant_membership(
        conn, principal_id="alice", collection_id="t1-own",
        grants=["read"], tenant_id="t1", now=NOW,
    )
    before = catalog.read_revisions(conn, "t1")

    assert catalog.revoke_membership(
        conn, principal_id="alice", collection_id="t1-own", tenant_id="t1"
    )

    after = catalog.read_revisions(conn, "t1")
    assert catalog.authorized_collections(conn, "alice") == ()
    assert after.revocation_epoch > before.revocation_epoch
    assert after.membership_revision > before.membership_revision


# --- revision counters never move backwards --------------------------------


def test_revisions_may_not_be_rolled_back_for_an_existing_tenant(
    conn: sqlite3.Connection,
) -> None:
    """A rolled-back counter UN-REVOKES: every context invalidated by a prior
    bump matches the smaller value again."""
    catalog.set_revisions(conn, "t1", catalog.CatalogRevisions(revocation_epoch=7))

    with pytest.raises(ValueError, match="backwards"):
        catalog.set_revisions(conn, "t1", catalog.CatalogRevisions(revocation_epoch=3))

    assert catalog.read_revisions(conn, "t1").revocation_epoch == 7


def test_a_new_tenant_may_start_at_any_revision(conn: sqlite3.Connection) -> None:
    """POSITIVE CONTROL: provisioning at the control plane's numbers is the
    reason set_revisions exists at all."""
    catalog.set_revisions(
        conn,
        "fresh",
        catalog.CatalogRevisions(
            policy_revision=4, membership_revision=9, revocation_epoch=2
        ),
    )

    assert catalog.read_revisions(conn, "fresh").membership_revision == 9


def test_advancing_an_existing_tenant_is_still_allowed(
    conn: sqlite3.Connection,
) -> None:
    catalog.set_revisions(conn, "t1", catalog.CatalogRevisions(policy_revision=1))
    catalog.set_revisions(conn, "t1", catalog.CatalogRevisions(policy_revision=2))

    assert catalog.read_revisions(conn, "t1").policy_revision == 2


# --- the audit write owns no transaction but its own ------------------------


def test_auditing_does_not_commit_the_callers_pending_writes(
    conn: sqlite3.Connection,
) -> None:
    """`Connection.commit()` is connection-WIDE.

    The audit write is best-effort and self-committing, but it ran on the
    caller's connection, so it promoted every pending statement on it --
    including a failed handler's -- to durable state. The caller was told the
    request failed and the store kept half the effect.
    """
    _two_tenants(conn)
    conn.execute(
        "INSERT INTO lcm_teams_collections(collection_id, tenant_id, kind,"
        " created_at) VALUES('half-applied', 't1', 'own', ?)",
        (NOW,),
    )
    assert conn.in_transaction, "the fixture no longer models an open transaction"

    catalog.record_audit_event(
        conn,
        occurred_at=NOW,
        tenant_id="t1",
        principal_id="alice",
        operation="collections.create",
        allowed=False,
        denial_reason="unavailable",
    )
    conn.rollback()

    remaining = conn.execute(
        "SELECT COUNT(*) FROM lcm_teams_collections WHERE collection_id='half-applied'"
    ).fetchone()[0]
    assert remaining == 0, "the audit write committed the caller's partial write"


def test_an_idle_connection_still_gets_a_durable_audit_row(
    conn: sqlite3.Connection,
) -> None:
    """POSITIVE CONTROL: every audit-only writer must still persist evidence."""
    assert not conn.in_transaction

    catalog.record_audit_event(
        conn,
        occurred_at=NOW,
        tenant_id="t1",
        principal_id="alice",
        operation="read",
        allowed=False,
        denial_reason="scope_forbidden",
    )
    conn.rollback()

    assert len(catalog.read_audit_events(conn, tenant_id="t1")) == 1


# --- structural verification ----------------------------------------------


def test_a_table_with_a_missing_column_is_reported_as_a_defect(
    conn: sqlite3.Connection,
) -> None:
    """`CREATE TABLE IF NOT EXISTS` cannot repair this, so a name-only check
    hands an operator a clean result for a store that cannot serve a lookup."""
    conn.execute("DROP TABLE lcm_teams_principals")
    conn.execute(
        "CREATE TABLE lcm_teams_principals (principal_id TEXT PRIMARY KEY)"
    )
    conn.commit()

    errors = catalog.verify_teams_catalog(conn)

    assert "missing column:lcm_teams_principals.status" in errors
    assert "missing column:lcm_teams_principals.tenant_id" in errors
    assert not any(error.startswith("missing table:") for error in errors)


# --- an unchanged write is not a change ------------------------------------
#
# The same shape the grant path was already corrected for: an upsert that
# writes no authorization state still bumped a revision, and a bump is what
# makes every already-issued context for the tenant stop validating. A
# desired-state reconciler replaying its inventory therefore locked out
# principals nothing had changed.


def test_reprovisioning_an_unchanged_principal_invalidates_nothing(
    conn: sqlite3.Connection,
) -> None:
    """Desired-state reconciliation re-asserts what is already true."""
    catalog.provision_principal(conn, principal_id="alice", tenant_id="t1", now=NOW)
    before = catalog.read_revisions(conn, "t1").membership_revision

    again = catalog.provision_principal(
        conn, principal_id="alice", tenant_id="t1", now=NOW + 1
    )

    assert again.status == "active"
    assert catalog.read_revisions(conn, "t1").membership_revision == before, (
        "a no-op provision invalidated every context issued for the tenant"
    )


def test_reactivating_a_suspended_principal_still_bumps(
    conn: sqlite3.Connection,
) -> None:
    """POSITIVE CONTROL: a real state change must still invalidate."""
    catalog.provision_principal(conn, principal_id="alice", tenant_id="t1", now=NOW)
    catalog.suspend_principal(
        conn, principal_id="alice", tenant_id="t1", now=NOW + 1
    )
    before = catalog.read_revisions(conn, "t1").membership_revision

    catalog.provision_principal(
        conn, principal_id="alice", tenant_id="t1", now=NOW + 2
    )

    assert catalog.read_revisions(conn, "t1").membership_revision > before
    assert catalog.read_principal(conn, "alice").status == "active"


def test_a_refused_revocation_does_not_bump_the_counters(
    conn: sqlite3.Connection,
) -> None:
    """`revoke_membership` returned False and bumped anyway.

    The tenant predicate was added to the DELETE alone, so a foreign or
    already-gone membership still advanced the SUPPLIED tenant's revocation
    epoch and membership revision -- invalidating that tenant's contexts for a
    deletion that never happened. Suspension and grant were both corrected for
    this; revocation's regression asserted only the return value.
    """
    _two_tenants(conn)
    catalog.grant_membership(
        conn, principal_id="mallory", collection_id="t2-secret",
        grants=["read"], tenant_id="t2", now=NOW,
    )
    before = catalog.read_revisions(conn, "t1")

    assert (
        catalog.revoke_membership(
            conn, principal_id="mallory", collection_id="t2-secret", tenant_id="t1"
        )
        is False
    )

    assert catalog.read_revisions(conn, "t1") == before, (
        "a refused revocation invalidated the caller's own contexts"
    )


# --- the tenant predicate on collections -----------------------------------


def test_claiming_another_tenants_collection_id_is_refused(
    conn: sqlite3.Connection,
) -> None:
    """`INSERT OR IGNORE` skipped the insert and reported the id as created.

    The management ledger then caches a false success while the caller's later
    grant fails, because the stored collection still belongs to the other
    tenant.
    """
    _two_tenants(conn)

    with pytest.raises(LookupError):
        catalog.create_collection(
            conn, collection_id="t2-secret", tenant_id="t1", kind="own", now=NOW + 1
        )

    row = conn.execute(
        "SELECT tenant_id, kind FROM lcm_teams_collections WHERE collection_id='t2-secret'"
    ).fetchone()
    assert tuple(row) == ("t2", "shared")


def test_recreating_an_identical_collection_is_idempotent(
    conn: sqlite3.Connection,
) -> None:
    """POSITIVE CONTROL: the same tenant re-asserting the same row succeeds."""
    _two_tenants(conn)

    assert (
        catalog.create_collection(
            conn, collection_id="t1-own", tenant_id="t1", kind="own", now=NOW + 1
        )
        == "t1-own"
    )


def test_redefining_a_collections_kind_is_refused(conn: sqlite3.Connection) -> None:
    """Same tenant, different kind: a conflicting request, not a replay."""
    _two_tenants(conn)

    with pytest.raises(ValueError):
        catalog.create_collection(
            conn, collection_id="t1-own", tenant_id="t1", kind="shared", now=NOW + 1
        )


# --- one grant name is one capability --------------------------------------


def test_a_delimiter_bearing_grant_name_is_refused(conn: sqlite3.Connection) -> None:
    """Grants are stored comma-joined and read back comma-split.

    So one supplied grant `"custom,write"` became TWO grants, and
    `authorized_collections(..., grant="write")` then returned the collection
    even though `write` was never an element of the request -- a control-plane
    payload turning one value into a capability it never asked for.
    """
    _two_tenants(conn)

    with pytest.raises(ValueError):
        catalog.grant_membership(
            conn, principal_id="alice", collection_id="t1-own",
            grants=["custom,write"], tenant_id="t1", now=NOW,
        )

    assert catalog.authorized_collections(conn, "alice", grant="write") == ()
    assert catalog.read_memberships(conn, "alice") == ()


# --- a missing revision row is not revision zero ---------------------------


def test_a_known_tenant_with_no_revision_row_is_inconsistent(
    conn: sqlite3.Connection,
) -> None:
    """All-zero is the answer for an UNKNOWN tenant, and only for one.

    A tenant whose principals are still there but whose revision row was lost
    to a partial restore read as all-zero, so every context previously revoked
    by a policy, membership or revocation bump matched again -- undoing every
    later invalidation at once.
    """
    _two_tenants(conn)
    conn.execute("DELETE FROM lcm_teams_revisions WHERE tenant_id='t1'")
    conn.commit()

    with pytest.raises(catalog.InconsistentCatalogError):
        catalog.read_revisions(conn, "t1")

    # An genuinely unknown tenant is still the ordinary all-zero answer.
    assert catalog.read_revisions(conn, "t-nobody") == catalog.CatalogRevisions()
