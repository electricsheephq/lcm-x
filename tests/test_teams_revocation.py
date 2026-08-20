"""Revocation, made live.

``resolve_policy`` called ``validate()`` with only ``now``, so every comparison
in the NOT_REVOKED stage was against ``None`` and short-circuited. The stage had
tests and they passed -- because they called ``validate()`` directly with the
arguments production never supplied. A revoked context validated exactly like a
current one, and nothing anywhere reported it.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.access_context import DenialReason
from hermes_lcm.access_context.model import AccessContextV1
from hermes_lcm.access_policy import (
    FailClosedPolicy,
    TeamsPolicy,
    TrustedOwnerPolicy,
    policy_for_engine,
    resolve_policy,
)
from hermes_lcm.teams import catalog


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _context(**overrides) -> AccessContextV1:
    fields = dict(
        authenticated_transport="host-session",
        context_id="ctx-1",
        request_id="req-1",
        source_kind="human",
        deployment_id="dep-1",
        tenant_id="tenant-1",
        principal_id="principal-a",
        profile_id="profile-a",
        profile_incarnation="incarnation-1",
        session_id="session-a",
        session_owner_principal_id="principal-a",
        conversation_id="conversation-a",
        conversation_lane="lane-a",
        read_policy_ref="policy-a",
        lease_id="lease-a",
        issued_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        expires_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    fields.update(overrides)
    return AccessContextV1.from_host(**fields)


def _provisioned(
    store: sqlite3.Connection,
    *,
    principal_id: str = "principal-a",
    tenant_id: str = "tenant-1",
    **overrides,
) -> AccessContextV1:
    """Provision the principal, then mint a context at the catalog's revisions.

    A catalog that exists is authoritative about its principals, so a context
    for a principal it has never heard of fails closed. Provisioning itself
    bumps ``membership_revision``, hence reading the counters back rather than
    minting at zero.
    """

    catalog.provision_principal(
        store, principal_id=principal_id, tenant_id=tenant_id, now=0.0
    )
    current = catalog.read_revisions(store, tenant_id)
    fields = dict(
        principal_id=principal_id,
        session_owner_principal_id=principal_id,
        tenant_id=tenant_id,
        policy_revision=current.policy_revision,
        membership_revision=current.membership_revision,
        revocation_epoch=current.revocation_epoch,
    )
    fields.update(overrides)
    return _context(**fields)


class _Store:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection


class _Engine:
    """An engine wired the way the host wires one."""

    def __init__(self, connection, context, *, teams_enabled: bool = True) -> None:
        self._store = _Store(connection)
        if teams_enabled:
            self.lcm_teams_enabled = True
        if context is not None:
            self.get_lcm_access_context = lambda: context


@pytest.fixture()
def store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "lcm.db")
    db_bootstrap.configure_connection(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.mark.parametrize(
    "field, context_kwargs",
    [
        ("revocation_epoch", {"revocation_epoch": 0}),
        ("membership_revision", {"membership_revision": 0}),
        ("policy_revision", {"policy_revision": 0}),
    ],
)
def test_a_stale_revision_is_refused(field: str, context_kwargs: dict) -> None:
    revisions = catalog.CatalogRevisions(**{field: 3})

    policy = resolve_policy(
        _context(**context_kwargs), True, NOW, current_revisions=revisions
    )

    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason is DenialReason.CONTEXT_REVOKED


def test_a_current_revision_is_permitted() -> None:
    policy = resolve_policy(
        _context(revocation_epoch=3),
        True,
        NOW,
        current_revisions=catalog.CatalogRevisions(revocation_epoch=3),
    )
    assert isinstance(policy, TeamsPolicy)


def test_without_revisions_the_same_stale_context_passes() -> None:
    """Pins the defect itself, so a regression is visible rather than silent."""
    stale = _context(revocation_epoch=0)

    unwired = resolve_policy(stale, True, NOW)
    wired = resolve_policy(
        stale, True, NOW, current_revisions=catalog.CatalogRevisions(revocation_epoch=3)
    )

    assert isinstance(unwired, TeamsPolicy)  # permitted, because nothing said otherwise
    assert isinstance(wired, FailClosedPolicy)


def test_bumping_the_epoch_invalidates_an_already_issued_context(
    store: sqlite3.Connection,
) -> None:
    """The operation an operator actually performs to revoke someone."""
    catalog.ensure_teams_catalog(store)
    issued = _provisioned(store)
    assert isinstance(policy_for_engine(_Engine(store, issued)), TeamsPolicy)

    catalog.bump_revision(store, "tenant-1", "revocation_epoch")

    revoked = policy_for_engine(_Engine(store, issued))
    assert isinstance(revoked, FailClosedPolicy)
    assert revoked.denial_reason is DenialReason.CONTEXT_REVOKED


def test_a_context_minted_after_the_bump_is_accepted(
    store: sqlite3.Connection,
) -> None:
    catalog.ensure_teams_catalog(store)
    catalog.bump_revision(store, "tenant-1", "revocation_epoch")

    policy = policy_for_engine(_Engine(store, _provisioned(store)))

    assert isinstance(policy, TeamsPolicy)


def test_teams_on_without_a_catalog_fails_closed(store: sqlite3.Connection) -> None:
    """The store cannot say whether this context was revoked, so it refuses.

    Defaulting to zero here would be worse than useless: zero is a real
    revision value, so a context minted at zero would validate against a
    catalog nobody could read.
    """
    policy = policy_for_engine(_Engine(store, _context()))

    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason is DenialReason.CONTEXT_INVALID


def test_teams_off_never_consults_the_catalog(store: sqlite3.Connection) -> None:
    """The negative control: default-off must be untouched by all of this."""
    policy = policy_for_engine(_Engine(store, None, teams_enabled=False))
    assert isinstance(policy, TrustedOwnerPolicy)


def test_each_counter_is_checked_independently(store: sqlite3.Connection) -> None:
    """A membership change must not be waved through by a matching epoch."""
    catalog.ensure_teams_catalog(store)
    issued = _provisioned(store)
    catalog.bump_revision(store, "tenant-1", "membership_revision")

    # Epoch still matches; only membership moved.
    policy = policy_for_engine(_Engine(store, issued))

    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason is DenialReason.CONTEXT_REVOKED


# --- the catalog's word on the principal ----------------------------------


def test_a_suspended_principal_is_refused_at_the_current_epoch(
    store: sqlite3.Connection,
) -> None:
    """The hole the tenant counters cannot close.

    Suspension bumps the revocation epoch, so contexts already issued go
    stale. But a context minted AFTER the bump carries the current epoch and
    validated cleanly, restoring access to a principal the catalog has
    suspended -- while `authorized_collections()` correctly returns nothing
    for it.
    """
    catalog.ensure_teams_catalog(store)
    _provisioned(store)
    catalog.suspend_principal(
        store, principal_id="principal-a", tenant_id="tenant-1", now=1.0
    )
    current = catalog.read_revisions(store, "tenant-1")
    minted_after = _context(
        policy_revision=current.policy_revision,
        membership_revision=current.membership_revision,
        revocation_epoch=current.revocation_epoch,
    )

    policy = policy_for_engine(_Engine(store, minted_after))

    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason is DenialReason.CONTEXT_REVOKED


def test_a_principal_the_catalog_never_provisioned_is_refused(
    store: sqlite3.Connection,
) -> None:
    """An existing catalog is authoritative about who its principals are."""
    catalog.ensure_teams_catalog(store)

    policy = policy_for_engine(_Engine(store, _context()))

    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason is DenialReason.CONTEXT_INVALID


def test_a_principal_belonging_to_another_tenant_is_refused(
    store: sqlite3.Connection,
) -> None:
    """A context may not borrow a principal id that lives in another tenant."""
    catalog.ensure_teams_catalog(store)
    catalog.provision_principal(
        store, principal_id="principal-a", tenant_id="tenant-other", now=0.0
    )

    policy = policy_for_engine(_Engine(store, _context()))

    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason is DenialReason.CONTEXT_INVALID


def test_a_re_provisioned_principal_is_served_again(
    store: sqlite3.Connection,
) -> None:
    """POSITIVE CONTROL: refusing everyone also passes the three tests above."""
    catalog.ensure_teams_catalog(store)
    _provisioned(store)
    catalog.suspend_principal(
        store, principal_id="principal-a", tenant_id="tenant-1", now=1.0
    )

    policy = policy_for_engine(_Engine(store, _provisioned(store)))

    assert isinstance(policy, TeamsPolicy)


def test_a_context_carrier_with_no_store_is_not_treated_as_an_inconsistent_store(
    store: sqlite3.Connection,
) -> None:
    """scripts/import_lossless_claw passes an object, not an engine.

    Its `engine` parameter is typed `object | None` and exists only to carry a
    context, so there is no catalog for it to be inconsistent with. Revocation
    is NOT enforced on that path -- stated here so the limitation is visible
    rather than discovered, and so the two absences stay distinguishable.
    """
    from types import SimpleNamespace

    carrier = SimpleNamespace(
        lcm_teams_enabled=True, get_lcm_access_context=lambda: _context()
    )

    assert isinstance(policy_for_engine(carrier), TeamsPolicy)
