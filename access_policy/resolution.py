"""The single policy-resolution point for the Teams authorization seam."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from enum import Enum

from ..access_context.denials import DenialReason
from ..access_context.model import AccessContextV1
from ..access_context.validation import ResolutionMode, resolve_mode, validate

from .fail_closed import FailClosedPolicy
from .teams_policy import TeamsPolicy
from .trusted_owner import TrustedOwnerPolicy


# The ONE place each binding is named. Hooks must not guess at attribute names:
# a cascade of getattr fallbacks silently yields a permissive policy when an
# attribute is renamed, which is the "unscoped fallback" #473 forbids and is
# invisible to tests because nothing fails.
TEAMS_ENABLED_ATTR = "lcm_teams_enabled"
ACCESS_CONTEXT_ACCESSOR = "get_lcm_access_context"


def policy_access_context(engine: object) -> AccessContextV1 | None:
    """Read the engine's carrier context through the same documented seam."""

    accessor = getattr(engine, ACCESS_CONTEXT_ACCESSOR, None)
    return accessor() if callable(accessor) else None


def policy_for_engine(
    engine: object,
) -> "TeamsPolicy | TrustedOwnerPolicy | FailClosedPolicy":
    """Resolve the policy for an engine through one documented seam.

    Every hook site calls this rather than reading the engine itself, so there
    is a single place to change when the host wiring lands, and F5 can assert
    that no hook resolves a policy any other way.

    Note the asymmetry that makes this safe: an engine with no Teams wiring at
    all reports ``teams_enabled=False`` and gets the permissive policy, which
    is correct default-off. But an engine that HAS Teams enabled and is missing
    its context accessor yields ``resolve_policy(None, True)`` -> FAIL_CLOSED.
    Enabled-but-unwired fails closed; it does not fall back to permissive.
    """

    teams_enabled = bool(getattr(engine, TEAMS_ENABLED_ATTR, False))
    context = policy_access_context(engine)
    if not teams_enabled or not isinstance(context, AccessContextV1):
        return resolve_policy(context, teams_enabled)

    status, revisions = _catalog_revisions(engine, context)
    if status is _CatalogLookup.UNREADABLE:
        # A store IS bound, and it cannot say whether this context has been
        # revoked. Falling through with "no revisions" would resolve permissive
        # on exactly the store that failed to answer -- the silent-permissive
        # shape this seam exists to prevent. Note zero is NOT a safe default
        # either: zero is a real revision value, so a context minted at zero
        # would validate against a catalog nobody could read.
        return FailClosedPolicy(DenialReason.CONTEXT_INVALID)

    principal_denial = _catalog_principal_denial(engine, context)
    if principal_denial is not None:
        return FailClosedPolicy(principal_denial)

    return resolve_policy(
        context,
        teams_enabled,
        current_revisions=revisions,
        audit_sink=_audit_sink_for_engine(engine),
        session_owner=_session_owner_for_engine(engine),
    )


def _catalog_principal_denial(
    engine: object, context: AccessContextV1
) -> DenialReason | None:
    """Refuse a context whose principal the catalog does not stand behind.

    The revision counters are per-TENANT, so they say nothing about the
    principal. A principal suspended after its context was minted is caught by
    the revocation-epoch bump, but a context minted AFTER that bump carries the
    current epoch and validated cleanly -- restoring access to a suspended
    principal even though ``authorized_collections()`` correctly returns
    nothing for it.

    A catalog that exists is authoritative about its principals, the same way
    ``_catalog_revisions`` treats an existing-but-unreadable catalog as
    inconsistent rather than absent. So an absent principal, a suspended one,
    and one belonging to another tenant all fail closed here.
    """

    from ..teams.catalog import read_principal

    store = getattr(engine, "_store", None)
    connection = getattr(store, "connection", None)
    if connection is None:
        return None
    try:
        principal = read_principal(connection, context.principal_id)
    except Exception:
        return DenialReason.CONTEXT_INVALID
    if principal is None:
        return DenialReason.CONTEXT_INVALID
    if principal.tenant_id != str(context.tenant_id):
        return DenialReason.CONTEXT_INVALID
    if principal.status != "active":
        return DenialReason.CONTEXT_REVOKED
    return None


class OwnerLookupError(RuntimeError):
    """The stamped owner of a session could not be determined.

    Raised rather than returned, because ``None`` on this seam means "nothing
    claims this session" and the policy DELIBERATELY allows an unclaimed
    target. Collapsing a failed lookup into that answer applies
    unclaimed-target behaviour after an ownership check that never ran.
    ``TeamsPolicy._target_owner`` turns any exception here into
    ``_OWNER_LOOKUP_FAILED`` -> ``CONTEXT_INVALID``, which is the fail-closed
    direction #68 records for callback failures.
    """


def _session_owner_for_engine(engine: object):
    """Bind a target-session owner resolver to this engine's store, or None.

    Answers "who owns this session" from the stamps already on disk, which is
    authoritative: the same value the write path assigns. A session with no
    stamped rows resolves to None and is treated as unclaimed by the policy.

    A callable rather than a connection, for the same reason as the audit sink:
    the policy keeps no database handle and stays testable with a plain dict.
    """

    store = getattr(engine, "_store", None)
    connection = getattr(store, "connection", None)
    if connection is None:
        return None

    def resolve(session_id: str) -> str | None:
        owners: set[str] = set()
        readable = 0
        for table in ("messages", "summary_nodes"):
            try:
                rows = connection.execute(
                    f'SELECT DISTINCT access_scope FROM "{table}" '
                    "WHERE session_id = ? AND access_scope IS NOT NULL",
                    (session_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                # Table or column absent on this store. Counted, not ignored:
                # if NEITHER table could be read the answer is unknown, not
                # "unclaimed".
                continue
            readable += 1
            owners.update(str(row[0]) for row in rows if row[0] is not None)
        if not readable:
            raise OwnerLookupError(f"no owner table could be read for {session_id!r}")
        if len(owners) > 1:
            # Rows for one session carrying more than one owner stamp is an
            # inconsistent store, and `LIMIT 1` used to pick one arbitrarily
            # and report it as authoritative -- enough to pass a session-level
            # reset, compaction or lifecycle gate over another principal's
            # rows. A conflict is not an answer.
            raise OwnerLookupError(
                f"session {session_id!r} carries {len(owners)} conflicting owner stamps"
            )
        return next(iter(owners)) if owners else None

    return resolve


def _audit_sink_for_engine(engine: object):
    """Bind an audit writer to this engine's store, or None.

    A CALLABLE rather than the connection itself: the policy stays free of a
    database handle, which is what keeps it unit-testable with a recording fake
    and consistent with how raw-identifier ownership is resolved (in the
    engine, before the policy is asked).
    """

    from ..teams.catalog import record_audit_event

    store = getattr(engine, "_store", None)
    connection = getattr(store, "connection", None)
    if connection is None:
        return None

    def sink(**fields: object) -> None:
        record_audit_event(connection, occurred_at=time.time(), **fields)

    return sink


class _CatalogLookup(Enum):
    OK = "ok"
    UNREADABLE = "unreadable"
    NOT_STORE_BACKED = "not_store_backed"


def _catalog_revisions(
    engine: object, context: AccessContextV1
) -> tuple["_CatalogLookup", object | None]:
    """Read the tenant's CURRENT revisions from the catalog.

    Two absences that look alike and must not be treated alike:

    UNREADABLE -- a store is bound but its catalog is missing or unreadable.
    That is an inconsistent Teams store and it fails closed.

    NOT_STORE_BACKED -- the caller passed a context carrier with no store at
    all. ``scripts/import_lossless_claw`` does this deliberately: its ``engine``
    parameter is typed ``object | None`` and exists only to carry a context, so
    there is no catalog for it to be inconsistent with. Revocation is therefore
    NOT enforced on that path; it gets a real authenticated surface in the
    connector phase, and until then this is a stated limitation rather than a
    silent one.
    """

    from ..teams.catalog import read_revisions, teams_catalog_exists

    store = getattr(engine, "_store", None)
    connection = getattr(store, "connection", None)
    if connection is None:
        return _CatalogLookup.NOT_STORE_BACKED, None
    try:
        if not teams_catalog_exists(connection):
            return _CatalogLookup.UNREADABLE, None
        return _CatalogLookup.OK, read_revisions(connection, context.tenant_id)
    except Exception:
        return _CatalogLookup.UNREADABLE, None


def resolve_policy(
    carrier_context: AccessContextV1 | None,
    teams_enabled: bool,
    now: datetime | None = None,
    current_revisions: object | None = None,
    audit_sink: object | None = None,
    session_owner: object | None = None,
) -> TeamsPolicy | TrustedOwnerPolicy | FailClosedPolicy:
    """Resolve the policy from the explicit Teams flag and carrier context.

    Teams-off is deliberately resolved before context validation: carrying a
    context does not enable Teams, and the default path remains trusted-owner.

    ``current_revisions`` carries the catalog's policy/membership/revocation
    counters. Passing them is what makes the NOT_REVOKED stage live: without
    them every comparison in that stage is against ``None`` and short-circuits,
    so a revoked context validated exactly like a current one.

    A valid context now resolves to :class:`TeamsPolicy`, which scopes to the
    acting principal. The permissive placeholder is gone.

    ⚠ Still inert HERE, and worth naming rather than leaving to be discovered:
    OWNERSHIP_CURRENT's generation check and LEASE_CURRENT, because the catalog
    does not yet track ownership generations or leases. SCOPE_PERMITTED and
    TARGET_RESOLUTION are per-OPERATION rather than per-context, so they belong
    in the policy's authorize_operation -- which is where TeamsPolicy now makes
    them, via the owner-of-target and store-wide rules.
    """

    mode = resolve_mode(carrier_context, teams_enabled)
    if mode is ResolutionMode.STANDARD_UNMANAGED:
        return TrustedOwnerPolicy(teams_enabled=False)
    if mode is ResolutionMode.FAIL_CLOSED:
        return FailClosedPolicy(DenialReason.CONTEXT_MISSING)

    assert mode is ResolutionMode.ENFORCING
    if not isinstance(carrier_context, AccessContextV1):
        return FailClosedPolicy(DenialReason.CONTEXT_INVALID)
    validation = validate(
        carrier_context,
        now=now if now is not None else datetime.now(timezone.utc),
        current_policy_revision=getattr(current_revisions, "policy_revision", None),
        current_membership_revision=getattr(
            current_revisions, "membership_revision", None
        ),
        current_revocation_epoch=getattr(current_revisions, "revocation_epoch", None),
    )
    if validation.allowed:
        return TeamsPolicy(
            carrier_context, audit_sink=audit_sink, session_owner=session_owner
        )
    assert validation.denial_reason is not None
    return FailClosedPolicy(validation.denial_reason)
