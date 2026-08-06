"""The single policy-resolution point for the Teams authorization seam."""

from __future__ import annotations

from datetime import datetime, timezone

from ..access_context.denials import DenialReason
from ..access_context.model import AccessContextV1
from ..access_context.validation import ResolutionMode, resolve_mode, validate

from .fail_closed import FailClosedPolicy
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


def policy_for_engine(engine: object) -> "TrustedOwnerPolicy | FailClosedPolicy":
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

    revisions = _catalog_revisions(engine, context)
    if revisions is None:
        # Teams is on with a valid context, but the catalog that owns the
        # revisions is missing or unreadable. Falling through with "no
        # revisions" would resolve permissive on a store that cannot say
        # whether this context has been revoked -- the exact silent-permissive
        # shape this seam exists to prevent.
        return FailClosedPolicy(DenialReason.CONTEXT_INVALID)
    return resolve_policy(context, teams_enabled, current_revisions=revisions)


def _catalog_revisions(engine: object, context: AccessContextV1):
    """Read the tenant's CURRENT revisions, or None if they cannot be read.

    Deliberately not defaulted to zero on failure: zero is a real revision
    value, and a context minted at zero would then validate against a store
    whose catalog could not be consulted.
    """

    from ..teams.catalog import read_revisions, teams_catalog_exists

    store = getattr(engine, "_store", None)
    connection = getattr(store, "connection", None)
    if connection is None:
        return None
    try:
        if not teams_catalog_exists(connection):
            return None
        return read_revisions(connection, context.tenant_id)
    except Exception:
        return None


def resolve_policy(
    carrier_context: AccessContextV1 | None,
    teams_enabled: bool,
    now: datetime | None = None,
    current_revisions: object | None = None,
) -> TrustedOwnerPolicy | FailClosedPolicy:
    """Resolve the policy from the explicit Teams flag and carrier context.

    Teams-off is deliberately resolved before context validation: carrying a
    context does not enable Teams, and the default path remains trusted-owner.

    ``current_revisions`` carries the catalog's policy/membership/revocation
    counters. Passing them is what makes the NOT_REVOKED stage live: without
    them every comparison in that stage is against ``None`` and short-circuits,
    so a revoked context validated exactly like a current one.

    ⚠ Still NOT enforced by this function, and worth naming rather than
    leaving to be discovered: OWNERSHIP_CURRENT's generation check and
    LEASE_CURRENT both remain inert, because the catalog does not yet track
    ownership generations or leases. SCOPE_PERMITTED and TARGET_RESOLUTION are
    per-OPERATION rather than per-context and belong in the policy's
    authorize_operation, not here. A valid context reaching the allowed branch
    below still resolves to ``TrustedOwnerPolicy``; revocation is now real, the
    rest of enforcement is not.
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
        return TrustedOwnerPolicy(teams_enabled=True)
    assert validation.denial_reason is not None
    return FailClosedPolicy(validation.denial_reason)
