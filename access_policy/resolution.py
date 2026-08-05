"""The single policy-resolution point for the Teams authorization seam."""

from __future__ import annotations

from datetime import datetime, timezone

from access_context.denials import DenialReason
from access_context.model import AccessContextV1
from access_context.validation import ResolutionMode, resolve_mode, validate

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
    return resolve_policy(policy_access_context(engine), teams_enabled)


def resolve_policy(
    carrier_context: AccessContextV1 | None,
    teams_enabled: bool,
    now: datetime | None = None,
) -> TrustedOwnerPolicy | FailClosedPolicy:
    """Resolve the policy from the explicit Teams flag and carrier context.

    Teams-off is deliberately resolved before context validation: carrying a
    context does not enable Teams, and the default path remains trusted-owner.

    ⚠ A VALID Teams context currently resolves to ``TrustedOwnerPolicy``, which
    permits everything. That is a placeholder, not enforcement. This slice adds
    the neutral seam only; the policy that actually scopes a principal --
    membership, roles, catalogs -- is the Teams adapter in #483. Until that
    lands, enabling Teams changes which policy object is constructed and
    nothing about what is permitted. Do not read a valid context reaching this
    branch as "Teams is enforced".
    """

    mode = resolve_mode(carrier_context, teams_enabled)
    if mode is ResolutionMode.STANDARD_UNMANAGED:
        return TrustedOwnerPolicy()
    if mode is ResolutionMode.FAIL_CLOSED:
        return FailClosedPolicy(DenialReason.CONTEXT_MISSING)

    assert mode is ResolutionMode.ENFORCING
    validation = validate(
        carrier_context,
        now=now if now is not None else datetime.now(timezone.utc),
    )
    if validation.allowed:
        return TrustedOwnerPolicy()
    assert validation.denial_reason is not None
    return FailClosedPolicy(validation.denial_reason)
