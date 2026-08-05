"""The single policy-resolution point for the Teams authorization seam."""

from __future__ import annotations

from datetime import datetime, timezone

from access_context.denials import DenialReason
from access_context.model import AccessContextV1
from access_context.validation import ResolutionMode, resolve_mode, validate

from .fail_closed import FailClosedPolicy
from .trusted_owner import TrustedOwnerPolicy


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
