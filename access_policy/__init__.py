"""Inert policy layer for the LCM authorization seam."""

from .errors import AuthorizationRequiredError
from .fail_closed import FailClosedPolicy
from .resolution import policy_access_context, policy_for_engine, resolve_policy
from .trusted_owner import TrustedOwnerPolicy

__all__ = [
    "AuthorizationRequiredError",
    "FailClosedPolicy",
    "policy_access_context",
    "policy_for_engine",
    "TrustedOwnerPolicy",
    "resolve_policy",
]
