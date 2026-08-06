"""Inert policy layer for the LCM authorization seam."""

from .errors import AuthorizationRequiredError
from .fail_closed import FailClosedPolicy
from .resolution import (
    ACCESS_CONTEXT_ACCESSOR,
    TEAMS_ENABLED_ATTR,
    policy_access_context,
    policy_for_engine,
    resolve_policy,
)
from .trusted_owner import TrustedOwnerPolicy

__all__ = [
    "AuthorizationRequiredError",
    "ACCESS_CONTEXT_ACCESSOR",
    "FailClosedPolicy",
    "TEAMS_ENABLED_ATTR",
    "policy_access_context",
    "policy_for_engine",
    "TrustedOwnerPolicy",
    "resolve_policy",
]
