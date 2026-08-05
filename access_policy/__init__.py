"""Inert policy layer for the LCM authorization seam."""

from .errors import AuthorizationRequiredError
from .fail_closed import FailClosedPolicy
from .resolution import resolve_policy
from .trusted_owner import TrustedOwnerPolicy

__all__ = [
    "AuthorizationRequiredError",
    "FailClosedPolicy",
    "TrustedOwnerPolicy",
    "resolve_policy",
]
