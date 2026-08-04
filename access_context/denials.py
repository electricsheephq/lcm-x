"""Typed authorization outcomes and the intentional public disclosure policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class DenialReason(str, Enum):
    CONTEXT_MISSING = "context_missing"
    CONTEXT_INVALID = "context_invalid"
    CONTEXT_UNSUPPORTED_VERSION = "context_unsupported_version"
    CONTEXT_EXPIRED = "context_expired"
    CONTEXT_REVOKED = "context_revoked"
    SCOPE_FORBIDDEN = "scope_forbidden"
    SCOPE_MISMATCH = "scope_mismatch"
    OWNERSHIP_CHANGED = "ownership_changed"
    LEASE_STALE = "lease_stale"
    TARGET_NOT_FOUND_OR_FORBIDDEN = "target_not_found_or_forbidden"


_CONTENT_FREE_DETAIL_KEYS = frozenset(
    {
        "context_id",
        "request_id",
        "principal_id",
        "tenant_id",
        "deployment_id",
        "profile_id",
        "session_id",
        "conversation_id",
        "lease_id",
        "target_id",
        "policy_revision",
        "membership_revision",
        "revocation_epoch",
        "ownership_generation",
        "lease_generation",
        "expected_ownership_generation",
        "expected_lease_generation",
        "expected_membership_revision",
        "expected_revocation_epoch",
    }
)


# The only keys a denied caller may see. Both are values the caller itself
# supplied, so echoing them teaches it nothing it did not already know.
_PUBLIC_DETAIL_KEYS: tuple[str, ...] = ("context_id", "request_id")


def _public_detail(detail: Mapping[str, Any]) -> Mapping[str, Any]:
    """Project detail to a FIXED key set, identical for every denial reason.

    The shape must not vary: several internal reasons blur to one public
    reason, so a key that appeared only for, say, ``LEASE_STALE`` would
    re-identify it and undo ``PUBLIC_DENIAL_PROJECTION``. Keys are always
    present, carrying ``None`` when absent internally.
    """

    return MappingProxyType({key: detail.get(key) for key in _PUBLIC_DETAIL_KEYS})


def _safe_detail(detail: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not detail:
        return MappingProxyType({})
    clean = {
        str(key): value
        for key, value in detail.items()
        if str(key) in _CONTENT_FREE_DETAIL_KEYS and isinstance(value, (str, int, bool, type(None)))
    }
    return MappingProxyType(clean)


# Keep this as one table so the public existence-disclosure policy can be
# reviewed without tracing conditionals through the validator.
PUBLIC_DENIAL_PROJECTION: Mapping[DenialReason, DenialReason] = MappingProxyType(
    {
        DenialReason.CONTEXT_MISSING: DenialReason.CONTEXT_MISSING,
        DenialReason.CONTEXT_INVALID: DenialReason.CONTEXT_INVALID,
        DenialReason.CONTEXT_UNSUPPORTED_VERSION: DenialReason.CONTEXT_UNSUPPORTED_VERSION,
        DenialReason.CONTEXT_EXPIRED: DenialReason.CONTEXT_EXPIRED,
        DenialReason.CONTEXT_REVOKED: DenialReason.CONTEXT_REVOKED,
        DenialReason.SCOPE_FORBIDDEN: DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN,
        DenialReason.SCOPE_MISMATCH: DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN,
        DenialReason.OWNERSHIP_CHANGED: DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN,
        DenialReason.LEASE_STALE: DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN,
        DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN: DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN,
    }
)


@dataclass(frozen=True)
class Decision:
    """Internal result retaining the exact denial reason and safe details."""

    allowed: bool
    denial_reason: DenialReason | None = None
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        reason = self.denial_reason
        if reason is not None:
            reason = DenialReason(reason)
        if self.allowed and reason is not None:
            raise ValueError("an allowed decision cannot carry a denial reason")
        if not self.allowed and reason is None:
            raise ValueError("a denied decision requires a denial reason")
        object.__setattr__(self, "denial_reason", reason)
        object.__setattr__(self, "detail", _safe_detail(self.detail))

    @classmethod
    def allow(cls, **detail: Any) -> "Decision":
        return cls(True, None, detail)

    @classmethod
    def deny(cls, reason: DenialReason, **detail: Any) -> "Decision":
        return cls(False, DenialReason(reason), detail)

    def __hash__(self) -> int:
        # ``detail`` is a mappingproxy, which is unhashable, so the generated
        # __hash__ raises despite frozen=True advertising the opposite. Values
        # are constrained to str/int/bool/None by _safe_detail, so the sorted
        # item tuple is hashable.
        return hash((self.allowed, self.denial_reason, tuple(sorted(self.detail.items()))))

    @property
    def reason(self) -> DenialReason | None:
        """Short alias used by consumers that call the field ``reason``."""

        return self.denial_reason

    @property
    def is_allowed(self) -> bool:
        return self.allowed

    def public(self) -> "PublicDecision":
        return project_public(self)


@dataclass(frozen=True)
class PublicDecision:
    """Public projection that does not reveal target existence."""

    allowed: bool
    denial_reason: DenialReason | None = None
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __hash__(self) -> int:
        # See Decision.__hash__ — same mappingproxy caveat.
        return hash((self.allowed, self.denial_reason, tuple(sorted(self.detail.items()))))

    @property
    def reason(self) -> DenialReason | None:
        return self.denial_reason


def project_public(decision: Decision) -> PublicDecision:
    """Apply the complete, explicit public denial projection table."""

    if decision.allowed:
        return PublicDecision(True)
    assert decision.denial_reason is not None
    return PublicDecision(
        False,
        PUBLIC_DENIAL_PROJECTION[decision.denial_reason],
        _public_detail(decision.detail),
    )


# Explicit names for callers that want to distinguish the two projections;
# both remain the same inert result shape.
InternalDecision = Decision
public_decision = project_public
