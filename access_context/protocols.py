"""Producer/consumer protocols for the future host-to-LCM authorization seam."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .denials import Decision, DenialReason, PublicDecision
from .model import AccessContextV1

TargetScope = Mapping[str, Any]


@runtime_checkable
class HostContextCarrier(Protocol):
    """Host-side producer; ``None`` means this operation has no Teams context."""

    def get_lcm_access_context(self) -> AccessContextV1 | None:
        """Return the authenticated context for the current operation."""


@runtime_checkable
class LcmAuthorizationConsumer(Protocol):
    """Consumer seam that must authorize before any disclosure primitive."""

    def authorize_operation(
        self,
        context: AccessContextV1 | None,
        operation: str,
        expected_scope: TargetScope,
    ) -> Decision:
        """Authorize an operation before selecting or disclosing targets."""

    def resolve_authorized_targets(
        self,
        context: AccessContextV1 | None,
        operation: str,
        requested_narrowing: TargetScope,
    ) -> Sequence[Any]:
        """Resolve targets only within the already-authorized narrowing."""

    def authorize_stored_scope(
        self,
        context: AccessContextV1 | None,
        operation: str,
        stored_scope: TargetScope,
    ) -> Decision:
        """Re-authorize scope persisted with a stored target or handle."""

    def audit_decision(
        self,
        context: AccessContextV1 | None,
        operation: str,
        internal_reason: DenialReason | None,
        public_result: PublicDecision,
    ) -> None:
        """Record the exact internal reason alongside its public projection.

        ``internal_reason`` is ``None`` for an allow; the seam audits allow and
        deny alike, so the reason is only present on the deny path.
        """

    def select_collection(self, target_scope: TargetScope) -> Any:
        """Select a backing collection only after authorization."""

    def count_candidates(self, candidates: Sequence[Any]) -> int:
        """Count candidates only after authorization."""

    def rank_candidates(self, candidates: Sequence[Any]) -> Sequence[Any]:
        """Rank candidates only after authorization."""

    def hydrate_targets(self, targets: Sequence[Any]) -> Sequence[Any]:
        """Hydrate target content only after authorization."""

    def issue_handle(self, target: Any) -> Any:
        """Issue a cursor/reference handle only after authorization."""
