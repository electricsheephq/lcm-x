"""Producer/consumer protocols for the future host-to-LCM authorization seam."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .denials import Decision
from .model import AccessContextV1

TargetScope = Mapping[str, Any]


@runtime_checkable
class HostContextCarrier(Protocol):
    """Host-side producer; ``None`` means this operation has no Teams context."""

    def get_access_context(self) -> AccessContextV1 | None:
        """Return the authenticated context for the current operation."""


@runtime_checkable
class LcmAuthorizationConsumer(Protocol):
    """Consumer seam that must authorize before any disclosure primitive."""

    def authorize(
        self,
        context: AccessContextV1 | None,
        operation: str,
        target_scope: TargetScope,
    ) -> Decision:
        """Authorize an operation before selecting or disclosing targets."""

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
