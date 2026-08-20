"""Neutral default-deny policy for an unusable Teams context."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..access_context.denials import Decision, DenialReason, PublicDecision
from ..access_context.model import AccessContextV1
from ..access_context.protocols import TargetScope

from .errors import AuthorizationRequiredError
from .teams_policy import principal_of


class FailClosedPolicy:
    """Deny every authorization or disclosure operation without a context.

    Carries the audit sink and the attributable context when the resolver has
    them. Without them, the denials this policy produces -- an expired,
    revoked, stale or otherwise rejected Teams context, which is exactly the
    set an operator investigates -- reached only the in-memory list below,
    while valid-context denials were written durably by ``TeamsPolicy``. The
    audit trail was missing precisely the attempts that failed hardest.
    """

    def __init__(
        self,
        denial_reason: DenialReason = DenialReason.CONTEXT_MISSING,
        *,
        audit_sink: "Callable[..., None] | None" = None,
        context: AccessContextV1 | None = None,
    ) -> None:
        self.denial_reason = DenialReason(denial_reason)
        self.audit_records: list[tuple[DenialReason | None, PublicDecision]] = []
        self._audit_sink = audit_sink
        self._context = context

    def _deny(self) -> Decision:
        return Decision.deny(self.denial_reason)

    def authorize_operation(
        self,
        context: AccessContextV1 | None,
        operation: str,
        expected_scope: TargetScope,
    ) -> Decision:
        return self._deny()

    def resolve_authorized_targets(
        self,
        context: AccessContextV1 | None,
        operation: str,
        requested_narrowing: TargetScope,
    ) -> TargetScope:
        # An empty SCOPE, not an empty sequence: callers read this with .get,
        # and the protocol declares a mapping. Both are falsy, so the practical
        # behaviour is unchanged -- this keeps the type honest.
        return {}

    def authorize_stored_scope(
        self,
        context: AccessContextV1 | None,
        operation: str,
        stored_scope: TargetScope,
    ) -> Decision:
        return self._deny()

    def audit_decision(
        self,
        context: AccessContextV1 | None,
        operation: str,
        internal_reason: DenialReason | None,
        public_result: PublicDecision,
    ) -> None:
        self.audit_records.append((internal_reason, public_result))
        if self._audit_sink is None:
            return None
        # Every decision this policy makes is a denial, so there is no
        # read-allow volume to bound the way ``TeamsPolicy.audit_decision``
        # does. The reason written is the PUBLIC one, for the same reason:
        # these rows reach tenant admins through `audit.read`.
        effective = context if context is not None else self._context
        reason = getattr(public_result, "denial_reason", None)
        try:
            self._audit_sink(
                tenant_id=str(getattr(effective, "tenant_id", "") or ""),
                principal_id=principal_of(effective),
                operation=str(operation),
                allowed=False,
                denial_reason=getattr(reason, "value", None) if reason else None,
            )
        except Exception:  # noqa: BLE001 - auditing is best effort
            # Same rule as TeamsPolicy: auditing must never be the reason work
            # fails -- and here the work has already been denied anyway.
            return None
        return None

    # The disclosure primitives RAISE rather than return a Decision. The
    # protocol declares int/Sequence returns here, and a Decision is truthy,
    # so returning one would let `if policy.select_collection(scope):` sail
    # straight through -- a fail-closed policy failing open at the call site.
    def _refuse(self, primitive: str) -> Any:
        raise AuthorizationRequiredError(
            primitive,
            self._deny().public().denial_reason,
        )

    def select_collection(self, target_scope: TargetScope) -> Any:
        return self._refuse("select_collection")

    def count_candidates(self, candidates: Sequence[Any]) -> int:
        return self._refuse("count_candidates")

    def rank_candidates(self, candidates: Sequence[Any]) -> Sequence[Any]:
        return self._refuse("rank_candidates")

    def hydrate_targets(self, targets: Sequence[Any]) -> Sequence[Any]:
        return self._refuse("hydrate_targets")

    def issue_handle(self, target: Any) -> Any:
        return self._refuse("issue_handle")
