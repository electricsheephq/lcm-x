"""DRAFT per-principal policy — the experiment the plan calls for first.

Purpose of this draft is to answer ONE question: do the two levers that already
exist -- per-row ``authorize_stored_scope`` and a ``resolve_authorized_targets``
that overrides rather than passes through -- cover every leak probe in the
isolation smoke, or does some read path need an ``access_scope`` predicate added
to its query?

Not the finished policy. Membership, shared collections and delegation all
resolve from the catalog in the real one; this draft decides from the context
alone, which is enough to find out which probes still leak.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..access_context.denials import Decision, DenialReason, PublicDecision
from ..access_context.model import AccessContextV1
from ..access_context.protocols import TargetScope


# Operations that touch the whole store rather than one principal's targets.
# A backup copies every principal's memory into one file, so it is an
# administrative capability the connector holds, not something a principal
# inherits by being the only one logged in.
_STORE_WIDE_KINDS = frozenset({"backup"})


def principal_of(context: AccessContextV1 | None) -> str:
    """The owner scope a row would be stamped with for this context.

    Must match engine._access_scope_for_storage_session exactly, or reads will
    disagree with writes and the store will look isolated while being broken.
    """

    if context is None:
        return ""
    return str(context.session_owner_principal_id or context.principal_id or "")


class TeamsPolicy:
    """Scope every operation to the acting principal."""

    def __init__(
        self,
        context: AccessContextV1 | None,
        audit_sink: "Callable[..., None] | None" = None,
    ) -> None:
        self._context = context
        self._audit_sink = audit_sink
        self.teams_enabled = True

    # -- authorization ----------------------------------------------------

    def authorize_operation(
        self,
        context: AccessContextV1 | None,
        operation: str,
        expected_scope: TargetScope,
    ) -> Decision:
        effective = context if context is not None else self._context
        principal = principal_of(effective)
        if not principal:
            return Decision.deny(DenialReason.CONTEXT_INVALID)

        # Store-WIDE operations are not a principal's to perform. `backup_database`
        # copies the entire file, every principal's memory included, so under
        # Teams no principal holds it -- not even the one who happens to be
        # first. It belongs to the connector, which #497 already gives the
        # backup/migration/audit families and which authenticates separately.
        #
        # This is the half of `owner_only` that is store-wide admin. The other
        # half -- owner OF THE TARGET -- is the session checks below. Treating
        # them as one authority is what made a policy that enforced the natural
        # `required <= operation_allowlist` rule deny principal A its own
        # session load: neither principal holds `owner_only`, and no fixture
        # tweak makes that coherent.
        if str(expected_scope.get("kind") or "") in _STORE_WIDE_KINDS:
            return Decision.deny(DenialReason.SCOPE_FORBIDDEN)

        # Owner OF THE TARGET. A session belonging to someone else is refused
        # before any row is read. The key differs by path: most carry
        # `session_id`, compression rollover carries `source_session_id`, and
        # rollup scheduling carries the session as `partition_key`.
        for key in ("session_id", "source_session_id", "partition_key"):
            target = expected_scope.get(key)
            if (
                target
                and effective is not None
                and str(target) != str(effective.session_id)
            ):
                return Decision.deny(DenialReason.SCOPE_FORBIDDEN)

        # Raw identifiers (store_id, node_id) name a row without saying who owns
        # it. The engine resolves the stored owner before authorizing precisely
        # so this comparison is possible; without it, expanding another
        # principal's message by id was allowed and returned their content.
        for owner in expected_scope.get("target_access_scopes") or ():
            if str(owner) != principal:
                return Decision.deny(DenialReason.SCOPE_FORBIDDEN)

        return Decision.allow()

    def resolve_authorized_targets(
        self,
        context: AccessContextV1 | None,
        operation: str,
        requested_narrowing: TargetScope,
    ) -> TargetScope:
        """Replace the caller's corpus with the principal's own.

        Explicitly SET rather than omit: run_knn reads this with
        ``authorized_scope.get("source", source)``, so omitting the key keeps
        the caller's value -- which is the leak, not the fix.
        """

        effective = context if context is not None else self._context
        narrowed = dict(requested_narrowing)
        if effective is not None:
            collection = str(effective.default_write_collection_id or "")
            if collection and "source" in narrowed:
                narrowed["source"] = collection
        return narrowed

    def authorize_stored_scope(
        self,
        context: AccessContextV1 | None,
        operation: str,
        stored_scope: TargetScope,
    ) -> Decision:
        effective = context if context is not None else self._context
        principal = principal_of(effective)
        if not principal:
            return Decision.deny(DenialReason.CONTEXT_INVALID)

        stored = stored_scope.get("access_scope")
        if stored is None:
            # Legacy, unstamped. Allowed for now so a partially-migrated store
            # is not bricked; the real policy has to decide this deliberately.
            return Decision.allow()
        if str(stored) != principal:
            return Decision.deny(DenialReason.SCOPE_FORBIDDEN)
        return Decision.allow()

    def audit_decision(
        self,
        context: AccessContextV1 | None,
        operation: str,
        internal_reason: DenialReason | None,
        public_result: PublicDecision,
    ) -> None:
        """Record what was decided, without recording why in internal terms.

        Volume is deliberately bounded. Every denial is recorded, but only
        NON-READ allows are: `audit_decision` has 39 call sites and reads
        dominate them, so a row per authorization would put an INSERT in the
        hot path of every retrieval. This branch already cost this branch a
        48s->174s regression once, from far less.

        The reason written out is ``public_result``'s, never
        ``internal_reason`` -- see record_audit_event for why that matters when
        an audit export leaves the store.
        """

        if self._audit_sink is None:
            return None
        allowed = bool(getattr(public_result, "allowed", False))
        if allowed and str(operation) == "read":
            return None
        effective = context if context is not None else self._context
        reason = getattr(public_result, "denial_reason", None)
        self._audit_sink(
            tenant_id=str(getattr(effective, "tenant_id", "") or ""),
            principal_id=principal_of(effective),
            operation=str(operation),
            allowed=allowed,
            denial_reason=getattr(reason, "value", None) if reason else None,
        )
        return None

    # -- disclosure primitives (no production call sites; protocol only) ---

    def select_collection(self, target_scope: TargetScope) -> Any:
        return target_scope

    def count_candidates(self, candidates: Sequence[Any]) -> int:
        return len(candidates)

    def rank_candidates(self, candidates: Sequence[Any]) -> Sequence[Any]:
        return candidates

    def hydrate_targets(self, targets: Sequence[Any]) -> Sequence[Any]:
        return targets

    def issue_handle(self, target: Any) -> Any:
        return target
