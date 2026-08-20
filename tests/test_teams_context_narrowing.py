"""The validated context's NARROWING is enforced, not merely recorded.

`resolve_policy` validates exactly one context -- expiry, revisions, principal,
delegation -- and builds `TeamsPolicy` for it. Three ways that validated
authority used to be ignored:

* the `operation` argument was never compared against anything, so a context
  narrowed to `operation:read` still reached `Decision.allow()` for a write;
* `resolve_authorized_targets` added the owner stamp and nothing else, so a
  context narrowed to one collection kept whatever corpus the caller asked for;
* every protocol method preferred the CALLER's context over the bound one, so a
  stale or differently scoped context could determine the principal and pass
  through a policy created for another request.

None of these are call-site defects. The policy is the thing that decides, and
it decided without looking.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from hermes_lcm.access_context.denials import DenialReason
from hermes_lcm.access_context.model import AccessContextV1
from hermes_lcm.access_policy import TeamsPolicy


def _context(
    principal: str = "acorn",
    *,
    grants: frozenset[str] | set[str] | None = None,
    narrowing: frozenset[str] | set[str] | None = None,
) -> AccessContextV1:
    now = datetime.now(timezone.utc)
    return AccessContextV1.from_host(
        authenticated_transport="test",
        context_id=f"ctx-{principal}",
        request_id=f"req-{principal}",
        source_kind="human",
        deployment_id="dep",
        tenant_id="tenant",
        principal_id=principal,
        profile_id=principal,
        profile_incarnation="inc",
        session_id=f"session-{principal}",
        session_owner_principal_id=principal,
        conversation_id="conv",
        conversation_lane="lane",
        default_write_collection_id="collection-a",
        read_policy_ref="policy",
        lease_id="lease",
        grants=frozenset(grants or ()),
        narrowing=frozenset(narrowing or ()),
        issued_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(hours=1),
    )


# --- the operation allowlist ----------------------------------------------


def test_a_read_only_context_may_not_write() -> None:
    """The finding: only `admin` was denied; `operation` itself was ignored."""
    policy = TeamsPolicy(_context(grants={"read"}))

    decision = policy.authorize_operation(None, "write", {"kind": "ingest"})

    assert not decision.allowed
    assert decision.denial_reason is DenialReason.SCOPE_FORBIDDEN


def test_a_read_only_context_may_not_write_via_required_scope() -> None:
    """`resolve_policy` validates without a requested scope, so this is the
    only place the gate site's `required_scope` is compared to the grants."""
    policy = TeamsPolicy(_context(grants={"read"}))

    decision = policy.authorize_operation(
        None, "ingest", {"kind": "ingest", "required_scope": "write"}
    )

    assert not decision.allowed


def test_an_explicit_operation_narrowing_removes_a_granted_operation() -> None:
    policy = TeamsPolicy(
        _context(grants={"read", "write"}, narrowing={"operation:read"})
    )

    assert not policy.authorize_operation(None, "write", {}).allowed
    assert policy.authorize_operation(None, "read", {}).allowed


def test_a_read_write_context_still_writes() -> None:
    """POSITIVE CONTROL. Denying every write also passes the three above."""
    policy = TeamsPolicy(_context(grants={"read", "write"}))

    assert policy.authorize_operation(None, "write", {"kind": "ingest"}).allowed


def test_owner_only_is_not_measured_against_the_operation_allowlist() -> None:
    """The conflation that once denied principal A its own session load.

    `owner_only` is an AUTHORITY word a gate site asks for, not an operation a
    host grants -- no principal holds it, so measuring it against the allowlist
    denies every principal its own session reset. It is decided by kind and by
    target ownership instead.
    """
    context = _context(grants={"read", "write"})
    policy = TeamsPolicy(context)

    decision = policy.authorize_operation(
        None,
        "owner_only",
        {
            "kind": "session_reset",
            "session_id": context.session_id,
            "required_scope": "owner_only",
        },
    )

    assert decision.allowed


def test_a_context_with_no_grants_declares_no_restriction() -> None:
    """An unmanaged carrier issues no operation vocabulary at all.

    Pinned so the empty case is a recorded decision rather than an accident:
    turning empty into deny-all is a call-site-slice question about what hosts
    must populate, and would deny every operation on every context today.
    """
    assert TeamsPolicy(_context()).authorize_operation(None, "write", {}).allowed


# --- the collection allowlist ---------------------------------------------


def test_a_narrowed_context_resolves_only_its_collections() -> None:
    policy = TeamsPolicy(
        _context(narrowing={"collection:collection-a"})
    )

    resolved = policy.resolve_authorized_targets(None, "read", {"session_scope": "all"})

    assert resolved["collection_allowlist"] == ("collection-a",)
    assert resolved["access_scope"] == "acorn"


def test_a_broader_caller_narrowing_cannot_widen_the_context() -> None:
    """A caller asking for more collections gets the intersection, not the union."""
    policy = TeamsPolicy(_context(narrowing={"collection:collection-a"}))

    resolved = policy.resolve_authorized_targets(
        None, "read", {"collection_allowlist": ("collection-a", "collection-b")}
    )

    assert resolved["collection_allowlist"] == ("collection-a",)


def test_a_disjoint_caller_narrowing_resolves_to_deny_all_explicitly() -> None:
    """SET, never omitted: a consumer reading `.get(key, caller_value)` keeps
    the caller's value when the key is absent, which is the leak not the fix."""
    policy = TeamsPolicy(_context(narrowing={"collection:collection-a"}))

    resolved = policy.resolve_authorized_targets(
        None, "read", {"collection_allowlist": ("collection-b",)}
    )

    assert resolved["collection_allowlist"] == ()


def test_an_unnarrowed_context_leaves_the_collection_dimension_alone() -> None:
    """Empty means UNRESTRICTED in the collection dimension -- do not invent a
    predicate that matches nothing (the defect test_teams_owner_predicate pins)."""
    resolved = TeamsPolicy(_context()).resolve_authorized_targets(
        None, "read", {"session_scope": "all"}
    )

    assert "collection_allowlist" not in resolved


def test_an_operation_naming_a_collection_outside_the_narrowing_is_denied() -> None:
    policy = TeamsPolicy(_context(narrowing={"collection:collection-a"}))

    assert not policy.authorize_operation(
        None, "read", {"collection_id": "collection-b"}
    ).allowed
    assert policy.authorize_operation(
        None, "read", {"collection_id": "collection-a"}
    ).allowed


# --- the bound context -----------------------------------------------------


def test_a_caller_supplied_context_cannot_replace_the_validated_one() -> None:
    """The substitution: a stale context could determine the principal."""
    policy = TeamsPolicy(_context("acorn"))

    decision = policy.authorize_operation(_context("carus"), "read", {})

    assert not decision.allowed
    assert decision.denial_reason is DenialReason.CONTEXT_INVALID


def test_the_same_context_by_value_is_still_the_bound_context() -> None:
    """POSITIVE CONTROL: a faithful reconstruction is the same authority."""
    context = _context("acorn")
    policy = TeamsPolicy(context)
    rebuilt = dataclasses.replace(context)

    assert rebuilt is not context
    assert policy.authorize_operation(rebuilt, "read", {}).allowed


def test_a_substituted_context_cannot_re_authorize_a_stored_row() -> None:
    policy = TeamsPolicy(_context("acorn"))

    decision = policy.authorize_stored_scope(
        _context("carus"), "read", {"access_scope": "carus"}
    )

    assert not decision.allowed
    assert decision.denial_reason is DenialReason.CONTEXT_INVALID


def test_a_substituted_context_resolves_to_no_corpus() -> None:
    """Explicitly emptied rather than returned unchanged."""
    policy = TeamsPolicy(_context("acorn"))

    resolved = policy.resolve_authorized_targets(
        _context("carus"), "read", {"session_scope": "all"}
    )

    assert resolved["access_scope"] == ""
    assert resolved["collection_allowlist"] == ()


@pytest.mark.parametrize("operation", ["read", "write"])
def test_an_unbound_policy_refuses_a_supplied_context(operation: str) -> None:
    """Nothing validated it, so nothing may be decided from it."""
    decision = TeamsPolicy(None).authorize_operation(_context(), operation, {})

    assert not decision.allowed
    assert decision.denial_reason is DenialReason.CONTEXT_INVALID
