# AccessContextV1 contract
`AccessContextV1` is the data-and-validation contract for the Hermes Agent
host carrier and the Hermes-LCM authorization seam. This document defines
data, validation order, denial projection, protocols, shared JSON vectors,
and the authority-path inventory.

Landing sequence (the Teams extraction train): this contract lands FIRST and
is initially unconsumed — no engine import, no feature flag, no enforcement,
no tool change. Enforcement arrives in later slices: the policy/catalog layer
(`access_policy/`, `teams/`, per-item scope storage), then read-path hooks,
then write/admin hooks; each slice re-states this boundary in its own PR. The
authority-path inventory below records, per path, the hook site that the
wiring slices are obligated to install — the completeness test enforces that
obligation once the hooks land.

## Authority and lifetime

The immutable context is derived from authenticated host transport and session
state. Profile, session, conversation, collection, cursor, reference, and
request IDs carry lineage; none grants authority by itself. `narrow()` and
`derive_child()` return new contexts and reject any attempted widening.
Delegation intersects operation, collection, audience/binding, expiry, and
current revision boundaries. The complete delegation chain and narrowing set
are retained for re-delegation inspection.

Four rules make "reject any widening" total:

- **Empty is unrestricted, so narrowing to empty is widening.** A collection
  allowlist or audience with no members is read as "any" by every consumer, so
  `narrow(collections=[])` and `narrow(audience=[])` raise `ScopeMismatchError`
  rather than erase the parent's restriction, and a candidate that dropped a
  restricted dimension is not a subset. (Operations are the exception and need
  no such rule: the operation allowlist falls back to `grants`, where empty
  really does mean deny-all.)
- **A narrowing token is a restriction, never a grant.** The effective
  operation allowlist is the INTERSECTION of `grants` and the `operation:*`
  tokens; a token outside the grants is a claim of authority, so `from_host()`
  and `from_payload()` reject it and `validate()` denies `context_invalid`.
- **Sibling delegations have distinct identities.** `derive_child()` mints a
  per-derivation nonce into `context_id` and `request_id`, because revocation
  is keyed by `context_id`: depth-derived IDs made two independent children of
  one parent revoke each other. Callers needing a stable identity pass
  `child_context_id`/`child_request_id`.
- **Subset proofs compare effective bounds and provenance.** `is_subset_of()`
  compares each effective dimension rather than demanding the parent's raw
  token set survive verbatim (a superseded `operation:write` is correctly
  dropped by a read-only child), and it verifies that a delegated candidate's
  `delegated_by` is the final entry of its own chain.

Validation is deterministic and injected-time only:

`CONTEXT_PRESENT → CONTEXT_WELL_FORMED → REVISION_SUPPORTED → NOT_EXPIRED → NOT_REVOKED → OWNERSHIP_CURRENT → LEASE_CURRENT → SCOPE_PERMITTED → TARGET_RESOLUTION`

Authorization is therefore before collection selection, existence/count,
ranking, hydration, and handle issuance. The consumer protocol exposes those
disclosure primitives so a seam conformance test can record that order.

The authority-path inventory includes the public `lcm_*` tools and the
non-tool paths that can bypass those handlers: store/compaction/rollup and
sidecar writes; maintenance, import, schema, and diagnostics; retrieval and
expansion; auxiliary/lifecycle session state; and host callbacks in
`engine.py`. The `cron` category is represented by the real
`_RollupMaintenanceScheduler`; this repository has no separate OS cron entry
point, so the scheduler note is the honest boundary rather than an invented
function.

## Denials

The internal `Decision` preserves the exact `DenialReason` and a content-free
detail mapping containing IDs or revisions only. `PUBLIC_DENIAL_PROJECTION` is
the single disclosure table: scope, ownership, lease, and target denials are
projected as `target_not_found_or_forbidden`; context lifecycle denials remain
typed.

## Standard single-user compatibility

The contract has no runtime call sites and is default-off by construction. The
carrier matrix is deliberately explicit:

| Host carrier | Teams | Mode |
| --- | --- | --- |
| absent (`None`) | disabled | `STANDARD_UNMANAGED`; standard single-user behavior is entirely unchanged and the contract is not consulted |
| absent (`None`) | enabled | `FAIL_CLOSED`; every Teams-governed path returns `context_missing`, with no unscoped fallback |
| present | disabled | `STANDARD_UNMANAGED`; the context is ignored and carrying one cannot enable Teams |
| present | enabled | `ENFORCING`; normal validation applies |

No context-var, thread-local, crypto, token verifier, policy DSL, collection
catalog, audit store, retry framework, or other hardening is part of this
contract. Enforcement and policy remain the explicitly named follow-up seams.
