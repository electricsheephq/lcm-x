# Architecture

LCM-X keeps raw messages in profile-local SQLite and builds a summary DAG to keep active context bounded.

Released baseline: `v0.23.1@81d8d41197dddc4c09b57097f4955ebae32366a9`.
Stable runtime identity and a later main checkout are separate proof planes.

## Core flow

1. The active context engine ingests messages into `lcm.db`.
2. Older eligible messages are compacted into leaf summaries.
3. Summary nodes can be condensed to higher DAG depths.
4. Context assembly combines selected summaries with a protected fresh raw tail.
5. Recall tools recover exact source rows or bounded expanded context when summaries are insufficient.

Raw messages are source truth. Summary nodes, embeddings, temporal rollups, query views, and assertions are derived and rebuildable layers with explicit provenance.

## LCM note carrier

On Hermes, the LCM note appears in the `lcm-x` plugin system-prompt section after
memory, only for a session served by an LCM-X engine. Its text is byte-stable and
conditional because it appears before the first compaction too. It explains that
LCM summaries are untrusted history and names the retrieval tools, including
`lcm_expand(externalized_ref="R")` for externalized tool output. The section never
enters the message store. Older hosts without the section API keep the existing
in-band note path: when a host passes a leading system row, assembly appends the
note only if that row does not already contain the LCM phrase.

## Cloud embedding privacy boundary

Known cloud embedding paths transform provider input without rewriting durable source. That
protection resolves ON automatically for recognized cloud providers and is independent of
durable redaction: `LCM_SENSITIVE_PATTERNS_ENABLED` (default off) governs the durable store
only, and the durable store is lossless by default. While provider-copy privacy is on, LCM
requires a nonempty recognized `LCM_SENSITIVE_PATTERNS` catalog; canonicalizes existing
placeholders; replaces matches with pattern-only placeholders; scans for residual matches;
and fails closed before transport on any invalid state. `LCM_EMBEDDING_PRIVACY_ENABLED=false`
is an explicit opt-out that dispatches raw input under the `privacy:off` revision. Optional
Voyage reranking is covered by the same resolution. A privacy-policy error on the
`lcm_recall` path (an invalid catalog, a residual match, dispatch-time drift) raises; it is
never a silent degrade to full-text. A stale vector identity is different: when no active
cloud profile matches, or the stored vectors carry an older privacy revision, `lcm_recall`
answers from its full-text arm (for every `include`) with `degraded=true` and an
`embedding_identity_stale:` reason, sends no query to the stale arm, and reads none of its
vectors. The remedy is `/lcm embed warmup`, then `/lcm embed backfill --apply`.

With embeddings on and a registered profile, new summaries are embedded by a bounded
background pass after each published summary and on session bind; new message chunks are
embedded the same way only with a local provider, because cloud chunk consent
(`--confirm-raw-text`) is per invocation. The pass shares the backfill lease and skips while a
manual backfill holds it. `/lcm status` reports the backlog per corpus and the last pass.

Vector identity binds provider, model, dimension, storage shape, and the active
privacy revision. A policy change requires a new warmup/identity rather than
mixing vectors. Evidence and status expose aggregate policy state, never matched
content.

Cloud raw-chunk embedding requires explicit raw-text consent in addition to whatever privacy
posture is active because
chunks derive from verbatim message/tool content. The pattern gate is not a
general classifier. `openai-compatible` is conservatively cloud-gated; Ollama
endpoint locality must not be inferred solely from the provider name (#337).

## Scope model

- Current-session DAG operations use the active engine/session binding.
- `lcm_recall` searches all conversations already stored in the local LCM database.
- `lcm_load_session` enumerates a known LCM session.
- Hermes `session_search` covers host-tracked history outside `lcm.db`.

Do not silently treat those stores or scopes as interchangeable.

## V4 derived state

The V4 branch adds same-database, default-off assertion/query-view state and provider-neutral reasoning/evidence components. They remain subordinate to raw messages:

- assertions require exact message IDs, spans, quotes, and lifecycle provenance;
- query views cache evidence dependencies and coverage, never final prose;
- computation validates exact operands and emits an immutable trace;
- evidence packs return bounded evidence/computation, not an authoritative answer.

Unknown source/event time and unresolved conflict are valid states. Derived data must fail closed rather than manufacture certainty.
