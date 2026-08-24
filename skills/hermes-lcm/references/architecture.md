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

## Cloud embedding privacy boundary

Known cloud embedding paths transform provider input without rewriting durable
source. Before warmup, document backfill, or semantic-query dispatch, LCM
requires an enabled, nonempty, recognized sensitive-pattern policy; canonicalizes
existing placeholders; replaces matches with pattern-only placeholders; scans
for residual matches; and fails closed before transport on any invalid state.

Vector identity binds provider, model, dimension, storage shape, and the active
privacy revision. A policy change requires a new warmup/identity rather than
mixing vectors. Evidence and status expose aggregate policy state, never matched
content.

Cloud raw-chunk embedding additionally requires explicit raw-text consent because
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
