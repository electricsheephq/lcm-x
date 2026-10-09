# Recall tools

Use recall tools when the answer depends on historical evidence that may have been compacted or lives in another LCM session.

## Current compacted conversation

### `lcm_grep`

Use for discovery across current-session raw messages and summary nodes.

- `query` is FTS5 text by default; it is not a regex.
- Indexed history full-text queries AND terms by default; uppercase OR/NOT, quoted phrases, and term* prefixes (2+ alphanumeric characters) are honoured.
- Indexed history zero hits with multiple terms retry once with any-term matching; `retried='any_term'` reports this. Top-level OR/NOT queries and CJK/emoji substring searches do not retry.
- `content_scope='externalized'` matches the literal query text; operators and retry do not apply.
- Keep `sort='recency'` for recent events, use `sort='relevance'` for the strongest older match, and use `sort='hybrid'` when both matter.
- `mode='semantic'` or `'hybrid'` is useful when embeddings are configured; degraded coverage is reported.
- Broader `session_scope='all'|'session'` is explicit, bounded, raw-message-only archive recovery inside `lcm.db`.
- Exact role/time/source/conversation filters apply before limiting where supported.
- If the payload carries `message_search_error`, the raw-message stage crashed and its
  results are missing: treat any returned results as INCOMPLETE (they come from the other
  stages only), never as proof of absence. Retry once, or fall back to `lcm_recall` /
  narrower filters, and disclose the gap if the answer depends on raw-message coverage.
  (`lcm_recall` already folds the same failure into `coverage.fts: 'none'`.)

Do not treat a short search snippet as sufficient evidence for a detail-heavy answer.

### `lcm_describe`

Use for inexpensive inspection of a known current-session summary node or externalized payload reference. With no handle it returns a current-session DAG overview. It is a planning step, not broad discovery.

A summary node produced by a model call also reports `escalation_level` (1 = full summary, 2 = compact summary, 3 = deterministic truncation of the source) and `model` (the route that answered; empty = the host default route, `deterministic` = level 3). Nodes written before this record existed report neither. A level-3 node is a lossy stand-in: prefer `lcm_expand` on it before relying on its text.

### `lcm_expand_query`

Use when current-session compacted material must be expanded and synthesized into a precise bounded answer.

- Always provide `prompt`.
- Provide either a small `query` or explicit `node_ids` when known.
- `query` ANDs terms and supports quoted phrases; prefer 1-3 distinctive terms.
- The expansion path is model-backed and bounded by answer/context token limits.

Recommended current-session escalation:

1. `lcm_grep` to locate relevant material.
2. `lcm_describe` when a known handle needs inspection.
3. `lcm_expand_query` when exact detail was compressed away.

### `lcm_expand`

Use as low-level drill-down after a known handle:

- `node_id` expands a current-session summary with source pagination;
- `store_id` recovers one raw message and works across LCM sessions;
- `externalized_ref` opens a payload of the current session, or of a session it replaced at a compression-boundary rotation, with content pagination.

Do not use it as broad first-step discovery.

Set `include_exact_ref=true` with `store_id` when the recovered slice will be
cited or passed to `lcm_evidence_pack`/`lcm_compile_evidence`/`lcm_compute`. The default remains off
for byte compatibility.

## Cross-conversation memory

### `lcm_recall`

Use for semantic discovery across all conversations stored in the local LCM database. It fuses raw full-text, summary-vector, and verbatim-chunk arms when available and degrades honestly when embeddings are unavailable.

- `scope_bias` and recency are ranking boosts, never hard filters.
- `include` selects all, summary, or verbatim hits.
- `detail='snippets'` is the default.
- `detail='answer_ready'` adds bounded per-session diversity and exact-ref hydration.
- In both details, a verbatim message hit carries `event_time` (ISO 8601 UTC, e.g. `2024-03-20T13:45:30Z`) when the host recorded when the message happened, plus `event_time_source` when known. A hit without `event_time` has no recorded event time (for example imported history): do not read its `timestamp` as one, since that is when LCM stored the row. Summary hits carry neither field. If the response has `event_time_unavailable: true`, the event time could not be read (`event_time_unavailable_reason` is `deadline`, `read_error`, or `response_cap` when the fields were dropped to fit the response size cap), so a missing `event_time` there says nothing about the row.
- With embeddings on, the full-text arm's best match is kept in the ranked window (`LCM_RECALL_FTS_ANCHOR`, default `true`). `provenance.fts_anchor = {fired, position, delivered}` reports whether it moved, its 1-based slot, and whether it is among the returned hits (`null` when it did not fire; `answer_ready` citation and per-session rules and the response cap can still drop it).
- Follow each result's `expand_hint`: verbatim hits use `lcm_expand(store_id=...)`, current-session summaries use `lcm_expand(node_id=...)`, and cross-session summaries use `lcm_expand(node_id=..., session_id=...)`. Reach for `lcm_load_session` when the whole transcript is wanted rather than one node.

### `lcm_load_session`

Use after a session ID is known. It enumerates raw rows in chronological `store_id` pages; it is not search.

- Continue with `after_store_id` from `next_cursor`.
- Increase `max_content_chars` only within its hard bound.
- Recover a truncated individual row with `lcm_expand(store_id=..., content_offset=...)`.
- Set `include_exact_ref=true` when transcript rows will feed exact evidence or computation.

Use Hermes `session_search` for host-tracked sessions that are not present in `lcm.db`.

## Time-bounded recall

### `lcm_recent`

Use for `today`, `yesterday`, `Nd`, `week`, `month`, `date:YYYY-MM-DD`, or `last Nh`. Periods are UTC. Ready rollups are used only when they cover the entire requested window; otherwise the tool falls back to bounded summaries and reports provenance.

For exact raw-message windows, use `lcm_grep` with explicit `time_from`/`time_to` instead.

## Exact evidence and computation

### `lcm_compile_evidence`

Use when a question needs multiple named facets, exact operands, conflict handling, or latest-state reasoning. Supply one bounded semantic proposal over already retrieved exact refs. When deterministic parsing yields only the catch-all `answer` facet, the proposal may provide up to 12 unique generic `requested_facets`; otherwise those facets can only extend, never erase, deterministic requirements. The proposal is only a hint: every requested facet must be closed by product-validated exact evidence, and product code revalidates every ref, unique quote span, entity, date, value, unit, distinct key, role, and source. Use `answer_sufficient` for a specifically answered open-ended question; require `finite_coverage` or `computation_sufficient` for exhaustive counts, lists, or arithmetic. On `partial`, `conflicted`, or `unknown`, preserve uncertainty and the ordinary answer path.

### `lcm_evidence_pack`

Use after bounded baseline refs exist. It validates/hydrates exact refs, keeps occurrence and observation time distinct, deduplicates, and may return a canonical computation trace. It returns evidence, not final prose. Open-cardinality stays partial unless coverage is product-verifiable.

### `lcm_compute`

Use only over exact cited evidence validated by the compiler for supported date intervals/filters, distinct counts, compatible-unit sums, directed or absolute differences, ordering, and latest-state selection. Invalid spans, mixed units, ambiguity, or unsupported closure fail closed.

## Default-off advanced paths

- `lcm_query_state` is hidden until `LCM_ASSERTIONS_ENABLED=true`, then queries the same-database assertion sidecar.
- `lcm_retrieve` is hidden until `LCM_ADAPTIVE_RETRIEVAL_ENABLED=true`. It is the default-off bounded adaptive controller, is not required for ordinary recall, and must not replace the stable workflow above without measured benefit.

Both host paths advertise the other 13 schemas by default. `LCM_DISABLED_TOOLS`
hides any named tool, including a flag-enabled one. Cached dormant calls still
return their existing `status: disabled` responses.

## Operator tools

`lcm_status`, `lcm_inspect`, and `lcm_doctor` report health and metadata. They do not replace content retrieval.
