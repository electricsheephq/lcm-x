# Opt-in async/background compaction with atomic publish

Design spike for preparing old stable chunks off the turn-critical path while keeping current LCM behavior unchanged unless explicitly enabled.

Refs:

- Lossless Claw #807 — prepare incremental summaries in the background and publish atomically
- Lossless Claw #942 — live config must beat stale persisted thresholds/debt
- Lossless Claw #902 — summary failure/backoff must not wedge compaction debt forever

## Problem

Today `LCMEngine.compress()` does the expensive work synchronously: ingest, select the oldest raw backlog outside the fresh tail, call the summarizer, write canonical DAG nodes, optionally condense, then assemble the active context. That preserves the important cache-friendly property: active context changes only at threshold/full-sweep boundaries. The downside is foreground latency, especially with slow local summarizers or serial leaf chains.

The safe target is not “write summaries in another thread and flip a boolean.” The target is a two-phase lifecycle:

1. **Prepare** non-canonical summaries for old, stable raw-message chunks in the background.
2. **Promote** a complete, still-valid batch atomically when normal foreground compaction would have run.

Until promotion, active context, search, recall, expansion, transcript GC, and doctor integrity checks behave as if the prepared summaries do not exist.

## Non-goals for the first slice

- No default behavior change.
- No mandatory background thread in normal installs.
- No prebuilt condensation layers in the first slice. Leaf-only promotion is already useful and is easier to prove atomic.
- No replacement for foreground compaction. If prepared work is absent, incomplete, stale, or invalid, foreground compaction falls back to today’s path.
- No persisted threshold override that can win over live config. Persisted metadata is evidence to validate against live policy, not policy itself.

## v0.28.0 decisions (2026-10-08)

This section records the design chosen for v0.28.0 "invisible compaction" (#787). It overrides the sections below
where they differ. The storage model, fingerprints, reader rules and test matrix stand. The flag table, preparation
step 4, the promotion transaction, the implementation sequence and the first two open questions are revised here.

**Goal and bar.** The turn that crosses the compaction threshold waits only to publish summaries that are already
written, not to write them. The GA bar is a visible-wait p90 of at most 5 s on the ≥ 24 h reference-agent soak, over at
least 10 events, with no compaction slower than on the previous GA. Visible wait is the wall time from LCM `compress()`
entry to return, for every `compress()` the host or the gateway's hygiene pass runs while a user turn waits. It does
not include host commit overhead (system-prompt rebuild, session end and start).

**Prepare in the background; promote inside `compress()`.** Every leaf publication and frontier move stays on the
thread that already owns it: the host's `compress()` call, or hygiene's. (Condensation, which moves no frontier, is
the one canonical write that may leave the turn; see Condensation below.) An off-turn worker that publishes canonical
leaves right after the turn is rejected for five reasons:
- assembly, FTS, rollups, condensation selection, store-complete and replay-drop read a canonical node at once;
- session end finalizes with the in-process frontier (`_last_compacted_store_id`), which an off-turn writer would
  have to set on the engine the host calls;
- hygiene runs on a fresh clone with fresh in-memory state;
- `on_session_end` waits on `_stable_use_lock` with no timeout;
- condensation publication has no fence (#988).

**Batches and promotion.**
- A batch is one leaf. Promotion publishes the longest chain of consecutive ready batches that starts at the live
  frontier, so the "prefix promotion" question below resolves to whole leaves in order.
- Promotion reuses today's fenced leaf publication: `dag.add_node(before_commit=…)` with a callback that runs
  `stage_compaction_publication` and marks the batch promoted, in one transaction on the DAG connection. No new
  publish path decides frontier contiguity, session binding or "already claimed". In-process markers update after
  the commit, and every post-commit hook a foreground leaf runs runs for each promoted leaf: rollup invalidation
  (`_invalidate_rollups_for_published_node`) and transcript GC (`_maybe_gc_compacted_tool_results`) today. Promotion
  shares those steps with the foreground leaf through one helper, so a hook added later cannot be skipped.
- Promotion runs at the top of `_compress_impl`, before the no-progress hold's cleanup-only return and before pre-leaf
  condensation. A ready chain at the live frontier is progress that costs no model call, so it lifts the hold, but
  only when the live prompt meets a normal compression trigger (the threshold or the full sweep): `should_compress`
  then reports true despite the hold. Below a trigger nothing is promoted early, so the active prompt and its cache
  prefix change only at the boundaries they change at today. If the estimate is then below the #671 target,
  `min(T − leaf_chunk, 0.95·T)`, the pass makes no foreground summariser call at all:
  no remainder leaf, no pre-leaf or post-drain condensation. Otherwise today's path runs for the gap.
- Each batch is checked inside the transaction. Any failure marks it `rejected` with a reason and the fallback
  continues. The checks are those listed under "Atomic promotion" below, plus a host-row check: the host rows from
  the candidate start must map contiguously into the batch's stored range (`_get_store_id_map_for_messages`) and be
  consumed exactly as a foreground leaf consumes them, dependent replies included. Otherwise the reason is
  `host_list_mismatch`. Without this check a promoted leaf would leave the host list unchanged (the hidden-only path,
  #904) and start a no-progress hold.
- Promotion stops at the first rejected batch. Batches made stale by a foreground leaf become `superseded`.
- Stub-first exits promote before they return.

**What a prepared leaf carries.**
- The summary route the foreground uses (prompt v1 today, #660). The route fingerprint covers every setting the
  foreground leaf call reads: model and fallbacks, `summary_prompt_version`, `summary_reasoning_effort`, and the leaf
  output budget (`leaf_target_ratio`, `leaf_target_min_tokens`, `leaf_target_max_tokens`). A batch prepared under any
  other value is rejected at promotion. S3's test derives the list from the config fields
  `_summarize_leaf_chunk_with_rescue` reads, so a new setting cannot be missed.
- No previous-summary continuity context, because the foreground leaf passes none. This replaces preparation step 4.
  The worker records the `focus_topic` it used on the batch; a mismatch with the live list is not a rejection reason.
  Track S at the v0.28.0 rc1 judges any quality effect.
- The producing model and escalation level, stored on the prepared node (`summary_model`, `escalation_level` in
  `pending_summary_nodes` below). Promotion copies both into `summary_node_provenance`, so a restart or a hygiene clone
  promotes with the model that wrote the summary. The worker never writes the shared `_last_leaf_summary_model` (#441).
- Row identity hashes `timestamp`, not `observed_at`. Exclusions are recomputed with
  `_stored_publication_filter_exclusions`, and carried ranges are passed.

**The worker.**
- One daemon thread per process. Dedup is keyed by the store (its profile home and database path), the
  conversation id and the session id: one process can serve several profile homes that use the same conversation id,
  and batches are bound to a session. It runs
  each job in the captured context of the turn that scheduled it, which carries the profile's secret scope and home
  (#987). A turn-end trigger that finds a job running for the same key sets a dirty flag instead of being dropped;
  when the job ends, the worker re-reads the estimate and runs again if the flag is set.
- Batch creation is arbitrated by the database, not the process: a partial unique index on the live generation of a
  frontier (below) plus `INSERT OR IGNORE`, so two processes that open the same profile store cannot both create and
  claim a batch for one frontier. The key is the frontier alone, without the fingerprints. A job whose live
  fingerprints differ from a live batch's first supersedes that batch (a compare-and-set to `superseded`), then
  inserts its own, so stale and current batches never compete for promotion. A new session in the conversation
  supersedes the previous session's live batches the same way.
- A job is scheduled at any turn end that has eligible rows past one leaf chunk. Prepared work is sized in tokens:
  the job loops until the predicted estimate after promotion, the current estimate minus the ready chain's savings
  (per batch, the source tokens minus the rendered cost of the summary as assembly emits it, header, expand hint and
  framing included), is at or below the target with one leaf of headroom, or the worker's
  spend share runs out. Source tokens covered are not the measure: a summary only has to be shorter than its source.
  There is no fixed batch cap, so `async_background_compaction_max_batches` is dropped.
- It has its own circuit-breaker key and a bounded share of the summary spend guard (default half of
  `LCM_SUMMARY_SPEND_MAX_CALLS`), so the foreground fallback always has budget.
- The worker never takes the foreground's model-call slot. When the host's `auxiliary.compression.max_concurrency`
  is set, the worker uses at most that value minus one; at 1 it does not run, and today's foreground path serves the
  profile. Unset means the host imposes no cap: the fleet's summary providers are remote and serve concurrent
  requests. A profile whose provider serializes requests (a local model) sets `max_concurrency: 1`, which turns the
  worker off. A call already in flight cannot yield, so reserving the slot up front is the mechanism; a test starts a
  foreground compression while a slow background call is in flight and asserts the foreground call does not wait on
  it.
- It holds no transaction across a model call (a gateway exit can kill it mid-call). Its writes are single statements,
  and each state change is a compare-and-set on the state it leaves plus the claim token the job wrote when it moved
  the batch to `preparing`: completion is `SET state='ready' … WHERE batch_id=? AND state='preparing' AND
  claim_token=?`, and promotion moves only `ready` batches. Prepared-node inserts are fenced the same way: each row
  carries the claim token and is inserted only while its batch is still `preparing` under that token
  (`INSERT … SELECT … WHERE EXISTS (… state='preparing' AND claim_token=?)`), and promotion reads only rows whose token
  matches the batch's. A cleanup or rejection that ran first therefore wins, and a reclaimed job's late result is
  never published.
  The cleanup timeout for a stale `preparing` batch is longer than the worst-case prepare.

**Condensation.** Prepared condensations stay out, as the non-goals say. Condensation leaves the turn by a different
route: a canonical off-turn condenser in the same worker, under the condensation fence (#988). It is the exception to
the ownership rule above, and S6 must show it is safe before it ships. Its inputs are canonical nodes and it moves no
frontier, so session end and the hygiene clone's in-memory state do not depend on it. Its node reaches the wire only
at the next `compress()` assembly, which reads the store. If S6 cannot show this, condensation publication moves back
onto `compress()` as a promotion of a prepared condensation.

**Hygiene (#626).** Prepared batches live in SQLite, so hygiene's fresh clone promotes them like any other caller.
Hygiene's bounded hold then covers promotion plus assembly.

**Slices.** Each slice is its own PR under the cross-model review rule.
1. S0a: a per-`compress()` record (wall time, foreground summariser calls, promoted leaves, trigger), as a log line
   and `lcm_status` counters. It is on by default with no behaviour change, and it is the GA instrument.
2. S0b: a reliability cell with a fake summariser that holds 20 s per call. It records visible wait per compaction
   with prepare off and on, and prompt-prefix breaks per cycle from the fake provider's request log. It must include a
   due condensation and a remainder.
3. S1: the background-job helper that captures context per job (#987); rollups, assertion extraction and the
   pre-compaction extraction (`_run_pre_compaction_extraction`, today tied to the foreground leaf) move onto it, so a
   promoted leaf's source rows still get their configured extraction.
4. S2: the condensation fence (#988).
5. S3: the two tables through a named `lcm_migration_state` step (no `SCHEMA_VERSION` bump, no `summary_nodes` column),
   with reader filters and status/doctor counts; the flag stays off.
6. S4: the one-shot preparer and promotion, with the exit rule and the host-row check.
7. S5: the worker, scheduling, capacity and restart cleanup.
8. S6: the off-turn canonical condenser.

**Gates.**
- The S0b cell: with prepare on and a 20 s summariser, visible wait at the crossing turn is at most 5 s when the
  prepared chain covers the target. Prefix breaks per cycle are not above prepare off.
- A harness test on a fleet-shaped list (6k stubs, carriers, dependent replies) asserts both zero summariser calls and
  a shorter host list at the crossing.
- Soak counters:
  - promotion outcomes by reason;
  - hidden-only passes that had promoted leaves, which must be 0;
  - the visible-wait p90 over the bar's whole population: every host or hygiene `compress()` that runs while a user
    turn waits, cleanup-only and held passes included.
- The test matrix below, the new tests, and a green reliability nightly.
- Track S at v0.28.0 rc1 with prepare on: facts kept not worse than the previous GA.
- The flags stay off in code until these pass. Turning prepare on by default is the owner's decision at the v0.28.0
  rc, with these numbers.

## Proposed flags

Add config fields, all disabled by default:

| Field | Env | Default | Meaning |
| --- | --- | ---: | --- |
| `async_background_compaction_enabled` | `LCM_ASYNC_BACKGROUND_COMPACTION_ENABLED` | `false` | Enables the feature surface. |
| `async_background_compaction_worker_enabled` | `LCM_ASYNC_BACKGROUND_COMPACTION_WORKER_ENABLED` | `false` | Allows automatic background preparation. Tests and hosts may still call one-shot prep manually when the feature is enabled. |
| `async_background_compaction_retry_backoff_seconds` | `LCM_ASYNC_BACKGROUND_COMPACTION_RETRY_BACKOFF_SECONDS` | `300` | Cooldown after summary failures. |

The enable flag should guard all writes to the new tables and all promotion attempts. Reader filters must still be robust if old pending rows exist after the flag is later disabled.

## Storage model

Add tables separate from canonical `summary_nodes`:

### `compaction_batches`

One row per prepared generation.

Suggested columns:

- `batch_id TEXT PRIMARY KEY`
- `conversation_id TEXT NOT NULL`
- `session_id TEXT NOT NULL`
- `state TEXT NOT NULL` — `pending`, `preparing`, `ready`, `promoting`, `promoted`, `rejected`, `failed`, `superseded`
- `claim_token TEXT` — written with the move to `preparing`; every later state change compares it
- `focus_topic TEXT` — the focus topic the worker summarised with (recorded for the rc1 quality read; not a rejection
  reason)
- `frontier_start_store_id INTEGER NOT NULL`
- `frontier_end_store_id INTEGER NOT NULL`
- `fresh_tail_count INTEGER NOT NULL`
- `leaf_chunk_tokens INTEGER NOT NULL`
- `policy_fingerprint TEXT NOT NULL`
- `summary_route_fingerprint TEXT NOT NULL`
- `source_coverage_hash TEXT NOT NULL`
- `expected_leaf_count INTEGER NOT NULL`
- `prepared_leaf_count INTEGER NOT NULL DEFAULT 0`
- `failure_count INTEGER NOT NULL DEFAULT 0`
- `next_retry_at REAL`
- `last_error TEXT`
- `created_at REAL NOT NULL`
- `updated_at REAL NOT NULL`
- `promoted_at REAL`
- `rejected_reason TEXT`

Indexes:

- `(conversation_id, state, created_at)`
- `(session_id, state, created_at)`
- `(next_retry_at, state)`
- `UNIQUE (conversation_id, session_id, frontier_start_store_id) WHERE state IN ('pending', 'preparing', 'ready',
  'promoting')` — one live generation per frontier, whatever its fingerprints; creation uses `INSERT OR IGNORE` after
  superseding a live batch whose fingerprints are stale

### `pending_summary_nodes`

Prepared but non-canonical leaf summaries.

Suggested columns:

- `pending_id TEXT PRIMARY KEY`
- `batch_id TEXT NOT NULL REFERENCES compaction_batches(batch_id)`
- `conversation_id TEXT NOT NULL`
- `session_id TEXT NOT NULL`
- `depth INTEGER NOT NULL DEFAULT 0`
- `summary TEXT NOT NULL`
- `token_count INTEGER NOT NULL`
- `source_token_count INTEGER NOT NULL`
- `source_ids TEXT NOT NULL`
- `source_identity_hashes TEXT NOT NULL`
- `source_range_start_store_id INTEGER NOT NULL`
- `source_range_end_store_id INTEGER NOT NULL`
- `previous_pending_ids TEXT NOT NULL DEFAULT '[]'`
- `created_at REAL NOT NULL`
- `earliest_at REAL`
- `latest_at REAL`
- `expand_hint TEXT DEFAULT ''`
- `claim_token TEXT NOT NULL` — the claim this row was written under; promotion reads only rows matching the batch's
- `summary_model TEXT` — the model that produced this summary (copied into `summary_node_provenance` at promotion)
- `escalation_level INTEGER`

Indexes:

- `(batch_id, source_range_start_store_id)`
- `(conversation_id, state)` is intentionally on the batch table; pending nodes should not have independent canonical state.

Do **not** add pending rows to `summary_nodes`. Reusing the canonical table with a lifecycle flag would make every reader, FTS query, and integrity check a footgun. Keeping pending rows in a separate table gives active-only defaults naturally.

## Fingerprints and validation inputs

A prepared batch is valid only for the exact policy and source frontier it was created for.

### Policy fingerprint

Hash a normalized JSON object of compaction policy inputs that affect chunking or active-context semantics:

- schema/protocol version, e.g. `async_compaction_protocol_v1`
- `fresh_tail_count`
- `leaf_chunk_tokens`
- `context_threshold` / effective preflight threshold policy
- `dynamic_leaf_chunk_enabled`
- `dynamic_leaf_chunk_max`
- `ignore_message_patterns` plus their source
- sensitive-pattern config that changes stored/summarized text
- large-output externalization settings that change serializer input
- `custom_instructions`
- `l2_budget_ratio`, `l3_truncate_tokens`
- `incremental_max_depth` only if the batch later supports pending condensation

### Summary route fingerprint

Hash the effective summarizer contract:

- `summary_model`
- `summary_fallback_models`
- `summary_prompt_version`
- `summary_reasoning_effort`
- `leaf_target_ratio`, `leaf_target_min_tokens`, `leaf_target_max_tokens` (they set the leaf's output budget)
- provider/model route after parsing, if available
- summarizer timeout class only if it changes produced summaries or failure policy
- plugin version/protocol version

A model/config change should not necessarily delete pending rows immediately, but promotion must reject rows whose fingerprints no longer match live config.

### Source coverage hash

For every source row in the prepared range, hash a canonical tuple:

```text
store_id | session_id | conversation_id | role | content_sha256 | tool_call_id | tool_calls_sha256 | tool_name | timestamp
```

Promotion validates both:

- the ordered set of `store_id`s is exactly what the batch claims;
- the identity hash for every row still matches.

This catches transcript reconciliation, late ingest of missing rows inside the range, externalization/GC rewrites, and accidental ordinal drift.

## Preparation lifecycle

Background preparation should operate only on raw messages that are outside the fresh tail at preparation time.

1. Resolve live config and compute the compactable prefix using the same filtering as foreground compaction.
2. Choose the oldest chunk(s) up to the current publishable frontier.
3. Create or resume a `pending`/`preparing` batch for that frontier.
4. Generate leaf summaries into `pending_summary_nodes`, optionally using previous pending summaries from the same batch as continuity context.
5. Mark the batch `ready` only when `prepared_leaf_count == expected_leaf_count` and every expected range is covered exactly once.

Preparation must not mutate:

- `summary_nodes`
- active replay context
- compaction count
- frontier markers
- transcript GC state
- generated placeholder ordinals

## Atomic promotion

Foreground compaction remains the owner of canonical changes. When `should_compress_preflight()` says compaction is needed, `compress()` may attempt prepared promotion before doing synchronous summarization.

Promotion runs in one SQLite transaction on a single connection, using `BEGIN IMMEDIATE` so foreground/background writers serialize.

The promotion path should not call existing helpers that perform their own commits on separate SQLite connections. Canonical node inserts, lifecycle frontier updates, batch state changes, and superseding older batches must share one transaction boundary. In-process markers such as `_last_compacted_store_id` should update only after the transaction commits.

Validation inside the transaction:

1. Feature flag still enabled.
2. Batch is `ready` for this `conversation_id` and `session_id`.
3. Live policy fingerprint equals batch policy fingerprint.
4. Live summary route fingerprint equals batch summary route fingerprint.
5. Current lifecycle frontier equals the batch’s expected start frontier.
6. Fresh-tail boundary still permits promoting the full prepared range. If the boundary moved such that only a prefix is safe, reject the batch for v1 rather than partially publish.
7. Source rows for the claimed range still exist, are ordered, and match every source identity hash.
8. Pending nodes cover the full source range without gaps or overlaps.
9. No canonical `summary_nodes` already cover the same source IDs. This handles a foreground compaction race that won before promotion.

Publish steps in the same transaction:

1. Insert canonical `summary_nodes` copied from pending rows.
2. Advance lifecycle frontier to the promoted end store id.
3. Mark the batch `promoted` with `promoted_at`.
4. Mark older pending/ready batches for the same conversation `superseded`.
5. Commit.

Only after commit may the engine assemble active context from canonical summaries. If any validation step fails, mark the batch `rejected` with a reason in a transaction that does **not** insert canonical summaries or advance the frontier, then fall back to today’s foreground compaction path.

## Race semantics

### Foreground compaction wins first

If a synchronous foreground pass inserts canonical nodes and advances the frontier while a background batch is pending, later promotion sees either frontier mismatch or canonical source overlap. It rejects/supersedes the stale batch and leaves the foreground result intact.

### Background ready wins first

Foreground promotion takes `BEGIN IMMEDIATE`, validates current source/frontier/fingerprints, publishes, then assembles. A concurrent background preparer trying to write the same batch waits or fails with normal SQLite busy behavior and must reload batch state before continuing.

### Background failure/backoff

Summary failure increments `failure_count`, stores a compact `last_error`, and sets `next_retry_at`. It must not create compaction debt that blocks foreground recovery. Foreground compaction can always ignore failed/pending work and use the current synchronous path.

### Restart

On startup/session bind:

- `promoting` from a crashed transaction should not be visible as canonical unless the transaction committed. If the batch row says `promoting` but no canonical nodes/frontier were advanced, mark it `rejected` or return it to `ready` after validation.
- `preparing` older than a timeout becomes `pending` for retry, or `failed` if backoff policy says so.
- `ready` batches are left ready, but promotion still revalidates live config and source coverage.
- Pending rows are never used by active context during recovery.

SQLite transaction atomicity should mean there is no half-published active state. Recovery still needs explicit cleanup of stale lifecycle labels so status/doctor is trustworthy.

## Reader and diagnostics rules

Active readers default to canonical rows only:

- `lcm_grep` summary search ignores pending rows.
- `lcm_expand(node_id=...)` cannot expand pending IDs through the canonical node path.
- `lcm_describe` active DAG overview excludes pending rows.
- transcript GC eligibility ignores pending rows.
- doctor active-context integrity checks ignore pending rows unless checking async health specifically.

Add an explicit async section to status/doctor instead:

```json
"async_compaction": {
  "enabled": false,
  "worker_enabled": false,
  "pending_batches": 0,
  "preparing_batches": 0,
  "prepared_batches": 0,
  "promoted_batches": 0,
  "rejected_batches": 0,
  "failed_batches": 0,
  "superseded_batches": 0,
  "pending_summaries": 0,
  "oldest_pending_age_seconds": null,
  "last_rejected_reason": null,
  "last_error": null
}
```

Doctor should warn, not fail, for normal disabled state. It should warn on:

- stale `preparing` batches beyond recovery timeout;
- ready batches whose live policy fingerprint no longer matches;
- failed batches whose backoff has expired but no worker has retried;
- pending rows whose batch is missing.

## Implementation sequence

1. **Acceptance tests first** for stale rejection, config change rejection, foreground/background race, summary failure/backoff, restart recovery, successful atomic promotion, pending invisibility, and status/doctor counts.
2. Schema only: create the two tables and reader filters, with feature disabled and no behavior change.
3. Manual one-shot preparer behind the flag, no automatic worker yet.
4. Atomic promotion path in `compress()` before foreground summarization, with fallback on any reject.
5. Status/doctor async counts.
6. Optional worker loop with backpressure and retry policy.
7. Later: pending condensed layers, if leaf-only promotion leaves too much foreground work.

## Test matrix

These are mirrored in `tests/test_async_background_compaction_design.py` as xfailed RED spike tests until the implementation exists.

| Test | Proves |
| --- | --- |
| `test_pending_summaries_are_invisible_until_atomic_promotion` | Pending rows do not affect active assembly/search/status counters except async diagnostics. |
| `test_atomic_promotion_rejects_stale_source_identity` | Transcript reconciliation or row rewrite invalidates the batch without canonical mutation. |
| `test_atomic_promotion_rejects_live_config_change` | Live config/policy wins over persisted prepared metadata. |
| `test_foreground_compaction_race_supersedes_pending_batch` | Foreground compaction and background promotion cannot double-compact or mix generations. |
| `test_summary_failure_backoff_does_not_wedge_foreground_compaction` | Failed background prep backs off but foreground synchronous compaction remains available. |
| `test_restart_recovers_or_discards_pending_batches_safely` | Restart never makes pending rows canonical and cleans stale lifecycle states. |
| `test_default_disabled_async_compaction_is_inert` | Default-off config performs no background preparation and reports zero async counts. |
| `test_atomic_promotion_rejects_summary_route_change` | Model/route changes reject stale prepared summaries before canonical mutation. |
| `test_atomic_promotion_rejects_live_threshold_policy_change` | Live threshold policy changes beat persisted prepared metadata. |
| `test_successful_atomic_promotion_is_all_or_nothing` | Canonical node insert, frontier advance, and batch promotion commit together. |
| `test_atomic_promotion_rolls_back_partial_publish_failure` | A mid-promotion failure leaves no canonical node/frontier/batch half-state. |
| `test_status_and_doctor_report_async_compaction_counts` | Operators see pending/prepared/promoted/rejected/failed counts. |

## Open questions

- Should v1 reject a ready batch when only a prefix is still publishable, or support prefix promotion? Recommendation: reject in v1. Prefix promotion makes continuity and expected leaf counts more complex. Resolved for v0.28.0: a batch is one leaf, and promotion publishes the longest ready chain from the frontier, whole leaves only.
- Should automatic workers live inside `LCMEngine`, a plugin lifecycle helper, or a host-managed scheduler? Recommendation: start with a manual one-shot preparer and make the automatic worker a later slice. Resolved for v0.28.0: one LCM daemon worker per process, after the one-shot preparer slice (S4 before S5).
- Should route fingerprint include fallback model order? Recommendation: yes. Different fallback order can change output after partial failures.
- Should summary timeout changes reject prepared work? Recommendation: no unless timeout policy changes the output contract; include route/model/policy version, not operational timing knobs.
