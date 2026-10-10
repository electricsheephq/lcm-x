# Configuration and activation

LCM-X is a general Hermes plugin and a context engine. Both identifiers must be
active:

```yaml
plugins:
  enabled:
    - hermes-lcm-x

context:
  engine: lcm-x
```

Migrating from v0.23.x (`hermes-lcm` / `lcm`, BREAKING in v0.24.0): stop Hermes,
update the code, replace `hermes-lcm` with `hermes-lcm-x` in `plugins.enabled`,
set `context.engine: lcm-x`, then start Hermes. Keep `hermes-lcm` listed only
when `plugins/hermes-lcm` is this same checkout; with a separate older copy
installed, LCM-X logs `Another LCM generation is already loaded` and stays inert.
A config that enables only `hermes-lcm` no longer loads LCM-X: Hermes logs
`Context engine 'lcm' not found — falling back to built-in compressor`. The
existing `lcm.db` is untouched, but turns handled while Hermes runs without
LCM-X are not in `lcm.db` and their compacted content may not be recoverable, so
update the config before restarting Hermes after the update. Legacy
`context.engine: lcm` still works through an alias with a warning and an
`identity_migration` field in `lcm_status` / `lcm_doctor`.

Restart Hermes after changing plugin or context-engine configuration. Verify with `hermes plugins list`, then use `lcm_status` after a normal message has bound the session.

## Installation

An existing checkout can install profile-aware plugin and skill links:

```bash
./scripts/install.sh
HERMES_PROFILE=myprofile ./scripts/install.sh
```

The installer exposes both:

- `plugins/hermes-lcm-x` for plugin loading;
- `skills/hermes-lcm-x` for normal skill discovery (the skill name stays `hermes-lcm`).

It refuses conflicting paths rather than overwriting an existing install, reuses
an existing `plugins/hermes-lcm` link to the same checkout, and prints migration
steps for a legacy install without editing config or deleting anything.

## High-impact controls

Use `docs/operator-guide.md` as the complete current source. Start with:

- `LCM_CONTEXT_THRESHOLD` (default `0.75`): when normal context pressure triggers compaction;
- `LCM_FRESH_TAIL_COUNT` (default `24`) and `LCM_FRESH_TAIL_MAX_TOKENS` (default `24000`): newest messages kept raw, and their token cap;
- `LCM_LEAF_CHUNK_TOKENS` (default `8000`): maximum raw material per leaf compaction group;
- `LCM_THRESHOLD_FULL_SWEEP_ENABLED` (default `true`): the threshold full sweep;
- `LCM_TEMPORAL_ROLLUPS_ENABLED` (default `true` from v0.27.0): background day/week rollup summaries; the first session bind creates the rollup tables and schedules maintenance; `false` opts out;
- `LCM_LEAF_TARGET_RATIO` (default `0.20`), `LCM_LEAF_TARGET_MIN_TOKENS` (default `2000`) and `LCM_LEAF_TARGET_MAX_TOKENS` (default `12000`): leaf summary target `min(MAX, max(MIN, int(source_tokens * RATIO)))`; the first summary call gets twice the target as `max_tokens`. Defaults are unchanged; change them only after measuring what a different ratio keeps (#614);
- `LCM_DATABASE_PATH`: profile-local SQLite path when the default is unsuitable;
- `LCM_NATIVE_RECOVERY` (default `false`): removed in v0.25.0; ignored if set
- `LCM_SURVIVAL_FIT` (default `true`) and `LCM_SURVIVAL_RESERVE` (default `0.15`): when compaction cannot bring the list under the model window, the oldest whole user turns leave live context (an oversized newest turn is projected) so the session survives; rows stay stored and searchable, a WARNING `LCM survival fit applied` is logged, the user is warned once and `/lcm doctor` reports `survival_fit`. The reserve is the share of the window kept free;
- `LCM_HOST_MESSAGE_UID` (default `shadow`; `off`, `shadow`, `on`; env only): on a host whose messages carry `message_uid`, record which stored row each uid was bound to in the droppable `host_uid_bindings` table and count agreement with the existing identity matching; ingest decisions are unchanged. `/lcm doctor` reports `host_uid_*` counts. `off`: no capture, no table writes, no counts, no log line. `on` (reserved) and unknown values behave as `shadow` (an unknown value warns once). No-op on hosts without uids;
- `FASTEMBED_CACHE_PATH` (fastembed's own variable, not an `LCM_*` key): the model cache the local `fastembed` provider loads from (default `~/.cache/fastembed`); it may be a shared, read-only cache. A read-only cache must be pre-seeded, because `/lcm embed warmup` cannot download into it. The model runs with 2 threads;
- `LCM_IGNORE_SESSION_PATTERNS` and `LCM_STATELESS_SESSION_PATTERNS`: storage ownership boundaries;
- summary/embedding provider settings only after confirming credentials, cost, and data handling; see `docs/embeddings-setup.md`, section "A host update can remove fastembed", for the embedding-dependency recovery note. Known cloud providers protect provider-bound copies automatically with the configured nonempty known pattern list while durable storage stays raw. `LCM_SENSITIVE_PATTERNS_ENABLED=true` is a separate irreversible durable-ingest opt-in. `LCM_EMBEDDING_PRIVACY_ENABLED=false` explicitly sends raw cloud copies under the `privacy:off` vector revision; warmup binds the chosen posture and later query/backfill calls fail closed on identity drift.

Optional slash commands are disabled by default with `LCM_ENABLE_SLASH_COMMAND=false`. Destructive cleanup apply is separately guarded. Do not enable mutation surfaces merely to diagnose a problem.

`LCM_SQLITE_MMAP_SIZE` sets SQLite mapped-read bytes (default `268435456`,
unchanged). Set `LCM_SQLITE_MMAP_SIZE=0` on SD-card or single-board hosts: a
storage read error on a mapped page kills the process with SIGBUS, while `0`
returns a catchable I/O error. Invalid or negative values use the default with
one WARNING per process. `/lcm doctor` reports effective `sqlite_mmap_bytes`,
including SQLite's cap. Example: `LCM_SQLITE_MMAP_SIZE=0 hermes` after restarting.

Change one tuning variable at a time, then re-check `lcm_status`, context pressure, summary health, latency, and actual answer quality.

### Tool exposure

Both native context-engine schemas and plugin-registry registration advertise
13 tools by default. `LCM_ASSERTIONS_ENABLED=true` exposes `lcm_query_state`;
`LCM_ADAPTIVE_RETRIEVAL_ENABLED=true` exposes `lcm_retrieve`. Restart after
changing these loaded process flags. `LCM_DISABLED_TOOLS` (comma-separated
names) hides any tool, including a flag-enabled one, and refuses direct calls.
Calls through a cached dormant schema retain the existing disabled response.
The manifest continues to declare all 15 tools.

### Recall ranking

- `LCM_RECALL_ARM_WEIGHTS` (default `fts=0.5,summary=1.0,chunk=1.0`): per-arm RRF weights for `lcm_recall`; see `docs/retrieval-tools.md`.
- `LCM_RECALL_FTS_ANCHOR` (default `true`, #950): with two or more arms, keeps the full-text arm's best match inside the `lcm_recall` ranked window when fusion pushed it out (at most one session displaced). The hit keeps its own score; `provenance.fts_anchor = {fired, position, delivered}` reports the move and whether the anchored hit is in the returned hits after `answer_ready` selection and the response cap (`delivered` is `null` when the anchor did not fire). Embeddings-off output is unchanged, and an `fts` weight of `0` turns it off. It is inert for `limit` below `4` (displacing one of three or fewer hits costs more than it gains; frozen in #950), and proactive recall does not use it. Set `false` to restore the previous order. Managed (evaOS fleet) profiles can set it only after a provisioning release adds it to the environment allowlist.

### Summary circuit

Each summary route (the summary model and each fallback model) has its own circuit. A route whose circuit is open is skipped until its cooldown ends. While every route is refused, a compaction that is not a forced overflow recovery writes no further leaf and no condensed node and keeps the remaining rows for a later pass (#628); a leaf whose own level 1 and level 2 results were rejected is not stored either while the survival fit can rescue the request, unless its level 3 cut is the whole source (#652); a forced overflow recovery, and a host whose model window is unknown or whose survival fit is off, still fall back to deterministic truncation. An accepted summary resets the route's counts.

A 400 or 404 saying the route cannot serve its model (unknown model, model not found, does not exist, not supported) is a configuration error (#682): it opens the route on the first failure and logs one WARNING per episode naming the provider and model the host used and whether LCM-X sent a model; a successful summary ends the episode. The fix is in the profile's `config.yaml`, not an LCM key: set `auxiliary.compression.provider` and `auxiliary.compression.model` together, or a consistent `model.provider` / `model.default` pair. `lcm_status.summary_route` shows `state` (`open`/`closed`), `seconds_left`, `last_error_class`, `provider` and `model`.

- `LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` (default `2`): provider failures (a call that raises or times out) before the route is refused;
- `LCM_SUMMARY_CIRCUIT_BREAKER_REJECTION_THRESHOLD` (default `6`, #630): results rejected for their content (empty, reasoning-only, integrity contract violated, not shorter than the source) before the route is refused;
- `LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS` (default `300`): seconds an open route is refused before a retry is allowed;
- `LCM_SUMMARY_TIMEOUT_MS` (default `60000`; when unset, Hermes `auxiliary.compression.timeout` applies if configured): how long one summary call may run before it counts as a failure.
- `LCM_FOREGROUND_SOFT_SECONDS` (default `60`; `0` = none) and `LCM_FOREGROUND_HARD_SECONDS` (default `120`; `0` or invalid = `120`) (#605): one time budget for a threshold sweep, counted from `compress()` entry. The progress call (a condensation when the summary frontier is over its target, else the first leaf) is tried while 15 s of usable time is left before the hard bound less a finalize reserve; after a stored leaf or condensed node a call starts only if its route's recent duration says it ends, with the reserve, by the hard bound and by the soft target (stop reason `soft_target_reached`). Each call's timeout is at most the usable time left. A step already running is not interrupted, so a compaction can end past the hard bound: read the INFO line `LCM compaction stop:` for the seconds. A call cut by the budget does not count against the route's circuit, and a compaction takes one spend-guard slot; rollups use their own guard. The budget covers the sweep-off, manual `/compress`, forced overflow and recovery paths too (manual and forced use the hard bound only), and a leaf or condensation source within `LCM_L3_TRUNCATE_TOKENS` is written whole as level 3 with no call.
- `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS` (default `10000`) and `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_AGED_THRESHOLD_TOKENS` (default `2000`; `0` = the first-sight value, never above it) (#671): with `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` and `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED` both on, a new tool result over the first value is replaced by its externalized-ref stub at ingest, and at a compaction a tool result outside the fresh tail is stubbed from the aged value. The stub keeps the call row; `lcm_expand(externalized_ref=...)` returns the original. An automatic compaction at or over the threshold first tries these free cuts without a model call and returns them when the host-measured list is at or under min(threshold - leaf chunk, 0.95 x threshold): stop reason `stub_first_exit` (partial; `lcm_status` `last_stub_first_exit`, the doctor, the `LCM compaction stop:` line); the rows no leaf summarised stay stored as backlog.
- `LCM_SUMMARY_PROMPT_VERSION` (default `1`; env only): `1` keeps the original summariser prompts and 2x output ceiling; `2` opts in to the v2 prompts (six fixed headings, focus directives in the trusted policy, topic label tagged in the transcript message) and a 3x ceiling. Other values fall back to `1` with a config warning. (#646)
