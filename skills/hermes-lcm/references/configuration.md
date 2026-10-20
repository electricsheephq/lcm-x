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
- `LCM_LEAF_CHUNK_TOKENS` (default `24000`): maximum raw material per leaf compaction group;
- `LCM_THRESHOLD_FULL_SWEEP_ENABLED` (default `true`): the threshold full sweep;
- `LCM_SUMMARY_PREFIX_TARGET_TOKENS` (default `0`): the summary-frontier size a threshold sweep condenses toward; `0` means one leaf budget capped at a quarter of the compaction threshold, so a small window's summaries cannot fill it (#1139);
- `LCM_BACKGROUND_LEAF_PREP_ENABLED` (default `false`, #787 slice 1): prepare leaf summaries off-turn above P = threshold − 0.15 × window and publish them at the threshold crossing with no model call, down to L = threshold − max(2 × chunk, 0.1 × window); `LCM_BACKGROUND_LEAF_PREP_MAX_CALLS` (default `8`) and `LCM_BACKGROUND_LEAF_PREP_WINDOW_S` (default `600`) bound its own spend guard;
- `LCM_TEMPORAL_ROLLUPS_ENABLED` (default `true` from v0.27.0): background day/week rollup summaries; the first session bind creates the rollup tables and schedules maintenance; `false` opts out;
- `LCM_LEAF_TARGET_RATIO` (default `0.20`), `LCM_LEAF_TARGET_MIN_TOKENS` (default `2000`) and `LCM_LEAF_TARGET_MAX_TOKENS` (default `2000`): leaf summary target `min(MAX, max(MIN, int(source_tokens * RATIO)))`; the first summary call gets twice the target as `max_tokens`. Defaults give a fixed 2,000-token target. Setting only RATIO has no effect while MIN equals MAX and warns; set `LCM_LEAF_TARGET_MAX_TOKENS` above MIN to enable the ratio. A MIN above the default cap also needs MAX, otherwise MIN resets with a warning naming that key (#614, #1138);
- `LCM_DATABASE_PATH`: profile-local SQLite path when the default is unsuitable;
- `LCM_NATIVE_RECOVERY` (default `false`): removed in v0.25.0; ignored if set
- `LCM_SURVIVAL_FIT` (default `true`) and `LCM_SURVIVAL_RESERVE` (default `0.15`): when compaction cannot bring the list under the model window, the oldest whole user turns leave live context (an oversized newest turn is projected) so the session survives; rows stay stored and searchable, a WARNING `LCM survival fit applied` is logged, the user is warned once and `/lcm doctor` reports `survival_fit` as history without changing its status. The `lcm_doctor` tool returns that history in its top-level `survival_fit` field (null when no fit is recorded); it is not a health check and does not affect `overall`. `lcm_status.last_survival_fit` describes the latest fit in the current session, or null, including `cause` when recorded; doctor history retains `last_cause`. The reserve is the share of the window kept free;
- `LCM_HOST_MESSAGE_UID` (default `shadow`; `off`, `shadow`, `on`; env only): on a host whose messages carry `message_uid`, record which stored row each uid was bound to in the droppable `host_uid_bindings` table and count agreement with the existing identity matching; ingest decisions are unchanged. `/lcm doctor` reports `host_uid_*` counts. `off`: no capture, no table writes, no counts, no log line. Under `shadow` and `on`, canonical uid evidence can also exclude a proven hidden re-ingest copy of an assistant message from leaf summary input (the row stays stored and searchable); `off` disables that. `on` (reserved) and unknown values behave as `shadow` (an unknown value warns once). No-op on hosts without uids;
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

The uniform boolean/integer/float environment loader keeps defaults for malformed
values and reports them through `config_source_warnings` in `lcm_status`,
`/lcm status`, and the `lcm_doctor` configuration check. For example,
`LCM_SURVIVAL_FIT=flase` keeps `true`; the warning names the key, shows the raw
value's representation limited to 40 characters, and states the value used.
Empty or whitespace-only values in this loader keep the default without warnings.
Range warnings (`LCM_CONTEXT_THRESHOLD`, `LCM_SURVIVAL_RESERVE`, `LCM_LEAF_CHUNK_TOKENS`, `LCM_SUMMARY_TIMEOUT_MS`), choice warnings (`LCM_EMBEDDING_PROVIDER`, `LCM_PREANSWER_EVIDENCE_MODE`, `LCM_EMBEDDING_STORAGE_DTYPE`), and dropped-input warnings (`LCM_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS`, `LCM_RECALL_ARM_WEIGHTS`, `LCM_MODEL_THRESHOLDS`) use the same channel without changing values or consumer fallbacks.

`LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED` and `LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED` are deprecated in v0.28.0 and removed in v0.29.0 (#1246). Each enabled flag adds one `config_source_warnings` entry without changing behavior; false or unset flags and direct `LCMConfig(...)` construction add no deprecation warning. These flags are environment-only; their `lcm.*` YAML keys are ignored and shipped presets do not set them.

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
- `LCM_SUMMARY_TIMEOUT_MS` (default `120000`, the host's own compression default; when unset, Hermes `auxiliary.compression.timeout` applies if configured): how long one summary call may run before it counts as a failure. The foreground time budget can cut a call shorter.
- `LCM_FOREGROUND_SOFT_SECONDS` (default `60`; `0` = none) and `LCM_FOREGROUND_HARD_SECONDS` (default `120`; `0` or invalid = `120`) (#605): one time budget for a threshold sweep, counted from `compress()` entry. The progress call (a condensation when the summary frontier is over its target, else the first leaf) is tried while 15 s of usable time is left before the hard bound less a finalize reserve; after a stored leaf or condensed node a call starts only if its route's recent duration says it ends, with the reserve, by the hard bound and by the soft target (stop reason `soft_target_reached`). Each call's timeout is at most the usable time left. A step already running is not interrupted, so a compaction can end past the hard bound: read the INFO line `LCM compaction stop:` for the seconds. On `reason=error`, a trailing `failure=` field names the known cause as one token, such as `publication_invariant_conflict` or `exception:RuntimeError` (#1180). A call cut by the budget does not count against the route's circuit, and a compaction takes one spend-guard slot; rollups use their own guard. The budget covers the sweep-off, manual `/compress`, forced overflow and recovery paths too (manual and forced use the hard bound only), and a leaf or condensation source within `LCM_L3_TRUNCATE_TOKENS` is written whole as level 3 with no call.
- `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` (default `true` from v0.27.0): a large tool result or inline media payload is stored as a payload file under the Hermes home and its row holds an `[Externalized payload: ...]` reference; `lcm_expand(externalized_ref=...)` returns the original bytes; `false` opts out;
- `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED` (default `true` from v0.27.0): replay carries the externalized-ref stub in place of a large tool result, per the two thresholds below; needs externalization on; `false` opts out;
- `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS` (default `10000`) and `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_AGED_THRESHOLD_TOKENS` (default `2000`; `0` = the first-sight value, never above it) (#671): with `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` and `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED` both on (the default from v0.27.0), a new tool result over the first value is replaced by its externalized-ref stub at ingest, and at a compaction a tool result outside the fresh tail is stubbed from the aged value. The stub keeps the call row; `lcm_expand(externalized_ref=...)` returns the original. An automatic compaction at or over the threshold first tries these free cuts without a model call and returns them when the host-measured list is at or under min(threshold - leaf chunk, 0.95 x threshold): stop reason `stub_first_exit` (partial; `lcm_status` `last_stub_first_exit`, the doctor, the `LCM compaction stop:` line); the rows no leaf summarised stay stored as backlog.
- `LCM_STUB_FIRST_EXIT_TARGET_RATIO` (default `0`; env only): lowers the stub-first exit's target to min(threshold - leaf chunk, 0.95 x threshold, int(ratio x threshold)); cuts over it take the leaf path. `0` keeps the built-in target; a negative value, a value over `1` or a non-number keeps it with a config warning. The applied target is the one in `last_stub_first_exit` and on the stop line. Use it on a large window where an exit leaves a long live list (with a 750,000-token threshold, `0.2` exits only at 150,000 tokens or under); the share follows the active model's threshold.
- `LCM_SUMMARY_PROMPT_VERSION` (default `1`; env only): `1` keeps the original summariser prompts and 2x output ceiling; `2` opts in to the v2 prompts (six fixed headings, focus directives in the trusted policy, topic label tagged in the transcript message) and a 3x ceiling. Other values fall back to `1` with a config warning. (#646)


## Generated scalar environment reference

Regenerate this table with `python scripts/gen_config_reference.py` from the
checkout root. It lists every scalar registry entry using the declared constructor
default, without reading the environment, credentials, or host configuration.
`""` is an empty string; `null` means unset (automatic provider-dependent protection
for `LCM_EMBEDDING_PRIVACY_ENABLED`). Empty model/path defaults resolve at runtime;
`LCM_CONTEXT_THRESHOLD` and `LCM_SUMMARY_TIMEOUT_MS` can inherit host settings as
described above. Some defaults are sentinels: `LCM_FOREGROUND_HARD_SECONDS` `0` means 120 s,
`LCM_PREANSWER_EVIDENCE_MODE` `""` means `legacy_selective` when pre-answer evidence is on,
`LCM_EMBEDDING_STORE_DIM` `0` means the provider's full dimension, and `LCM_MAX_ASSEMBLY_TOKENS`
and `LCM_RESERVE_TOKENS_FLOOR` `0` mean no cap. The type is the environment parser type, so the privacy boolean
can still have an unset default.

Ranges and nonempty choices are the loader's diagnostic metadata; consumers keep
their existing clamping, fallback, and normalization rules. `—` means no metadata
in that column, not that every input is valid or that a removed option is active.
`LCM_NATIVE_RECOVERY` was removed in v0.25.0; it is parsed and ignored.
The operator guide and the prose above explain behavior and coupled settings.
Deprecation releases describe the two retiring experiments; enabling either still
warns without changing its behavior until removal.

<!-- BEGIN GENERATED CONFIG REFERENCE -->
| Environment key | Type | Default | Diagnostic range / choices | Deprecation |
| --- | --- | --- | --- | --- |
| `LCM_FRESH_TAIL_COUNT` | `int` | `24` | — | — |
| `LCM_FRESH_TAIL_MAX_TOKENS` | `int` | `24000` | — | — |
| `LCM_FRESH_TAIL_PRESSURE_YIELD_ENABLED` | `bool` | `true` | — | — |
| `LCM_FRESH_TAIL_PRESSURE_YIELD_MIN_OBSERVATIONS` | `int` | `3` | — | — |
| `LCM_LEAF_CHUNK_TOKENS` | `int` | `24000` | `>= 1` | — |
| `LCM_LEAF_TARGET_RATIO` | `float` | `0.2` | — | — |
| `LCM_LEAF_TARGET_MIN_TOKENS` | `int` | `2000` | — | — |
| `LCM_LEAF_TARGET_MAX_TOKENS` | `int` | `2000` | — | — |
| `LCM_CONTEXT_THRESHOLD` | `float` | `0.75` | `> 0.0`, `<= 1.0` | — |
| `LCM_INCREMENTAL_MAX_DEPTH` | `int` | `3` | — | — |
| `LCM_CONDENSATION_FANIN` | `int` | `4` | — | — |
| `LCM_DYNAMIC_LEAF_CHUNK_ENABLED` | `bool` | `false` | — | — |
| `LCM_DYNAMIC_LEAF_CHUNK_MAX` | `int` | `40000` | — | — |
| `LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED` | `bool` | `false` | — | Deprecated v0.28.0; removed v0.29.0 |
| `LCM_CACHE_FRIENDLY_MIN_DEBT_GROUPS` | `int` | `2` | — | — |
| `LCM_DEFERRED_MAINTENANCE_ENABLED` | `bool` | `false` | — | — |
| `LCM_DEFERRED_MAINTENANCE_MAX_PASSES` | `int` | `4` | — | — |
| `LCM_CRITICAL_BUDGET_PRESSURE_RATIO` | `float` | `0.0` | — | — |
| `LCM_THRESHOLD_FULL_SWEEP_ENABLED` | `bool` | `true` | — | — |
| `LCM_SUMMARY_PREFIX_TARGET_TOKENS` | `int` | `0` | — | — |
| `LCM_FOREGROUND_SOFT_SECONDS` | `float` | `60.0` | — | — |
| `LCM_FOREGROUND_HARD_SECONDS` | `float` | `0.0` | — | — |
| `LCM_L2_BUDGET_RATIO` | `float` | `0.5` | — | — |
| `LCM_L3_TRUNCATE_TOKENS` | `int` | `512` | — | — |
| `LCM_MAX_ASSEMBLY_TOKENS` | `int` | `0` | — | — |
| `LCM_RESERVE_TOKENS_FLOOR` | `int` | `0` | — | — |
| `LCM_CUSTOM_INSTRUCTIONS` | `str` | `""` | — | — |
| `LCM_SUMMARY_PROMPT_VERSION` | `int` | `1` | — | — |
| `LCM_EXTRACTION_ENABLED` | `bool` | `false` | — | — |
| `LCM_EXTRACTION_MODEL` | `str` | `""` | — | — |
| `LCM_EXTRACTION_OUTPUT_PATH` | `str` | `""` | — | — |
| `LCM_ASSERTIONS_ENABLED` | `bool` | `false` | — | — |
| `LCM_QUERY_VIEWS_ENABLED` | `bool` | `false` | — | — |
| `LCM_ADAPTIVE_RETRIEVAL_ENABLED` | `bool` | `false` | — | — |
| `LCM_ASSERTION_EXTRACTION_ENABLED` | `bool` | `false` | — | — |
| `LCM_ASSERTION_EXTRACTION_MODEL` | `str` | `""` | — | — |
| `LCM_ASSERTION_EXTRACTION_MAX_SOURCES_PER_PASS` | `int` | `4` | — | — |
| `LCM_ASSERTION_EXTRACTION_TIMEOUT_SECONDS` | `float` | `30.0` | — | — |
| `LCM_SENSITIVE_PATTERNS_ENABLED` | `bool` | `false` | — | — |
| `LCM_EMBEDDING_PRIVACY_ENABLED` | `bool` | `null` | — | — |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` | `bool` | `true` | — | — |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS` | `int` | `12000` | — | — |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH` | `str` | `""` | — | — |
| `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED` | `bool` | `true` | — | — |
| `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS` | `int` | `10000` | — | — |
| `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_AGED_THRESHOLD_TOKENS` | `int` | `2000` | — | — |
| `LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED` | `bool` | `false` | — | Deprecated v0.28.0; removed v0.29.0 |
| `LCM_STUB_FIRST_EXIT_TARGET_RATIO` | `float` | `0.0` | — | — |
| `LCM_SUMMARY_MODEL` | `str` | `""` | — | — |
| `LCM_NATIVE_RECOVERY` | `bool` | `false` | — | — |
| `LCM_SURVIVAL_FIT` | `bool` | `true` | — | — |
| `LCM_SURVIVAL_RESERVE` | `float` | `0.15` | `>= 0.0`, `<= 0.9` | — |
| `LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `int` | `2` | — | — |
| `LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `int` | `300` | — | — |
| `LCM_SUMMARY_CIRCUIT_BREAKER_REJECTION_THRESHOLD` | `int` | `6` | — | — |
| `LCM_SUMMARY_SPEND_MAX_CALLS` | `int` | `24` | — | — |
| `LCM_SUMMARY_SPEND_WINDOW_SECONDS` | `float` | `600.0` | — | — |
| `LCM_SUMMARY_SPEND_BACKOFF_SECONDS` | `float` | `1800.0` | — | — |
| `LCM_EXPANSION_MODEL` | `str` | `""` | — | — |
| `LCM_EXPANSION_CONTEXT_TOKENS` | `int` | `32000` | — | — |
| `LCM_SUMMARY_TIMEOUT_MS` | `int` | `120000` | `> 0` | — |
| `LCM_EXPANSION_TIMEOUT_MS` | `int` | `120000` | — | — |
| `LCM_DATABASE_PATH` | `str` | `""` | — | — |
| `LCM_EMBEDDINGS_ENABLED` | `bool` | `false` | — | — |
| `LCM_RERANK_ENABLED` | `bool` | `false` | — | — |
| `LCM_RERANK_MODEL` | `str` | `"rerank-2.5-lite"` | — | — |
| `LCM_RERANK_WINDOW_LIMIT` | `int` | `0` | — | — |
| `LCM_RERANK_MARGIN` | `float` | `0.0` | — | — |
| `LCM_RECALL_SCAN_ROWS` | `int` | `25000` | — | — |
| `LCM_RECALL_SCAN_MAX_ROWS` | `int` | `0` | — | — |
| `LCM_RECALL_SCAN_BUDGET_S` | `float` | `0.0` | — | — |
| `LCM_RECALL_REFERENCE_STRICT` | `bool` | `true` | — | — |
| `LCM_RECALL_FTS_ANCHOR` | `bool` | `true` | — | — |
| `LCM_PROACTIVE_RECALL_ENABLED` | `bool` | `false` | — | — |
| `LCM_PROACTIVE_RECALL_MIN_SCORE` | `float` | `0.01` | — | — |
| `LCM_PROACTIVE_RECALL_BUDGET_TOKENS` | `int` | `500` | — | — |
| `LCM_PROACTIVE_RECALL_PROVIDER` | `str` | `""` | — | — |
| `LCM_PREANSWER_EVIDENCE_ENABLED` | `bool` | `false` | — | — |
| `LCM_PREANSWER_EVIDENCE_MODE` | `str` | `""` | `"off"`, `"legacy_selective"`, `"requirements_v1"` | — |
| `LCM_SELECTIVE_COMPILER_ENABLED` | `bool` | `false` | — | — |
| `LCM_SELECTIVE_COMPILER_MODEL` | `str` | `""` | — | — |
| `LCM_EMBEDDING_BOUNDED_SCAN_ROWS` | `int` | `2000` | — | — |
| `LCM_EMBEDDING_STORAGE_DTYPE` | `str` | `"float32"` | `"float32"`, `"int8"` | — |
| `LCM_EMBEDDING_STORE_DIM` | `int` | `0` | — | — |
| `LCM_EMBEDDING_BINARY_PRESCREEN` | `bool` | `false` | — | — |
| `LCM_KNN_PRESCREEN_MULTIPLIER` | `int` | `4` | — | — |
| `LCM_EMBEDDING_PROVIDER` | `str` | `""` | `"voyage"`, `"voyageai"`, `"ollama"`, `"fastembed"`, `"fast-embed"`, `"openai-compatible"`, `"openai"`, `"siliconflow"` | — |
| `LCM_EMBEDDING_MODEL` | `str` | `""` | — | — |
| `LCM_EMBED_CONTENT_POLICY` | `str` | `"conversational"` | — | — |
| `LCM_OLLAMA_BASE_URL` | `str` | `"http://localhost:11434"` | — | — |
| `LCM_EMBEDDING_BASE_URL` | `str` | `""` | — | — |
| `LCM_EMBEDDING_API_KEY_ENV` | `str` | `"LCM_EMBEDDING_API_KEY"` | — | — |
| `LCM_EMBEDDING_QUERY_TIMEOUT_S` | `float` | `3.0` | — | — |
| `LCM_RECALL_QUERY_TIMEOUT_S` | `float` | `8.0` | — | — |
| `LCM_EMBEDDING_BACKFILL_TIMEOUT_S` | `float` | `120.0` | — | — |
| `LCM_EMBEDDING_MAX_BATCH_ITEMS` | `int` | `1000` | — | — |
| `LCM_EMBEDDING_QUERY_SPEND_MAX_CALLS` | `int` | `600` | — | — |
| `LCM_EMBEDDING_QUERY_SPEND_WINDOW_SECONDS` | `float` | `60.0` | — | — |
| `LCM_EMBEDDING_QUERY_SPEND_BACKOFF_SECONDS` | `float` | `60.0` | — | — |
| `LCM_NEW_SESSION_RETAIN_DEPTH` | `int` | `2` | — | — |
| `LCM_DOCTOR_CLEAN_APPLY_ENABLED` | `bool` | `false` | — | — |
| `LCM_EMPTY_LIFECYCLE_GC_ENABLED` | `bool` | `true` | — | — |
| `LCM_EMPTY_LIFECYCLE_GC_THRESHOLD` | `int` | `200` | — | — |
| `LCM_TEMPORAL_ROLLUPS_ENABLED` | `bool` | `true` | — | — |
| `LCM_ROLLUP_DAILY_TARGET_TOKENS` | `int` | `5000` | — | — |
| `LCM_ROLLUP_DAILY_MAX_TOKENS` | `int` | `15000` | — | — |
| `LCM_ROLLUP_AGGREGATE_MAX_TOKENS` | `int` | `20000` | — | — |
| `LCM_ROLLUP_BUILDS_PER_PASS` | `int` | `2` | — | — |
| `LCM_ROLLUP_MAINTENANCE_BUDGET_MS` | `int` | `5000` | — | — |
| `LCM_BACKGROUND_LEAF_PREP_ENABLED` | `bool` | `false` | — | — |
| `LCM_BACKGROUND_LEAF_PREP_MAX_CALLS` | `int` | `8` | — | — |
| `LCM_BACKGROUND_LEAF_PREP_WINDOW_S` | `float` | `600.0` | — | — |
<!-- END GENERATED CONFIG REFERENCE -->

## Hand-written reference: explicit configuration readers

These entries are deliberately outside the scalar registry and generated table.
All are read by `config.py` in `LCMConfig.from_env()`; list entries accept
comma-separated values. Keep their existing source tracking and parsing rules.

| Environment key | Format / declared default |
| --- | --- |
| `LCM_MODEL_THRESHOLDS` | `model:threshold` pairs; `{}`; each threshold must be > 0 and <= 1 |
| `LCM_SUMMARY_REASONING_EFFORT` | Empty (task/provider default); `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `LCM_EXPANSION_REASONING_EFFORT` | Same reasoning choices; empty (task/provider default) |
| `LCM_SENSITIVE_PATTERNS` | List; `api_key,bearer_token,password_assignment,private_key` |
| `LCM_SUMMARY_FALLBACK_MODELS` | List; empty |
| `LCM_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS` | Float; `24.0` |
| `LCM_RECALL_ARM_WEIGHTS` | `arm=weight` pairs; `fts=0.5,summary=1.0,chunk=1.0` |
| `LCM_IGNORE_SESSION_PATTERNS` | Session glob list; empty |
| `LCM_STATELESS_SESSION_PATTERNS` | Session glob list; empty |
| `LCM_IGNORE_MESSAGE_PATTERNS` | Message regex list; empty |

## Hand-written reference: side-channel environment readers

These 11 keys are read outside `config.py`, directly or through helper
wrappers. They are not loaded into `LCMConfig`; listing them does not
consolidate or change their readers. Reader filenames are relative to the
checkout root. Credential entries list only variable names, never values.

| Environment key | Reader file(s) | Purpose | Read when |
| --- | --- | --- | --- |
| `LCM_ABSOLUTE_THRESHOLD_TOKENS` | `engine.py` | Absolute compaction threshold override | each call — `_set_context_length()` with a positive context length, through model updates, session-start metadata, or cloning |
| `LCM_ENABLE_SLASH_COMMAND` | `__init__.py` | Optional slash-command registration | plugin registration |
| `LCM_HERMES_BASE_DIR` | `config.py` (called by `engine.py`, `externalize.py`, `diagnostics.py`) | Storage containment boundary; process-wide, never read from a routed profile | each call — storage-directory resolution or state-database containment validation; `_get_allowed_hermes_base()` also reads it when called (tests only; no production caller) |
| `LCM_IDENTITY_ANCHOR` | `identity_anchor.py` | Identity-anchor switch | each call — `identity_anchor_enabled()` checks during ingest, replay identity handling, and compaction |
| `LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS` | `db_bootstrap.py` | Startup integrity-check interval | database open — startup FTS validation, after structural checks pass |
| `LCM_FTS_INTEGRITY_BACKGROUND` | `db_bootstrap.py` | Background integrity-check switch | database open — startup FTS validation, when a deep check is due |
| `LCM_EMBEDDING_BACKFILL_LEASE_TTL_S` | `command.py` | Backfill claim lifetime | each call — each summary or chunk backfill apply run, before lease acquisition |
| `LCM_EMBEDDING_BACKFILL_HEARTBEAT_S` | `command.py` | Backfill claim renewal cadence | each call — each summary or chunk backfill apply run, before lease acquisition |
| `LCM_EMBEDDING_BACKFILL_BUDGET_S` | `command.py` | Operation-wide backfill time budget | each call — each summary or chunk backfill apply run, before lease acquisition |
| `LCM_EMBEDDING_BACKFILL_BATCH_SIZE` | `command.py` | Backfill claim batch size | each call — backfill batch estimates, apply-run batch sizing, and automatic embedding-maintenance pass sizing |
| `LCM_EMBEDDING_API_KEY` | `embedding_provider.py` | Default credential variable for the compatible endpoint; name selectable via `LCM_EMBEDDING_API_KEY_ENV` | plugin registration — availability probe when compatible embeddings are enabled; also re-read at each compatible-provider `_request()` and availability probe (doctor and automatic maintenance) |

Restart the host for `LCM_ENABLE_SLASH_COMMAND`; the other keys take effect on
the next named operation (FTS controls on the next database open, and
`LCM_EMBEDDING_API_KEY` on the next request or probe despite its initial
registration-time check), provided the running process's environment has been
updated; changing a parent shell's environment alone requires restarting the
host to pass it the new values.

On routed profiles, registration snapshots the absolute override for that home;
model updates and clones keep that snapshot. Routed readers use the host-bound
profile mapping or that home's registration pin. LCM-X background work runs with
the profile that scheduled it.
A multiplexed messaging gateway builds every profile's scope, including the launch
profile, from its `.env` and configured secret sources; process-only settings and
provider keys do not apply there. `hermes serve` instead supplies its captured launch
environment overlaid by the launch profile's files; LCM-X preserves that composition.
Put routed settings and provider keys in each profile's files or secret sources.
`LCM_HERMES_BASE_DIR` is the exception: the containment root always comes from the
process environment, so a profile can neither drop nor widen it.

Three additional standalone helpers in `config.py` are also outside the scalar
registry and `LCMConfig.from_env()`:

| Environment key | Reader file(s) | Purpose |
| --- | --- | --- |
| `LCM_DISABLED_TOOLS` | `config.py` (called by `engine.py`, `__init__.py`) | Call-time tool exclusion list |
| `LCM_HOST_MESSAGE_UID` | `config.py` (called by `host_uid.py`, `host_uid_emit.py`, `message_analysis.py`) | Host message-identity capture mode |
| `LCM_SQLITE_MMAP_SIZE` | `config.py` (called by `db_bootstrap.py`) | Connection-time mapped-read budget |
