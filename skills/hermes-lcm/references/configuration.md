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

- `LCM_CONTEXT_THRESHOLD`: when normal context pressure triggers compaction;
- `LCM_FRESH_TAIL_COUNT`: newest messages kept raw;
- `LCM_LEAF_CHUNK_TOKENS`: maximum raw material per leaf compaction group;
- `LCM_LEAF_TARGET_RATIO` (default `0.20`), `LCM_LEAF_TARGET_MIN_TOKENS` (default `2000`) and `LCM_LEAF_TARGET_MAX_TOKENS` (default `12000`): leaf summary target `min(MAX, max(MIN, int(source_tokens * RATIO)))`; the first summary call gets twice the target as `max_tokens`. Defaults are unchanged; change them only after measuring what a different ratio keeps (#614);
- `LCM_DATABASE_PATH`: profile-local SQLite path when the default is unsuitable;
- `LCM_NATIVE_RECOVERY` (default `false`): ingest sources normally, then allow a cancellation-fenced Hermes host to summarize the active context through its native compressor and archive transaction. LCM history and recall stay available; its frontier is not advanced and LCM publication is not attempted. Sanitized user text remains in active replay even when its durable copy is externalized, allowing the native summarizer to read it. This increases active token usage; previously published references are not automatically expanded. Failed, cancelled, placeholder or non-shrinking summaries retain the active context. This is recovery, not repair of the underlying coverage mismatch;
- `LCM_SURVIVAL_FIT` (default `true`) and `LCM_SURVIVAL_RESERVE` (default `0.15`): when compaction cannot bring the list under the model window, the oldest whole user turns leave live context (an oversized newest turn is projected) so the session survives; rows stay stored and searchable, a WARNING `LCM survival fit applied` is logged, the user is warned once and `/lcm doctor` reports `survival_fit`. The reserve is the share of the window kept free;
- `LCM_IGNORE_SESSION_PATTERNS` and `LCM_STATELESS_SESSION_PATTERNS`: storage ownership boundaries;
- summary/embedding provider settings only after confirming credentials, cost, and data handling. Known cloud providers protect provider-bound copies automatically with the configured nonempty known pattern list while durable storage stays raw. `LCM_SENSITIVE_PATTERNS_ENABLED=true` is a separate irreversible durable-ingest opt-in. `LCM_EMBEDDING_PRIVACY_ENABLED=false` explicitly sends raw cloud copies under the `privacy:off` vector revision; warmup binds the chosen posture and later query/backfill calls fail closed on identity drift.

Optional slash commands are disabled by default with `LCM_ENABLE_SLASH_COMMAND=false`. Destructive cleanup apply is separately guarded. Do not enable mutation surfaces merely to diagnose a problem.

Change one tuning variable at a time, then re-check `lcm_status`, context pressure, summary health, latency, and actual answer quality.

### Summary circuit

Each summary route (the summary model and each fallback model) has its own circuit. A route whose circuit is open is skipped until its cooldown ends. While every route is refused, a compaction that is not a forced overflow recovery writes no further leaf and no condensed node and keeps the remaining rows for a later pass (#628); a leaf whose own level 1 and level 2 results were rejected is not stored either while the survival fit can rescue the request, unless its level 3 cut is the whole source (#652); a forced overflow recovery, and a host whose model window is unknown or whose survival fit is off, still fall back to deterministic truncation. An accepted summary resets the route's counts.

- `LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` (default `2`): provider failures (a call that raises or times out) before the route is refused;
- `LCM_SUMMARY_CIRCUIT_BREAKER_REJECTION_THRESHOLD` (default `6`, #630): results rejected for their content (empty, reasoning-only, integrity contract violated, not shorter than the source) before the route is refused;
- `LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS` (default `300`): seconds an open route is refused before a retry is allowed;
- `LCM_SUMMARY_TIMEOUT_MS` (default `60000`; when unset, Hermes `auxiliary.compression.timeout` applies if configured): how long one summary call may run before it counts as a failure.
