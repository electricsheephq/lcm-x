# Diagnostics

Use read-only product tools before changing configuration or running an apply path.

## Fast path

1. `hermes plugins list`: confirm `hermes-lcm-x` is enabled and the selected context engine is `lcm-x` (`lcm` is the deprecated alias; `lcm_doctor` flags it under `identity_migration`).
2. Send one normal message if the session has not been bound since restart.
3. `lcm_status`: inspect runtime identity, database path, context pressure, summary/store counts, filters, and lifecycle state.
   `dag.nodes_by_escalation_level` counts the session's summary nodes by the level that produced them (`1`, `2`, `3`, plus `unrecorded` for nodes written before the record existed); a growing `3` count means summaries are being truncated rather than written by the model.
4. `lcm_inspect`: inspect current-session lineage, frontiers, fresh tail, externalized-ref readability, and skip/no-op reasons without retrieving content.
5. `lcm_doctor`: run database, FTS, lifecycle, configuration, and context-pressure diagnostics. `embedding_provider_health` warns when the configured provider cannot run in this process; see `docs/embeddings-setup.md`, section "A host update can remove fastembed", for the embedding-dependency recovery note. With an available cloud provider it also warns when the store has no active vector identity for it or the identity carries an older privacy revision (`embedding_identity_stale` in the detail). It checks the summary identity, and the chunk identity once the store holds chunk vectors; `stale_tasks` names which is stale. Recall then answers from full text with an `embedding_identity_stale:` reason until `/lcm embed warmup` and `/lcm embed backfill --apply` run. An unavailable provider is reported first, because warmup cannot succeed until it is fixed.
6. `lcm_doctor` with `action: repair_level3`: read-only scan for level 3 truncation fragments and their condensed ancestors in the foreground session. Each page returns at most 50 `flagged` nodes and 50 `ancestors`, with `total_flagged`, `total_ancestors` and `next_cursor` when more remain; pass that integer as `cursor` for the next page. It writes nothing and refuses `apply`. The operator's slash command `/lcm doctor repair level3` keeps the store-wide view; `/lcm doctor repair level3 apply` backs up first, then re-summarises each group in place and schedules enabled rollup maintenance. Suggest it to the user, never run it yourself.

If optional slash commands are enabled, `/lcm status` and `/lcm doctor` expose the corresponding operator views.

## Safe mutation order

For cleanup, repair, source normalization, or rotate:

1. run the read-only preview;
2. inspect exact candidates and paths;
3. create/confirm a backup;
4. obtain user authorization for the specific apply operation;
5. run one bounded apply and verify integrity afterward.

Cleanup apply is separately feature-gated. Never infer permission to enable it from a diagnosis request.

## Common states

- Unbound status after restart: send a normal message, then check again.
- Database exists but stays empty: verify plugin enablement, `context.engine`, profile, database path, and ignore/stateless patterns.
- Weak exact recall: verify source rows exist, query construction/scope is correct, summary health is sound, and embedding coverage/provenance matches the requested mode.
- Conflicting summary and raw evidence: prefer the newer exact raw evidence and inspect lineage.
- Path B/context-engine schema log: expected on hosts where plugin-registry handlers do not receive active messages; context-engine schemas and dispatch remain the healthy route.
- Proactive recall injects nothing and `lcm_status` shows `proactive_recall.privacy_policy_errors > 0`:
  a deterministic embedding-privacy configuration fault, not load shedding. Check the
  `LCM_SENSITIVE_PATTERNS` catalog (nonempty, recognized names) and whether the registered
  vector revision matches the current posture; re-run `/lcm embed warmup` after any change.
