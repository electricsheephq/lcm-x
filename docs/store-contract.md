# Store contract

Contract version: 1
Digest: sha256:76644a990e50b9613697704dcb05b349a952d44ea22187bd768b0eab9a9badc9

## What a consumer may read

Use plain SQL over a read-only (`mode=ro`) connection. These are the contracted tables and columns; feature tables may be absent until their named migration first runs. Probe presence before reading them. JSON stored in TEXT remains JSON, not a separate SQL type.
Column entries below give name, declared type, nullability, default SQL and primary-key position. Index names are sorted. The digest covers only what these tables show. It does not describe which columns an index covers, whether an index is unique, index expressions, triggers, CHECK constraints or foreign keys. For callables it covers parameter names, kinds and defaults, not behaviour or return values. Column order is a fresh store's: a store migrated from an older release can list the same columns in another order, so read columns by name.

<!-- Generated from a fresh store with all contracted named features enabled. -->
| Table | Columns (name: type; nullability; default; primary key position) | Index names |
| --- | --- | --- |
| `capture_events` | event_id: INTEGER; nullable; no default; pk=1; session_id: TEXT; NOT NULL; no default; pk=0; lineage_key: TEXT; NOT NULL; no default; pk=0; source_kind: TEXT; NOT NULL; no default; pk=0; generation: INTEGER; NOT NULL; no default; pk=0; position: INTEGER; NOT NULL; no default; pk=0; uid: TEXT; nullable; no default; pk=0; entry_type: TEXT; nullable; no default; pk=0; entry_class: TEXT; NOT NULL; no default; pk=0; line_digest: TEXT; NOT NULL; no default; pk=0; content_digest: TEXT; nullable; no default; pk=0; disposition: TEXT; NOT NULL; no default; pk=0; store_ids: TEXT; nullable; no default; pk=0; notes: TEXT; nullable; no default; pk=0; raw: BLOB; nullable; no default; pk=0; raw_from: INTEGER; nullable; no default; pk=0; projection_version: TEXT; NOT NULL; no default; pk=0; captured_at: REAL; NOT NULL; no default; pk=0 | idx_capture_events_file, idx_capture_events_line, idx_capture_events_shared, idx_capture_events_uid |
| `host_uid_bindings` | store_id: INTEGER; NOT NULL; no default; pk=0; uid: TEXT; NOT NULL; no default; pk=0; lineage_key: TEXT; NOT NULL; no default; pk=0; kind: TEXT; NOT NULL; no default; pk=0; binding_version: INTEGER; NOT NULL; no default; pk=0; proof_kind: TEXT; nullable; no default; pk=0; created_at: REAL; nullable; no default; pk=0; first_check: TEXT; nullable; no default; pk=0; disagree_seen: INTEGER; NOT NULL; 0; pk=0 | idx_host_uid_canonical, idx_host_uid_lineage_uid, idx_host_uid_store |
| `ingest_views` | lineage: TEXT; NOT NULL; no default; pk=1; view_seq: INTEGER; NOT NULL; no default; pk=3; kind: TEXT; NOT NULL; no default; pk=2; state: TEXT; nullable; no default; pk=0; entries: BLOB; NOT NULL; no default; pk=0; prefix_chain: BLOB; NOT NULL; no default; pk=0 | sqlite_autoindex_ingest_views_1 |
| `lcm_lifecycle_state` | conversation_id: TEXT; nullable; no default; pk=1; current_session_id: TEXT; nullable; no default; pk=0; last_finalized_session_id: TEXT; nullable; no default; pk=0; current_frontier_store_id: INTEGER; NOT NULL; 0; pk=0; last_finalized_frontier_store_id: INTEGER; NOT NULL; 0; pk=0; debt_kind: TEXT; nullable; no default; pk=0; debt_size_estimate: INTEGER; NOT NULL; 0; pk=0; current_bound_at: REAL; nullable; no default; pk=0; last_finalized_at: REAL; nullable; no default; pk=0; debt_updated_at: REAL; nullable; no default; pk=0; last_maintenance_attempt_at: REAL; nullable; no default; pk=0; last_rollover_at: REAL; nullable; no default; pk=0; last_reset_at: REAL; nullable; no default; pk=0; updated_at: REAL; NOT NULL; strftime('%s','now'); pk=0 | idx_lcm_lifecycle_current_session, idx_lcm_lifecycle_last_finalized_session, sqlite_autoindex_lcm_lifecycle_state_1 |
| `lcm_migration_state` | step_name: TEXT; nullable; no default; pk=1; completed_at: REAL; NOT NULL; no default; pk=0 | sqlite_autoindex_lcm_migration_state_1 |
| `ledger_emissions` | emission_id: INTEGER; nullable; no default; pk=1; lineage: TEXT; NOT NULL; no default; pk=0; view_seq: INTEGER; NOT NULL; no default; pk=0; output_index: INTEGER; NOT NULL; no default; pk=0; descriptor_digest: BLOB; NOT NULL; no default; pk=0; kind: TEXT; NOT NULL; no default; pk=0; source_store_ids: TEXT; NOT NULL; '[]'; pk=0; state: TEXT; NOT NULL; no default; pk=0; next_occ_id: INTEGER; nullable; no default; pk=0; observed_stamp: TEXT; nullable; no default; pk=0; recovered_store_id: INTEGER; nullable; no default; pk=0 | idx_emissions_lineage_digest, sqlite_autoindex_ledger_emissions_1 |
| `ledger_meta` | key: TEXT; nullable; no default; pk=1; value: BLOB; NOT NULL; no default; pk=0 | sqlite_autoindex_ledger_meta_1 |
| `lineage_marks` | lineage: TEXT; nullable; no default; pk=1; max_store_id: INTEGER; nullable; no default; pk=0; legacy_proof_digest: BLOB; nullable; no default; pk=0; max_leaf_id: INTEGER; nullable; no default; pk=0; stamp_policy: TEXT; NOT NULL; 'normal'; pk=0 | sqlite_autoindex_lineage_marks_1 |
| `message_relations` | relation_id: INTEGER; nullable; no default; pk=1; store_id: INTEGER; NOT NULL; no default; pk=0; kind: TEXT; NOT NULL; no default; pk=0; related_store_id: INTEGER; nullable; no default; pk=0; ordinal: INTEGER; nullable; no default; pk=0; observed_at: REAL; nullable; no default; pk=0; created_at: REAL; nullable; no default; pk=0 | idx_msg_relations_observed, idx_msg_relations_related, idx_msg_relations_store |
| `messages` | store_id: INTEGER; nullable; no default; pk=1; session_id: TEXT; NOT NULL; no default; pk=0; source: TEXT; nullable; ''; pk=0; conversation_id: TEXT; nullable; ''; pk=0; role: TEXT; NOT NULL; no default; pk=0; content: TEXT; nullable; no default; pk=0; tool_call_id: TEXT; nullable; no default; pk=0; tool_calls: TEXT; nullable; no default; pk=0; tool_name: TEXT; nullable; no default; pk=0; timestamp: REAL; NOT NULL; no default; pk=0; token_estimate: INTEGER; nullable; 0; pk=0; pinned: INTEGER; nullable; 0; pk=0; ingested_at: REAL; nullable; no default; pk=0; observed_at: REAL; nullable; no default; pk=0; observed_at_source: TEXT; nullable; no default; pk=0; sender_id: TEXT; nullable; no default; pk=0; sender_source: TEXT; nullable; no default; pk=0 | idx_msg_conversation_observed, idx_msg_conversation_session, idx_msg_session, idx_msg_session_ts, idx_msg_source_session |
| `metadata` | key: TEXT; nullable; no default; pk=1; value: TEXT; nullable; no default; pk=0 | sqlite_autoindex_metadata_1 |
| `occurrence_bindings` | occ_id: INTEGER; nullable; no default; pk=1; lineage: TEXT; NOT NULL; no default; pk=0; ordinal: INTEGER; nullable; no default; pk=0; store_id: INTEGER; nullable; no default; pk=0; identity_digest: BLOB; NOT NULL; no default; pk=0; stamp: TEXT; nullable; no default; pk=0; disposition: TEXT; NOT NULL; no default; pk=0; tag: TEXT; nullable; no default; pk=0; key_rank: INTEGER; nullable; no default; pk=0; view_seq: INTEGER; NOT NULL; no default; pk=0 | idx_occ_lineage_digest, idx_occ_lineage_ordinal, idx_occ_lineage_stamp_digest, idx_occ_store |
| `session_lineage` | child_session_id: TEXT; nullable; no default; pk=1; parent_session_id: TEXT; NOT NULL; no default; pk=0; child_subagent_id: TEXT; nullable; no default; pk=0; parent_subagent_id: TEXT; nullable; no default; pk=0; child_role: TEXT; nullable; no default; pk=0; source: TEXT; NOT NULL; no default; pk=0; created_at: REAL; nullable; no default; pk=0 | idx_session_lineage_parent, sqlite_autoindex_session_lineage_1 |
| `stamp_observations` | lineage: TEXT; NOT NULL; no default; pk=0; occ_id: INTEGER; NOT NULL; no default; pk=0; stamp: TEXT; NOT NULL; no default; pk=0; kind: TEXT; NOT NULL; no default; pk=0; view_seq: INTEGER; NOT NULL; no default; pk=0; source: TEXT; NOT NULL; 'ledger'; pk=0 | idx_stamp_obs_lineage_stamp, sqlite_autoindex_stamp_observations_1 |
| `summary_nodes` | node_id: INTEGER; nullable; no default; pk=1; session_id: TEXT; NOT NULL; no default; pk=0; depth: INTEGER; NOT NULL; 0; pk=0; summary: TEXT; NOT NULL; no default; pk=0; token_count: INTEGER; nullable; 0; pk=0; source_token_count: INTEGER; nullable; 0; pk=0; source_ids: TEXT; NOT NULL; '[]'; pk=0; source_type: TEXT; NOT NULL; 'messages'; pk=0; created_at: REAL; NOT NULL; no default; pk=0; earliest_at: REAL; nullable; no default; pk=0; latest_at: REAL; nullable; no default; pk=0; expand_hint: TEXT; nullable; ''; pk=0 | idx_nodes_session_depth, idx_nodes_session_depth_node, idx_nodes_session_latest, idx_nodes_session_node |
<!-- End generated table. -->

## How a consumer may write

Write core rows only through the contracted `MessageStore` methods below. `MessageStore` opens the store; `SummaryDAG(db_path)` is contracted for schema creation only. Close opened stores after use. `connection` is a read-oriented property and may be `None` after close; only consumer-owned rows have the SQL-write exception below.
`append_batch` returns stable store ids; `ingest_atomic` and `atomic` own transaction boundaries. Bind host uids and relations through their methods; `record_session_lineage` keeps the first parent link. Metadata helpers read and update JSON values. `register_session_purge` registers a named callback for consumer-owned session data in the caller transaction; it grants no automatic-maintenance deletion authority.
`db_bootstrap.SCHEMA_VERSION` is 5 (the on-disk schema stamp, distinct from this contract version). `mark_migration_step_complete` records a named additive step. The reader API is `StoreReader` (construction, preparation, envelope formatting and tool calls) and `handle(reader, request)`, which handles a JSON-RPC request and returns a response or `None` for a notification.

<!-- Generated signatures; annotations omitted by inspect.Signature.replace. -->
| Callable | Signature |
| --- | --- |
| `dag.SummaryDAG` | `(db_path)` |
| `db_bootstrap.mark_migration_step_complete` | `(conn, step_name)` |
| `mcp_reader.StoreReader` | `(store_path, hermes_home, *, config, environ=None)` |
| `mcp_reader.StoreReader.call` | `(self, name, arguments)` |
| `mcp_reader.StoreReader.envelope` | `(self, raw, context)` |
| `mcp_reader.StoreReader.prepare` | `(self, *, tokenizer_wait_s=30.0)` |
| `mcp_reader.handle` | `(reader, request)` |
| `store.MessageStore` | `(db_path, *, ingest_protection_config=None, hermes_home='')` |
| `store.MessageStore.add_host_uid_bindings` | `(self, lineage_key, rows)` |
| `store.MessageStore.add_message_relations` | `(self, groups)` |
| `store.MessageStore.append_batch` | `(self, session_id, messages, token_estimates=None, source='', conversation_id='')` |
| `store.MessageStore.atomic` | `(self, *, immediate=False)` |
| `store.MessageStore.close` | `(self)` |
| `store.MessageStore.connection` | `(self)` |
| `store.MessageStore.host_uid_uids_of_stores` | `(self, lineage_key, store_ids, kinds=('canonical', 'version'), *, require_disagree=False)` |
| `store.MessageStore.ingest_atomic` | `(self, *, admission_retry=False, prelock=False)` |
| `store.MessageStore.read_metadata_json` | `(self, key)` |
| `store.MessageStore.record_session_lineage` | `(self, child_session_id, parent_session_id, *, source, child_subagent_id='', parent_subagent_id='', child_role='')` |
| `store.MessageStore.update_metadata_json` | `(self, key, update)` |
| `store.register_session_purge` | `(name, purge)` |
<!-- End generated signatures. -->

## What a consumer owns

A consumer owns its own tables and metadata keys under its own prefix. It may write only those rows through `connection` and only inside `store.atomic()` or `store.ingest_atomic()`, which hold the store's write lock. This is the one exception to `connection`'s read-only rule: it is read-only for everything else. The adapter owns `capture_events` and `capture:` metadata; its DDL remains in `claude_code_store.py`, only in a capture-owned store, under the exception in [Schema changes](process/schema-changes.md). Consumers must not write another consumer’s rows.
The adapter spool is a JSON pointer object, not transcript content. It is assembled dynamically in `adapters/claude-code/bin/lcmx_hook.py`; there is no standalone record schema/type definition to hash.
Fields: `host` (host string), `event` (event name), `session_id` and optional `agent_id` (identity); `transcript_path` and `agent_transcript_path` (absolute paths or null); `source`, `reason`, `agent_type` (optional strings); `observe_only` (boolean) and `problem` (optional refusal code).
`observed_at` is wall-clock seconds; `seq`, `hook_started_ns` and `newest_ns` are diagnostic nanosecond stamps (the newest coalesced start may be null). They do not define transcript order. `prompt_id` is a hash or null. `local_gc` is a boolean qualification flag.
`seen` and `compact_prompts` are coalesced lists of event names and prompt hashes. `transcript_seen` and `agent_transcript_seen` are booleans or null (unknown). `start_source` and `started_at` retain session-start provenance when available. A hook atomically publishes a `.json` pointer; the worker claims it as `.claimed`, replays claims after crashes, and preserves failed pointers as `.failed`.

## Reached today, not contracted

These existing private dependencies are not promoted in version 1. Locations refer to the main tree at introduction.
| Private name | Consumer and location |
| --- | --- |
| `MessageStore._ingest_protection_config` | Capture writer, `claude_code_store.py:796` |
| `MessageStore._hermes_home` | Capture writer, `claude_code_store.py:797` |
| `command._resolve_storage_dtype` | Reader, `mcp_reader.py:314` (import at 36) |
| `command._resolve_store_dim` | Reader, `mcp_reader.py:315` (import at 36) |
| `vector_store._VECTOR_BYTEORDER` | Reader, `mcp_reader.py:322` (import at 52) |
| `store._database_identity` | Read-only store view, `readonly_store.py:319` (import at 46) |
| `db_bootstrap._fts_missing_triggers` | Read-only capability probe, `readonly_store.py:246` (import at 34) |

## Invariants

Preserve one stored row per host occurrence per lineage ([I1](invariants.md), with its named exceptions); no stored row deleted or altered ([I2](invariants.md)); summary nodes, provenance and frontier committed together ([I14](invariants.md)); every emitted tool result after its call ([I8](invariants.md)), with assistant/tool groups never split by the fresh tail or a leaf cut ([I9](invariants.md)); the newest user turn reaching the model ([I4](invariants.md)); stable message ids across hosts; and session ownership and chronology. Order message rows by `store_id` within their session, not pointer arrival or diagnostic time; forks retain their own lineage. This contract adds no writer-sharing protocol or migration.
The SHA-256 covers UTF-8 canonical JSON (`sort_keys=True`, compact separators, `ensure_ascii=False`): sorted tables with their ordered PRAGMA table_info rows and sorted index names, callables keyed by sorted module-qualified name with annotation-free signatures, and SCHEMA_VERSION. A fresh isolated store enables the contracted named features before inspection; no stored user content is hashed.
Run `tests/compat/test_store_contract.py`. For a surface change, raise the contract version by exactly one, refresh the digest and add a dated changelog row with that digest and a one-line reason. An unchanged digest keeps the same version. `scripts/store_contract_gate.py --base-ref <ref>` checks that discipline against the PR base; introduction without a base doc requires version 1.

## Changelog

| Version | Digest | Date | Reason |
| --- | --- | --- | --- |
| 1 | sha256:76644a990e50b9613697704dcb05b349a952d44ea22187bd768b0eab9a9badc9 | 2026-11-16 | Record the shared store, capture ledger and reader surface. |
