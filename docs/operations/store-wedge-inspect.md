# Read-only store wedge inspection

Use this for repeated `zero_leaf_streak` / `compaction_progress` doctor warnings
and named retained-row stops. See [#1162](https://github.com/electricsheephq/lcm-x/issues/1162)
and [#1598](https://github.com/electricsheephq/lcm-x/issues/1598).

Stop the host and make a separate, consistent SQLite copy. Never point this
script at a running store. It opens only the operator-made copy with `mode=ro`,
`query_only=ON`, and one read transaction. It refuses schema mismatches and any
`-wal`, `-shm`, or `-journal` sidecar. It also refuses a WAL-mode header, which
SQLite could create sidecars for; prepare a standalone rollback-journal copy
using SQLite backup, retaining the original and its companions unchanged.

```sh
python scripts/store_wedge_inspect.py inspect --copy COPY.db \
  --session-id SESSION --conversation-id CONVERSATION --json
```

Output contains fixed codes, counts and numeric source IDs, never message text,
scope identifiers, UIDs, SQL parameters or paths. It makes no model calls.

| Field | Meaning and operator action |
| --- | --- |
| `polluted_copies=suspected` | Equal stored identities or a salvage pair; inspect IDs locally. Equal text, timestamps and reused call IDs do not prove duplication. Legitimate repeated user turns remain distinct. |
| `polluted_copies=confirmed` | Equal stored identities share a captured host UID, lineage and version. Keep every source; this is diagnosis, not permission to delete. |
| `prefix_over_leaf_budget` | Cost through the named retained source exceeds the leaf budget, respecting the first oversized-group exception. Total backlog alone is insufficient. Follow #1162; do not widen budgets automatically. |
| `blocker_in_fresh_tail` | The named source has an occurrence inside the captured protected tail. Keep the tail; follow #1162. |
| `empty_input_kind` | Actual attribution branch: `retained`, `contiguity`, `no_source`, or `none`; budget diagnosis does not relabel contiguity. |
| `unreachable_source_ids` | Sources neither covered by validated summary lineage nor represented in the view, including below the frontier. Preserve the copy and report IDs under #1598. `null` means unassessed. |
| `view_state=unknown` | Missing/mismatched view, config, counter or host evidence. This is never healthy. Preserve evidence; do not infer a blocker from the first source above the frontier. |
| `healthy=true` | Complete supported evidence has no identified failure for this snapshot; future summarizer success is unproven. |

For view-dependent checks add `--view-snapshot SNAPSHOT.json` and
`--effective-config CONFIG.json`. These sensitive local inputs must not be
attached to a public report. Snapshot keys: `source_sha256`, `session_id`,
`conversation_id`, nonempty `capture_id`, exact ordered `messages`, and
`effective_config` equal to CONFIG.json. Source IDs supplied in messages are
not trusted as mapping evidence.

Config keys: `leaf_chunk_tokens`, effective `selection_budget`, `effective_trigger`,
`fresh_tail_count`, `fresh_tail_max_tokens`, `pressure_yield_limit`,
`context_window`, `token_counter`, and full `lcm_config` (all `LCMConfig` fields).
Capture actual effective settings, not environment defaults. V1 supports the
`fallback-v1` counter and ordinary textual raw views. Unavailable counters,
external payloads, summary carriers and host-dependent identity forms produce
`unknown`, retaining database-only findings. No host files are opened.

**Does NOT repair:** no reconstruction, frontier rewind, replay manifest,
ingest, compression, schema migration or live host adoption is performed.
Reconstruction waits for supported host adoption in
[#1731](https://github.com/electricsheephq/lcm-x/issues/1731).
