# Read-only store wedge inspection

Use this for repeated `zero_leaf_streak` / `compaction_progress` doctor warnings
and named retained-row stops. See [#1162](https://github.com/electricsheephq/lcm-x/issues/1162)
and [#1598](https://github.com/electricsheephq/lcm-x/issues/1598).

Stop the host and make a separate, consistent SQLite copy. Never point this
script at a running store. It opens only the operator-made copy with `mode=ro`,
`query_only=ON`, and one read transaction. It refuses schema mismatches and any
`-wal`, `-shm`, or `-journal` sidecar. It also refuses a WAL-mode header, which
SQLite could create sidecars for; prepare a standalone rollback-journal copy
using SQLite backup **and then converting the copy to DELETE journal mode**.
Backup alone can preserve a WAL-mode header. With the host stopped, keep the
original and its companions, choose a new destination, and run:

```python
import sqlite3
from contextlib import closing
from pathlib import Path

source = Path("SOURCE.db").resolve(strict=True)
copy = Path("COPY.db")
copy.touch(exist_ok=False)
with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
    with closing(sqlite3.connect(copy)) as destination:
        original.backup(destination)
        assert destination.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
assert not any(Path(str(copy) + suffix).exists() for suffix in ("-wal", "-shm", "-journal"))
with copy.open("rb") as header:
    assert header.read(20)[18:20] == b"\x01\x01"
```

If either check fails, stop and prepare another standalone copy; do not delete
sidecars to make a copy pass. The conversion writes only the copy and happens
before inspection and before capturing its hash.

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
| `polluted_copies=none` / `unknown` | No suspected pair found / not assessed. A first stored tool placeholder without an earlier different-content copy is not a salvage pair. |
| `polluted_source_ids`, `confirmed_source_ids` | Sorted numeric source IDs suspected of pollution / confirmed by the captured binding evidence. Empty lists mean none found. |
| `source_rows`, `covered_rows`, `frontier` | Selected-session stored-row count, validated lineage-covered count (including verified rotation predecessors), and stored integer frontier. Non-integer frontiers are refused, never echoed. |
| `prefix_over_leaf_budget` | Cost through the named retained source exceeds the leaf budget, respecting the first oversized-group exception. Total backlog alone is insufficient. Follow #1162; do not widen budgets automatically. |
| `blocker_in_fresh_tail` | The named source has an occurrence inside the captured protected tail. Keep the tail; follow #1162. |
| `empty_input_kind` | Actual attribution branch: `retained`, `contiguity`, `no_source`, or `none`; `unknown` means unassessed. `none` also covers the product's no-op when no raw backlog exists outside the fresh tail and no hidden backlog is eligible. Budget diagnosis does not relabel contiguity. |
| `blocker_source_ids` | The named retained source, or an empty list when no blocker is found. |
| `prefix_tokens`, `prefix_rows`, `leaf_budget`, `selection_budget` | Measured prefix cost/count through that source and the captured effective budgets. Prefix cost/count are zero when no blocker is found. |
| `selection_failed` | The assessed selection cannot proceed, or the protected tail reaches the effective trigger. A no-backlog product no-op is not a selection failure. |
| `unreachable_source_ids` | Sources neither covered by validated summary lineage nor represented in the view, including below the frontier. Preserve the copy and report IDs under #1598. `null` means unassessed. |
| `view_state=complete` / `unknown` | Complete supported occurrence evidence / missing, mismatched, malformed or unsupported evidence. Unknown is never healthy. Preserve evidence; do not infer a blocker from the first source above the frontier. |
| `evidence_code` | Optional fixed reason why view evidence could not be used; database findings remain available. See codes below. |
| `healthy=true` | Complete supported evidence has no identified failure for this snapshot; future summarizer success is unproven. |

Unassessed view booleans (`prefix_over_leaf_budget`, `blocker_in_fresh_tail`) and
`unreachable_source_ids` are `null`. Detailed selection fields are absent until
the probe succeeds. Database findings remain available even if optional view
evidence is malformed. Without view/config evidence, `empty_input_kind` and
`view_state` are `unknown`, and `healthy` is false.

Exit **0** means inspection completed, including an unknown or unhealthy
diagnosis; it is not a health assertion. Exit **2** with `{"error": CODE}` means
the inspection was refused or failed. `--json` is optional because JSON is the
only inspection output. Normal argument help exits 0; argument syntax errors
exit 2 using argparse's usage message. Inspection failures emit no traceback.

| Error code | Meaning |
| --- | --- |
| `sidecar_present` | A WAL, SHM or journal companion exists; refused before opening SQLite, or detected by the final check. |
| `wal_mode_copy` | Header is not standalone rollback-journal mode; convert a new copy first. |
| `schema_mismatch` | SQLite header, schema stamp or required columns do not match. |
| `scope_mismatch` | Conversation lifecycle is absent or its active session differs; finalized-only sessions remain refused. |
| `invalid_dag` | Summary lineage has an invalid edge, source list, source type or cycle. |
| `unverified_ancestry` | Summary ancestry is outside verified session ownership, or rotation-predecessor metadata is missing, contradictory, foreign to the conversation, or cyclic. |
| `invalid_store` | Malformed store values or database read failure, including a non-integer frontier. |
| `input_changed` | Copy hash changed during inspection; preserve the evidence and stop. |
| `invalid_input` | Other input failure or unexpected exception, including output serialization failure. |

| Evidence code | Meaning (`view_state=unknown`, database findings retained) |
| --- | --- |
| `config_incomplete` | Effective config is absent, incomplete, malformed, has extra LCM fields or invalid required budgets. |
| `token_counter_unavailable` | Captured counter cannot be reproduced without a download or global counter change. |
| `snapshot_invalid` | View is absent, malformed, mismatched to the copy/scope/capture/config, or cannot be processed. |
| `unsupported_view_evidence` | Special host identity, external payload or carrier evidence is required. |
| `source_time_unavailable` | Supported schema-v5 copy lacks optional source-time columns; database findings remain available, but view attribution is unassessed. |
| `identity_anchor_unsupported` | A timestamped user message has an owned composite witness, or an unmapped user message is an exact join of stored replay-identity forms with a same-stamp user donor in the conversation and supported session scope (including alternate stamps). Mapped replays skip the candidate-composite guard. Engine claim resolution can write relations, so all view findings remain unassessed. |
| `leaf_admission_unassessed` | Raw backlog is below the context-capped working leaf threshold without a captured pressure-yield waiver, or dynamic sizing is enabled. V1 lacks the force, fit, sweep and deferred-maintenance state needed to assess admission safely. |
| `copy_exclusions_unassessed` | Nonempty persisted host UID bindings require active host lineage and UID mode unavailable in V1, so view copy exclusions are unassessed. With bindings absent or empty, the production helper reproduces E1 byte-identical assistant tool-call/tool-result exclusions using selected-session ownership and node/leaf coverage. Database copy findings remain available. |
| `ignore_patterns_unavailable` | Configured ignore patterns cannot all be compiled with the production engine in this process (including missing optional `regex`). View filtering and attribution remain unassessed. |
| `unverified_ancestry` | Captured compaction carry ancestry, or the host ancestry of a same-stamp user donor outside the selected session, cannot be established. |
| `blocker_evidence_missing` | The retained blocker cannot be located in the captured view. |

For view-dependent checks add `--view-snapshot SNAPSHOT.json` and
`--effective-config CONFIG.json`. These sensitive local inputs must not be
attached to a public report. Snapshot keys: `source_sha256`, `session_id`,
`conversation_id`, nonempty `capture_id`, exact ordered `messages`, and
`effective_config` equal to CONFIG.json. Source IDs supplied in messages are
not trusted as mapping evidence. `source_sha256` is the lowercase SHA-256 hex
digest of the **exact COPY.db file bytes after journal-mode conversion and all
copy preparation**, not the original store, a SQL dump, or the snapshot JSON.
A changed or mismatched hash leaves view findings unknown.

Config keys: `leaf_chunk_tokens`, effective `selection_budget`, `effective_trigger`,
`fresh_tail_count`, `fresh_tail_max_tokens`, `pressure_yield_limit`,
`context_window`, `token_counter`, and full `lcm_config` (all `LCMConfig` fields).
Capture actual effective settings, not environment defaults. V1 supports the
`fallback-v1` counter and ordinary textual raw views. Unavailable counters,
external payloads, summary carriers and host-dependent identity forms produce
`unknown`, retaining database-only findings. No host files are opened.
Legacy blank, NULL and padded conversation IDs share the selected conversation's
ownership. Replay matching considers every row in the selected session, while
coverage and pollution stay conversation-scoped. A replay mapping to a foreign
conversation leaves all view findings unassessed. A summary DAG wholly owned by
another conversation is skipped; mixed or malformed target lineage is refused.
Source lists must be unique; message sources follow store order, and child-node
sources must be nonempty and follow creation time. Empty message-source leaves
are valid (filtered dependent replies) and contribute no coverage. Parent message sources inside a
persisted carry range are accepted after validating the range and conversation;
view ancestry remains unassessed without host evidence. Rotation-predecessor
chains allow empty hops (which contribute no rows), and every hop with rows
must have consistent conversation ownership, including all-blank or NULL predecessors;
missing or contradictory evidence and cycles refuse with `unverified_ancestry`.
Validated predecessor message sources contribute to coverage even though raw
rows remain in their original session.
Configured ignored active or mapped stored messages are removed before selecting
the probe leaf. Dependent replies remain in selection and are omitted only from
summarizer input. Persisted dependent-reply records, active replay placeholders,
and special identity metadata leave view findings unassessed before attribution.
The no-op check keeps shown-but-unmapped retained occurrences in the backlog,
so they can produce a named stop. It uses production's bounded, extendable scan
and subtracts node-covered copies only when that scan is not truncated.
Admission uses production's working-leaf helper with the captured context window
and honours an already-engaged pressure-yield waiver. The raw configured leaf
budget still governs attribution and prefix-budget reporting. Attribution uses
the larger of that budget and the selected input's token cost, matching
production's budget floor.

**Does NOT repair:** no reconstruction, frontier rewind, replay manifest,
ingest, compression, schema migration or live host adoption is performed.
Reconstruction waits for supported host adoption in
[#1731](https://github.com/electricsheephq/lcm-x/issues/1731).
