# Identity ledger (O3)

Status: decided plan. Nothing in it changes behaviour by itself. It applies from v0.26.0 (host message identity in
shadow mode, #643), and each step below ships as its own reviewed PR.

## Why this exists

On every ingest, LCM-X decides for each host row whether it is new or a row it already stored. Today that decision
is made by content rules built up release by release since #111:
- replay identity tuples;
- commit proofs and emission descriptors;
- the #436 identity anchor;
- payload and prefix matchers;
- the placeholder ledger;
- recognisers for rows LCM generated.

An inventory at commit `9afb80d0` counts **460 rules in nine systems**.

A host that stamps a `message_uid` on each message answers most of those questions directly. v0.26.0 records the
uids in shadow mode only: it counts agreement and decides nothing.

This ledger records two things:
- for every rule, what happens to it once a uid decides identity (`LCM_HOST_MESSAGE_UID=on`, "slice C");
- which rules are dead code today.

The per-rule table is in [identity-ledger-inventory.md](identity-ledger-inventory.md).

## Decisions

| decision | meaning | rules |
|---|---|---|
| **KEEP** | Stays, and keeps deciding. Either it is not a question a uid answers (lifecycle, publication, store hygiene, record formats, byte splits, ownership ranges), or it is the uid route itself. | 183 |
| **KEEP (legacy reader)** | Reads state that a deleted writer persisted: native recovery, removed in v0.25.0, and proofs written before v4 descriptors. Frozen. Deleted only when a read-only count finds none of that state on managed stores. | 7 |
| **DEMOTE-UNDER-ON** | Becomes the fallback. In `on` mode it runs only for rows the uid route cannot decide. On hosts that send no uid it keeps deciding exactly as today. Frozen: bug fixes only. | 189 |
| **DEMOTE+MERGE** | Demoted, and first folded into one sibling implementation. | 29 |
| **MERGE** | Copies of one predicate become one function. The refactor is behaviour-neutral. A parity test runs every copy's existing test inputs through the merged function. | 33 |
| **DELETE** | No production caller, or unreachable. Removed with proof: a grep of code, tests and docs, the git history, and the full suite. | 19 |

No rule is deleted just because uids exist. LCM keeps supporting hosts that send no uid, so every demoted rule stays
for as long as that support does.

### Counts per system

| system | KEEP | KEEP (legacy reader) | DEMOTE-UNDER-ON | DEMOTE+MERGE | MERGE | DELETE | total |
|---|---|---|---|---|---|---|---|
| SYS-1 Cursor/replay matcher | 15 | 2 | 53 | 18 | 10 | 5 | 103 |
| SYS-2 Commit proof, emission descriptors & occurrence projection | 18 | 5 | 48 | 1 | 3 | 4 | 79 |
| SYS-3 #436 identity anchor | 17 | 0 | 47 | 1 | 4 | 1 | 70 |
| SYS-4 Externalized payload & prefix matching | 17 | 0 | 32 | 1 | 2 | 7 | 59 |
| SYS-5 Host uid (shadow/on, engine uids) | 57 | 0 | 0 | 0 | 5 | 0 | 62 |
| SYS-6 Placeholder ledger & ignore policy | 5 | 0 | 5 | 3 | 4 | 1 | 18 |
| SYS-7 Generated-row recognition (scaffold, carrier, survival fit, retained anchor) | 8 | 0 | 3 | 5 | 4 | 0 | 20 |
| SYS-8 Lifecycle, rotation, publication & DAG lineage | 41 | 0 | 0 | 0 | 1 | 0 | 42 |
| SYS-9 Store columns & metadata hygiene | 5 | 0 | 1 | 0 | 0 | 1 | 7 |
| **all** | **183** | **7** | **189** | **29** | **33** | **19** | **460** |

## Principles

1. **A uid decides only what it can prove.**
   - A bound uid identifies the stored row.
   - An engine uid identifies a row LCM generated.
   - `_absorbed_message_uids` names the constituents of a host merge.

   A uid does not split bytes (composite remainders), own ranges (carry ranges) or validate a persisted record.
   The rules that do those things stay.
2. **First sight goes through the content rules once.** An unbound uid can belong to a row stored before the host
   stamped uids. So the content rules decide that row once, and the result is bound (binding kinds `canonical` and
   `version`). From then on the uid decides. The content matchers run once per row, when it is bound, instead of on
   every replay.
3. **Readers of persisted state outlive their writers.** This is the rule v0.25.0 applied when it removed native
   recovery. A store written by an older version can still hold that state, and a plugin-only rollback to the
   previous GA must read what this version writes.
4. **Fail open to today's decision.** Any exception on the uid route returns the content decision and is counted,
   as the #436 anchor's fail-open does today.
5. **No schema version change.** Bindings live in the droppable `host_uid_bindings` side table. A new column comes
   only through `add_column_if_missing`, with a rollback-reader test.
6. **One step per PR.**
   - Steps: the DELETE batch first, then one MERGE cluster per PR, then the planner with `on` mode.
   - A step that changes behaviour is red-first.
   - Every step is reviewed by a model other than its author.

## The ingest planner (slice C sketch)

Today an ingest decides in four stages:
1. The ingest cursor comes from an OR of replay proofs in `_find_reconciled_cursor_for_store_tail` (the 19 rules
   of the cursor cluster below).
2. The #436 anchor then adjusts composites and remainders.
3. The placeholder ledger then handles ignored rows.
4. The host-uid shadow classifies the outcome after the decision.

Slice C puts one planner in front of that chain:

```text
plan_ingest(view):
  for each host row, in order:
    engine uid of this lineage          -> GENERATED (never stored)
    uid bound in this lineage           -> REPLAY(bound store_id)          binding kind canonical or version
    all _absorbed_message_uids bound    -> COMPOSITE(constituents), remainder by the byte split (KEEP rules)
    otherwise                           -> FALLBACK: today's content rules decide once, then the uid is bound
  check the plan invariants (KEEP rules):
    a stored row is consumed at most once; a tool segment is replayed whole or not at all;
    carry ranges and the frontier hold; nothing inside the fresh tail is dropped
  apply in one store transaction (metadata rows with the INSERT, as today)
  any exception -> today's content decision, counted
```

The planner adds no new answers. Shadow mode already computes the uid answer for each row and compares it with the
content decision:
- the `host_uid_*` counters in `/lcm doctor`;
- the proof kinds `stored_new`, `version_new` and `anchor_replay`.

`on` mode changes which answer is applied. The counters keep running, so a disagreement stays visible.

**Entry conditions for `on` mode:**
- **Shadow gate (v0.26.0).** At least 99% agreement over at least 200 counted bindings (`stored_new`,
  `version_new`) across at least 24 h. Every disagreement is explained with row evidence. Durable shadow errors are 0.
- **A deployed host carries uids.** Today no managed runtime sends them.
- **Persistence-only probe.** The host declares `message_uid` persistence-only. The probe runs once per process
  and fails closed.
- **Default stays `shadow`.** `on` ships as an opt-in value, and the default stays `shadow` for at least one minor.

## What each system keeps

- **SYS-1 Cursor/replay matcher (103).**
  - **KEEP:** the replay identity tuple (role, normalised content, tool call id, tool calls, tool name). It is the
    only identity a row without a uid has.
  - **DEMOTE:** the OR-ed replay proofs (sanitised suffix, raw suffix, tool-pair end, stale window, durable proof
    cursor). 19 of them are the planner cluster.
  - **Legacy readers:** the native-recovery snapshot digests.
- **SYS-2 Commit proof, emission descriptors and occurrence projection (79).**
  - **KEEP:** the proof record format, its digest and version contract, and proof-source selection. A rollback
    reader reads them.
  - **DEMOTE:** descriptor binding, because an engine uid identifies the generated occurrence.
  - **Legacy readers:** the durable native proof walk, the native digest adopted by an empty rollover session,
    and pre-v4 occurrence identities.
- **SYS-3 #436 identity anchor (70).**
  - **DEMOTE:** candidate scope, prematch and rewind.
  - **KEEP:** the byte split of a composite remainder, and the relation writes that record it.
- **SYS-4 Externalised payload and prefix matching (59).**
  - **KEEP:** the payload session-ownership guards. They protect privacy scope, not identity.
  - **DEMOTE:** the prefix matchers.
- **SYS-5 Host uid (62).** **KEEP.** This becomes the deciding route; five rules merge into their siblings.
- **SYS-6 Placeholder ledger and ignore policy (18).**
  - **KEEP:** the persisted record formats.
  - **MERGE, then DEMOTE:** the ordinal and budget matchers.
- **SYS-7 Generated-row recognition (20).**
  - **DEMOTE to engine uids:** summary-carrier recognition.
  - **KEEP:** the carrier split.
  - **MERGE:** the eight scaffold classifier copies become one.
- **SYS-8 Lifecycle, rotation, publication and DAG lineage (42).** **KEEP.** These rules decide when to compact and
  publish, not which row is which.
- **SYS-9 Store columns and metadata hygiene (7).** **KEEP,** except one dead helper.

## DELETE list (19)

Each row was checked against the code at `9afb80d0`. The deleting PR repeats the grep at its own base.

| id | where | why it is dead |
|---|---|---|
| R017 | `reconcile._session_end_replay_snapshot_metadata_key` | no caller |
| R040 | `reconcile._strip_inline_persisted_output_generation_identity` and its two OR terms | both terms need an unrecoverable marker, which always hits the earlier `continue` veto (since the #313 era) |
| R057 | parameter `session_count` of `_find_reconciled_cursor_for_store_tail` | never read |
| R086 | `engine._is_active_context_droppable_identity` | no caller; `reconcile` has its own inline copy |
| R087 | `engine._ignored_message_is_quarantinable_assistant` | no caller |
| R133 | second `_cursor_from_durable_commit_proof` call in `_reconcile_ingest_cursor_from_store` | same input as the first call, and nothing between them changes what the proof reads; the only difference is a diagnostic label on a zero cursor |
| R164 | native-lossy veto in `_ingest_messages` | the in-memory commit proof is always written with `native = False` (compaction is its only writer since v0.25.0) |
| R168 | `engine._remap_cursor_through_native_host_repair` | reachable only through a native in-memory proof (see R164) |
| R181 | legacy native fields written into the commit proof | written as false; readers use `.get`, so the previous GA reads a proof without them unchanged |
| R232 | `identity_anchor._identity_anchor_record_versions` (the `supersedes` relation) | written, never read; existing rows are inert and are purged with their messages |
| R258 | `recovered_identity_content` fallback in `_message_replay_identity` | unreachable: the earlier `recovered_with_stat` branch always fires first |
| R282 | `prefix_matching._messages_match_fingerprint_prefix` | no reference in code, tests or docs |
| R283 | `prefix_matching._messages_match_lcm_bypass_prefix` | no reference |
| R284 | `prefix_matching._matching_lcm_bypass_prefix_count` | only caller is R283; the engine calls `_matching_lcm_bypass_prefix_evidence` directly |
| R285 | `prefix_matching._messages_match_lcm_normal_prefix` | no reference |
| R302 | marker filters `require_missing_file_generation_metadata` and `persisted_output_file_size/mtime_ns/ctime_ns` of `find_externalized_tool_result_content_for_call` | no caller passes them |
| R306 | `externalize.reassign_externalized_payloads` | only a test calls it; the production call was removed in #269, and #680 replaced it with rotation lineage |
| R381 | `placeholder_ledger._subtract_placeholder_digest_counts` | no caller; the engine inlines the same arithmetic |
| R460 | `placeholder_ledger._ignored_placeholder_metadata_key` (singular) | no caller |

## MERGE clusters

| cluster | rules | size | condition |
|---|---|---|---|
| Cursor OR-lattice → the ingest planner | R018 R027 R028 R036 R041 R045–R056 R064 R134 | 19 | merges into the planner (slice C), not on its own |
| Merge-survivor recognisers (#535, #436 R2–R3, host-uid composite) | R077 R114 R115 R117 R209 R220 R240 R334 R335 R343 R392 R402 | 12 | |
| Scaffold classifiers | R125 R196 R395 R396 R397 R406 R407 R408 | 8 | |
| Session-end prefix arbitration | R274 R287–R291 R309 R310 | 8 | waits on the tie-break check (finding (b) below) |
| Placeholder ordinal and budget matchers | R062 R375 R378 R386 R388 R389 | 6 | |
| State-store lineage walkers | R193 R195 R325 R326 R459 | 5 | waits on the fork-child check (finding (a) below) |
| Content identity tuples and digests | R001 R118 R269 R275 R370 | 5 | the digest bytes must not change: rollback readers compare them |
| Payload session-ownership guards | R270 R271 R294 R307 R308 | 5 | |
| Lossy-redaction fence copies | R009 R035 R065 R198 | 4 | |
| Ignored-backlog classifier copies | R384 R385 R391 R411 | 4 | |
| Proof consult vs reconcile digest walk | R141 R163 R165 R166 | 4 | |
| Tool-segment dedupe | R078 R079 R233 | 3 | |
| Row-ownership predicate | R242 R423 R447 | 3 | |
| Host stamp and `alt_stamp` identity | R090 R201 R236 | 3 | |

## Contradictions resolved

1. **Native-recovery state.** Readers proposed DELETE, DEMOTE and KEEP for the same rules.
   - **Resolution, in-memory side.** The in-memory commit proof has one writer, compaction, which always sets
     `native = False`. So its consumers (R164, R168) and the legacy fields it writes (R181) are deleted.
   - **Resolution, persisted side.** The durable native proof walk (R143–R145) and the native snapshot digests
     (R015, R088, R131) read state that a store written before v0.25.0 can still hold. They are KEEP (legacy reader).
   - **Sunset:** a read-only count of native digest keys and native durable proofs on managed stores returns zero.
2. **Store-id reuse (R250, R317, R401, R454).** One reader said ids are never reused; others found guards against
   reuse.
   - `messages.store_id` has been `INTEGER PRIMARY KEY AUTOINCREMENT` since the first commit, so ids are not reused.
   - The plain `INTEGER PRIMARY KEY` table in `db_bootstrap.py` is an in-memory scratch for a schema-contract check,
     not a store.
   - **Resolution:** the purge of relations and bindings together with their rows stays (KEEP). Its real job is that
     nothing points at a deleted row.
3. **Three suspected defects** (findings (a)–(c)) are recorded in the next section.
4. **Reader disagreements (23 rules).** Readers proposed different buckets for 23 rules. The rules applied:
   - a uid answers the question and the rule is still needed without one → DEMOTE (or DEMOTE+MERGE when a copy
     exists);
   - a record format, a byte split, an ownership range or the uid route → KEEP;
   - a reader of state a deleted writer persisted → KEEP (legacy reader).

   The per-rule results are in the inventory table.

## Findings to verify before the merges that depend on them

Readers flagged three possible defects in existing rules. Each is checked against the code before the merge that
depends on it; a real defect gets its own issue and a red-first fix.

- **(a) Fork-child scope in the anchor chain (R193 vs R326).** The host-uid lineage walk excludes fork children, but
  `_identity_anchor_chain` has no such check. A fork child of a session that ended by compression may count as a
  verified ancestor in #436 candidate scope.
- **(b) Session-end prefix tie-breaks (R288).** The same-id and off-current tie-breaks compare in opposite
  directions (`>=` and `>`). The copies must agree before they merge.
- **(c) Lineage coverage (R438 vs R448).** The lineage-claimed check covers only the same session; the
  node-covered check in `store_complete` accepts a node from any session.

Status: being verified. The two merge clusters that depend on (a) and (b) wait for the result.

## Execution order

1. **This ledger** (a v0.26.0 GA criterion).
2. **The DELETE batch**, one PR after v0.26.0 GA.
   - It changes no behaviour, except R133's diagnostic label.
   - Proof: the full suite and the rollback-reader tests.
3. **The MERGE clusters**, one PR each, with parity tests. The two clusters marked above wait on their findings.
4. **The planner with `on` mode (slice C)** in its own minor, once the entry conditions hold.
5. **The legacy readers** are deleted once the sunset count is zero, with a rollback-reader test.
