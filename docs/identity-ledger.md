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
| **KEEP** | Stays, and keeps deciding. Either it is not a question a uid answers (lifecycle, publication, store hygiene, record formats, byte splits, ownership ranges), or it is the uid route itself. | 189 |
| **KEEP (legacy reader)** | Reads state that a deleted writer persisted: native recovery, removed in v0.25.0, and proofs written before v4 descriptors. Frozen. Deleted only when a read-only count finds none of that state on managed stores. | 7 |
| **SPLIT** | Does two jobs. Its non-identity job (a persisted record it writes or validates, an ownership or carry update, a scope guard, a byte split) stays and keeps running. Only its matching part demotes. | 80 |
| **DEMOTE-UNDER-ON** | Becomes the fallback. In `on` mode it runs only for rows the uid route cannot decide. On hosts that send no uid it keeps deciding exactly as today. Frozen: bug fixes only. | 111 |
| **DEMOTE+MERGE** | Demoted, and first folded into one sibling implementation. | 21 |
| **MERGE** | Copies of one predicate become one function. The refactor is behaviour-neutral. A parity test runs every copy's existing test inputs through the merged function. | 33 |
| **DELETE** | Changes no identity decision. Each row names its kind: **dead** (no caller), **unreachable** (no input reaches it), **ineffective** (it runs, but its result cannot change a decision), **redundant** (it repeats a computation already made), or **retired write** (it writes a value no reader needs: nothing reads it, or every reader, the previous GA's included, falls back to the same default when it is absent). Removed with proof: a grep of code, tests and docs, the git history, and the full suite. | 19 |

No rule is deleted just because uids exist. LCM keeps supporting hosts that send no uid, so every demoted rule stays
for as long as that support does.

### Counts per system

| system | KEEP | KEEP (legacy reader) | SPLIT | DEMOTE-UNDER-ON | DEMOTE+MERGE | MERGE | DELETE | total |
|---|---|---|---|---|---|---|---|---|
| SYS-1 Cursor/replay matcher | 16 | 2 | 14 | 38 | 18 | 10 | 5 | 103 |
| SYS-2 Commit proof, emission descriptors & occurrence projection | 18 | 5 | 21 | 27 | 1 | 3 | 4 | 79 |
| SYS-3 #436 identity anchor | 20 | 0 | 19 | 26 | 0 | 4 | 1 | 70 |
| SYS-4 Externalized payload & prefix matching | 18 | 0 | 16 | 15 | 1 | 2 | 7 | 59 |
| SYS-5 Host uid (shadow/on, engine uids) | 57 | 0 | 0 | 0 | 0 | 5 | 0 | 62 |
| SYS-6 Placeholder ledger & ignore policy | 5 | 0 | 3 | 4 | 1 | 4 | 1 | 18 |
| SYS-7 Generated-row recognition (scaffold, carrier, survival fit, retained anchor) | 8 | 0 | 7 | 1 | 0 | 4 | 0 | 20 |
| SYS-8 Lifecycle, rotation, publication & DAG lineage | 41 | 0 | 0 | 0 | 0 | 1 | 0 | 42 |
| SYS-9 Store columns & metadata hygiene | 6 | 0 | 0 | 0 | 0 | 0 | 1 | 7 |
| **all** | **189** | **7** | **80** | **111** | **21** | **33** | **19** | **460** |

## Principles

1. **A uid decides only what it can prove.**
   - A bound uid identifies the stored row.
   - An engine uid identifies a row LCM generated.
   - `_absorbed_message_uids` names the constituents of a host merge.

   A uid does not split bytes (composite remainders), own ranges (carry ranges) or validate a persisted record.
   The rules that do those things stay.
2. **Only the match demotes.** A DEMOTE decision covers a rule's identity matching and nothing else. Anything the
   rule also does keeps running in `on` mode:
   - a persisted record it writes or validates, which this version or the previous GA reads;
   - the identity bytes that a persisted proof or digest is computed from: a rollback reader recomputes the
     digest, so that form stays byte-for-byte;
   - an ownership, carry or frontier update, or the compaction boundary it sets;
   - a session or privacy guard on payload content;
   - routing between normal ingest and the bypass or host-fallback path;
   - chronology columns;
   - a byte split.

   Such a rule is SPLIT in the inventory. Every DEMOTE row was read against the code by an independent reader, and
   72 of them turned out to be SPLIT. The slice C PR lists, for each rule it stops consulting, the jobs that keep
   running, and its review checks that list.
3. **The content rules decide what the uid cannot.**
   - **First sight.** An unbound uid can belong to a row stored before the host stamped uids. The content rules
     decide that row, and the result is bound (binding kind `canonical`).
   - **Changed bytes.** One uid can hold several stored versions. A row whose bytes match one bound version
     replays that version. A row whose bytes match none of them goes to the content rules, and a new version is
     bound (kind `version`).

   Otherwise the uid decides, so the content matchers run only on first sight and on changed bytes, not on every
   replay.
4. **Readers of persisted state outlive their writers.** This is the rule v0.25.0 applied when it removed native
   recovery. A store written by an older version can still hold that state, and a plugin-only rollback to the
   previous GA must read what this version writes.
5. **Fail open to today's decision.** Any exception on the uid route returns the content decision and is counted,
   as the #436 anchor's fail-open does today.
6. **No schema version change.** Bindings live in the droppable `host_uid_bindings` side table. A new column comes
   only through `add_column_if_missing`, with a rollback-reader test.
7. **One step per PR.**
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
  for each host row, in order (the first matching branch decides):
    1. composites first
       carries _absorbed_message_uids          -> COMPOSITE: each constituent by its own binding; a constituent with
                                                   no binding goes to the content rules; never skipped as a whole
       engine uid of this lineage, with bytes   -> CARRIER: the generated part is GENERATED; the remainder is split by
       beyond the generated row                    the byte rules (KEEP) and decided as its own row
    2. engine uid of this lineage, bytes equal to the generated row
                                              -> GENERATED (never stored)
    3. uid bound in this lineage (canonical or version)
       bytes match one bound version            -> REPLAY(that version's store_id)
       bytes match none                         -> content rules decide; a stored row is bound as a new version
    4. otherwise (unbound uid, or no uid)       -> content rules decide; the uid is bound canonical, except when the
                                                   target row already holds another uid (a host that re-minted uids,
                                                   e.g. after a downgrade): no canonical binding, as shadow does today
  alias candidates stay observational: they never decide
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

**What `on` mode saves, and what it does not.**
- **What stops.** The 132 matching rules (DEMOTE and DEMOTE+MERGE) are no longer consulted on a replay that the
  uid decides.
- **What keeps running.** The SPLIT rules keep doing their other jobs: they write the proof and digest records,
  keep the identity forms those records hash, update carry ranges and the frontier, and guard payload scope. This
  is the price of a plugin-only rollback.
- **When those writers can go.** Only in a release whose rollback target no longer reads those records.

**Entry conditions for `on` mode:**
- **Shadow gate (v0.26.0).** At least 99% agreement over at least 200 counted bindings (`stored_new`,
  `version_new`) across at least 24 h. Every disagreement is explained with row evidence. Durable shadow errors are 0.
- **The planner decides in qualification first.** Before `on` ships, a build where the planner decides runs the
  existing version, carrier, composite and rollback cases, and the shadow window reports the generated, carrier and
  composite outcomes next to the counted bindings. Any disagreement that would lose, duplicate or reorder a row is
  fixed, or that case stays on the content rules. #891 (fork-child anchor scope) is fixed.
- **A deployed host carries uids.** Today no managed runtime sends them.
- **Persistence-only probe.** The host declares `message_uid` persistence-only. The probe runs once per process
  and fails closed.
- **Default stays `shadow`.** `on` ships as an opt-in value, and the default stays `shadow` for at least one minor.

## What each system keeps

- **SYS-1 Cursor/replay matcher (103).**
  - **KEEP:** the replay identity tuple. It is the only identity a row without a uid has.
  - **SPLIT:** the parts of that identity which set the bytes persisted proofs and digests hash; the write-failure
    guard; and the digest writers.
  - **DEMOTE:** the OR-ed replay proofs (sanitised suffix, raw suffix, tool-pair end, stale window, durable proof
    cursor). 19 of them are the planner cluster.
  - **Legacy readers:** the native-recovery snapshot digests.
- **SYS-2 Commit proof, emission descriptors and occurrence projection (79).**
  - **KEEP:** the proof record format, its digest and version contract, and proof-source selection. A rollback
    reader reads them.
  - **SPLIT:** descriptor binding, which picks the occurrence the persisted proof records; the proof writer and its
    carry-range rewrite. Their matching use demotes.
  - **Legacy readers:** the durable native proof walk, the native digest adopted by an empty rollover session, and
    pre-v4 occurrence identities.
- **SYS-3 #436 identity anchor (70).**
  - **DEMOTE:** candidate matching, prematch and rewind.
  - **SPLIT:** the composite and `alt_stamp` relation writes, the host-rewrite override records, the
    `observed_at` backfills, the session and conversation scope of candidates, and the ownership and frontier
    checks on summary claims.
  - **KEEP:** the byte split of a composite remainder.
- **SYS-4 Externalised payload and prefix matching (59).**
  - **KEEP and SPLIT:** the payload session-ownership guards and the persisted marker metadata.
  - **DEMOTE:** the prefix and occurrence matches.
- **SYS-5 Host uid (62).** **KEEP.** This becomes the deciding route; five rules merge into their siblings.
- **SYS-6 Placeholder ledger and ignore policy (18).**
  - **KEEP and SPLIT:** the persisted count, ordinal and dependent-reply records, which the previous GA reads.
  - **DEMOTE:** the matchers.
- **SYS-7 Generated-row recognition (20).**
  - **SPLIT:** carrier recognition (its byte split stays), and the survival-fit layout and compaction-boundary
    decisions.
  - **MERGE:** the scaffold classifier copies become one.
- **SYS-8 Lifecycle, rotation, publication and DAG lineage (42).** **KEEP.** These rules decide when to compact and
  publish, not which row is which.
- **SYS-9 Store columns and metadata hygiene (7).** **KEEP,** except one dead helper.

## DELETE list (19)

Each row was checked against the code at `9afb80d0`, and an independent review confirmed each kind. The deleting PR
repeats the grep at its own base. Rows not marked otherwise are **dead** or **unreachable**.

| id | where | why it is dead |
|---|---|---|
| R017 | `reconcile._session_end_replay_snapshot_metadata_key` | no caller |
| R040 | `reconcile._strip_inline_persisted_output_generation_identity` and its two OR terms | **ineffective**: the helper runs, but its result feeds only two OR terms that need an unrecoverable marker, and such a candidate always hits the later `continue` veto first (since the #313 era) |
| R057 | parameter `session_count` of `_find_reconciled_cursor_for_store_tail` | never read |
| R086 | `engine._is_active_context_droppable_identity` | no caller; `reconcile` has its own inline copy |
| R087 | `engine._ignored_message_is_quarantinable_assistant` | no caller |
| R133 | second `_cursor_from_durable_commit_proof` call in `_reconcile_ingest_cursor_from_store` | **redundant**: same input as the first call, and nothing between them changes what the proof reads; the only difference is a diagnostic label on a zero cursor |
| R164 | native-lossy veto in `_ingest_messages` | the in-memory commit proof is always written with `native = False` (compaction is its only writer since v0.25.0) |
| R168 | `engine._remap_cursor_through_native_host_repair` | reachable only through a native in-memory proof (see R164) |
| R181 | legacy native fields written into the commit proof | **retired write**: always written as false or empty; every reader, the previous GA's included, uses `.get` with the same default |
| R232 | `identity_anchor._identity_anchor_record_versions` (the `supersedes` relation) | **retired write**: written on every rewritten-row store, never read by this version or the previous GA; its two test assertions go with it; existing rows are inert and are purged with their messages |
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
| Session-end prefix arbitration | R274 R287–R291 R309 R310 | 8 | keeps the tie-break asymmetry as an explicit input (finding (b) below) |
| Placeholder ordinal and budget matchers | R062 R375 R378 R386 R388 R389 | 6 | |
| State-store lineage walkers | R193 R195 R325 R326 R459 | 5 | after #891 (finding (a) below) |
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
   - Every store LCM creates has `messages.store_id INTEGER PRIMARY KEY AUTOINCREMENT` (since the first commit), so
     ids are not reused there. A `messages` table that something else created with a plain primary key keeps that
     shape (`CREATE TABLE IF NOT EXISTS`), and its ids can be reused; the hygiene tests open such a table.
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

## Findings checked before the merges that depend on them

Readers flagged three possible defects in existing rules. Each was checked against the code. A real defect gets its own
issue and a red-first fix.

- **(a) Fork-child scope in the anchor chain (R193 vs R326).** The host-uid lineage walk excludes fork children, but
  `_identity_anchor_chain` has no such check. A fork child of a session that ended by compression may count as a
  verified ancestor in #436 candidate scope.
- **(b) Session-end prefix tie-breaks (R288).** The same-id and off-current tie-breaks compare in opposite
  directions (`>=` and `>`). The copies must agree before they merge.
- **(c) Lineage coverage (R438 vs R448).** The lineage-claimed check covers only the same session; the
  node-covered check in `store_complete` accepts a node from any session.

Results (checked against the code, with synthetic repros):

- **(a) is a real defect: #891.**
  - A branch child of a compression-ended parent stores 2 of its 10 rows and carries the parent's rows instead.
    With host uids, its lineage binds to the parent's store ids (proof kind `anchor_replay`, which the shadow gate
    reports apart).
  - No row was lost or duplicated.
  - Trigger: a host that rotates sessions, a gateway chat (the parent and the branch share one conversation id),
    and a branch that ingests the copied rows after the parent rotated.
  - Fix: the anchor chain stops at a fork child, with the same predicate as the host-uid walk. The lineage-walker
    merge then makes the two walks one resolver.
- **(b) is deliberate.**
  - While the session is still bound, a tie between the normal and bypass prefixes stores the final row (`>=`),
    unless the bypass prefix was truncated and the end view is longer than it. That ambiguity vetoes the normal
    path even on a tie.
  - Once the engine has moved on, a tie does not store it (`>`): it fails closed. The host keeps the reply, and a
    later ingest of that session stores it.
  - Tests pin both directions and the veto. The merged arbitration keeps the veto, and keeps the asymmetry behind an
    explicit "the session still has a live normal binding" input; neither operator can replace both.
- **(c) is a real difference, but not a defect on its own.**
  - A session can summarise only rows it owns, and compression moves the parent's summaries to the child. So two
    live sessions own the same rows only through (a).
  - The any-session coverage check stays as it is.

## Execution order

1. **This ledger** (a v0.26.0 GA criterion).
2. **The DELETE batch**, one PR after v0.26.0 GA.
   - It changes no identity decision. Observable differences: R133's diagnostic label on a zero cursor, and R181
     stops writing fields whose readers fall back to the same default when they are absent, and R232 stops writing a
     relation that nothing reads.
   - Proof: the full suite and the rollback-reader tests.
3. **The MERGE clusters**, one PR each, with parity tests. The lineage-walker merge follows #891.
4. **The planner with `on` mode (slice C)** in its own minor, once the entry conditions hold.
5. **The legacy readers** are deleted once the sunset count is zero, with a rollback-reader test.
