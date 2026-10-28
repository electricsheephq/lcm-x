# Exactly-once ingest: one model for retries, replays and restarts (design r3.4)

Tracker: #1993 (v0.30.0, "nothing lost"). Status: reviewed; ready for phase P1 (2026-10-28). Phase P1 is tracked in its own issue, linked from #1993.

**Review history.** Four rounds of independent review shaped this design: a Codex gpt-6-astra reviewer and Claude review workflows (several lenses, skeptics and a critic). Each round's findings were folded before the next. The last two rounds checked only the P1 parts, and both reviewers returned READY FOR P1 on r3.4.
- Finding ids from those reviews (A1-A12, L1-1 to L5-10, "Astra n") stay in parentheses for traceability. The review record itself is not published. In §5a, A and C mark the Codex and Claude reviewers' findings.
- In §1 and §5, C1-C4 name the three commit points and the injected-crash cells built on them.
- E1-E7 are this design's rules (§2) and M1-M16 its decisions (§6). I-numbers (for example I1 and I5) are the invariants in [docs/invariants.md](invariants.md).
- File and line references are at commit 0eb6e8736ac0361341b3f29ebfb8e30e5b4dabc5 (v0.29.1-rc2), the tree this design reads.

## 1. Problem
- `messages` has no identity key (store.py:429-447). Exactly-once is an in-memory decision made before the INSERT.
- The cursor and the in-process proof live in memory only (frontier_state.py:7-16). A cold start rebuilds them by tail matching, and when that fails it re-stores whole batches ("persisted ambiguous delta", reconcile.py:3029-3038).
- Decision evidence commits apart from the rows: before the INSERT (C1), after it (C2), and with the cursor advance (C3).
- Unkeyed rows are decided by position. A replace or a merge counts as not-new, so the change is lost (#7).
- The host uid is shadow only (#643).
- These cause the open children. Each gets a point fix, and new shapes keep arriving.

## 2. Invariants
- **E1 No skip without proof.** A host row is skipped only when a proof binds it to one specific stored occurrence or LCM emission. Ranks 1-4 and the anchored run are strong proofs. Rank 5 is weak and is accepted only where §3.4.5's counts make the match certain. With no proof, store.
- **E2 One binding per occurrence.** An occurrence has an immutable id and a sortable ordinal in its lineage (§3.1 item 1). UNIQUE (lineage, ordinal) over stored, migrated and ambiguous bindings enforces E2 and is the M5 backstop.
- **E2b Consume once, and one key order.**
  - Within one incoming view, each stored occurrence is bound at most once across all ranks.
    - A tool segment debits its assistant row and all its results together, or nothing.
    - A surplus copy falls to the next rank, then to NEW.
    - One stored row is bound twice only through a composite_part or split_part witness.
  - **Key order:** uid > stamp > tool_call_id > position. A disagreement on a stronger key vetoes every weaker rank before it runs: the candidate is not bound, and the row is stored with `possible_replay_of`.
  - A disagreement needs both values known; missing never disagrees.
  - **Stamp disagreement:** the incoming row has a stamp, the candidate has recorded stamps (its own plus its usable observations, §3.1 item 4), and the incoming stamp is none of them. It vetoes rank 3 (#1967), rank 4 once the emission's observed stamp is recorded (#1003), and rank 5.
  - **Unkeyed** means no eligible uid and no stamp. On a lineage marked stamp_unreliable (M14), stamps count as missing.
- **E3 Atomic ingest.** Rows, bindings, vectors, relations and evidence commit in ONE `BEGIN IMMEDIATE` transaction on the store connection.
  - The classification may run first on a ledger snapshot outside the lock. Inside the lock the transaction re-reads the lineage's view_seq and marks, and classifies again under the same rules if either changed (§3.2).
  - The one exception is pending session-end tails (§3.5).
- **E4 Visible doubt.** Nothing is deleted.
  - A **plausible candidate** for a row is an occurrence in the same lineage that a rank proposed for it and that a rule then rejected: the window, the order, a veto, a tie, the cost fallback, or a refused emission.
  - Every row stored while it had a plausible candidate carries `possible_replay_of` to that candidate.
  - Every changed-slot NEW carries `supersedes`; its target may be a generated entry.
  - A NEW row with an LCM generated marker and no descriptor carries `generated_unproven` (§3.3).
  - The doctor reports all three (§3.9).
  - A row with no candidate is plain NEW, so repeated ordinary content ("ok", "continue") gets no relation.
- **E5 Ranked proof.**
  1. **Host uid, eligible.** The binding is canonical or a recorded revision, `disagree_seen` = 0, and a pre-ledger binding has `proof_kind` = stored_new. Rank 1 decides only in uid mode `on` (M16).
     - An ineligible binding authorizes no skip. Its uid still vetoes weaker ranks when it disagrees.
     - Rank 1 is subject to the Window and to consume-once, so a reused uid that names an occurrence the view has moved past is stored visibly (#1449). It is exempt from the Order condition: keyed reorders stand, and are logged.
  2. **The anchor:** role, host stamp and identity digest (§3.1).
  3. **tool_call_id plus identity,** scoped to the user anchor before it. A stamp disagreement vetoes it. When the anchor is NEW, its tool segment is NEW unless each row has its own eligible uid (the #1869/#1838 retry loss).
  4. **The emission descriptor,** for generated rows. It binds at its output index within its emission vector, at most once per view. Once the emission's observed stamp is recorded, a disagreeing stamp vetoes it (#1003).
  5. **Ordered alignment** (weak, §3.4.5).
- **E5a A rank-1 hit only resolves the uid to its revisions (A1, #1428; Astra 5).**
  - Incoming content equal to the latest revision (identity digest or alias), with the absorbed-uid set a subset of that revision's constituents: REPLAY.
  - Equal to an earlier revision: a revert. It is stored NEW, with `supersedes` to the latest revision and `possible_replay_of` to the matched one.
  - An extension of the latest revision: PARTIAL (§3.4.6), with that revision as base. The uid witnesses the extension even when the stamp changed.
  - Any other content: NEW plus `supersedes` to the latest revision, recorded in the same transaction as the uid's next revision.
  - The canonical binding never moves. UNIQUE (lineage, uid) holds on it.
  - An unknown absorbed uid classifies its own constituent. A merged scaffold's outer uid is engine-minted, so its match may come from rank 2.
- **E5b Same-stamp rules.**
  - Same stamp, and the identity is in the stored occurrence's alias set: REPLAY, subject to E5c and E2b.
    - The alias set covers the pruned-tool alias (#773), host-rewrite overrides, carrier strip (the reconcile.py:924 aliases), and an externalized stub equal to its raw output (reconcile.py:993-1051, #588).
    - It applies at rank 2, and at rank 3 when the incoming row has no stamp (scoped to the user anchor).
  - Same stamp, and the identity extends the stored one: PARTIAL (the §3.4.6 guards).
  - Same stamp, any other identity: NEW plus `possible_replay_of`.
  - A row that goes from no uid to a known uid gets a uid binding for its existing occurrence (#1836 item 4).
- **E5c Position agreement.** A rank-2 or rank-3 match, or an alias match, is accepted only when it is position-agreeing:
  - **Window:** the occurrence's ordinal is at or after the view's head (§3.4.3).
  - **Order:** inside its gap, the match is in EVERY largest order-consistent set of matches (§3.4.4).
  - A match that fails either is NEW plus `possible_replay_of`: at worst a visible duplicate, never a loss.
  - Rank 1 needs the Window only.
- **E6 Generated rows** bind by emission descriptor only. Descriptors are durable before the host sees the list (§3.3). If the emission write failed, a returned generated row is stored with `generated_unproven`: visible, never silent.
- **E7 Frontier, diagnostic only (A5).**
  - The frontier gate stays today's exact contiguous-coverage and duplicate-source checks (lifecycle_state.py:1149-1168).
  - E7 itself (the frontier is at most the highest store_id bound in the adopted generation) is checked and logged at each adoption (§3.3). A lagging frontier loses nothing (engine.py:10602-10616 already reports a stale advance).

## 3. The model
### 3.1 The ledger: one write rule and its tables
**M11, the write rule.** The ledger records only facts the view vectors cannot reproduce:
- a stored row;
- a uid or a uid revision;
- an LCM emission;
- a relation between stored rows;
- a new stamp observation for a bound occurrence.

A replay with an unchanged stamp writes nothing except the view vector.

1. **Occurrence bindings:** `occurrence_bindings(occ_id, lineage, ordinal, store_id, identity_digest, stamp, disposition, key_rank, view_seq)`.
   - `occ_id` is immutable and never reused. Every other table, every vector and every parent link refers to an occurrence by occ_id. Only this table holds the ordinal, so re-spacing ordinals rewrites one column (Astra 9).
   - The disposition is one of: stored, migrated, ambiguous, composite_part, split_part. The tag `migration_residual` marks a row that migration stored (§3.6).
   - composite_part and split_part rows are relations: they bind existing stored rows as parts of another occurrence.
   - UNIQUE (lineage, ordinal) WHERE the disposition is stored, migrated or ambiguous (E2).
   - **Ordinals** are sparse 64-bit sortable labels.
     - An appended row takes the lineage's last ordinal plus 2^20.
     - A row placed inside a gap takes a midpoint between the two ADJACENT EXISTING ordinals of the lineage, of any disposition (L3-1).
     - Rows in one gap are assigned in view order.
     - A NEW row is placed after every existing ordinal in its gap that this view did not place, because new events follow dropped ones. A supersedes row is the exception: it takes the midpoint right after the row it replaces.
     - Generated entries have no ordinal (L3-7).
   - **Re-spacing:** when a gap has no integer midpoint, the ingest re-spaces the smallest window of neighbouring ordinals that leaves every gap in it at least 2^10 apart. It runs in the same transaction, and is logged and counted.
   - Indexes: (lineage, ordinal), (lineage, stamp, identity_digest), and (lineage, identity_digest). Ranks 1-3 find candidates lineage-wide through them, from any epoch; E5c then decides.
   - Kept forever.
2. **Uid bindings (A7)** reuse the canonical `host_uid_bindings` shape (store.py:1239) and are written inside the ingest transaction.
   - The canonical binding carries UNIQUE (lineage, uid). This is the only place uniqueness on a uid is enforced; view entries carry no uid.
   - Revision rows (lineage, uid, revision, occ_id) record the content revisions (E5a), using host_uid.py's existing version kind (host_uid.py:299-310).
   - The absorbed-uid constituent bindings (A1) live here too.
   - `disagree_seen` and `proof_kind` decide eligibility (E5 rank 1).
   - The uid modes are defined in M16.
3. **Emissions:** `emissions(lineage, view_seq, output_index, descriptor_digest, kind, source_store_ids, state, next_occ_id, observed_stamp NULL, recovered_store_id NULL)`.
   - Kind is one of: summary, carrier, scaffold, recovery, projection. These are the rank-4 descriptors.
   - The state is proposed, adopted or refused (§3.3).
   - `next_occ_id` is the first occurrence that followed the generated rows in the returned list. It gives a head when the emission's vector has been evicted (§3.4.3).
   - A recovery emission records (emission, observed stamp, recovered store_id) on the first ingest that observes it (A11, #1523).
4. **Stamp observations:** `stamp_observations(lineage, occ_id, stamp, kind, view_seq, source)`.
   - **alt_stamp:** the occurrence arrived bound strongly (rank 1 or the anchored run) with a stamp not yet recorded for it.
   - **carried_stamp:** an LCM carrier row arrived carrying its source occurrence's stamp.
   - **source** is `ledger`, or `legacy` when it is read from a legacy relation (§3.6).
   - An observation is usable from the next view on. A row never certifies itself with the stamp it brought (Astra 7).
   - At most 4 per occurrence and kind (L3-8, L5-9). Reaching the cap on 3 occurrences of a lineage marks the lineage stamp_unreliable (M14).
   - Readers that need host time for a NULL `observed_at` take the binding's stamp, or else the earliest observation (#1978, L5-10).
5. **View vectors (M10):** `ingest_views(lineage, view_seq, kind, state, entries BLOB, prefix_chain BLOB)`.
   - Kind is ingest or emission.
   - Each entry is (identity_digest, stamp_digest or NULL, uidset_digest or NULL, occ_id or descriptor id). uidset_digest covers the uid and the absorbed-uid set, so a view whose only change is a uid or an absorbed uid does not match its previous entry (#1428).
   - The prefix chain is a running digest over the three digests of every entry.
   - The last 2 of each kind are kept per lineage.
   - Only ingests write vectors. A tail retry (§3.5) and an S3 end (§3.7) write bindings and no vector (L3-5).
6. **Lineage marks (§3.6):** `lineage_marks(lineage, max_store_id, legacy_proof_digest, max_leaf_id, stamp_policy)`, one row per lineage.
   - The ingest transaction writes the store mark, only when max_store_id changes (L4-10).
   - `dag.add_node` (dag.py:267), the single DAG insert point, writes the leaf mark in the same transaction as the node. Condensations and conversation-less leaves therefore update it too (L4-7).
   - Every writer of the legacy commit proof updates `legacy_proof_digest` in the same transaction: the compress() exit write, the rotation re-key and the reset clear (L4-2, L5-4). The digest covers the per-session proof keys and excludes created_at.
7. **Meta:** the per-store digest salt lives in a ledger meta row in the same file, so backups and copies keep it (M4). A ledger whose salt is missing is unreadable. The store is then treated as pre-ledger and re-migrated with a new salt (§3.6), which is a re-migration, never a mass re-store.

Two definitions:
- **The durable cursor** is the newest ingest vector. A cold start finds the anchored run against it (§3.4.2). At P1 that run is one more candidate in today's cursor choice, never a replacement for tail matching (§5); at P2 it is the classifier's base.
- **identity_digest** is a per-store salted digest of the PRE-redaction identity tuple (M4). A redaction-policy change therefore cannot change a ledger-era identity (#758), The salt stops precomputed lookups across stores. A reader of the same file holds the salt and can still test a guessed text, so the digests are not a confidentiality control. Pre-ledger rows keep today's matching until they are migrated (§3.6).

### 3.2 One transaction per ingest
In order:
1. **Outside the lock:** compute every row's digests and the prefix chain, take a ledger snapshot (the lineage's vectors, marks and view_seq), and classify against it (§3.4). The view factories must be pure and cheap, because a retry re-runs them.
2. **`BEGIN IMMEDIATE`,** opened explicitly before the first read of this phase. Today it starts at the first INSERT, so the reads run outside the lock (L5-7). Then re-read view_seq, the marks and the store's write generation (`PRAGMA data_version`, store.py:1196-1200). The write generation also moves when an older reader on another connection commits an `alt_stamp` relation or an observed_at backfill. If any of them changed, classify again inside the lock under the same rules and budget.
3. **Insert the rows.**
   - `messages.observed_at` is written ONCE at INSERT from the row's own stamp, and never updated.
   - At P2 the observed_at backfill UPDATEs are retired, so an old row never takes a new stamp (#1512). At P1 today's NULL-only backfill stays, inside the ingest transaction, because today's anchor finds its candidates through observed_at (identity_anchor.py:291).
4. **Write the complete list (L5-6):**
   - occurrence bindings, including any re-spacing;
   - the ingest view vector;
   - the store mark, when it changed;
   - the relations: composite (forward composite_part is today's `composite` relation), split, supersedes, possible_replay_of, generated_unproven and adopted_from;
   - uid canonical, revision and constituent bindings;
   - stamp observations;
   - the anchor carry metadata;
   - a recovery-emission binding on its first observation (A11);
   - emission state changes (adoption);
   - migration bindings (§3.6);
   - the stamp_policy mark.
5. **COMMIT.**

Rules:
- A rank-1 disagreement blocks any rebind.
- Concurrent writers serialize on the lock. The UNIQUE indexes are a backstop that sends the loser back to classify again (M5).
- Today's decision evidence (the C1 data) is staged in memory and read back by the classifier; it gets no commit of its own.
- Faulted ingests close here (#1815: three shapes that duplicate or skip the retry). A failed transaction leaves no partial evidence, and the retry classifies again.
- Each ingest counts its lock-hold time, rows classified, rank-5 DP cells, bytes written, cost fallbacks and re-spacings. These are the §5 instruments.

### 3.3 Emissions: proposed, adopted or refused (A4)
**M10, the generation model.**
- A generation is an emission epoch: it changes only when `compress()` returns a list.
- Every ingest writes an ingest view vector inside its own transaction.

The pieces:
- **Leaf mark.** It is written by `dag.add_node` (§3.1 item 6). A failed exit write after a leaf publish therefore cannot trip rollback detection.
- **Final-output write.** `compress()` gets ONE exit function, and every return path calls it (L5-8): stub-first, recovery, fit-only and exception.
  - The function writes the emission view vector (state proposed) and a descriptor, with `next_occ_id`, for every generated row of the list actually returned.
  - This is the write a fit-only rotation was missing (#2029, #606).
  - A failure of this write is handled inside the exit function, not by the generic exception handler.
- **If that write fails:**
  - Set a process-wide unproven mark. Each entry holds the store, the lineage, the emission descriptor, the output index and the row's digest. It lives at module level, so every engine instance and clone in the process sees it.
  - Return the unrewritten list where it fits. If the list is over-window (#582), return the fit.
  - Every later ingest in the process first retries the emission write inside a SAVEPOINT, then classifies. The mark clears on success.
  - While the mark holds, a generated row that comes back to the same store and lineage with a digest in the mark is stored NEW, with `generated_unproven` and `possible_replay_of` to its source rows.
  - After a restart the mark is gone. The row still carries an LCM generated marker and has no descriptor, so it is stored NEW with `generated_unproven`. It also gets `possible_replay_of` when its digest matches a rendering of one of the lineage's DAG nodes. The row is visible, never silent (Astra 4, L5-1; this replaces r3.1's "plain NEW").
  - P1 makes the marker universal: every generated kind carries one (the summary header, the carrier tag, the scaffold prefix). A user row that contains the marker text is flagged visibly, never lost.
- **Adoption.** The emission becomes adopted on the host's boundary signal (engine.py:4444) or when an incoming view anchors on it.
  - A host refusal or rollback (#1116, `record_rejected_compaction`, engine.py:2686) sets it to refused. A refused emission is never a base and never adopted (L5-2), and the previous epoch stays current.
  - E7 is checked and logged at adoption.
- **Crash between the write and the host's ingest:** safe. The host never held an output that was not durable, so a summary or scaffold that comes back always matches its descriptor (E6, the #534 carrier shape).
- No ATTACH and no separate marker file.

### 3.4 Classification of an incoming view
#### 3.4.1 One evaluation order
No step reopens an earlier one.
1. Compute the digests and keys (pure, outside the lock).
2. Find the base and the anchored run (§3.4.2), then the head (§3.4.3).
3. Fix the strong placements:
   - the anchored run;
   - accepted rank-1 matches (eligible, inside the Window, consume-once);
   - accepted rank-4 matches.
4. Find candidates at ranks 2-3 and aliases, lineage-wide, applying the key-order vetoes (E2b).
5. Apply the Window and Order per gap (§3.4.4). Accepted candidates become placements.
6. Run PARTIAL for the rows still unplaced (§3.4.6).
7. Run rank 5 per gap (§3.4.5).
8. Everything left is NEW, with `possible_replay_of` when a candidate was rejected (E4).

#### 3.4.2 Base and the anchored run
- **Eligible records:** this lineage's retained vectors (2 ingest, 2 emission), excluding refused emissions.
- **Entry match:** two entries match when their identity digests are equal and no known key disagrees. Uid sets that are both known must be equal. Stamps that are both known must be equal, except on a stamp_unreliable lineage.
  - A generated entry whose observed stamp is not yet recorded has a missing stamp, so the host stamping it does not break the run.
- **The anchored run** is the longest run V[0..k-1] that matches R[j..j+k-1], over eligible records R and offsets j, with k ≥ 1.
  - It starts at V's first row, skipping rows of a kind LCM-X never stores, such as system rows. Generated rows take part like any other row: their entries compare by identity digest.
  - Ties go to the newest record, then the smallest j.
  - Its rows are REPLAY: a strong proof, because the run is exact, in order and contiguous from the view's start. They bind to R's entries, consume-once.
  - The fast path (A12b) is the case j = 0 against the durable cursor. When the prefix chains are equal it is decided with O(1) ledger I/O, otherwise entry by entry.
  - An offset j > 0 is a sliding-window host (#1566).
  - An anchored run on a proposed emission adopts it.
- **The base** is that record R. With no anchored run, the base is the history: the lineage's kept bindings in ordinal order.
- **All-restamp control (M14, #1511).** Suppose the search finds at least 3 consecutive entries that match on identity digest and uid set, where every one has known stamps that differ. Then the lineage is marked stamp_unreliable (logged; the doctor shows it), and the search runs again without stamps.

#### 3.4.3 The head: one definition (L1-2)
The head is the ordinal from which this view may bind occurrences. Take the first rule that applies:
- **(h1) Anchored run:** the ordinal of R's first ordinal-carrying entry at or after j. When R's own entries after j carry none (an evicted emission's tail), use the emission's `next_occ_id`.
- **(h2) Full-history replay (#1674):** the view's first row that is not generated, and not of a kind LCM-X never stores, has the identity digest of one of the lineage's first 8 kept occurrences. The head is the ordinal of the earliest such occurrence.
- **(h3) Keyed view:** the ordinal of the first eligible rank-1 match in VIEW order.
- **(h4) No head.** The view is anchorless. Ranks 2-3, aliases and rank 5 accept nothing, and every row with a candidate is stored NEW with `possible_replay_of` (Astra 1: without independent placement evidence, store visibly).

An alias, a stamp observation, or a rank-2, rank-3 or rank-5 match never sets the head.

Traces:
- **#1519:** X is stored at ordinal 10 with an alternate stamp, and the view has moved past X. The incoming X@alternate is either anchorless (h4) or lands after a head above 10. Either way it is stored visibly. The Window rejects it; it never sets its own head.
- **#1449:** a reused uid naming an occurrence before the head fails the Window, so it is stored with `possible_replay_of`.

#### 3.4.4 Window and Order, per gap
- **Gaps.** The placements cut the view into gaps.
  - A gap between placements p and q (in view order) has the ordinal interval (ord(p), ord(q)).
  - The leading gap's interval starts at the head, and the trailing gap's interval is open-ended.
  - A rank-4 placement has no ordinal, so the interval uses the nearest ordinal-carrying placement on each side.
- **Keyed reorder (Astra 9, L1-7).** If ord(p) > ord(q), both rank-1 placements stand, and the reorder is logged and counted. The gap's interval is empty, so nothing else is accepted in it.
- A candidate is **position-agreeing** when its ordinal lies inside its gap's interval and at or after the head.
- **Order.** Accepted candidates must strictly increase in view order. Compute the longest increasing subsequences of the position-agreeing candidates (O(n log n), forward and backward lengths). A candidate is accepted only if it lies in EVERY longest one (L1-3); the others are NEW with `possible_replay_of`. Accepted candidates become placements and split the gap further.
- **Units (L1-6).**
  - A tool segment is one unit, placed at its assistant row.
  - A split group is one unit, at its stored composite's ordinal.
  - A composite's parts must increase, and its remainder is placed right after its last part.

#### 3.4.5 Rank 5, per gap (weak)
- **Target.** The lineage's kept bindings (stored, migrated, ambiguous) inside the gap's ordinal interval, in ordinal order, minus the occurrences this view already bound.
  - Aligning against history instead of one vector keeps dropped rows reachable across evictions (Astra 6, L2-2, #1573, #1547). Example: [A,B] → [A] → [A,A] → restart → [A,B] binds B to the stored B.
  - A superseded occurrence in the target matches only as history. A slot that matches it is a revert: stored NEW, with `supersedes` to the current row of its chain and `possible_replay_of` to the matched row (L2-3).
- **Counts (Astra 10, L2-1).**
  - Definitions: n_d is the number of the gap's incoming slots with digest d. lo_d and hi_d are the fewest and the most of those slots matched over all maximum alignments (order-preserving, consume-once).
  - lo_d slots are REPLAY (weak, accepted).
  - n_d − hi_d slots are NEW.
  - hi_d − lo_d slots are AMBIGUOUS: stored with `possible_replay_of` at their view position. A tie never skips a row.
  - One canonical alignment, the leftmost maximum alignment, decides which slots get which label, and their ordinals. Its matched d-slots, taken in view order, are REPLAY up to lo_d and AMBIGUOUS after that. Its unmatched d-slots are AMBIGUOUS up to (hi_d − its matched count) and NEW after that.
  - Twins alone never create ambiguity. With base [x, y] and incoming [x, x, y], every maximum alignment matches exactly one x, so the leftmost x is REPLAY and the other is NEW.
  - A swap [p, q] → [q, p] gives lo = 0 and hi = 1 for both digests, so both rows are stored with `possible_replay_of`.
  - The number stored is never below the number of new rows (no loss), and never more than Σ(hi_d − lo_d) above it, all of them visible.
  - An identical run of unchanged length is matched in full by every maximum alignment. That is M2.
- **Supersedes pairing (L3-2).** The accepted matches (of any rank) cut the gap into sub-gaps.
  - When a sub-gap holds as many unmatched entries of the base VECTOR as NEW rows, they pair in order as changed slots: NEW plus `supersedes`.
  - Otherwise nothing pairs, and each unmatched entry is a plain drop.
  - Rows dropped in older views never pair.
- **Computation and bound (Astra 12, L2-4).**
  1. Trim the common prefix and suffix (O(n)).
  2. Run one LCS DP over the core, with forward and backward tables. For digests that are unique on both sides, these give each slot exactly: lo_d = 1 when the slot is essential, and hi_d = 1 when it is possible.
  3. A digest repeated on either side adds two lexicographic DPs: the most and the fewest d-matches at maximum length.
  - The budget is B = 4,000,000 DP cells per ingest, counted across all gaps and DPs.
  - **Over budget, the gap takes the visible fallback:** its slots with a candidate in the target are stored with `possible_replay_of`, and the rest are NEW. Each fallback is counted.

#### 3.4.6 PARTIAL, in both directions (A3, widened)
**Shapes:**
- An incoming composite whose parts are stored: bind the parts as `composite_part`, and store only the remainder.
- A stored composite whose parts arrive separately: bind them as `split_part`, with a composite relation (#1506).
- A Hermes merge-append (R, then R+N1, then R+N1+N2): store only the new tail, so the copies never grow quadratically.
- **An emission merged with a host row** (#1446, #1961, #534): a row that begins or ends with an emission's generated text, at that emission's output index, is split. The generated part binds to its descriptor as composite_part, and only the remainder is stored, in the row's own role. Summary text is never stored as user content.

**Guards.** Each shape applies only when all five guards hold; otherwise the row is NEW plus `possible_replay_of`. A same-stamp prefix part follows the same rule.
- (a) A donor part is witnessed by one of:
  - the composite's stamp;
  - a usable stamp observation of it;
  - an eligible rank-1 uid (this covers a same-uid extension whose stamp changed);
  - a **retained-prefix witness** (Astra 7, #545). The base vector holds the donor at this slot in the immediately previous view. The slot's left neighbour is placed, or the slot leads its gap at the head. The composite begins with the donor's exact bytes. This covers the unstamped merge-append: each step stores only the new tail.
  - Neither NULL equality nor the candidate's own stamp witnesses a part.
- (b) The parts are back-to-back rows in one view run, with nothing between them.
- (c) The decomposition is unique and ordered, and uses each stored row once.
- (d) No part is reserved as its own occurrence (B-ID-1).
- (e) No rank-1 or rank-2 key disagrees.

**M2 (decided, extended).** An unkeyed row re-sent identically in place, with the run length unchanged, cannot be told apart from a replay without host keys. It is REPLAY.
- The same holds for identical re-growth after a shrink (A6), #1566's all-identical sliding window, and an all-identical unkeyed overlap in migration or a full-history replay.
- It is a disclosed limit for hosts without uids or stamps, the ACP and editor paths. Class D closes it (host uid), not guessing: storing instead would duplicate on every plain replay.
- It is measured on the ACP-driven cells (§5).

### 3.5 Pending session-end tails (A9)
A tail exists because the store hit SQLite busy at session end. The ledger is in the same file, so it cannot take the tail at that moment either.
- A tail is queued for ANY lock failure at session end, including a locked reconcile read (#1805).
- Each queued tail records its frozen head: the occ_id of the last placed row of the view it followed (L5-5).
- Tails stay in memory. Each one is retried inside one ledger transaction, and writes bindings but no vector.
  - **P1 rule:** the tail's leading rows that match, in order and on all three digests, the kept bindings right after the frozen head are REPLAY. Every other row is stored, with `possible_replay_of` when a binding after the frozen head has its identity digest. This replaces the positional `after_store_id` check (engine.py:10674-10692, #1440). A tail frozen before any decision was made (a locked reconcile read, #1805) is retried as a cold-start ingest of its own session, so fix 2's candidates apply (§5). When that session is no longer the engine's current one, the retry takes today's S3 end route (prefix_matching.py:345-386), which has no fix-2 candidates. Either way, any row it stores whose identity digest matches a kept binding of the lineage carries `possible_replay_of`.
  - **P2:** the tail goes through §3.4, with the frozen head as its head and the history after it as its target.
- **M9 (re-ruled).** The loss window needs all three of:
  - busy at session end;
  - busy again at every later retry (the next start, the next end, a sibling bind, and the 5 s shutdown drain);
  - then process death or a busy exit.
- Recovery happens only through a host resume of that session; `/new` does not recover it.
- Accepted and disclosed, with no sidecar journal. Every busy-at-end event and every drain failure is logged and counted. Any drain failure during the parity audit fails the gate; it is never relabelled as deferred (§5).

### 3.6 Migration and rollback
- **When:** the first ledger ingest of a lineage that has pre-ledger rows, and again after a detected rollback.
- **Base (L4-1):** the lineage's pre-ledger rows across its session and its compression ancestors, plus the carry ranges, in store order.
- **P1, binding only.** Today's code classifies the view, and its decisions take effect.
  - Migration then binds each stored row it can tie to a byte-exact copy by today's matching, consume-once.
  - It binds an aligned row to its own byte-exact copy, never to a neighbour's copy set (#588), and records the incoming row's pre-redaction digest.
  - It leaves every other row unbound and inserts no rows.
  - An unbound pre-ledger row stays a lineage-wide rank-2 candidate (by its stamp and content). A NEW row that matches one carries `possible_replay_of`.
- **P2.** Migration runs the §3.4 classifier with this base. Ranks 1-4 are the anchors, including persisted legacy-proof emissions as rank-4 descriptors, and rank 5 runs between them.
  - Misfits and ties are stored with `possible_replay_of` and tagged `migration_residual`. The doctor counts them, and that count is the #758(a) metric (L4-6).
  - An all-identical unkeyed overlap is the M2 limit.
- **Rollback detection.** The lineage marks (§3.1 item 6) record its max store_id, the legacy proof digest, and its max DAG leaf id.
  - A mark that changed behind the ledger's back means an older reader wrote this lineage.
  - Re-migration then covers only the rows past the marks. Rows already bound are fixed anchors; the §3.1 insert rule places the rest; a new vector and new marks are written; epoch 0 is untouched (L4-3).
  - Writes to other lineages never trip it.
- **Legacy identity-only writes (Astra 8).** An older reader can add an alt_stamp relation or backfill observed_at without changing any mark. So on every ingest the classifier reads the legacy identity relations of its candidates (the alt_stamp relations and the reconcile.py:924 aliases) as observations with source `legacy`. No mark is needed for them.
- **Re-upgrade (#534, L4-9).** A summary that an older reader published is read as a rank-4 descriptor when that reader persisted its proof. A memory-only legacy proof leaves no descriptor, and the replayed summary is stored with `generated_unproven`. That is visible, and disclosed.
- **Disclosed residual (#758(a)):** a redaction-policy change before the first ledger ingest leaves a one-time visible duplicate.
- **Rollback is a supported path** (docs/process/schema-changes.md:14-18; invariant I5).

### 3.7 Other insert routes and state
- S3 is the session end for a non-current session (prefix_matching.py:156-192, 345-386). It goes through the same classifier and writes bindings but no vector.
- The process-local carrier seen/retired ledger, capped at 256 (identity_anchor.py:57), becomes `emissions` rows.

### 3.8 Lineage (A8)
- **Key.** A lineage's key is the persisted host_uid key `<home_tag>:<root>` (host_uid.py:77).
  - It is fixed per session at first resolution and never re-derived.
  - A partial chain is never cached. This fixes identity_anchor.py:99-110, where a missing child row caches the short chain.
- **Provisional lineage.** While the key is unresolved, writes go under a provisional session-scoped lineage P. When the key resolves to L, P merges into L inside one ingest transaction, in five steps (L4-4, Astra 11):
  1. Re-read L's key, marks and view_seq inside the lock.
  2. Classify P's rows against L's ledger by §3.4, with L's history as the base. A P row that replays an L occurrence keeps both stored rows, and P's copy gets `possible_replay_of` to L's.
  3. Give EVERY retained P occurrence, matched or not, a fresh ordinal after L's max, in P's order; occ_ids and relations are unchanged, so UNIQUE (lineage, ordinal) holds. Re-key P's vectors to L; P's newest becomes L's cursor when it is newer.
  4. A uid with a canonical binding in both: L's stays canonical, and P's becomes a revision with `possible_replay_of`. Nothing is deleted.
  5. Set max_store_id and max_leaf_id to the larger value, recompute legacy_proof_digest from the merged proof keys, and record `adopted_from` (the #1520 mechanism). The same merge corrects ownership invented on a cold post-turn ingest (#1250).
- **At P1 the merge binds only:** step 2 uses migration's byte-exact, consume-once matching and writes relations only; every occurrence of P, replayed or not, is re-keyed to L with a fresh ordinal (step 3), together with every reference to it; view_seq is store-global, so "newer" in step 3 is defined.
- **Fork child.** It starts a new lineage with a parent link (by occ_id). While the parent lineage is provisional, the link stays provisional. A missing parent stays provisional and never becomes a root (L4-8).
- The host's view model is not declared (M7).

### 3.9 Readers (A5, A12a)
These ship with the P2 flip:
- **The doctor reads the ledger.**
  - Per session, it reports counts and lists of the ambiguous, possible_replay_of, supersedes, generated_unproven and migration_residual rows.
  - It also reports the counters: cost fallbacks, re-spacings, stamp_unreliable lineages, keyed reorders, anchorless views, and busy-at-end events.
- **The compaction source mapper** resolves sources through occurrence bindings, lineage-wide, for ledger-era rows.
  - **A split_part group is indivisible (Astra 2).** The chunker never cuts inside one: a leaf's source range holds all of a group or none of it. The group's stored composite is claimed only by the leaf that holds the whole group. The exact-ID frontier checks stay as they are (E7).
  - This closes #2029, #1402 and #553's split half.
- **Other readers** (assembly, recall, grep, expand, embeddings) treat marked rows as ordinary stored rows at v0.30.0. Nothing is hidden or deduplicated, and M3 bounds how many there are.
- `store_complete` coverage moves to a ledger reader after the gate.

## 4. Child map (every v0.30.0 milestone issue; the closing phase in brackets)
| Class | Mechanism | Issues | Fixed by |
|---|---|---|---|
| A | Evidence not atomic with rows | [P1] #574; #1440 item 1 (the P1 tail rule; items 2, 3, 5, 6 split out as non-identity); #1451 item 4; #534 residuals (the atomic exit write); #1805; #1815; #1978 (reader stamp); #1904 item 1 (items 2-5 are test hygiene, folded into #1451's tests). [P2] #1506; #1508 (needs carried_stamp reads) | §3.2, §3.3, §3.5 |
| B | Cursor or proof lost on restart or rotation | [P1] #758(b); #1523 (recovery-emission binding); the #1328 regression family (the durable-cursor fast path). [P2] #545 (the retained-prefix witness); #1838 (rank-3 scoping, unkeyed definition); #2029 (§3.9 mapper; its exit write is P1); #588 (both halves: migration's alignment is P2) | §3.1, §3.3, §3.4.2, §3.6, §3.9 |
| C | Unkeyed rows decided by position or content | [P2] #7; #1525; #1566 (anchored offset; the all-identical window is the M2 limit); #1446; #1967 (rank-3 stamp veto; the engine.py:7750 before 7752 ordering hazard); #1451 item 3; #1674 (h2); #1563 (ordinals); #1573; #1547; #607 (multiplicity and legacy twins); #1519 (h4 and the Window: a visible duplicate, not a loss); #1133's default-chunk replay cell (from #1208) | §3.4, E2b, E5b, E5c, §3.6 |
| D | Host-uid evidence ignored or misused | [P1 tables, P2 rules and uid `on`] #1836; #1837; #1869; #1179; #1428; #1449 (Window on rank 1); #1512; #1520; #1922; #1924; #1795; #1451 rewind variant | E5 rank 1, E5a, §3.1 item 2, §3.2, §3.8, M16 |
| E | Identity drift from transforms | [P1 digest, P2 aliases] #758(a) (one-time residual disclosed); #1836-#1838 (pruned outputs); #588 stub-vs-raw alias. #10's spillover half is fixed by #739 (merged 10-28; keep its test), and its Windows CRLF half is split out as #2046 (same class, P2) | §3.1 salted digest, E5b aliases |
| F | Composite and carrier claims | [P2] #1506; #1508; #2032-F1; #2032-D1; #1003 (rank-4 stamp veto); #1961 and #1446 (emission merged with a host row); #1402; #553 (coverage guard; close as fixed by #581 if a real-host reliability run compacts cleanly); #606 (P3 note: fixed via E6 once §3.3 lands); #923 (now visible: supersedes may target a generated entry, plus possible_replay_of; eliminated only by class D) | §3.4.6, E6, §3.9 |
| G | Admission: rows never offered to ingest (NOT fixed by the ledger) | [own children, in parallel with P1] #1931 (delegated-child sessions bypassed, bypass.py:151-163); #2042 (after its measurement split: children go to #1931, top-level is P1 v0.30.0); #1250 (ownership; its lineage half is §3.8) | Separate gate children, own PRs |

- #1451 items 1-2 are done.
- Milestone items outside this design, each with its own issue and PR: #1607, #1650, #1652, #1656, #1666, #1898, #1959, #2045.
- Not product shapes:
  - #824 is an instrument artefact (see #861).
  - #1328's point fix stands; its regression family is in class B, and its test is kept.
  - #491 is out of scope (redaction).

**Regressions.** Every child gets a regression that fails on v0.29.0, or is strict-xfail there.
- The drafts' strict-xfail and RED tests carry over: #742, #756, #769, #783, #784, #924, #1544, #1914, #1935, #1937, and #1133's default-chunk replay cell.
- Each rule added in revision r3.2 has a named case:
  - test_issue_1519 (anchorless and moved-past);
  - #1449 uid reuse;
  - #1967 re-stamped tool pair on a cold resume;
  - #1003(c);
  - [x, x, y] twins;
  - [p, q] → [q, p];
  - [A, B] → [A] → [A, A] → restart → [A, B];
  - unstamped R → R+N1 → R+N1+N2;
  - [A(u)] → [B(u)] → [Z] → [B(u)];
  - a split group at a chunk boundary;
  - a P → L merge with a shared uid;
  - a legacy alt_stamp written after a downgrade;
  - a failed emission write followed by a restart;
  - an all-restamp view.
- Parity guards that are green on v0.29.0 are kept and labelled as guards, not as a child's regression. For example, E2b's two identical unkeyed segments are a guard; #607's fresh-bind twin is the regression.
- test_restart_after_shrunk_resend keeps its pin as an upper bound. The ledger-era target is 0 re-stores.

## 5. Phases (v0.30.0 ceiling 11-18; ships when the gate passes)
- **P0, refactor moves: NOT prerequisites (M13).** The cycle break (#2037) and the scheduler-family move land in the refactor window after v0.29.2 if they are ready. P1 and P2 do not wait for them; whichever lands second rebases.
- **P1, ledger and atomic ingest (M15).**
  - Contents:
    - the ledger schema (docs/process/schema-changes.md);
    - one transaction per ingest, with the explicit BEGIN IMMEDIATE, the complete write list and the snapshot re-check (§3.2);
    - the compress() single exit, its emission records and its visible failure path (§3.3);
    - the lineage marks with all their writers;
    - the durable cursor;
    - the lineage key and the provisional merge (§3.8);
    - binding-only migration (§3.6);
    - the P1 tail rule (§3.5);
    - stamp observations;
    - the per-ingest counters (§3.2);
    - the reliability scorer reads relations, so a visible surplus row is told apart from a silent one.
  - **Behaviour:** today's classifier keeps deciding. Four fixes take effect at P1, because each one either stores more or skips only on exact evidence:
    1. atomic evidence (class A);
    2. on a cold start, the anchored run at j = 0 on the durable cursor, or on the newest non-refused emission vector, becomes one more candidate in today's largest-cursor choice (reconcile.py:3085-3103). It never replaces tail matching, so rows stored by a tail retry or an S3 end, which write no vector, are still found (the class B restart family, and a crash between the compress() return and the next ingest);
    3. the single exit write and its visible failure path (#2029's exit half, #606, the #534 residuals);
    4. the P1 tail rule (#1440 item 1).
  - It closes the [P1] children in §4.
  - **Gate for P1:**
    - The injected-crash cells (C1-C4, the session-end tail including one frozen before any decision and retried after other ingests, and an emission-write failure followed by a restart) show 0 loss and 0 silent surplus. A surplus row that carries a relation is visible; a surplus row without one fails.
    - The release spec's existing turn-latency bars hold (D5 and D7). D2 measures facts kept, not latency.
    - Write-lock hold is at most 50 ms at p90 and 250 ms at p99 on views of up to 2,000 rows, read from the new counters.
    - Busy errors stay at or below the v0.29.x count on the same cells.
- **G, admission** (in parallel with P1; a gate prerequisite). #1931 and its delegated-child reliability cell, then #2042 after its split. Own spec, own PR.
- **P2, classifier.**
  - Contents: E2b's key order and M14's stamp control, E5a-E5c, the §3.4 evaluation order, PARTIAL with the witness, rank 5 with counts and the budget, migration's classifier, the §3.9 readers, and uid mode `on` (M16).
  - **Shadow first.** Decisions are logged next to today's on every reliability cell (including the uid-capable upstream-main cell and the ACP-driven cells) and on the parity corpus: the reference agent's recorded host views, replayed engine-direct.
  - **Flip bar:**
    - Scope: every reliability cell, whose ground truth comes from the scripted material, and at least 7 days of the reference agent, whose ground truth is the parity audit's exact-content comparison with the host transcript.
    - 0 new rows that the shadow decisions would skip.
    - 0 silent surplus.
    - Visible-doubt rows at most 1% of stored rows (M3).
    - Anchorless views at most 0.1% per host.
    - The P1 latency and lock bars hold.
    - The uid default moves to `on` under the same bar, read on the uid-capable cells.
  - It closes the [P2] children in §4.
- **Gate (v0.30.0).**
  - **The parity audit runs on the reference agent,** a host without uids: at least 72 h, at least 10 delegated children and at least 200 turns. It must show 0 sessions with 0 rows, 0 exact-content deficits and 0 silent surplus.
  - Class D is measured on the uid-capable reliability cell (upstream main), with its regressions green under uid `on`. If upstream main stops emitting uids, class D stays open, and the gate receipt says so.
  - The new delegated-child reliability cell is green, and every v0.30.0 child regression is green.
  - **No child whose effect is loss stays strict-xfail at the gate** (an earlier revision allowed #1449, #1512 and #1566). The only exception is a child whose trigger needs a host capability the audit host lacks (named, for example uid emission), AND whose regression is green where that capability exists (a reliability cell or an engine-direct test). The receipt names both.
  - The disclosed limits are measured: M2 on the ACP-driven cells, where it can occur; M9 and the #758(a) residual on the audit. A hit during the audit fails the gate.
  - **M3:** visible-doubt rows are at most 1% of stored rows across the audit's sessions, and the doctor lists every one.
    - The numerator is rows stored with `possible_replay_of` or `generated_unproven`.
    - The denominator is rows stored in the audit window.
  - The P1 latency and lock bars hold.

## 5a. P2 open items (P2-blocking; closed in the P2 spec, each with a named regression, before the P2 flip)
Sources: the r3.2 findings of the Codex reviewer (A) and of the Claude reviewer (C).
- A1: the M14 restamp control can erase stamps that tell new rows apart. Keep the veto unless restamping has independent evidence; otherwise store visibly.
- A2: h2 and h3 let a stale candidate or a reused uid set its own head. Require head evidence independent of the candidate; otherwise h4.
- A3: the anchored run can accept an older revision of a uid before E5a runs. Validate revisions before accepting anchored entries.
- A5 and C7/C10: head selection defeats the [p, q] → [q, p] and keyed [B, A] rules, and rows outside every interval get no candidate. A NEW row equal to an unbound entry of the base record (under h4, of the durable cursor) gets possible_replay_of; state the keyed-reorder outcome.
- A6 and C6: the retained-prefix witness is unreachable for [R] → [R+N1], and an emission merged into V[0] has no anchor. Select the previous vector independently of the run; split an emission-merged V[0] by its descriptor before the search; let the run start at a small V[i] with an empty leading gap.
- C8: P2's first ingest of a lineage that still has unbound pre-ledger rows runs P2 migration over them once, so #758(a) stays measurable.
- C9: the first ingest that binds a generated row records its observed stamp, for every kind, so the #1003 veto arms.
- PR review 1: after a rollback, an older reader can backfill `messages.observed_at` with no relation. At re-upgrade, import those backfills as legacy stamp observations, so the stamp veto sees them. Named regression: downgrade, backfill, re-upgrade.
- PR review 2: rank 5's target includes superseded occurrences, so an incoming row equal to a superseded one is counted as REPLAY. Exclude superseded occurrences from the lo_d/hi_d counts; a match to one takes the NEW path with `supersedes` and `possible_replay_of`. Named regression: x, then x superseded by x', then x again.
- C11: the 4M-cell budget against one h2 core; M14's trigger strength; "in every LIS" with several same-stamp candidates; a history base never pairs supersedes (#7).

## 6. Decisions (the release manager decides; review may challenge them)
- **M1 Lineage:** §3.8, keyed by the persisted host_uid key (A8), with the five-step provisional merge.
- **M2 Identical unkeyed re-send:** REPLAY, disclosed. It extends to re-growth after a shrink and to an identical overlap in migration or a full-history replay, and is measured on the ACP-driven cells.
- **M3 Duplicate budget:** a tie stores its rows and never skips them. Visible-doubt rows are at most 1% of stored rows (numerator and denominator in §5), and the doctor shows every one.
- **M4 Digest:** salted and pre-redaction, no catalog versioning. The salt lives in the same file.
- **M5 Locking:** commit under `BEGIN IMMEDIATE`, re-classifying when the snapshot changed; the UNIQUE indexes are the backstop.
- **M6 Retention:**
  - Binding, uid, revision and emission rows are kept forever.
  - Stamp observations are capped at 4 per occurrence and kind.
  - View vectors keep the last 2 of each kind.
  - The doctor and the source mapper become ledger readers at P2; store_complete after the gate.
- **M7 View model:** not declared.
- **M8 Rollback:** per-lineage marks, written by every writer of what they cover, plus migration mode. Legacy identity-only writes are read directly (§3.6).
- **M9 Session-end kill window:** accepted, narrowed and measured. A hit during the audit fails the gate (§3.5).
- **M10 Generation model:** one vector per ingest and per emission. A generation is an emission epoch. A refused emission is never a base. Rank 5 aligns each gap against the lineage's kept bindings (§3.4.5).
- **M11 Write rule:** the ledger records only stored rows, uids and revisions, emissions, relations and new stamp observations (§3.1).
- **M12 Admission:** class G is in scope (ROADMAP v0.30.0) and is a gate prerequisite. It is a separate child, not part of the ledger.
- **M13 Refactor moves:** not prerequisites (§5).
- **M14 Key order and stamp policy (new):**
  - The key order is uid > stamp > tool_call_id > position, and a known stamp disagreement vetoes ranks 3-5 (E2b).
  - A lineage whose views re-stamp rows is marked stamp_unreliable once, and for good, and its stamps then count as missing (#1511).
- **M15 Phase split (new):** P1 keeps today's decisions apart from four named effective fixes. The classifier and migration's alignment are P2 and run in shadow first (§5).
- **M16 Uid modes (new; config.py:254-267):**
  - `off`: no uid capture and no uid reads.
  - `shadow` (today's default): bindings are written, and rank 1 is computed and logged but never used to skip.
  - `on`: rank 1 decides. Today `on` behaves exactly like `shadow`; P2 gives it this meaning, and the default flips to `on` under the P2 bar.

**Next:**
1. Phase P1 is implemented against this design, in its own issue (linked from #1993).
2. The G child (#1931) gets its own spec.
3. The P2 issue's spec review closes the §5a items, each with a named regression, before the P2 flip.
