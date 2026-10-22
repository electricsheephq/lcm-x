# ADR: entity identity, aliases, merge/split and tenancy (Entities v1, E0)

Status: ACCEPTED by the release manager on 2026-10-22, with review amendments A1–A6 folded in. Findings from the pending cross-model architecture review land as an amendment before E2 is dispatched. Tracking: #1862. Refs #1191, #1171, #995, #75.

This ADR fixes the parts of the entity layer that are hard to reverse once rows exist:
- how an entity is identified;
- how names map to it;
- how merges and splits work;
- how it bridges to #1191's assertion subject keys;
- how it is scoped.

It does not decide:
- name discovery (E3);
- the review UI (E3);
- consolidation (E5);
- typed links (E6).

"Entity identity" here is unrelated to the message identity ledger (`docs/identity-ledger.md`).

## Context

`lcm_assertions.subject_key` is free text (`namespace:value`, e.g. `person:alex`). Two failures follow:
- **Split identity.** `person:alex` and `person:alexander` are different subjects.
- **Collapsed identity.** Every speaker's "my sister" becomes one `person:sister`.

An earlier internal triple-store prototype keyed facts on (entity, predicate) and lost true facts on overwrite. Opaque ids and append-only curation avoid both failures.

## Decisions

### D1. Opaque entity ids
- `entity_id = "ent_" + 32 lowercase hex characters` (128 random bits).
- An id never encodes a name, kind or time, and is never reused.
- All ids come from one generator function that accepts an injected random source, so tests are deterministic.
- Entities are curated rows. Derived tables point at them; they are not rebuilt from text.

### D2. Kinds
- Closed set: `person`, `agent`, `org`, `project`, `place`, `thing`, `topic`, `other`.
- Changing an entity's kind is a recorded event.
- The agent itself is one `agent` entity per profile.

### D3. Aliases
- **Normalisation of names:** for `name`, `nickname` and `relation_phrase` only, `alias_norm` = NFKC, casefold, whitespace collapsed to single spaces, leading and trailing punctuation stripped.
- **Platform ids are never name-normalised.** A `platform_id` is stored as `<platform>:<id>`: the platform lowercased, the id byte-exact.
- **`alias_kind`:**
  - `name`, `nickname`;
  - `relation_phrase`: requires `owner_entity_id`, the speaker who uses the phrase. "my sister" said by U1 and by U2 are different aliases.
    - Ownership comes only from a **direct first-person statement by a verified sender**, and extraction must mark the statement as first person.
    - Quoted, reported or second-person phrases ("your sister", "she said her sister") stay unresolved.
  - `platform_id`: created only from a verified gateway `sender_source` (#995).
- **Provenance.** Every alias carries the source rows it came from (`source_store_ids`). An alias with no source rows is visible to the profile owner only (see D8).
- **One alias, several entities.** The same `alias_norm` may map to several entities. Lookups return all candidates and mark the result ambiguous. Nothing silently picks one.
- **The one hard identity.** At most one active entity per (`scope_key`, `platform_id`).
- **Removal.** Aliases are retired (`retired_at`), never deleted.

### D4. Merge and split are append-only events
- **Merges work on canonical ids.** Merge(A ← B) first resolves both sides to their canonical entities: merging from or into an already-merged entity uses that entity's canonical.
  - The merge is refused when `canonical(A) == canonical(B)`. That covers both A == B and any cycle.
  - The canonical B gets `status = merged` and `merged_into = canonical(A)`.
  - B's aliases, mentions and notes keep pointing at B. Readers resolve `canonical(B)` by following `merged_into`.
  - Undo restores B to `active`.
- **Split(E → E2, a selected subset).**
  - Selected aliases are retired on E and added to the new entity E2. A moved `platform_id` alias is retired on E and added on E2 in **one transaction**, so D3's one-active-holder rule never fails midway.
  - Selected notes are superseded by copies that point at E2.
  - Mentions are derived, so they are rebuilt for the moved aliases.
  - Undo replays the inverse event.
- **Logging.** Every merge, split, alias change, kind change, forget and undo is a row in `lcm_memory_events` (`event_id`, `op`, `payload_json`, `actor`, `created_at`, and `target_event_id` for an undo). Nothing curated is updated in place except four projected fields: `status`, `merged_into`, `retired_at` and a note's `superseded_by`. Each change to them is mirrored by an event in the same transaction, and replaying the log reproduces all four.
- **One op, one event.** A composite operation (merge, split, undo) writes exactly one event, whose payload records the before and after value of every projected field it changed and the ids of the rows it created.
- **Kind is derived.** A kind change is an event. The current kind is the latest kind event that has not been undone, else the kind the entity was created with. Every kind comparison uses the current kind.
- **Undo is an event, and "undone" is derived.**
  - An undo appends an event whose `target_event_id` names the event it reverses.
  - Whether an event is undone is derived from the log; no column on the original event is updated.
  - **Chained undo.** An undo is refused while any later event that is not itself an undo, and has not been undone, touches the same entities, aliases or notes (including rows that event created). Undo events never block. Dependents are undone first, last-in first-out per affected object.
  - **Void, not delete.** Rows created by an event that is later undone are excluded from the resolved view. Aliases among them are also retired.
  - **Undo order.** An undo first retires the aliases its target created, then restores the ones it retired. Restoring a `platform_id` that another entity now holds is refused; it is never left to the unique index.
  - **No redo in v1.** Undoing an undo is refused; the operation is run again instead.

### D5. Automatic merges
The agent or a background pass acts without asking only in these cases:
- **(a) A verified platform id.** A new sender row whose verified `platform_id` already has an active holder **attaches** to that holder; no new entity is created. This is not a merge of two entities, since D3 forbids two active holders of one `platform_id`.
- **(b) An explicit cited statement** ("Alex is Alexander") auto-merges only when it comes from the profile's user or from a verified sender. The merge always appears on the review queue and can be undone.
- **(c) User confirmation.**

Otherwise the agent asks: in chat when the current answer depends on it, else on the review queue.

### D6. The bridge to #1191 (`lcm_entity_keys`)
- **Shape.** `lcm_entity_keys(subject_key, owner_entity_id, entity_id, method, source_store_ids, created_at, retired_at)`, with `method` ∈ {`alias_match`, `override`}.
  - The key's identity is (`subject_key`, `owner_entity_id`).
  - **Relation-phrase keys** map per (`subject_key`, owner). So do **`user` keys resolved through a source row's verified sender**: two senders' `user:self` are two keys, never one shared row. In a single-user profile, `user` keys resolve to the profile's user entity. Every other key has `owner_entity_id = NULL`.
  - Keys carry the source rows they were resolved from (`source_store_ids`).
- **It is derived.** It is rebuilt from current aliases, plus override rows that are themselves recorded as events.
- **Overrides are never global while a key is ambiguous.** An override applies only:
  - (a) when the key has exactly one candidate, to all of the key's source rows it cites (a global override); or
  - (b) scoped to the assertion or source rows it disambiguates.

  A global override is refused while the key is ambiguous. This check counts every candidate in the scope, independent of the requester, and the requester gets a generic refusal with no ambiguity flag or candidate count, so it never reveals an entity they cannot see (D8).
- **Resolution:**
  - Take the `value` part of `namespace:value`, normalise it as in D3, and look it up among aliases.
  - Exactly one candidate of a compatible kind gives a resolved key. Zero or several candidates leave it unresolved; nothing guesses.
  - **Namespace → kind compatibility.** The namespaces are #1191's `_SUBJECT_NAMESPACES` in `assertion_extraction.py`:

    | Namespace | Entity kind | Notes |
    |---|---|---|
    | `person` | `person` | |
    | `user` | `person` | Resolves to the profile's user entity, or the verified sender of the source row |
    | `assistant` | `agent` | The profile's agent entity |
    | `organization` | `org` | |
    | `project` | `project` | |
    | `place` | `place` | |
    | `object` | `thing` | |
    | `account` | `thing` | |
    | `document` | `thing` | |
    | `event` | not bridged in v1 | Events stay assertions; typed links are E6 |
    | `conversation` | not bridged | A conversation is a session, not an entity |

    The match is exact. Entities of kind `other` or `topic` never resolve a key automatically; an override can link them. A namespace added later is unbridged until this table lists it.
  - **Relation-phrase keys** (`person:sister`) resolve only with an owner: the verified sender of the assertion's `source_store_id`, and only when the source statement is marked first person (D3). With no verified sender, or with quoted, reported or second-person phrasing, the key stays unresolved. It is never collapsed into one shared entity.
- **Subject-key churn.** A re-extraction under a new `extraction_version` may produce a different key for the same subject. The bridge resolves the new key through aliases again.
  - Override rows are keyed by `subject_key` text and carry the source rows they were made for. **An override, global or scoped, binds only the source rows it cites.** New evidence that produces the same key is not bound by it until it is reviewed. The override stays dormant for those rows even while an original source is current. For each cited row, it applies while that row is still a current source of the key, so it survives an `extraction_version` change of the same evidence; a reused key text never binds unrelated new evidence.
  - A renamed key with no matching alias stays unresolved.
  - Source-scoped overrides are recorded as events and applied per source row when a key is read; they are not materialised as key rows.
  - A global override's key row (`method = override`) carries its cited `source_store_ids` and resolves only assertions whose source row is among them. Other assertions with the same key resolve through aliases, or stay unresolved.
  - A key that disappears is retired, not deleted.
- **#1191 is unchanged.** `lcm_assertions` is not altered.
- **No delivered content changes yet.** Bridge resolution changes no delivered content until E2 passes its gate (invariant 5).

### D7. Tenancy
- Every entity, alias, note, link, key and event row carries `scope_key`.
- In v1, `scope_key` is the bound profile identity (the same identity D10 records on file revisions), set on every insert. Normally each profile has its own store. When a configured `database_path` makes one database serve several profile homes, rows from different profiles still never share a `scope_key`. So D3's hard identity, D5(a) attach, D6 keys and every read are per profile, and no read crosses `scope_key`. Every mutation target lookup (D4 merge, split and undo, D5(a) attach) and every canonical traversal is scoped to the bound `scope_key`: an id from another scope is "not found".
- A future Teams scope is an additive value with an explicit share. It is not a migration.

### D8. Labels are not permissions (invariant 6)
- A sender or author label never decides recall access.
- When AccessContextV1 (`docs/access-context-v1.md`) is active, an entity note, link, alias or key is visible to a principal only if its cited set is **non-empty** and every cited row is visible to that principal.
  - Uncited records are hidden unless an explicit grant exists.
  - An alias with no source rows is visible to the profile owner only.
  - **Citation kinds.** A citation is either a store row (`store_id`) or a host memory-file revision (`lcm_memory_host_file_revisions.revision_id`, see D10). A file-revision citation counts toward the non-empty cited set. A record citing any file revision is visible to the profile owner only, and every other citation it carries must also be visible.
  - **Forget hiding is a delivery filter, not a visibility change (D11).** Hidden rows still count as visible when D8 evaluates the citations of the retraction note that hides them, so the note and its withdrawal marker stay visible to the forgetting principal. Derived records (alias, key, link, card) whose ONLY supporting provenance is hidden for the forgetter are filtered for that forgetter by the same per-principal delivery rule; the retraction note is the one exception.
  - **Entity visibility.** Entity rows carry no citations of their own. An entity (its canonical-map entry, card and candidate slot) is visible to a principal only when at least one of its own aliases or notes, or those of an entity merged into it, is visible.
- **Ambiguity is counted after visibility.** Under AccessContextV1, alias and key candidates are filtered by visibility **before** ambiguity is counted. Stored key rows (D6) are the principal-independent derivation. Resolution and ambiguity for a reader are evaluated at read time over that reader's visible candidates. Candidate sets and ambiguity flags therefore never reveal an entity that the principal cannot see.
- Entity cards apply the same filters.
- Without AccessContextV1, no entity content is injected automatically. The agent reaches entity content only by calling `lcm_memory`, the same way it reaches `lcm_recall` results today.
- Rows without a gateway `sender_source` carry unverified labels.
- **Scope of D8.** D8 covers entity notes, links and cards reached through `lcm_memory`. #1191's own profile-block injection (gap 3) is governed by #1191's spec, where invariant 6 is also written. D8 neither permits nor forbids it.

### D9. Byte-identity (invariant 5)
- Messages gain nullable `sender_id` and `sender_source` columns.
- Until #995 rules on rendering the sender, these columns are not rendered into FTS text, chunk-embedding text or delivered hit text.
- With the feature flag on and `lcm_memory` unused, every recall result and every assembled context is byte-identical to the flag being off.
- The one declared difference is the `lcm_memory` tool schema in the tool list. Its size is measured and reported.
- **New columns reach every whole-row reader.** Adding columns changes every `SELECT * FROM messages` consumer. Those found on main:
  - `bench/instruments/local_catchup/audit.py` (a whole-row digest; a change would break #1586 instrument comparability);
  - `bench/instruments/local_catchup/measurement.py`;
  - `bench/instruments/reliability/probe.py` (four sites);
  - `bench/instruments/reliability/stop_turn.py`;
  - `scripts/store_wedge_inspect.py`;
  - about 20 test files, mostly before/after comparisons of the same query.
- **E1 requirement:**
  - Enumerate every consumer of the messages table (product, bench, scripts, tests).
  - For each one, either pin an explicit column list, or declare the change with an instrument-hash note for any digest that moves.
  - A test shows those outputs are unchanged, or changed only as declared, once the new columns exist.

### D10. Host memory files (MEMORY.md and USER.md)
- **Flag.** Capture runs only with the entities flag on (`LCM_ENTITIES_ENABLED`, default off). With the flag off, no revisions or notes are written and the files are never opened.
- **Why.** Hermes injects both files into every system prompt. LCM-X reads them for history, removals and entity-card coverage, not for current facts.
- **Capture.** At session start, and at idle where the engine has an idle point, LCM-X reads the bound profile home's `memories/MEMORY.md` and `memories/USER.md`.
  - Entries are split the way Hermes's own memory store parses them. The entry, not the line, is the unit.
  - Each content change observed at a snapshot point (session start, idle) becomes one `lcm_memory_host_file_revisions` row. Changes made and replaced between two snapshot points are not seen; capturing at Hermes' write boundary is a later host-hook option. Each such row has `revision_id`, absolute `path`, `revision_sha`, `content_hash`, `captured_at`, and the bound profile identity the engine already uses for ownership.
  - A file-revision citation is visible only to the owner of that same profile. When one database serves several profile homes (a configured `database_path`), each profile's files are captured under its own identity, and one profile never sees notes from another profile's files. The entry text itself is kept in the note it creates.
  - No message rows are written (invariant 5). The files are never written.
- **Unreadable file.** An unreadable or undecodable file records no revision and changes no notes, so an uncertain read is never treated as an empty file. Capture never blocks or fails session start; the failure is logged and counted in `lcm_status`.
- **Notes.**
  - An added entry becomes a note with `author = host_memory_file`, cited to its revision.
  - Authorship is never inferred from the file name or a role: `asserted_by_entity_id` stays empty.
  - A removed entry supersedes its note, which remains as history.
  - A removal is a retraction only when it traces to an explicit user forget request (a message row). Otherwise it may become a review-queue candidate.
- **Trust.** Inside LCM-X delivery, file-derived notes carry the same trust label as other recalled memory, with no elevation. Hermes's own system-prompt copy of these files is outside LCM-X and keeps its host behaviour; this ADR makes no claim about it.

### D11. Forget (E2)
- **Flag.** Forget hiding runs only with the entities flag on (`LCM_ENTITIES_ENABLED`, default off). With the flag off, no forget notes are written and nothing is hidden. The default stays off until the E2 measurement reports zero false hides.
- **What a forget is.** `lcm_memory forget` writes a `kind = retraction` note. It cites the user's request and names the retracted message rows or note. The note is the only flag.
- **Authorization.** A forget may name only rows and notes visible to the requesting principal. Any other id, foreign or nonexistent, gets the same refusal, writes no note and reports no count.
- **Hiding.** While the note stands, the retracted rows are hidden from every agent-facing read path. The hide is bound to the requesting principal: it applies only to reads evaluated in that principal's access context. Other principals who can see the same row still see it, until they forget it themselves. Only the principal who forgot a row can undo that forget. A single-user profile is unaffected. Responses report a count of hidden rows, never their content. The fresh tail and the live host transcript are not hidden: LCM-X does not rewrite the live conversation. Hiding is a delivery filter, not a visibility change (D8): the retraction note and its withdrawal marker stay visible to the forgetting principal.
- **The request row.** The request row that carries a forget is hidden together with its targets, on the same paths and under the same principal binding, because it usually restates the fact. The retraction note keeps the citation for audit and undo. The fresh tail and live transcript stay excluded, as for the targets.
- **Lossless.** Message rows, summaries, FTS and embeddings are never modified. Undo of the forget restores byte-identical retrieval. Real deletion (purge) is a separate, owner-confirmed operation and never a side effect of forget.
- **Rollback.** Readers older than D11 do not honour forgets. A rollback to them makes forgotten rows visible again, because forget is not purge. A user who needs removal on every version uses purge. The rollback test asserts that the older reader opens the store, and the release notes state this boundary.
- **Scope in v1.** A forget targets rows and notes, not a whole entity, so it needs no cutoff rule for future statements. A later statement is new evidence.
  - Summaries are never rewritten or hidden. A delivered summary that spans a forgotten row is **marked**: it carries the withdrawal note.
  - Host memory files (D10) are outside this mechanism. An entry there is removed through Hermes's own memory tool.
- **Detector links are not forgets.** The query-time retraction detector only changes how `lcm_recall` renders a hit. It never hides anything on any other path.

## Consequences
- The bridge needs aliases to exist before it resolves anything. Until E1 and E2 populate them, #1191 keys stay unresolved. That is safe: today's behaviour.
- Ambiguity is visible to the agent and on the review queue. That is extra UX work, accepted in exchange for avoiding silent wrong merges.
- Relation phrases depend on S0 sender capture, in single-user profiles too. There, once S0 verifies the user's rows, the owner is the profile's user entity.

## Tests this ADR requires (E1/E2 specs carry them)

**"Identical state"** means an identical **resolved view**: the canonical map, alias → candidate sets, entity cards and bridge keys. Rows are append-only, so the raw tables cannot be identical after an undo. The event log may differ only by the appended events. Each property is run against the candidate before it gates.

- **Property:**
  - merge then undo returns the identical resolved view;
  - split then undo returns the identical resolved view;
  - a merge is refused when `canonical(A) == canonical(B)`, including A == B and cycles;
  - merging an already-merged entity resolves to its canonical;
  - a split that moves a `platform_id` alias never leaves two active holders, even mid-transaction;
  - **chained split:** split E → E2, then E2 → E3. Undoing the first split is refused until the second is undone. After undoing both in LIFO order, the resolved view is identical.
- **Undo log:** an undo appends an event with `target_event_id`; the original event row is unchanged; the undone state is derived.
- **Platform-id storage:** platform ids keep the id byte-exact (no casefold, no NFKC); only the platform prefix is lowercased.
- **Platform-id attach:** a new verified sender row with a held `platform_id` attaches to the holder and creates no entity.
- **Ambiguity:** two entities share an alias; lookup returns both and marks the result ambiguous.
- **Relation phrase:**
  - "my sister" from two verified senders gives two entities, and two bridge keys differing by owner;
  - with no verified sender, the key stays unresolved;
  - "your sister" and reported speech stay unresolved.
- **Overrides:** a global override on an ambiguous key is refused; a scoped override applies only to its rows.
- **Bridge churn:** after re-extraction under a new `extraction_version` with a changed subject key, the key resolves through aliases, and override rows survive.
- **Override applicability:** unchanged key under a new version keeps applying; a renamed key with and without an alias; a disappeared key; reused key text with unrelated evidence leaves the old override dormant.
- **Owner-scoped `user` keys:** two verified senders each with `user:self` give two keys and two entities.
- **Chained undo, executable:** undo the second of two chained splits, then the first; both succeed and the resolved view is identical.
- **One op, one event:** each operation adds exactly one event; replaying the log reproduces the projected fields.
- **Host files (D10):** a revision with an added entry gives one cited note, visible to the owner only under AccessContextV1. A removed entry supersedes the note and creates no retraction. No message rows are written: the messages-table digest is unchanged.
- **Forget is per principal (D11):** a row is visible to principals A and B. After A forgets it, A's reads hide it and B's reads are unchanged, and B cannot undo A's forget.
- **Host files under a shared database (D10):** two profiles share one `database_path`. Profile B's reads never see notes derived from profile A's MEMORY.md.
- **Shared database scope (D7):** two profiles share one `database_path` and see the same verified `platform_id`. Each gets its own entity (no cross-profile attach), and profile B's reads never see A's entities, aliases, notes, keys or events.
- **Schema compatibility:** `lcm_memory_host_file_revisions`, like every E2 table, follows `docs/process/schema-changes.md`: a forward migration test, a rollback test against the previous release (v0.28.x), and the oldest compatible reader recorded. Readers older than D11 do not honour forgets: the rollback test asserts that the older reader opens the store, and the release notes state that forgotten rows become visible again after a rollback (forget is not purge).
- **Forget (D11):** the same fact, present as raw text, inside a summary and as a note, is asked about before and after a forget, including from a new session. After the forget, the forgotten raw rows and notes are absent on every agent-facing path, and a summary that spans them is delivered unchanged with the withdrawal note present (marked, not hidden). Undo restores byte-identical retrieval, and a principal can never hide another principal's rows.
- **Forget request row (D11):** after "forget that my access code is 1234", neither the target nor the request row is retrievable on any agent-facing path. The retraction note keeps the citation, and undo restores both.
- **Forget keeps its own note visible (D8, D11):** after a forget, the forgetter's spanning summary still carries the withdrawal note, and the retraction note stays visible to that principal although it cites the hidden request row.
- **Override scope (D6):** an original source stays current, new unrelated evidence reuses the key, and that new evidence stays unbound.
- **Forget authorization (D11):** a forget naming a foreign row id and one naming a nonexistent id give byte-identical responses: the same refusal, no note written, no count reported.
- **Host files unreadable (D10):** a permission error and an undecodable file each record no revision, change no notes, never fail session start, and are logged and counted in `lcm_status`.
- **Byte-identity:** with the flag off and on, and the tool unused, recall output and assembled context are identical across a fixture corpus.
- **Whole-row consumers (D9):** with the new nullable columns present, every enumerated `SELECT *` consumer's output is unchanged, or changed exactly as declared.
- **Namespace table (D6):** each namespace resolves only to its listed kind; `event` and `conversation` keys stay unbridged.
- **Access (D8):**
  - under AccessContextV1, a note citing one invisible source row is hidden;
  - an uncited note is hidden without an explicit grant (uncited denial);
  - **two principals:** P1 and P2 each have an entity named "Alex", with P1's sourced only from rows P2 cannot see. For P2, lookup returns one candidate, not ambiguous, and never reveals P1's entity.
