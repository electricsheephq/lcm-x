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
- **Normalisation:** `alias_norm` = NFKC, casefold, whitespace collapsed to single spaces, leading and trailing punctuation stripped.
- **`alias_kind`:**
  - `name`, `nickname`;
  - `relation_phrase`: requires `owner_entity_id`, the speaker who uses the phrase. "my sister" said by U1 and by U2 are different aliases;
  - `platform_id`: `<platform>:<id>`, created only from a verified gateway `sender_source` (#995).
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
- **Logging.** Every merge, split, alias change, kind change, forget and undo is a row in `lcm_memory_events` (`op`, `payload_json`, `actor`, `created_at`, `undone_by`). Nothing curated is updated in place except `status`, `merged_into` and `retired_at`. Each change to those three fields is mirrored by an event.

### D5. Automatic merges
The agent or a background pass acts without asking only in these cases:
- **(a) A verified platform id.** A new sender row whose verified `platform_id` already has an active holder **attaches** to that holder; no new entity is created. This is not a merge of two entities, since D3 forbids two active holders of one `platform_id`.
- **(b) An explicit cited statement** ("Alex is Alexander") auto-merges only when it comes from the profile's user or from a verified sender. The merge always appears on the review queue and can be undone.
- **(c) User confirmation.**

Otherwise the agent asks: in chat when the current answer depends on it, else on the review queue.

### D6. The bridge to #1191 (`lcm_entity_keys`)
- **Shape.** `lcm_entity_keys(subject_key, entity_id, method, created_at, retired_at)`, with `method` ∈ {`alias_match`, `override`}.
- **It is derived.** It is rebuilt from current aliases, plus override rows that are themselves recorded as events.
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
  - **Relation-phrase keys** (`person:sister`) resolve only with an owner: the verified sender of the assertion's `source_store_id`. With no verified sender, the key stays unresolved. It is never collapsed into one shared entity.
- **Subject-key churn.** A re-extraction under a new `extraction_version` may produce a different key for the same subject. The bridge resolves the new key through aliases again.
  - Override rows are keyed by `subject_key` text, so they survive an `extraction_version` change.
  - A key that disappears is retired, not deleted.
- **#1191 is unchanged.** `lcm_assertions` is not altered.
- **No delivered content changes yet.** Bridge resolution changes no delivered content until E2 passes its gate (invariant 5).

### D7. Tenancy
- Every entity, alias, note, link, key and event row carries `scope_key`.
- In v1, `scope_key = 'profile'` for every row: each profile has its own store, and there are no cross-store reads.
- A future Teams scope is an additive value with an explicit share. It is not a migration.

### D8. Labels are not permissions (invariant 6)
- A sender or author label never decides recall access.
- When AccessContextV1 (`docs/access-context-v1.md`) is active, an entity note or link is visible to a principal only if every source row it cites is visible to that principal.
- Entity cards apply the same filter.
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

## Consequences
- The bridge needs aliases to exist before it resolves anything. Until E1 and E2 populate them, #1191 keys stay unresolved. That is safe: today's behaviour.
- Ambiguity is visible to the agent and on the review queue. That is extra UX work, accepted in exchange for avoiding silent wrong merges.
- Relation phrases depend on S0 sender capture. In single-user profiles the owner is the user, and the profile's user entity serves.

## Tests this ADR requires (E1/E2 specs carry them)

**"Identical state"** means an identical **resolved view**: the canonical map, alias → candidate sets, entity cards and bridge keys. Rows are append-only, so the raw tables cannot be identical after an undo. The event log may differ only by the appended events. Each property is run against the candidate before it gates.

- **Property:**
  - merge then undo returns the identical resolved view;
  - split then undo returns the identical resolved view;
  - a merge is refused when `canonical(A) == canonical(B)`, including A == B and cycles;
  - merging an already-merged entity resolves to its canonical;
  - a split that moves a `platform_id` alias never leaves two active holders, even mid-transaction.
- **Platform-id attach:** a new verified sender row with a held `platform_id` attaches to the holder and creates no entity.
- **Ambiguity:** two entities share an alias; lookup returns both and marks the result ambiguous.
- **Relation phrase:** "my sister" from two verified senders gives two entities; with no verified sender, the key stays unresolved.
- **Bridge churn:** after re-extraction under a new `extraction_version` with a changed subject key, the key resolves through aliases, and override rows survive.
- **Byte-identity:** with the flag off and on, and the tool unused, recall output and assembled context are identical across a fixture corpus.
- **Whole-row consumers (D9):** with the new nullable columns present, every enumerated `SELECT *` consumer's output is unchanged, or changed exactly as declared.
- **Namespace table (D6):** each namespace resolves only to its listed kind; `event` and `conversation` keys stay unbridged.
- **Access (D8):** under AccessContextV1, a note citing one invisible source row is hidden.
