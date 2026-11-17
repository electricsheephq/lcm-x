# Cloud capture, step 1: design record

Status: decided 2026-11-16 by the release manager; this is step 1's design. Five review rounds by a fresh
Claude reviewer and `gpt-6-astra` at high; both approved the fifth edit (Appendices C to G).
A sixth edit (2026-11-16) folds the bot review of the docs PR and lane 0's first results (Appendix H). A seventh
edit (2026-11-16), after a delivery checkpoint, removes `delete-hosted`, `--resume` and the pinning of superseded
copies from step 1, so every chain conflict ends in `--new-copy` and retention is the only thing that deletes
(Appendix I). The page-delta design is folded into 3.7 (Appendix J). Drafted against
`origin/main` at `43e8484c`.
It replaces draft 1 (2026-11-09). Every `file:line` below was read on that commit. A sentence marked
**(assumption)** has no citation and no measurement behind it yet.

Words used with one meaning each:
- **Origin store:** one machine's Claude Code capture store (`<home>/lcm.db` plus `<home>/lcm-large-outputs/`).
- **Origin id:** that store's `capture:owner.store_uuid` (claude_code_store.py:182, :204-208). It names the store.
- **Copy id:** a uuid that `enable` mints on one machine and keeps in `<home>/cloud/cloud.json`. It names the
  hosted copy. It differs from the origin id only so that a home copied to a second machine can get its own hosted
  copy without writing the live store (3.4, Chain).
- **Hosted copy:** the files of one origin store in object storage, under its copy id.
- **Generation:** one published snapshot of a hosted copy, named by its manifest.
- **Replica:** a local, read-only download of another machine's hosted copy.

## 1. Step 1 in one paragraph

Step 1 keeps a checked copy of each machine's Claude Code capture store in storage that only its user can read,
and lets every machine of that user recall from the others. A user who opts in gets three things. Each machine
uploads snapshots of its own store. Each machine downloads read-only replicas of the user's other machines. Recall
answers from the local store as it does today and adds what the replicas hold. Nothing runs in the cloud: the
hosted copy is files in a bucket. Capture still writes only the local store and works with the network off
(VISION.md:68).

**A user with one machine** gets an off-machine copy with a restore that was proven on their own data before
uploads began. Recall gains nothing until a second store exists. When that user replaces or loses the machine, the
old machine's history comes back on the new one as a read-only replica beside the new store. It is never restored
over a live store.

## 2. Decisions the release manager can take now

These need no money and no privacy ruling on a bucket we operate. Each is one line of decision and one of reason.

| # | Decision | Reason |
|---|---|---|
| D1 | One hosted copy per enabled machine, named by its copy id. No merge. | Merging is roadmap row 13 and needs the single writer (ROADMAP.md:24, :127). A copy keeps `store_id`, so identity is exact. |
| D2 | The host stores files and runs none of our code. Recall reads local replicas. | No service to build or secure. The recall wait does not depend on the network. Any S3-compatible store works. |
| D3 | Step 1 covers Claude Code capture stores. Grok Build stores join once Grok capture ships and its own restore proof and copy checker pass. Hermes stores are out. | Grok capture is hard off (docs/claude-code-adapter.md:467-468). A Hermes store has no origin stamp: only capture writes `capture:owner` (claude_code_store.py:50). |
| D4 | Step 1 code never writes a live capture store, its payload folder or its spool, and never takes `capture.lock`. | A held lock tells hooks and workers that a holder will rescan the spool (adapters/claude-code/bin/lcmx_hook.py:348-357; scripts/lcm_x_capture.py:82-83). An uploader that held it would inherit that duty. |
| D5 | The cloud code is a consumer of store contract v1 and changes nothing in it. | The contract already gives the read rule, the tables and the reader API it needs (docs/store-contract.md:8, :14-28, :38-59). |
| D6 | A snapshot is a page-for-page copy taken with SQLite's online backup over a `mode=ro` connection, not `MessageStore.backup`. | `MessageStore.backup` is not a contracted callable (docs/store-contract.md:38-59) and runs on a live `MessageStore` connection (store.py:3427), which the cloud code never opens. |
| D7 | The publish guard is product code. The 0-lost and 0-duplicate measurement is a separate, independent checker. | The outside-in checker is the precedent: it shares no code with what it checks (scripts/capture_outside_in.py:2-6). The worker must not judge itself. |
| D8 | A deliberate delete reaches the hosted copy at the next upload. The opt-in says so, with the limits stated in 5.1. | Decision in force. The forgotten marker makes it cheap (claude_code_store.py:100-108). |
| D9 | Restore writes only into a fresh directory. It refuses a live capture home. | Restoring over a live store would re-capture forgotten sessions and drop rows whose transcripts are gone (critic 4, first item). |
| D10 | Recall across stores is a merger started through `reader-override.json`. It runs one stock reader per store. | The override is the existing seam, and "the plugin never writes it" (docs/claude-code-adapter.md:375-377). Each reader still serves one store. |
| D11 | The opt-in command (`lcm_x_cloud.py enable`) is the one writer of `cloud/cloud.json`. It writes `reader-override.json` only where no other tool's file is, and marks it as its own (4.1). | Critic 3, last item: draft 1 named no writer. |
| D12 | `enable` seeds the hosted copy, restores it into a scratch folder, checks it, and only then schedules uploads. | It makes "one proven restore before any capture" true for each user, not only in our lab. Cost: one full download at opt-in. |
| D13 | Uploads and replica refreshes are started by the merger process: at reader start, then on a timer, plus a manual `sync`. Every Claude Code session starts its own reader, so several mergers can run at once (each session starts the stdio server in adapters/claude-code/.mcp.json): uploads, refreshes and retention run one at a time per home under `cloud/upload.lock` (3.4). | No hook and no capture-worker change, so the capture path stays as gated. |
| D14 | Transport is our own S3 client on the standard library (SigV4, path-style), plus a directory transport for tests. No new dependency. | The unit's runtime is stdlib-only: a found interpreter runs with `-I -S` (adapters/claude-code/bin/lcmx_runtime.py:21-22). **(Assumption:** path-style SigV4 works on R2, B2, Hetzner and MinIO; lane 6 proves it on the owner's test bucket (O1) and on the host that ships.) |
| D15 | Encryption: TLS in transit and the host's default at rest. No client-side encryption for a user's own bucket. For a bucket we operate it is owner decision O2. | The isolation bar needs transport encryption only. Posture: nothing beyond the gate. |
| D16 | Moved to the owner as O3 (section 9). | A lost or stolen machine exposes the history of every one of the user's machines it holds a replica of, not only its own. That widens a privacy promise, so it is the owner's (both reviewers). |
| D17 | Step 1 ships as a Claude Code unit release. A lane that must change `store.py`, `claude_code_store.py`, `mcp_reader.py`, `readonly_store.py` or `tools.py` ships in a core train first. | Decision in force. The plan below needs no such change. |
| D18 | Amend ROADMAP.md:125 when step 1 ships (4.6). | The merged endpoint serves more than "one profile's store" (ROADMAP.md:125). The Grok Build rows (ROADMAP.md:22, :126, :201) already carry section 7's condition. |
| D19 | Forgetting on a replica is done where the replica is materialized: a reader opens a view built without the removed sessions (4.7). A row that links a kept session to a removed one leaves with the removed one; a kept session's own rows stay byte-identical. The merger holds no forget logic per argument. | Fifteen rounds of per-argument patches did not converge (#2796). A session that is not in the file cannot answer. |

## 3. The hosted copy on contract v1

**The hosted copy is a byte copy of the origin store plus a manifest. Everything that inspects it reads only
contracted tables, by column name, over `mode=ro`, plus SQLite's own `sqlite_sequence`, read-only, for the id
check (3.4).** This section adds what the contract does not cover and
restates none of it.

### 3.1 What is uploaded

- **The database file,** whole at a base and as changed pages between bases (3.7). It carries the contracted
  tables (docs/store-contract.md:14-28) and also tables the contract does not list, such as full-text and vector tables (db_bootstrap.py:1527, :1795). Those travel
  uninterpreted. Raw transcript lines are inside it, in `capture_events.raw`.
- **Payload files that rows in the snapshot reference,** and no others. A forgotten session's payload file is
  then absent from the next generation. Payload files are written before their batch commits
  (docs/claude-code-adapter.md:280-282), so every reference the product wrote has its file on disk. References are
  found by scanning text, which also finds placeholders that sessions only quote (lane 0's real store held five,
  such as `...` and `{ref}`). Such a reference names no file, and 3.3 reports it under one of two counts.
- **A manifest,** written last. Its `database` block names the base and complete delta chain (3.7.3).
- **Not uploaded:** transcripts, the spool, logs, the lock, the runtime, `.env`, `capture-owner.json`, the
  override file, cloud settings and credentials.
- **The `.records` sidecar.** Main now has a second database beside the store, `lcm.db.records`
  (store.py:479-496). No production code registers a records table today: outside `store.py:499-514` the only
  callers are tests (tests/test_records_sidecar.py:23). So a capture store should have none. If the file exists, the upload refuses and reports
  `records_pair_unsupported`. Copying the pair is a later lane, pinned in the order `MessageStore.backup` uses
  (store.py:3440-3445).
- **Privacy fact for the opt-in text.** `capture_events.raw` is the lossless copy of each transcript line. Rows
  may be redacted or externalized by ingest protection; the raw line is not (docs/claude-code-adapter.md:618-619).
  The hosted copy therefore holds text that redaction removed from the searchable rows, although the redaction
  opt-in is described as discarding content by design (VISION.md:26). Uploading it is part of owner decision O3.

Layout in the bucket, under one prefix per user:

```
<prefix>/<copy id>/parts/<sha256>.<token>     database parts, 32 MiB each, named by content and by the attempt that uploaded them
<prefix>/<copy id>/deltas/<sha256>.<token>    changed-page objects, at most 32 MiB, named by content and attempt (3.7)
<prefix>/<copy id>/payloads/<sha256>.<token>  payload files, named the same way
<prefix>/<copy id>/gen/<number>.json          one manifest per generation, created once with a conditional PUT, never rewritten
```

Parts are small on purpose. No object needs a multipart upload. A part or payload that the head names is reused
under its existing key. Any other part or payload is uploaded under a key unique to the attempt
(`<sha256>.<token>`, with the publish token of 3.4), never onto an existing key, and a resumed attempt skips its
own keys. Generations share their base and every earlier delta (3.7).

### 3.2 Identity and the import stamp

- **Origin id:** `capture:owner.store_uuid`. It is the only id that survives a copy. Capture stores have no
  `metadata.store_uuid`: `ensure_store_uuid` has no caller outside tests (db_bootstrap.py:2573;
  tests/test_issue_1192_mcp_reader.py:1279), and store creation stamps only the owner key
  (claude_code_store.py:204-208).
- **Row identity across stores:** (copy id, `store_id`). A byte copy keeps `store_id`. Two copies of one copied
  home share an origin id but not their later rows, so the origin id alone cannot name a row.
- **Ledger identity:** (lineage key, generation, position) for lines and (lineage key, uid, content digest,
  position) for files, the ledger's own unique keys (claude_code_store.py:74-77). `event_id` is not used: it is
  not AUTOINCREMENT, so SQLite can reuse it (claude_code_store.py:69).
- **Cross-check key:** (lineage key, uid, content identity), the harness's own identity. Content identity is the
  sha256 of the row's role, its content, and its tool-call and tool-result fields, with every externalized
  placeholder expanded to its payload's text, read from that store's own payload folder. So two revisions of one
  entry that keep the text but change a tool call stay apart (lane 7b tests one). Two stores can hold one output differently: externalized under different file names
  (externalize.py:1547), or inline in one store and externalized in the other, when externalization is off, below
  its threshold or without storage there (externalize.py:1490-1503). Expansion gives all three the same identity. A placeholder whose payload file is missing or unreadable falls
  back to its own fingerprint, marked as a fallback: its character and byte counts (externalize.py:866-876) and
  the 12-hex sha256 prefix in the payload's name (externalize.py:111-112, :1547). The duplicate rule never drops
  a hit on a fallback match (4.2). Identities are computed once per generation, not per call. The key finds one occurrence held by two stores, and it is the key step 2's merge will
  use. Content alone never merges two rows.
- **The import stamp** is one row in the copy's `metadata`, key `import:origin`, under the cloud consumer's own
  prefix (docs/store-contract.md:64). It is written to the staging copy only, never to the live store. The same
  JSON is in the manifest. Fields: stamp format; origin id; copy id; origin harness; generation number; parent generation
  and its manifest digest; contract version and digest; `SCHEMA_VERSION`; projection versions present; unit
  version and core commit; highest `store_id`; time.
- **Not stamped:** the copy gets no `metadata.store_uuid`. That key belongs to core. Section 4.3 shows id
  stability does not need it.
- **The copy cannot be written by capture.** It has no `capture-owner.json`, so `admit()` refuses it as
  `store_unstamped` (claude_code_store.py:163-166).

### 3.3 Taking a snapshot

1. Take `cloud/upload.lock` (an OS file lock, released when its holder exits; a merger that cannot take it skips
   this cycle) and clear what a crashed run left in staging. Open the live store `mode=ro`. Copy it to
   `<home>/cloud/staging/` with the standard library's
   `Connection.backup`, a few pages per step. A step holds a read lock briefly. If capture writes between steps,
   SQLite starts the copy again **(assumption: SQLite's documented behaviour; section 6, row 1 measures it)**. So
   capture always wins and the snapshot waits for a quiet moment. Right after the database copy, the payload
   files the staging rows reference are listed, and each one whose file exists is cloned or hard-linked into
   staging (copied through an open file when the folders are on different devices; the free-disk check counts
   those bytes).
2. On the staging copy only: switch it to rollback-journal mode and write the import stamp. The live store flips
   between WAL and rollback mode (claude_code_store.py:250-265, :1169-1173), and the reader refuses a WAL header
   with no sidecar files (docs/claude-code-adapter.md:416-419). So the copy is normalised before it is hashed.
3. Read the staging copy's contracted tables and build the manifest. Hash the staged payload files; they are
   uploaded from staging. A reference that names no file is not uploaded. The manifest lists it under one of two
   counts: `unresolved_minted` when its name has the shape the product gives payload files (externalize.py:115-134),
   which is either a payload lost on this machine or a name quoted from another installation, and
   `unresolved_quoted` for any other name. The product's own integrity check reports the first kind as missing
   rather than go silent, for the same reason (ingest_integrity.py:400-412). Neither count stops a snapshot: the
   hosted copy never held such a file, so a refusal would protect no copy and would stop the backup of every other
   row. `status` names both counts by session, and the restore check and the copy checker report them (5.3). A payload the head names that is now missing, and
   whose session is not forgotten, is a loss, and the guard refuses it (3.4).
4. Run the publish guard (3.4). Record the attempt as pending, or resume the pending one (3.4, Chain), then
   upload the base parts or delta objects, and payloads, the head does not name, under the attempt's keys (3.1).
   Just before the manifest, check every named part, delta and payload exists. A delta attempt with a missing
   base part or delta becomes a base (3.7.3); a base attempt re-uploads missing parts under its attempt key,
   and missing payloads are uploaded again. Re-read live forgotten markers over `mode=ro` and `state.json`'s
   `--as-delete` decisions: drop and retry a copy holding rows forgotten since the copy, or a delta with a new
   item whose session the head's chain may hold, even if staging holds no forgotten rows. Refuse a copy holding
   ids covered by a pending decision with `as_delete_rows_present`; the final-check rules are in 3.7.3.
   Write the manifest last, as a create-if-absent (conditional) PUT of
   `gen/<N+1>.json`. Only when this attempt's own create wins and a head read then shows that manifest is the head
   does it run retention (3.5), still under the lock; a resumed attempt whose create is refused because the manifest
   already exists runs none.

A crash leaves orphan parts or delta objects and no manifest. An outage queues nothing: the next snapshot is simply newer. A batch
is never half in a snapshot, because rows, bindings, ledger rows and the cursor commit together
(docs/claude-code-adapter.md:280-282). Free disk is checked first: staging needs one store's size, plus the payloads it copies rather than links.

### 3.4 The publish guard

**The guard enforces the contract's "no stored row deleted or altered" (docs/store-contract.md:85) for messages,
bindings, relations, ledger lines and payloads between two generations, with one exemption: a session the user
forgot.** It compares generation N+1 with N before N+1's manifest is written.

- **Messages:** every `store_id` in N is in N+1 with the same content digest, or its session has a
  `capture:forgotten:<session>` marker (claude_code_store.py:52, :100-108). Compare the per-id `[id, hash]` pairs; a matching whole-session digest is a fast path for messages, bindings and relations only.
  Ledger entries are always compared individually; the digest does not cover them. An id of N that is missing is a loss (below); an id of N that holds a
  different row is `store_id_reused` (next). The digest covers every column of the row. No step-1 writer changes a guarded column of a capture-store row in place (the ledger's notes and the binding check columns are outside the digests); a later writer that does needs its own guard rule first, or the next publish refuses `store_id_reused`.
- **Bindings and relations:** every `host_uid_bindings` row and every `message_relations` row in N is in N+1
  unchanged, or its message's session is forgotten; a relation names two messages and is exempt when either one's session is forgotten, and an `--as-delete` decision covers it when either one's session is the decision's (store.py:399 deletes a relation with either end's session). For a binding, unchanged means its identity columns
  (`store_id`, `uid`, `lineage_key`, `kind`): the check columns `first_check` and `disagree_seen` are updated in
  place (store.py:1877-1885). Both are contracted tables (docs/store-contract.md:14-28). The
  delete helpers remove them only with their rows (store.py:393-434), but nothing else stops one from vanishing
  alone, so the guard compares them. They are in the same per-session digests.
- **Ids are never reused under one copy id.** `messages.store_id` is `INTEGER PRIMARY KEY AUTOINCREMENT`
  (store.py:870-871), so one database file never gives a new row an id it gave before, even after its top rows
  are deleted: forgetting the newest session reuses nothing. Reuse needs an older file. A store rolled back to a
  backup or an earlier copy has an older `sqlite_sequence`, its new rows can get ids that a published generation
  gave to other rows, and an id issued to an agent would then resolve to a different row. Each manifest records
  the highest `store_id` ever published under its copy id (the mark) and carries forward every removed session
  (forgotten, or accepted lost) with its ids, as a compressed id set (lanes 2 and 3 measure its size and the guard's
  time on a long, sparse history). The publish is refused with `store_id_reused` when the
  store's `sqlite_sequence` for `messages` is below the mark, which catches a rollback before any id is reused,
  or when an id at or below the mark holds a row other than the one published under it: for an id of N, a row
  whose session or content digest differs; for a removed id, any row. A store that holds a forgotten session
  again, as a rolled-back one can, is refused with `forgotten_session_returned`, and the user forgets it again
  through 5.2's path. Rows that are only missing go through the loss and forget rules. The exit from
  `store_id_reused` is `enable --new-copy`, never `accept-loss`. The old copy id keeps its ids and
  becomes this machine's superseded copy (3.5); on this path only, the new copy id starts with the old one's forgotten
  sessions, which are this machine's own history; a `--new-copy` after a chain conflict starts with none.
- **Ledger:** every ledger key in N is in N+1, or its session is forgotten. Like the message check, it
  compares N+1 with N. For an entry with bytes, the sha256 of its decompressed raw line (resolving `raw_from`)
  equals the value N's manifest recorded, and also its `line_digest` when it did in N; an accepted cut keeps N's
  values and is not judged again. A `held` file entry has
  no bytes: its text is held whole by the rows it names (claude_code_store.py:765-782). For it, the digest is
  unchanged and every row in its `store_ids` that N held is present with an unchanged content digest; a row N
  already lacked was judged when it went. A key that N lacks is judged against itself: an entry with bytes needs its
  raw sha256 to equal its `line_digest`, and a held entry needs a non-empty `store_ids` that names only rows present in its session. An entry with neither bytes nor such a list fails.
  A cut raw line with an unchanged digest column fails both for a key N held intact and for a new key.
  Damage before the first generation is not caught, because no guard runs without a head.
- **Payloads:** every payload name in N is in N+1 with the same decoded content. An absent name is a `guard_loss` if N+1 still references it in `unresolved_minted` or `unresolved_quoted`; without a reference it is exempt here because the message rule judges its rows. Core
  can replace a payload file in place (externalize.py:264-268); whether capture stores ever reach that path is
  not established, so a content change refuses.
- **Chain:** each manifest names its publisher: a digest of the copy id and the machine's hardware id (the macOS
  platform uuid, Linux's `/etc/machine-id`), which a copied home does not carry **(assumption: both are readable
  without privileges and unique per machine; a cloned VM image that keeps its `/etc/machine-id` breaks the
  second; a later lane that adds `--resume` measures it; in step 1 every chain conflict
  ends in `--new-copy`)**. Each publish attempt records (N+1, a random 128-bit token) as pending in `cloud/state.json`
  before it uploads anything, and the manifest carries the token. The token comes from the OS random source mixed
  with `time_ns` and the pid. A publish goes ahead only when the bucket's newest generation names this publisher
  and its (generation, token) pair equals this machine's recorded last or pending pair for that copy; a null or different publisher is `chain_conflict`; and the
  manifest is created only if `gen/<N+1>.json` does not exist. Otherwise the publish stops with `chain_conflict`
  and overwrites nothing. With no generation in the bucket, a publish goes ahead only when this machine records no last pair for that copy and no pending pair above generation 1; otherwise it stops with `chain_conflict` (the hosted copy was deleted in the console or lost), whose exit is `--new-copy`. A pending pair is replaced only by the next attempt, never cleared. An interrupted
  attempt resumes with its pending token. The pending record keeps the bucket's `Date` from the attempt's first
  response; an attempt that resumes more than 3.5 days (G/2) after that, by the bucket's clock, is abandoned and a
  new attempt starts. Two machines taking turns are caught as well as two racing ones, so a
  second machine never meets a loss refusal for the first one's rows, except a home copied during a publish (next
  item).
- **A copied home on the same machine, or a cloned VM with the same hardware id,** is indistinguishable from the
  original until their attempts diverge. A copy taken during a publish holds the same pending pair, so it can take
  the generation its twin created as its own. If the twin re-snapshotted after the copy, that generation holds the
  twin's later rows, and the copy's next publish meets `store_id_reused` or a loss refusal instead of
  `chain_conflict`; `status` never calls the twin's copy safe to delete (idle copies, below). Nothing in step 1
  deletes except retention after its own winning create (below); every machine but one then meets
  `chain_conflict`, `store_id_reused` or a loss refusal. No kept manifest names a missing key,
  because no attempt writes onto an existing key (3.1) and the object check re-uploads available content or
  cuts a base when a delta attempt lacks a base part or delta (3.7.3), with one exception that step 1 states and
  does not fence. It needs two stalls in such a pair of copies:
  the other copy's retention, after its winning create of N+1, stalls for 7 days or more between its head read and
  its deletes, and this copy's attempt for N+2, more than 7 days old, passes its part check before that delete and
  creates N+2 after it. N+2 is then a head that names a key deleted as never named. No local row is lost: a restore
  of that generation fails visibly on the missing object (5.3), and the generations before it stay kept unless a
  forget pruned them. If this copy publishes again, a missing base part or delta makes it cut a base; other
  available content is uploaded again (3.7.4), and the newest generation is whole; otherwise the other copy meets
  `chain_conflict` and leaves by `--new-copy`, and
  the hosted copy stays idle with a newest generation whose restore and refresh fail visibly (lane 3 tests this
  interleaving). An
  attempt paused for a long time can also create a manifest number that retention deleted. That manifest is not the
  head, so no replica or guard reads it and it runs no retention; `restore --generation` can still name it, and then
  fails visibly on any missing object (5.3).
- **One invariant: nothing writes under a copy id unless its head names this machine's publisher and carries this
  machine's recorded last or pending (generation, token) pair for that copy, and nothing deletes under it except
  retention, which runs only when this attempt's own conditional create wins and a head read then shows that
  manifest is the head.** No command deletes by an attempt's token: a twin holding the same token may have published those
  keys. Every command that writes to the bucket reads the head first and refuses otherwise: a publish, retention
  and `sync`. `status` says why.
- **A chain conflict never offers `accept-loss`.** Its one exit in step 1:
  - `enable --new-copy`, in every case: a lost `cloud/state.json` (taking the old copy back is a later lane's
    `--resume`, Appendix I), a changed hardware id, a replaced machine, or a home copied with its
    `cloud/` folder to the same path on a second machine, which passes `admit()` because it compares only the
    home path (claude_code_store.py:168-169). It mints a new copy id and seeds a fresh hosted copy from this
    machine's store, with D12's restore check. This machine never writes to the old copy again. The old copy
    keeps its publisher if another machine still publishes to it, and every machine reads it as a replica;
    the sessions they share before the copy are handled by the duplicate rule (4.2).
- **An idle copy** is one that no machine publishes to any more, after a chain conflict, a changed hardware id or
  a replaced machine. Nothing prunes it, so a later forget does not reach it. `status` lists every copy under the
  prefix with its head's publisher, number and time by the bucket's clock, and says that a whole copy is deleted in
  the bucket host's own console; it never names a copy as safe to delete. Step 1 has no command that deletes a copy
  (Appendix I).
- **On refusal for a loss (`guard_loss`):** N stays current. Nothing in the bucket changes. `status` names what disappeared, by
  session and count. The user then chooses:
  - `accept-loss` records the acceptance in `cloud/state.json`, not in the store, and the next publish goes ahead;
    every later manifest carries the pin, so a lost `state.json` does not lose it.
    It **pins the generation before the loss**, dated only by the bucket clock at the winning publish: retention keeps it, and the parts it names, for 30 days or until
    `accept-loss --release`, which only records the release: the next manifest omits the pin, and that publish's
    retention prunes. So a loss caused by a bug is not mirrored into the hosted copy's only intact
    generation. `status` lists pinned generations.
  - `accept-loss --as-delete` is for a loss the user meant, such as a Hermes `/lcm doctor clean apply` pointed at
    the store. It is treated like a forgotten session: no pin, and retention prunes as in 3.5.

The guard makes "uploads never mirror a local deletion that was not deliberate" mechanical. It also covers the
old payload-GC worry: the rewrite seam has no production caller on main (store.py:5-8, :1277), and a rewritten
row would fail the content compare anyway. Draft 1's "GC and cloud are mutually exclusive" rule is dropped.

### 3.5 Retention

- Every writer of `cloud/state.json`, including decision recording, release, withdrawal and `--new-copy`, takes `cloud/upload.lock`. Retention runs only under `cloud/upload.lock`, and only when this attempt's own conditional create wins and a head read then shows that manifest is the head (3.4). Every command that writes to the bucket (`enable` with `--new-copy`, `sync`) takes the same lock; every network call has a timeout, and `status` shows the lock's age, so a hung holder is visible.
- Keep the current generation, the two before it, and any generation pinned by `accept-loss` (3.4). Every manifest
  carries the head's pins forward. Retention deletes keys named only by manifests being pruned before those
  manifests, and deletes a manifest only after its exclusive keys returned success or not found; otherwise the
  next winning publish retries. This covers `parts/`, `deltas/` and `payloads/` (3.7.4). A never-named key is
  deleted only when the bucket's clock (a response's `Date`, never the machine's clock) shows it older than G = 7 days.
- Every publish deletes each older kept generation that is not pinned and whose manifest still names a session
  that the current store marks `capture:forgotten:`, or an identity accepted with `--as-delete`; it also prunes
  by the cumulative `forgotten` list's base-generation cutoff (3.7.4). **A pinned
  generation is never pruned by a forget:** it would take with it the rows a bug lost. `status` reports
  `pin_holds_forgotten` with the session count until the user releases the pin (`accept-loss --release`), and the
  opt-in says that a forgotten session stays in a pinned generation until then. `status` also warns 7 days and
  1 day before a pin expires. Because the check runs on
  every publish, a crash or a failed delete is retried by the next winning publish. A delete that keeps failing, for example under a credential without delete permission, is retried on every publish, so nothing is pruned and the forgotten rows stay until it succeeds; `status` reports repeated retention failures, and `enable` checks delete permission (#2801). Once a publish has finished its retention
  step, no object in the bucket holds the forgotten rows outside a pinned generation, a superseded copy or an idle copy (3.4), which
  `status` reports (`pin_holds_forgotten`, `superseded_holds_forgotten`), except parts and delta objects of failed
  or interrupted attempts or dropped staging copies that no manifest named: they can hold a forgotten session
  for at least 7 days, until the next successful retention after that. Free-page residue has 5.2's separate
  limit. The opt-in says both.
- **Superseded copies.** When `enable --new-copy` leaves a copy whose head names this machine (after
  `store_id_reused`), `cloud/cloud.json` and the new copy's manifests list it as superseded. It is an idle copy
  (3.4): nothing prunes it, and no command of step 1 deletes it. A forget does not physically reach it; `status`
  reports `superseded_holds_forgotten` with the session count, the opt-in says so (5.1), and those sessions are
  absent from the copy's view (V3) on every machine that has read the forgetting copy's manifest (4.2). Every
  machine, this one included,
  reads a superseded copy as a replica, so rows a rollback lost stay recallable and ids issued from it keep naming
  their rows; the duplicate rule applies as to any replica (4.2). Each superseded or idle copy counts as one more
  store in 4.4's bars. `status` lists each one with what it holds that no current copy holds; the user deletes a
  whole copy in the bucket host's own console (5.1).
- Where the host keeps object versions, deleted objects must expire. The opt-in states the expiry the user's
  bucket is set to, or that we could not read it.
- The two older generations are the hosted copy's own backup. The restore proof restores one of them (5.3).
- After a delete has pruned them, only one generation exists until two more publishes. In that window the hosted
  copy has no older generation to fall back to. `status` says so.

### 3.6 Versioning

**Three version numbers travel in every manifest: the bucket layout format, the store contract version with its
digest, and `SCHEMA_VERSION`.**

- The cloud module holds the contract version and digest as constants. A new compat test fails when they differ
  from `docs/store-contract.md`. So a contract change cannot merge without someone reading the cloud consumer,
  on top of the existing gate (docs/store-contract.md:87).
- **A machine that meets a newer contract version in a manifest** does not download that generation. It keeps
  serving its last readable replica, reports `replica_newer_contract`, and asks for an update. Nothing is lost:
  the hosted copy is untouched.
- **A machine that meets an older version** serves it while its build still reads that version. Below that, the
  replica is not served until its origin machine publishes again.
- **The guard across a version change** compares only presence and content of rows and ledger lines, by column
  name. A contract change that renames those columns must change the guard in the same PR; the compat test
  forces it.
- **An old generation** is restored by the unit version its manifest names.
- The reader already refuses a store whose schema stamp is above its build (readonly_store.py:198-200). That
  stays the last line.

### 3.7 Base-plus-delta upload

**The unit of change is the SQLite page. A base holds the whole file in 3.1's 32 MiB parts; a delta holds the
pages that differ from the generation before it.** Row 2's real-store replay found that busy publishes changed
pages across nearly every part, so whole-part uploads exceeded the daily bar. This is the decided upload design;
row 2D below measures its real snapshot path before the bars are claimed.

#### 3.7.1 The delta object

- Read page size P from bytes 16-17 of the staging file's header, unsigned big-endian, with 1 meaning 65,536.
  It must equal `PRAGMA page_size` on the staging connection, and file length must be a whole number of pages;
  otherwise refuse the snapshot. The live store is not read for this.
- A delta holds pages only. The header is page 0; the import stamp travels inside pages and remains
  in the manifest. File length, page size and digests are in the `database` block (3.7.3).
- Object bytes, from which lane 4 writes its own decoder:
  - bytes 0-7: `LCMXPD1\n`; bytes 8-11: P; bytes 12-15: record count n; both integers unsigned big-endian;
  - bytes 16 to the end: one zlib stream, written with the standard library's `zlib` at level 6. It decompresses
    to exactly n records: a 4-byte unsigned big-endian page index i, then P bytes replacing file bytes i·P to
    (i+1)·P. Indexes rise strictly within an object and across the objects of one delta entry;
  - at most ⌊(2^25 - 2^16) / (4 + P)⌋ records per object (8,168 at 4 KiB), allowing for zlib's growth on
    incompressible pages. `MAX_OBJECT_BYTES` is 2^25 (32 MiB); a larger delta uses several objects. In format 1 a reader does
    not refuse on these two caps (Appendix K, F8 and F9). #2800 proposes that readers refuse, before fetching, a
    part or object reference whose `bytes` exceed 2^25, or an object reference whose `records` exceed the limit
    above (`bad_database_block`); Appendix K changes with it if it lands;
  - key: `<prefix>/<copy id>/deltas/<sha256 of the object>.<token>`, with 3.1's content-and-attempt naming.
- Readers refuse bad object sha256 (`object_sha256`), byte length (`object_bytes`), magic (`delta_magic`), page
  size (`delta_page_size`), manifest/header count (`delta_count`), decompressed length or stream (`delta_stream`),
  trailing bytes (`delta_trailing`), and non-increasing indexes (`delta_order`). A missing object is
  `object_missing`. They also refuse an index at or past the entry's `file_pages` (`delta_index`), a growing
  entry that leaves any appended page unwritten (`delta_gap`), and generations not exactly
  `base.generation + 1` through the manifest's generation (`chain_generation`); malformed `database` fields are
  `bad_database_block`. A part or object key that fails 3.1's key grammar is `bad_name`. When a replica refresh
  patches the generation it holds (4.1), a held generation outside the chain, or a held file whose length is not that
  generation's, is `start_mismatch`; a full rebuild whose output file already exists is `output_exists`. When one
  malformed input trips two checks, an implementation may report either code
  (a growing entry that skips a page then holds a corrupt object: `delta_gap` or `delta_stream`; a missing or
  non-integer manifest `generation`: `chain_generation` or `bad_database_block`), and every consumer treats any
  refusal the same way.
- This is row 2's measured encoding, index plus page in one level-6 zlib stream, with a 16-byte object header.
  It needs only the standard library (D14); the raw ledger already uses `zlib` (claude_code_store.py:20, :736).
- No stable page layout is assumed. Shrink, reshuffle and any other change are changed pages; rebuild sets the
  file length after each entry (3.7.3).

#### 3.7.2 Page lists, self-check and the unchanged-store skip

After 3.3 step 2's journal switch and stamp, one sequential pass over the staging file computes each page's
sha256, the whole-file sha256, and `page_list_sha256`: the sha256 of the raw 32-byte page digests concatenated
in page order from 0 through `file_pages - 1`, with no separator. Lanes 2d and 4 share one test vector for it.
After that pass the staging copy is opened read-only only. No step takes `capture.lock` or hashes the live store.

- The page list stays local, in `<home>/cloud/pages/<copy id>.last` and `.pending`, mirroring `state.json`'s
  last and pending pairs. Write `.pending` by replace before uploading; promote it to `.last` when the attempt's
  manifest is recorded as head. Read the head first and promote a `.pending` that matches it before writing a
  new one. A delta requires a local list whose digest equals the head's `page_list_sha256`; otherwise cut a base.
  Lost or damaged lists, a crash after manifest create, and a head adopted from a twin's different staging copy
  therefore never cause a delta against the wrong predecessor. Lost `state.json` still requires `--new-copy`.
- Compare the head's page digests with the new list. Every index at or past the head's page count is changed,
  including an appended all-zero page. The base page count is `base.file_bytes / page_size`; that is also the
  previous count for its first delta. Encoding checks each emitted page's sha256 against the new list and checks
  that the emitted indexes equal exactly the changed-page set; a mismatch refuses the publish (`delta_selfcheck`).
- Before the backup, the timer compares the size and mtime of `lcm.db` and `lcm.db-wal` with those recorded when
  the last published snapshot began. It skips only if both match, no decision (`accept-loss`, `--as-delete`,
  `--release`) is unpublished, the pending pair equals the last winning pair, and `retention_done_generation`
  in `state.json` equals the head's generation. Lane 2's skip check reads both the pending pair and `retention_done_generation`.
  Manual `sync` never skips. `SKIP_CEILING_S` is provisionally
  86,400: at most 24 hours from the last successful publish. An unchanged store still publishes a stamp-only
  delta then, to expire pins and sweep orphans. The stamp prevents using page hashes as the skip test.
  Capture's return to rollback mode moves the signature **(assumption)**; a wrong unchanged result delays a
  change by up to 24 hours and loses nothing. A dropped staging copy restarts at the hash pass; base rules 2 to 6
  and `.pending` are evaluated on the copy that will publish.
- Local cost per publish that is not skipped:

  | | 1 GB store | 3 GB store |
  |---|---|---|
  | Staging copy written, at least (3.3's existing cost) | 1 GB | 3 GB |
  | Hash pass read | 1 GB | 3 GB |
  | Page lists: memory for two lists, bytes written | 17 MB, 8.4 MB | 50 MB, 25 MB |
  | Disk held between publishes (two lists) | 17 MB | 50 MB |
  | Staging writes per day at 16 / 48 publishes, at least | 16 GB / 48 GB | 48 GB / 144 GB |

  The hash pass's synthetic probe was about 1.4 s per GiB on the shared development machine; no timing there
  is judged. Row 2D reports local bytes written per day and bench timing. Its results trigger revisiting a
  copy-on-write snapshot under a read lock as follow-up; step 1 keeps D6's consistent backup copy.

#### 3.7.3 Manifest, rebuild and base rules

**Every manifest names all base parts and deltas needed to rebuild its generation, in order. No reader follows
a pointer to another manifest.** The manifest's `database` block is:

| Field | Meaning |
|---|---|
| `page_size`, `file_bytes`, `file_sha256` | This generation's staged file. |
| `page_list_sha256` | Digest of its raw page-digest list (3.7.2). |
| `base.generation`, `base.file_bytes`, `base.file_sha256` | The base generation and its whole file. |
| `base.parts[]`: `key`, `bytes` | Base parts, in file order. |
| `deltas[]`: `generation`, `file_pages`, `objects[]` (`key`, `bytes`, `records`) | One entry per generation after the base, oldest first; objects in ascending page-index order. |

Rebuild writes base parts in order and checks `base.file_bytes` and `base.file_sha256`
(`base_digest_mismatch` on failure). Apply each delta entry in order, enforcing 3.7.1's checks, then set length to
`file_pages × page_size`. Finally check the rebuilt file's `file_bytes`, `file_sha256` and `page_list_sha256`;
a mismatch is `rebuild_digest_mismatch`, with the generation. A base has empty `deltas` and equal base and file fields.
Each object's sha256 is in its key; there are no per-page or per-range digests in a manifest. A page list of
8 to 25 MB is about four busy publishes' worth of deltas.

Every manifest also carries a cumulative `forgotten` list that never shrinks. Each entry has a key (the session
for a forget, the decision's own id for an `--as-delete`), its kind (`forgotten` or `as-delete`), its session and
`base_generation` (or `null`). A new item is a live forgotten marker or `--as-delete` decision whose key is absent
from the head's list; a second decision on the same session is a new item. The head's chain may hold a session
when its per-session digests name it or its cumulative removed list carries it, including a plain accepted loss.

Cut a base when any rule holds, recording its reason in the manifest and showing it in `status`:

1. No generation exists for the copy id (`enable`, `--new-copy`; `first`).
2. Page size changed (`page_size`), or no local list matches the head's `page_list_sha256` (`page_list`).
3. At staging, a new item's session is one the head's chain may hold (`forget`). A session never published
   triggers nothing. A forget after plain `accept-loss`, a forget after `--as-delete` followed by new published
   rows, and a second `--as-delete` of new published rows all trigger a base.
4. On a delta attempt, the check before the manifest finds a missing base part or delta (`missing_object`). List
   `parts/` and `deltas/` once each, rather than one request per key. An old base part cannot be reconstructed
   from the current staging file: the delta becomes a base. A base attempt re-uploads a missing part from staging
   under its attempt key, as in 3.3.
5. Stored delta-object bytes since the base, including this delta and excluding base parts, reach
   `SIZE_FACTOR × file_bytes`, provisionally 2 times the current file (`size`).
6. The chain would pass `MAX_CHAIN_GENERATIONS`, provisionally 1,024 generations including the base (`length`).

Just before the manifest, re-read the live forgotten markers and `state.json`'s `--as-delete` decisions. Drop the
staging copy and restart at the hash pass only if it holds rows of a session forgotten since the copy, or if the
attempt is a delta and a new item's session is one the head's chain may hold. The latter restarts even a delta
whose staging copy holds no forgotten rows; it never restarts a base. Re-evaluate the rules on the new copy.
An `--as-delete` decision covers rows already missing and removes nothing itself. If a staging copy holds ids
covered by a pending decision after a rollback, refuse with `as_delete_rows_present` until the user withdraws
the decision or the rows go again; checked at staging, before the guard's codes, and again at the final check; never restart in a loop. Record every new item read at the final check:
`base_generation` is this generation only when it is a base and the item's session is one the head's chain may
hold; otherwise it is `null`. Inherited entries on `--new-copy` take that copy's first base.

There is no age cap. The size and length constants and skip ceiling are provisional pending row 2D; readers
rebuild any chain a manifest names, so changing them needs no format change. `status` shows chain length, stored
bytes and base reason. The existing bucket layout format covers this block; no hosted copy exists in the earlier
layout, so no migration is needed.

Storage usually peaks near 4 times the database after a size base. That is not a bound: large deltas immediately
after a size base can keep two chains at once, under 6 times. Pins, 7-day orphans, manifests and payloads come on
top, as do idle and superseded copies (section 9).

#### 3.7.4 Consumers, retention and forget

- **Publish guard (3.4).** Row semantics are unchanged. It reads the staging copy's contracted rows,
  `sqlite_sequence` and the head's digests, mark and removed-id sets, never parts or deltas. Transport digests
  do not judge loss.
- **Retention (3.5).** List and delete `deltas/` as well as `parts/` and `payloads/`. A manifest names its entire
  base and chain, so a kept manifest protects all of them. Delete keys named only by manifests being pruned
  before deleting those manifests. Delete a manifest only after every key it alone names returned success or
  not found; otherwise leave it for the next winning publish. A crash between keys and manifests leaves unkept
  manifests with missing keys; the next retention deletes them. Write `retention_done_generation` only when
  every delete returned success or not found. Never-named objects retain the 7-day rule.
- **Forget.** Besides 3.5's session rule, every publish deletes each unpinned generation whose `base.generation`
  is older than the largest non-null `base_generation` in the head's cumulative `forgotten` list. Reading the
  cutoff from the head makes a kill before retention retryable. A base part reused under its old key is a slice
  of the current file and holds nothing the local file does not. The promise holds after the forget publish
  completes retention, subject to pins, idle and superseded copies, and 5.2's limits: objects from failed or
  interrupted attempts or dropped staging copies that no manifest named wait for the 7-day rule. Free-page
  residue travels in base parts and delta objects; it stays while the local file keeps it, and a later base can
  copy it again. LCM-X does not set `secure_delete`; SQLite reuses free pages over time. Older objects keep
  residue until their generations go. Step 1 adds no scrubbing.
- **The missing-key exception (3.4).** If the stalled-retention interleaving deletes a base part or delta, this
  copy's next publish cuts a base instead of re-uploading that old content. Content still available to upload is
  uploaded again. The newest generation is whole either way; lane 3 tests the interleaving with delta keys.
- **Restore (5.3).** Verify each downloaded object and rebuild as in 3.7.3 before checks 1 to 4. A rebuild digest
  mismatch stops restore and reports `rebuild_digest_mismatch` with the generation. The proof publishes base,
  delta, delta, forced base; restore the newest, delete it as simulated loss and restore the preceding base with
  two deltas, then delete the first delta's object and show that the same restore fails and names it.
- **Replica refresh (4.1).** If the held generation has the same base parts and its delta list is a prefix of
  the head's, clone its `lcm.db` into the new generation folder where supported (`cp -c` on APFS, a reflink on
  Linux), otherwise copy it in full. Hold the source folder's shared lock during clone or copy. Apply only the
  remaining deltas and check the rebuilt digests. Only those deltas need database download; payload handling
  stays as in 4.1. If the prefix condition fails or patching fails, download the base and chain and rebuild,
  checking `base.file_sha256` and `file_sha256`. On a mismatch the previous generation keeps serving and
  `status` names the generation. The old folder is never modified; 4.1's handover and peak disk stay. Crossing
  a re-base downloads the new base once.
- **Independent checker (lane 4).** Import no uploader module. Decode and rebuild independently, check
  `file_sha256` and `page_list_sha256`, compare byte for byte with restore's file, and run row checks on the
  checker's own file. Add three faults to the existing five: a delta omitted, two deltas swapped, and one
  record's page replaced by that page's earlier bytes with the object's key recomputed. Only the result digest
  catches the last one.
- **Upload retry (section 6, row 3).** Re-send no complete unchanged base part and at most one generation's
  delta objects. A resumed attempt takes a new snapshot, so its delta may differ from the interrupted one.

#### 3.7.5 Daily bar and row 2D

The daily bar is the largest of 20% of the database at the day's start, 3 times that day's growth, and 32 MiB.
The floor matters below 160 MiB (168 MB). Count every uploaded byte: manifests, delta objects, base parts and
payloads. Each day without a base must meet its bar. Bases are cut only by 3.7.3's rules, with the reason shown;
any 30 days with no forced base must total at most their 30 daily bars. Forced-base and forget cells report
their day against its bar plus one database.

Row 2's 1,390,985,216-byte store had a 278 MB 20% term. These are delta bytes only, scaled from seven bins:

| Day | Scaling | Delta bytes | Share of the 20% term |
|---|---|---:|---:|
| 16 publishes, replay mix | 7-bin sum × 16/7 | 32.5 MB | 12% |
| 16 publishes, all like the largest bin | 3,376,727 × 16 | 54 MB | 19% |
| 48 publishes, replay mix | 7-bin sum × 48/7 | 98 MB | 35% |
| 48 publishes, all like the largest bin | 3,376,727 × 48 | 162 MB | 58% |

The day's delta/growth ratio over the seven bins was 14,232,909 / 9,867,264 = 1.44 (1.25 to 1.62 for five busy
bins; 2.97 uncompressed). The two trickle bins were 2.9 and 3.9. Churn followed activity rather than file size
in the two measured stores. A user several times busier is unmeasured; proportional delta growth is an
**(assumption)**. Evidence is one replay, two sessions, 3.5 hours, embeddings off, hashing the live file after
worker runs rather than staging, and no full-text merge spike. It proves no delta restore time, heavy-user day,
embeddings-on default or real manifest size.

A prototype manifest for a separate payload-heavy 2.75 GB store was 1,263,640 bytes (82 parts, 5,804 payloads),
not the final format. That proxy adds 20 MB at 16 publishes and 61 MB at 48. At 48 the replay mix becomes about
159 MB (57% of the 20% term), passing line (5); the all-largest-bin day becomes 223 MB, 66% of its governing
3-times-growth term.
Every share must name the term that governs it. Manifests stay plain JSON; row 2D measures the actual writer.

Size bases cost a whole database but count only delta bytes toward the 2-times threshold. Growth alone does not
trip it while delta/growth stays below 2. With churn and no growth a size base averages 1.5 times delta bytes;
staying below two thirds of the bar leaves room for that base. Length bases add about 1.6% of a database per day
at 16 publishes and 4.7% at 48 (8% and 23% of the 20% term). When growth governs, their contributions are at most
0.23 and 0.70 times growth. These arithmetic estimates do not replace the simulation gate.

**Row 2D runs once lane 2's real snapshot path exists, after lane 2 merges, and moves no date unless it fails.**

- Size cells run on the shared development machine: seed a scratch store from real transcripts by row 2's
  method at a cut T0 at least 24 hours back, then replay the following 24 hours in 30-minute bins (48 publishes)
  through lane 2's real backup, journal switch, stamp, page-list and delta path, embeddings off and on. Real
  transcripts stay there. Timing cells (hash and encode seconds per GB, rebuild) run on the bench hosts using
  row 1's recorded replay; those need no 24 hours. Include one forced base and one forget.
- Report per publish: changed pages, stored delta bytes, growth, manifest bytes, hash and encode seconds per GB.
  Per day: total uploaded bytes, largest publish, ratio and local bytes written. Measure every replay manifest;
  beyond 48 deltas run lane 2's manifest writer on synthetic chains of 128, 256, 512 and 1,024 deltas repeating
  replay entries, interpolating linearly. Scale the store-growing part (per-session digests, payload names and
  id sets) with file bytes.
- Repeat the replayed day for 90 days from a fresh base at its starting size. Advance file bytes by measured
  per-publish growth. Publish deltas unless size or length fires; a base costs that publish's file bytes and
  manifest. Every simulated publish adds its measured new-payload bytes, including bases. Compute each daily
  bar from the file at that day's start, that day's growth and the floor. No forced base is simulated.
- Pass lines: (1) every day with no base is within its bar; (2) every 30-day window starting on days 0 through 60
  totals at most its summed daily bars; (3) replay-day delta bytes divided by whole-day growth is at most 2.0;
  (4) every generation rebuilds to its `file_sha256`, and snapshot, guard, hash and encode together fit row 3's
  60 s per GB; (5) the replayed day's whole delta bytes plus manifests are at most two thirds of 20% of the file.
  Do not subtract raw growth from compressed delta bytes. The continuation with growth zero after day 1,
  keeping each day's whole delta bytes and no new payloads, must also pass (1) and (2).
- Report the worst window's share with and without manifests. If that window passes two thirds of its summed
  bars, manifest compression becomes a follow-up lane decided on those numbers. If the ratio fails, raise the
  zlib level once, then report; the size constant must stay above the ratio. If (5) fails, return the size
  constant to the release manager. The provisional size, length and skip constants can be revisited on row 2D.

#### 3.7.6 Lane assignments

Lane 2d holds `cloud_delta.py` and `tests/test_cloud_delta.py`: page lists and digests, last/pending list files,
encode and apply, the pure base rule and skip signature. Its surface includes the object format, page lists,
changed pages, rebuild, emitted-page self-check, record cap, skip completion checks and appended-zero-page test.
It takes the forget trigger computed by lane 3 and the missing-object input checked by lane 2 as given. Its
estimated ~220 non-generated lines, tests apart, exceed the 200-line signal; it is a separate High lane so lane 2
does not grow past 600 lines. It needs no transport, manifest writer or snapshot. Codex (`gpt-6.1-sol`, high)
authors it with one fresh Claude reviewer.

Lanes 2 and 4 receive the delta-object and `database` format blocks from one source. Lane 2 adds about 80 lines
(to ~480): manifest block, base-or-delta publish and pre-manifest chain check. Lane 3 adds about 30: `deltas/`
retention, forget input, and interleaving tests with delta keys. Lane 4 adds about 60 (to ~310): independent
decoder, rebuild and three faults. Lane 5 adds about 80 (to ~380): restore rebuild and local patch with full
fallback. Lane 8 adds about 15: chain length, bytes and last base reason in `status`. All are High; these are
estimates **(assumption)**. Dependencies and dates are in section 8; lane 7a is unaffected.

#### 3.7.7 What stays

D4 and D6 stand: no writes to the live store, its payload folder or spool, no `capture.lock`, and a staging
backup, journal switch and stamp before hashing. Payload upload, 32 MiB base parts, attempt keys, the publisher,
pending pair and conditional create, row guard semantics, `accept-loss`, pins, contract constants and D14 stay.
The retention window and 7-day orphan rule stay, with the key-before-manifest order and cumulative forget cutoff
above. The merger's routing, recall semantics and handover stay; only refresh's database rebuild changes.
A single-machine user's chain is not re-read between bases, so damage may first appear at restore, as with whole
parts. `sync --verify` is follow-up work, outside this lane.

## 4. Recall across stores

**Recall stays local. A merger runs one stock reader per store in parallel, returns the local answer unchanged,
and adds hits from the user's other machines in a separate list.**

### 4.1 Topology

- `enable` writes `reader-override.json` naming the merger, with the key `"owner": "lcm_x_cloud"`. Before any
  bucket write it refuses when a file without that key exists, and names the path. The plugin starts the merger
  and passes the plain reader's argv in `LCMX_LOCAL_READER_CMD` (docs/claude-code-adapter.md:381-383).
- The merger starts that plain reader as a child, and one more stock reader per replica home. Each replica home is
  `<home>/cloud/replicas/<copy id>/gen-<n>/` with `lcm.db` and `lcm-large-outputs/`, the layout the reader expects.
  The home a reader opens is the view that `view.json` names (4.7).
- **A refresh never changes files under a running reader.** The stock reader opens several connections in turn and
  resolves payloads separately (mcp_reader.py:360), so a home swapped mid-call could mix generations. Each
  generation rebuilds into its own directory, which is never modified once served, with every downloaded
  object's sha256 verified. Rebuild uses the base and chain, or patches a clone/copy of the held generation
  under its shared folder lock, with full download as fallback and rebuilt-file digest checks (3.7.4). A missing or bad object stops the refresh: the previous generation keeps serving, and `status` names the
  object (lane 5 tests it on a generation with a deleted part). Otherwise the refresh writes
  `replicas/<copy id>/current` (the generation number, replaced atomically). `current` names the byte copy's
  generation: refresh writes it, and builds and the merger's V9 check read it. Every merger in the home reads
  `view.json` before each call, so all of them learn of a new generation. A merger starts a new stock reader on it and
  switches new calls there only after that reader has answered one probe call. Each reader holds a shared OS lock
  on its generation's folder; a folder is deleted only by a process that takes the lock exclusively without
  waiting, so a folder another session still reads stays. An old reader is stopped once its in-flight calls end,
  or when the tool budget runs out, so no reader holds a folder longer than one call. Row 5 runs with two mergers.
  Peak disk: V11 (4.7). A refresh starts only when free disk exceeds twice the replica plus 5 GB; otherwise it
  waits and `status` says why.
- A machine never reads the hosted copy of its own current copy id. It reads its superseded copies like any other
  replica (3.5).
- The override owns recall once started (docs/claude-code-adapter.md:413-415). So the merger must fall back to
  the local child alone whenever a replica reader fails, is warming, or is late.

### 4.2 The merge

- The merged response is the local reader's JSON with one added top-level key, `other_stores`. Removing that key
  restores the local bytes exactly. The reader's own envelope already works this way (mcp_reader.py:431-452).
- `other_stores` has one entry per replica: copy id, origin id, generation, age, recall mode (vector or full text), status,
  and its own hit list with its own limit. No cross-store score is invented.
- A replica uses vectors only when its stored embedding identity equals this machine's (mcp_reader.py:337-352).
  Otherwise it answers from full text and says so. Every proof and every wait number records each replica's mode.
- **Forgotten here, forgotten everywhere on this machine.** A forgotten session is not in any view a reader opens
  (4.7), so every reader tool answers for it exactly as for an id that never existed, the anchor error included.
  The merger forwards ids unchanged and only checks that a view is current (V2).
- **Duplicates.** A duplicate in recall is one harness occurrence, (lineage key, uid, content identity) as defined
  in 3.2, returned twice in one response. It should only happen when one session's transcript reached two
  machines, or a copied home became two copies **(assumption; row 2 counts sessions held by more than one real
  store)**. The merger finds such sessions once per replica generation by comparing session ids over contracted
  SQL. For hits in those sessions only, it drops a replica hit whose occurrence is **already emitted in this
  response** by the local reader or an earlier replica. A hit is never dropped because another store merely holds
  the occurrence: that store may not have returned it (ranking, limits or lateness). Rows that exist only in the
  replica stay visible. Fork copies are not duplicates: they sit in another lineage, and
  the audit already counts them apart (docs/claude-code-adapter.md:725-726). A fork and its parent can still both
  appear in one store's hits today (docs/claude-code-adapter.md:441-443); step 1 does not change that.

### 4.3 Id stability

- The merger qualifies every id in a replica hit with the copy id in its 32-hex form (the dashes removed), which
  is the grammar the reader already uses (mcp_reader.py:95-98). Local ids stay bare. The merger advertises its own
  tool schemas: every argument that takes an id (`store_id`, and the cursor and anchor arguments) accepts a bare
  integer or a qualified string, so a schema-valid call can name a replica row (lane 7a tests it).
- A follow-up call with a qualified id goes to that replica's reader with the prefix removed. A bare id goes to
  the local reader. The stock reader accepts bare ids (mcp_reader.py:230-244), so its own derived uuid, which
  changes at every download (readonly_store.py:175-179), never matters and `ForeignStoreError` cannot fire.
- An id handed out before a refresh still resolves after it: the prefix names the hosted copy, a byte copy keeps
  `store_id`, and no id ever names a different row (3.4). Only a row deleted on purpose, or lost and accepted, stops
  resolving, and so does a row a view lacks as forgotten or removed (4.2), even while a superseded copy or a
  pinned generation still holds it.
- **No bare id may leave a replica hit.** `store_id` values are small integers in every store, so a bare id from a
  replica would resolve, with no error, to a different row in the local store. Ids also sit inside text the
  reader builds: expand hints (tools.py:183, :209; recall_delivery.py:270) and `lcm:<store_id>:<range>` references
  (recall_delivery.py:935; tools.py:1040). So the merger does not pass a replica hit through. It emits an allow-list
  of fields, qualifies every id field, and rebuilds hints and references from qualified ids. A field it does not
  know is dropped from replica hits, never forwarded.
- **Pass line:** every id in `other_stores` is qualified, checked by a test that scans replica hits for id-shaped
  tokens, and by the recall kit: the split corpus gives both stores the same low `store_id` values with different
  content, so a wrongly routed expand returns the wrong bytes and fails arm T. What an agent does with a prefix it
  decides to strip is model behaviour; the agent arm measures it and section 10 lists it.
- Step 2 will need a map from (copy id, `store_id`) to the merged store's ids. The stamp records both halves.

### 4.4 The user's wait

**The wait gets a bar relative to the local-only wait measured in the same run, because no bar may be one the
baseline cannot meet (ROADMAP.md:174).**

- Readers run in parallel on local disk. The merger waits for a replica until the local reader has finished plus
  250 ms, and never past the tool's own budget (`lcm_recall` has 8 s: config.py:1471, tools.py:4960). A late
  replica is left out of that call and named in the status.
- A replica that is still loading never delays the local answer.
- A forget makes a copy that held the session unavailable while its view is rebuilt (V12). No call waits for a
  rebuild.
- **Bars** (per tool, warm calls, measured with the GA recall kit, section 6 row 4):
  - merger with no replica: p90 no more than 50 ms above the plain reader;
  - two stores: p90 no more than 300 ms above local-only, no call more than 1 s above;
  - four stores: p90 no more than 500 ms above local-only;
  - every cell: p90 at most 1 s and no call above 5 s, the kit's existing tripwire
    (bench/instruments/cc_mcp_recall/README.md:45). If local-only already exceeds it, the relative bars govern
    and the result is classified in writing;
  - memory: all readers together at four stores stay within 4 GB, measured with one Claude Code session open and
    again with four (four mergers, the unit's R6 shape). **(Assumption:** the figure is borrowed from
    the budget the roadmap applies to an embedding model with its vectors (ROADMAP.md:170); no reader's memory
    has been measured for this record.) If not, replicas answer from full text only.
- **Kill line:** two stores more than 1 s above local-only at p90, with no small fix. Step 1 then folds into
  step 2 (ROADMAP.md:44-47).

### 4.5 Freshness

Each `other_stores` entry carries the replica's age. At the default cadence a fact captured on machine A is
recallable on machine B within one upload interval plus one refresh interval (proposed: 30 minutes each). A manual
`sync` on both machines makes it minutes. The bound holds only while a Claude Code session on A lasts one upload
interval past the capture. A capture made after A's last upload reaches B one refresh interval after A's next
session starts or a manual `sync` on A. Until then it is also not off-machine. A lock skip, the disk floor or a
guard refusal adds more delay.

### 4.6 Roadmap wording to amend

"The endpoint ... serves exactly one profile's store" (ROADMAP.md:125) becomes: "serves one user's stores: the
local store, and read-only replicas of that user's other machines, each read by its own reader. It accepts no
connection from another host."

### 4.7 Replica views: a forgotten session is missing by construction
- **V1. Two folders per copy.** `gen-<n>/` is the byte copy, as 4.1 builds it: checked against the manifest, never
  opened by a reader, never modified. It stays because a changed file cannot be a delta base and cannot match
  `file_sha256` (3.7.4; section 7, link 3). `view-<n>-<r>-<e>/` is a copy home built from it, and the only folder a
  reader opens. `view.json` beside them names the served view (F13).
- **V2. One safety rule.** The merger serves a copy for a call only when `view.json` names a complete view whose
  `forgotten_sha256` equals the digest of the local forgotten set read for that call, whose `removed_sha256` equals
  the digest of the copy's `removed.json` (F13), and whose `rules` equals the merger's own. Otherwise that copy
  answers as a warming one does (4.1) and none of its readers receives the call. The merger sends every call
  unchanged, so to the reader a forgotten id and a missing id are both ids with no rows. Reason: these equalities
  are all that safety needs. Every rule below only decides how soon a fresh view exists.
- **V3. What a view lacks.**
  - *The removed set R* is the local `capture:forgotten:` set, plus the sessions and row ids in the copy's
    `removed.json`, which refresh writes from the copy's own lineage and from that of each copy that supersedes it
    (3.5, F12, F13). V2 checks both, so a session that either lineage lists as removed is absent from the first call
    after refresh records it (3.5).
  - *Ownership.* A row belongs to the session its session column names. A row that links a kept session to a removed
    one belongs to the removed one and leaves with it, since it would not exist had that session never existed: a
    `session_lineage` edge with either end in R, a `rotation_predecessor_session:<V>` row whose value is in R, an
    `lcm_lifecycle_state` row with either session column in R, a `message_relations` row with either end a removed
    message. A kept session's own rows stay byte-identical: no kept session loses or changes a row of its own,
    except a row `removed.json` names by id, which the origin deleted (a partial delete, F12's `as-delete`).
  - *Names match whole.* A session id matches by equality only. A lineage value matches only as an encoding the
    product writes, parsed and compared whole: today `<source>:<row session id>` (claude_code_capture.py:63-74). A
    kept subagent `P/agent:<A>` whose agent id happens to equal a removed id is not that session, and a removed
    session's subagent `<H>/agent:<A>` is a separate session that stays (V13). A `metadata` key matches only through
    its family: the capture families are keyed by the row session or by the lineage (claude_code_store.py:52-59),
    `rotation_predecessor_session:` by the session (engine.py:6001). The view module's class map lists every family
    and every lineage encoding. Containment of an id in a key or value is never a reason to delete (V4 uses it only
    to refuse).
  - *Sets first.* Before any delete the build collects R's message `store_id`s and summary `node_id`s, adds the row
    ids `removed.json` names, and runs every keyed condition below, and V4's recheck, on those sets. A subquery on
    rows already deleted would match nothing.
  - *The deletes*, in one transaction on a private copy of `lcm.db`, with plain SQL from a standard-library module:
    - rows whose session column is in R: `messages`, `summary_nodes`, `capture_events`;
    - `messages` rows whose `store_id` is a row id `removed.json` names, even when their session stays: refresh
      writes the file before its download, so a build on the held generation must leave such a row out itself;
    - the link rows named under *Ownership*;
    - rows keyed by a collected message or node id: full-text rows (through the store's own delete triggers), chunk
      rows and their vectors, `lcm_embedding_*` rows (keyed by node id, db_bootstrap.py:1513-1521),
      `summary_node_provenance`, `host_uid_bindings`, `occurrence_bindings`, `stamp_observations`, `ledger_emissions`;
    - rows whose lineage value parses to a row session in R: `capture_events`, `host_uid_bindings`, `ingest_views`,
      `lineage_marks`, and the lineage columns of the tables above;
    - `metadata` rows whose family key names a session in R. With the links, that takes
      `rotation_predecessor_session:<V>` when its value is the forgotten session, so the reader's lineage walk
      (mcp_reader.py:213-227) cannot reach it.
- **V4. The build refuses what it cannot classify, and checks itself.** Every table has a class in one map in the
  view module (session, message, node, link, lineage, metadata, full-text, global), and so do every `metadata` family
  and lineage encoding. When anything is to be left out:
  - a table outside the map, or one the map classes `unsupported`, that holds rows stops the build with
    `view_unsupported_table`. The six entity-memory tables are `unsupported` in rules 1: only the Hermes engine
    creates them (db_bootstrap.py:3923), step 1 copies capture stores (D3), and their ownership (scope-wide entity
    records, notes citing several sessions, an append-only event log) is not decided (V13);
  - a `metadata` key outside the families, or a lineage value that does not parse, that holds an id of R stops it
    with `view_unsupported_key`.
  A compat test builds a store with every feature on and fails on a table the map lacks (as 3.6 does for the
  contract). It then builds that store's view without one of its sessions and runs the lane's independent scanner
  over it, so a new key family, a new value shape or a new session column also fails the test. After its deletes
  the build runs every condition again on the collected sets, because a delete trigger can write a row that names
  the session (query_view_store.py:431-443), and puts back `metadata` rows that a trigger removed and that name no
  session of R (db_bootstrap.py:1811-1816). Before the rename it requires zero matches, `integrity_check` ok, a
  rollback journal and no sidecar file. The merger runs as its own standard-library process and imports no sibling
  module, so it and the view module each read the forgotten set (the merger's query is cloud_reader.py:194-201
  at 593cfced (#2796)), compute F13's digests and carry `RULES`; one test holds both to the same query text, the
  same `RULES` and F13's vector.
- **V5. Summaries.** A summary node of a removed session goes with it. If a node that stays lists a removed message
  or node in `source_ids`, the build stops with `view_summary_spans_forgotten` and the copy is not served. Step 1
  cannot reach this (claude_code_store.py:207); a step that copies stores that compact must first decide the cut.
- **V6. Payload files.** The view's `lcm-large-outputs/` holds hard links to the byte copy's files (a copy where
  linking fails), as the uploader stages them (cloud_copy.py:365-374). A reference is not ownership. A file is left
  out only when its own `session_id` field, the field the reader checks (tools.py:269-281), names a session in R, or
  when it has no `session_id` and every row that references it (F3) belongs to a session in R. A file that a kept row
  references stays even when a removed row quotes it too, so the kept session's own `lcm_expand` still resolves. So
  a removed session's payload, asked for through a visible session's lineage (expand_tools.py:189-198) or by name
  with no session (tools.py:255, :272), is absent.
- **V7. Identity, and the forget that removes nothing.** `<r>` is the rules version; a rule fix retires every older
  view. `<e>` is the first 16 hex of a digest of everything the build leaves out of this generation: each session of
  R that matches a row, each payload file left out, each removed row id present. A forget that leaves nothing more
  out gives the same `<e>`: the build rewrites `view.json` with the new digests, the folder is untouched, and the
  running reader keeps serving. When nothing is left out at all, the view's `lcm.db` is a hard link to the byte
  copy's file (a reader opens `mode=ro`, readonly_store.py:155), so nobody pays a second copy. The link is made only
  when the byte copy has a rollback journal and no sidecar file (V4's check); otherwise the build clones or copies
  it. Reason: a forget usually names a session of this machine, which other machines' copies do not hold (4.2).
- **V8. Building.** A builder waits for `replicas/<copy id>/view.lock` (exclusive) for up to 10 minutes, then exits
  `view_busy`. Holding it, the builder:
  1. reads `current` for the generation and holds that `gen-<n>/`'s shared lock (4.1) until it ends, so refresh
     cannot delete the byte copy under it;
  2. reads R and `removed.json`, then finds what would be left out, with the deletes' own conditions run as reads
     on the byte copy and V6's test on its payload files;
  3. if that view exists, rewrites `view.json` and stops;
  4. otherwise builds in `view-...tmp-<token>/`: a hard link when nothing is left out (V7); else it checks the disk
     floor (V11), clones `lcm.db` where the filesystem can (`cp -c`, a reflink) or copies it, deletes and checks
     (V4); then links payloads and syncs;
  5. reads R and `removed.json` again and starts over if either changed (three passes at most);
  6. renames the folder and replaces `view.json` atomically.

  A folder without `.tmp-` in its name is therefore complete, and `view.json` never names anything else. A crash
  leaves a `.tmp-` folder, or a complete folder that `view.json` does not name; the next build deletes the first and
  adopts or deletes the second. A view folder is deleted only under 4.1's rule (an exclusive lock, never waited for).
- **V9. Who starts a build.** Refresh builds the view of a new generation before it replaces `view.json`. A merger
  that finds V2 failing starts one build per copy as a child process (the `view` command, at low priority) and goes
  on answering; it never waits. The child's stdin, stdout and stderr are `/dev/null` and it runs in a new session,
  so nothing it writes reaches the MCP stream and it holds none of the client's pipes; the merger reaps it on a
  later call without waiting. `sync` builds too. Each build records its end in `view-result.json` (F13). A merger
  starts no build for a copy while its own child for that copy runs; while the record names this call's digests,
  the current generation and the merger's `rules` with a lasting refusal (`view_unsupported_table`,
  `view_unsupported_key`, `view_summary_spans_forgotten`); or while it names a passing one (`view_busy`,
  `view_disk_floor`) less than 60 s old. So a lasting refusal costs one build, and a passing one is retried.
  Reason: with only the 30-minute timer (4.5), a forget would leave every copy unavailable for up to 30 minutes,
  even a copy that never held the session, because V8's no-rebuild path also runs inside a build.
- **V10. Concurrency.** Builds of one copy never overlap: a second builder waits for `view.lock` (V8) and then reads R
  itself, so a forget that lands during a build gets its view from the next builder. Refresh holds `upload.lock`,
  then waits for `view.lock`, so a generation that arrives during a build gets its view next. Two forgets in a row:
  the second changes the digest, so the first view is served only if it already lacks both (V8 step 5). Each merger
  reads the forgotten set once per call, before its serial dispatch; a forget that commits during a call counts as
  after it. Handover: on a new generation with unchanged digests the old reader serves until the new one has
  answered its probe, as today; on a changed digest it receives no further call.
- **V11. Disk.** S is one copy's database. Payloads are stored once wherever hard links work; on a filesystem
  without them V6 copies the kept files, which costs their bytes again. The view's database costs nothing when
  nothing is left out, a few changed pages when cloned, and S when copied: at worst 2S plus payloads at rest. A
  forget deletes the stale view first when no reader holds it; otherwise it costs one more S until that reader
  closes. A refresh deletes the old byte copy once the new one is verified, then builds the new view while the old
  one serves: at worst 3S, plus S per older view another session still reads. Two view folders can exist during a
  refresh; only the one `view.json` names is served, and V2 holds for it because a refresh leaves the digests
  unchanged unless the new lineage lists new removals. Floors: refresh keeps 4.1's. A build checks its floor only
  before it clones or copies the database or copies payload files: 5 GB free plus S, whether it clones or copies
  (the deletes can change every page of a clone), plus the kept files' bytes when it must copy them. Below it the
  copy stays unavailable (`view_disk_floor`); the no-rebuild path and hard links write almost nothing and need no
  floor.
- **V12. The wait. (Assumption: every number in this item; row 5V measures them.)** Each call adds two small file
  reads per copy (`view.json`, `removed.json`), one digest of the local forgotten set and one per copy. A routed call
  becomes one reader call; the parked merger made up to three and opened the copy's database itself
  (cloud_reader.py:506-536 at 593cfced (#2796)). After a forget, a copy with nothing more to leave out is back after
  the match alone, which reads the session columns and every payload file: at most 5 s per GB. A copy that held the
  session is back after copy, deletes, check and the new reader's probe: at most 30 s per GB, plus the reader's start.
  Lane 0's real store is 2.75 GB with 2,414 sessions: about 14 s and 83 s at those bars. Until then that copy alone is
  `warming`. Local answers never wait and 4.4's bars are unchanged: the build is another process.
- **V13. Limits kept.** Bytes of deleted rows can stay in the view's free pages and in the full-text index's
  internal tables (`*_data`, `*_idx`), as in the local file (5.2); a full-text query does not return them. A
  forgotten parent's subagent sessions are separate sessions and stay, as they do locally (#2186). A step that copies
  stores with entity memory decides first which entity rows a session owns (V4).

## 5. Isolation, delete and restore

### 5.1 Isolation and opt-in

| Bar | Mechanism | Test |
|---|---|---|
| Capture is opt-in | Nothing in step 1 runs until `cloud/cloud.json` exists and passes the ownership check the override already uses (adapters/claude-code/bin/lcmx_mcp.py:35-41). `enable` prints the opt-in text and needs `--accept`. `disable` stops uploads and removes the override only when it carries `"owner": "lcm_x_cloud"`; otherwise it leaves the file and says so. Step 1 has no command that deletes a hosted copy: the user deletes it in the bucket host's own console, and the opt-in says so. | On a home that never opted in, capture and recall run with sockets blocked: 0 connection attempts, the audit unchanged. After `disable`: 0 uploads, plain reader starts. With another tool's `reader-override.json` in place, `enable` refuses before any bucket write and names the path, and `disable` leaves that file unchanged and says so. |
| Each hosted store reachable only by its own user's agents | The storage vendor's access control: one prefix or bucket per user and a credential scoped to it. We write no authorisation code. Our client also refuses any key outside its configured prefix, and names read from a manifest (copy ids, part and payload names) must match their grammar before they are used as paths. TLS certificates are verified; `enable` refuses a host it cannot verify and names a missing CA bundle **(assumption: some found interpreters, such as the python.org macOS build, ship without one; lane 6 tests it)**. | Negative test with two credentials, sent with our client's prefix check bypassed, so that the host itself must refuse (HTTP 403): B cannot list, read, write or delete under A's prefix. Positive control first: A can. It runs on the owner's test bucket (O1) when it exists, and on the host that ships at the gate. There is no MinIO run: installing it is a download, and the bar names a hosted store. |
| Credentials | A file in `<home>/cloud/` that only its owner can read or write (mode 0600 or stricter). Never in `cloud.json`, never printed, never typed by an agent. No environment-variable option: Claude Code's environment reaches the agent's shell, and an `env` the agent runs would be captured into the store and uploaded. | The file is refused when group or others can read or write it (the override's ownership check tests only writability, adapters/claude-code/bin/lcmx_mcp.py:35-41, so this check is new). A planted credential value is absent from logs, `status`, and the captured store after a session that runs every cloud command. |

The opt-in text must state these facts: what is uploaded (every captured session, the saved tool outputs, and
the raw transcript lines, including text that redaction removed from search, as O3 decides); who can read the
bucket; that the user's other machines will hold a copy; that a lost machine exposes the copies it holds (O3); the
delete rule and its limits (5.2); that an upload refuses rather than mirror an unexplained local loss; that a forget does not reach an idle copy
(one no machine publishes to any more, after a chain conflict, a lost `cloud/state.json`, a changed hardware id or a
replaced machine) until the user deletes it (3.4); that a forget does not reach
a superseded copy until the user deletes it (3.5); that a whole hosted copy is deleted in the bucket host's own
console, since step 1 has no delete command; and that a
Hermes `/lcm doctor clean apply` pointed at a capture store is refused at upload until the user runs
`accept-loss --as-delete`.

### 5.2 Deliberate delete

- **Mechanism.** A deliberate delete of a capture-store session writes `capture:forgotten:<session>` in the same
  transaction as the delete (claude_code_store.py:100-108), through the purge registry that both delete paths run
  (store.py:453-466, :1261-1275; command.py:1458). The next snapshot has no row, ledger line or payload of that
  session. The guard exempts it. Retention removes every older generation that is not pinned in the same publish (3.5).
- **Today there is no forget command for Claude Code** (.github/release-notes/claude-code-v1.0.0.md:38). The only
  supported path is `delete_session_messages` in a process that has loaded the adapter. The test drives exactly
  that, on a scratch store.
- **A delete with no marker** happens when the deleting process never loaded the adapter, for example a Hermes
  `/lcm doctor clean apply` pointed at a capture store, which is unsupported (docs/claude-code-adapter.md:269-271).
  The guard refuses. That is the safe direction. `accept-loss --as-delete` is the way out for a delete the user
  meant; plain `accept-loss` keeps the generation before the loss (3.4).
- **Other machines.** A session forgotten on A leaves B's replica at B's next refresh. Until then B can still
  recall it. The opt-in says so. A session forgotten on B is absent from every replica view on B from the first
  call after the forget (V2), and stays absent after any refresh.
- **The stated limits.** The bar is met at the level of rows, ledger lines and payload files. Bytes of deleted
  rows can remain in unused pages of the database file and in the full-text index's internal tables, in the
  hosted copy exactly as in the local file; search no longer returns them. Row 6's synthetic run found a planted
  id in the full-text index's backing tables even after `VACUUM` (section 6).
  Free-page residue travels in base parts and delta objects while the local file keeps it; a later base can
  copy it again, and older objects keep it until their generations go. LCM-X does not set `secure_delete` and
  step 1 adds no scrubbing (3.7.4). Parts and delta objects of failed or interrupted attempts or dropped staging
  copies that no manifest named can hold a forgotten session for at least 7 days, until the next successful
  retention after that (3.5). The opt-in says each in one line.
- **Tests.** (a) With no pin in the bucket, forget a session that holds a planted string; publish; restore the current generation: 0 rows,
  0 ledger lines, 0 payloads of that session, and recall finds nothing. List the bucket: no older generation
  holds the session, and no key holds it except one that no manifest ever named, younger than 7 days by the
  bucket's clock. Then repeat with the process killed after the manifest is written and before retention: the
  next publish that wins its own create leaves no generation that holds it. Then forget a session after the staging copy is taken and
  before the manifest is created: that copy is dropped, and the retried publish is a base holding none of it.
  Kill between deleting exclusive keys and deleting manifests: the next winning retention finishes (3.7.4).
  (b) Delete rows with no marker: the publish is refused and the bucket is unchanged. (c) `accept-loss`, then publish: passes, and the generation before the loss is still in the
  bucket and restores with the rows. (d) `accept-loss --as-delete`, then publish: passes, and no generation holds
  the rows. (e) Control: a publish with no delete passes. (f) Two publishers under one copy id, interleaved at each
  step (parts, part check, manifest, retention): at most one manifest per number, no part a kept manifest names is
  deleted, and the loser reports `chain_conflict`; a publish paused past the other's retention recreates a
  deleted number below the head, including one a forget emptied inside the kept window; it runs no retention, no
  replica or guard reads it, and a restore of it by number fails visibly when it names a missing object. (g) Forget a session that a pinned generation holds: `status`
  reports `pin_holds_forgotten`; after `accept-loss --release`, the next publish leaves no generation that holds
  it. (h) Forget the newest session, capture more, publish: it passes, with no `store_id_reused`. (i) Replace the
  store with an older copy of itself: the next publish is refused with `store_id_reused`, both before and after
  capture writes past the mark; a forgotten session that the older copy holds again is refused with
  `forgotten_session_returned`; `--new-copy` passes, the old copy is superseded (never pinned or pruned), an id qualified with
  the old copy id never resolves to a row of the new one, and after a forget of a session the superseded copy
  holds, `status` reports `superseded_holds_forgotten` and recall finds nothing, and an id of that session
  qualified with the old copy id, issued before the forget, answers exactly as a missing id does. (j) Delete `cloud/state.json`: `chain_conflict`, then `--new-copy` passes and publishes, and the old copy is idle. Stop the process after the
  manifest is created and before it is recorded: the next publish finds its own pending pair and passes.
  Change the hardware id, or copy the home to a second machine: `--new-copy` passes, no
  command writes to the old copy afterwards, and `status` lists it as idle or as the other machine's. Run a
  second machine's publish attempts against the old publisher's retention at every step: no kept manifest names a
  missing part. Copy a home to a second home on the same machine, or clone a VM that keeps its hardware id, once
  with `cloud/state.json` at its last pair and once with a pending pair, and run both publishers' attempts and
  retention interleaved at every step: after at most one generation each, all but one meet `chain_conflict`,
  `store_id_reused` or a loss refusal; no attempt writes onto an existing key; no retention runs except after its
  own winning create with that manifest at the head, and a resumed attempt whose create is refused runs none;
  `status` never calls the other copy safe to delete; and no kept manifest names a missing key, except in the case
  3.4 states: with a pending pair, hold the other copy's retention after its winning create of N+1 between its head
  read and its deletes for more than 7 days, while this copy's attempt for N+2 passes its part check before the
  delete and creates N+2 after it; the head then names a missing key, its restore and refresh fail visibly, and the
  newest generation is whole again only if this copy publishes again. Resume an
  attempt more than 3.5 days after it started: it is abandoned and a new attempt starts. Copy a home, forget session S on the first machine, run `--new-copy` on the second: it uploads S.
  (k) The cases found on #2796 before it was parked (padded and empty routed ids and references, expansion budgets
  spent on forgotten sources, the review threads left open) and a forgotten session reached through a kept
  session's lineage, each as a forgotten-against-missing pair through the merger (the lane packet's B1 to B8).

- **Delta lifecycle tests (lane 3).** Publish S, lose S, plain `accept-loss`, stage a snapshot, then forget S
  before the manifest: exactly one restart ends in a winning base. Repeat with a kill before retention: the
  next winning publish prunes older unpinned generations by the head's cutoff. `--as-delete` S, publish new rows for S,
  then forget S: a base. Separately, a second `--as-delete` decision on a session whose new rows were published
  also cuts a base. Release pins in each case and
  verify no unpinned object holds the removed rows. Roll back under a pending
  decision: `as_delete_rows_present` once, no restart loop (3.7.3).

### 5.3 Restore

- **What it is.** `restore --generation N --to <fresh directory>` downloads the manifest, base parts,
  delta objects and payloads, verifies every sha256, rebuilds `lcm.db` by base plus chain (3.7.3), assembles
  `lcm-large-outputs/`, and checks the result. A named object that is
  missing or fails its sha256 stops the restore and names the object (lane 5 tests it on a generation with a deleted
  part). A rebuild digest mismatch stops restore with `rebuild_digest_mismatch` and the generation (3.7.4).
- **Refusals.** The target of `restore --to` must be empty. It must not hold `capture-owner.json` or `lcm.db`, and
  must not be a capture home, a Hermes home, or anything inside one. Replica refresh is a separate path that
  writes only under `<home>/cloud/replicas/`; staging writes only under `<home>/cloud/staging/`. Test: each
  refusal, by name.
- **Checks, in order.** (1) `PRAGMA integrity_check`. (2) The independent checker against the manifest and, where
  it is at hand, against the origin store as of that generation's cut; a reference that names no file is
  reported under its count (3.3) for each side: the same unresolved names on both sides pass, with the counts
  reported, and a reference unresolved on one side only fails (lanes 4 and 5 test both, and that a store holding only
  quoted placeholders passes D12). (3) Only when a saved copy of every transcript
  of every session as of that cut exists (the gate proof always has one): the existing outside-in checker run
  against the restored folder with those transcripts: `--home` names the folder, and every transcript of each
  session is passed with `--transcript SESSION=PATH`, oldest first. It reads `mode=ro`
  (scripts/capture_outside_in.py:99-108). Otherwise restore reports `check3_not_run` and passes on 1, 2 and 4.
  (4) Start the stock reader on the restored folder, run one `lcm_grep`, and run one `lcm_expand`: of a payload
  the manifest names when it names any, otherwise of one message by `store_id`; with no rows, (4) is the
  reader's start and one empty `lcm_grep`. A failed check that does run stops the restore and keeps the folder.
- **The proof, before any user upload.** Seed a store from real sessions. Publish four generations to the gate
  bucket (base, delta, delta, forced base), with capture running between publishes. At each publish keep a scratch copy of the origin store and of
  its transcripts, so each generation has its own oracle. Restore the newest: all four checks pass against its own
  saved copy. Then delete the newest manifest and its unique parts from the test bucket, as a simulated loss, and
  restore the one before: all four checks pass against that generation's saved copy. The generations differ, so this
  is a real test. Then delete the first delta's object: that same restore must fail and name it (3.7.4).
  Record: generation ids, digests, counts, no content.
- **Per user.** D12: `enable` repeats checks 1, 2 and 4 on the user's first generation before uploads are
  scheduled.

## 6. The design check

**Six rows plus rows 2D and 5V decide the open numbers for their dependent lanes. Each has a method,
a bar set now, and what happens if it fails.** Rows 1 to 6 run before the lanes that depend on them start,
on throwaway prototypes; row 2D runs after lane 2 merges, on its real snapshot path and manifest writer,
with size and timing cells separated as in 3.7.5; row 5V runs on lane 5v's build. The prototypes are a snapshot script
(the standard library's backup, about 60 lines), a comparer that digests every contracted table of two stores
(about 80 lines), a merger stub (two to four stock readers, prefixes, a separate hit list; about
200 lines) and a file-swap loop (about 60 lines).

**Where it runs (release manager's edit).** The shared development machine is used by several lanes, so no timing on it is judged.
- Rows 2 and 6, and row 2D's size cells, measure no time. They run on the shared development machine, on scratch copies.
- Rows 1, 3, 4 and 5, and row 2D's timing cells, run on the bench hosts, Linux, adding the cross-platform case:
  - the first bench host once the v0.30.1 D1 row ends (about 11-18 22Z);
  - the second bench host as the second machine once D4 ends (about 11-19 09Z).

  Row 1 replays recorded transcripts through the real hooks and worker, as the unit's tests drive capture, so no
  live model session runs on a bench host. Every number names its machine, since neither host is the reference
  laptop of VISION.md:70.
- Bench access stays with the release manager. Codex writes the throwaway prototypes from a spec; the
  release manager runs them on the hosts.
- Storage is a directory transport, except for row 3. Row 3 needs a real S3 endpoint and uses the owner's test
  bucket (O1), so it waits for that bucket rather than for a MinIO download. No agent creates a paid resource.
  Row 3 decides nothing in the design: its fallback is a transport fix. So it gates the gate run (lane 10), not
  lanes 2 to 9, and runs when the bucket exists.

Ground rules: every store used is a scratch copy under one scratch root with a stated size cap and a `df` check
before each cell; no script accepts a path that is a live capture home; throwaway code lives in a scratch folder
outside the repository; the comparer is first shown one removed row and one changed row, and must name both, before its
result counts.

| # | What | How it is measured | Bar | If it fails |
|---|---|---|---|---|
| 1 | **Snapshot beside live capture** | Four sessions in parallel on one scratch store (the unit's R6 shape, replayed from recorded transcripts) for 30 minutes: first without the snapshot loop (the control), then with a snapshot every 60 s, then the control again, on the same host. | Audit `ok` and `verified`; outside-in checker passes; 0 failed batches, 0 `still_wal`, 0 `.failed` spool records in the worker log; hook wall p95 no more than 5 ms above the higher control's, or no more than the two controls' difference if that is larger. The absolute wall p95 is recorded beside the unit's 50 ms target (docs/claude-code-adapter.md:727) and is not gated (bench/specs/RELEASE-READINESS-V1.md:90); a run over 50 ms is reported to #2141 (docs/claude-code-adapter.md:740-741) and does not fail this row. Every snapshot passes `integrity_check` and the comparer against the origin at its cut; 95% of cycles finish within three restarts. | Snapshot only when no worker is running and the store has been quiet for a set time. If that also fails, the snapshot moves into the capture worker, a core-train lane. |
| 2 | **Store and upload sizes** | Three stores: the GA gate's growth-check store (about 690,000 lines); a store seeded from a maintainer's real transcripts by the growth check's method (bench/specs/RELEASE-READINESS-V1.md:94-96); the second machine's store. For each: database bytes, payload bytes, the raw-ledger share, the vector share, bytes against transcripts. Then one working day of publishes at the default 30-minute cadence: bytes uploaded by each, and in total, counting base parts or delta objects, manifests and payloads (3.7). Also the number of session ids held by more than one of the three stores. | Sizes have no pass line; they are recorded. Daily bar and base-day rules: 3.7.5, including the 32 MiB floor and all uploaded bytes. The same bytes are what the user's other machines download, so O1's cost figures use this number. | Whole parts exceeded the bar; base-plus-delta is adopted (3.7), with row 2D as its gate. |
| 2D | **Base-plus-delta path** | The replay, manifest scaling, 90-day simulation, forced-base and forget cells of 3.7.5, embeddings off and on. Size cells on the shared development machine; timing cells on bench hosts with row 1's replay. | All five pass lines in 3.7.5, including the no-growth continuation; all uploaded bytes count. | Raise zlib level once on ratio failure, then report; return the size constant on line (5) failure; worst-window share above two thirds triggers manifest-compression follow-up. |
| 3 | **Upload time** | First upload and incremental upload from each machine. A fixed 1 GiB transfer first gives the link's throughput. Then `kill -9` mid-upload three times and rerun. | Work outside the transfer (snapshot, guard, hashing) at most 60 s per GB. Transfer at least 80% of the link's throughput. After each kill: the rerun resumes the same attempt with its pending token, completes, re-sends no complete unchanged base part and at most one generation's delta objects (3.7.4), writes nothing onto an existing key, and the bucket never holds a manifest that names a missing part. | Fix in the transport lane; no design change. |
| 4 | **Recall wait across 1, 2 and 4 stores** | The GA recall kit (bench/instruments/cc_mcp_recall/README.md:9-17, :23-25), arm T. Cells: plain reader, one store; merger, no replica; corpus split over 2 stores; over 4 stores; then 2 and 4 again with each replica padded to the largest real store from row 2. At least 200 warm calls per tool per cell, first calls reported apart, memory of all readers recorded with one session open and with four, each replica's mode recorded. | The bars in 4.4. Also correctness: 44 of 44 values found with byte-equal expand in every cell; 0 duplicates; with one session planted in two stores, including at least one externalized tool result, it is returned once; the kit's empty-store control passes first. | The kill line in 4.4: step 1 folds into step 2. |
| 5 | **Replica refresh** | The second machine pulls each new generation into its own directory while two sessions' mergers send recall calls, with the handover in 4.1 (pointer file, probe call, shared folder locks, then delete). 100 handovers, some timed to land mid-call, with 20 forgets among them; same bars. | 0 failed calls; no call more than 300 ms above local-only (the merger waits 250 ms for a replica, so 50 ms is left for its own work); an id issued before a handover expands after it; the first call after a handover finds a value that only the new generation holds; no call reads two generations; refresh takes at most the transfer time plus 30 s per GB; peak disk at most twice the replica, or three times while another session still reads the oldest generation (4.1). | If calls still fail under this handover, the kill line applies. |
| 5V | **Replica view build** | On row 2's largest real store, time the match, a cloned build, a copied build and the reader's start, embeddings off and on. | V12's. | If the build misses: `quick_check` for `integrity_check`. If the match misses: a per-generation list (section 11). |
| 6 | **Deleted bytes** | On a scratch store, forget a session that holds a planted string, through `delete_session_messages` with the adapter loaded. Take a snapshot. Search the snapshot's bytes and its full-text index for the string. | No pass line. The result decides the wording of the opt-in's limit in 5.2. | Not applicable: it is a disclosure, not a bar. |

The product instruments prove themselves later, in their own lanes: lane 4's checker must name eight seeded faults
(a row removed, a row doubled, a row's content changed, a payload removed, a raw ledger blob cut, and the three
delta faults in 3.7.4) and pass a clean copy; lane 3's guard must refuse an unexplained delete and pass a
deliberate one. Neither gates anything before that.

Order: row 1 first, because lane 2 waits for it. Then rows 2, 3 and 6.
Rows 4 and 5 last, on the merger stub. Row 2D runs once lane 2's real snapshot path exists, after lane 2 merges.

The numbers still assumed today: 6 GB per user (draft 1's hosting estimate) and every upload time. The shared development machine's own
capture store is 13 MB (created 2026-11-13), so it is not a size sample. Draft 1's question on whether the reader
honours a stamped `metadata.store_uuid` is answered on main (readonly_store.py:169-172) and is not a check row.

## 7. The step 1 gate mapped to evidence

**Every roadmap bar has one instrument and one pass line. The instrument for a bar is never the code that
implements it.**

| Roadmap bar (ROADMAP.md:22, :126) | Instrument | Pass line |
|---|---|---|
| Capture is opt-in | Socket-blocked run on a home that never opted in; `enable` and `disable` tests (5.1) | 0 connection attempts before opt-in; uploads start only after `--accept` and a passed restore check; 0 uploads after `disable` |
| Each hosted store reachable only by its own user's agents | Two-credential negative test on the host that ships, positive control first, with our client's own prefix check bypassed (5.1) | Every cross-prefix list, read, write and delete refused by the host (HTTP 403); own-prefix operations succeed |
| One proven restore before any capture | The restore proof in 5.3, on a store seeded from real sessions, before any user upload; plus `enable`'s own check | All four checks pass on the newest generation and on the one before it after a simulated loss, each against its own cut |
| 0 lost and 0 duplicate rows across stores | Chain of four links, on two machines. (1) Transcript to local: the audit and the outside-in checker, as the unit gate runs them (bench/specs/RELEASE-READINESS-V1.md:57-66). (2) Local to hosted: the copy checker, origin store against restored generation, every contracted table by key, plus payload digests and raw ledger digests; a reference that names no file is reported under its count (3.3); the same unresolved names on both sides pass, and one-sided ones fail. (3) Hosted to replica: every downloaded object's sha256 and the rebuilt file's `file_sha256` (3.7.4). (4) Transcript to restored copy: the outside-in checker on the restored folder. The list of hosted copies comes from each machine's `cloud/cloud.json` (copy id) and `capture-owner.json` (origin id), not from the bucket, because a checker cannot see an input it is not given (docs/claude-code-adapter.md:730). | 0 lost and 0 duplicated at every link; order kept; 0 unpaired. No copy id published by two machines past one generation each (the invariant and the publish token, 3.4; test (j)); an origin id under two copy ids (a copied home) is listed. Merged recall returns no occurrence twice. |
| One real recall proof | The recall kit's agent arm through the merger: a value captured only on machine A is asked for on machine B in a fresh Claude Code session told nothing about stores. One more run on a machine with no local history, the replaced-machine case. Question and expected value written down first by someone other than the runner. | The agent answers with the value; the hit carries A's copy id; `lcm_expand` returns the stored source byte-equal. With the replica removed, the same question gets a local-only answer and no tool error. Claim class: one real session, not a pass rate. |
| Copied stores carry an import stamp | The copy checker reads `import:origin` in every restored generation and replica and compares it with the manifest | Present, equal to the manifest, origin id equal to the origin machine's owner file |
| A deliberate delete reaches the hosted copy, or the opt-in says it does not | The tests in 5.2 | (a) to (k) pass; the opt-in text carries the delete rule and its limits |
| The design check shows recall can read across stores | Section 6, rows 4 and 5 | Their bars |
| Unstated but binding: local capture is unharmed (VISION.md:60) | Section 6 row 1, repeated on the candidate; the unit's own gate on the candidate | As row 1; R1-R10 pass |

Gate evidence covers Claude Code stores only. A Grok Build store qualifies later by the same restore proof and
copy checker, once its capture ships.

## 8. The build plan

**Fourteen lanes: the design check, a document lane, eleven code lanes and the gate run. No lane changes the core
store, ingest or the stock reader. Only lane 9 enters the unit's import closure.** Sizes are non-generated lines,
tests apart. Tiers follow the diff: anything that guards, deletes, authenticates or sits on a user's data path is
High. **(Assumption:** every size and every day count below is an estimate from the scope in this record, not
from a prototype.)

| Lane | What | Files (new unless marked) | Size | Tier | Needs |
|---|---|---|---|---|---|
| 0 | Design check: throwaway prototypes, kit runs, result card | none in the repository; prototypes in a scratch folder outside the repository; the result card goes into lane 1's document | ~400 throwaway | none (no product code) | this record's bars |
| 1 | This record as a repo document; roadmap and adapter-doc wording | `docs/cloud-capture-step1.md`; `ROADMAP.md`, `docs/claude-code-adapter.md` (edits) | docs | Low | decision by 11-21 |
| 2d | Page lists/digests, last/pending lists, delta encode/apply, pure base rule, skip signature (3.7.6) | `cloud_delta.py`, `tests/test_cloud_delta.py` | ~220 | High | 3.7's codec interface |
| 2 | Snapshot, manifest including `database`, base-or-delta publish through 2d, pre-manifest chain check, part store, directory transport, contract constants | `cloud_copy.py`, `scripts/lcm_x_cloud.py` (`snapshot`, `publish`), `tests/test_cloud_copy.py`, `tests/compat/test_cloud_copy_contract.py` | ~480 (+~80) | High | row 1's result; 2d before merge |
| 3 | Publish guard (`store_id_reused`, `forgotten_session_returned`, the chain), retention including `deltas/`, forget trigger for the base rule, tests of `as_delete_rows_present`, delta-key interleavings, superseded copies, `accept-loss` decisions, pins and their effect on the guard and retention (library) | `cloud_guard.py`, `cloud_copy.py` (edit), `scripts/lcm_x_cloud.py` (edit), `scripts/maintainer_gate.py` (edit), tests | ~330 (+~30) | High | 2 |
| 4 | Independent decoder, rebuild and copy checker, written from section 3 only, by a different author than lanes 2 and 3; eight seeded faults | `scripts/cloud_copy_outside_in.py`, `tests/compat/test_cloud_copy_outside_in.py` | ~310 (+~60) | High | section 3 |
| 5 | Restore rebuild/check, replica refresh by local patch into a new directory with full download as fallback, reader handover (4.1, 3.7.4) | `cloud_restore.py`, tests | ~380 (+~80) | High | 2 |
| 5v | Replica views: build, identity, payloads, the view command | `cloud_view.py`, `scripts/lcm_x_cloud.py` (edit), tests | ~300 | High | a hand-built byte copy |
| 6 | S3 transport (conditional create, TLS verification), credentials, isolation test | `cloud_s3.py`, tests (recorded responses; live tests against the O1 bucket, skipped without its credentials, required at the gate) | ~350 | High | 2; the bucket for its live proof |
| 7a | Merger: children, routing, id qualification, fallback | `cloud_reader.py`, tests | ~300 | High | a hand-built replica home; rows 4 and 5 decide whether it continues |
| 7b | Merger: duplicate rule, status, deadline; kit cells for 2 and 4 stores | `cloud_reader.py`, `bench/instruments/cc_mcp_recall/` (edit) | ~250 | High | 7a |
| 8 | Opt-in: `enable` (with `--new-copy`; a plain `enable` re-run on a home whose `cloud.json` names a copy id reads the head first, per the invariant), `disable`, `status`, `sync`, `accept-loss` (`--as-delete`, `--release`; calls lane 3's decision API); `cloud/upload.lock` around every bucket write; opt-in text; the sync timer; `status` chain length, bytes and last base reason | `scripts/lcm_x_cloud.py` (edit), `cloud_reader.py` (edit), tests | ~315 (+~15) | High | 3, 5, 6, 7b |
| 9 | Unit packaging, smoke, release-spec gate items, user docs | `scripts/build_claude_code_unit.py:32`, `scripts/claude_code_unit_delta.py:24`, `scripts/claude_code_unit_smoke.py`, `bench/specs/RELEASE-READINESS-V1.md`, `adapters/claude-code/README.md` (all edits) | ~200 | High | 8 |
| 10 | Gate run on the candidate: two machines, the real bucket, records | none | none | gate | 4, 6 and 9; row 3; owner decisions O1 and O3 |

- **Parallel.** Lane 2d precedes lane 2's merge; lane 2 starts against its interface. Lanes 4 and 7a start with 2:
  4 and 7a need only section 3's layout. After 2 merges, lanes 3, 5 and 6 run together. Lane 5v needs only a
  hand-built byte copy; lane 5's refresh calls it, so 5v merges before refresh does. 7b follows 7a. The week of
  11-30 holds five writers (3, 5, 5v, 6 and 7b), each Codex lane in its own leased home; review, not homes, is the
  limit that week, so 7b's review queues behind 3, 5, 5v and 6.
- **One writer each.** The chain 2d to 2 to 3 to 8 to 9 to 10 is dependent and is never split across parallel writers.
- **Lane 2d** has a closed codec surface (3.7.6); lanes 2 and 4 receive the delta-object and `database` blocks
  from one source, and 2d and 4 share the page-list vector. It takes rule 3 and rule 4 inputs from lanes 3 and 2.
- **Lanes 7a and 7b** are split because one merger PR would pass 200 lines, the stop-and-report signal.
- **Lane 5v.** Lane 7a returns after 5v with its forget code deleted and V2 added. Lane 5's refresh writes
  `removed.json` before its download and calls 5v's build at its end. **(Assumption:** 5v takes about four working days and 7a's return two, before
  review; the reader side then has no slack against lane 8's week of 12-07, so the stated risk to 2027-01-04
  grows.)
- **Unit closure.** The unit asset ships only two entry scripts today (scripts/build_claude_code_unit.py:32;
  docs/process/claude-code-unit-release.md:30-31). Lane 9 adds the third. Until then nothing in lanes 2 to 8 can
  change a shipped unit. The candidate runs the unit's full gate, R1-R10 included, as every unit release does.
- **Core store.** No lane touches it. If the design check forces one (row 1's last fallback, or a reader change
  for swaps), that lane ships in a core train and the unit pins that core commit.
- **Authors (release manager's edit).**
  - Lanes 2d, 2, 3, 5, 5v, 6, 7a, 7b, 8 and 9 go to Codex (`gpt-6.1-sol`, high) against a spec with its
    acceptance tests.
  - Lane 4, the independent checker, goes to a Claude `coder`. A different model family from the author of lanes 2
    and 3 writes what judges them.
  - Lane 1 is the release manager's.
- **Review.** A Claude-authored High lane takes two reviewers; a Codex-authored one takes one fresh Claude reviewer.
  Lanes 3 and 4 decide what counts as loss, so they get the strongest reviewers available.

**Proposed ship date: 2026-12-28, with a stated risk to 2027-01-04.** Dates are ceilings; the count is
Monday to Friday and prices no holiday. It is derived, not chosen:

- 11-17 to 11-24: design check, beside the decision on this record (due 11-21).
  - Prototypes and rows 2 and 6 from 11-17.
  - Row 1 on the first bench host from 11-19, done by 11-20, because lane 2 waits for it.
  - Rows 4 and 5 by 11-24.
  - Six rows on two hosts that are busy until 11-18 and 11-19 is the weakest number in this plan.
- Row 2D runs once lane 2's real snapshot path exists, after lane 2 merges (3.7.5); it reports manifest bytes
  and local writes as well as the upload bar.
- Lane 2d was dispatched when the design was decided, 11-16, and merged the same day (#2797), so its four
  working days are spent. Lanes 2, 4 and 7a also started on 11-16, a week before the weekly plan below. The
  dates in this section are the plan as decided on 11-16 and are not re-dated here.
  Lanes 4 and 7a start with 2; 7a is unaffected. Lanes 3, 5 and 6 follow 2; 7b follows 7a.
- Week of 11-23: lanes 2, 4, 7a. Week of 11-30: lanes 3, 5, 5v, 6, 7b. Week of 12-07: lanes 8 and 9.
- Week of 12-14: candidate and gate, including the unit's 24-hour lane.
- Lanes 3, 5 and 6 start once lane 2 merges, about three working days later than without the delta work; later lanes and the gate move with them, and the critical path below carries the end dates.
- Lane 3 adds under a day. Lane 5's extra ~80 lines can delay lane 8, which needs it, by about a day, within
  the same four-working-day move.
- The critical path is 2d, 2, 3, 8, 9, then the gate. From lane 2's 11-23 start, its
  4 + 4 + 4 + 2 + 4 working days plus the delta work's four working days land on 12-22 with no slip at all.
  Two review rounds on two High lanes of that path add four working days: 12-28. All four code lanes after 2d
  are High; if each needs two rounds, it is about 01-04. Best case, with one review round, is 12-24 if
  2d merges by 11-25 and lane 2 merges in its slot.
- It moves day for day if the owner's bucket (O1) arrives after 12-07 (lane 6's live proof, row 3 and the gate
  wait for it), if row 1 is not in by 11-23, or if rows 4 and 5 end after 11-24.
  The week of 12-14 also holds facts that update (12-16) and compaction part 1 (12-19), so reviewer time is
  the likeliest slip.
- The gate decides. If it passes earlier, step 1 ships earlier, but not before 12-16, the roadmap's floor.

## 9. Owner decisions

**Three decisions need the owner. Two are about a bucket: who pays for it, and what we promise about reading it.
The third is what a user's other machines and the bucket hold, and what a lost machine exposes.**

**O1. Whose bucket (money).** Prices below were read 2026-11-10 for draft 1 and were not re-read. They assume
6 GB per user, which is unmeasured. They price one database, while step 1 keeps three generations (3.5).
Base-plus-delta storage usually peaks near 4 times the database after a size base; with large new deltas,
two kept chains can reach under 6 times. These are database-only figures: pins (up to 30 days), 7-day orphans,
manifests and payloads come on top (3.7.3). Idle and superseded copies add more, since step 1 never deletes a copy.
The figures below are single-copy estimates before those multipliers.

| Answer | What it costs | What follows |
|---|---|---|
| **a. Users bring their own bucket. We operate none. (recommended)** | Nothing for us. About $0.10 to $0.30 a month for a user at 6 GB. | We hold no user data and issue no credentials. The gate still needs one test bucket with two scoped keys from the owner: free at proof sizes on Cloudflare R2 or Backblaze B2 (each has 10 GB free). |
| b. We also operate one bucket, one prefix per user | About $1, $9 and $110 a month at 10, 100 and 1,000 users on R2 for one copy, before the storage multipliers above; or about EUR 6.50 flat up to 1 TB on Hetzner Object Storage. | Credentials issued and revoked by hand for the first users. Decision O2 becomes live. The date holds only if the bucket exists by 12-07. |
| c. No real bucket; prove on MinIO only | Nothing. | The isolation bar is then proven on a local server, not on a host a user would use. Not recommended: the bar names a hosted store. |

What we need under the recommendation: one bucket and two prefix-scoped keys, created by the owner, by 12-07 for
the gate. If they exist by 11-21, the design check's upload row (row 3) uses them; until then row 3 waits. Lane 6's
live tests and the isolation proof also run on them, so the bucket is the first real test of our S3 client.

**O2. If we operate a bucket: what we say about reading it (privacy).** It only arises under O1-b.

| Answer | What it costs | What follows |
|---|---|---|
| **Say plainly that the operator can read the hosted copy, raw transcript lines included. No client-side encryption. (recommended, if O1-b is chosen at all)** | Nothing to build. | Some users will not opt in to our bucket; they can use their own. |
| Encrypt on the user's machine before upload | One more High lane, about a week; key handling on every machine. | A lost key makes the hosted copy unreadable, which the restore proof would then have to cover. It also rules out a hosted reader or writer in step 2 without a redesign. |

Recommendation for both together: launch on O1-a, decide O1-b and O2 when there are users asking for it.

**O3. What the user's other machines and the bucket hold (privacy).** It arises under every O1 answer, as two
separate questions.

**O3a. A lost machine.** A lost or stolen machine exposes the history of every one of the user's machines it holds
a replica of, not only its own. Today a lost machine exposes only its own store.

| Answer | What it costs | What follows |
|---|---|---|
| **Accept it and state it in the opt-in text. (recommended)** | Nothing to build. | Step 1 as designed. |
| Keep replicas off until step 2's hosted reader; upload only | Recall across machines waits for step 2. | Step 1 becomes an off-machine backup with a proven restore. A replaced machine still gets its history back, by restore into a fresh folder. |

**O3b. Raw lines in the bucket.** The hosted copy carries the raw transcript lines (`capture_events.raw`),
including text that the redaction opt-in removed from the searchable rows. VISION.md:26 describes that opt-in as
discarding content by design; locally the raw line is already kept (docs/claude-code-adapter.md:618-619). Every
replica carries them too.

| Answer | What it costs | What follows |
|---|---|---|
| **Accept it and state it in the opt-in text. (recommended)** | Nothing to build. | The hosted copy is whole, and a restore is lossless. |
| Upload without `capture_events.raw` for users who decline | One more lane: the snapshot blanks the raw bytes in the staging copy, the guard and the copy checker skip them, and the restore proof checks the rest. | A restore then loses the raw history; the opt-in says so. "Nothing is lost" holds only on the machine. |
| Redacted text never leaves the machine, and no partial copy | Nothing to build. | A user who wants that does not opt in. |

## 10. What this record does not settle

- **Every size and every wait** is unmeasured until section 6 runs. The date assumes rows 1, 4, 5 and 6 pass
  by 11-24, with row 2D now the gate after row 2 fired its fallback; row 3 runs when the O1 bucket exists,
  before the gate.
- **A session-end publish.** A capture made after A's last upload waits for A's next session or a manual `sync`
  (4.5). Publishing when a session ends needs a hook, and D13 has none, so this stays open.
- **Conditional create on a user's host.** The chain and retention rely on a create-if-absent PUT. `enable` accepts
  only a host that lane 6 has shown honours it, and refuses every other host by name **(assumption: R2, B2 and
  Hetzner do; lane 6 checks each host it names, including that a second create of the same key is refused with 412
  and never answered with a 2xx)**.
- **TLS on found interpreters.** Some ship without a CA bundle; `enable` then refuses and says so.
- **Grok Build.** Covered by design only. The roadmap rows say so: Grok Build stores join once Grok capture ships
  and its own restore proof and copy checker pass.
- **Hermes stores** have no origin id and stay out.
- **A forget command for Claude Code.** Step 1 carries deletes that happen; it adds no way to make them.
- **Bytes of deleted rows in unused database pages and full-text index tables** (5.2). Step 1 measures and discloses. It does not scrub.
- **The `.records` sidecar.** Refused if present. Carrying it is future work when a capture store first has one.
- **Contract version 2.** The rules in 3.6 are untested until a second version exists.
- **Bucket settings we cannot see:** object versioning and lifecycle on a user's own bucket.
- **Replica disk use.** Each machine holds a full copy of the user's other machines. No eviction in step 1. A copy
  that held a forgotten session costs up to one more database on a filesystem without clones (V11).
- **Backfill** of sessions from before the plugin was installed, and **Windows.**
- **The merge into one store,** and any hosted reader or writer. That is step 2.
- **Cost at scale.** The O1 figures rest on an assumed 6 GB per user and an unmeasured share between generations.
- **An agent that strips an id's prefix** would reach a local row. The merger never emits a bare replica id; what
  a model does with one is measured by the agent arm, not prevented.

## 11. Rejected alternatives

- **Hold `capture.lock` during the snapshot.** The holder inherits the worker's rescan duty; a crash delays capture.
- **Snapshot inside the capture worker.** It changes the gated capture module; any bug becomes a capture bug.
- **`MessageStore.backup` for the snapshot.** Not contracted; runs on a live `MessageStore` connection (store.py:3427).
- **Compact the snapshot with `VACUUM`.** It defeats part sharing and does not clear full-text index remnants.
- **Row deltas instead of snapshots.** Needs an apply process on the host and its own cursor; a skip loses rows silently.
- **Re-ingest raw lines on the host.** Builds a hosted writer before step 2's single writer.
- **An outbox inside the ingest transaction.** Touches the gated ingest path.
- **Stamp `metadata.store_uuid` on the copy.** Writes a core-owned key; the merger's prefix makes it unnecessary.
- **Extend the capture audit with a store-only mode.** `audit()` reads transcripts by design (claude_code_audit.py:823); an independent checker is the precedent.
- **Two MCP servers and let the agent merge.** No duplicate rule, no deadline, and the proof would test the model.
- **A hosted reader over HTTPS or SSH.** Code and authorisation on the host; one bug exposes every user.
- **Rank fusion or score merge across stores.** Changes local ordering and invents a cross-store score.
- **Hide every session the local store also holds.** It hides replica rows the local store lacks.
- **One process serving all stores through `StoreReader`.** Kept as the fallback if four readers break the memory bar.
- **Hosted SQLite or Postgres (D1, Turso, Supabase, Neon).** None takes a SQLite file as is; Postgres is a second engine.
- **`boto3` or an external sync tool.** A dependency or a binary the stdlib-only unit would have to install and test.
- **Turning cloud off deletes the hosted copy.** One slip destroys the only off-machine copy.
- **Prune every older generation on an accepted loss** (draft 2's first edit). A loss caused by a bug, accepted with
  one command, would also destroy the hosted copy's only intact generation. Replaced by the pin (3.4).
- **Credentials in environment variables.** Claude Code's environment reaches the agent's shell, and the store
  would capture any `env` the agent runs.
- **Key the hosted copy by the origin id alone.** A home copied to a second machine would publish two diverging
  stores under one name, and its only way out would destroy one machine's history.
- **Swap a replica's files under a running reader.** The reader opens several connections per call, so one call
  could read two generations.
- **Drop a replica hit because another store holds the same occurrence.** That store may not have returned it, so
  the only copy in the response would vanish.
- **Restore into the live home.** See D9.

For base-plus-delta upload (3.7), these alternatives are also rejected:

- **Smaller parts.** 64 KiB re-sends about 0.75 GB a day. At 16 KiB, 16 publishes send about 172 MB of parts
  (62% of the 278 MB term) but about 85,000 entries per base make a manifest about 18 MB at the prototype's
  215 bytes per entry, about 292 MB a day on top. At 48 publishes, parts alone send about 515 MB.
- **Cumulative deltas since the base.** Re-sending about 800 additional distinct pages per busy bin costs
  about 210 MB on day one and 610 MB on day two, over the bar.
- **Rolling re-base.** Smooths size/length bases, but a forget still needs every part; adds per-part generations
  and a daily budget. The ordered manifest list could admit it later.
- **Daily roll-up deltas.** Send each day's pages twice and need a second local page list; 1,024 small objects
  cost less.
- **Page lists in the bucket.** 8 to 25 MB per generation, more than deltas; a lost local list costs one base.
- **Per-page content objects.** About 1,500 PUTs per publish and 340,000 objects per base.
- **Row deltas, WAL shipping, SQLite's session extension.** Row deltas have the earlier objection above; WAL
  shipping controls live checkpoints against D4; the session extension is absent from stdlib `sqlite3`.
- **Bridge delta on a base generation.** Helps only a replica within two publishes of re-base and adds a second
  path; revisit if replica downloads matter.
- **Chain age cap.** Re-sends an unchanged store; the missing-object rule already heals an incomplete chain.
- **Compressed base parts or `lzma`.** Compressing parts is separate work; some builds lack `lzma`.

## Appendix A. The critic's objections to draft 1, each with its disposition

| Critic item | Disposition | Where |
|---|---|---|
| 1.1 Snapshot design contradicts "the hosted copy keeps what was deleted" | Adopt. Deletes reach the hosted copy; the opt-in says so. | 3.5, 5.2 |
| 1.2 Snapshot mirrors the payload-GC rewrite; guard checked identity only | Adopt the guard change: it compares content. The rewrite has no production caller on main, so the exclusion rule is dropped. | 3.4 |
| 1.3 Replica id changes at every refresh; `ForeignStoreError` | Adopt the problem, change the fix: the merger owns the prefix and passes bare ids. | 4.3 |
| 1.4 Two host contracts (SSH reader against files-only) | Adopt: files only. | D2 |
| 1.5 Hosted databases cannot take a snapshot; the object-only cost assumed base plus segments, not whole snapshots | Adopt both halves: hosted databases are rejected, and the cost figures now say they can be up to three times higher until part sharing is measured. | 11, 9, 6 row 2 |
| 1.6 Two different uuids under one word | Adopt: one id, defined once. | Terms, 3.2 |
| 2a `ensure_store_uuid` has no production caller | Adopt; re-verified on main. | 3.2 |
| 2b `admit()` passes a home copied to the same path | Adopt: the manifest chain detects a second publisher. | 3.4 |
| 2c The reader opens a restored or moved store | Adopt as a fact the restore check relies on. | 5.3 |
| 2d Deletes with no marker | Adopt: the guard refuses; `accept-loss` records it outside the store. | 3.4, 5.2 |
| 2e Confirmed facts | Kept only where still used, with new line numbers: `event_id` (claude_code_store.py:69), exclusion list (tools.py:378-394), recall budget (config.py:1471). The rank-fusion facts are no longer used. | 3.2, 4 |
| 3 "Claude Code and Grok Build": Grok not delivered | Adopt: evidence covers Claude Code; the roadmap rows say so (sixth edit, Appendix H). | D3, 7, 10 |
| 3 "Reachable only by its own user's agents": host not chosen; merged endpoint against "one profile's store" | Adopt: owner decision O1 names the host; the test is defined; the roadmap sentence is amended. | 5.1, 4.6, 9 |
| 3 "One proven restore": no host, no runner, no upload path | Adopt: lanes 2 and 5 come first; the proof and its runner are named; `enable` repeats it per user. | 5.3, 8 |
| 3 "0 lost and 0 duplicate": no definition; recall hides rows; who runs it; store-only audit | Adopt: definitions in 4.2 and 7; row-level duplicate rule; each origin machine runs its link; no audit change. | 4.2, 7 |
| 3 "One real recall proof" shows only the second-home case | Adopt: said plainly; the replaced-machine run is added. | 1, 7 |
| 3 "Opt-in": no writer for the two files | Adopt: `enable`. | D11 |
| 4 Restore into the live home | Adopt as a rule with a test. | D9, 5.3 |
| 4 Outbox inside ingest | Adopt: rejected. | 11 |
| 4 Long read locks on a rollback-journal store | Adopt: stepped copy, measured against capture, with two fallbacks. Draft 1's "small page steps" was an assumption and is now row 1. | 3.3, 6 |
| 4 A delete script pointed at a live home | Adopt: no check script accepts a live home. | 6 |
| 4 Marker-less deletes | As 2d. | |
| 5 The three questions | Answered by D1, D2 and D8. | 2 |

## Appendix B. What changed from draft 1

- **Built on contract v1.** Snapshot, guard, checker and merger read contracted tables only, plus SQLite's own `sqlite_sequence` for the
  guard's id check; lane 1 notes that read in the contract's prose, which its digest does not cover. No contract
  change.
- **The snapshot no longer assumes it is harmless to capture.** It takes no lock, copies in steps, and is measured.
- **The copy is normalised** to rollback-journal mode before hashing; the `.records` sidecar is handled by refusal.
- **Guard and measurement are split.** The guard is product code; an independent checker gates.
- **Id stability moved to the merger.** No `metadata.store_uuid` is written.
- **Duplicates are decided per occurrence,** not by hiding whole sessions.
- **The wait bar is relative** to local-only, measured with the GA recall kit, for 1, 2 and 4 stores.
- **Parts named by content** replace one snapshot object, so uploads resume and generations share bytes.
- **`enable` proves a restore per user.** A replaced machine gets its old history back as a replica.
- **Dropped:** the payload-GC exclusion rule; the store-only audit mode; the keychain; the reader-stamp check.
- **Owner decisions: three.** The lost-machine exposure stays with the owner (O3), together with the raw-line
  upload. Draft 2's first edit had moved the exposure to the release manager (D16); both reviewers objected.
- **Every kept citation was re-read** on `43e8484c`; draft 1's line numbers had drifted by up to several hundred lines.

## Appendix C. Review of the first edit, each finding with its disposition

Reviewers: `gpt-6-astra` at high (all eight design-blocking findings checked and found real by a separate verify
pass) and a fresh Claude reviewer (`claude-opus-5-5`, high). Both: NEEDS CHANGES.

| Finding | Class | Disposition | Where |
|---|---|---|---|
| Astra 1, scout 1: no arbitration between two publishers or several mergers; retention could delete another upload's parts; one shared staging folder | design-blocking | Adopted: `cloud/upload.lock` for uploads, refreshes and retention; conditional create of each manifest; a part check before it; `chain_conflict`; interleaving tests; memory measured at four sessions | D13, 3.3, 3.4, 3.5, 5.2 (f), 4.4, row 4 |
| Scout 1: a copied home's only exit was `accept-loss` | design-blocking | Adopted: the copy id, `enable --new-copy`; a chain conflict never offers `accept-loss` | Terms, 3.2, 3.4, 4.1, 4.3, 7 |
| Astra 2: forgotten-data pruning not retried after a crash | design-blocking | Adopted: every publish prunes any older generation that names a forgotten session; kill test | 3.5, 5.2 (a) |
| Scout 2: `accept-loss` destroyed the off-machine copy of what it accepted losing | design-blocking | Adopted: plain `accept-loss` pins the generation before the loss for 30 days; `--as-delete` for a meant delete | 3.4, 3.5, 5.2, 11 |
| Astra 3: the guard trusted `line_digest`, not the raw bytes; bindings and relations unchecked | design-blocking | Adopted: digest of the decompressed raw line; bindings and relations go with their rows; the copy checker covers them | 3.4 |
| Astra 4: the duplicate rule could drop the only returned hit and missed copies externalized under other names | design-blocking | Adopted: content identity with placeholder fingerprints; drop only what this response already emitted | 3.2, 4.2, row 4, 11 |
| Astra 5: a replica swap could mix generations within one call | design-blocking | Adopted: per-generation directories, new reader, drain, then delete | 4.1, row 5, lane 5, 11 |
| Astra 6: restoring the older generation had no oracle of its own | design-blocking | Adopted: a saved origin copy and transcripts per generation | 5.3, 7 |
| Astra 7: 50 ms cited as the unit's bar | design-blocking | Adopted: a target, not gated; the row's bar is relative; over 50 ms is reported to #2141 | row 1 |
| Astra 8, scout 9: D16 and the raw-line upload widen privacy promises | design-blocking | Adopted: owner decision O3 | D16, 3.1, 5.1, 9 |
| Astra 9: bucket date against row 3; lane 10's dependencies | non-blocking | Adopted: row 3 gates only the gate run; lane 10 needs 4, 6, 9, row 3, O1 and O3 | 6, 8, 10 |
| Scout 4: the isolation test could pass on our own client's refusal | non-blocking | Adopted: our prefix check bypassed; the host must answer 403 | 5.1, 7 |
| Scout 5: MinIO named in one place, avoided in another | non-blocking | Adopted: no MinIO run; recorded-response tests, then the O1 bucket | 5.1, lane 6 |
| Scout 6: credential file readable by others; environment variables leak into the store | non-blocking | Adopted: file readable and writable only by its owner; no environment option; planted-value grep extended to the store; manifest names checked before use as paths; TLS verified | 5.1, 10, 11 |
| Scout 7: row 5's bar equalled the merger's own wait; row 1 had one control; four writers in a week capped at three | non-blocking | Adopted: the wait is 250 ms; row 1 runs the control twice; the week of 11-30 is stated as four writers; the review risk to 12-29 is stated | 4.4, rows 1 and 5, 8 |
| Scout 8: row 2 did not bound a day of 30-minute publishes | non-blocking | Adopted: row 2 measures a day of publishes and bounds their total | row 2 |
| Scout 10: the opt-in did not say a Hermes clean apply is refused at upload | non-blocking | Adopted | 5.1, 5.2 |
| Scout R1: D6 and D3 citations imprecise | nit | Fixed: store.py:3427 (a live connection) and claude_code_store.py:50 | D3, D6 |

## Appendix D. Review of the second edit, each finding with its disposition

Reviewers: the same two (`gpt-6-astra` at high; the same fresh Claude reviewer, resumed). Both: NEEDS CHANGES.

| Finding | Class | Disposition | Where |
|---|---|---|---|
| Astra 1: retention on one machine could delete parts of another machine's publish under the same copy id | design-blocking | Adopted: the publisher chain allows one publisher per copy id, so retention can delete only parts a refused publish left | 3.4 Chain |
| Astra 2: two machines taking turns under a copied home were not caught; the second met a loss refusal | design-blocking | Adopted: each manifest names its publisher (copy id plus hardware id); the head must be this machine's last | 3.4 Chain |
| Astra 3, scout 1: a later forget pruned a pinned generation, and the rows a bug lost with it | design-blocking | Adopted: a pinned generation is never pruned by a forget; `pin_holds_forgotten`; expiry warnings | 3.5 |
| Astra 4: an accepted loss or a rollback lets SQLite reuse ids, so an issued id could resolve to another row | design-blocking | Adopted: published high-water mark per copy id; `store_id_reused` refusal; exit `--new-copy` only | 3.4 |
| Astra 5, scout 3: one merger could delete a generation folder another session still reads; no bound on retirement or disk | design-blocking | Adopted: pointer file, probe call, shared folder locks, retirement at the tool budget, disk floor for refresh, row 5 with two mergers | 4.1, row 5 |
| Astra 6: bindings and relations could vanish alone and pass the guard | design-blocking | Adopted: both compared like messages, in the per-session digests | 3.4 |
| Astra 7: `held` file entries have no bytes, so the raw-line rule would refuse them | non-blocking | Adopted: checked through their digest and the rows they name | 3.4 Ledger |
| Astra 8: an inline copy and an externalized copy of one output had different identities | design-blocking | Adopted: identity expands placeholders to their payload text | 3.2, 4.2 |
| Astra 9: O3's "upload only" still uploaded raw lines | non-blocking | Adopted: O3 split into O3a (lost machine) and O3b (raw lines), each with its own answers | 9 |
| Scout 2: `--new-copy` on the wrong machine orphaned a hosted copy | design-blocking | Adopted: allowed only on the machine whose own publish conflicted | 3.4 Chain |
| Scout 4: the lock did not cover every bucket write; a hung holder was invisible; the handover could switch before the new reader answered | nit | Adopted: every bucket-writing command takes the lock; timeouts; lock age in `status`; switch after a probe call | 3.5, 4.1 |
| Astra 10, scout Q3: MinIO still named in D14; section 11's backup wording; the storage estimate ignored pins; the gate map named four tests; lane 8's scope; the store list's source; the outside-in checker's inputs | nit | Fixed | D14, 11, 9, 7, 8, 5.3 |

## Appendix E. Review of the third edit, each finding with its disposition

Reviewers: the same two. `gpt-6-astra` at high: APPROVE, with six non-blocking findings. The fresh Claude reviewer
(resumed): NEEDS CHANGES, with one design-blocking finding. Both found the same wrong citation. This was the third
review round, so the release manager ran the delivery checkpoint: the one design-blocking finding has a bounded fix
that its reviewer specified, and is fixed now; the rest are folded as text or left to the named lanes' tests. The
changes go back to the same two reviewers as a delta only.

| Finding | Class | Disposition | Where |
|---|---|---|---|
| Scout 1, Astra 1: the id-reuse rule cited a scratch table (db_bootstrap.py:2385-2386); the live column is AUTOINCREMENT (store.py:870-871), so the rule refused an ordinary forget of the newest session | design-blocking (scout); non-blocking (Astra) | Fixed: refused on a `sqlite_sequence` below the mark, an id of N holding a different row, or a removed id holding any row; missing rows go through the loss and forget rules; tests (h), (i) | 3.4 Messages, Ids |
| Scout 1, Astra 2: the only exit from `store_id_reused` was refused on the machine that needed it | design-blocking (scout); non-blocking (Astra) | Fixed: `--new-copy` is that exit; the old copy becomes a superseded copy | 3.4 Ids, 3.5 |
| Scout 2, Astra 3: a conflict from a lost `state.json`, a changed hardware id or a stop after the manifest left this machine's own copy without a publisher after `--new-copy`; forgets and pins never reached it | non-blocking | Adopted: `enable --resume` under the whole guard against the head; `--new-copy` refused while `--resume` would pass; superseded copies pinned for 30 days, then pruned by forgets, read as replicas, removable with `delete-hosted --copy`; test (j) | 3.4 Chain, 3.5, 4.1, lanes 3, 8 |
| Scout 3, Astra 4: content identity undefined for a missing or unreadable payload | nit (scout); non-blocking (Astra) | Adopted: the placeholder's own fingerprint, marked as a fallback; never drops a hit; computed once per generation | 3.2 |
| Scout 4: binding check columns are updated in place | nit | Adopted: bindings compared on their identity columns | 3.4 Bindings |
| Astra 5: complete-removal promises contradicted the pin exception | nit | Fixed: the exception stated in 3.5 and 5.2; test (a) runs with no pin; test (g) | 3.5, 5.2 |
| Scout Q3, Astra 6: row 5's disk bar contradicted 4.1 | nit | Fixed: twice the replica, three times while another session reads the oldest generation | row 5 |
| Release manager, while fixing scout 1: a rolled-back store could hold a forgotten session again and upload it, also into a new copy id | design-blocking | Adopted: manifests carry removed sessions forward; `forgotten_session_returned`; a new copy id starts with the old one's forgotten sessions; test (i) | 3.4 Ids |

## Appendix F. Review of the fourth edit (a delta), each finding with its disposition

Reviewers: the same two, on the changed text only. The fresh Claude reviewer: APPROVE, with one non-blocking
finding and five nits. `gpt-6-astra` at high: NEEDS CHANGES, with one design-blocking finding. The release manager
verified it and wrote a delivery checkpoint before this edit. The fourth edit had added a writer to a copy id, and
this is the third time the retention race reopened that way. So the takeover is removed rather than fenced. If the
confirm round finds another blocking item in this cluster, there is no sixth round: step 1 ships with every chain
conflict ending in `--new-copy`, and each recovery convenience moves to a later lane.

| Finding | Class | Disposition | Where |
|---|---|---|---|
| Astra 1: `--resume` let a second publisher take over while the first ran retention, so the takeover's manifest could name a part the first had just deleted | design-blocking | Fixed by removal: `--resume` only where the head names this machine's current publisher; every other conflict exits through `--new-copy`; one invariant (nothing writes or deletes under another publisher's copy id), checked by every bucket-writing command; test (j) | 3.4 Chain, invariant |
| Astra 2: pins and superseded copies were recorded only in local files | non-blocking | Adopted: every later manifest carries the pin; the new copy's manifests list superseded copies | 3.4, 3.5 |
| Astra 3, scout 6: `sqlite_sequence` is not a contracted table | non-blocking; nit | Adopted: stated as the one non-contracted read; lane 1 notes it in the contract's prose | 3, Appendix B |
| Astra 3, scout 5: the gate's "no copy id published by two machines" contradicted the takeover | non-blocking; nit | Closed by the removal; the line cites the invariant | 7 |
| Astra 4: the carried removed ids could grow without bound | non-blocking | Adopted: a compressed id set; lanes 2 and 3 measure its size and the guard's time on a long, sparse history | 3.4 Ids |
| Scout 1: a copied home's `--new-copy` inherited the other machine's forgotten list | non-blocking | Adopted: the list is carried only on the `store_id_reused` path; test (j) | 3.4 Ids |
| Scout 2: superseded-copy pruning after a changed hardware id | nit | Adopted: pruning runs only while the head names this machine's current publisher; otherwise the copy is idle and `status` says so | 3.5 |
| Scout 3: superseded copies are extra stores | nit | Adopted: each counts as a store in 4.4's bars | 3.5 |
| Scout 4: 3.5's duplicate wording contradicted 4.2 | nit | Fixed: 3.5 defers to 4.2 | 3.5 |
| Release manager: a copy left idle by a changed hardware id is no longer pruned by forgets | non-blocking | Stated limit: `status` lists idle copies and the opt-in says a forget does not reach one until the user deletes it; a safe prune is a lane 8 item | 3.4 |

## Appendix G. Final review (the fifth edit), each finding with its disposition

Reviewers: the same two, on one question: does the round-4 interleaving still exist, and does any command write
to, or delete under, a copy id whose head names another publisher? Both answered no. Both: APPROVE.

| Finding | Class | Disposition | Where |
|---|---|---|---|
| Scout 1, Astra 1: two machines that share a hardware id (a cloned VM image keeping its `/etc/machine-id`) compute one publisher, hold separate locks, and `--resume` would let both write under one copy id | non-blocking | Lane 2 tests it on each supported platform: two homes with one machine id and a copied home; `--resume` must be refused, and no manifest may name a missing part. Where the id cannot be shown unique, `--resume` does not ship there and every chain conflict ends in `--new-copy` (the fallback named in the round-4 checkpoint). Without `--resume`, such machines are fenced by the publish token (3.4); this sentence was corrected in the sixth edit (Appendix H). | 3.4 Chain, lane 2 |
| Scout 2: a plain `enable` re-run on a home whose `cloud.json` names a copy id was not in the invariant's command list | nit | Lane 8 names it and tests it | lane 8 |

## Appendix H. Bot review of the docs PR and lane 0's first results, each finding with its disposition

The docs PR's bots found that the fifth edit's chain rule did not fence a copied home or a cloned VM, because both
machines pass the last-or-pending test; the publish token fences them independently of the hardware id.

| Finding | Class | Disposition | Where |
|---|---|---|---|
| T1 (Codex connector): a cloned VM with the same hardware id passes the last-or-pending test without `--resume`, so two machines keep publishing and one's retention can delete parts the other's next manifest names | design-blocking | Fixed: a per-attempt publish token fences every bucket-writing command; a part or payload the head does not name goes under a key unique to the attempt; retention spares never-named keys for 7 days by the bucket's clock; Appendix G's last sentence corrected; test (j), row 3 | 3.1, 3.3, 3.4 Chain, 3.5, 5.2, G |
| T2 (NeonDiff): `exclude_session_ids` filters search only, so a direct `lcm_expand` of a replica id still returns a forgotten session | design-blocking | Fixed: the merger enforces forgetting for every reader tool, fails closed on a result with no session, and answers as for a missing id; test (i) | 4.2, 4.3, 5.2 |
| T3 (NeonDiff): a copied home that shares a hardware id passes the ordinary publish check, and the original's retention can delete the copy's parts | design-blocking | Fixed with T1 | 3.4 Chain |
| T4 (NeonDiff): a forget never reaches a superseded copy's head | design-blocking (as text) | Disclosed: the head stays until `delete-hosted --copy`; `status` reports `superseded_holds_forgotten`; the merger hides those sessions; the opt-in says so; a prune that keeps unrelated rows is a lane 8 item; D8 and test (i) changed | 3.5, 5.1, D8, 5.2 |
| T5 (NeonDiff): restore check 4 needed a payload, which a store without one lacks | doc-fix | Fixed: the payload expand only when the manifest names a payload, otherwise one message by `store_id`; with no rows, the reader's start and one empty `lcm_grep` | 5.3 |
| T6 (CodeRabbit): a referenced payload could change or vanish between the database copy and hashing | doc-fix | Fixed: referenced payloads are cloned or hard-linked into staging right after the database copy and uploaded from there; a reference that names no file is unresolved and never stops a snapshot; a missing payload the head names is a loss; forgotten markers are read again before the manifest | 3.1, 3.3, 5.2 (a) |
| T7 (CodeRabbit): `enable` and `disable` could overwrite or delete another tool's `reader-override.json` | doc-fix | Fixed: an owner key; `enable` refuses a foreign file before any bucket write; `disable` leaves it and says so; D11 no longer claims to be its one writer | D11, 4.1, 5.1 |
| T8 (CodeRabbit): the freshness bound held only while a session runs on the origin | doc-fix | Fixed: the condition is stated; a session-end publish is an open item, since it needs a hook | 4.5, 10 |
| T9 (CodeRabbit): restore check 3 has no input without saved transcripts | doc-fix | Fixed: check 3 runs only with a saved copy of every transcript as of the cut, otherwise `check3_not_run`; a failed check that runs stops the restore; the gate proof keeps all four checks | 5.3 |
| T10 (CodeRabbit): the roadmap put Grok Build in step 1's scope | doc-fix | Fixed: the roadmap rows say Claude Code, and Grok Build stores join once Grok capture ships and its own restore proof and copy checker pass; D18 amends only ROADMAP.md:125 | D18, 10, Appendix A, ROADMAP.md |
| L0-1: quoted placeholders. Real sessions quote placeholder-shaped strings that name no file, and the first two real snapshot attempts stopped on them | doc-fix | Fixed: such a reference is counted as unresolved; it never stops a snapshot and never makes two stores unequal | 3.1, 3.3, 5.3, 7 |
| L0-2: full-text remnants after a forget. A planted id stayed in the full-text index's backing tables, even after `VACUUM` | wording | The stated limit names the full-text index's internal tables; the real-store run is still to come | 5.2, 10 |
| L0-3: real sizes | measurement | Recorded: the raw ledger is 42 to 50 percent of the database in two real stores of 2.5 and 2.75 GB; a snapshot of the 2.75 GB store matched every contracted table and payload | 6 (inputs to rows 1 and 2; no text change) |

## Appendix I. Round 6 confirm review, the delivery checkpoint, and the seventh edit

Both reviewers found that a home copied during a publish shares the pending pair, so the fence let its twin run
`delete-hosted`, or delete the other machine's live copy as its own superseded copy, while the other machine still
used the keys. It was the fourth time the copy-identity rules reopened, and each earlier fix had added protocol.
The delivery checkpoint (release manager, 2026-11-16) cut step 1's scope instead: step 1 has no command that
deletes a copy, no `--resume` and no pins on superseded copies, and retention after its own winning create is the
only thing that deletes. Rows of Appendices E to H that name `delete-hosted`, `--resume` or a superseded copy's pin,
and Appendix H's row L0-1 on references that name no file, describe earlier edits; this appendix supersedes them.

| Finding | Class | Disposition | Where |
|---|---|---|---|
| Astra 1, scout 1 (first case): a home copied during a publish holds the same pending pair, so its twin passes the fence for `delete-hosted` and the superseded-copy path, and the other machine's next manifest can name a deleted key | design-blocking (Astra); Important (scout) | Fixed by removal: `delete-hosted` and `--resume` are out of step 1; a superseded copy is never pinned or pruned; `accept-loss --release` only records; retention runs only after its own winning create with that manifest at the head; whole copies are deleted in the host's console | 3.4, 3.5, 5.1, 5.2 (j), lanes 2, 3, 8 |
| Astra 2: a payload lost before its first publish was counted unresolved and published without anyone being told | design-blocking | Fixed: two counts (`unresolved_minted`, `unresolved_quoted`) that `status`, the restore check and the copy checker report, never passed as equal. Not refused: the hosted copy never held the file, and a refusal by name shape would also stop on names quoted from another installation (lane 0's real store held two among five) | 3.1, 3.3, 5.3, gate map |
| Scout 1 (second case): an attempt paused past retention can create a deleted manifest number that names a deleted key | Important | Restated: such a manifest is below the head, never read, runs no retention, and the next retention deletes it; the invariant speaks of kept manifests; test (f) | 3.4, 3.5, 5.2 (f) |
| Astra nit: seven days is when a key becomes eligible, not when it is deleted | nit | Fixed: at least 7 days, until the next successful retention | 3.5, 5.2 |
| Scout nits: an empty `session_id`; the superseded-copy pin and pruning text contradicted itself; row 6's real-store run was undefined; quoted and product-shaped references shared one count | nit | Fixed: an empty `session_id` counts as missing; the pin and pruning text is gone; row 6 runs on a scratch store; two counts | 4.2, 3.5, 5.2, 3.3 |
| Scout's suggested fixes: the pending record keeps the built manifest's digest, and the head is read again just before the manifest's create | follow-up | Lane 3 tests both as candidates | lane 3 |

### Review of the seventh edit

Both reviewers re-read the seventh edit's delta. The fresh Claude reviewer approved with Important wording items;
`gpt-6-astra` asked for changes. Every item is folded below as wording, a stated limit or a named lane test, as the
checkpoint required; none adds protocol. Where this table and the one above differ, this table holds.

| Finding | Class | Disposition | Where |
|---|---|---|---|
| Astra 1: in a home copied during a publish, two stalls (the other copy's retention between its head read and its deletes, and this copy's attempt across that delete) can create a head naming a key deleted as never named | design-blocking (Astra) | Stated limit, with a lane 3 test of the exact interleaving. No local row is lost; a restore of that generation fails visibly, and so does a replica refresh, which keeps the previous generation serving; the generations before it stay kept unless a forget pruned them; the newest generation is whole again if that copy publishes again, and otherwise the hosted copy stays idle with a newest generation that fails visibly. Removing the deletion of never-named keys would close it, but would leave a forgotten session in the bucket indefinitely after any forget that lands during an upload, on a single machine. A blind adjudication against four pre-declared bars (no lost row, the forget promise, no new protocol, no silent residual) found that both options pass and chose this one as the smaller residual. | 3.4, 5.2 (j), 5.3 |
| Astra 2: "never read" was false, since `restore --generation` can name a recreated manifest | Important | Fixed: such a manifest is not the head, so no replica or guard reads it and it runs no retention; a restore of it by number fails visibly on a missing object; lane 5 test | 3.4, 5.2 (f), 5.3 |
| Astra 3, scout: Appendix E still described the removed commands | nit | Fixed: the supersession covers Appendices E to H and Appendix H's row L0-1 | this appendix |
| Scout: "own conditional create" left room to run retention on a resumed attempt whose create is refused | Important | Fixed: retention runs only after this attempt's create wins; a refused resumed create runs none; tests (a) and (j) | 3.3, 3.4, 3.5, 5.2 |
| Scout: nothing said whether unresolved counts pass the restore check and the copy checker | Important | Fixed: the same unresolved names on both sides pass and are reported, and one-sided ones fail; lanes 4 and 5 test both, and that a store holding only quoted placeholders passes D12 | 5.3, gate map |
| Scout: a recreated manifest inside the kept window can be kept and named for restore | Important | Narrowed: it is not the head, runs no retention, and fails visibly on restore when it names a missing object; test (f) | 3.4, 5.2 (f) |
| Scout: a copy taken during a publish can adopt a generation holding its twin's later rows, and then meets `store_id_reused` or a loss refusal | Important | Stated, with test (j): every copy but one meets `chain_conflict`, `store_id_reused` or a loss refusal; nothing deletes outside its own winning retention; `status` never calls the other copy safe to delete | 3.4, 5.2 (j) |
| Scout: the opt-in named idle copies too narrowly; O1's storage figure left out idle copies; lane 2 measured hardware-id uniqueness that only a later `--resume` needs | Important; nits | Fixed: the opt-in covers every idle copy; O1 counts idle and superseded copies; the measurement moves to that later lane | 5.1, O1, lane 2 |

## Appendix J. Review of the page-delta design

The base-plus-delta design was reviewed separately from the whole-copy record in four rounds.
Each round used one fresh Claude review and `gpt-6-astra` at high, with delta passes after the first round.
The release manager's ruling closed the design.
Section 3.7 states its decided rules, including all four rounds' corrections.
The object format, page-list digest and rebuild checks are shared specifications, with an independent decoder.
The final forget rule keys each decision separately and rechecks live items just before the manifest.
A racing forget restarts a delta once into a base; a rolled-back pending delete refuses rather than loops.
Cumulative base-generation cutoffs make pruning retryable, without pruning for sessions the chain never held.
Retention deletes exclusive objects before manifests and records completion only after successful deletes.
The daily bar counts all uploads and includes a 32 MiB floor; row 2D gates both growth and no-growth runs.
Its sizes use a transcript replay on the shared development machine; timing uses the bench hosts.
Manifest scaling, payloads, size bases and length bases all enter the 90-day simulation.
Lane 2d separates the codec from the uploader; section 8 carries the dependencies and revised date.
This fold is documentation evidence only. It proves no codec behavior, upload bar, restore time or release gate.

## Appendix K. Hosted copy format 1 (F1 to F13)

This appendix is the format text the hosted-copy code was built from. It is the whole agreement between the code
that makes a hosted copy (`cloud_copy.py`) and the code that checks one (`scripts/cloud_copy_outside_in.py`).
F7 to F11 are also the agreement of the page-delta codec (`cloud_delta.py`). The two sides were written by
different authors who never read each other's work, so a sentence here that can be read two ways is a defect in
this text: report it rather than pick a reading. Code comments cite these labels.

Known differences, each tracked:
- the codec has refusal codes this text does not list, `start_mismatch` and `output_exists`; section 3.7 lists
  them;
- this text has no reader refuse on F8's writer limits (F8, F9), and neither merged reader does. #2800 proposes
  that the codec refuse, before fetching, a part or object reference over them (`bad_database_block`). If it
  lands, F8 to F10 change with it, so that every reader refuses the same references;
- F12 does not yet cover `--new-copy`: section 3.7.3 gives inherited entries that copy's first base generation.
  F12 gains the rule with the lane that builds `--new-copy` (#2801).

**F1. Names.** Hex digits are lower case. A payload name keeps its case.
- copy id: a UUID with dashes, `[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}`. Its 32-hex form is
  the same with the dashes removed.
- sha256: `[0-9a-f]{64}`. token: `[0-9a-f]{32}`. generation: `[1-9][0-9]*`.
- payload name: a file name of 1 to 255 bytes with no `/`, no `\`, no NUL, and not `.` or `..`.
- A name read from a manifest, a pointer file or a directory listing is matched against its grammar before it
  becomes part of a path. A name that fails is a refusal, `bad_name`.

**F2. Folders.**
- **Origin home:** `lcm.db`, `lcm-large-outputs/` (payload files), `capture-owner.json` (its `store_uuid` is the
  origin id; claude_code_store.py:49, :180-183).
- **Copy home** (a restored generation, or one generation of a replica): `lcm.db` and `lcm-large-outputs/<payload
  name>`. It has no `capture-owner.json`. Other entries are ignored.
- **Replica:** `<home>/cloud/replicas/<copy id>/gen-<generation>/` is a copy home and is never modified.
  `<home>/cloud/replicas/<copy id>/current` holds the generation number in ASCII digits, with an optional
  newline. Beside them, in `<home>/cloud/replicas/<copy id>/`: `view-<generation>-<rules>-<e>/` (a copy home, never
  modified once renamed), `view.json`, `view-result.json`, `removed.json` and `view.lock` (F13, 4.7).
- **Own copy id:** `<home>/cloud/cloud.json` is a JSON object whose `copy_id` names this home's own hosted copy.
  Only `enable` (a later step) writes it. Other keys are not fixed here.
- **Working files:** `<home>/cloud/staging/` and the lock `<home>/cloud/upload.lock`.
- **Bucket,** under `<prefix>/<copy id>/`: `parts/<sha256>.<token>`, `deltas/<sha256>.<token>` (F8),
  `payloads/<sha256>.<token>`, `gen/<generation>.json`. The sha256 is of the object's bytes. The token is the
  publish attempt that uploaded it, which can be an earlier generation's.

**F3. References.** A reference is a payload name that stored text points at. Both tools find references the
same way:
1. Read `messages.content` and `messages.tool_calls` of every row, by column name, where the value is text
   (Python `str` through `sqlite3`'s default decoding; a value that does not decode is a refusal,
   `undecodable_text`).
2. Apply these two patterns, each on its own with `finditer`, with no flags, and take group 1 of every match:
   - `\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]`
   - `\[Externalized LCM ingest payload:.*?;\s*ref=([^;\]\s]+)\]`
3. Each distinct captured string is one reference. It is:
   - **resolved** when it is a payload name (F1) and `lcm-large-outputs/<name>` in that home is a regular file,
     not a symlink;
   - otherwise **unresolved minted** when it fully matches `\d{8}_\d{6}_.+_[0-9a-f]{12}_[0-9a-f]+\.json` and is a
     payload name;
   - otherwise **unresolved quoted** (for example `...` or `{ref}`).
4. No other table, column or pattern adds a reference. A payload file that no reference names is not part of the
   copy.
(Sources in the code when this was written: externalize.py:31-33, :123; ingest_content.py:90. Real stores quote
placeholders: one real store held five unresolved references, two of them minted names from other stores.)

**F4. The import stamp.** One row in the copy's `metadata` table, key `import:origin`. Its value is the JSON of
this object (`sort_keys=True`, separators `(",", ":")`, `ensure_ascii=False`). The live store never gets it.

| Field | Type | Value |
|---|---|---|
| `format` | int | 1 |
| `origin_id` | string | `store_uuid` in the store's `capture:owner` metadata row |
| `copy_id` | string | copy id |
| `origin_harness` | string | `host` in the `capture:owner` row |
| `generation` | int | this generation |
| `parent` | null or object | `{"generation": int, "manifest_sha256": sha256 of that manifest's stored bytes}` |
| `contract` | object | `{"version": 1, "digest": "sha256:76644a990e50b9613697704dcb05b349a952d44ea22187bd768b0eab9a9badc9"}` (docs/store-contract.md:3-4) |
| `schema_version` | int | the store's own `metadata` value for key `schema_version`, as an integer; 0 when the row is absent |
| `projection_versions` | list of strings | distinct `capture_events.projection_version`, sorted; `[]` when the table is absent |
| `unit` | object | `{"version": string or null, "core_commit": string or null, "core_version": string or null}` |
| `max_store_id` | int | highest `messages.store_id`, 0 when there is none |
| `created_at` | number | seconds since the epoch on the publishing machine |

The copy gets no `metadata.store_uuid` row and no other new row.

**F5. The manifest.** `gen/<generation>.json` is one UTF-8 JSON object. Key order and whitespace are free; its
digest is the sha256 of the stored bytes. A reader ignores keys it does not know. It refuses a `layout` above 1
(`newer_layout`), and a `stamp.contract.version` above its own constant (`replica_newer_contract`).

| Key | Type | Value |
|---|---|---|
| `layout` | int | 1 |
| `copy_id`, `generation` | string, int | as in the stamp |
| `created_at` | number | as in the stamp |
| `stamp` | object | equal to the F4 object in the copy |
| `database` | object | the block of F9: this generation's file, the base it rests on and every delta after that base |
| `payloads` | list | `{"name", "sha256", "bytes", "key"}` for each resolved reference, sorted by name; sha256 and bytes are of the file; `key` is `payloads/<sha256>.<token>`. |
| `unresolved_minted`, `unresolved_quoted` | lists | `{"name": string, "sessions": [session_id, ...]}`, sorted by name, sessions sorted and distinct: the `messages.session_id` of each row that holds the reference. |
| `tables` | object | row count of each contracted table (F6) that exists in the copy, counted in the copy as hashed (so `metadata` includes the stamp row); an absent table has no key |
| `sequence` | object | `{"messages": int or null}`: `seq` of the `messages` row in `sqlite_sequence`, null when there is no row |
| `attempt` | object | `{"generation": int, "token": token}` |
| `publisher` | string | sha256 of the copy id, a newline and the hardware id (3.4); required from generation 1 |
| `pins`, `superseded` | lists | pins: `{generation, decision, from, until}`, dated only by the bucket clock; superseded: `{copy_id, generation, manifest_sha256}`, written by `--new-copy` |
| `forgotten` | list | the list of F12 |
| `guard` | object | inputs of the publish guard. The uploader side (`cloud_copy.py` and `cloud_guard.py`) authors it; lane 3 owns its guard semantics. **The checker never reads it.** |

**F6. When a copy equals its origin.** The origin is the store as of the copy's cut. Contracted tables and their
columns are the generated table at docs/store-contract.md:12-28 (15 tables); hold them as constants and pin them
with a test that parses that file.
- A contracted table exists in both or in neither, with the same contracted columns present. Columns are read by
  name; column order means nothing (docs/store-contract.md:9).
- Rows are equal by key and by every contracted column. Keys: `host_uid_bindings` by (`store_id`, `uid`,
  `lineage_key`, `kind`); `capture_events` by (`lineage_key`, `generation`, `position`) where
  `source_kind = 'line'` and by (`lineage_key`, `uid`, `content_digest`, `position`) where it is `'file'`
  (claude_code_store.py:74-77); every other table by its primary key, or by all its contracted columns when it
  has none. Rows that share a key are compared as a multiset.
- One exemption: the copy's `metadata` holds exactly one row the origin does not, key `import:origin`.
- The `sqlite_sequence` row for `messages` is equal.
- The resolved references are the same names with the same file sha256; the unresolved minted names are the
  same; the unresolved quoted names are the same.
- Tables the contract does not list are not compared row by row. In the copy they are covered by `file_sha256`
  (F10).

F7 to F11 below are shared word for word with the page-delta codec. Where they say they add to or replace a sentence of
F2, F5 or F6, that edit is already made above.

**F7. Pages and their digests.**
- `lcm.db` is a whole number of pages, at least one. The page size P is bytes 16-17 of the file, unsigned
  big-endian, where the value 1 means 65,536. P is a power of two from 512 to 65,536. Page `i` is file bytes
  `i`·P to (`i`+1)·P − 1. Page 0 holds the file header and is a page like any other.
- A page digest is the sha256 of the page's P bytes. The page list is every page digest, from page 0 to the last
  page, in page order.
- `page_list_sha256` is the sha256 of the page list as raw bytes: the 32-byte digests joined in page order, with
  no separator, no hex text and no length in front.

**F8. The delta object.** A delta object holds whole pages and nothing else.
- **Key** (this adds one line to F2's bucket list): `deltas/<sha256>.<token>` under `<prefix>/<copy id>/`. The
  sha256 is of the object's bytes. The token is the publish attempt that uploaded it.
- **Bytes.**
  - Bytes 0-7: the magic `LCMXPD1\n` (hex `4c 43 4d 58 50 44 31 0a`).
  - Bytes 8-11: the page size P. Bytes 12-15: the record count `n`. Both unsigned big-endian.
  - Bytes 16 to the end: exactly one zlib stream (RFC 1950, as the standard library's `zlib` writes and reads
    it). A writer uses level 6. A reader accepts any level.
  - The stream decompresses to exactly `n` records and nothing else. A record is 4 + P bytes: a page index `i`
    (4 bytes, unsigned big-endian), then the P bytes that replace file bytes `i`·P to (`i`+1)·P − 1.
  - Indexes rise strictly from each record to the next.
- **What binds a writer only.** An object has at least 1 record and at most ⌊(2^25 − 2^16) / (4 + P)⌋ (8,168 at
  4 KiB; 510 at 64 KiB), and is at most 2^25 bytes long, so it never needs a multipart upload. A larger delta is
  several objects. A reader does not enforce these limits: it reads a count of 0 as an object that changes
  nothing, and it never sizes memory by the count.
- **A reader refuses an object** when:

  | Code | Condition |
  |---|---|
  | `object_sha256` | the sha256 of its bytes is not the sha256 in its key |
  | `delta_magic` | it is shorter than 16 bytes, or bytes 0-7 are not the magic |
  | `delta_page_size` | bytes 8-11 are not the manifest's `page_size` |
  | `delta_stream` | bytes 16 on are not a zlib stream; the stream ends early (a stream cut inside its last 4 bytes included); or the decompressed bytes are not exactly `n` × (4 + P) |
  | `delta_trailing` | any byte follows the end of the zlib stream |
  | `delta_order` | an index is not above the index before it |

**F9. The manifest's `database` block.** This replaces the `database` row of F5's table. Every manifest is
self-contained: it names the base parts and every delta needed to rebuild its generation, in order. No reader
follows a pointer to another manifest.

| Key | Type | Value |
|---|---|---|
| `page_size` | int | P of this generation's file (F7) |
| `file_bytes`, `file_sha256` | int, sha256 | length and sha256 of this generation's whole file |
| `page_list_sha256` | sha256 | of this generation's file (F7) |
| `base` | object | the chain's base, below |
| `base.generation` | int | the generation whose whole file the parts are |
| `base.file_bytes`, `base.file_sha256` | int, sha256 | that file |
| `base.reason` | string | why that base was cut: `first`, `page_size`, `page_list`, `forget`, `missing_object`, `size` or `length` |
| `base.part_size` | int | default 33554432 |
| `base.parts` | list | `{"key", "bytes"}` in file order. `key` is `parts/<sha256>.<token>`. The parts joined in order are the base file. Every part but the last has `bytes == part_size` |
| `deltas` | list | one entry per generation after the base, oldest first |
| `deltas[].generation` | int | that generation |
| `deltas[].file_pages` | int | the file's page count as of that generation, at least 1 |
| `deltas[].objects` | list | `{"key", "bytes", "records"}`. `key` is `deltas/<sha256>.<token>`, `bytes` the object's length, `records` its header count |

- The sha256 of a part or an object is the 64 hex digits in its key. No entry carries it a second time.
- An int here is a JSON integer, zero or above; `true` and `1.0` are not ints. A reader ignores keys it does not
  know, in the block and in every entry.
- **A base generation** has an empty `deltas`, `base.generation` equal to the manifest's `generation`, and file
  fields equal to its base fields. **A delta generation** has at least one entry; entry k (from 0) is generation
  `base.generation` + 1 + k, and the last entry is the manifest's `generation`. F10 checks both.
- The previous page count of an entry is `base.file_bytes` / `page_size` for the first entry, and the entry
  before's `file_pages` for the others.
- The objects of one entry are listed in ascending index order. `objects` may be empty: that generation changed
  no page, or only got shorter.
- The manifest holds no digest of a single page or range. An object's sha256 is in its key, and the result has
  one whole-file sha256.
- **What a reader checks is in F10, and only there.**
- **What binds the writer only.** A reader does not check these and never refuses on them:
  - every part but the last has `bytes == part_size`;
  - `base.reason` is one of the seven words (a reader that shows it prints what it finds);
  - a delta generation carries the head manifest's `base` (its `reason` included) and `deltas` unchanged, plus
    one new last entry;
  - an entry's objects hold every page of that generation's file that differs from the generation before. A
    reader cannot see a changed page left out below the previous page count; only `file_sha256` shows it. Pages
    at or past the previous count are checked (F10, `delta_gap`);
  - F8's writer limits.

**F10. Rebuilding a generation, and when a reader refuses.** This replaces "`lcm.db` is the parts joined in that
order" (F5) and "covered by the part digests" (F6, last bullet): tables the contract does not list are covered by
`file_sha256`.
1. **Shape first.** Before any object is read, refuse the block when one of the conditions below holds, and for
   no other reason. When several hold, any one of their codes is a right answer.
   - `bad_database_block`:
     a. a key of F9's table, other than `base.reason` and `base.part_size`, is missing or is not of its type (a
        sha256 that fails its grammar included);
     b. `page_size` is not a value F7 allows;
     c. `base.file_bytes` is 0 or is not a whole number of pages, or the parts' `bytes` do not sum to it;
     d. a `file_pages` is below 1;
     e. `file_bytes` is not the last entry's `file_pages` × `page_size`; or, with an empty `deltas`, is not
        `base.file_bytes`;
     f. `deltas` is empty and `file_sha256` is not `base.file_sha256`.
   - `bad_name`: a part's key does not match `parts/<sha256>.<token>` in full, or an object's key does not match
     `deltas/<sha256>.<token>` in full.
   - `chain_generation`: some `deltas[k].generation` is not `base.generation` + 1 + k; or the last entry's
     generation is not the manifest's `generation`; or `deltas` is empty and `base.generation` is not the
     manifest's `generation`.
2. **The base.** Write `base.parts` in order. Refuse a named object that is absent (`object_missing`, with its
   key), whose length is not its `bytes` (`object_bytes`), or whose sha256 is not its key's (`object_sha256`).
   Then the file must be `base.file_bytes` long and hash to `base.file_sha256` (`base_digest_mismatch`).
3. **Each `deltas` entry, in order.** For each object in order, refuse as in step 2 and as in F8, and also when:

   | Code | Condition |
   |---|---|
   | `delta_count` | the header count is not the manifest's `records` |
   | `delta_order` | its first index is at or below the last index of the object before it in the same entry |
   | `delta_index` | an index is at or past this entry's `file_pages` |

   Write each record's page at `i`·P. When the entry's `file_pages` is above the previous page count (F9), every
   page from the previous count to `file_pages` − 1 must have been written by this entry (`delta_gap`). Then set
   the file's length to `file_pages` × `page_size`. Do this after every entry, so that a shorter generation
   followed by a longer one never keeps old bytes.
4. **The result.** Bytes 16-17 of the file must state `page_size` (F7), and the file must be `file_bytes` long
   and hash to `file_sha256`. Each failure is `rebuild_digest_mismatch`, with the generation. The result's page
   list hashes to `page_list_sha256`; the copy checker checks that as well, under the same code.
- A reader rebuilds any chain a manifest names. No reader limits a chain's length or stored size.
- After a refusal the file being built is not a copy of anything and must not be served or kept as one.

**F11. Shared test vectors.** Every hex string is lower case.

*File A*, 1,536 bytes, P = 512: page `k` (k = 0, 1, 2) is 512 bytes of value `k`, and then bytes 16 and 17 of
the file are set to `02 00`.

| | sha256 |
|---|---|
| page 0 | `b6d17a1a271154e2d74bdf05d114d2f55626bd2898e9c147c7489e3c1c878f75` |
| page 1 | `6caf38d537984e261527b8caef5f990fb91415a1db917198821a79ed28997973` |
| page 2 | `1bd8d04bb127c9dbdb406c3c213b202c84dc75f122ecd553b02fb322bffafdc9` |
| `page_list_sha256` | `5651d68c2542dc58414e77e1816c5af341e54e5b2daf96396af4cdae06813e8a` |
| `file_sha256` | `144056a8546425225385ebdb956b7584a8a34f1ebecac3e68c3f73553c8f9a46` |

*File B*, 2,048 bytes: file A with page 1 replaced by 512 bytes of `aa` and a page 3 of 512 zero bytes appended.

| | sha256 |
|---|---|
| page 1 | `799edf40e8115dc980109a64ff0a7ae2c6b62e20313c4a01f9871d0e189aa7c2` |
| page 3 | `076a27c79e5ace2a3d47f9dd2e83e4ff6ea8872b3c2218f66c92b89b55f36560` |
| `page_list_sha256` | `0b4ae118a4c210fabb5d988c90f4d4a3fdfafd6727f7426b7e08bd461147d1f4` |
| `file_sha256` | `916e79b77b05ac39fbb92433771f87ced54de932e04c3a01042be863e537ac4f` |

Pages 0 and 2 of B are those of A. The pages that differ from A to B are 1 and 3: page 3 is all zeros and is
still sent, because it is past A's page count.

*File C*, 65,536 bytes: all zeros, and then bytes 16 and 17 are set to `00 01`. P = 65,536, one page.
`file_sha256` and the page 0 digest are `0a0c88c08b7fcf477db1ae6d68f48abc6b2ef3e3e7fad9936e518f94e26c2d7b`.
`page_list_sha256` is `b03d47e0fc0d7bb43d78f933243b667f09b8b8a212f7bfa74f7442bf9c90cf0f`.

*One valid delta object from A to B*, 42 bytes, sha256
`e467bc42aa2753ec64ece09b0f147c8b3f6a14265916145d222328707e7c69ea`:

```
4c434d585044310a0000020000000002789c636060605c350a463460606060661805231a0000442e5414
```

Its stream decompresses to 1,032 bytes: `00000001`, 512 bytes of `aa`, `00000003`, 512 bytes of `00`. Every
reader must accept this object. A writer need not produce these 42 bytes: zlib builds compress differently, so a
writer is checked by decompressing what it wrote, never by comparing compressed bytes.

*The `database` block of generation 2 on base generation 1*, with A cut into parts of 1,024 bytes:

```json
{"page_size": 512, "file_bytes": 2048,
 "file_sha256": "916e79b77b05ac39fbb92433771f87ced54de932e04c3a01042be863e537ac4f",
 "page_list_sha256": "0b4ae118a4c210fabb5d988c90f4d4a3fdfafd6727f7426b7e08bd461147d1f4",
 "base": {"generation": 1, "file_bytes": 1536,
          "file_sha256": "144056a8546425225385ebdb956b7584a8a34f1ebecac3e68c3f73553c8f9a46",
          "reason": "first", "part_size": 1024,
          "parts": [
            {"key": "parts/3b2318e81fd1de9fc1e3fbd184bfdca9efdac6ee7d10bcf78cba505b2bfb475b.00000000000000000000000000000001", "bytes": 1024},
            {"key": "parts/1bd8d04bb127c9dbdb406c3c213b202c84dc75f122ecd553b02fb322bffafdc9.00000000000000000000000000000001", "bytes": 512}]},
 "deltas": [{"generation": 2, "file_pages": 4, "objects": [
            {"key": "deltas/e467bc42aa2753ec64ece09b0f147c8b3f6a14265916145d222328707e7c69ea.00000000000000000000000000000002", "bytes": 42, "records": 2}]}]}
```

The first part is A's bytes 0 to 1,023 and the second is A's page 2. Rebuilding this block gives file B.

**F12. The `forgotten` list.** A list at the manifest's top level (the `forgotten` row of F5).
- Each entry is `{"key": string, "kind": "forgotten" | "as-delete", "session": session_id,
  "base_generation": int or null}`, sorted by kind, then key.
- `key` is the session id for a forget, and the decision's own id for an `--as-delete` decision. Two
  `--as-delete` decisions on one session are two entries.
- `base_generation`: a manifest records each new item with its own generation only when it is a base and the
  item's session is one the head's chain may hold. Every other new item gets `null`.
- The list is cumulative: a manifest carries every entry of the head's list unchanged and adds its new items. It
  never shrinks. It is `[]` until something is forgotten.
- The copy checker does not judge this list. It is an input of the publish rule and of pruning.

**F13. Replica views.** The files of 4.7 beside a replica's byte copies (F2).
- `view.json` is one JSON object: `format` 1, `generation`, `rules`, `folder`, `forgotten_sha256`,
  `removed_sha256`.
- `removed.json` is written by refresh, atomically (a temporary file, then a rename): one JSON object, `format` 1,
  `session_ids` and `row_ids`, each sorted. `removed_sha256` is the sha256 of the file's bytes, and of no bytes
  when the file is absent. A refresh writes it once the head has verified and before any download, and only adds:
  the head's forgotten sessions (kind `forgotten`, F12) and its guard's removed row ids go into the copy's own
  file, and the same sessions, never the row ids, into the file of each copy the head lists as superseded, since
  a row id names a row of one copy (3.5).
- `view-result.json` holds `code`, `generation`, `rules`, `forgotten_sha256`, `removed_sha256` and `ended_at`
  (epoch seconds), so a rules fix retries a build that older rules refused; a build that succeeds writes `code`
  `ok`.
- The digest of a set of session ids is the sha256 of the ids as UTF-8, sorted by bytes, each followed by `\n`.
  Vector: `0b1c2d3e-1111-4222-8333-444455556666` and `3f2a1b0c-5555-4666-8777-888899990000` give
  `d32b75cc4f118d8e709d786a1d10998880f9517537e7df297302a09905e6f88c`; the empty set gives
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. The merger and the view module each
  compute it; one test holds both to this vector.

## Appendix L. Design reviews of replica views (4.7), each finding with its disposition

Section 4.7 replaces the parked merger's per-argument forget filter (#2796), after fifteen rounds of patches to it
did not converge (the release manager's ruling; D19). The design was reviewed twice by the same two reviewers,
the fresh Claude reviewer and the Codex reviewer. On the first revision both asked for changes. A release
manager's ruling decided the blocking items, and the second revision folded them and every non-blocking
item. On that revision's delta both answered SOUND, with no blocking finding.

| Finding | Class | Disposition | Where |
|---|---|---|---|
| Codex 2, Claude 1: a payload file was left out when a removed row referenced it, so a kept session's own `lcm_expand` could lose a file it also references | design-blocking | Fixed: a file leaves only by its own `session_id`, or, with none, when every row that references it belongs to a removed session; the lane packet tests a removed row quoting a kept session's file | 4.7 V6 |
| Codex 1: matching a lineage value that "ends with `:<id>`" could hit a kept subagent whose agent id equals a removed id | design-blocking | Fixed: names match whole, only as an encoding the product writes; an id of R in a key or lineage value the map cannot parse stops the build (`view_unsupported_key`); a collision fixture | 4.7 V3, V4 |
| Codex 3: rows shared by a kept and a removed session (a lineage edge, `rotation_predecessor_session:<V>`, a lifecycle row naming both) had no rule, and a JSON containment test stood in for one | design-blocking | Fixed by the release manager's ruling: a row that links a kept session to a removed one belongs to the removed one and leaves with it; a kept session's own rows stay byte-identical; goal 2 restated so | 2 (D19), 4.7 V3 |
| Codex 4: the lane packet's pagination test held a kept summary with forgotten sources, which V5 refuses, and its probe pairs had no expected outcome | design-blocking | Fixed in the lane packet: the visible-summary case is split from the mixed-summary refusal, and each probe pair names its outcome | lane 5v |
| Claude 2: the packet's scanner must skip the full-text index's internal tables and free pages | non-blocking | Adopted in the lane packet, as V13 allows | 4.7 V13 |
| Claude 3: keyed conditions run after deletes would match nothing | non-blocking | Adopted: the build collects the removed message and node ids first, and the deletes and the recheck use those sets; `lcm_embedding_*` rows are keyed by node id | 4.7 V3, V4 |
| Claude 4: the compat test checked tables only | non-blocking | Adopted: the lane's independent scanner runs over the every-feature store's view after one session's build | 4.7 V4 |
| Claude 5: refusals, waiting and the disk floor | non-blocking | Adopted: only lasting refusals stop retries, a second builder waits for `view.lock`, and the floor applies only before a clone or copy | 4.7 V8 to V11 |
| Claude 6: the build child could write into the MCP stream or hold the client's pipes | non-blocking | Adopted: its stdio is `/dev/null`, it runs in a new session, and the merger reaps it without waiting | 4.7 V9 |
| Claude 7: a removed set written by refresh was not in the safety rule | non-blocking | Adopted: `removed.json` and `removed_sha256`, compared by V2 | 4.7 V2, F13 |
| Claude 8: builds did not read `current` or hold the generation's lock; relations by one end only; the set query and `RULES` could drift between the merger and the view module | non-blocking | Adopted: a build reads `current` and holds `gen-<n>`'s shared lock, relations go by either end, and one test holds both modules to the same query, `RULES` and F13's vector | 4.1, 4.7 V3, V4, V8, F13 |
| Codex, re-review: the refusal record had no rules version, so a rules fix would not retry a build an older rule refused | non-blocking | Adopted as proposed: `rules` joins `view-result.json` and the merger's retry key | 4.7 V9, F13 |
| Claude, re-review N1: a kept fork of a removed session keeps its own rows, its cursor's `fork_parent` included, but the packet's scanner would fail on it and would not catch a build that deleted it | non-blocking (Important) | Adopted in the lane packet: a kept-fork fixture whose own rows stay byte-identical, and a scanner allowance for the own rows of the kept sessions the fixture names | lane 5v |
| Claude, re-review N2: nothing said who writes `removed.json` for idle and superseded copies, or when | nit | Adopted: a refresh writes it once the head has verified and before the download, for its own copy and, with the forgotten sessions only, for each copy its head lists as superseded; a superseded copy's own head never moves, so its file is written by the copy that supersedes it | 4.7 V3, F13; lane 5 |
| Lane 5v's first run stopped at its classification gate: six entity-memory tables had no class | Important | Adopted by the release manager's ruling: they are `unsupported`, so a build that would leave out any of their rows refuses `view_unsupported_table`, and none is deleted or edited; the every-feature test store covers the capture writer's features | 4.7 V4, V13 |
| Bot review of this record: the delete list removed messages by session only, so a message named only by a row id in `removed.json` (a partial delete whose session stays) could survive in a view built on the held generation | non-blocking | Fixed: the deletes name `messages` rows whose `store_id` is such a row id, and the byte-identical rule for kept sessions excepts them; the lane packet already required it (its test A4) | 4.7 V3 |
| Bot review of this record: V6 copies payload files where hard links fail, but V11 said payloads are never stored twice and set no floor for those copies | non-blocking | Fixed: the floor also counts the kept files' bytes when they must be copied, and V11 says where payloads are stored twice | 4.7 V11 |
| Bot review of this record: a clone passed the 5 GB floor, yet its deletes can change pages across the whole database and fill the disk | non-blocking | Fixed: a build that clones also reserves S, as a copy does; the no-rebuild path and hard links stay exempt | 4.7 V11 |

Both re-reviews are static design reviews with synthetic probes on capture stores the product built. They prove no
built view, real-store timing, embeddings-on run, live interleaving or crash behavior; row 5V and the lanes'
tests do.
