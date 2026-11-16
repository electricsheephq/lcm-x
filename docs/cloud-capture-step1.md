# Cloud capture, step 1: design record

Status: decided 2026-11-16 by the release manager; this is step 1's design. Five review rounds by a fresh
Claude reviewer and `gpt-6-astra` at high; both approved the final text (Appendices C to G). One condition goes to
lane 2: `--resume` ships only on a platform where lane 2 shows the hardware id is unique per machine; elsewhere every
chain conflict ends in `--new-copy`. Drafted against
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
| D3 | Step 1 covers Claude Code capture stores. Grok Build stores join when Grok capture ships. Hermes stores are out. | Grok capture is hard off (docs/claude-code-adapter.md:467-468). A Hermes store has no origin stamp: only capture writes `capture:owner` (claude_code_store.py:50). |
| D4 | Step 1 code never writes a live capture store, its payload folder or its spool, and never takes `capture.lock`. | A held lock tells hooks and workers that a holder will rescan the spool (adapters/claude-code/bin/lcmx_hook.py:348-357; scripts/lcm_x_capture.py:82-83). An uploader that held it would inherit that duty. |
| D5 | The cloud code is a consumer of store contract v1 and changes nothing in it. | The contract already gives the read rule, the tables and the reader API it needs (docs/store-contract.md:8, :14-28, :38-59). |
| D6 | A snapshot is a page-for-page copy taken with SQLite's online backup over a `mode=ro` connection, not `MessageStore.backup`. | `MessageStore.backup` is not a contracted callable (docs/store-contract.md:38-59) and runs on a live `MessageStore` connection (store.py:3427), which the cloud code never opens. |
| D7 | The publish guard is product code. The 0-lost and 0-duplicate measurement is a separate, independent checker. | The outside-in checker is the precedent: it shares no code with what it checks (scripts/capture_outside_in.py:2-6). The worker must not judge itself. |
| D8 | A deliberate delete reaches the hosted copy at the next upload. The opt-in says so, with one stated limit (section 5.2). | Decision in force. The forgotten marker makes it cheap (claude_code_store.py:100-108). |
| D9 | Restore writes only into a fresh directory. It refuses a live capture home. | Restoring over a live store would re-capture forgotten sessions and drop rows whose transcripts are gone (critic 4, first item). |
| D10 | Recall across stores is a merger started through `reader-override.json`. It runs one stock reader per store. | The override is the existing seam, and "the plugin never writes it" (docs/claude-code-adapter.md:375-377). Each reader still serves one store. |
| D11 | The opt-in command (`lcm_x_cloud.py enable`) is the one writer of `cloud/cloud.json` and `reader-override.json`. | Critic 3, last item: draft 1 named no writer. |
| D12 | `enable` seeds the hosted copy, restores it into a scratch folder, checks it, and only then schedules uploads. | It makes "one proven restore before any capture" true for each user, not only in our lab. Cost: one full download at opt-in. |
| D13 | Uploads and replica refreshes are started by the merger process: at reader start, then on a timer, plus a manual `sync`. Every Claude Code session starts its own reader, so several mergers can run at once (each session starts the stdio server in adapters/claude-code/.mcp.json): uploads, refreshes and retention run one at a time per home under `cloud/upload.lock` (3.4). | No hook and no capture-worker change, so the capture path stays as gated. |
| D14 | Transport is our own S3 client on the standard library (SigV4, path-style), plus a directory transport for tests. No new dependency. | The unit's runtime is stdlib-only: a found interpreter runs with `-I -S` (adapters/claude-code/bin/lcmx_runtime.py:21-22). **(Assumption:** path-style SigV4 works on R2, B2, Hetzner and MinIO; lane 6 proves it on the owner's test bucket (O1) and on the host that ships.) |
| D15 | Encryption: TLS in transit and the host's default at rest. No client-side encryption for a user's own bucket. For a bucket we operate it is owner decision O2. | The isolation bar needs transport encryption only. Posture: nothing beyond the gate. |
| D16 | Moved to the owner as O3 (section 9). | A lost or stolen machine exposes the history of every one of the user's machines it holds a replica of, not only its own. That widens a privacy promise, so it is the owner's (both reviewers). |
| D17 | Step 1 ships as a Claude Code unit release. A lane that must change `store.py`, `claude_code_store.py`, `mcp_reader.py`, `readonly_store.py` or `tools.py` ships in a core train first. | Decision in force. The plan below needs no such change. |
| D18 | Amend two roadmap sentences when step 1 ships (section 4.6 and D3). | The merged endpoint serves more than "one profile's store" (ROADMAP.md:125). Row 11 names Grok Build (ROADMAP.md:22). |

## 3. The hosted copy on contract v1

**The hosted copy is a byte copy of the origin store plus a manifest. Everything that inspects it reads only
contracted tables, by column name, over `mode=ro`, plus SQLite's own `sqlite_sequence`, read-only, for the id
check (3.4).** This section adds what the contract does not cover and
restates none of it.

### 3.1 What is uploaded

- **The database file,** whole. It carries the contracted tables (docs/store-contract.md:14-28) and also tables the
  contract does not list, such as full-text and vector tables (db_bootstrap.py:1527, :1795). Those travel
  uninterpreted. Raw transcript lines are inside it, in `capture_events.raw`.
- **Payload files that rows in the snapshot reference,** and no others. A forgotten session's payload file is
  then absent from the next generation. Payload files are written before their batch commits
  (docs/claude-code-adapter.md:280-282), so every reference in a snapshot has its file on disk.
- **A manifest,** written last.
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
<prefix>/<copy id>/parts/<sha256>            database parts, 32 MiB each, named by content
<prefix>/<copy id>/payloads/<sha256>         payload files, named by content
<prefix>/<copy id>/gen/<number>.json         one manifest per generation, created once with a conditional PUT, never rewritten
```

Parts are small on purpose. No object needs a multipart upload. An interrupted upload resumes by skipping parts
that exist. Two generations share every part that did not change. How much they share is measured (section 6,
row 2); the plan does not depend on it.

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
  sha256 of the row content with every externalized placeholder expanded to its payload's text, read from that
  store's own payload folder. Two stores can hold one output differently: externalized under different file names
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
   capture always wins and the snapshot waits for a quiet moment.
2. On the staging copy only: switch it to rollback-journal mode and write the import stamp. The live store flips
   between WAL and rollback mode (claude_code_store.py:250-265, :1169-1173), and the reader refuses a WAL header
   with no sidecar files (docs/claude-code-adapter.md:416-419). So the copy is normalised before it is hashed.
3. Read the staging copy's contracted tables and build the manifest. List the payload files its rows reference
   and hash them.
4. Run the publish guard (3.4). Upload missing parts and payloads. Just before the manifest, check that every part
   and payload it names exists in the bucket, and upload any that are missing. Write the manifest last, as a
   create-if-absent (conditional) PUT of `gen/<N+1>.json`. Then run retention (3.5), still under the lock.

A crash leaves orphan parts and no manifest. An outage queues nothing: the next snapshot is simply newer. A batch
is never half in a snapshot, because rows, bindings, ledger rows and the cursor commit together
(docs/claude-code-adapter.md:280-282). Free disk is checked first: staging needs one store's size.

### 3.4 The publish guard

**The guard enforces the contract's "no stored row deleted or altered" (docs/store-contract.md:85) for messages,
bindings, relations, ledger lines and payloads between two generations, with one exemption: a session the user
forgot.** It compares generation N+1 with N before N+1's manifest is written.

- **Messages:** every `store_id` in N is in N+1 with the same content digest, or its session has a
  `capture:forgotten:<session>` marker (claude_code_store.py:52, :100-108). The manifest holds one prefix digest
  per session, so the compare is one pass. An id of N that is missing is a loss (below); an id of N that holds a
  different row is `store_id_reused` (next).
- **Bindings and relations:** every `host_uid_bindings` row and every `message_relations` row in N is in N+1
  unchanged, or its message's session is forgotten. For a binding, unchanged means its identity columns
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
  `store_id_reused` is `enable --new-copy`, never `accept-loss` or `--resume`. The old copy id keeps its ids and
  becomes this machine's superseded copy (3.5); on this path only, the new copy id starts with the old one's forgotten
  sessions, which are this machine's own history; a `--new-copy` after a chain conflict starts with none.
- **Ledger:** every ledger key in N is in N+1, or its session is forgotten. For an entry with bytes, the sha256 of
  its decompressed raw line (resolving `raw_from`) equals both its `line_digest` and the value N's manifest
  recorded, so a cut raw line with an unchanged digest column fails. A `held` file entry has no bytes: its text is
  held whole by the rows it names (claude_code_store.py:765-782). For it, the digest is unchanged and every row in
  its `store_ids` is present with an unchanged content digest.
- **Payloads:** every payload name in N is in N+1 with the same decoded content, or its session is forgotten. Core
  can replace a payload file in place (externalize.py:264-268); whether capture stores ever reach that path is
  not established, so a content change refuses.
- **Chain:** each manifest names its publisher: a digest of the copy id and the machine's hardware id (the macOS
  platform uuid, Linux's `/etc/machine-id`), which a copied home does not carry **(assumption: both are readable
  without privileges and unique per machine; a cloned VM image that keeps its `/etc/machine-id` breaks the
  second; lane 2 checks, and where uniqueness cannot be shown `--resume` does not ship and every chain conflict ends
  in `--new-copy`)**. A publish goes ahead only when the bucket's newest generation names this publisher
  and is the one `cloud/state.json` records as this machine's last, or the one it recorded as pending just before
  creating it; and the manifest is created only if `gen/<N+1>.json` does not exist. Otherwise the publish stops
  with `chain_conflict` and overwrites nothing. Two machines taking turns are caught as well as two racing ones,
  so a second machine never meets a loss refusal for the first one's rows.
- **One invariant: nothing writes or deletes under a copy id whose head names another publisher.** So one
  machine only ever publishes, runs retention or deletes under a copy id, and retention, which runs under that
  publisher's lock after its own manifest exists, can delete only parts that a refused publish left behind. Every
  command that writes to the bucket reads the head first and refuses a copy whose head names another publisher:
  `sync`, `accept-loss --release`, the pruning of superseded copies (3.5) and `delete-hosted`. `status` says why.
- **A chain conflict never offers `accept-loss`.** It has two exits:
  - `enable --resume`, only when the head names this machine's current publisher, which happens when
    `cloud/state.json` was lost. It goes ahead only when the store passes the whole guard against the head (every
    row, binding, relation, ledger key and payload of the head present and unchanged, or forgotten) and the
    store's `sqlite_sequence` is at least the head's mark (above). It records the head as this machine's last.
    `enable` lists the copy ids under the prefix whose head names this machine, so `--resume <copy id>` also
    works when `cloud/cloud.json` is lost.
  - `enable --new-copy` in every other case: a changed hardware id, a replaced machine, or a home copied with its
    `cloud/` folder to the same path on a second machine, which passes `admit()` because it compares only the
    home path (claude_code_store.py:168-169). It mints a new copy id and seeds a fresh hosted copy from this
    machine's store, with D12's restore check. This machine never writes to the old copy again. The old copy
    keeps its publisher if another machine still publishes to it, and every machine reads it as a replica;
    the sessions they share before the copy are handled by the duplicate rule (4.2).
- **An idle copy** is one that no machine publishes to any more, after a changed hardware id or a replaced
  machine. The invariant forbids pruning it, so a later forget does not reach it. `status` lists it with its last
  publish time and says that the user deletes it in the bucket host's own console. A safe way to prune or delete
  it from here is a lane 8 item (test (j)).
- **On refusal for a loss:** N stays current. Nothing in the bucket changes. `status` names what disappeared, by
  session and count. The user then chooses:
  - `accept-loss` records the acceptance in `cloud/state.json`, not in the store, and the next publish goes ahead;
    every later manifest carries the pin, so a lost `state.json` does not lose it.
    It **pins the generation before the loss**: retention keeps it, and the parts it names, for 30 days or until
    `accept-loss --release`. So a loss caused by a bug is not mirrored into the hosted copy's only intact
    generation. `status` lists pinned generations.
  - `accept-loss --as-delete` is for a loss the user meant, such as a Hermes `/lcm doctor clean apply` pointed at
    the store. It is treated like a forgotten session: no pin, and retention prunes as in 3.5.

The guard makes "uploads never mirror a local deletion that was not deliberate" mechanical. It also covers the
old payload-GC worry: the rewrite seam has no production caller on main (store.py:5-8, :1277), and a rewritten
row would fail the content compare anyway. Draft 1's "GC and cloud are mutually exclusive" rule is dropped.

### 3.5 Retention

- Retention runs only under `cloud/upload.lock`, after the new manifest exists. Every command that writes to the bucket (`enable` with `--resume` or `--new-copy`, `sync`, `delete-hosted`, `accept-loss --release`) takes the same lock; every network call has a timeout, and `status` shows the lock's age, so a hung holder is visible.
- Keep the current generation, the two before it, and any generation pinned by `accept-loss` (3.4). Older manifests
  are deleted, then any part or payload that no kept manifest names.
- Every publish deletes each older kept generation that is not pinned and whose manifest still names a session
  that the current store marks `capture:forgotten:`, or an identity accepted with `--as-delete`. **A pinned
  generation is never pruned by a forget:** it would take with it the rows a bug lost. `status` reports
  `pin_holds_forgotten` with the session count until the user releases the pin (`accept-loss --release`), and the
  opt-in says that a forgotten session stays in a pinned generation until then. `status` also warns 7 days and
  1 day before a pin expires. Because the check runs on
  every publish, a crash or a failed delete is retried by the next one. Once a publish has finished its retention
  step, no object in the bucket holds the forgotten rows outside a pinned generation, which `status` reports
  (`pin_holds_forgotten`).
- **Superseded copies.** When `enable --new-copy` leaves a copy whose head names this machine (after
  `store_id_reused`), `cloud/cloud.json` and the new copy's manifests list it as superseded, and its generations
  are pinned for 30 days, as an accepted loss's are (3.4), with the same `pin_holds_forgotten`, warnings and
  `--release`. While its head still names this machine's current publisher (the invariant, 3.4), every publish
  runs the forget pruning above over it too, so a forget reaches it once the pin ends (D8); after a changed
  hardware id it becomes an idle copy (3.4). Every machine, this one included, reads a superseded copy as a
  replica, so rows a rollback lost stay recallable and ids issued from it keep naming their rows; the duplicate
  rule applies as to any replica (4.2), and a replica of a generation that pruning deleted is dropped at the next
  refresh. Each superseded or idle copy counts as one more store in 4.4's bars. `status` lists each one, with what
  it holds that no current copy holds; `delete-hosted --copy <id>` removes a superseded copy.
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

## 4. Recall across stores

**Recall stays local. A merger runs one stock reader per store in parallel, returns the local answer unchanged,
and adds hits from the user's other machines in a separate list.**

### 4.1 Topology

- `enable` writes `reader-override.json` naming the merger. The plugin starts it and passes the plain reader's
  argv in `LCMX_LOCAL_READER_CMD` (docs/claude-code-adapter.md:381-383).
- The merger starts that plain reader as a child, and one more stock reader per replica home. Each replica home is
  `<home>/cloud/replicas/<copy id>/gen-<n>/` with `lcm.db` and `lcm-large-outputs/`, the layout the reader expects.
- **A refresh never changes files under a running reader.** The stock reader opens several connections in turn and
  resolves payloads separately (mcp_reader.py:360), so a home swapped mid-call could mix generations. Each
  generation downloads into its own directory, which is never modified, and the refresh then writes
  `replicas/<copy id>/current` (the generation number, replaced atomically). Every merger in the home reads that
  file before each call, so all of them learn of a new generation. A merger starts a new stock reader on it and
  switches new calls there only after that reader has answered one probe call. Each reader holds a shared OS lock
  on its generation's folder; a folder is deleted only by a process that takes the lock exclusively without
  waiting, so a folder another session still reads stays. An old reader is stopped once its in-flight calls end,
  or when the tool budget runs out, so no reader holds a folder longer than one call. Row 5 runs with two mergers.
  Peak disk is two generations of one replica, plus one more while another session still reads the oldest. A
  refresh starts only when free disk exceeds twice the replica plus 5 GB; otherwise it waits and `status` says
  why.
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
- **Forgotten here, forgotten everywhere on this machine:** sessions with a local forgotten marker are excluded
  from replica queries with the existing `exclude_session_ids` argument (tools.py:378-394, :2780, :4898).
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
  is the grammar the reader already uses (mcp_reader.py:95-98). Local ids stay bare.
- A follow-up call with a qualified id goes to that replica's reader with the prefix removed. A bare id goes to
  the local reader. The stock reader accepts bare ids (mcp_reader.py:230-244), so its own derived uuid, which
  changes at every download (readonly_store.py:175-179), never matters and `ForeignStoreError` cannot fire.
- An id handed out before a refresh still resolves after it: the prefix names the hosted copy, a byte copy keeps
  `store_id`, and no id ever names a different row (3.4). Only a row deleted on purpose, or lost and accepted, stops
  resolving.
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
`sync` on both machines makes it minutes.

### 4.6 Roadmap wording to amend

"The endpoint ... serves exactly one profile's store" (ROADMAP.md:125) becomes: "serves one user's stores: the
local store, and read-only replicas of that user's other machines, each read by its own reader. It accepts no
connection from another host."

## 5. Isolation, delete and restore

### 5.1 Isolation and opt-in

| Bar | Mechanism | Test |
|---|---|---|
| Capture is opt-in | Nothing in step 1 runs until `cloud/cloud.json` exists and passes the ownership check the override already uses (adapters/claude-code/bin/lcmx_mcp.py:35-41). `enable` prints the opt-in text and needs `--accept`. `disable` stops uploads and removes the override. A separate `delete-hosted` removes the hosted copy. | On a home that never opted in, capture and recall run with sockets blocked: 0 connection attempts, the audit unchanged. After `disable`: 0 uploads, plain reader starts. |
| Each hosted store reachable only by its own user's agents | The storage vendor's access control: one prefix or bucket per user and a credential scoped to it. We write no authorisation code. Our client also refuses any key outside its configured prefix, and names read from a manifest (copy ids, part and payload names) must match their grammar before they are used as paths. TLS certificates are verified; `enable` refuses a host it cannot verify and names a missing CA bundle **(assumption: some found interpreters, such as the python.org macOS build, ship without one; lane 6 tests it)**. | Negative test with two credentials, sent with our client's prefix check bypassed, so that the host itself must refuse (HTTP 403): B cannot list, read, write or delete under A's prefix. Positive control first: A can. It runs on the owner's test bucket (O1) when it exists, and on the host that ships at the gate. There is no MinIO run: installing it is a download, and the bar names a hosted store. |
| Credentials | A file in `<home>/cloud/` that only its owner can read or write (mode 0600 or stricter). Never in `cloud.json`, never printed, never typed by an agent. No environment-variable option: Claude Code's environment reaches the agent's shell, and an `env` the agent runs would be captured into the store and uploaded. | The file is refused when group or others can read or write it (the override's ownership check tests only writability, adapters/claude-code/bin/lcmx_mcp.py:35-41, so this check is new). A planted credential value is absent from logs, `status`, and the captured store after a session that runs every cloud command. |

The opt-in text must state these facts: what is uploaded (every captured session, the saved tool outputs, and
the raw transcript lines, including text that redaction removed from search, as O3 decides); who can read the
bucket; that the user's other machines will hold a copy; that a lost machine exposes the copies it holds (O3); the
delete rule and its limit (5.2); that an upload refuses rather than mirror an unexplained local loss; that a forget does not reach a copy left
idle by a changed hardware id or a replaced machine until the user deletes it (3.4); and that a
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
  recall it. The opt-in says so.
- **The stated limit.** The bar is met at the level of rows, ledger lines and payload files. Bytes of deleted rows
  can remain in unused pages of the database file, in the hosted copy exactly as in the local file
  **(assumption; section 6, row 6 measures it)**. The opt-in says this in one line.
- **Tests.** (a) With no pin in the bucket, forget a session that holds a planted string; publish; restore the current generation: 0 rows,
  0 ledger lines, 0 payloads of that session, and recall finds nothing. List the bucket: no older generation
  holds the session. Then repeat with the process killed after the manifest is written and before retention: the
  next publish leaves no generation that holds it. (b) Delete rows with no marker: the publish is refused and the
  bucket is unchanged. (c) `accept-loss`, then publish: passes, and the generation before the loss is still in the
  bucket and restores with the rows. (d) `accept-loss --as-delete`, then publish: passes, and no generation holds
  the rows. (e) Control: a publish with no delete passes. (f) Two publishers under one copy id, interleaved at each
  step (parts, part check, manifest, retention): exactly one manifest per number, no part a kept manifest names is
  deleted, and the loser reports `chain_conflict`. (g) Forget a session that a pinned generation holds: `status`
  reports `pin_holds_forgotten`; after `accept-loss --release`, the next publish leaves no generation that holds
  it. (h) Forget the newest session, capture more, publish: it passes, with no `store_id_reused`. (i) Replace the
  store with an older copy of itself: the next publish is refused with `store_id_reused`, both before and after
  capture writes past the mark; a forgotten session that the older copy holds again is refused with
  `forgotten_session_returned`; `--new-copy` passes, the old copy is superseded and pinned, an id qualified with
  the old copy id never resolves to a row of the new one, and a forget reaches the superseded copy once its pin
  ends. (j) Delete `cloud/state.json`: `chain_conflict`, then `--resume` passes and publishes. Stop the process after the
  manifest is created and before it is recorded: the next publish finds its own pending generation and passes.
  Change the hardware id, or copy the home to a second machine: `--resume` is refused, `--new-copy` passes, no
  command writes to the old copy afterwards, and `status` lists it as idle or as the other machine's. Run a
  second machine's publish attempts against the old publisher's retention at every step: no manifest names a
  missing part. Copy a home, forget session S on the first machine, run `--new-copy` on the second: it uploads S.

### 5.3 Restore

- **What it is.** `restore --generation N --to <fresh directory>` downloads the manifest, parts and payloads,
  verifies every sha256, assembles `lcm.db` and `lcm-large-outputs/`, and checks the result.
- **Refusals.** The target of `restore --to` must be empty. It must not hold `capture-owner.json` or `lcm.db`, and
  must not be a capture home, a Hermes home, or anything inside one. Replica refresh is a separate path that
  writes only under `<home>/cloud/replicas/`; staging writes only under `<home>/cloud/staging/`. Test: each
  refusal, by name.
- **Checks, in order.** (1) `PRAGMA integrity_check`. (2) The independent checker against the manifest and, where
  it is at hand, against the origin store as of that generation's cut. (3) The existing outside-in checker run
  against the restored folder with the origin's transcripts as of that cut: `--home` names the folder, and every
  transcript of each session is passed with `--transcript SESSION=PATH`, oldest first. It reads `mode=ro`
  (scripts/capture_outside_in.py:99-108). (4) One `lcm_grep` and one `lcm_expand` of a payload through
  the stock reader on the restored folder.
- **The proof, before any user upload.** Seed a store from real sessions. Publish three generations to the gate
  bucket, with capture running between publishes. At each publish keep a scratch copy of the origin store and of
  its transcripts, so each generation has its own oracle. Restore the newest: all checks pass against its own
  saved copy. Then delete the newest manifest and its unique parts from the test bucket, as a simulated loss, and
  restore the one before: all checks pass against that generation's saved copy. The generations differ, so this
  is a real test. Record: generation ids, digests, counts, no content.
- **Per user.** D12: `enable` repeats checks 1, 2 and 4 on the user's first generation before uploads are
  scheduled.

## 6. The design check

**Six rows of measurements decide the open numbers before the lanes that depend on them start. Each has a method,
a bar set now, and what happens if it fails.** The check runs on throwaway prototypes, not on product code: a
snapshot script (the standard library's backup, about 60 lines), a comparer that digests every contracted table
of two stores (about 80 lines), a merger stub (two to four stock readers, prefixes, a separate hit list; about
200 lines) and a file-swap loop (about 60 lines). None of it is lane code.

**Where it runs (release manager's edit).** The shared development machine is used by several lanes, so no timing on it is judged.
- Rows 2 and 6 (sizes, deleted bytes) measure no time. They run on the shared development machine, on scratch copies.
- Rows 1, 3, 4 and 5 are timing rows. They run on the bench hosts, Linux, which also adds the cross-platform case:
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
| 2 | **Store and upload sizes** | Three stores: the GA gate's growth-check store (about 690,000 lines); a store seeded from a maintainer's real transcripts by the growth check's method (bench/specs/RELEASE-READINESS-V1.md:94-96); the second machine's store. For each: database bytes, payload bytes, the raw-ledger share, the vector share, bytes against transcripts. Then one working day of publishes at the default 30-minute cadence: bytes uploaded by each, and in total, since parts that capture keeps writing are sent again at every publish. Also the number of session ids held by more than one of the three stores. | Sizes have no pass line; they are recorded. Daily bar: the day's publishes together upload at most the larger of 20% of the database and 3 times that day's store growth. The same bytes are what the user's other machines download, so O1's cost figures use this number. | Tune the part size once. If still over, add a base-plus-delta upload lane before build and move the date by its size. |
| 3 | **Upload time** | First upload and incremental upload from each machine. A fixed 1 GiB transfer first gives the link's throughput. Then `kill -9` mid-upload three times and rerun. | Work outside the transfer (snapshot, guard, hashing) at most 60 s per GB. Transfer at least 80% of the link's throughput. After each kill: the rerun completes, re-sends no complete part, and the bucket never holds a manifest that names a missing part. | Fix in the transport lane; no design change. |
| 4 | **Recall wait across 1, 2 and 4 stores** | The GA recall kit (bench/instruments/cc_mcp_recall/README.md:9-17, :23-25), arm T. Cells: plain reader, one store; merger, no replica; corpus split over 2 stores; over 4 stores; then 2 and 4 again with each replica padded to the largest real store from row 2. At least 200 warm calls per tool per cell, first calls reported apart, memory of all readers recorded with one session open and with four, each replica's mode recorded. | The bars in 4.4. Also correctness: 44 of 44 values found with byte-equal expand in every cell; 0 duplicates; with one session planted in two stores, including at least one externalized tool result, it is returned once; the kit's empty-store control passes first. | The kill line in 4.4: step 1 folds into step 2. |
| 5 | **Replica refresh** | The second machine pulls each new generation into its own directory while two sessions' mergers send recall calls, with the handover in 4.1 (pointer file, probe call, shared folder locks, then delete). 100 handovers, some timed to land mid-call. | 0 failed calls; no call more than 300 ms above local-only (the merger waits 250 ms for a replica, so 50 ms is left for its own work); an id issued before a handover expands after it; the first call after a handover finds a value that only the new generation holds; no call reads two generations; refresh takes at most the transfer time plus 30 s per GB; peak disk at most twice the replica, or three times while another session still reads the oldest generation (4.1). | If calls still fail under this handover, the kill line applies. |
| 6 | **Deleted bytes** | On a scratch store, forget a session that holds a planted string, through `delete_session_messages` with the adapter loaded. Take a snapshot. Search the snapshot's bytes and its full-text index for the string. | No pass line. The result decides the wording of the opt-in's limit in 5.2. | Not applicable: it is a disclosure, not a bar. |

The product instruments prove themselves later, in their own lanes: lane 4's checker must name five seeded faults
(a row removed, a row doubled, a row's content changed, a payload removed, a raw ledger blob cut) and pass a clean
copy; lane 3's guard must refuse an unexplained delete and pass a deliberate one. Neither gates anything before
that.

Order: row 1 first, because lane 2 waits for it. Then rows 2, 3 and 6. Rows 4 and 5 last, on the merger stub.

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
| 0 lost and 0 duplicate rows across stores | Chain of four links, on two machines. (1) Transcript to local: the audit and the outside-in checker, as the unit gate runs them (bench/specs/RELEASE-READINESS-V1.md:57-66). (2) Local to hosted: the copy checker, origin store against restored generation, every contracted table by key, plus payload digests and raw ledger digests. (3) Hosted to replica: sha256 of every part and payload. (4) Transcript to restored copy: the outside-in checker on the restored folder. The list of hosted copies comes from each machine's `cloud/cloud.json` (copy id) and `capture-owner.json` (origin id), not from the bucket, because a checker cannot see an input it is not given (docs/claude-code-adapter.md:730). | 0 lost and 0 duplicated at every link; order kept; 0 unpaired. No copy id published by two machines (the invariant, 3.4); an origin id under two copy ids (a copied home) is listed. Merged recall returns no occurrence twice. |
| One real recall proof | The recall kit's agent arm through the merger: a value captured only on machine A is asked for on machine B in a fresh Claude Code session told nothing about stores. One more run on a machine with no local history, the replaced-machine case. Question and expected value written down first by someone other than the runner. | The agent answers with the value; the hit carries A's copy id; `lcm_expand` returns the stored source byte-equal. With the replica removed, the same question gets a local-only answer and no tool error. Claim class: one real session, not a pass rate. |
| Copied stores carry an import stamp | The copy checker reads `import:origin` in every restored generation and replica and compares it with the manifest | Present, equal to the manifest, origin id equal to the origin machine's owner file |
| A deliberate delete reaches the hosted copy, or the opt-in says it does not | The tests in 5.2 | (a) to (e) pass; the opt-in text carries the delete rule and its limit |
| The design check shows recall can read across stores | Section 6, rows 4 and 5 | Their bars |
| Unstated but binding: local capture is unharmed (VISION.md:60) | Section 6 row 1, repeated on the candidate; the unit's own gate on the candidate | As row 1; R1-R10 pass |

Gate evidence covers Claude Code stores only. A Grok Build store qualifies later by the same restore proof and
copy checker, once its capture ships.

## 8. The build plan

**Twelve lanes: the design check, a document lane, nine code lanes and the gate run. No lane changes the core
store, ingest or the stock reader. Only lane 9 enters the unit's import closure.** Sizes are non-generated lines,
tests apart. Tiers follow the diff: anything that guards, deletes, authenticates or sits on a user's data path is
High. **(Assumption:** every size and every day count below is an estimate from the scope in this record, not
from a prototype.)

| Lane | What | Files (new unless marked) | Size | Tier | Needs |
|---|---|---|---|---|---|
| 0 | Design check: throwaway prototypes, kit runs, result card | none in the repository; prototypes in a scratch folder outside the repository; the result card goes into lane 1's document | ~400 throwaway | none (no product code) | this record's bars |
| 1 | This record as a repo document; roadmap and adapter-doc wording | `docs/cloud-capture-step1.md`; `ROADMAP.md`, `docs/claude-code-adapter.md` (edits) | docs | Low | decision by 11-21 |
| 2 | Snapshot, manifest, part store, directory transport, contract constants; hardware-id uniqueness per platform, which decides where `--resume` ships (Appendix G) | `cloud_copy.py`, `scripts/lcm_x_cloud.py` (`snapshot`, `publish`), `tests/test_cloud_copy.py`, `tests/compat/test_cloud_copy_contract.py` | ~400 | High | row 1's result |
| 3 | Publish guard (`store_id_reused`, `forgotten_session_returned`, the chain and `--resume`'s check), retention, superseded copies, `accept-loss` | `cloud_guard.py`, tests | ~300 | High | 2 |
| 4 | Independent copy checker, written from section 3 only, by a different author than lanes 2 and 3; accepted on five seeded faults | `scripts/cloud_copy_outside_in.py`, `tests/compat/test_cloud_copy_outside_in.py` | ~250 | High | section 3 |
| 5 | Restore, restore check, replica refresh into per-generation directories and the reader handover (4.1) | `cloud_restore.py`, tests | ~300 | High | 2 |
| 6 | S3 transport (conditional create, TLS verification), credentials, isolation test | `cloud_s3.py`, tests (recorded responses; live tests against the O1 bucket, skipped without its credentials, required at the gate) | ~350 | High | 2; the bucket for its live proof |
| 7a | Merger: children, routing, id qualification, fallback | `cloud_reader.py`, tests | ~300 | High | a hand-built replica home; rows 4 and 5 decide whether it continues |
| 7b | Merger: duplicate rule, status, deadline; kit cells for 2 and 4 stores | `cloud_reader.py`, `bench/instruments/cc_mcp_recall/` (edit) | ~250 | High | 7a |
| 8 | Opt-in: `enable` (with `--resume` and `--new-copy`; a plain `enable` re-run on a home whose `cloud.json` names a copy id reads the head first, per the invariant), `disable`, `status`, `sync`, `delete-hosted` (`--copy`), `accept-loss` (`--as-delete`, `--release`); `cloud/upload.lock` around every bucket write; opt-in text; the sync timer | `scripts/lcm_x_cloud.py` (edit), `cloud_reader.py` (edit), tests | ~300 | High | 3, 5, 6, 7b |
| 9 | Unit packaging, smoke, release-spec gate items, user docs | `scripts/build_claude_code_unit.py:32`, `scripts/claude_code_unit_delta.py:24`, `scripts/claude_code_unit_smoke.py`, `bench/specs/RELEASE-READINESS-V1.md`, `adapters/claude-code/README.md` (all edits) | ~200 | High | 8 |
| 10 | Gate run on the candidate: two machines, the real bucket, records | none | none | gate | 4, 6 and 9; row 3; owner decisions O1 and O3 |

- **Parallel.** Lanes 2, 4 and 7a start together: 4 and 7a need only section 3's layout. After 2 merges, lanes 3, 5
  and 6 run together. 7b follows 7a. The week of 11-30 holds four writers (3, 5, 6 and 7b), each Codex lane in its
  own leased home; review, not homes, is the limit that week, so 7b's review queues behind 3, 5 and 6.
- **One writer each.** The chain 2 to 3 to 8 to 9 to 10 is dependent and is never split across parallel writers.
- **Lanes 7a and 7b** are split because one merger PR would pass 200 lines, the stop-and-report signal.
- **Unit closure.** The unit asset ships only two entry scripts today (scripts/build_claude_code_unit.py:32;
  docs/process/claude-code-unit-release.md:30-31). Lane 9 adds the third. Until then nothing in lanes 2 to 8 can
  change a shipped unit. The candidate runs the unit's full gate, R1-R10 included, as every unit release does.
- **Core store.** No lane touches it. If the design check forces one (row 1's last fallback, or a reader change
  for swaps), that lane ships in a core train and the unit pins that core commit.
- **Authors (release manager's edit).**
  - Lanes 2, 3, 5, 6, 7a, 7b, 8 and 9 go to Codex (`gpt-6.1-sol`, high) against a spec with its acceptance tests.
  - Lane 4, the independent checker, goes to a Claude `coder`. A different model family from the author of lanes 2
    and 3 writes what judges them.
  - Lane 1 is the release manager's.
- **Review.** A Claude-authored High lane takes two reviewers; a Codex-authored one takes one fresh Claude reviewer.
  Lanes 3 and 4 decide what counts as loss, so they get the strongest reviewers available.

**Proposed ship date: 2026-12-22, with a stated risk to 12-29.** It is derived, not chosen:

- 11-17 to 11-24: design check, beside the decision on this record (due 11-21).
  - Prototypes and rows 2 and 6 from 11-17.
  - Row 1 on the first bench host from 11-19, done by 11-20, because lane 2 waits for it.
  - Rows 4 and 5 by 11-24.
  - Six rows on two hosts that are busy until 11-18 and 11-19 is the weakest number in this plan.
- Week of 11-23: lanes 2, 4, 7a. Week of 11-30: lanes 3, 5, 6, 7b. Week of 12-07: lanes 8 and 9.
- Week of 12-14: candidate and gate, including the unit's 24-hour lane.
- The critical path is 2, 3, 8, 9, then the gate: about 4 + 4 + 4 + 2 + 4 working days from 11-23, which lands on
  12-16, the roadmap's floor, with no slip at all. Two review rounds on two High lanes of that path add four
  working days: 12-22. All four lanes of that path are High; if each needs two rounds, it is about 12-29.
- It moves day for day if the owner's bucket (O1) arrives after 12-07 (lane 6's live proof, row 3 and the gate
  wait for it), if row 1 is not in by 11-23, or if rows 4 and 5 end after 11-24.
  It moves by one lane if row 2 forces delta uploads. The week of 12-14 also holds facts that update (12-16) and
  compaction part 1 (12-19), so reviewer time is the likeliest slip.
- The gate decides. If it passes earlier, step 1 ships earlier, but not before 12-16.

## 9. Owner decisions

**Three decisions need the owner. Two are about a bucket: who pays for it, and what we promise about reading it.
The third is what a user's other machines and the bucket hold, and what a lost machine exposes.**

**O1. Whose bucket (money).** Prices below were read 2026-11-10 for draft 1 and were not re-read. They assume
6 GB per user, which is unmeasured. They also price one copy. Step 1 keeps three generations (3.5), plus any
generation pinned by `accept-loss` for up to 30 days, and how many bytes generations share is unmeasured
(section 6, row 2). So stored bytes, and each figure below, can be three times higher, or more while a pin
lasts.

| Answer | What it costs | What follows |
|---|---|---|
| **a. Users bring their own bucket. We operate none. (recommended)** | Nothing for us. About $0.10 to $0.30 a month for a user at 6 GB. | We hold no user data and issue no credentials. The gate still needs one test bucket with two scoped keys from the owner: free at proof sizes on Cloudflare R2 or Backblaze B2 (each has 10 GB free). |
| b. We also operate one bucket, one prefix per user | About $1, $9 and $110 a month at 10, 100 and 1,000 users on R2 for one copy, so up to about $3, $27 and $330 with three generations; or about EUR 6.50 flat up to 1 TB on Hetzner Object Storage. | Credentials issued and revoked by hand for the first users. Decision O2 becomes live. The date holds only if the bucket exists by 12-07. |
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

- **Every size and every wait** is unmeasured until section 6 runs. The date assumes rows 1, 2, 4, 5 and 6 pass
  by 11-24; row 3 runs when the O1 bucket exists, before the gate.
- **Conditional create on a user's host.** The chain and retention rely on a create-if-absent PUT. `enable` refuses
  a host that does not honour it **(assumption: R2, B2 and Hetzner do; lane 6 checks each host it names)**.
- **TLS on found interpreters.** Some ship without a CA bundle; `enable` then refuses and says so.
- **Grok Build.** Covered by design only. The roadmap row names it; D18 proposes the wording.
- **Hermes stores** have no origin id and stay out.
- **A forget command for Claude Code.** Step 1 carries deletes that happen; it adds no way to make them.
- **Bytes of deleted rows in unused database pages** (5.2). Step 1 measures and discloses. It does not scrub.
- **The `.records` sidecar.** Refused if present. Carrying it is future work when a capture store first has one.
- **Contract version 2.** The rules in 3.6 are untested until a second version exists.
- **Bucket settings we cannot see:** object versioning and lifecycle on a user's own bucket.
- **Replica disk use.** Each machine holds a full copy of the user's other machines. No eviction in step 1.
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
| 3 "Claude Code and Grok Build": Grok not delivered | Adopt: evidence covers Claude Code; wording change proposed. | D3, D18, 7 |
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
| Scout 1, Astra 1: two machines that share a hardware id (a cloned VM image keeping its `/etc/machine-id`) compute one publisher, hold separate locks, and `--resume` would let both write under one copy id | non-blocking | Lane 2 tests it on each supported platform: two homes with one machine id and a copied home; `--resume` must be refused, and no manifest may name a missing part. Where the id cannot be shown unique, `--resume` does not ship there and every chain conflict ends in `--new-copy` (the fallback named in the round-4 checkpoint). Without `--resume`, such machines still meet `chain_conflict`, because their recorded last generations differ. | 3.4 Chain, lane 2 |
| Scout 2: a plain `enable` re-run on a home whose `cloud.json` names a copy id was not in the invariant's command list | nit | Lane 8 names it and tests it | lane 8 |
