# Claude Code adapter: capture and recall (design r2)

Status: draft, 2026-11-01. Tracker: #1192 (one memory across harnesses), with #1010 (portable recall).
Prior work: draft #1011 (portable capture and packaging) and draft #1216 (read-only MCP reader).
r2 folds two independent design reviews (a Codex `gpt-6-astra` review and a fresh-context Claude
review, both "sound with fixes"; dispositions are on the PR) and the first real-session probe (R0).

## Goal and scope

Claude Code uses LCM-X memory now:

- **Capture.** Every message and tool result of a Claude Code session is stored losslessly in an
  LCM-X store that Claude Code owns, including the files Claude Code saves large tool outputs to.
- **Recall.** The agent searches and reads that store through the read-only MCP reader (#1216).
- **No compaction.** Claude Code keeps its native compaction. The adapter never blocks, rewrites
  or replaces Claude Code's context, and injects nothing in this version. Compaction stays a
  Hermes feature.

What recall reaches in this version: user text and images, assistant text and tool calls, tool
results in full, and system notes. Thinking, attachments (`@`-file contents, plans, hook-injected
context) and bookkeeping entries are kept losslessly in the source ledger but are not searchable
yet; a later projection version can expose them because their raw lines are kept.

**Qualification boundary.** Lossless capture requires Claude Code's opt-in local transcript GC to
be off (`CLAUDE_CODE_TRANSCRIPT_LOCAL_GC` unset). With it on, the GC can delete history a drain
has not read yet; capture still runs, records the flag per session (`capture:local_gc:<session>`, kept
once any hook saw it on), and the audit lists those sessions without a lossless claim.

It ships as a Claude Code plugin: hooks for capture, a bundled MCP server for recall. Claude Code
plugins and settings hooks are also run by Grok Build, so the same package is the base for other
harnesses (see "Grok Build").

Non-goals: Hermes behaviour, paths or defaults (unchanged; a Hermes-side refusal of capture stores
is a separate follow-up); writing to a Hermes profile's store; hosted or cloud ingest; customer or
reference-agent stores; a continuation capsule or any context injection; importing old transcripts
automatically (an explicit backfill command comes later).

## Architecture

```
Claude Code ──hook (stdin JSON)──> hook entry ──spool record──> capture worker ──one txn per batch──> lcm.db
   │                                (stdlib, <30 ms,              (Python >= 3.11, one per store,
   │                                 exit 0, no stdout)            reads the transcript from its cursor)
   └──MCP stdio──> lcm_x_mcp.py (#1216 reader, read-only) <─────────────────────────────────────────── lcm.db
```

- The transcript file (with the session's `subagents/` and `tool-results/` folders) is the capture
  source. Hook payloads only say which transcript changed, so a payload that arrives early, late,
  twice or never cannot lose or double a row.
- One store has one owning harness (H3, #1192 design v2). The writer admits only a store it created
  and stamped itself (see "Admission"), so it never writes a Hermes profile's store.

## Capture triggers

Observed on Claude Code 2.1.314 (R0, see "Evidence"). Every hook exits 0 with no stdout; a failure is
logged without content and capture catches up on a later run.

| Hook | What the adapter does | Waits? |
|---|---|---|
| `SessionStart` (`startup`, `resume`, `fork`, `clear`, `compact`) | Spool the session; start the worker. A new session's transcript may not exist yet. The `fork` payload names no parent: the parent is found by the fork's copied uuids (below). | No |
| `UserPromptSubmit`, `PostToolBatch`, `PostCompact` | Spool; start the worker only if none is running. `PostToolBatch` fires once per batch of tool calls, also inside subagents (with `agent_id`). | No |
| `Stop`, `StopFailure` | Spool; the worker drains the transcript, then settles (below). `StopFailure` replaces `Stop` when the turn ends on an API error. | No |
| `SubagentStop` | Spool the subagent transcript (`<session>/subagents/agent-<id>.jsonl`) as a child lineage. Internal agents (`agent_type` empty) are captured and tagged; R0 saw the compaction summarizer fire it with a transcript path that does not exist, which is logged with nothing to ingest. | No |
| `PreCompact` | Drain synchronously, bounded to 10 s, waiting for the store lock if a detached worker holds it. The hook logs `drained` only for a worker-confirmed complete drain, otherwise `drain_incomplete`, `drain_failed:<code>` or `drain_timeout`; any of these but a refused store then starts a detached worker for the rest (`+worker`). | Bounded |
| `SessionEnd` | Spool; start a detached drain and return at once. | No |

### Spool and worker handshake

Two reviews found a lost wake-up: a hook that fires while the worker finishes its last scan sees the
lock held and starts nobody. The handshake closes it:

1. **Hook:** write the spool record first (one file per session or subagent, atomic replace, with a
   nanosecond `seq`), then probe the lock. Free: start a detached worker. Held: the holder will see
   the record (step 3).
2. **Claim by rename:** the worker renames `<key>.json` to `<key>.claimed` before reading it. A hook
   firing meanwhile writes a new `<key>.json` and never touches the claimed file, so no record is
   deleted unread. The worker deletes the claimed file only after that transcript's batch commits. A
   claimed file found by the next lock holder comes from a crashed worker and is processed again
   (ingest is idempotent).
3. **Exit:** the worker releases the lock, then rescans the spool. If a record is there, it retakes
   the lock without blocking and loops. If another worker holds the lock, that worker acquired it
   after the record was written, and every worker scans after acquiring, so it sees the record.
4. **Settle:** after draining a transcript to its end, the worker re-checks it until the file is
   quiet (bounded to about 2 s): R0 found the final reply not yet written at `Stop` in 2 of 4 runs.
5. **Sweep:** every worker run also checks each registered transcript, with its `subagents/` and
   `tool-results/` folders, for growth past its cursor (a `tool-results` file is read again only
   when its size or modification time changed). A line written after the settle, or a session
   killed before `SessionEnd`, is captured by the next run of any session. Claude Code deletes
   transcripts after `cleanupPeriodDays` (30 by default), so capture is complete once any session
   runs within that period.

A synchronous `PreCompact` drain waits for the lock until its deadline, so it does not exit early
while a detached worker holds it, and waits again for what is left of the deadline when it retakes
the lock after a rescan. A bounded worker does not rescan once its deadline has passed, so when it
ends incomplete its hook starts a detached worker: a record another session spooled while it held
the lock is not left for a later hook. A record whose drain raises is set aside as `<key>.failed`
and logged by error type only; the session's next hook spools a fresh record, and a transcript's
cursor is written before its first batch, so the sweep retries a transcript whose first batch
failed. One bad transcript cannot stall the others or loop the worker.

### Transcript facts

- **Lag.** The transcript is written asynchronously and may lag the hooks, even `Stop` (R0: the
  tool result was not yet in the file at `PostToolUse`, and the final reply was missing at `Stop` in
  2 of 4 runs). Handled by the settle and the sweep.
- **Not strictly append-only.** A resume could rewrite the file (fixed in 2.1.288); the opt-in local
  GC trims pre-compaction history; set-aside copies (`.orphaned-`, `.superseded-`) can appear.
  `/compact` appended a boundary and a summary and rewrote nothing (R0).
- **Large tool outputs** are saved to `<session>/tool-results/<name>` (text, images, PDFs); the
  transcript keeps a `<persisted-output>` preview, and `toolUseResult.persistedOutputPath` names the
  file on most entries.
- **Subagents** write `<session>/subagents/agent-<id>.jsonl`; their entries carry the parent's
  `sessionId`, plus `agentId` and `isSidechain`. The main transcript holds no sidechain entries.
- **One API response, several entries.** Claude Code writes each content block of a response as its
  own entry with its own uuid; the entries share `message.id`.
- **Format** is internal and changes between versions, hence the per-type classes below and the
  `unknown_type` count.

### Cursor

- Per transcript file: device and inode, first-line digest, byte offset, line index, and the offset
  and digest of the last consumed line. Only complete, newline-terminated lines are consumed; a
  partly written last line waits for the next drain. No line length limit stalls a cursor (the
  largest line seen is 3.5 MB); a line that is not valid JSON is kept raw and counted.
- Each drain checks, in constant time: same device and inode, first line unchanged, the last
  consumed line unchanged at its offset, size not below the offset. A failed check starts a new
  generation read from the top: known (uid, content) pairs are no-ops and nothing is deleted. The
  full prefix is not re-hashed per drain (the largest local transcript is 1.6 GB); the audit
  re-reads every line against its ledger digest.
- **Set-aside copies:** when the inode changes, the worker first drains a sibling `.orphaned-` or
  `.superseded-` file whose device and inode match the cursor, so lines written to the old file
  after the last drain are kept.

## Store layout

- **One store per harness.** Claude Code: default `~/.lcm-x/claude-code/lcm.db`, overridable with
  `LCMX_CC_HOME` (an absolute path; a relative one is refused). The store holds all sessions of all
  projects, so recall works across sessions. It lives outside the plugin's install and data
  directories: removing or updating the plugin never removes memory. The home also holds
  externalized payloads, the spool, the lock, the hook log and the runtime. Grok Build gets its own
  home (see "Grok Build").
- **Admission (H3).** Before any write-capable open:
  - The home must not be a Hermes home: it is refused when it holds Hermes markers (`config.yaml`,
    `state.db`) or lies inside the Hermes home.
  - **Creation** happens only when the home has no `lcm.db`. The writer creates `capture-owner.json`
    exclusively (`O_EXCL`) with a new store uuid and the home's real path, builds the database
    under a temporary name with the full core schema (`MessageStore` and `SummaryDAG` tables, so the
    reader's required `summary_nodes` exists; nothing compacts), stamps `capture:owner` = {host,
    format, store uuid} in `metadata`, and links it into place with `os.link`, which fails if a
    `lcm.db` appeared meanwhile.
  - **Every later open** reads the stamp through a read-only connection first. The stamp's store
    uuid must match `capture-owner.json`, and that file's recorded home must match the home's real
    path. A mismatch (a copied or moved store) or a missing stamp refuses the store: logged, nothing
    written. An existing unstamped database is never adopted.
  - Hermes does not check the stamp, so a stamped store copied into a Hermes profile would be opened
    by Hermes; refusing it there is a Hermes path change, proposed as a follow-up issue.
- **Rows.** The normal `messages` table through `MessageStore.append_batch`, so ingest protection
  (externalized large and media payloads) applies exactly as in Hermes. `session_id` is the Claude
  Code session id; a subagent's rows use `<session_id>/agent:<agent_id>`, which keeps the parent's
  chronology clean and gives each child its own lineage link. `source` is `claude-code`.
- **Identity.** The existing `host_uid_bindings` table. Lineage key `claude-code:<row session_id>`;
  uid = the entry's `uuid`, plus `#<n>` when one entry yields several rows. The first binding of a
  uid is `canonical`; a later content revision is a `version` binding (below).
- **Lineage.** The existing `session_lineage` table: a subagent (child `<sid>/agent:<aid>`, parent
  `<sid>`, `child_subagent_id`, role `subagent`) and a fork (role `fork`). The link is written before
  the batch's ingest transaction, because `record_session_lineage` runs its own transaction and is
  idempotent: a crash after the batch commits cannot lose the link.
- **Source ledger (new, additive): `capture_events`**, the one table this adapter adds. One row per
  transcript line and per `tool-results` file: lineage, generation, line index, uid, entry type,
  class, line digest, content digest, disposition (`new`, `duplicate`, `version`, `fork_copy`, or
  `raw` for a line without rows), the store ids it produced, and the raw bytes (zlib). Request
  snapshots keep their raw bytes once per session and line digest; later identical lines of the
  same session point at the row that holds them, so deleting one session never takes another
  session's bytes. An index on uid serves the version and fork lookups (the queries name it:
  without statistics SQLite would scan the whole lineage on every batch).
  - It makes capture provable and lets a later projection version re-derive rows after Claude Code
    has deleted the transcript.
  - The adapter owns it: its DDL is the named migration step `capture_events_v1` in
    `claude_code_store.py`, created only in stamped capture stores; a test shows it is never created
    in a Hermes store. An older reader ignores it. Core never imports the adapter (a test checks
    every core module), so the Hermes plugin loads neither the DDL nor the purge below.
  - Deleting a session deletes its ledger rows and raw bytes and marks the session forgotten
    (`capture:forgotten:<session>`): the worker never reads its transcript or `tool-results` files
    again, even after Claude Code rewrites the transcript, so forgotten content cannot come back.
    Later turns of a forgotten session are not captured either. A subagent is its own session
    (`<session>/agent:<agent>`) and is forgotten on its own. The adapter registers this purge with
    core (`register_session_purge`) when it loads, and both session delete paths run every
    registered purge: core's `delete_session_messages` and the doctor clean apply; one test covers
    both. Hermes never loads the adapter, so a Hermes delete on a capture store (unsupported: see
    the stamp follow-up above) would leave the ledger and write no marker.
  - Per `docs/process/schema-changes.md`, the PR records the format, the oldest reader and the
    removal version, and adds forward and rollback compat tests.
- **Cursor storage.** One `metadata` key per transcript (`capture:cursor:<lineage>`), written inside
  the batch transaction with a plain upsert (`update_metadata_json` opens its own transaction).
- **Growth gate.** Measured on one real ~670,000-line transcript before merge: store bytes against
  transcript bytes, the raw-line share, ingest time per 10,000 lines and peak memory. Above about 1.5
  times the transcript, a cheaper raw-line policy is proposed before merge.

Rows, bindings, relations, ledger rows and the cursor of one batch commit in one `ingest_atomic()`
transaction (BEGIN IMMEDIATE). Externalized payload files are written before it, so a crash leaves
at worst an unreferenced payload file, never a lost row; a rerun is idempotent by uid.

## Identity and projection

- **Duplicate:** the same (lineage, uid, content digest) again. No rows; the ledger records the
  occurrence as `duplicate`. The content digest covers the entry without the envelope fields that
  change when Claude Code copies an entry into another session (`sessionId`, `promptId`; R0 found
  fork copies differ in exactly these).
- **Revision:** a known uid with a different content digest. A new row, a `version` binding (the
  canonical binding of a uid is unique per lineage) and `supersedes` relations from the new rows to
  the earlier ones, row for row in projection order, in the same transaction. When a revision changes
  the number of rows, extra new rows link to the last earlier row and extra earlier rows to the last
  new row, so every earlier row has a successor. Rows are never overwritten. Claude Code's own revision
  fields (`supersedesUuids`, `retractedMessageUuids`) stay in the ledger and become relations, never
  deletions. R0 and the review census saw 1,206 repeated uuids within single files, all with
  identical content.
- **Forks (changed in r2 and r3).** A fork's copied entries become the fork's own rows, so every
  session reads back complete through the unchanged reader (`lcm_load_session` reads one session's
  rows, and `session_lineage` cannot express a fork point).
  - Only a session that started as a fork has copies. The hook carries the `SessionStart` source
    `fork` and its time onto the session's later spool records, and the worker records both for
    every claimed record before it drains any.
  - A copy is an entry written before the fork started: Claude Code keeps each copied entry's
    timestamp. So an entry the fork wrote itself is never a copy, even when a later fork's copy of
    it was captured first. Without a usable start time or entry timestamp, a copy is an entry whose
    uid and content digest another session wrote; that rule can take the fork's own entry for a
    copy when a later fork's copy of it was captured first. The hook records every fork's start
    time.
  - A copy's binding carries proof kind `fork_copy`. Its `fork_copy_of` relation points at the row
    that first held that uid and content in the session that wrote it (a `new` or `version` row,
    never another fork's copy), so sibling forks point at the same writer.
  - A copied revision (a second content version of a uid the fork already holds, written before
    the fork started) is a copy too: disposition `fork_copy`, a `version` binding with proof kind
    `fork_copy`, `supersedes` the fork's earlier version and `fork_copy_of` its writer's row. So a
    later fork never takes one fork's copy as the row its writer wrote.
  - The fork's parent session is the one that wrote its last copied entry, recorded when the fork
    writes its first own entry. A copy whose writer is not captured yet gets no link, and the parent
    is then the nearest captured ancestor.
  - A session not known to be a fork keeps such entries as ordinary rows without a link, because
    the direction would be a guess. Content equality alone never merges. The cost is a second copy
    of the forked prefix (measured in R3). If the parent was never captured, the copies have no link
    and the audit reports `fork_parent_unknown`.
- **Projection v1, by entry:**

| Entry | Rows |
|---|---|
| `user`, text or image parts | One `user` row. Meta entries, compact summaries and transcript-only entries become `system` rows. |
| `user`, `tool_result` parts | One `tool` row per result, `tool_call_id` = `tool_use_id`. When Claude Code saved the output to a file inside the session's own `tool-results/` folder (`persistedOutputPath`, or the path in the marker), the row holds the file's full text (up to 64 MiB, as Hermes's persisted-output recovery does); the preview stays in the ledger's raw line. |
| `assistant` | One row per entry: text parts as content, `tool_use` parts as `tool_calls`. An entry with only thinking gives no row. `message.id` stays in the ledger, so a reader can regroup one response. API-error entries become `system` rows. |
| `tool-results` files | Ledger rows each. Files not already held in full by a tool row (images, PDFs, unreferenced or large files) keep their bytes in the ledger, in rows of at most 64 MiB, with no size limit; a file whose bytes change is kept again, a held one included. A `tool-results` folder that is a symlink, or in a symlinked session folder, is never read, for files or for saved outputs. |
| `system`, `attachment`, bookkeeping (`queue-operation`, `last-prompt`, `mode`, `custom-title`, `file-history-*`, `tools-used`, desktop types such as `bridge-session` and `frame-link`) | None; raw line in the ledger. |
| Request snapshots (`api-request`, `api-request-blob`, `api-request-shape`) | None; raw line kept once per line digest. They hold the exact model input, including system text and injected reminders that appear nowhere else in the transcript. |
| Unknown types | None; raw line in the ledger, counted as `unknown_type`. |

Nothing is dropped silently: the audit counts every entry by class. A `user` or `assistant` entry
without a `uuid` (none seen in real transcripts) and an entry whose projection fails keep their raw
line with a note (`projected_entry_without_uuid`, `projection_error:<type>`), and the audit lists them.

## Recall surface

- The plugin declares one MCP server, `lcm-x`, which runs #1216's `scripts/lcm_x_mcp.py --store
  <home>/lcm.db --hermes-home <home>`. Its five tools (`lcm_recall`, `lcm_grep`, `lcm_expand`,
  `lcm_load_session`, `lcm_describe`) search across sessions. The `StoreReader` and `handle()`
  signatures stay as they are. #1216 gets two behaviour changes in its own PR: host-neutral error
  texts, and the offline check re-reads the header once before refusing.
- **Between worker runs.** When the last connection to a WAL store closes, SQLite deletes the
  `-wal` and `-shm` files, and the reader refuses a WAL store without them. So the worker
  checkpoints and switches the store to rollback mode (`journal_mode=DELETE`) before it exits; the
  reader then opens it read-only with no sidecars, and the next worker run switches it back to WAL.
  The switch runs only under the store lock, so no other writer is active. If it fails because a
  reader is mid-call, the store stays in WAL with its sidecars (a read-only connection cannot delete
  them), recall keeps working, and the next run retries; the reader's refusal text says to retry once
  the writer has run. If that reader closes before the worker does, the worker's close removes the
  sidecars of a store still marked WAL, so the worker checks the header after its close and switches
  the store again, still under the lock, through a connection in exclusive locking mode: from its
  first read to its close no reader can attach, so none can block the switch or leave that close to
  remove sidecars again. A store still in WAL without sidecars after that close is logged and leaves
  the run incomplete: a bounded run's hook then starts a detached worker, and otherwise the next
  hook's worker retries. R9 runs after the worker has exited and the reader has restarted.
- **One configuration.** The worker loads `LCM_*` settings from `<home>/.env`, exactly as the
  reader does, so both compute the same embedding identity.
- The worker runs one bounded embedding catch-up pass (`run_incremental_embedding_pass`) after each
  drain, so vector recall follows capture; until it runs, recall falls back to full text.
- Fork copies can make a search return the same text from the parent and the fork. R3 measures
  how often; collapsing a copy onto its source through `fork_copy_of` is a reader change, filed
  as its own issue with that measurement.
- One memory, two harnesses: a second, read-only server over a Hermes profile's store is
  configuration only.

## Grok Build

Grok Build runs Claude Code plugins and `~/.claude/settings.json` hooks, with differences the
hook entry handles:

- Detect Grok by `GROK_SESSION_ID` or camelCase keys (`hookEventName`, `sessionId`) and
  normalise the payload.
- Grok sessions get their own store home (`~/.lcm-x/grok-build`, `LCMX_GROK_HOME`), so each store
  keeps one owning harness; a read-only server can read the other store.
- There is no `transcript_path`. The session folder is
  `${GROK_HOME:-~/.grok}/sessions/<URL-encoded cwd>/<sessionId>/`, matched by its `.cwd` sidecar
  for long paths; never the newest session as a guess.
- The end-of-session `Stop` (reason not `end_turn`) and `SessionEnd` (1.5 s budget) are
  observe-only: logged, never spooled, so they cannot replace a pending record.
- The Grok reader (`updates.jsonl`) is experimental and off (`LCMX_GROK_CAPTURE=1` enables it)
  until a real-session proof shows tool results are complete, `/compact` and `/rewind` never
  rewrite the file, and event ids are stable. Without stable ids, uid = session, line index and
  line digest; a rewritten file starts a new generation.

## Runtime and failure modes

- The hook entry runs on the system `python3` (3.9 or newer, stdlib only). The worker and the MCP
  server need Python 3.11 or newer with LCM-X's dependencies: `LCMX_PYTHON`, else a venv in the
  plugin's data directory (`CLAUDE_PLUGIN_DATA`, which survives plugin updates and is removed on
  uninstall), else `<home>/venv`.
  - Otherwise the plugin uses a Python 3.11+ on the hook's `PATH`: a versioned `python3.1x`, or
    `python3` when it is the hook's own 3.11+ interpreter.
  - That interpreter runs with `-I -S`, so only the standard library loads and nothing installed
    in it, Hermes included, can change core. Core needs nothing else; FastEmbed needs a venv.
  - Without any of these, hooks record `runtime_missing`, and the server's error names the one
    setup step: install Python 3.11+, or run `uv venv --python 3.11 <home>/venv`.
- Hooks print nothing: on `SessionStart` and `UserPromptSubmit` stdout would enter Claude's
  context. The worker is started fully detached with its output discarded, so a `-p` session's
  teardown cannot kill it.
- One worker per store holds an exclusive lock; parallel sessions share it.
- Capture reads only transcripts named by a hook payload under the Claude Code projects directory
  (a relative `CLAUDE_CONFIG_DIR` is refused), plus those sessions' own `subagents/` and
  `tool-results/` folders. `LCMX_CAPTURE=off` disables it.

## The audit command

`lcm_x_capture.py audit --home <store home> [--session <id>]` checks a capture store against its
transcripts and prints one JSON report of counts and ids, never paths or content. It is the unit
gate's instrument (see "The audit" in the test plan).
- **Read-only.**
  - The connection is `mode=ro`. It is `immutable` only for a WAL header without sidecars, which
    means no connection is open.
  - The audit takes no capture lock and writes nothing to the store or the home.
  - Each lineage's cursor, ledger rows, message rows and relations are read in one read
    transaction; its transcript and `tool-results` files are read after it.
- **Failures** (exit 1 when any is above 0):
  - `lost_lines`, `digest_mismatch`, `surplus_lines`: every line before the cursor has a ledger
    row with that line's digest, and no ledger row lies past the cursor;
  - `corrupt`: each ledger row's bytes, or those of the shared row its `raw_from` names, hash to
    its digest; a kept file's chunks rebuild the file's digest;
  - `lost`, `duplicates`, `missing_rows`: the multiset compare of projected (uid, content digest)
    against the `new`, `version` and `fork_copy` rows, whose store ids must exist;
  - `orphans`, `doubles`: tool calls and results pair one to one, over the rows no revision
    superseded;
  - `files_lost`: every `tool-results` file of a session has a ledger row with its current digest.
- **Reported, never failures:**
  - lines and files written after the last capture: `pending_lines`, and `files_pending` for a file
    newer than the lineage's last capture;
  - a transcript that is gone, replaced or rewritten (`transcript_state`), and older generations
    without a set-aside copy (`generations_gone`);
  - entry classes and unknown types, and projection errors;
  - unanswered calls, which an interrupted turn leaves;
  - fork copies, `fork_parent_unknown` and copies without an origin;
  - local-GC sessions;
  - spool and worker-log counts;
  - hook p95 per event, with `hook_budget_ok` (`PostToolBatch` p95 at most 50 ms);
  - lag and store growth.
- **Exit 2:** the home is not absolute, holds no store, or the store has no capture stamp.
- **A run passes the gate** when the audit exits 0 and every lineage of the run was verified: its
  transcript was read in full (`transcript_state` `ok`, no generation gone). A local-GC session is
  listed without a lossless claim.

## Release mechanics

LCM-X for Claude Code is its own release unit, versioned by the plugin manifest
(`adapters/claude-code/.claude-plugin/plugin.json`; `0.1.0` is the preview).
- A release is an annotated tag `claude-code-v<version>` on a `main` commit. The Hermes release
  workflows trigger on `v*` tags, so they never fire on it.
- `scripts/build_claude_code_unit.py <tag>` builds the release asset from the tag's tree alone
  (`git archive`). The asset holds:
  - the plugin folder;
  - core, vendored at `lcm-x/`, without tests, bench, docs or repository tooling (the license and
    the files core reads at run time stay);
  - `bundle.json`: the unit version, the tag, the full core commit and core's own version.

  The asset is `lcm-x-claude-code-<version>.tar.gz`, with its `.sha256` beside it. The build
  refuses a tag that does not name the manifest's version. The release notes repeat the core
  commit and version.
- A unit release needs two passes on its candidate:
  - the unit gate: the audit gate, with its local-GC qualification;
  - the CLI smoke. It unpacks the asset into a fresh Python 3.11 venv with no Hermes, runs the
    hook entry on a synthetic transcript and then the worker, and passes when the reader finds a
    planted string through the MCP entry. It needs #1216 on `main`.
- The gate re-runs whenever a file the unit depends on changed since the last `claude-code-v*`
  tag. `scripts/claude_code_unit_delta.py` computes that set and lists what changed in it:
  - the set is the import closure of the worker and reader entry points, plus the plugin folder;
  - `--check` exits 1 when anything in it changed;
  - before the first tag, everything counts as changed.

## Mapping to exactly-once ingest (#2055, design r3.4)

The adapter writes no #2055 ledger tables. This is how its records adopt them once P1 lands:

| Adapter | #2055 r3.4 | At adoption |
|---|---|---|
| Lineage key `claude-code:<session_id>` | Lineage (§3.8), keyed by the persisted host_uid key | Each session or subagent is a root lineage. A fork is a fork child whose parent link (by occ_id) is the parent occurrence of its last copied entry (§3.8). |
| uid (`uuid`, `uuid#n`), `canonical` binding | Uid bindings (§3.1 item 2): UNIQUE (lineage, uid) on the canonical binding | Carried over unchanged. |
| Content revision: `version` binding plus `supersedes` | Revision rows through host_uid's `version` kind (E5a) | Carried over; a supersedes row takes the ordinal right after the row it replaces (§3.1 item 1). |
| Line index within a generation (parts in part order) | `occurrence_bindings.ordinal` | P1g's binding-only migration (§3.6) binds each stored row in store order, each appended at the lineage's last ordinal plus 2^20. No rows are inserted. |
| New generation (rewritten file) | A known (uid, content) is a replay and writes no occurrence; new content under a known uid is a revision | Nothing to migrate. |
| File cursor (inode, offset, line index) | The durable cursor = the newest ingest vector (P1d) | The byte cursor stays the source position; each drain batch then also writes one ingest view, and the newest becomes the durable cursor. |
| Content digest (sha256, unsalted, without `sessionId` and `promptId`) | `identity_digest` (salted, pre-redaction identity tuple, M4) | Recomputed with the store salt from the rows or the ledger's raw lines; never copied. Content equality never merges. |
| Entry `timestamp` | Host stamp | Read from the ledger's raw line. |
| Disposition `new`, `duplicate`, `version`, `fork_copy` | `stored`; replay (no occurrence); revision; `stored` in the fork lineage | — |
| — | `key_rank`, `view_seq` | `key_rank` NULL at P1 (X4); `view_seq` store-global (X8). |

## Test plan with real Claude Code sessions

Headless `claude -p` sessions with the plugin loaded by `--plugin-dir` and an isolated store home,
plus one interactive lane as client zero. R0-R10 run before 2026-11-03 on Claude Code backed by an
Anthropic-compatible GLM endpoint; each run records its backend, and at least one R-test is repeated
on Anthropic models afterwards. Only synthetic stores and the test lane's own sessions are used.

| Run | What it proves |
|---|---|
| R0 (done) | Hook shapes and transcript lag with successful turns (see "Evidence"). |
| R1 | Three turns with Bash, Read, Edit and Grep calls, one output large enough for Claude Code to save it to `tool-results/`: the audit passes and `lcm_expand` returns the full output. |
| R2, R3 | `--resume` and `--fork-session`: the fork reads back complete; its copies carry `fork_copy`; lineage recorded; parent rows unchanged. |
| R4 | Manual `/compact` and one auto-compaction, local GC off: 0 lost and 0 duplicate rows across the boundary. |
| R5 | Two parallel subagents: two child sessions, each linked to the parent. |
| R6 | Four sessions in parallel on one store. |
| R7 | SIGKILL of the worker mid-batch; a session killed before `SessionEnd`; a hook firing at the worker's final scan: reruns give 0 lost, 0 duplicate. |
| R8 | Capture rerun twice over every transcript: 0 new rows. |
| R9 | After the worker has exited and the reader restarted, a new session recalls an exact string from R1 through the MCP tools; `lcm_expand` returns the stored source. |
| R10 | One non-critical interactive lane for 24 h, then the audit. |
| R11 | Grok Build: synthetic payloads now; real sessions once it is installed. |

**The audit (the gate):** per lineage, a multiset compare of projected (uid, content digest) against
stored rows; counts per entry class; tool calls and results paired with no orphan and no double;
every ledger row's line digest re-read from its file; every `tool-results` file present with its
digest; sessions with local GC on listed; fork copies counted in their own column, never as
duplicates or surplus, with `fork_parent_unknown`; spool records set aside as `.failed`. Also
measured: hook wall time (target p95 at most 50 ms on `PostToolBatch`), lag from transcript line to
committed row, and store growth (the growth gate above).

## Evidence

- **R0** (Claude Code 2.1.314, GLM-backed, five headless sessions with a metadata-only probe plugin):
  `PostToolBatch` fires, also inside subagents with `agent_id`; the tool result is not yet in the
  file at `PostToolUse`; the final assistant line was absent at `Stop` in 2 of 4 runs; `/compact`
  appends `compact_boundary` and a summary entry with no rewrite; the compaction summarizer fires
  `SubagentStop` with an empty `agent_type` and a transcript path that does not exist; fork copies
  differ from the parent's entries only in `sessionId` and `promptId`; the `fork` `SessionStart`
  payload names no parent; subagent files carry the parent's `sessionId`; a new entry type
  `tools-used` appeared.
- An earlier probe on the same version: `SessionStart` sources `startup`, `resume` and `fork`; a new
  session's transcript does not exist at `SessionStart`; `UserPromptSubmit` carries `prompt` and
  `prompt_id`; `PreCompact` carries `trigger` and `custom_instructions`; `SessionEnd` carries
  `reason` and saw the complete file; plugin hooks get `CLAUDE_PLUGIN_ROOT`, `CLAUDE_PLUGIN_DATA`
  and `CLAUDE_PROJECT_DIR`.
- **Census** of one development machine's own transcripts (types, keys and counts; no content):
  2,962 `tool-results` files (text, JPEG, PNG, PDF); in a 230-file sample, 153 persisted-output
  markers, 112 of them with `persistedOutputPath`; desktop entry types beyond the documented ones
  (`atis-latch`, `bridge-session`, `frame-link`, `history-suppression`, `artifact-*`, `started`,
  `result`, `launched`); no set-aside copies.

Claim class: design. Nothing here is measured capture or recall quality yet.
