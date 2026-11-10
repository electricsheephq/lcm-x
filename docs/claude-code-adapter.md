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
has not read yet. Capture still runs, best effort: the `PreCompact` drain reads the history before
the trim only when the store lock is free and within 10 s. It records the flag per session
(`capture:local_gc:<session>`, kept once any hook saw it on, or found the session's spool pointer
unreadable), and the audit lists those sessions
without a lossless claim. One limit holds with the GC off too: a subagent transcript removed before
the worker first reads it leaves nothing behind (no cursor, no set-aside record, no file), so a
whole-home audit can still pass. Claude Code's own cleanup removes a whole session, main transcript
included, and the selected-session audit (`--session`) reports that case; whether Claude Code ever
removes only a subagent's transcript is unverified.

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
| `UserPromptSubmit`, `PostCompact` | Spool; start the worker only if none is running. | No |
| `PostToolBatch` | Spool; start a worker, which exits at once if one is running: this async hook never probes the lock (step 1 below). It fires once per batch of tool calls, also inside subagents (with `agent_id`). | No |
| `Stop`, `StopFailure` | Spool; the worker drains the transcript, then settles (below). `StopFailure` replaces `Stop` when the turn ends on an API error. | No |
| `SubagentStop` | Spool the subagent transcript (`<session>/subagents/agent-<id>.jsonl`) as a child lineage. Internal agents (`agent_type` empty) are captured and tagged; R0 saw the compaction summarizer fire it with a transcript path that does not exist, which is logged with nothing to ingest. | No |
| `PreCompact` | Drain synchronously, bounded to 10 s, only when the store lock is free; it never waits for the lock. If a worker holds it, the hook returns at once (`worker_running`) and the holder takes the spooled record. The hook logs `drained` only for a worker-confirmed complete drain, otherwise `drain_bound` (the worker's deadline ended it), `drain_incomplete` (work left for another reason), `drain_failed:<code>` or `drain_timeout`; any of these but a refused store then starts a detached worker for the rest (`+worker`), or leaves it to the worker that has taken the lock since (`+holder`). The drain captures only; it never embeds. The hook kills a drain still running 2 s after its bound. | Bounded, never on the lock |
| `SessionEnd` | Spool; start a detached drain and return at once. | No |

`PostToolBatch` is registered `async`. Other hooks return after forking a detached child,
except SessionStart and PreCompact, which stay synchronous; without fork, hooks stay synchronous.
The child validates paths, spools and starts the worker under a 30 s deadline, ignores SIGHUP
and closes inherited pipes; a kill between create and rename can leave a zero-byte `.tmp`.
A sibling spool lock preserves an unclaimed end pointer against late non-end hooks (`kept_end`)
and rejects older same-kind records (`stale`); a claimed pointer is gone, so a late write only
adds an idempotent drain. An extra settle costs at most 2 s, within PreCompact's 10 s worker
deadline; an ordinary worker has no overall deadline. Based on the `registered(store)` sweep,
the worst case of a lost pointer or a skipped settle is a delay until the next hook on the
machine, not a loss (for already registered lineages).

Each hook writes one line to `logs/hook.log` in the home: the event, the session id, the outcome and
its in-process milliseconds (`ms`), without content or paths. For #2164 the line also carries `t0`,
the wall-clock second the hook's own code began, so a row joins Claude Code's debug line for the same
run and the gap before `t0` is the start-up of the shell and the interpreter, and `phases`, the
milliseconds of each step that ran, in order:
- `imports`;
- `read` (the payload) and `spool`;
- `runtime` (finding the runtime and the worker) and `probe` (the lock);
- `subprocess` (its import), then `start` (a detached worker). For `PreCompact` it is instead
  `drain` (the bounded run) and `recheck` (the lock again), with `start` if a worker takes the rest;
- `log`, the log's own set-up.

Each step ends when the next begins, and a run that returns early or fails logs the step it was in.
So the steps after `imports` add up to about `ms`: each is rounded on its own, and they begin just
after `ms` starts and end just after it stops. Neither counts the writing of the line itself. The steps
use a monotonic clock, while `t0` and `t` are wall-clock times. These are measurements only: nothing
reads them to decide anything.

### Spool and worker handshake

Two reviews found a lost wake-up: a hook that fires while the worker finishes its last scan sees the
lock held and starts nobody. The handshake closes it:

1. **Hook:** write the spool record first (one file per session or subagent, atomic replace, with a
   nanosecond `seq`), then probe the lock. Free: start a detached worker. Held: the holder will see
   the record (step 3). The probe holds the lock for an instant, and a hook or a starting worker
   that meets it gives way, so a hook probes only once it has found a runtime and the worker
   script: the prober must be able to start the worker itself. The async `PostToolBatch` hook
   never probes, since Claude Code may kill it at any point, mid-probe too. It starts a worker on
   every tool batch, and a worker that finds the lock held exits before it loads any store code.
2. **Claim by rename:** the worker renames `<key>.json` to `<key>.claimed` before reading it. A hook
   firing meanwhile writes a new `<key>.json` and never touches the claimed file, so no record is
   deleted unread. The worker deletes the claimed file only after that transcript's batch commits. A
   claimed file found by the next lock holder comes from a crashed worker and is processed again
   (ingest is idempotent).
3. **Exit:** the worker releases the lock, then rescans the spool. If a record is there, it retakes
   the lock without blocking and loops. If another worker holds the lock, that worker acquired it
   after the record was written, and every worker scans after acquiring, so it sees the record.
   A probe (step 1) holds the lock without scanning, but a prober that finds the lock free goes on
   to a scan: a hook starts a worker, and a starting worker takes the lock. A prober killed before
   then leaves the record for the next worker run.
4. **Settle:** after draining a transcript to its end, the worker re-checks it until the file is
   quiet (bounded to about 2 s): R0 found the final reply not yet written at `Stop` in 2 of 4 runs.
5. **Sweep:** every worker run also checks each registered transcript, with its `subagents/` and
   `tool-results/` folders, for growth past its cursor (a `tool-results` file is read again only
   when its size or modification time changed). A line written after the settle, or a session
   killed before `SessionEnd`, is captured by the next run of any session. Claude Code deletes
   transcripts after `cleanupPeriodDays` (30 by default), so capture is complete once any session
   runs within that period.

No hook waits on the store lock, and no worker waits for it: a held lock means a holder that sees
every spooled record when it rescans. So a synchronous `PreCompact` drain runs only when the lock is
free. It stops at the first batch boundary after its deadline, keeping every batch it committed
whole; a batch still running 2 s after the deadline is killed with the drain and rolls back. Either
way its record stays for the next worker, since a worker removes a drained record only after the
sweep that captures its saved outputs. It exists for sessions with Claude Code's local GC on, which
trims pre-compaction history: there it captures what it can before the trim, best effort. A bounded
worker does not rescan once its deadline has passed, so when it ends incomplete its hook starts a
detached worker, or leaves the rest to the worker that has taken the lock since, which rescans: a
record another session spooled while it held the lock is not left for a later hook. A record whose
drain raises is set aside as `<key>.failed` and logged by error type only; the session's next hook
spools a fresh record, and a transcript's cursor is written before its first batch, so the sweep
retries a transcript whose first batch failed. One bad transcript cannot stall the others or loop
the worker. A claimed record the worker cannot read (an empty file after a power cut, or one that is
not a JSON object) is set aside the same way and never drained as nothing; nothing retries it, and the
audit stays unverified with `spool:unreadable_failed_record`, which names no session. The facts that
record held are lost to the audit: the session it came from can still expire by the ordinary pending
rule, so this marker is what keeps every audit unverified. A later record of the same key is set
aside beside it, never over it. Moving the `*.failed` file out of the home's `spool/` folder clears
that state: a session that expired meanwhile then reads verified, though its record was lost. So
check the session first: the file name up to its first dot is the pointer key, the first 32 hex digits
of sha256 of `host`, `session_id` and `agent_id` joined by NUL (an empty `agent_id` for a session), and
its modification time is the hook's write time.

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
  Workflow agents write `<session>/subagents/workflows/<run-id>/agent-<id>.jsonl`, and their
  `SubagentStop` hook (`agent_type` `workflow-subagent`) names that path. The sweep takes every
  `agent-*.jsonl` up to four folders below `subagents/`, never through a symlink, each as its own
  child lineage of the session. Each run's `journal.jsonl` in that run folder is captured too, as
  child lineage `journal/<run-id>` with every line kept raw (`unknown_type`): some of its results
  appear in no transcript. Its lines are workflow events (started, result, launched, failed), not
  model-visible messages, so they become no message rows. The `<session>/workflows/` folder holds
  run state and scripts, not transcripts, so it is not read. If Claude Code reuses an agent id
  within one session, each transcript gets its own lineage: a lineage is found by its path first,
  the first path keeps the agent id, and a later path is keyed by its path below `subagents/`
  while the first transcript is still on disk (#2566). A store written before that change, whose one lineage already moved between the two
  paths, keeps its rows and gains the missing lines; its audit stays unverified for that agent.
  Known limits of such a store, each pinned by a test, none losing a row or storing one twice:
  a tool call the old lineage holds stays in its session when the result is written after the
  upgrade, so the result's row is in the path session and the old lineage counts one unanswered
  call; the audit's failures count the path lineage's duplicate entries as lost, so `ok` is false
  although every row is stored once. If the user then deletes the id-keyed child session after the
  second path got its key, capture stores nothing more, but that path session keeps its ledger
  events (duplicates and shared raw lines, each holding its raw line) and the audit still reads
  them and lists that lineage unverified. They never re-enter messages, which recall reads; deleting
  the path session too removes them.
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
| `tool-results` files | Ledger rows each. A saved output is held, with no bytes of its own, only when a row of the entry that references it, as stored, holds its exact text, inline or as one externalized payload. Every other file (images, PDFs, unreferenced or large files, and saved outputs that ingest protection redacts or externalizes in parts, such as a long base64 run, or that changed after their line was read) keeps its bytes in the ledger, in rows of at most 64 MiB, with no size limit; a file whose bytes change is kept again, a held one included. A `tool-results` folder that is a symlink, or in a symlinked session folder, is never read, for files or for saved outputs. Only files directly in `tool-results/` are captured as files; its subfolders (for example page renders of a PDF) are not (#2554). |
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
- **Saved tool outputs in `lcm_grep` (#2225).** In this reader, a full-text `lcm_grep` with no
  `content_scope` searches raw messages first. When they return fewer than `limit` results, it then
  searches this store's saved tool outputs (externalized payloads), newest first. `content_scope`
  `externalized` and `both` work with `session_scope` `all` and `session`. With `session_scope`
  `current` the default stays history-only, and an explicit `externalized` or `both` is an error,
  because the reader has no active session.
  - A payload hit carries `type: externalized`, its `ref` and its `session_id`. A payload matches the
    query as a literal phrase.
  - `lcm_expand(externalized_ref, session_id)` returns the output in pages of `max_tokens`; follow
    `has_more` and `next_content_offset` for the rest.
  - The pass is bounded: 4096 names discovered, 128 files and 8 MiB read per call, 512 KB per file.
    Whenever a bound, an unreadable payload or the store-containment check cuts it, the response
    carries `partial` with its `reasons`, which means incomplete coverage. The remedy depends on the
    reason:
    - `discovery_cap`, `file_cap` and `byte_cap`: narrow `session_scope` or pass `externalized_refs`.
    - `prefix` (a file past 512 KB was searched only in its first part): page that ref with
      `lcm_expand`.
    - `containment` (the payload pass is off for this store, named refs included) and a folder-level
      `unreadable`: nothing in the payload folder was searched, so the result is history only.
  - A named ref that cannot be read is an error: no payload folder, a name missing from it, a symlink,
    an unparseable payload, or one outside the requested scope. There are two exceptions, both
    reported as `partial` rather than as an error: a named ref cut short by the byte budget
    (`byte_cap`), and the containment check turning the payload pass off (`containment`, #2518).
  - Hermes' own `lcm_grep` is unchanged.
- **Reader override.** A valid `reader-override.json` in the store home makes the server start
  another reader in place of the plain one. Something outside the plugin writes the file to opt in
  and removes it to opt out; the plugin never writes it. Each harness reads the file in its own home.
  - The file is `{"version": 1, "command": ["<absolute path>", "<arg>", ...]}`. `version` is the
    integer 1, `command` is a non-empty array of strings whose first item is the absolute path of
    an executable file, and other keys are ignored.
  - The plugin starts `command` on the same stdio and adds one environment variable,
    `LCMX_LOCAL_READER_CMD`: a JSON array holding the exact argv the plugin would have run for the
    plain reader. The rest of the environment passes through unchanged.
  - With no file, the plain reader starts and nothing is logged. The plain reader also starts, and
    one line on the server's stderr names the reason, when the file is not a regular file (a FIFO
    or a device is never read), cannot be read, is not valid JSON, has a field of the wrong type, a
    `version` other than 1, an empty `command`, or a first item that is relative, missing or not
    executable. The same holds when starting the command fails with an OS error, such as a script
    whose interpreter is gone.
  - Another local user must not be able to choose the command. So the plain reader also starts when
    the file is not owned by the user the server runs as, or when `command`'s first item is owned
    by anyone but that user or root (root owns system interpreters), or when either one is
    writable by group or others.
  - Nor may another local user replace or redirect the command before it starts (#2261). Resolving
    `command[0]` looks entries up in `/`, in each folder on the way, and in the folder of each
    symlink on it, and each such folder must pass: owned by that user or root, and writable by
    nobody else. A folder writable by others fails, except a sticky one whose entry belongs to that
    user or root. A folder writable by its group passes only when the group's members (its listed
    members and every user whose primary group it is, read when the check runs) are that user and
    root; groups nested in another group are not expanded. Members are only trusted where the host's
    user database is provably local, since a directory service may not list its users: on Linux,
    glibc, with every `passwd` and `group` line of `nsswitch.conf`, and any `initgroups` line,
    naming only `files`, `systemd` or `altfiles`; on macOS, a search path in use
    (`dscl /Search -read / SearchPath`) of local nodes only (`/Local/...` and `/BSD/local`, not an
    NIS domain under `/BSD/`). Anything else fails such a folder:
    another C library, no `nsswitch.conf`, a node that is not local, an error or a timeout. The check
    runs once per launch and is never stored. The line on stderr names the folder; for a folder
    writable by its group it names the group too, and says when the user database is not provably
    local. The command then starts by its configured path, not its resolved one:
    once every folder passes, nobody else can redirect it, and a venv's Python finds its venv from
    the path it was started by. Script arguments are not checked, nor are a script's `#!`
    interpreter, the `PATH` search of `#!/usr/bin/env`, or folder ACLs.
  - Once started, the override owns recall. The plugin does not supervise it; falling back to
    `LCMX_LOCAL_READER_CMD` on its own failures is the override's job. The file is read only when
    the plain reader could start; otherwise the server reports the missing setup step as before.
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
  - Only the detached worker runs it, with FastEmbed only, and never downloads a model.
  - Its 8 s budget is soft. A new hook record or the budget stops it between slices, but one
    slice runs to its end: the first slice loads the model, and a message with many chunks is
    embedded whole. A slice holds the capture lock, so capture waits behind it; nothing is lost,
    since hooks keep spooling. The growth run measures the longest slice.
  - Failures go to `worker.log` as tokens only. A worker run by hand also shows core's own
    warning on stderr; a hook-started worker discards it.
- Fork copies can make a search return the same text from the parent and the fork. R3 measures
  how often; collapsing a copy onto its source through `fork_copy_of` is a reader change, filed
  as its own issue with that measurement.
- One memory, two harnesses: a second, read-only server over a Hermes profile's store is
  configuration only.

## Grok Build

Grok Build runs Claude Code plugins and `~/.claude/settings.json` hooks, with differences the
hook entry handles:

- Grok reads `.grok-plugin/plugin.json` before `.claude-plugin/plugin.json`, and the `hooks` path
  it names (`.grok-plugin/hooks.json`) replaces `hooks/hooks.json`. Grok's hook runner drops
  `args`, so that file holds the same hooks in the shell form
  (`python3 -I -S "${CLAUDE_PLUGIN_ROOT}/bin/lcmx_hook.py"`). Grok sets `CLAUDE_PLUGIN_ROOT` for
  plugin hooks. Until real Grok sessions (R11), only a static check of this layout covers it.
- Detect Grok by `GROK_SESSION_ID` or camelCase keys (`hookEventName`, `sessionId`) and
  normalise the payload.
- Grok sessions get their own store home (`~/.lcm-x/grok-build`, `LCMX_GROK_HOME`), so each store
  keeps one owning harness; a read-only server can read the other store.
- There is no `transcript_path`. The session folder is
  `${GROK_HOME:-~/.grok}/sessions/<URL-encoded cwd>/<sessionId>/`, matched by its `.cwd` sidecar
  for long paths; never the newest session as a guess.
- The end-of-session `Stop` (reason not `end_turn`) and `SessionEnd` (1.5 s budget) are
  observe-only: logged, never spooled, so they cannot replace a spooled hook record that the worker
  has not yet claimed.
- The Grok reader (`updates.jsonl`) is not in this release, so the hook refuses every Grok event
  (`grok_capture_off`) and spools nothing; no setting enables it. It lands only after a
  real-session proof shows tool results are complete, `/compact` and `/rewind` never
  rewrite the file, and event ids are stable. Without stable ids, uid = session, line index and
  line digest; a rewritten file starts a new generation.
- While Grok capture is off, the MCP launcher starts no reader under Grok Build (and says why on
  stderr) unless the Grok store home holds an `lcm.db` or a `reader-override.json`.

## Runtime and failure modes

A session whose transcript did not exist at its first drain is remembered as pending and swept
every worker run until its transcript appears. Pending rows are never deleted; the audit counts them
and reports the age of the oldest one.

A pending lineage none of whose named transcripts exists is unverified, because a transcript removed before its first
drain looks the same on disk. Two shapes seen on real sessions are instead counted as expired, never written
(`expired_never_written`, and in words in the one-line verdict), and only when no hook of the lineage ever found its
transcript on disk:

- the compaction summarizer: its only hook is a `SubagentStop` with `agent_type` "" whose prompt id is one of its
  session's compactions (`PreCompact`, `PostCompact` or a `SessionStart` with source `compact`);
- a session that ended with no prompt: its hooks were only `SessionStart` and `SessionEnd`. Any other event (a prompt,
  a `Stop`, a tool hook) keeps it unverified. A built-in command such as `/usage` writes a transcript without a
  `UserPromptSubmit`, which is why the hooks' own look at the disk is required too.

To tell these apart the hook keeps a hash of each record's prompt id, its `agent_type`, and whether it found each
transcript on disk (one stat). The spool pointer merges these facts from every record that reached it since the worker
last claimed it, so a record the pointer rules replace or keep out still counts. A session still open with no prompt
yet, or one whose `SessionEnd` was lost, stays unverified. Pending rows and hook records from a build before this rule
lack these facts and stay unverified.

Known issue in the shipped preview `claude-code-v0.1.0`: before #2533, such a session's final rows could be lost if its
later hook records were also lost, and before #2535 the audit counted, but did not fail on, a session whose transcript
file never appeared.

A lineage's records can name more than one transcript path: a resume from another directory sends `SessionStart` with
a path under the new directory that never appears. Capture follows one path. The pending row and the cursor list every
path named in a record the worker reads. The audit does not match another named path against the store: if one exists and holds lines, or saved
outputs or subagent transcripts under its session folder, or cannot be checked, the lineage is unverified (`named_other_paths`, and in words in
the verdict: "a second transcript path holds rows; not matched"). For a lineage with a cursor, a named path that never
appeared, or an empty one with no saved outputs, holds nothing. A pending lineage (no cursor yet) is stricter: any named
path that exists, even empty, or cannot be checked, or saved outputs or subagent transcripts under any named path's
session folder, leave it unverified. No real session has shown a second path holding rows; matching and capturing one
is a follow-up for when one does.

Detached hook logs include `detached: true` and parent `blocking_ms`; `ms` still measures
through child completion. The parent parses stdin and forks before filesystem validation.

- Claude Code starts the hook in exec form (`hooks/hooks.json`): `python3` found on `PATH`, with
  `${CLAUDE_PLUGIN_ROOT}` substituted in its arguments and no shell before it, which saves the
  shell's start (about 4 ms per synchronous hook on the measurement machine, #2141). The exec form
  needs Claude Code 2.1.139 or later, the release that added `args`. The adapter is tested on
  2.1.314 and later. Grok Build drops `args`, so it has its own hooks file in the shell form (see
  "Grok Build"). A test keeps the two files the same apart from the command form.
- The hook entry runs on the system `python3` (3.9 or newer, stdlib only), started with `-I -S`:
  the user's `PYTHONPATH`, `PYTHONHOME`, user site-packages and `.pth` files never reach it. Each
  module loads only on the path that needs it: a synchronous hook that finds the worker running
  loads no `subprocess`, and with a set-up runtime no `shutil` either (#2141). Under `-I`, Apple's
  system Python (Xcode or the Command Line Tools) loses its bytecode cache in
  `~/Library/Caches/com.apple.python` and stops writing one, so each start compiled the standard
  library from source and took about twice as long. Before its standard-library imports, the hook gives that
  Python back the cache and the writing it has without `-I`; any other Python keeps its own. The worker and the MCP
  server need Python 3.11 or newer with LCM-X's dependencies: `LCMX_PYTHON`, else a venv in the
  plugin's data directory (`CLAUDE_PLUGIN_DATA`, which survives plugin updates and is removed on
  uninstall), else `<home>/venv`. A relative `LCMX_PYTHON` or `LCMX_PACKAGE_ROOT` is refused, as
  a relative home is: the hook checks paths from the project folder, while the worker runs in the
  store home. Hooks then record `runtime_missing` or `package_missing`, records wait in the spool,
  and the server names the setting.
  - **Zero steps** when a Python 3.11+ exists. Without `LCMX_PYTHON` or a venv, the plugin finds
    one on the hook's `PATH`: the hook's own `python3` if it is 3.11+, else `python3.14` down to
    `python3.11`. Only absolute `PATH` entries are searched: a relative one would name the project
    folder Claude Code runs the hook in.
    - It always runs that interpreter with `-I -S`, so only the standard library loads, and
      nothing installed in it, Hermes included, can change core.
    - Core needs nothing else; FastEmbed needs a venv.
    - The CLI smoke runs the release asset exactly this way, in a fresh environment.
  - **One step** otherwise: install Python 3.11+, or run `uv venv --python 3.11 <home>/venv`.
    - Until then, hooks record `runtime_missing`, and the server's error names that step.
    - The plugin never provisions an interpreter itself: that would download one without
      consent.
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
  - `stray_rows`: no row of the lineage's session lacks a ledger row that references it;
  - `row_count`: each projected entry's store ids are its own (none listed twice, by it or by
    another projected entry) and as many as a re-projection of its own raw line gives;
  - `held_mismatch`: a held file's referenced rows hold its exact text, inline or externalized.
- **Reported, never failures:**
  - lines after the cursor (`pending_lines`) and `tool-results` files whose current bytes have no
    captured version (`files.uncaptured`): both leave the lineage unverified;
  - an unterminated last record after the cursor (`partial_tail_bytes`), which Claude Code may still be writing:
    counted and named in the verdict ("N lineages end in an unterminated record"). It leaves the lineage unverified
    only when that lineage's last observed hook was its end (`SessionEnd`, or `SubagentStop` for a subagent): a crash
    mid-append left a record capture will never take (#2571). An end marks it whatever its timestamp (the hook never
    drops it as stale behind an undrained hook, nor lets another end-class hook replace it); only a newer hook of that lineage (the session resumed) clears it.
    A subagent's partial tail also counts after its session's end: Claude Code sends no `SubagentStop` to a session
    ended while a subagent is mid-task (by exit, `/clear` or a closed terminal; checked on real sessions), so the
    session's end ends its subagents too, until the session resumes (#2566). Known limits: once the session resumes
    (`claude --resume`), a subagent cut off at its end, which never resumes, has its torn last record reported only
    again while the session reads verified; a Claude Code process
    killed before any end hook (no `SessionEnd`, no `SubagentStop`) leaves a torn last record counted in the verdict
    while the session still reads verified; a resumed session whose newer hook's pointer is replaced by a delayed older
    `Stop` before the worker drains it stays unverified until a newer hook of that session drains (the safe side); and
    on one real session, a subagent still working at a `/clear` went on in a new transcript under the new session,
    and the new session's audit read not ok with one orphan (it fails closed; nothing is lost);
  - files under a `tool-results` subfolder (`subfolder_files`; a symlink to a folder, or a folder that cannot be
    listed, counts one), which capture does not store as files: until #2554 they leave the lineage unverified, and the
    verdict names them ("N files in tool-results subfolders not stored (M lineages)");
  - forgotten lineages, which are not checked; rows of an older projection version
    (`other_projection`); projected entries without a `uuid` (`projected_without_uuid`);
  - a transcript that is gone, replaced or rewritten (`transcript_state`), and older generations
    without a set-aside copy (`generations_gone`);
  - entry classes and unknown types, and projection errors;
  - unanswered calls, which an interrupted turn leaves;
  - fork copies, `fork_parent_unknown` and copies without an origin;
  - local-GC sessions;
  - spool and worker-log counts;
  - each hook's in-process time per event (`hooks_in_process_ms`: n, p50, p95, max), with each
    session's first hook the log still holds apart (`first`), and `PreCompact`'s paths
    (`precompact`: drains; pokes of the worker holding the lock; other runs, which started
    neither, such as no runtime, package or worker, a record the hook would not spool, or a hook
    error; and drains that reached their bound: `drain_bound`, as the worker reports it, or killed
    at the hook's timeout);
  - lag and store growth: the database and its externalized payloads (`lcm-large-outputs/` in
    the home) against the captured transcripts.
- **Exit 2:** the home is not absolute, holds no store, or the store has no capture stamp.
- **It does not re-derive row text.** Ingest protection may redact, normalize or externalize it by
  design; the ledger's raw line is the lossless copy, which the `corrupt` check and the
  comparison with the transcript verify. To find externalized payloads it reads the `LCM_*`
  settings of its own environment, not the store's `.env`: a mismatch can fail `held_mismatch`,
  never pass it.
- **A run passes the gate** when the audit exits 0 and `verified` is true:
  - at least one lineage was checked;
  - every selected lineage's transcript was read in full (`transcript_state` `ok`, no generation
    gone), with nothing pending: no line after the cursor, no uncaptured file, no file under a
    `tool-results` subfolder, and no other named transcript path holding rows (`named_other_paths`);
  - every selected session, and every subagent transcript on disk (a forgotten session's too),
    has a cursor;
  - closed world: below a session folder it checks, nothing it cannot tie to a capture (a `*.jsonl` no cursor
    names, a symlink, a folder that cannot be listed), whatever the folder names around it;
  - every lineage a set-aside (`.failed`) spool record names has a cursor, unless that lineage
    was forgotten. A record names its subagent even when that transcript is not on disk: one never
    written (the compaction summarizer's `SubagentStop` names one) looks the same as one removed
    before it was captured, so such a home stays unverified. A record that cannot be read
    withholds `verified`;
  - no spool record waits.

  A store that is only behind is unverified, never failed. A local-GC session is best effort and
  is listed without a lossless claim: Claude Code's local GC trims pre-compaction history, and the
  `PreCompact` drain captures it first only when the store lock is free and within 10 s.

## Release mechanics

The release rules (the gate, no rc tag, the draft release) are in
`bench/specs/RELEASE-READINESS-V1.md`, "Claude Code unit releases"; the steps are in
`docs/process/claude-code-unit-release.md`.

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
  refuses a tag that does not name the manifest's version. The release body repeats the core
  commit and version: the tag workflow adds them from `bundle.json`.
- A unit release passes the four-item gate in `bench/specs/RELEASE-READINESS-V1.md` ("Claude Code
  unit releases") on its candidate: the audit gate (with its local-GC qualification) on real
  sessions R1-R10, each with its record; the growth check; and the CLI smoke; with install records
  beside its receipt. The CLI smoke (`scripts/claude_code_unit_smoke.py`) works as follows.
    - It unpacks the asset with no Hermes, in two runtime modes: a fresh venv of the running
      Python (3.11 or newer), and that Python found on `PATH` and run with `-I -S`.
    - It runs the `SessionStart`, `UserPromptSubmit`, `PostToolBatch` (20 times) and `Stop` hooks
      registered in `hooks/hooks.json` as Claude Code does: in exec form, with `python3` found on
      `PATH`, `${CLAUDE_PLUGIN_ROOT}` substituted in the arguments and no shell. It runs them on a
      synthetic transcript and waits for each, `async` ones included. It then
      waits for the worker those hooks started; it never runs a worker itself.
    - It passes when every hook run logged a started or running worker, no spool record failed,
      `bundle.json` names the commit and core version, and the reader, started from `.mcp.json`,
      finds a planted string and returns its stored source.
    - It reports each hook's wall time (`PostToolBatch` run 20 times) beside its in-process time.
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
plus one interactive lane as client zero. R0 ran on 2026-11-01; R1-R10 run on each unit release's
candidate, on Claude Code backed by an Anthropic-compatible GLM endpoint; each run records its backend, and at least one R-test is repeated
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
measured: hook wall time (target p95 at most 50 ms on `PostToolBatch`; see the run record below), lag from transcript line to
committed row, and store growth (the growth gate above).

**Each run's record**, which is also evidence for #2141:
- the backend;
- the audit's `ok` and `verified`, and every failure counter;
- the machine (model and RAM), the runtime mode (venv or found) and the exact interpreter version;
- the per-turn hook latency per event (n, p50, p95, max): in-process from the audit
  (`hooks_in_process_ms`), and wall time from a smoke run on the same machine right after
  (`hook_wall_ms`), which adds the interpreter start-up the hook log cannot see. Each session's
  first hook, a cold interpreter start, is reported apart (`first`: n and max). #2141 reopens on a
  wall p95 over 50 ms in either mode on any R1-R10 machine, or a first hook over 500 ms; R1 states
  the margin;
- `PreCompact` as its own event, never pooled with the per-turn hooks: p50, p95 and max, and each
  run's path (a drain, or a poke of the worker holding the lock). A p95 over 2 s, or any drain that
  reaches the 10 s bound, goes to the release manager;
- the lag and the growth.

**Install records**, one per supported platform (macOS and Linux):
- **Setup:** only the documented install steps, on a fresh machine. On macOS, and on Linux when no
  container runtime is available, this may be a clean environment (a scratch `HOME` and Claude Code
  config directory) on a real host, named as such in the record, with the Linux distribution and
  version. `scripts/claude_code_install_record_linux.sh` takes the native Linux record from a scratch folder
  with no instruction file (`CLAUDE.md` or `AGENTS.md`) in any parent folder (two cases: a
  Python 3.11+ on `PATH`, and only Python 3.10 on `PATH` plus the one documented `uv venv` step). A
  native record does not prove a fresh image's package state.
- **Recorded:** whether capture writes rows or records `runtime_missing`, and how many manual steps
  it took.
- **Without Claude Code:** the CLI smoke (`scripts/claude_code_unit_smoke.py`) runs the same path
  in both runtime modes, a fresh venv and a found Python.

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
