# Claude Code adapter: capture and recall (design)

Status: draft, 2026-11-01. Tracker: #1192 (one memory across harnesses), with #1010 (portable recall).
Prior work: draft #1011 (portable capture and packaging) and draft #1216 (read-only MCP reader).

## Goal and scope

Claude Code uses LCM-X memory now:

- **Capture.** Every message and tool result of a Claude Code session is stored losslessly in an
  LCM-X store that Claude Code owns.
- **Recall.** The agent searches and reads that store through the read-only MCP reader (#1216).
- **No compaction.** Claude Code keeps its native compaction. The adapter never blocks, rewrites
  or replaces Claude Code's context, and injects nothing in this version. Compaction stays a
  Hermes feature.

It ships as a Claude Code plugin: hooks for capture, a bundled MCP server for recall. Claude Code
plugins and settings hooks are also run by Grok Build, so the same package is the base for other
harnesses (see "Grok Build").

Non-goals: Hermes behaviour or defaults (unchanged); writing to a Hermes profile's store; hosted
or cloud ingest; customer or reference-agent stores; a continuation capsule or any context
injection; importing old transcripts automatically (an explicit backfill command comes later).

## Architecture

```
Claude Code ──hook (stdin JSON)──> hook entry ──spool record──> capture worker ──one txn per batch──> lcm.db
   │                                (stdlib, <30 ms,              (Python >= 3.11, one per store,
   │                                 exit 0, no stdout)            reads the transcript from its cursor)
   └──MCP stdio──> lcm_x_mcp.py (#1216 reader, read-only) <─────────────────────────────────────────── lcm.db
```

- The transcript file is the single capture source. Hook payloads only say which transcript
  changed, so a payload that arrives early, late, twice or never cannot lose or double a row.
- One store has one owning harness (H3, #1192 design v2). The capture writer refuses any store
  not stamped as a Claude Code capture store, so it never writes a Hermes profile's store.

## Capture path per hook

Observed on Claude Code 2.1.314 (see "Evidence"). Every hook exits 0 with no stdout; a failure is
logged without content and capture catches up on the next hook.

| Hook | What the adapter does | Waits? |
|---|---|---|
| `SessionStart` (`startup`, `resume`, `fork`, `clear`, `compact`) | Register the session (cwd, source, transcript path); start the worker. A new session's transcript may not exist yet. A `fork` session links to its parent (below). | No |
| `UserPromptSubmit` | Mark the session dirty; start the worker. | No |
| `PostToolBatch` | Mark dirty; start the worker only if none is running. Fires once per batch of tool calls (including inside subagents), so the entry stays stdlib-only and fast. | No |
| `Stop`, `StopFailure` | Mark dirty; the worker drains the transcript to its end. `StopFailure` replaces `Stop` when the turn ends on an API error (observed: no `Stop` then). | No |
| `SubagentStop` | Ingest `agent_transcript_path` (`<session>/subagents/agent-<id>.jsonl`) as a child lineage linked to the session. It also fires for internal agents (`agent_type` empty); those are captured and tagged. | No |
| `PreCompact` | Drain synchronously, bounded to 10 s, before native compaction. Compaction normally appends a boundary and a summary, so this is insurance, except with the opt-in local transcript GC, which trims pre-compaction history from the file. | Bounded |
| `PostCompact` | Mark dirty. | No |
| `SessionEnd` | Start a detached drain and return at once. The transcript was complete when SessionEnd fired. | No |

A crashed session that never reaches SessionEnd is caught up by the next worker run: the worker
re-checks every registered transcript whose size passed its cursor.

What Claude Code's documentation says about the transcript, and how capture copes:

- It is written asynchronously and may lag the hook, even at `Stop`. The worker re-checks the
  file after a short settle before it exits, and the next trigger catches anything later.
- It is not strictly append-only: a resume could rewrite it (fixed in 2.1.288), an opt-in local GC
  trims pre-compaction history, and set-aside copies (`.orphaned-`, `.superseded-`) can appear.
  A rewrite starts a new cursor generation; stored rows are never deleted.
- Its format is internal and changes between versions, hence the per-type classes below and the
  `unknown_type` count.
- Claude Code deletes transcripts after `cleanupPeriodDays` (30 by default). The store keeps
  them.

## Store layout

- **One store per Claude Code configuration**, default `~/.lcm-x/claude-code/lcm.db`, overridable
  with `LCMX_CC_HOME`. The store holds all sessions of all projects, so recall works across
  sessions. It lives outside the plugin's install and data directories: removing or updating the
  plugin never removes memory. The store home also holds externalized payloads, the spool, the
  worker lock, the hook log and the runtime.
- **Owner stamp.** On creation the writer records `capture:owner` (host `claude-code`, format 1) in
  `metadata`. The writer opens only stores with that stamp.
- **Rows.** The normal `messages` table through `MessageStore.append_batch`, so ingest protection
  (externalized large and media payloads) applies exactly as in Hermes. `session_id` is the Claude
  Code session id; `source` is `claude-code`.
- **Identity.** The existing `host_uid_bindings` table. Lineage `claude-code:<session_id>` (a
  subagent: `claude-code:<session_id>/agent:<agent_id>`); uid = the entry's `uuid`, plus `#<n>`
  when one entry yields several rows; kind `canonical`.
- **Lineage.** The existing `session_lineage` table links subagents and forks to their parent.
- **Source ledger (new, additive).** `capture_events`: one row per transcript entry with lineage,
  uid, entry type, line digest, content digest, the protected raw line (compressed; empty for
  digest-only types), the store ids it produced, generation and line index. It makes capture
  provable and lets a later projection version re-derive rows after Claude Code has deleted the
  transcript. It is created by a named migration step in `store.py`, only in Claude Code stores;
  an older reader ignores it. Per `docs/process/schema-changes.md`, the PR records the format and
  the oldest reader and adds forward and rollback compat tests. Metadata keys (#1011's approach)
  do not scale: one real desktop session holds about 670,000 transcript lines.
- **Cursor.** Per transcript file: device and inode, first-line digest, byte offset, line index
  and a digest of the consumed prefix (#1011's validated scheme). A rewritten or truncated file
  starts a new generation read from the top; known uids are no-ops and nothing is deleted.

Rows, bindings, ledger rows and the cursor of one batch commit in one `ingest_atomic()`
transaction (BEGIN IMMEDIATE). A crash leaves the whole batch or none of it; a rerun is
idempotent by uid.

## Identity and projection

- **Same uuid, same content:** nothing is written. A `--fork-session` transcript copies its
  parent's entries with the same uuids and a rewritten `sessionId` (observed). The content digest
  excludes envelope fields that change on copy, so the copies are recognised: the fork gets a
  `session_lineage` link to its parent at the fork point and no duplicate rows. The uuid reuse is
  observed, not documented (the Agent SDK's own fork remaps uuids). If a version remaps them, the
  fork's copied prefix is stored again as the fork's own rows: repeated content, never loss, and
  the audit reports it as `fork_prefix_restored`. Content equality alone never merges events.
- **Same uuid, changed content:** a new row with a `supersedes` relation to the earlier one. Rows
  are never overwritten. Claude Code's own revision fields (`supersedesUuids`,
  `retractedMessageUuids`) are kept in the ledger and recorded as relations, never as deletions.
- **Projection v1, by entry type:**

| Entry | Rows |
|---|---|
| `user`, text or image parts | One `user` row. Meta entries, compact summaries and transcript-only entries become `system` rows. |
| `user`, `tool_result` parts | One `tool` row per result, `tool_call_id` = `tool_use_id`. |
| `assistant` | One `assistant` row: text parts as content, `tool_use` parts as `tool_calls`. Thinking and other parts stay in the ledger. API-error entries become `system` rows. |
| `system`, `attachment`, bookkeeping (`queue-operation`, `last-prompt`, `mode`, `custom-title`, `file-history-*`, ...) | None; raw line in the ledger. |
| Request snapshots (`api-request`, `api-request-blob`, `api-request-shape`) | None; digest only. They copy the conversation that is already stored. |
| Unknown types | None; raw line in the ledger, counted as `unknown_type`. |

Nothing is dropped silently: the audit counts every entry by class.

## Recall surface

- The plugin declares one MCP server, `lcm-x`, which runs `scripts/lcm_x_mcp.py --store <home>/lcm.db
  --hermes-home <home>` from #1216 unchanged. Its five tools (`lcm_recall`, `lcm_grep`,
  `lcm_expand`, `lcm_load_session`, `lcm_describe`) search across sessions. The `StoreReader` and
  `handle()` signatures stay as they are.
- The worker runs one bounded embedding catch-up pass (`run_incremental_embedding_pass`) after
  each drain, so vector recall follows capture. Until it does, the reader falls back to full text
  and says so (`degraded`).
- One memory, two harnesses: a second, read-only server over a Hermes profile's store is
  configuration only.

## Grok Build

Grok Build runs Claude Code plugins and `~/.claude/settings.json` hooks, with differences the
hook entry handles:

- Detect Grok by `GROK_SESSION_ID` or camelCase keys (`hookEventName`, `sessionId`) and
  normalise the payload.
- There is no `transcript_path`. The session folder is
  `${GROK_HOME:-~/.grok}/sessions/<URL-encoded cwd>/<sessionId>/`, matched by its `.cwd` sidecar
  for long paths; never the newest session as a guess.
- The end-of-session `Stop` (reason not `end_turn`) and `SessionEnd` (1.5 s budget) are
  observe-only.
- The Grok reader (`updates.jsonl`) is experimental and off (`LCMX_GROK_CAPTURE=1` enables it)
  until a real-session proof shows tool results are complete, `/compact` and `/rewind` never
  rewrite the file, and event ids are stable. Without stable ids, uid = session, line index and
  line digest; a rewritten file starts a new generation.

## Runtime and failure modes

- The hook entry runs on the system `python3` (3.9 or newer, stdlib only). The worker and the MCP
  server need Python 3.11 or newer with LCM-X's dependencies: `LCMX_PYTHON`, else a venv in the
  plugin's data directory (`CLAUDE_PLUGIN_DATA`, which survives plugin updates and is removed on
  uninstall), else `<home>/venv`. Without one, hooks record `runtime_missing` and the server
  fails with a clear message.
- Hooks print nothing: on `SessionStart` and `UserPromptSubmit` stdout would enter Claude's
  context. The worker is started fully detached with its output discarded, so a `-p` session's
  teardown cannot kill it.
- One worker per store holds an exclusive lock; parallel sessions share it. WAL mode and short
  read transactions keep the reader and the writer apart (#1216).
- Capture only reads transcripts named by a hook payload, under the Claude Code projects
  directory; it never scans other folders. `LCMX_CAPTURE=off` disables it.

## Test plan with real Claude Code sessions

Headless `claude -p` sessions on Opus 5.5 with the plugin loaded by `--plugin-dir` and an isolated
store home, plus one interactive lane as client zero.

| Run | What it proves |
|---|---|
| R0 | Hook shapes and transcript lag with successful turns: `PostToolBatch`, `Stop`, `StopFailure`, `SubagentStop`, `PostCompact`; whether `PostToolBatch` fires on 2.1.314. |
| R1 | Three turns with Bash, Read, Edit and Grep calls: audit passes. |
| R2, R3 | `--resume` and `--fork-session`: fork copies add no rows; lineage recorded. |
| R4 | Manual `/compact` and one auto-compaction: 0 lost and 0 duplicate rows across the boundary. |
| R5 | Two parallel subagents: child lineages linked to the session. |
| R6 | Four sessions in parallel on one store. |
| R7 | SIGKILL of the worker mid-batch, and a session killed before SessionEnd: rerun gives 0 lost, 0 duplicate. |
| R8 | Capture rerun twice over every transcript: 0 new rows. |
| R9 | A new session recalls an exact string from R1 through the MCP tools; `lcm_expand` returns the stored source. |
| R10 | One non-critical interactive lane for 24 h, then the audit. |
| R11 | Grok Build: synthetic payloads now; real sessions once it is installed. |

**The audit (the gate):** for every transcript, a multiset compare of projected uid and content
digest against stored rows; each uuid appears once per store (fork copies as lineage links);
counts per entry class; tool calls and results paired with no orphan and no double; every ledger
row matches its transcript line digest. Also measured: hook wall time (target p95 ≤ 50 ms on
`PostToolUse`), lag from transcript line to committed row, and store growth per 1,000 entries.

## Evidence

- A metadata-only probe plugin on Claude Code 2.1.314 recorded payload keys and transcript state
  per hook in five headless sessions. The model calls failed on a usage limit, so tool, Stop and
  subagent payloads are still to be observed (R0). Observed: `SessionStart` sources `startup`,
  `resume` and `fork`; a new session's transcript does not exist at `SessionStart`;
  `UserPromptSubmit` carries `prompt` and `prompt_id`; `PreCompact` carries `trigger` and
  `custom_instructions`; `SessionEnd` carries `reason` (`other` in print mode) and saw the complete
  file; `Stop` did not fire when the model call failed; a fork copies the parent's entries with
  the same uuids; plugin hooks get `CLAUDE_PLUGIN_ROOT`, `CLAUDE_PLUGIN_DATA` and
  `CLAUDE_PROJECT_DIR`.
- A key census of one long real transcript (no content read) found 22 entry types and the content
  parts `text`, `thinking`, `tool_use`, `tool_result`, `image` and `fallback`.

Claim class: design. Nothing here is measured capture or recall quality yet.
