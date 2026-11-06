# Gate kit: the v0.30.0 "nothing lost" audit

`audit.py` reads one driven run (docs/exactly-once-ingest.md section 5, "Gate (v0.30.0)" and "The driven run") and
reports every gate count in one JSON file. It works on any Hermes home: the host's `state.db`, the LCM-X store and
the run's log files, including a reliability-harness cell's kept `db/` copies.

**Claim class: `instrument`.** It measures; it does not prove the gate by itself. The release manager's receipt
cites its output next to the run's identity (rc tree, host sha, run window) and the hand checks listed below.

## Run
Run it after the run has stopped (no Hermes or LCM-X process writing either database).
```
python -m bench.instruments.gate_kit.audit --state-db <hermes home>/state.db --lcm-db <lcm.db> \
  [--log <file> ...] [--log-tz <zone>] [--since <ISO-8601 or epoch seconds>] [--events <driver events.jsonl>] \
  --out <result.json>
```
- Both databases are opened through `mode=ro&immutable=1` file URIs: never written, and no `-wal` or `-shm` file is
  created beside them. An immutable open does not read a `-wal` file, so a database with a non-empty `-wal` beside it
  is refused (exit 2), as is one with a non-empty (hot) rollback `-journal`: checkpoint or roll it back first (open it once after the run has stopped, or copy it with its `-wal` and
  checkpoint the copy).
- `--since` (a naive ISO time is UTC): a lineage is scored when any member started at or after it, and only those
  members count: a rotation child in the window is scored without its earlier parent's rows or the copies of them it
  carries. Stored rows under a session the host does not know count when ingested at or after it. Driver events,
  compaction lines and recorder lines count only for audited sessions, at or after it.
- `--log`: the run's Hermes logs, in either Hermes format (`<asctime> <LEVEL> [<session>] <logger>: <message>` or the
  verbose `<asctime> - <logger> - <LEVEL> [<session>] - <message>`). A pattern counts only at the start of a message
  logged by LCM-X's own logger (a `hermes_lcm` module); a line logged to two files counts once.
- `--log-tz`: the zoneinfo name the log times are written in (for example `UTC` or `Europe/Berlin`). Default: the
  audit machine's local zone. Pass the run host's zone when they differ: it decides whether a compaction or a
  recorder line came before a driver restart event. An unknown zone exits 2.
- `--events`: the driver's event log, one JSON object per line: `{"t": epoch, "kind": "restart_cold" | "restart_warm"
  | "undo_resend" | "delegate_prompt" | "compress_request", "session": id}`.
- Exit 0 with the result written (the verdict is inside); exit 2 on an unreadable input. Stdout is a four-line summary.

The output holds counts, rates, booleans, session ids and store ids only. Message content, tool arguments and
results are compared in memory and never written (a test plants a sentinel in every content field to hold this).

## What it measures
Expected rows come from the host's own `state.db`, per lineage (a session plus its rotation children: a child whose
parent ended with `end_reason = 'compression'` on the same source), counted the way the host's lineage display counts
them. Hermes keeps a rotation parent's originals in the closed parent and gives the child only the compressed handoff;
an in-place compaction archives the displayed history (`active = 0, compacted = 1`) and rewinds the originals of the
carried tail (`active = 0, compacted = 0`). The displayed rows are `active = 1 OR compacted = 1`, and the display
collapses rows with the same display identity: role, content (edge-stripped, as B2 keys it), timestamp and tool
fields, or a tool-calling assistant row's call ids. A carried handoff copy takes its original's timestamp, so it
collapses with it; a genuine repeat has its own timestamp. One insert batch stamps all its rows with one time, so two
identical sends in one batch share an identity: an identity counts as many times as the most displayed rows any one
segment holds of it, and a key's expected count is the sum of those over the lineage's distinct identities. Host summary rows (`_compressed_summary`) and user rows
that begin with LCM-X's own generated-carrier markers are not expected.

Stored rows are scored with the reliability harness's B2 instruments, unchanged: `multiset.score` (user and assistant
rows), `tool_calls.stored_keys`/`compare` (tool calls and results), `externalized.resolve` (externalized payloads) and
`multiset.split_surplus` (visible versus silent surplus).

| Output | Fields |
|---|---|
| `parity` | per lineage and in total (every scored lineage, #1931 bucket B children included): `expected_rows`, `stored_rows`, `missing_rows` (exact-content deficit), `surplus_rows`, `visible_surplus_rows`, `silent_surplus_rows`, `split_assistant_keys`, `externalized_unresolved`; `zero_row_lineages` (expected rows but none stored); per session: lineage, source, stored rows. |
| `coverage` | each floor's `value`, `floor`, `met`; `unknown` when its input is missing (never met). Compactions are LCM-X's `LCM compaction #` lines tagged with a scored session; delegated children are the ended, scored bucket A children; host sources leave out `subagent` and `tool`. `coverage_basis` names each basis and gives the store's depth-0 message leaves (a pass may write several, so they are not the count). |
| `uid_coverage` | per host source: rows stored, rows with a canonical `host_uid_bindings` row; `table absent` when the store has none. |
| `m3` | rows in the window's lineages that carry `generated_unproven` or are the `store_id` side of `possible_replay_of`, over the rows stored there; the rate and every store id. |
| `recorder` | `LCM ingest route:` lines per route; the #2305 rate (`persisted_batch` at cursor 0 over storing ingests); stripped links per replay pair; `recorder absent` when there is no line. |
| `exempted_surplus` | silent surplus rows exempted under #1674 or #2203, rows listed for a hand check, unattributed rows, store ids for each, and the exempted fraction against the 1% bar. The rules are `attribute_silent_surplus`. |
| `issue_1931` | buckets A (`source = 'subagent'`) and B (any other source) of ended non-rotation children: n, children with 0 rows, missing, silent surplus, lineage rows matching the host parent; `pass_A`; AUXILIARY sessions excluded. |
| `gate` | each bar's value and pass, and the verdict. |

Bars: no lineage with expected rows and none stored; no exact-content deficit; no unresolved externalized payload (the
unresolved row is also a deficit of the host row it held, as in B2); no unattributed silent surplus; exempted surplus
at most 1%; M3 at most 1%; every floor met; #1931 PASS(A). The verdict is `GATE FAIL` when a data bar fails, else
`INCOMPLETE` when a floor is unmet or the recorder is absent, else `GATE PASS PENDING HAND CHECK` when any row was
exempted (its store ids are listed per limit), else `GATE PASS`.

## Surplus limits and recorder routes
A silent surplus row is attributed only from the one `LCM ingest route:` line of its session whose stored range holds
it; none or several leave it unattributed. Route tokens are the recorder's (spaces become underscores): `foreground`,
`append`, `advanced_cursor`, `persisted_batch`, `skipped_batch`, `cold`, `session_end`, `warm_tail` and `S3`.

| Limit | Route | Also checked | Result |
|---|---|---|---|
| #1674 restart replay | `advanced_cursor`, cursor > 0 (a re-bound engine's reconciliation) | copy unstamped; an earlier copy of its key in the session stamped; a driver restart of the lineage before the recorder line | exempted; hand check without `--events` |
| #2203 cold-tail retry | `cold` or `S3` | copy unstamped | exempted |
| #2203 exit drain | `session_end` whose reason is a tail match rule (`digest`, `off`, `no_ledger`, `salt_changed`, `read_failed`) | copy unstamped | exempted |
| #849 composite | none: the recorder has no composite route | a user row equal to another prompt, `\n\n`, and a user row stored apart in the session | hand check; stays silent |
| #1961 carrier merge | `foreground` at cursor 0 (a warm engine) | the carrier merge needs the host view | hand check; stays silent |
| #2305 | `persisted_batch` at cursor 0 | | never exempted |

`session_end` without a tail rule is an ordinary session-end flush, and `warm_tail` is a tail retry that is not the
exit drain; neither is a section 5 shape, so they are not attributed. Rows listed for a hand check count as silent
surplus.

## AUXILIARY sessions
A session LCM-X classes AUXILIARY stores nothing, so it is not scored. It is excluded when LCM-X's
`LCM reclassified session <id> as auxiliary at first ingest` line names it, or when the session decision line
`LCM session decision: session=<id> ... decision=auxiliary reason=<token>` names it with reason `background_review` or
`toolsets_memory_only`, whatever its source. A session classed auxiliary for any other reason (`frame_marker`,
logged for an unproven child: the #1931 bypass) stays scored, in bucket A when it is a subagent child, and is listed
under `issue_1931.auxiliary_unproven`.

## What it does not prove
- The run itself: that the driver covered every reliability shape, that the host and rc trees are the ones named,
  and the latency, lock and class D bars. The receipt states those.
- Every exemption is a hand check (hence `GATE PASS PENDING HAND CHECK`): the audit checks the route, cursor,
  `observed_at` and restart conditions, not the view-order conditions of #1674 and #2203 (no mid-history insertion
  or reordering; a #2305 insert drained through a retry route is not #2203). #849 and #1961 rows are never exempted
  by the audit: their routes cannot be told apart from ordinary ones.
- The expected count relies on the host carrying a handoff copy's timestamp from its original. The host does this
  best-effort, and a copy it rewrote (a pruned tool result) keeps no match: such a copy counts as one more expected
  row, so the audit errs toward reporting a deficit, never toward hiding one. A host without the `compacted` column
  counts every row as displayed. The host's user-handoff normalisation is not repeated; LCM-X's carriers are
  excluded by marker instead.
- A user row the host persisted with an LCM-X carrier is not expected; if the host held a user's text only inside
  such a row, its loss would not show here.
- A log without the session decision line (builds before it) can show an auxiliary child as a child with 0 rows.
- More than 10 silent copies of one key: the rest are counted but not listed, so they stay unattributed.
