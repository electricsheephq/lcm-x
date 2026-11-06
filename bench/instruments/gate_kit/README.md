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
  is refused (exit 2), as is one with a non-empty (hot) rollback `-journal`; a symlinked path is checked beside its target: checkpoint or roll it back first (open it once after the run has stopped, or copy it with its `-wal` and
  checkpoint the copy).
- `--out` is refused (exit 2) when it is, or resolves to, an input database, log, events or externalized payload file,
  or anywhere under the audited home or a database's directory. The result is written to a temporary file in `--out`'s
  directory and renamed over it.
- `--since` (a naive ISO time is UTC): a lineage is scored when any member started at or after it, and only those
  members count: a rotation child in the window is scored without its earlier parent's rows or the copies of them it
  carries. Stored rows under a session the host does not know count when ingested at or after it. Driver events,
  compaction lines and recorder lines count only for audited sessions, at or after it.
- `--log`: the run's Hermes logs, in a Hermes format (`<asctime> <LEVEL> [<session>] <logger>: <message>`, the
  verbose `<asctime> - <logger> - <LEVEL> [<session>] - <message>`, or the ACP host's stderr
  `<asctime> [<LEVEL>] <logger>: <message>`, which has no session field). A pattern counts only at the start of a
  message logged by LCM-X's own logger (a `hermes_lcm` module); a compaction or recorder line logged to two files counts once. A
  session-less copy matching a session-bearing compaction's timestamp and message is dropped; distinct sessions stay
  distinct. A line without a session takes the `session=` its message names, else the session of the nearest unused driver
  `compress_request` it falls within (2 s before to 120 s after the request); else it counts toward `compactions` but
  not toward `compactions_then_restart` (`coverage_basis.compactions_without_session`).
- Log times keep their milliseconds, so a restart and an ingest in the same second keep their order.
- `--log-tz`: the zoneinfo name the log times are written in (for example `UTC` or `Europe/Berlin`). Default: the
  audit machine's local zone. Pass the run host's zone when they differ: it decides whether a compaction or a
  recorder line came before a driver restart event. An unknown zone exits 2.
- `--events`: the driver's event log (`drive.py`, below), one JSON object per line: `{"t": epoch, "kind":
  "restart_cold" | "restart_cold_shared" | "restart_warm" | "undo_resend" | "delegate_prompt" |
  "compress_request" | "compress_request_failed", "session": id, "detail":
  short non-content text}`. Lines of any other kind (the driver also writes `turn`, `source2_session`,
  `context_window`, `host_exit`, `error` and a final `summary`) are ignored.
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
collapses with it; a genuine repeat has its own timestamp (the host offsets each row of an insert batch by 1e-6 s). An
identity counts as many times as the most displayed rows any one segment holds of it, and a key's expected count is
the sum of those over the lineage's distinct identities. With `--since`, a scored segment expects only the copies of
an identity beyond the most any earlier, unscored segment holds. Host summary rows (`_compressed_summary`) are not
expected. A user row that begins with LCM's summary carrier (`[<Label> Summary (dN, node N)]` parts, each ending
`[Expand for details: ...]`, as the host merges it with the next user row) is the user text behind the carrier, as
LCM-X matches it, only when every part matches its node in the store byte for byte (LCM-X's verifier, engine.py
`_verified_lcm_summary_prefix`; the node's session in the row's lineage); otherwise the whole row is ordinary user
content. A merged row is one occurrence with a row of the same text and timestamp. A merged row of a text the lineage
also holds at another timestamp may be the host's merged copy (a rotation handoff stamps it anew) or a resend: it is
not counted and is listed for a hand check (`merged_carrier_repeat`). A row that is only a verified carrier, or begins with one of LCM-X's prefix carriers (objective,
task list, overflow recovery), is not expected.

Stored rows are scored with the reliability harness's B2 instruments, unchanged: `multiset.score` (user and assistant
rows), `tool_calls.stored_keys`/`compare` (tool calls and results), `externalized.resolve` (externalized payloads) and
`multiset.split_surplus` (visible versus silent surplus).

| Output | Fields |
|---|---|
| `parity` | per lineage and in total (every scored lineage, #1931 bucket B children included): `expected_rows`, `stored_rows`, `missing_rows` (exact-content deficit), `surplus_rows`, `visible_surplus_rows`, `silent_surplus_rows`, `split_assistant_keys`, `externalized_unresolved`; `zero_row_lineages` (expected rows but none stored); per session: lineage, source, stored rows. |
| `coverage` | each floor's `value`, `floor`, `met`; `unknown` when its input is missing (never met). Compactions are LCM-X's `LCM compaction #` lines tagged with a scored session; delegated children are the ended, scored bucket A children; host sources count scored sessions only, leaving out `subagent` and `tool`. `coverage_basis` names each basis and gives the store's depth-0 message leaves (a pass may write several, so they are not the count). |
| `uid_coverage` | per host source: rows stored, rows with a canonical `host_uid_bindings` row; `table absent` when the store has none. |
| `m3` | rows in the window's lineages that carry `generated_unproven` or are the `store_id` side of `possible_replay_of`, over the rows stored there; the rate and every store id. An `undo_retained` row leaves the numerator for `possible_replay_of` only. |
| `undo_retained` | `rows` and `store_ids`: stored rows that are a host undo's retained occurrence (`undo_retained_rows`). A TUI undo soft-deletes the undone rows in the host (`active = 0`, `compacted = 0`), so they are not expected; LCM-X keeps its copy and stores the resend as a new occurrence. A stored row is excused when the host holds such an undone row of the same session and row key and the stored row's `observed_at` equals that row's timestamp; each undone row excuses one stored copy, and a row with NULL `observed_at` is never excused. An inactive row with the same session, key and timestamp as a displayed row (the original an in-place compaction, `replace_messages` or `deactivate_message` leaves beside its live clone) excuses nothing. Excused rows are neither expected nor surplus and stay out of M3's numerator (they count in its denominator). `host_rewind_count`: the host's `sessions.rewind_count` per audited session, as a cross-check. |
| `recorder` | `LCM ingest route:` lines per route; the #2305 rate (`persisted_batch` at cursor 0 over storing ingests); stripped links per replay pair; `recorder absent` when there is no line. |
| `exempted_surplus` | silent surplus rows exempted under #1674 or #2203, rows listed for a hand check, unattributed rows, store ids for each, and the exempted fraction against the 1% bar. The fraction counts distinct store rows: a row with content and a tool call is one row under two keys. `unattributed_silent_rows` (the bar's value) counts key copies; `unattributed_silent_distinct_rows` counts store rows. The rules are `attribute_silent_surplus`. |
| `issue_1931` | buckets A (`source = 'subagent'`) and B (any other source) of ended non-rotation children: n, children with 0 rows, missing, silent surplus, lineage rows matching the host parent; `pass_A`; AUXILIARY sessions excluded. |
| `gate` | each bar's value and pass, the verdict, and `hand_checks`: each reason the verdict is held for a hand check (see "Hand checks"). |

Bars: no lineage with expected rows and none stored; no exact-content deficit; no unresolved externalized payload (the
unresolved row is also a deficit of the host row it held, as in B2); no unattributed silent surplus; exempted surplus
at most 1%; M3 at most 1%; every floor met; #1931 PASS(A). The verdict is `GATE FAIL` when a data bar fails, else
`INCOMPLETE` when a floor is unmet or the recorder is absent, else `GATE PASS PENDING HAND CHECK` when `hand_checks`
names any reason, else `GATE PASS`.

## Surplus limits and recorder routes
A silent surplus row is attributed only from the one `LCM ingest route:` line of its session whose stored range holds
it; none or several leave it unattributed. Route tokens are the recorder's (spaces become underscores): `foreground`,
`append`, `advanced_cursor`, `persisted_batch`, `skipped_batch`, `cold`, `session_end`, `warm_tail` and `S3`.

| Limit | Route | Also checked | Result |
|---|---|---|---|
| #1674 restart replay | `advanced_cursor`, cursor > 0 (a re-bound engine's reconciliation) | copy unstamped; view order; an earlier copy of its key in the session stamped; a driver restart of the lineage before the recorder line | exempted; hand check without `--events` |
| #2203 cold-tail retry | `cold` or `S3` | copy unstamped; view order | exempted |
| #2203 exit drain | `session_end` whose reason is a tail match rule (`digest`, `off`, `no_ledger`, `salt_changed`, `read_failed`) | copy unstamped; view order | exempted |
| #849 composite | none: the recorder has no composite route | a user row equal to another prompt, `\n\n`, and a user row stored apart in the session | hand check; stays silent |
| #1961 carrier merge | `foreground` at cursor 0 (a warm engine) | the carrier merge needs the host view | hand check; stays silent |
| #2305 | `persisted_batch` at cursor 0 | | never exempted |

View order (`view_order`): the session's stored rows before the copy each take one host row of the session
(displayed or undone, in id order, carried clones dropped), each host row at most once, in order, with no exception.
When no such matching exists (a host insertion stored before the copy, as a #2305 insert drained through a retry
route, a reordering, or an earlier re-stored copy) the row stays unattributed. When repeated text allows more than one
matching, the row is listed under `order_ambiguous`: a hand check, not an exemption (it counts in the fraction). #2203
also needs an earlier copy of the key in the session. A rewrite that stores two rows again exempts only the first:
the second's prefix holds the first copy, which no host row takes.

`session_end` without a tail rule is an ordinary session-end flush, and `warm_tail` is a tail retry that is not the
exit drain; neither is a section 5 shape, so they are not attributed. Rows listed for a hand check count as silent
surplus.

## Hand checks
Each entry of `gate.hand_checks` names a reason; any entry makes the verdict `GATE PASS PENDING HAND CHECK` (unless a
data bar fails or a floor is unmet). The detectors marked "detector" only add an entry: they never change a count.

| Reason | Ids | What to check |
|---|---|---|
| `exempted_1674`, `exempted_2203` | store ids | the exempted rows (see "Surplus limits") |
| `view_order_ambiguous` | store ids | repeated text allowed more than one in-order matching |
| `merged_carrier_repeat` | host ids | a merged row of a text the lineage also holds at another time was not counted: a merged copy or a resend |
| `merged_carrier_window_repeat` (detector) | host ids | with `--since`, a merged row in the window whose text an earlier, unscored segment holds: the window does not expect the parent's occurrence, so a resend and a replay look alike |
| `merged_carrier_same_stamp_multiplicity` (detector) | host ids | two or more merged rows of one session with one text and one time, counted as one occurrence |
| `stamp_swap` (detector) | store ids | parity matches by content: for a key the host holds more than once in a session, every stored copy is stamped and their `observed_at` values are not among the host's times of that key (as multisets), so one occurrence may stand in for another (an undone row kept twice while its resend is lost). Unstamped copies are not judged. |

Known false FAIL, cleared by hand and not fixed: a #2203 or #1674 re-store of several rows (host A, B, C; stored A, B,
C, A, B, C) exempts at most the first copy; the later copies' prefix holds the earlier copy, which no host row takes,
so rows 5 and 6 stay unattributed and the gate fails. It fails safe.

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
  `observed_at`, restart and view-order conditions, but view order is read from the host's final rows, not from the
  view the ingest saw (the store keeps no view). #849 and #1961 rows are never exempted by the audit: their routes
  cannot be told apart from ordinary ones.
- The expected count relies on the host carrying a handoff copy's timestamp from its original. The host does this
  best-effort, and a copy it rewrote (a pruned tool result) keeps no match: such a copy counts as one more expected
  row, so the audit errs toward reporting a deficit, never toward hiding one. A host without the `compacted` column
  counts every row as displayed. The host's user-handoff normalisation is not repeated.
- `--state-db` may be a symlink: externalized payloads are read beside its target. A row
  that begins with a prefix carrier (objective, task list, overflow recovery) is not expected, since where the carrier
  ends is not marked: if the host held a user's text only behind such a carrier, its loss would not show here.
- A log without the session decision line (builds before it) can show an auxiliary child as a child with 0 rows.
- More than 10 silent copies of one key: the rest are counted but not listed, so they stay unattributed.

## The driver (`drive.py`)
`drive.py` runs the driven run against a prepared Hermes home and writes the events log the audit reads. It plans the
operations from the seed and every section 5 floor x1.2 (rounded up), then runs them one at a time through two real
host processes:

| Entry point | Operations |
|---|---|
| `hermes acp` (`reliability/acp_driver.py`) | turns from a fixed prompt bank (plain questions, `read_file`/todo/`lcm_grep` tool prompts, word-for-word repeats of earlier tool prompts, long pasted text), `/compress` after a long-text turn, `delegate_task` prompts, cancel-and-resend (`session/cancel` while the prompt is in flight, then the same prompt), cold restarts (SIGKILL the process group, a new process, `session/load`) |
| TUI gateway (`python -m tui_gateway.entry`) | the second host source (`tui` sessions), warm restarts (`session.close`, which ends the session in the context engine, then `session.resume` in the same process), undo-and-resend (`session.undo`, then the same prompt), `session.compress` |

ACP has no session end and no undo, which is why the TUI gateway carries those three. Every planned restart follows a
compaction in its session and is followed by more turns. Delegated children are counted from `state.db` (read-only)
using the audit's ended, non-rotation subagent-child predicate, excluding proven AUXILIARY sessions in the host logs;
when they fall short after the plan, the driver sends more delegate prompts, up to twice the target in total
attempts (including failed attempts and those made by the plan). The summary separates attempts from completed prompts.

```
python -m bench.instruments.gate_kit.drive --hermes-python <host venv python> --hermes-src <host tree> \
  --home <HERMES_HOME> --workdir <dir> --events <events.jsonl> [--turns N] [--children N] [--seed 2055] \
  [--max-hours 6] [--prompt-timeout 600] [--cancel-after 2] [--append-events] [--dry-run]
```
- `--dry-run` prints the targets and the planned counts and exits. `--turns`/`--children` can raise a target, never
  lower it below the floor x1.2.
- A `--home`, `--workdir` or `--events` at or under the live `~/.hermes` is refused.
- All CLI paths become absolute against the invoking directory before use, including the host Python and source tree;
  the Python executable's symlink is preserved so its virtualenv and sibling `hermes` launcher remain in use.
- A non-empty `--events` file is refused with exit 2 unless `--append-events` explicitly permits retaining its events.
- The host processes get the invoking environment with `HERMES_HOME` set to the home and `HOME`/`TMPDIR` set to
  `<workdir>/host-home`; their stderr goes to `<workdir>/host-logs/`. The read_file prompts name files the driver
  writes under `<workdir>/notes/`.
- A failed operation is an `error` event (a model error, a cancel that came too late, a JSON-RPC error with its code, a
  host process that died) and the run continues; nothing is retried silently. A process that died is replaced on its
  next operation; stored TUI sessions are resumed in the replacement gateway before use. A shared ACP process kill
  records one `restart_cold` for the planned slot only after its load succeeds. Siblings get `restart_cold_shared`
  at the kill; crash-affected ACP and TUI sessions get it after reload/resume. Shared events support restart-copy
  attribution but never the cold-restart floor. `restart_warm` requires a successful close and resume; `undo_resend`
  requires a successful first send (ACP cancellation or TUI completion) and a successful second send. A rejected
  first TUI send skips undo. After a successful close, a failed resume leaves the slot unbound so its next operation
  resumes it again. `--max-hours` stops the plan and still writes the summary.
- `compress_request` is recorded before the triggering long-text turn; `compress_result` records the later manual
  compression outcome separately, so the audit's attribution window starts before automatic compaction can occur.
  A failed long-text turn writes `compress_request_failed`; the audit excludes that request from attribution.
  Inactive-host scans restart at byte zero when a log shrinks below the saved offset.
- The events log never holds prompt or reply text: a prompt is its bank index (a test plants a sentinel to hold this).
  The last line is `{"kind": "summary", "planned", "executed", "errors", "delegate", "context_length", ...}`.

**Preparing the home.** The driver never reads, prints or edits keys or model config. The operator prepares:
- the model through the home's own provider config (`model:` in `config.yaml` and the home's `.env`);
- a small context window, `model.context_length` in `config.yaml`, so compactions follow from the long-text turns;
  the driver reports the value the host uses (ACP `/context`) as a `context_window` event and in the summary;
- `delegate_task` enabled: the `delegation` toolset in `platform_toolsets` for `acp` (and `tui`), with `file`, `todo`
  and `context_engine`;
- LCM-X installed and enabled as the context engine, with `plugins.load_timeout_seconds: 60` in `config.yaml`: under
  the default 10 s a slow plugin load leaves a host on the built-in compressor, writing nothing to `lcm.db`;
- a host tree whose TUI gateway has `session.close`, `session.resume` and `session.undo`, run from the host venv.

Then checkpoint both databases once the run has stopped (the audit refuses a non-empty `-wal`; see "Run" above) and
audit the run: `audit.py --state-db <home>/state.db --lcm-db <home>/lcm.db --log <home>/logs/agent.log --log
<workdir>/host-logs/acp.stderr --events <events.jsonl> --out <result.json>`. Audit only a run whose events show no
`lcm_inactive` error: after every operation the driver scans new bytes in each host's stderr and `<home>/logs/agent.log` for
LCM-X's "LCM-X is NOT active in this process" notice and the host's "Context engine 'lcm-x' not found" fallback; on
either it records `{"kind": "error", "detail": "lcm_inactive entry=<acp|tui> source=<file>"}`, stops the run and exits
3, because that host's rows would otherwise read as a loss. The ACP host logs only to its stderr, as
`<asctime> [<LEVEL>] <logger>: <message>` with no session field (acp_adapter/entry.py `_setup_logging`); the TUI
gateway logs to `<home>/logs/agent.log`. The run ends with stdin EOF to both hosts (a `host_exit` event each), so the
gateway ends its sessions before the audit reads the home.

**Exit-3 recovery.** Raise `plugins.load_timeout_seconds` and rerun with a new `--events` path, preferably in a fresh
prepared home. The audit scores every session in `state.db`, not only those in the events file, so in a retained home
record the retry's start time and pass it to the audit as `--since`; otherwise the failed run's sessions read as
deficits. Existing log bytes are excluded from inactive-host detection. Do not combine failed-run coverage with a fresh run:
use `--append-events` only when intentionally retaining the earlier events and accounting for them in the audit.

**Tests** (`tests/test_drive.py`): the planner (floors x1.2, deterministic, repeats, restart placement), the executor
against stub processes (no prompt or reply text in the events, a late cancel is an error), the events contract with
`audit.py --events`, and a host-backed smoke: 12 operations, one of each kind, through a real `hermes acp` and TUI
gateway against the localhost fake provider (`reliability/fake_provider.py`, under `sandbox-exec` where present), then
the audit over the home. The smoke takes its host from the reliability hosts file (`$LCM_RELIABILITY_HOSTS` or the
default path; `$LCM_GATE_KIT_HOST` picks one by name) and skips with the reason when none is configured.

**Claim class: `instrument`.** The driver produces the run; it proves nothing about the gate by itself.
