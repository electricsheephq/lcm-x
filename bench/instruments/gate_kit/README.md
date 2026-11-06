# Gate kit: the v0.30.0 "nothing lost" audit

`audit.py` reads one driven run (docs/exactly-once-ingest.md section 5, "Gate (v0.30.0)" and "The driven run") and
reports every gate count in one JSON file. It works on any Hermes home: the host's `state.db`, the LCM-X store and
the run's log files, including a reliability-harness cell's kept `db/` copies.

**Claim class: `instrument`.** It measures; it does not prove the gate by itself. The release manager's receipt
cites its output next to the run's identity (rc tree, host sha, run window) and the hand checks listed below.

## Run
```
python -m bench.instruments.gate_kit.audit --state-db <hermes home>/state.db --lcm-db <lcm.db> \
  [--log <file> ...] [--log-tz <zone>] [--since <ISO-8601 or epoch seconds>] [--events <driver events.jsonl>] \
  --out <result.json>
```
- Both databases are opened with read-only `mode=ro` URIs; nothing is copied or written beside them.
- `--since`: only host sessions started at or after it (a naive ISO time is UTC). Stored rows under a session the
  host does not know count when ingested at or after it.
- `--log`: the run's Hermes/LCM logs (`<asctime> <LEVEL> [<session>] <logger>: <message>`).
- `--log-tz`: the zoneinfo name the log times are written in (for example `UTC` or `Europe/Berlin`). Default: the
  audit machine's local zone. Pass the run host's zone when they differ: it decides whether a compaction came before a
  driver restart event. An unknown zone exits 2.
- `--events`: the driver's event log (`drive.py`, below), one JSON object per line: `{"t": epoch, "kind":
  "restart_cold" | "restart_warm" | "undo_resend" | "delegate_prompt" | "compress_request", "session": id, "detail":
  short non-content text}`. Lines of any other kind (the driver also writes `turn`, `source2_session`,
  `context_window`, `error` and a final `summary`) are ignored.
- Exit 0 with the result written (the verdict is inside); exit 2 on an unreadable input. Stdout is a four-line summary.

The output holds counts, rates, booleans, session ids and store ids only. Message content, tool arguments and
results are compared in memory and never written (a test plants a sentinel in every content field to hold this).

## What it measures
Expected rows come from the host's own `state.db`, per lineage (a session plus its rotation children: a child whose
parent ended with `end_reason = 'compression'` on the same source). A key's expected count is the reliability
harness's host count (`scorers/host_parity.py`): the most active rows one session of the lineage holds, else 1 when
the lineage holds it only on inactive rows. Host summary rows (`_compressed_summary`) and user rows that begin with
LCM-X's own generated-carrier markers are not expected. Stored rows are scored with the harness's B2 instruments,
unchanged: `multiset.score` (user and assistant rows), `tool_calls.stored_keys`/`compare` (tool calls and results),
`externalized.resolve` (externalized payloads) and `multiset.split_surplus` (visible versus silent surplus).

| Output | Fields |
|---|---|
| `parity` | per lineage and in total: `expected_rows`, `stored_rows`, `missing_rows` (exact-content deficit), `surplus_rows`, `visible_surplus_rows`, `silent_surplus_rows`, `split_assistant_keys`, `externalized_unresolved`; `zero_row_lineages`; per session: lineage, source, stored rows. A lineage with `gates: false` is a #1931 bucket B child: reported, left out of the totals. |
| `coverage` | each floor's `value`, `floor`, `met`; `unknown` when its input is missing (never met). Compactions are `LCM compaction #` log lines; `coverage_basis` names each basis and also gives the store's depth-0 message leaves (a pass may write several, so they are not the count). |
| `uid_coverage` | per host source: rows stored, rows with a canonical `host_uid_bindings` row; `table absent` when the store has none. |
| `m3` | rows in the window's lineages that carry `generated_unproven` or are the `store_id` side of `possible_replay_of`, over the rows stored there; the rate and every store id. |
| `recorder` | `LCM ingest route:` lines per route; the #2305 rate (`persisted_batch` at cursor 0 over storing ingests); stripped links per replay pair; `recorder absent` when there is no line. |
| `exempted_surplus` | silent surplus rows attributed to #1674 or #2203, rows listed for a #849 or #1961 hand check, unattributed rows, store ids for each, and the exempted fraction (#1674 and #2203 only) against the 1% bar. The rules are `attribute_silent_surplus`. |
| `issue_1931` | buckets A (`source = 'subagent'`) and B (any other source) of ended non-rotation children: n, children with 0 rows, missing, silent surplus, lineage rows matching the host parent; `pass_A`; AUXILIARY sessions excluded. |
| `gate` | each bar's value and pass, and the verdict: `GATE FAIL` when a data bar fails, else `INCOMPLETE` when a floor is unmet or the recorder is absent, else `GATE PASS`. |

## Surplus limits and recorder routes
A silent surplus row is attributed only from the one `LCM ingest route:` line of its session whose stored range holds
it; none or several leave it unattributed. Route tokens are the recorder's (spaces become underscores): `foreground`,
`append`, `advanced_cursor`, `persisted_batch`, `skipped_batch`, `cold`, `session_end`, `warm_tail` and `S3`.

| Limit | Route | Also checked | Result |
|---|---|---|---|
| #1674 restart replay | `advanced_cursor`, cursor > 0 (a re-bound engine's reconciliation) | copy unstamped, an earlier copy of its key in the session stamped | exempted |
| #2203 cold-tail retry | `cold` or `S3` | copy unstamped | exempted |
| #2203 exit drain | `session_end` whose reason is a tail match rule (`digest`, `off`, `no_ledger`, `salt_changed`, `read_failed`) | copy unstamped | exempted |
| #849 composite | none: the recorder has no composite route | a user row equal to another prompt, `\n\n`, and a user row stored apart in the session | hand check; stays silent |
| #1961 carrier merge | `foreground` at cursor 0 (a warm engine) | the carrier merge needs the host view | hand check; stays silent |
| #2305 | `persisted_batch` at cursor 0 | | never exempted |

`session_end` without a tail rule is an ordinary session-end flush, and `warm_tail` is a tail retry that is not the
exit drain; neither is a section 5 shape, so they are not attributed.

## What it does not prove
- The run itself: that the driver covered every reliability shape, that the host and rc trees are the ones named,
  and the latency, lock and class D bars. The receipt states those.
- Attribution checks the route, cursor and `observed_at` conditions only. The view-order conditions of #1674 and
  #2203 (no mid-history insertion or reordering) and the restart in the log for #1674 are hand checks. #849 and
  #1961 rows are never exempted by the audit: their routes cannot be told apart from ordinary ones.
- The host count undercounts a key only when its genuine repeats are spread over sessions that do not re-issue each
  other's rows; Hermes re-issues every durable row on a rotation, so a lineage's latest session holds them all.
- A user row the host persisted with an LCM-X carrier is not expected; if the host held a user's text only inside
  such a row, its loss would not show here.
- AUXILIARY sessions are recognised by `LCM reclassified session <id> as auxiliary at first ingest` and by the
  session decision line `LCM session decision: session=<id> ... decision=auxiliary`. A log without the decision line
  (builds before it) can show an auxiliary child as a child with 0 rows.
- More than 10 silent copies of one key: the rest are counted but not listed, so they stay unattributed.

## The driver (`drive.py`)
`drive.py` runs the driven run against a prepared Hermes home and writes the events log the audit reads. It plans the
operations from the seed and every section 5 floor x1.2 (rounded up), then runs them one at a time through two real
host processes:

| Entry point | Operations |
|---|---|
| `hermes acp` (`reliability/acp_driver.py`) | turns from a fixed prompt bank (plain questions, `read_file`/todo/`lcm_grep` tool prompts, word-for-word repeats of earlier tool prompts, long pasted text), `/compress` after a long-text turn, `delegate_task` prompts, cancel-and-resend (`session/cancel` while the prompt is in flight, then the same prompt), cold restarts (SIGKILL the process group, a new process, `session/load`) |
| TUI gateway (`python -m tui_gateway.entry`) | the second host source (`tui` sessions), warm restarts (`session.close`, which ends the session in the context engine, then `session.resume` in the same process), undo-and-resend (`session.undo`, then the same prompt), `session.compress` |

ACP has no session end and no undo, which is why the TUI gateway carries those three. Every restart follows a
compaction in its session and is followed by more turns. Delegated children are counted from `state.db` (read-only);
when they fall short after the plan, the driver sends more delegate prompts, up to twice the target.

```
python -m bench.instruments.gate_kit.drive --hermes-python <host venv python> --hermes-src <host tree> \
  --home <HERMES_HOME> --workdir <dir> --events <events.jsonl> [--turns N] [--children N] [--seed 2055] \
  [--max-hours 6] [--prompt-timeout 600] [--cancel-after 2] [--dry-run]
```
- `--dry-run` prints the targets and the planned counts and exits. `--turns`/`--children` can raise a target, never
  lower it below the floor x1.2.
- A `--home`, `--workdir` or `--events` at or under the live `~/.hermes` is refused.
- The host processes get the invoking environment with `HERMES_HOME` set to the home and `HOME`/`TMPDIR` set to
  `<workdir>/host-home`; their stderr goes to `<workdir>/host-logs/`. The read_file prompts name files the driver
  writes under `<workdir>/notes/`.
- A failed operation is an `error` event (a model error, a cancel that came too late, a JSON-RPC error with its code, a
  host process that died) and the run continues; nothing is retried silently. A process that died is replaced on its
  next operation. `--max-hours` stops the plan and still writes the summary.
- The events log never holds prompt or reply text: a prompt is its bank index (a test plants a sentinel to hold this).
  The last line is `{"kind": "summary", "planned", "executed", "errors", "delegate", "context_length", ...}`.

**Preparing the home.** The driver never reads, prints or edits keys or model config. The operator prepares:
- the model through the home's own provider config (`model:` in `config.yaml` and the home's `.env`);
- a small context window, `model.context_length` in `config.yaml`, so compactions follow from the long-text turns;
  the driver reports the value the host uses (ACP `/context`) as a `context_window` event and in the summary;
- `delegate_task` enabled: the `delegation` toolset in `platform_toolsets` for `acp` (and `tui`), with `file`, `todo`
  and `context_engine`;
- LCM-X installed and enabled as the context engine;
- a host tree whose TUI gateway has `session.close`, `session.resume` and `session.undo`, run from the host venv.

Then audit the run: `audit.py --state-db <home>/state.db --lcm-db <home>/lcm.db --log <home>/logs/agent.log --log
<workdir>/host-logs/acp.stderr --events <events.jsonl> --out <result.json>`. The ACP host logs only to its stderr, as
`<asctime> [<LEVEL>] <logger>: <message>` with no session field (acp_adapter/entry.py `_setup_logging`); the TUI
gateway logs to `<home>/logs/agent.log`. The run ends with stdin EOF to both hosts (a `host_exit` event each), so the
gateway ends its sessions before the audit reads the home.

**Tests** (`tests/test_drive.py`): the planner (floors x1.2, deterministic, repeats, restart placement), the executor
against stub processes (no prompt or reply text in the events, a late cancel is an error), the events contract with
`audit.py --events`, and a host-backed smoke: 12 operations, one of each kind, through a real `hermes acp` and TUI
gateway against the localhost fake provider (`reliability/fake_provider.py`, under `sandbox-exec` where present), then
the audit over the home. The smoke takes its host from the reliability hosts file (`$LCM_RELIABILITY_HOSTS` or the
default path; `$LCM_GATE_KIT_HOST` picks one by name) and skips with the reason when none is configured.

**Claim class: `instrument`.** The driver produces the run; it proves nothing about the gate by itself.
