# Gate kit: the v0.30.0 "nothing lost" audit

`audit.py` reads one driven run (docs/exactly-once-ingest.md section 5, "Gate (v0.30.0)" and "The driven run") and
reports every gate count in one JSON file. It works on any Hermes home: the host's `state.db`, the LCM-X store and
the run's log files, including a reliability-harness cell's kept `db/` copies.

**Claim class: `instrument`.** It measures; it does not prove the gate by itself. The release manager's receipt
cites its output next to the run's identity (rc tree, host sha, run window) and the hand checks listed below.

## Run
```
python -m bench.instruments.gate_kit.audit --state-db <hermes home>/state.db --lcm-db <lcm.db> \
  [--log <file> ...] [--since <ISO-8601 or epoch seconds>] [--events <driver events.jsonl>] --out <result.json>
```
- Both databases are opened with read-only `mode=ro` URIs; nothing is copied or written beside them.
- `--since`: only host sessions started at or after it (a naive ISO time is UTC). Stored rows under a session the
  host does not know count when ingested at or after it.
- `--log`: the run's Hermes/LCM logs (`<asctime> <LEVEL> [<session>] <logger>: <message>`). Log times are read in the
  audit machine's local time zone; set `TZ` to the run host's zone when they differ.
- `--events`: the driver's event log, one JSON object per line: `{"t": epoch, "kind": "restart_cold" | "restart_warm"
  | "undo_resend" | "delegate_prompt" | "compress_request", "session": id}`.
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
| `recorder` | `LCM ingest route:` lines per route; the #2305 rate (cursor-0 persisted batches over storing ingests); stripped links per replay pair; `recorder absent` when there is no line. |
| `exempted_surplus` | silent surplus rows attributed to #1674, #2203 or #849, rows needing a manual #1961 check, unattributed rows, store ids for each, and the exempted fraction against the 1% bar. The rules are `attribute_silent_surplus`. |
| `issue_1931` | buckets A (`source = 'subagent'`) and B (any other source) of ended non-rotation children: n, children with 0 rows, missing, silent surplus, lineage rows matching the host parent; `pass_A`; AUXILIARY sessions excluded. |
| `gate` | each bar's value and pass, and the verdict: `GATE FAIL` when a data bar fails, else `INCOMPLETE` when a floor is unmet or the recorder is absent, else `GATE PASS`. |

## What it does not prove
- The run itself: that the driver covered every reliability shape, that the host and rc trees are the ones named,
  and the latency, lock and class D bars. The receipt states those.
- Attribution is from recorder lines only, and checks the route, cursor and `observed_at` conditions. The view-order
  conditions of #1674 and #2203, #1961's carrier merge and #849's "next prompt also stored" are hand checks; a
  #1961 candidate stays silent surplus until checked. The recorder's route names are constants at the top of
  `audit.py` and must match the recorder as merged.
- The host count undercounts a key only when its genuine repeats are spread over sessions that do not re-issue each
  other's rows; Hermes re-issues every durable row on a rotation, so a lineage's latest session holds them all.
- A user row the host persisted with an LCM-X carrier is not expected; if the host held a user's text only inside
  such a row, its loss would not show here.
- AUXILIARY sessions are recognised only by `LCM reclassified session <id> as auxiliary at first ingest`; the #1931
  AUXILIARY class logs no line of its own, so an unlogged auxiliary child appears as a child with 0 rows.
- More than 10 silent copies of one key: the rest are counted but not listed, so they stay unattributed.
