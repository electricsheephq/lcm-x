# LCM-X reliability harness (R1): the in-process real-Hermes matrix

Runs the REAL Hermes turn loop (`AIAgent.run_conversation`, the plugin loader, SessionDB, the host
ContextCompressor) under each host's own python, against a deterministic scripted provider, over a
host x cell x lcm-x-ref matrix, and scores every cell against mechanical bars. $0, no network.

**Claim class: `advisory` / `code_green_local`.** A PASS proves that this lcm-x tree, on this host sha,
under this scripted in-process scenario, meets the bars. It does not prove behaviour under real models,
the real ACP/gateway processes, real transports or customer boxes.

## Run
```
uv run --no-project python bench/instruments/reliability/run_matrix.py \
  --hosts-file <hosts.local.json> --hosts ref-0.21.5,customer-0.21.2 \
  --plugin-ref origin/main[,v0.24.2,...] --cells 'crash-*,baseline/*' | all --jobs 8 --out <dir> [--keep-homes] [--keep-dbs none|fail|all] [--scratch-root <dir>] \
  [--lcm-env LCM_KEY=VAL ...]
```
- Hosts file: `--hosts-file`, else `$LCM_RELIABILITY_HOSTS`, else the host-prep lane's file; see
  `hosts.example.json`. A host whose path, lexical or resolved, is under the live `~/.hermes` is refused.
- Host identity is verified before any cell runs: `git rev-parse HEAD` == sha with no modified, untracked or
  ignored-`*.py` file (every git command must succeed) for a git tree; for an exported tree, `tree-manifest.json` next to it (written at host prep by
  `python hosts.py --write-manifest <src> --sha <sha> --git-repo <repo>`, which proves the tree equals
  `git archive <sha>`) must match the sha and the tree's current hash. An unverified host's cells are ERROR.
- Each ref is exported once with `git archive` into `<out>/plugins/<sha12>/`; the plugin dir name,
  `plugins.enabled` entry and engine name are read from that tree (v0.23.x = `hermes-lcm`/`lcm`).
- Per cell: `<out>/cells/<host>/<sha12>/<cell-slug>/` holds cell.json, transcript.jsonl, phase-*.json,
  probe logs and verdict.json: the scored outputs only. The cell's Hermes home (state.db, lcm.db), its HOME/TMPDIR
  and the `db/` sqlite backup-API copies the scorers read live in a private scratch dir under `--scratch-root`
  (default `$TMPDIR`), deleted once the cell is scored, an ERROR or a harness failure included. A full state.db is
  several GB per cell, so no Hermes database is left in `<out>` by default.
- Debugging: `--keep-dbs fail` keeps the `db/` copies of non-PASS cells and of a PASS that carries host-parity
  licences, `--keep-dbs all` those of every cell, and `--keep-homes` the hermes-home; each is moved into the cell
  dir (`<cell>/db/`, `<cell>/hermes-home/`); re-scoring a cell later (`scorers/cli.py --cell-dir`) needs its kept
  `db/`. If a move fails, nothing is deleted and the error names both dirs. A caller that sandboxes host writes to one
  dir passes a `--scratch-root` inside it.
- `--lcm-env` overrides LCM_* on every cell and is recorded in run.json and MATRIX.md.
- Output: `results.jsonl`, `MATRIX.md`, `ISSUE-MAP.md`. Re-render: `python report.py <out>`.
- Standalone scoring: `python -m bench.instruments.reliability.scorers.cli --db <lcm.db> --gauntlet-run <dir>`
  (copies the DB into a private temp dir first; the source file is never opened).

## How a cell runs
`probe.py` runs one phase: `<host python> probe.py --cell <cell.json> --phase A --start-turn N --cell-dir <dir>`
with cwd = host src, `HERMES_HOME=<scratch>/hermes-home`, `HOME=<scratch>/home`. It refuses (exit 3) a
HERMES_HOME/HOME at or under the real home's `.hermes`, and a cell dir under /tmp. Sockets are blocked;
the provider is a MagicMock scripted per turn (unique or repeated replies, tool plans, usage that is
estimated, provider-real or scaled); the host aux LLM and the LCM summariser (tag-preserving) are stubbed.
Tools execute for real (`lcm_*`, `todo`, `read_file` on files inside the cell dir).
Faults (`os._exit` crash after a compaction commit, after a rotation, between on_session_end and
on_session_start, mid tool call; a clean exit; an ACP cancel + re-send; an injected publication failure)
end a phase; the runner starts the next phase (a fresh host process on the same HERMES_HOME).
A gateway cell reloads the state.db transcript every turn and restarts at the tip after a rotation, and builds
its agent with a stable `gateway_session_key` (`rel:<cell>:chat1`), which the host forwards to LCM as
`conversation_id` (cited per host; recorded in phase-*.json).
Every emulated host shape is located at run time in the host tree and recorded as `file:line`
(`citations` in phase-A.json); a shape that cannot be cited makes the cell UNSUPPORTED, as does a fault
whose trigger never fires at that host sha.

## Bars (all applicable bars must pass)
B1 and B2 are scored per session lineage: the chat lineage is S0 and its compression children (state.db
`parent_session_id`), each cron fire its own lineage; a row stored in the wrong lineage fails both.
The expected assistant rows come from the provider log, not the host result: exactly one scripted reply per
completed non-cancel attempt (a reply the host dropped is a deficit), plus the host's own cited failed-turn copy
(`agent/turn_failure_copy.py` `FAILED_TURN_NOTICE`/`PARTIAL_FAILED_TURN_NOTICE`, exact text) only on an attempt
the host reported `interrupted`. Any other assistant row is surplus.
- **B1** every `[Tnn]` user tag (`[A-Z]\d{2,3}`: T100-T103 on 100-turn cells) and `reply to Tnn` sits in
  exactly as many stored rows as expected,
  and every stored `continue` row is followed by the reply of its own turn (position-bound).
- **B2** multiset-v1 (port of the gauntlet's `lossless_bar_multiset.py`), per session lineage: per (role,
  sha256(content with only leading/trailing whitespace stripped, the host's ACP prompt strip
  `acp_adapter/server.py` `_extract_text(prompt).strip()`)) stored count == expected count; internal whitespace
  is exact. Tool call/result rows are keyed per lineage too. Surplus and deficit reported apart. Every
  stored-only key and every split reply is surplus and fails. Expected =
  what the host held per attempt after its ACP strip and consecutive-user merge (a crashed prompt folded
  into the next composite counts once).
  B2 strips assistant edges too; Phase C alone uses exact assistant bytes (see `RELEASE-READINESS-V1`).
- **Host-parity licence (D-A, DESIGN-436 REVISION 2 P-HOST; `scorers/host_parity.py`).** A stored USER-row surplus
  of a B2 key (or a B1 user tag) in lineage L is licensed only up to what the cell's own host state.db holds in L:
  `min(surplus, host_count - expected)`, floored at 0. host_count is what one host view holds: the most ACTIVE rows
  one session of L holds at once, else 1 if L holds it only on inactive rows; rotation copies, in-place generation
  copies and H1's fresh-stamp re-issues never add up. Deficits, assistant and tool rows are never licensed. No
  readable state.db, no licence (`unavailable` records why). Every licence is reported: `host_parity_licensed`
  under `numbers.B1`/`numbers.B2` (count + up to 10 records: LCM store ids, host row ids, content sha256, tags), the
  MATRIX `host-dup` column and "PASS with host-parity licences" section, and the ISSUE-MAP licence list.
- **B3** zero `publication_invariant_conflict` log lines across phases.
- **B8** zero `LCM survival fit applied` log lines across phases (#582): a survival fit keeps an over-window
  session alive, so on a normal cell it marks a compaction that did not bring the list under the window.
  Like B3, it does not apply to the injected publication-failure cells.

`native-on-off/<ref>/<mode>` (#581/#582) runs turns 1-30 on lcm-x `<ref>` (v0.23.3, v0.24.3) with native recovery ON;
the `plugin_switch` fault then exits between turns and the candidate takes over the same HERMES_HOME and store with
native OFF. B3 and B8 count every event of the candidate's phases after the plugin switch, before its first
publication too; `numbers.pre_publication_counts` reports the pre-publication share as a diagnostic, not a bar. B4
asks it to publish. R1 only.
- **B4** no failed turn, and the final forced compaction through the host's ACP `/compress` entry point
  (`compress_now`, or `_compress_context(force=True)` on hosts without it, selected once before invoking;
  re-invoked once after a cleanup-only `sanitized`) does not end in error, conflict or exception; an
  exception from the selected entry point fails with no fallback. It publishes only on a host result
  `compressed` from this invocation AND an engine compress() pass inside it that committed; `sanitized`,
  `skipped`, `lock_skipped` or no fresh engine pass is INCONCLUSIVE.
- **B5** depth-0 message-sourced summary nodes grow after every LCM pass; every host-native pass has its own
  host `commit_status: committed` telemetry line in the same turn; published passes (LCM or host-native,
  the final forced compaction excluded) >= `min_compactions`; `LCM compaction #` log lines == LCM passes;
  every message-sourced node covers >= 1 stored message (`empty_source_nodes`), only of its own lineage.
- **B6** (tool cells) no message-sourced summary covers part of a tool group (#559 invariant); zero host
  orphan-tool-result drops; every planned call (name, id, args) ran for real at the cited host dispatch hook
  (`model_tools.handle_function_call` with its call id, or the context engine's `handle_tool_call` via
  `agent/tool_executor.py`, bound by name + args) with a successful, complete result (`read_file`: the whole
  file; `expect.min_chars` where a cell asks for a large result), and no unplanned dispatch. Tool rows are in B2:
  `(call id, name, args)` per tool-call entry and `(call id, sha256 of the result the next provider request
  carried)` per result row, stored vs expected as a multiset; a `tool_calls` value that is not a JSON list of objects
  is a `malformed_tool_calls` surplus key. A planned call the host never dispatched is
  UNSUPPORTED.
- **B7** (native cells) no `native recovery did not produce a usable summary`, no host
  `summary_generation_aborted`, at most one native attempt per turn.

`publication-failure/rotation-child` (G-REL-1, #541's bar) injects ONE publication failure, on the first rotation-child
publication; the next child compaction must commit. `publication-failure/rotation-child-persistent` fails every
rotation-child publication: a data cell (no target, `ci.NON_GATE`) for the degraded mode where no child compaction can
ever publish, a product question outside stabilization.

Native cells: native recovery was removed (#777), so the native-only cells (`native-short-prefix/*`,
`native-long-prefix/*`) were retired with it. `native-on-off/*` remain: an older lcm-x runs with native recovery ON,
then the candidate (which ignores the key) takes over the same store.

Verdicts: PASS, FAIL (failed bars with numbers), INCONCLUSIVE (no bar fails, one could not decide), ERROR (harness or host failure; never a PASS),
UNSUPPORTED (with the reason). A cell must prove its scenario ran or it is UNSUPPORTED, never PASS: every
planned tool call dispatched and its result seen by the next provider call, at least one stored tool group
for B6, at least one native attempt in a native cell, a cancel that reported `interrupted` before the
retry, and every host citation its faults and transport need. A runner job that raises is ERROR, and so is a
phase whose imported host modules (every loaded `run_agent`, `hermes_*`, `agent.*`, `model_tools`, `tools.*`,
`acp_adapter.*`, `gateway.*`, `cron.*`, plus each cited module) do not resolve under the verified host `src`,
whose plugin module is not the exported tree, or whose interpreter is not the host's (receipt:
`provenance` in phase-*.json).
`sql_dup_counter.py`, `summary_nodes_report.py` and `compaction_ledger.py`
are ported as `scorers/dupes.py` and `scorers/summary.py` (diagnostics and the B5 ledger).

## First-request continuity (diagnostic, non-gating)

R2 records the first `main` provider request after each committed compaction:
F1 is the system LCM note, F2 the first user's tag in sole-user cells, F3 the
last completed assistant reply's tag, and F4 the current user's tag. The new
`continuity/sole-user-tool-loop/{in-place,rotation}` cells send one prompt and
12 tool rounds reading distinct files because the host deduplicates identical reads,
with a 15,360-token trigger; `continuity/survival-fit/{in-place,rotation}` lowers the survival
ceiling and follows large turns with ordinary turns.
`continuity/survival-fit-older-turns/{in-place,rotation}` uses a moderate reserve
and short later turns to measure F3 while the newest turn fits. The optional
`continuity/forced-compaction-then-turn/{in-place,rotation}` reuses `/compress`
after turn 19 and sends turn 20. Existing `baseline/{mode}/acp` cells supply
the F3 negative control. These checks are outside `BARS`; the new cells are in
`ci.NON_GATE`. The two survival-fit shapes keep B1–B7 and B9, but omit B8
because survival fitting is their required scenario. Other cell definitions and
the bar implementations are unchanged.

In continuity cells only, a missing current tag in the last user message is
a diagnostic `FAIL` on F4, targeting #916 in `survival-fit`; a last user message
starting with `[LCM survival fit:` records `current_user_projected=true`.
Other cells still report this condition as `ERROR`. Infrastructure errors
still gate; continuity findings do not.

`MATRIX.md` reports per-host/cell counts and each compaction's kind, turn,
previous-reply tag and whether its input row was replaced. Unknown data and
compactions with no following request are explicit; a missing scenario is
reported, never inferred from configuration. New provider metadata contains
only booleans, synthetic tags and message role/counts, with no request text.
This answers #900/#798 on pinned CI hosts and seeds #659; literal presence
does not prove an instruction remains operative. No real model is called.

### #659 continuity probes (recorded, non-gating, opt-in)
`continuity/probe-{host-instruction,todo,constraint-head,constraint-mid,constraint-shape}/{in-place,rotation}`
(`cells.probe_cells`, R2 only) are selected by an explicit pattern and never by `--cells all`, so the nightly matrix,
`ci.expected_cells` and the gate are unchanged. Run them with
`run_matrix.py --transport acp-process --hosts <host> --plugin-ref <ref> --cells 'continuity/probe-*' --out <dir>`.
Each cell plants nonces (letters G-Z only, so no hex-shaped token) and the fake provider checks them at request time
(`provider-requests.jsonl` keeps no text): presence, count and carrier class (system, user_raw, summary_or_head,
todo_fold, tool_result, tool_args, assistant) in the first main request after each committed compaction, beside F1-F4.
`apply_probes` records a per-probe rate in `probes_recorded`; it never sets a bar, the verdict or the gate.
- **P1 host instruction** (`host-instruction`, bar 100%): a SOUL.md nonce line, exactly once in a system message.
  SOUL.md is rewritten before turn 16 with no restart: the new line must appear only after the next commit, and every
  main request between the rewrite and that commit must still carry the old one (`host-instruction-rebuild`).
- **P2 open todos** (`todo-open`): the scripted model writes todos through the default `tool_call` bridge (the defer
  list is not changed); the observer wraps the inline `todo_list` executor and B6 binds the bridge call to the unwrapped
  dispatch. Open items must sit in the host's fold row; the completed item never may (`todo-done-control`). Its own
  cell: the fold merge shape is #538.
- **P5 constraint before the fresh tail** (`constraint`): a turn-3 constraint at the head, mid-text, or in another
  nonce shape (the three "seeds"), then three large reads in turn 6; a turn-27 control inside the tail.

Not measured: the summariser is a scripted fake that keeps tags only, so P5 on LCM-X measures structural retention
(fresh tail and head anchor), not whether a real summary carries the constraint; literal presence does not prove an
instruction is obeyed. P3/P4 (plan, goal) and the plain-Hermes arm are #659 PR B/C.

## Positive controls
`controls.py` holds each control's refs, hosts, cells and expected red/green pattern; `run_matrix.py --control
PC-1 --out <dir>` runs it and writes CONTROL.json (HOLDS or the mismatches). PC-1 is a differential: lcm-x `47bd28e7` (before #498, the #494 fix) vs `ae1fb16d` on ref-0.21.5, rs34-0.21.5
and upstream-main: `baseline/in-place/acp` PASSes at both, `acp-trailing/in-place` FAILs only at
`47bd28e7` (the host's post-commit-proof persist strip). customer-0.21.2 passes both refs (no such strip).

## Limits
In-process only: no real `hermes acp`/gateway process, transport, model or timing. Gateway timestamp
rendering stays at its default (off). Upgrades from pre-fix DBs (#485/#542) and the Desktop/tui transport
(#463) are not covered; see ISSUE-MAP.md for every uncovered issue and the capability it needs.

## R2 transport cells (`--transport acp-process`)
`run_matrix.py ... --transport acp-process` runs the same cell ids and scorers through `<host venv>/bin/hermes acp`
over stdio (`acp_driver.py`, ported from the WS3 gauntlet driver), one process per phase, HERMES_HOME/HOME under the
cell dir. Every model route (main, `LCM_SUMMARY_MODEL`, host aux) is the localhost `fake_provider.py` (OpenAI JSON/SSE,
Anthropic messages, `/v1/models`); its request log is reconciled with the transcript (`accounting` in verdict.json).
`observer/` is put on the host's PYTHONPATH and records, without changing arguments or results, the host seams R1's
probe traces (R1's transcript schema). Faults are real: SIGKILL of the host process group while the provider holds the
turn's request; ACP `session/cancel` while it is in flight; the final check is the ACP `/compress` command.
Containment: localhost base URLs, no API-key env, every proxy variable at a recording sink that refuses
(`proxy-attempts.jsonl`), `model_catalog: {enabled: false, excluded_providers: [opencode-free]}`, a seeded models.dev
cache, macOS `sandbox-exec` (localhost-only network), and the socket guard; any attempt makes the cell ERROR (STOP).
`excluded_providers` is read at `hermes_cli/inventory.py:54`; without it customer-0.21.2 fetches the keyless
opencode-free catalog on ACP `session/new` (`hermes_cli/models.py:2105 _fetch_opencode_free_models`). The R2
summariser runs with `LCM_SUMMARY_SPEND_MAX_CALLS=100000` (60 turns in ~20 s would trip the per-window spend guard).
Cells whose fault is injected in-process (publication failure, crash between end/start), cron cells and the R1
gateway cells are UNSUPPORTED with the reason; `--transport gateway-process` is UNSUPPORTED with per-host citations
(the webhook platform is one-shot per delivery; api_server bypasses TurnRunner). `scorers/chronology.py` reports
(never gates) user-row tag order per lineage.

- **Backlog guarantee (R1 and R2).** Before the final check, if fewer than 3 turns have passed since the last
  committed non-final compaction, up to 3 extra scripted turns run (tagged and scored like any turn); the checks
  are in `final_check.backlog_checks`. B4 still `sanitized` after that is INCONCLUSIVE with the checks in the reason.
- **Private byte-code.** Every host invocation (R1 probe, R2 host process) sets `PYTHONPYCACHEPREFIX=<cell>/pycache`;
  `hosts.verify` refuses a sourceless `.pyc` (outside `__pycache__`, or with no matching `.py`).
- **Legacy `/compress` fallback.** `final_check` falls back to `_compress_context(force=True)` only on
  `ModuleNotFoundError` for exactly `agent.conversation_compression_manual`; any other ImportError is recorded and B4 fails.
- **crash-after-rotation over acp-process.** `RotationKiller` polls `lcm_lifecycle_state` every 20 ms and SIGKILLs the
  host group while the rotated child session has 0 lcm rows; a rotation seen after the child has rows is skipped
  (`missed_rotations`) and the next one is used. No provider request is held, so accounting does not count it as one.
- **anthropic-route/acp-process** routes main to the fake provider's `/anthropic` endpoint and needs the host's
  `anthropic` extra (`hermes-agent[anthropic]`); without the SDK the cell is UNSUPPORTED. The local hosts lack it.
- **customer-0.21.2 lcm-tool-mid-turn is UNSUPPORTED over ACP.** That host defers plugin tools behind the
  `tool_search`/`tool_call` bridge by default (`tools.tool_search.enabled: "auto"`, `hermes_cli/config_defaults.py:1793`),
  so the scripted direct `lcm_grep` call is never dispatched ("scenario not proven"). The default is not changed.

### Nightly CI (`.github/workflows/reliability-nightly.yml`)
Triggers: daily schedule and `workflow_dispatch` (effective once on main), and `pull_request` path-filtered to
`bench/instruments/reliability/**` and the workflow file. Not a required check; default token only. Matrix over
`hosts.ci.json` (pinned shas): ref-0.21.5, customer-0.21.2 and r34.4-0.21.5 on Python 3.11, upstream-main and upstream-uid
on 3.14. upstream-uid (2667c960) is upstream after its message-uid change: the one CI host with message uids and the P8
flush seams. upstream-main (6f7a7991) stays before it, where the host archive copies uncovered rows behind the running
turn (the shape that exposed #845). `ci.py prep`
fetches the sha and installs it editable with `[acp,edge-tts,bedrock,vertex,anthropic]` (the harness verifies git HEAD
and cites source); R1 all cells and R2 acp-process all cells run with `--plugin-ref HEAD`; MATRIX.md is the job
summary and results are uploaded. `ci.py gate` fails on any ERROR, on a FAIL in the G-REL-1 cell set unless every
failed bar is declared by at least one open target in `cells.ISSUES`, and on an empty set or any missing, duplicate or unexpected row per (host, transport,
plugin sha) against that transport's `--cells all` list. Linux has no `sandbox-exec`: there containment is the proxy sink plus the socket guard.
A FAIL with empty or missing `failed_bars` always gates; an open target absent from `ISSUES` declares no bars, and a
target listed in `cells.ISSUE_HOSTS` declares its bars only on hosts whose name starts with one of its prefixes.
G-REL-1 diagnostics list uncovered bars, targets and open targets.

### Claim boundary
R2 proves the plugin's behaviour through a real `hermes acp` process on the pinned host shas, with every model route
at a deterministic localhost fake. It does not prove live-model behaviour, the gateway process, several sessions in
one process, or the process-side publication-failure hook (R2b, #569), and it says nothing about customer boxes.

P8 / B9 audits R1 in-process and R2 `acp-process` host flushes using the host's own resolvers and digests:
I0 pins committed live addresses; I1 forbids archived writes/adopts; I2 pins session/role/uid
and unique uid-snapshot resolution; I3 forbids adopts; I5 rejects active uid twins
involving LCM output (host-only twins are reported). Events contain no payload.
Missing host seams, audit errors, no observed commit or no observed host flush give B9 UNSUPPORTED. At least one
flush must follow a commit of the same session in log order, across phases; otherwise B9 is UNSUPPORTED too.
The audit records the resolved target's session and fails I2 on a mismatch; older events without that field
keep their previous scoring. A flushed dict whose address resolves to no row is counted as `UNRESOLVED`
(report-only: after a rotation a parent-session `_row_id` legitimately resolves to nothing), not as `INSERT`.
`p8-control/{archived,other-active,random-snapshot}` fail I1/I2/I3; `none` passes.
On R2 the same controls inject after the second commit inside the ACP subprocess.
The nightly gate checks this pattern per host and transport, with B9 required in each FAIL's failed bars.
`controls.py` records the measured must-support pairs, where missing or UNSUPPORTED controls gate; other pairs
may be UNSUPPORTED, but any PASS/FAIL must match.
Disable wraps with `LCM_RELIABILITY_P8=off` or `--lcm-env LCM_RELIABILITY_P8=off`; faults stay.
