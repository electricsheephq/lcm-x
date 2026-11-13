# P1 busy-bar instrument

Claim class: **instrument**. This counts busy operations on public-engine versions
of four gate-cell families. It does not prove that P1 passes the busy bar, the crash
gate, losslessness, the lock-hold bar, D5/D7, or host/ACP compatibility. The actual
gate runs on a bench host; local smoke numbers are instrument checks only. These
cells do not inject crashes, savepoint failures, or SQLite locks.

Use the same Python and host `ContextEngine` import path for both trees. Supply
that host path through `PYTHONPATH`; no host shim is bundled here. Each process
registers exactly one checkout as `hermes_lcm`, using the occurrence-fuzz
load-by-path pattern without executing the plugin registration entry point.

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$HOST_IMPORT_PATH:$DEPENDENCIES"
python -B bench/instruments/busy_bar/orchestrate.py \
  "$BASELINE_TREE" "$CANDIDATE_TREE" "$OUTDIR" --runs 20 --work-root "$SCRATCH/busy"
python -B bench/instruments/busy_bar/gate.py \
  "$OUTDIR/baseline.jsonl" "$OUTDIR/candidate.jsonl" \
  --control-baseline "$OUTDIR-control/baseline.jsonl" \
  --control-candidate "$OUTDIR-control/candidate.jsonl"
# One tree/cell (new output file and fresh work directories):
python -B bench/instruments/busy_bar/driver.py \
  "$CANDIDATE_TREE" compress "$OUTPUT" "$WORK" --runs 1 --k 2 --b-view-rows 2000 --busy-timeout-ms 100
```

`orchestrate.py` defaults to 20 runs per cell per tree, with one child process
at a time. It alternates baseline/candidate for each cell of each run, reversing
the pair order on odd runs. Baseline is v0.29.2's rc1 tree, `7594beb7`; the
candidate source references below are from `4bf0a9d6`. Keep the baseline read-only
(including disabling bytecode writes). Outputs include the two combined JSONL
ledgers, individual run ledgers/logs, fresh stores, and `verdict.txt`/`verdict.json`.
The same comparison then runs with a 100 ms connection override in the sibling
`OUTDIR-control` (or `--control-outdir`). Both use the same seeds. Stores go under
`--work-root` (a fresh temporary directory if omitted), separately from evidence.
Run without `nice` where sandbox policy denies it. Driver runs default to one; each uses a new `WORK/run-N` and refuses an existing
directory. Orchestrator refuses an existing output directory. Exit status is 0
for PASS, 1 for FAIL, and 2 for INCONCLUSIVE (or a failed driver operation). A gate verdict needs
the full session (runs 0-19 of every cell in both trees, 0-79 for an extended cell; ten runs per tree for
holds) and the workload below unchanged (K=2, 2,000-row B views, W=400 ms, seed 2055+i; holds: 2,000 rows);
anything shorter or different prints SMOKE PASS/FAIL (holds: LOCK-SMOKE PASS/FAIL) and exits 3. Holds are
INCONCLUSIVE when any 2,000-new-row call shows no observed hold, a hold or statistic is not finite and
positive (NaN and inf never reach the tripwire), or a run lacks its proof: a distinct `run_id`, shape-n calls
whose `view_rows` and `rows_stored` equal the run's rows, `ingest_failures` 0 on every call, and a monitor whose
samples cover every call.

Tree identity: every busy ledger row and holds run records `tree_identity`, the tree's `plugin.yaml` version plus a
sha256 over its product `.py` files (relative path and content hash, sorted; top-level `tests`, `bench`,
`benchmarks`, `benchmarking`, `scripts`, `docs`, `release`, `skills`, `companions`, hidden and `__pycache__`
directories excluded), computed by the driver or probe from the tree path it was given. The baseline must be the
v0.29.2 GA tree (`driver.BASELINE_IDENTITY`, computed from a `git archive` extract of `7594beb7`); production,
control and extension ledgers of one role must share one identity; the candidate must differ from the baseline.
Anything else is INCONCLUSIVE. `--baseline-identity-override` (orchestrator, gate, holds) skips only the v0.29.2
check and makes the verdict a smoke. Each busy run must also contain exactly the operations the driver prescribes
(`driver.prescribed_operations`, a dry run of `cells.sequence` with K B ingests per A operation); equally
incomplete ledgers are INCONCLUSIVE, not a pass.

## Opt-in storage arms

Both driver and orchestrator accept these diagnostic flags. With none set,
commands, inputs, scheduling, ledger fields and summaries retain their defaults.

| Flag | Default | Effect |
| --- | --- | --- |
| `--wal-autocheckpoint N` | unset (product: 500) | Call the original connection configurator, then apply N to every configured connection, including imported aliases. Private unconfigured connections retain their SQLite defaults. |
| `--checkpointer off\|passive:<ms>` | `off` | A daemon with its own configured connection runs PASSIVE checkpoints at the positive integer interval. Use with `--wal-autocheckpoint 0`. |
| `--background-writers none\|stem\|integrity\|embedding\|all` | `none` | Seed due work, retain the selected real workers, and release them when the first scheduled operation begins. Nonselected background FTS dispatches are suppressed only in a writer arm. |
| `--stem-backlog N` | 20000 | Positive number of synthetic messages left unindexed for the stem worker. |
| `--record-holds` | off | Wrap SQLite transaction paths in this process and record foreground and background writes. |

An integrity arm seeds 2,000 messages and expires its scan stamp. An embedding
arm seeds 64 leaf summaries and registers a deterministic in-process fake
provider, retaining the real scheduler, maintenance pass and VectorStore batch
publication. It never needs a model or network. Preparation finishes before
workers are released; synthetic messages belong to a separate seed session.

`--wal-autocheckpoint` does not reach the background writers' private stem,
integrity-scan, or integrity-claim connections. They keep SQLite's 1000-page
autocheckpoint setting. An H2 run (`--wal-autocheckpoint 0`) combined with these
writers is therefore not autocheckpoint-free.

Opt-in ledgers add `storage_arms`, including configured-connection PRAGMA
readbacks and monotonic schedule boundaries, plus sibling `storage_arm_drains`
records with duration and skipped status. On `wait=False`, the engine's own
`timeout=0` retirement drops the old owner's queued, unshared passes; `finish()`
waits only for active ones, outside the timed window. The embedding arm's
background load at session end can therefore be lower than at the base, which
forced a drain of up to 35 s inside the timed window. The arm follows the product's own
path (ruling T1-88). Beside each output ledger,
`<output-stem>-run-N/` holds `holds.jsonl` and/or `checkpoints.jsonl` when requested.
Checkpoint records contain start, duration_ms, result (busy, log, checkpointed)
and error. Hold records contain writer, connection_id, database (the main file's
path as opened), start_basis (`acquired` or `call_entry`), start, end, hold_ms and
outcome (commit, rollback or busy). Only the cell's main database is recorded;
in-memory and other databases and preparation writes are excluded. Any ledger
carrying `storage_arms` is diagnostic and can only receive a SMOKE verdict (exit
3), or INCONCLUSIVE (exit 2), even with the standard workload and run count.

Host percentages are report-only; a null percentage gets a `reason`.
`host_load_peak1m` is the larger of the two endpoint readings, not a sampled peak.

Explicit IMMEDIATE transaction timing starts after lock acquisition. The end is
sampled before a separate release call (connection commit, rollback, close or
context exit; SQL COMMIT, END, ROLLBACK or RELEASE), so the recorded tenure is a
lower bound. These records use
`start_basis=acquired` and also include
`acquired_at`, the successful BEGIN's return timestamp. Implicit first writes
(including foreground stores' Python implicit BEGIN IMMEDIATE) and integrity
INSERTs use `start_basis=call_entry`: their acquisition point is not observable
through this wrapper, so they omit `acquired_at` and can include acquisition wait.
Their ends also precede a separate release call. An autocommit write that opens
and closes its own transaction instead ends after the write call: its nonempty
`call_entry` span contains the call, including any time after lock release.
Overlaps involving these spans are not evidence of simultaneous lock ownership.
Integrity spans end at their savepoint rollback. Busy records describe failed
acquisition attempts, use `call_entry`, and omit `acquired_at`. These
wrappers add overhead; they are diagnostic instrument evidence, not native lock
sampling or a release timing result. The product tree identity is unchanged.

## Workload and schedule

`success_wall_ms` reports `n`, `p50`, `p95`, and `max` (milliseconds, nearest-rank
percentiles) per operation kind and overall, per cell and per tree. It is
**reported, not gated; includes any SQLite busy wait**. Only judged calls with
no exception, busy signal, or inconclusive flag count; untimed older records are
omitted. The comparison adds candidate/baseline `p95_ratio` per kind and overall
(null if either p95 is missing or the baseline p95 is zero). Verdicts are unchanged.

Both engines use the same database and isolated home, with distinct sessions and
conversation ids A/B. B is created, bound, driven, and shut down on its own thread;
A runs on the main thread. Startup completes before contention begins. Each counted A operation releases one fixed batch of B operations. Setup, session
starts and shutdowns are outside the counted-operation denominator; their logs
remain in the raw ledger. Restart is shutdown plus a fresh engine and session
start, not a process restart. The orchestrator provides process isolation between
runs.

- Ingest: two-row prefix ingest, same-session start, three-row view, identical
  re-ingest (3 A operations).
- Compress: 12 cumulative two-row turns, one fresh user row, forced public
  `compress()`, restart, ingest the returned view (15 A operations).
- Session end: ingest, end with one further turn, next same-session start,
  ingest that view (3 A operations).
- Restart: ingest, restart, same view plus one turn (2 A operations).

Rows, timestamps, summary text, fresh tail (6 rows) and leaf chunk budget (400
tokens) are fixed in code. Each A operation and B's corresponding batch meet
at a two-party barrier. B starts immediately; A sleeps for its seeded arrival offset
before entering its operation. Offsets use `random.Random(seed).random() * W`,
uniform in [0, W), with seed 2055 and W=400 ms by default. Run i uses seed 2055+i
for both trees, both budgets and every cell. No tree-specific tuning is allowed. B performs exactly K sequential ingests (default K=2)
on its own thread; a completion barrier waits for both A and the entire B batch
before the next A operation. B does not contend with uncounted session starts or
restarts. This gives B exactly K times A's operation count regardless of tree
speed. Within a batch, OS scheduling still controls which writer obtains the lock.

Every B ingest presents a 2,000-row view of entirely new, deterministic rows with
unique content and timestamps across batches. The window stays bounded rather
than growing a cumulative transcript. Production mode sets no PRAGMA. Each engine's effective store timeout is read
back after opening, including A's restart. `--busy-timeout-ms N` on the driver
is the symmetric scaled-budget option; it overrides only that connection's PRAGMA,
leaving all product retries and temporary timeouts intact. The control uses 100 ms
and models writers that outlast the busy budget. It never determines PASS/FAIL.

The driver accepts `--k`, `--b-view-rows`, `--seed`, `--window-ms` and optional
`--busy-timeout-ms`. The orchestrator accepts the same schedule options and always
runs production plus the fixed 100 ms control. Ledgers record constants, seeds,
offsets, batch indexes, view sizes, timeout read-backs and operation intervals.
A/B operation exceptions record the type and, only for `sqlite3.Error`, the first
200 characters of its message. Exception chains retain only types, SQLite codes and
SQLite-only bounded messages, never wrapper messages. A raised error is busy only
when it or an error in either its `__cause__` or `__context__` chain is a
`sqlite3.Error` with `sqlite_errorcode & 0xff` equal to 5 (BUSY) or 6 (LOCKED).
Extended codes (including 517, BUSY_SNAPSHOT) count. Message text alone does not
classify a raise. Any other raise makes that run INCONCLUSIVE; no run is replaced.
Exceptions continue to the next operation. Failed compression
retains the original uncompressed host view. Setup, restart and coordination errors
remain driver errors. Unequal exposure remains a driver bug.

### Product-owned budgets and retries

Sources below are baseline `7594beb7` / candidate `4bf0a9d6`, respectively.
The requested `busy_timeout|SQLITE_BUSY|database is locked|_temporary_sqlite_busy_timeout`
search covered product Python files in both trees; test/bench copies are excluded.

| Policy | Baseline source | Candidate source |
| --- | --- | --- |
| Store connect 5 s, **then bootstrap overrides to 30,000 ms** | `store.py:421-423`, `db_bootstrap.py:54,144` | `store.py:514-516`, `db_bootstrap.py:55,145` |
| Session-end/finalization 50 ms; pending tails share remaining deadline | `engine.py:669,5690,5709,10838-10848` | `engine.py:276,5375,5394,11552-11565` |
| Emission write temporary 100 ms | absent | `engine.py:7155` |
| Foreground ingest retries complete owner once after busy/savepoint rollback; admission retry uses 0 ms | absent | `engine.py:6957-6958,7040-7044`, `store.py:2075` |
| Write-lock helper retries BUSY_SNAPSHOT once after rollback; ordinary BUSY gets connection budget only | `sqlite_util.py:53-84` | `sqlite_util.py:70-106` |
| Temporary timeout helper restores previous value | `sqlite_util.py:112-129` | `sqlite_util.py:154-171` |
| Single-row metadata writes: no custom budget or retry; read-to-write upgrades can fail immediately | `store.py:1701-1739` | `store.py:2011-2049` |
| Sender update temporarily 0 ms | `store.py:747-779` | `store.py:856-888` |
| Compaction telemetry reservation 100 ms | `store.py:1839` | `store.py:2197` |
| Bootstrap/WAL conversion lock retry: 30 s deadline, 5 ms backoff doubling to 250 ms | `db_bootstrap.py:54,162-193` | `db_bootstrap.py:55,163-194` |
| Integrity scan/meta connections 30 s; claim windows 50 ms, restored; maintenance lease 30 s, restored | `db_bootstrap.py:2805,2938,2949,3056-3059,3279-3290,3544-3563` | `db_bootstrap.py:2853,2986,2997,3104-3107,3327-3338,3592-3611` |
| Entity read-only 30 s (writable connect 5 s then bootstrap) | `entity_store.py:52-54` | `entity_store.py:52-54` |
| Assertion/trajectory read-only 30 s; writable connect 5 s then bootstrap | `assertion_store.py:235-243`, `trajectory_store.py:608-616` | `assertion_store.py:235-243`, `trajectory_store.py:608-616` |
| Standalone read-only reader 5 s default, explicit PRAGMA | absent | `readonly_store.py:51,118-134` |

The specs' description of a 5 s production timeout refers to the connect argument;
the actual effective value is read back rather than inferred. No product policy
is changed to match that description. A deferred read-to-write upgrade can return
BUSY immediately without invoking the busy handler; the single-row metadata
helper does not add a retry. The snapshot-retry helper above explains the ingest
path separately. Sender updates explicitly use a zero budget in both trees. Read-only, integrity and disabled auxiliary
paths above are inventory, not claims that each runs in these cells.

Both trees patch `_schedule_rollup_maintenance` and
`_schedule_embedding_maintenance` to no-ops when present. Embeddings, extraction,
assertion extraction and background leaf preparation are disabled. The model
adapter `engine.summarize_with_escalation` returns a constant local summary while
the public compression pipeline remains real. Driver home lookups are isolated
before tree import, inherited `LCM_*`/`HERMES_*` settings are cleared, and socket
network connections are denied. The three identity/ledger settings are not pinned:
each tree uses its shipped defaults, read through its loaded config functions and
recorded as `effective_defaults` (`absent` when that tree lacks the setting).
No credentials or model calls are needed.

## Pinned counting signals

The root logging handler records full messages, emitting thread/process ids,
monotonic times, levels and bounded exception-chain metadata. LCM loggers are
captured at DEBUG in both trees. These exact substrings are shared by both pinned trees:

| Signal | Match | Baseline source | Candidate source |
| --- | --- | --- | --- |
| write-lock-recovery | `SQLite write lock recovery exhausted` | `sqlite_util.py:77` | `sqlite_util.py:99` |
| ingest-busy | `LCM ingest failed` AND case-insensitive `busy` or `locked` | `engine.py:2931` | `engine.py:2554` |
| skipped-lock | `skipped due to SQLite lock` | `engine.py:5743`, `:5770`, `:5798` | `engine.py:5197`, `:5429`, `:5456`, `:5484` |

Candidate counters: `ledger.emission_busy` (`engine.py:7224`) during compression,
and `ledger.tail.busy_at_end` (`engine.py:5195`, `:5427`) during session end.
They are read from the engine's in-memory `_ingest_ledger_counters` before and
after the operation; absent counters read as zero. Baseline lacks `_ledger_count`
and never enables counter attribution. These observational reads change no
product behavior. `compress()` exists in both trees at `compaction.py:732`;
`ingest`, `on_session_start`, `on_session_end` and `shutdown` are public engine
methods and checked at driver startup.

An operation is busy once if any pinned log signal occurs on its own driver
thread/process during its public API invocation, its applicable candidate counter
rises, a raised exception is busy by the code rule above, or an LCM record's
attached `exc_info` is busy by that same rule. Caught DEBUG failures count too:
`LCM successful compaction telemetry update failed` is emitted by
`engine.py:2349` in the baseline / `engine.py:1971` in the candidate for the telemetry write's 100 ms budget.
Repeated lines, multiple signals, and a counter plus a log still count once.
A signal belongs to its emitting engine and call, never to another engine whose
call overlaps it. Internal writes belong to their containing public call's class.
Outside-call records and other-thread/process records do not count. A caught
non-busy error neither counts nor makes a run inconclusive. Output reports A, B
and total counts.

## Gate interpretation

Only production counts gate: a cell fails when candidate busy operations exceed
`ceil(1.25 * baseline) + 1`. Thus baseline 0 allows 1; baseline 4 allows 6.
The default gate session runs **20** production runs per tree/cell. If baseline
has fewer than 8 busy operations in a cell and candidate has more, both trees run
**60 additional** production runs of that cell, numbered 20..79, with seeds
2075..2134 continuing the sequence. The rule then judges the 80-run totals.
Both initial 20-run and final counts are reported. Missing extension runs make
the verdict INCONCLUSIVE. `--runs 2` is the local smoke; it does not extend or
establish the gate's prescribed exposure.

The control always covers the initial runs only (20 for a gate session), with
the same seeds and offsets. It must show at least one busy failure in **each
operation class** (`ingest`, `compress`, `session-end`) in **each tree**, including
signals from internal writes attributed to that class. A missing class/tree is
named and makes the verdict INCONCLUSIVE. Unequal operation counts/run coverage/
schedules, any driver error, or any inconclusive run also makes it INCONCLUSIVE.
Missing control ledgers cannot pass. Production/control exposure must match,
ignoring their timeout difference and production extension runs. Control counts
are printed separately and never compared for a gating regression. A smoke with
zero busy operations in either control tree stops without tuning.

The gate constants stay unchanged: K=2, 2,000 new B rows per operation,
W=400 ms, `random.Random(2055 + run).random()*W`, 20 initial runs, 60 extension
runs when required, and 20 control runs. The control overrides `PRAGMA
busy_timeout=100` on each A/B **main store connection**, immediately after opening
and after A's restart; the ledger records each read-back. It does not lower
auxiliary/read-only connections or replace temporary product budgets. Maintenance,
embeddings and extraction are disabled, so their auxiliary connections do not
participate. Production applies no override. The error-code and log/counter
classification rules above apply identically across budgets.

## Relative lock-bar probe

```sh
python -B bench/instruments/busy_bar/holds.py \
  "$BASELINE_TREE" "$CANDIDATE_TREE" "$HOLDS_OUT" --runs 10 --rows 2000 \
  --work-root "$SCRATCH/holds"
```

Each run/tree is a fresh process and fresh WAL store, interleaved with reversed
pair order on odd runs, serially except for its monitor child. It ingests six
2,000-new-row views into the same session (shape n), then the last stored
2,000-row view plus a two-row suffix (shape t). Journal mode is checked and
recorded; any non-WAL store fails the run. Shipped defaults and offline switches
match the busy driver.

The monitor opens only the database's `-shm` file **read-only** and polls
`fcntl(F_GETLK)` for a write lock at byte 120 (`UNIX_SHM_BASE=120` plus
`WAL_WRITE_LOCK=0`). It never acquires a lock or opens a SQLite connection. A
separate process is essential because POSIX locks owned by the querying process
are invisible to F_GETLK. Darwin's 64-bit struct flock is start/length/pid/type/
whence (24 bytes); Linux's is type/whence/start/length/pid (32 bytes, including
padding). Unknown platforms/ABIs fail loudly. Native-layout tests cover both;
a live known-holder PID check confirms the running kernel's response. This
Mac's layout was also confirmed by compiling against its SDK's `sys/fcntl.h`.
Linux kernel execution requires a Linux host; packing tests alone do not prove it.

**Poll constant: 150,000 ns (0.15 ms)** of sleep per loop, plus syscall/loop
work. Achieved intervals (median/p99/max) and all time/PID samples are recorded
per run. A local Darwin/Python 3.12 known-hold control measured median 0.194 ms,
p99 0.201 ms and maximum 0.231 ms; these are observations, not guaranteed bounds.
Both processes use system-wide `time.monotonic_ns()`, never wall time.
The engine process stamps only call start/end; it does no per-statement work.
A held period starts at the first held sample and ends at the first unlocked
sample. Periods separated by less than the fixed poll interval merge, then are
clipped to call boundaries; the longest is that call's hold. Sampling may miss
short holds (reported as zero), and scheduling stalls enlarge edge uncertainty;
use achieved intervals, not the requested sleep, to assess this uncertainty.

Per tree/shape the summary reports median/p90 longest hold, median first observed
start and median call duration. Shape n reports its ratio to v0.29.2 at **all eight**
statistics, and fails only when one exceeds the 2.25 regression tripwire (design r3.7; r3.6's
1.25 bar no longer applies; the busy bar above is the P1 lock gate): the all-calls median (60 calls in a default session); six per-position
medians over runs; and the median of each run's longest hold. FAIL names every
exceeded statistic. Shape t has no verdict: its absolute bar comes from the
harness's per-ingest counters. `--trace` optionally retains keyword-only statement
traces/windows for diagnostics; it produces `DIAGNOSTIC ONLY`, never a gate verdict.
The default never installs a trace callback.

The gate value comes from one ten-run interleaved session on a declared quiet
host. Local R=3 smoke on a shared host validates the instrument only. It does not
establish either release bar. A monitor that misses the known 200 ms control hold,
or an unconfirmed native layout, is a stop condition, not permission to substitute
a statement-trace measurement.
