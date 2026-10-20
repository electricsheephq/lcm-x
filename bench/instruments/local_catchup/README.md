# LOCAL catch-up instrument (I1)

Binding authority: [#1586, Gate declaration (release manager), v1](https://github.com/electricsheephq/lcm-x/issues/1586#issuecomment-6283060211).
Read the declaration before running; the instrument does not change its bars.
Use Python 3.11+ with pytest, psutil (mandatory), FastEmbed and ruff 0.15.13.
The Hermes source supplies the host ContextEngine contract; its CLI is never run.
Set shell variables to absolute paths: `PRODUCT` (any product checkout), `HOST`
(Hermes source), `SCRATCH` (new regenerable bulk), `RUN` (new evidence directory),
`CACHE` (prewarmed model cache with **all** write bits removed), `PY` (Python).
No live profile is used. Never pass a live DB or home.

```sh
python -m pytest bench/instruments/local_catchup/tests -q
ruff check --no-cache bench/instruments/local_catchup
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.corpus --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --out "$SCRATCH/corpus" --rows 250000 --shape short --baseline empty --pilot 1000 --peak-disk-gb 8
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.corpus --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --out "$SCRATCH/corpus-mixed" --rows 250000 --shape mixed --baseline empty --pilot 1000 --peak-disk-gb 12
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.catchup --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --db "$SCRATCH/corpus/store.db" --manifest "$SCRATCH/corpus/manifest.json" --home "$SCRATCH/homes" --model-cache "$CACHE" --scratch "$SCRATCH" --out "$RUN/catchup" --max-wall 24h
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.probes drain96 --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --db "$SCRATCH/probe/store.db" --manifest "$RUN/probe-manifest.json" --home "$SCRATCH/probe/homes" --model-cache "$CACHE" --scratch "$SCRATCH" --out "$RUN/probe" --max-wall 20m --restart-after 30s
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.probes cold --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --db "$SCRATCH/corpus/pilot.db" --manifest "$SCRATCH/corpus/pilot.json" --home "$SCRATCH/cold/home" --model-cache "$CACHE" --scratch "$SCRATCH" --out "$RUN/cold"
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.probes screen16 --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --scratch "$SCRATCH/screen" --model-cache "$CACHE" --out "$RUN/screen" --rows 250000 --peak-disk-gb 96
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.rig --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --scratch "$SCRATCH/rig" --model-cache "$CACHE" --out "$RUN/rig" --processes 14 --rows 250000 --peak-disk-gb 80
```

Use the matching corpus directory for catchup; repeat with `--baseline upgrade` in new directories.
`--pilot 1000 --pilot-only` builds only a pilot and records the full-size disk estimate.
The estimate includes float32 vectors, SQLite overhead, WAL/copy margin; excessive
estimates refuse allocation. The rig estimates the entire N-store allocation first.
Run large cells only on the authorized host/window with its declared disk budget.

The corpus strips system rows and uses at most 250 pairs/session. Every message,
canary and session has a shard identity. Times, IDs, logical digests and token
histograms are deterministic; SQLite files need not be byte-identical. Leaves
are deterministic SummaryDAG fixtures with real source IDs, not a compaction test.
Upgrade keeps short-turn gaps and one >32-chunk message with its first vector only.
Its pre-existing vectors are fixture vectors, never counted as measured throughput.

Catchup binds once per interpreter lifetime, then only observes. Restart defaults
to 40% coverage, with a declared 25%-of-max-wall fallback if that fraction is unreachable;
the small control uses 30 seconds. Shutdown has a 10-second bound, then kill.
Restart and interpreter startup count toward elapsed time. Every 30 seconds a
read-only connection counts expected IDs and digests newly seen vectors only;
previously seen vector replacements are detected at the next audit. Full raw-row/vector integrity audits
run at baseline, restart and end. Zero progress includes
persisted scan progress, and STALLED requires a successfully constructed provider.
CPU samples are every second, CPU-time deltas in cores, weighted means and full
60-second rolling means with nearest-rank p95. Missing 60-second windows are
insufficient evidence. Default bars: .6/.75 per PID, 7/8.75 aggregate; abort at
>12 cores for 10 seconds or increased host swap. Existing swap is recorded.
`--cpu-cap`, `--cpu-sustained`, `--per-mean`, `--per-p95`, `--aggregate-mean`,
`--aggregate-p95` expose the declared supervisor thresholds. psutil is mandatory. A dedicated sampler stays independent of audits; any sample
gap over 2 s is insufficient. CPU bars are N/A for STALLED/ABORTED cells.
Statistics clip to launch/first-complete; aggregate bars apply only to N=14.
Host one-minute load above the core count flags a non-quiet window.

Raw CPU/progress/events, worker status, manifest and summary are receipts.
Final DB copies stay in --scratch. Rescore with `python -m bench.instruments.local_catchup.rescore <cell>`.
Screen statistics start after the last bind; any completed backlog makes it INVALID.
The 16-process 60-second screen is reported and never gates the default flip.
The cold probe pairs OFF/ON binds and requires FTS canary hits with honest semantic
degradation. A control stall is an expected negative, not a product acceptance pass.
Small controls prove instrument sensitivity only: no 250k, floor-host, turn,
compaction/recall, release, customer, or default-flip claim follows.
At closeout initialize/add/headline/finalize/check a result card with `bench-card.py`
for the positive-control evidence directory. The release manager owns publication.

## I2: turns, compaction recall, floor surrogate

These extend I1 without changing its behavior. Use the same variables above;
`SCRATCH` and `RUN` must name new directories for each invocation.

```sh
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.turns --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --scratch "$SCRATCH/turns" --out "$RUN/turns" --model-cache "$CACHE" --rows 250000 --blocks 3 --samples 600 --seed 1586
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.recall --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --scratch "$SCRATCH/recall" --out "$RUN/recall" --model-cache "$CACHE" --rows 250000 --blocks 3 --queries 200 --seed 1586
# Functional smoke only: add --smoke --rows 2000 --blocks 1 --samples 50 --queries 20
```

Turns use alternating OFF/ON pairs with a seeded starting state, equivalent
baseline stores and a seeded host trace. Per-block nearest-rank p90s feed a
paired block bootstrap (10,000 resamples); mean matched p90 differences have a
one-sided 95% upper bound. The tolerance is max(50 ms, 10% of mean OFF p90).
Both ingest+preflight and compression-inclusive dispatch wait must qualify;
first-bind deltas must each be at most 500 ms. Wrappers observe preparation,
backfill, embedding-thread joins and denied network attempts. ON completion
before the trace ends invalidates catch-up evidence. At least two pairs are
needed for judgment; short smoke blocks are INVALID.
ON blocks also require a positive corpus covered-count delta and evidence of
background model preparation unless already prepared at block start. Raw summaries
record start/end counts; inactive blocks are INVALID (`on_no_catchup_activity`).

Recall uses product recall defaults, a pinned query-ID manifest derived from
corpus canaries, separate thread-owned connections, and the bgprep deterministic
summary route. Real leaf inserts with fenced publication callbacks define the
committed publication intervals; a call that misses every interval is excluded.
Every block needs 200 overlapping calls and 10 publications. Nearest-rank p50/p90
and their 10,000-resample upper95 bounds must meet 2 s/8 s. Errors, timeouts,
missing FTS hits or undisclosed incomplete coverage in overlapping calls fail health.
Honestly empty semantic arms during catch-up neither fail health nor count toward
the 200 semantic latency samples; covered cells still require both arms. ON-covered
explicitly prepares vectors in a setup thread before measurement; this is not
autonomous coverage evidence. Repeated canaries in small smoke stores do not
qualify the 200-query full-size cell. Raw samples and per-block summaries persist.
ON-catching-up requires positive coverage progress or a background backfill that
embedded rows within the overlapping recall window. Per-recall identity counts
exclude new trace rows; full expected-ID audits run at block start and end.

Dispatch `lcmx-1586-floor.yml` manually with `lcmx_ref`, `host_ref`, `cell`, `rows`,
`shape` (`short`/`mixed`), `baseline` (`empty`/`upgrade`), and
`max_minutes` (121–330). It prepares the cache before measurement, then enters
a cgroup v2 with MemoryMax=4 GiB and runs under `taskset -c 0,1`. The measured
process verifies its effective cgroup and CPU list before starting. All rows are
labelled **floor surrogate**. Coverage-rate requires a backlog lasting at least
two hours; early completion is INVALID for the sustained-rate estimate. Recall
uses the same three states and bars. Workflow artifacts include limits, SHAs,
summaries and raw samples; the release manager owns result-card publication.
The host defaults to the Mac bench pin `71d34bad1045610252eda02ee79549436896e784`;
artifacts also record `pip freeze`. Each summary's PASS/FAIL covers only its named
shape and baseline. The floor bar is met only when every declared shape and
baseline has a PASS row. The workflow moves its shell into the cgroup as root,
then executes the unprivileged Python process; Python only verifies the limits.
Neither synthetic summaries nor functional smoke establish live summary-route,
quiet-host timing, smallest-host, release, customer, or default-flip readiness.
