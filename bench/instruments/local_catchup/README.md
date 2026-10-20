# LOCAL catch-up instrument (I1)

Binding authority: [#1586, Gate declaration (release manager), v1](https://github.com/electricsheephq/lcm-x/issues/1586#issuecomment-6283060211).
Read the declaration before running; the instrument does not change its bars.
Use Python 3.11+ with pytest, psutil (optional), FastEmbed and ruff 0.15.13.
The Hermes source supplies the host ContextEngine contract; its CLI is never run.
Set shell variables to absolute paths: `PRODUCT` (any product checkout), `HOST`
(Hermes source), `SCRATCH` (new regenerable bulk), `RUN` (new evidence directory),
`CACHE` (prewarmed model cache with **all** write bits removed), `PY` (Python).
No live profile is used. Never pass a live DB or home.

```sh
python -m pytest bench/instruments/local_catchup/tests -q
ruff check --no-cache bench/instruments/local_catchup
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.corpus --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --out "$SCRATCH/corpus" --rows 250000 --shape short --baseline empty --pilot 1000 --peak-disk-gb 8
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.catchup --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --db "$SCRATCH/corpus/store.db" --manifest "$SCRATCH/corpus/manifest.json" --home "$SCRATCH/homes" --model-cache "$CACHE" --out "$RUN/catchup" --max-wall 24h
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.probes drain96 --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --db "$SCRATCH/probe/store.db" --manifest "$RUN/probe-manifest.json" --home "$SCRATCH/probe/homes" --model-cache "$CACHE" --out "$RUN/probe" --max-wall 20m --restart-after 30s
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.probes cold --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --db "$SCRATCH/corpus/pilot.db" --manifest "$SCRATCH/corpus/pilot.json" --home "$SCRATCH/cold/home" --model-cache "$CACHE" --out "$RUN/cold"
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.probes screen16 --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --scratch "$SCRATCH/screen" --model-cache "$CACHE" --out "$RUN/screen" --rows 96
nice -n 19 "$PY" -B -m bench.instruments.local_catchup.rig --lcmx-repo "$PRODUCT" --hermes-source "$HOST" --scratch "$SCRATCH/rig" --model-cache "$CACHE" --out "$RUN/rig" --processes 14 --rows 250000 --peak-disk-gb 80
```

Repeat corpus/catchup for `--shape mixed` and `--baseline upgrade` in new directories.
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
to 40% coverage, with a declared 10-minute fallback if that fraction is unreachable;
the small control uses 30 seconds. Shutdown has a 10-second bound, then kill.
Restart and interpreter startup count toward elapsed time. Every 30 seconds a
separate read-only connection audits all expected IDs, vector lengths, raw-row
digests, pre-existing vectors and uncertain/inflight state. Zero progress includes
persisted scan progress, and STALLED requires a successfully constructed provider.
CPU samples are every second, CPU-time deltas in cores, weighted means and full
60-second rolling means with nearest-rank p95. Missing 60-second windows are
insufficient evidence. Default bars: .6/.75 per PID, 7/8.75 aggregate; abort at
>12 cores for 10 seconds or increased host swap. Existing swap is recorded.
`--cpu-cap`, `--cpu-sustained`, `--per-mean`, `--per-p95`, `--aggregate-mean`,
`--aggregate-p95` expose the declared supervisor thresholds. psutil is preferred;
the fallback reads cumulative `ps time`, RSS and platform-specific thread counts.

Raw CPU/progress/events, worker status, final DB, manifest and summary are receipts.
The 16-process 60-second screen is reported and never gates the default flip.
The cold probe pairs OFF/ON binds and requires FTS canary hits with honest semantic
degradation. A control stall is an expected negative, not a product acceptance pass.
Small controls prove instrument sensitivity only: no 250k, floor-host, turn,
compaction/recall, release, customer, or default-flip claim follows.
At closeout initialize/add/headline/finalize/check a result card with `bench-card.py`
for the positive-control evidence directory. The release manager owns publication.
