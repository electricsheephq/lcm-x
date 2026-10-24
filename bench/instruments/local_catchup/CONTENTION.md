# Warm LOCAL recall contention diagnostic

`contention.py` ports the r1 serial exact-recall harness. It reads an existing
instrument-built synthetic store (`store.db` and `manifest.json` with canaries).
Never supply a live profile, transcript, or customer store. All arguments reject
`.hermes` paths, including resolved symlinks. Home and output must be new,
distinct, and outside the supplied source, cache, and store paths.

Run from the instrument checkout with its supported Python environment and an
already populated model cache. Replace each placeholder with an absolute path:

```sh
python -m bench.instruments.local_catchup.contention \
  --repo <product-checkout> \
  --hermes-source <hermes-checkout> \
  --model-cache <offline-model-cache> \
  --store <synthetic-store-directory> \
  --home <new-isolated-home> \
  --out <new-result-directory> \
  --offset 0 --n 100 --busy-threads 1
```

Use `--busy-threads 0`, `1`, or `2` in separate processes with new homes and
outputs. Each busy thread runs the original `sum(range(10000))` loop, starting
after warmup. Run only one measured process at a time. The caller must apply the
run's declared load-admission rule and retain `uptime` before and after every
block; the script does not enforce or measure host load. The default quiet-host
rule is load1 below 4. An explicit run amendment may permit current-load ABAB
measurement. For #1586 P2, exactness and RSS may run at any load, while every
C1/C2/C3 ABAB block requires load1 below 4 immediately before admission. Wait
in 10-minute checks for up to 3 hours, then record PENDING-QUIET if necessary.
For the P5 continuation, that historical P2 wait is replaced by 60 minutes from
the width fixture's PASS. Record that UTC clock once; do not reset it after a
prototype, commit, fixture refresh, or CI run. Admit C1/C2 only at load1 below 4
and before the recorded cutoff, otherwise retain PENDING-QUIET.
Score ranking exactness only on paired queries where both arms have full
summary and chunk coverage, no timeout and no error. Preserve censored pairs
as latency events, never silently discard them from latency results. Stop on
any ordered top-k ID or score-tolerance mismatch in a full/full pair.
Follow the operator's disk and CPU budgets;
the retained per-query 40 GiB disk stop is not permission to bypass a stricter
launch budget.

The query sequence uses shuffle seed 1586 and alternates class `a` (exact canary
values) with class `b` (the original three broad natural-language queries).
`--n` is the number per class and `--offset` indexes that same repeating sequence.
One canary warmup runs first and is recorded separately, excluded from summaries.
The product's default config, recall deadline, model, and ORT thread count remain
unchanged by default. For P2's additional no-deadline exactness pass, repeat the
same paired queries with `--recall-timeout-s 3600`, a harness-only override of
the existing `recall_query_timeout_s` config field. This equals the existing
`LCM_RECALL_QUERY_TIMEOUT_S` knob; the isolated instrument builds config directly
and clears ambient LCM environment variables. Require full coverage for every
pair in that pass and identical ordered top-k IDs within four float32 ULP
(`4 * 2**-23`); an incomplete pass cannot prove uncensored exactness. Never
combine raised-deadline timings with C1/C2/C3. BLAS/OMP environment thread
counts are explicitly 1; socket network
operations are denied and model preparation disallows downloads when supported.

Outputs:

- `identity.json`: product/Hermes Git SHAs and dirty flags (or a contract digest for a host stub), instrument SHA-256,
  full effective config, product ORT threads, environment, seed, offset, and arm.
- `warmup.json`: warmup timing and complete recall payload.
- `recalls.jsonl`: each query, latency, timeout, error, coverage, FTS hit count,
  FTS latency/error, and complete recall payload.
- `summary.json`: original per-class latency percentiles, maximum, timeouts,
  errors, coverage counts, and zero-hit FTS counts; added FTS totals/latencies
  and `_process` peak RSS. `ru_maxrss` is converted from bytes on macOS or KiB
  on Linux into `peak_rss_bytes`. This is the process high-water mark including
  startup and warmup, not an incremental query allocation or system memory total.
- `memory-attribution.json`: a separate untimed, raised-deadline recall after
  acceptance summaries are saved and busy threads stop. It records transfer
  object sizes, matrix shape/dtype/ownership, traced sites and settled RSS using
  the RCA recipe. Its tracing and diagnostic peak must not enter acceptance
  latency or RSS. A warm summary cache may produce no summary allocation event.

For a separate diagnostic of failed C2, add `--profile` and use a new output/home.
It retains only class `b` after warmup, and writes `profile.pstats`, `profile.txt`,
`spans.json`, and `sqlite-spans.json`. Spans include vector stages and pool-lock
acquisition; SQLite splits chunk BLOB reads, chunk IDs, and other SQL into
execute/fetch time. Profiling perturbs timing: never use it in latency blocks or
combine its rows with unprofiled measurements.

These are GIL-contention diagnostics on one host. They do not prove 2-vCPU
performance, CPU saturation behavior, concurrent compaction, fleet behavior, or
GA readiness. Dirty source flags identify uncommitted work but do not snapshot
it; retain exact diffs and immutable input identities with any measured run.

Citable numbers require a run-root `RESULT-MANIFEST.json` and `RESULT.md` made
with the result-card tooling, raw output snapshots, and durable publication under
the project's evidence policy. This instrument does not create cards or publish
results. Preserve all per-query outputs and inputs needed to re-score. A compile
check alone produces no benchmark claim and needs no result card.
