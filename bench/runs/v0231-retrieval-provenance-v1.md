# v0.23.1 retrieval-provenance v1 run sheet

Status: registered instrumentation candidate; no score or runtime claim.

This sheet records the bounded benchmark design for the content-free retrieval
funnel sidecar. It is not a product telemetry contract and must not be pointed
at private customer or live Eva data.

## Frozen identity

- Registered product: `v0.23.1@81d8d41197dddc4c09b57097f4955ebae32366a9`.
- Delivery base: `main@3d4fbb4c979dc09aef0b831bb50d928e0e18d68f`.
- Instrument: schema v1, `--dump-retrieval-funnel PATH`, `--expected-dim 1024`.
- Dataset: pinned 500-question LongMemEval V1-M public corpus and the registered
  100-question subset selected with seed `20260802` (95 scored questions and
  five abstention cases). Record the source, prepared-manifest, and
  subset-manifest digests before inspection.
- Provider/model, cache state/key, environment, config, seed, weights, and
  top-k are frozen in the sidecar header.
- Retrieval configuration: `voyage-4-large`, 1024-dimensional float32,
  rerank/prescreen/proactive off, and weights FTS `0.5`, summary `1.0`, chunk
  `1.0`.

## Run sequence

1. Estimate billable tokens using the current official Voyage price. Stop
   before a provider call if the full registered run could exceed USD 10.
2. Run a seeded-random smoke from the registered seed-`20260802` subset.
   First-N selection is invalid.
3. Run the same registered 100-question subset twice (A/A-prime) from an
   immutable cache and score its 95 answerable questions. After normalizing
   generated/timing fields, IDs, ranks, metrics, counts, and configuration
   hashes must be byte-identical.
4. If A/A-prime is accepted, run all 500 questions (470 scored plus 30
   abstention cases) with the same frozen manifests and configuration.
5. Preserve the JSONL sidecar, checkpoint, metrics, markdown, and manifests in
   a durable run directory. Do not append a different corpus or configuration.

Example command:

```bash
python scripts/lcm_longmemeval.py run \
  --prepared-dir /path/to/public-or-scrubbed-prepared \
  --output /path/to/run-root \
  --provider voyage --model voyage-4-large \
  --dump-retrieval-funnel /path/to/run-root/retrieval-funnel.jsonl \
  --expected-dim 1024
```

The 100-question A/A-prime uses its registered prepared subset directly; it
must not be produced with `--limit`, because that would select the first N
questions instead of the seeded sample.

## Metrics and decision

For each of the 470 scored questions, `gold_at_1` is one only when the first
deduplicated shipped `lcm_recall` session belongs to the registered gold set.
Report `gold@1_rate = sum(gold_at_1) / 470`, shipped recall@1/10, nDCG@10,
hard misses, delivered-but-not-first cases, per-arm oracle rescue, coverage and
fallback classes, reference validity, latency, token accounting, and cost. No
reader or judge participates.

The default decision is `KEEP CURRENT`. Record `FUSION DESIGN EARNED` only if
at least 15 of 470 current-run misses are delivered-but-not-first while another
existing arm ranks a gold session first, with zero instrument, reference, or
fallback failures. That result is oracle headroom only; it does not authorize a
retrieval change. A later label-blind candidate must independently improve
gold@1 by at least 3.0 points, preserve standard recall@1, keep recall@10 and
nDCG@10 inside the zero A/A-prime band, and have no more than one demotion per
four promotions.

## Sidecar acceptance

Each row may contain only question/gold IDs, arm IDs/ranks, shipped hit kind
and reference IDs, coverage, degraded status, safe reason codes, embedding
accounting, counts, timing, and `reference_valid`. Message/chunk store IDs and
summary node IDs must resolve to the matching question session. Unknown
fallbacks, dimension mismatches, invalid references, foreign ownership,
duplicate IDs, and torn non-final lines fail closed before appending a row.

The default-off path must retain byte-identical aggregate/checkpoint behavior,
candidate-dump behavior, runtime APIs, product schemas, and provider defaults.
The sidecar must never contain raw text, snippets, prompts, queries, answers,
provider payloads, secrets, or raw configuration.

## Evidence boundary

This run sheet proves only deterministic source/tests/docs registration and the
benchmark sidecar's local behavior. It does not prove retrieval quality,
provider determinism, release readiness, runtime safety, customer readiness,
or a score until both frozen runs complete with their own manifests and
artifacts.
