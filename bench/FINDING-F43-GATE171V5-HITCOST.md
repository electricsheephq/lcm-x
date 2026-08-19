# F43 — Gate #5: KILL (PASS-3, PASS-5@2k). Mechanism fully restored (v3 shape, recall net 0 everywhere); the residual is a per-HIT staleness recheck costing up to 334ms/query. One repair; execution #6.

**Run:** gate171-v5 on ea66eaa, spec v2 unchanged, pins/stores/probes sha-matched, both chains complete,
zero LLM, 17m19s. Artifacts: hermes-r3-1/artifacts/gate171-v5-run/.

## Read
Residency: 1 build/99 hits/100 ranked at every rung (exactly v3); temp tables GONE at 8k/19,829; recall
net 0 vs both baselines (70/70 pairs concordant); scoring/reach disclosures clean; B-control shows the
machine ran FASTER than v3's session (0.93× at top rung) — the regression is the build's.
**KILL:** p50 @19,829 = 466.93ms (>388); 2k = 1.66–2.03× raw (>1.6×). Cold first-touch also grew
(8.6–9.5s vs 5.0–7.4s).

## Root cause (from the run's own evidence)
Cache-HIT second-call cost exploded 0.01–0.02ms → 1.5–333.9ms (334ms chunk@19,829): the disposition
batch's row-2/row-3 correctness rechecks (profile version + live cardinality before publishing) run PER
QUERY on the hit path — a full recount against 185k rows per hit. 222.7 (v3) + ~240 recheck ≈ the 467
observed. The check is redundant per-hit IF deletes/writes bump data_version (residency is already
data_version-keyed); if any mutation path does NOT bump it, bump it there instead.

## Repair (execution #6 after)
Make hit-path staleness free: validate via the existing data_version key only; move cardinality/profile
rechecks to build time + invalidation events (and close any mutation path that fails to bump
data_version, with a regression). Add a hit-cost regression: cache-hit service time must be O(1) —
assert no COUNT/scan SQL executes on the hit path (statement-count telemetry as a test, the F42 pattern).
