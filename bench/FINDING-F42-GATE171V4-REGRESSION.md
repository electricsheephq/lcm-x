# F42 — Gate #4 (confirmation): KILL — the 20-finding disposition batch broke both performance mechanisms while fixing correctness. Two surgical repairs identified; the confirmation-run rule is now twice-proven.

**Date:** 2026-07-29 · **Run:** gate171-v4 on 1516214, spec v2 unchanged, pins/stores/probes sha-matched,
zero LLM. Artifacts: session-notes/2026-07-29/hermes-r3-1/artifacts/gate171-v4-run/.

## What the run found
1. **Residency disengaged on the live path — 0 builds, 0 resident rankings at every rung** (v3: 1
   build/99 hits/100 ranked). The registry re-keying from disposition rows 5/7 (budget-partitioned keys,
   per-store namespaces) evidently derives DIFFERENT keys at build-site vs lookup-site on the live path;
   the direct structural probe still builds fine. Correct fixes, broken wiring.
2. **The row-8 recency restoration reintroduced a per-query O(N) ordered temp table** — the exact pattern
   R1 existed to kill. With residency dark, nothing offsets it: p50 @19,829 = **19,994ms** (89.8× vs v3;
   **3.17× WORSE than the F34 pre-fix baseline**); 44/50 top-rung queries blew the 8s deadline (PASS-4
   now failing too). The regression GROWS with N.
3. The batch's correctness content VERIFIED GOOD: suppression/orphan joins produce zero reach divergence
   (identical id sets, 40/40 full-reach comparisons); scoring disclosure field correct everywhere; recall
   parity held (net 0 vs F34; 0/−1/0/0 vs v3).

## Adjudication
- **KILL confirmed for 1516214; PR #184 must not merge at this head.**
- Two surgical repairs (revision lane): (a) unify residency key derivation between build and lookup on
  the live path (add a regression asserting resident ranking count == query count on a ≥crossover
  fixture — the v4 telemetry read as a TEST, so this class can never ship dark again); (b) newest-first
  on truncated scans WITHOUT a per-query temp table — order the candidate enumeration itself (ORDER BY
  on the streaming cursor / in-memory id sort before batching).
- Gate execution #5 after the repairs. Spec v2 unchanged.

## The rule this proves (second time)
A review-fix batch on a measured surface is an UNMEASURED build until the gate re-runs. v4's batch would
have shipped a 3× regression labeled as fixes. The confirmation run is not overhead; it is the difference
between a fix and a claim.
