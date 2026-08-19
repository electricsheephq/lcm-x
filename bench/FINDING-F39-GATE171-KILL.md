# F39 — #171 parity gate: KILL as registered (p50 clause). Root cause: R2 residency was structurally unreachable (dtype gate); design revised, re-registration required.

**Date:** 2026-07-29 · **Run:** gate171 on wt-171-fastscan @4dc5bdd, instrument scale389 @bd23407, pins
PASS pre+post, stores sha-identical pre/post, 0 fail-closes, ~30 min machine. Artifacts:
session-notes/2026-07-29/hermes-r3-1/artifacts/gate171-run/.

## 1. The raw reading (spec clauses)
PASS-1 recall net (b−c): **0/0/0/0** — 20/20 concordant every rung (R1's bit-identity claim held in the
live instrument). PASS-2 A3≥B every rung: MET. PASS-4 deadline: MET (0/600 censored; F34 had 8/150 at
top rung). PASS-6 coverage truthful: MET. **PASS-3 FAILED:** p50 @19,829 = 2,864.6ms vs ≤350ms (7.4×
over; 1,520ms on the most favorable warm reading — still 4.3× over). **PASS-5 FAILED:** small-rung
regression 2.61×/3.85× at 500/2k vs ≤1.25×. **KILL clause fired** (p50 @19,829 > 388ms). Same-session
B-control: recall bit-identical to F34, latency within 0.99–1.20× — the machine is not the confound.

## 2. Root cause — a design assumption the gate caught
`resident_eligible` requires `dtype == int8`; all four frozen 389× stores carry **float32** profiles with
zero binary rows. **R2 (int8 residency, the 39ms design measurement) never engaged** — the gate measured
R1 (vectorized streaming loader) alone: 6.31s → 2.86s at 19,829 (2.2×, real but nowhere near the bar).
The design phase measured residency on synthetic int8 and never verified the deployed/benchmark corpora's
dtype. Additionally R1's streaming setup costs more than the old temp-table path at SMALL rungs — a
product regression for typical small stores.

## 3. Adjudication
- **KILL stands as registered.** No claim ships; the PR does not merge in this form (the small-rung
  regression alone blocks it).
- **The mechanism is revised, not abandoned (design decision):**
  **R2′ — quantize-on-load residency:** build the resident int8 matrix FROM float32 vectors at
  residency-build time (dtype-agnostic eligibility; same RAM budget; ~71MB at 185k). This changes
  scoring vs exact float32 on resident paths, so recall parity is NOT assumed — it is exactly what the
  re-registered gate must test. **Plus size-aware path selection:** below a measured N threshold, keep
  the original simple loader (kill the small-rung regression by not entering the streaming path at all).
- **Re-registration required:** SPEC-171-FAST-SCAN-GATE v2 with identical bars + one addition — the
  small-rung clause gains GRAY/KILL counterparts (v1's PASS-5 had no failure semantics; that was a spec
  authoring gap this run exposed).

## 4. Positives banked (no claims, but real)
Deadline hardening verified live (0 censored vs F34's 8; the R2-train deadline work holds at scale).
Instrument stability re-confirmed (B-arm bit-identical recall, ≤1.2× latency). And the run produced the
FIRST answer-turn delivered-completeness baseline at scale: **0.545 / 0.386 / 0.205 / 0.136** across the
rungs — the session-expansion scale gate's b-pool is far above its ≥20 floor at the top rungs; that gate
is now data-ready once the instrument's sidecar turn-join fallback is fixed (deviation 1: union.jsonl
date-dedup vs per-question dates — the run's private enriched-corpus workaround was sound; the instrument
fallback needs the same fix properly).

## 5. Queue
(1) Revision lane: R2′ + size-aware path selection on feat/fast-scan-residency. (2) Spec v2 freeze.
(3) Gate re-run. (4) Instrument turn-join fallback fix. The #171 PR stays open, labeled not-mergeable
pending the re-registered gate.
