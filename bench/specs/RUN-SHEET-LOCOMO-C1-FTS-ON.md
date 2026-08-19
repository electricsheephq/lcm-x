# RUN SHEET — LoCoMo declared config C1 "FTS-ON" (registered; launch gated on machine slot)

Registered 2026-08-02, BEFORE launch. G0 channel: performance lever. Baseline of record:
F48 declared config 54.6% (1085/1986), A/A′ discordance 3.47%, aggregate spread 0.76pt.

## 1. Config delta vs F48 declared (exactly two coupled knobs, coupling disclosed)
1. `LCM_FTS_PROSE_MODE=1` (F49: the flag whose absence kept #183 dark in the banked run).
2. `HERMES_MB_FUSION=quota:fts=1,chunk=1` (re-derived by SPEC-FUSION-RESIM under the a-priori
   rule; the old 1:2 was tuned against a near-empty arm and is known-stale for a live arm).
Coupling rationale: the ratio derivation is CONDITIONAL on prose mode; registering them
separately would run a config (prose@1:2) the sim already measured as dominated (50.24% vs
53.66% delivery). The sim table IS the per-knob attribution: chunk-only control 40.00%,
prose@1:2 50.24%, prose@1:1 53.66% (recomputed from raw CSV by the orchestrator).
Everything else IDENTICAL to F48 pins (product 9d181aa, harness chain, judge gpt-5.6-sol@low
narrowed rubric, fastembed bge-small, thresholds, answer-ready chars).

## 2. Sim-to-run expectation (honest framing)
Sim measures turn-level GOLD DELIVERY on a category-balanced 100q sample: as-run ≈ 40.0%
(FTS-dark ⇒ chunk-only control) → C1 53.66% (+13.7pt delivery). Delivery→answer conversion is
lossy (F48: answer-variance and judge strictness absorb part); bars below are set on ANSWER
accuracy, conservatively.

## 3. Pre-declared verdict bands (aggregate scored read, arm A; A/A′ agreement per F48 practice)
- **PASS:** ≥ 56.1% (≥ +1.5pt over 54.6, ~2× the F48 A/A′ spread) AND no category falls >4.0pt
  below its F48 §3-CORRECTION value (adversarial floor 28.7 — attribution is C2's job, not C1's).
- **GRAY:** +0.76 to +1.5pt → disposition in writing; default ADOPT only if the category
  pattern matches the sim's prediction (single-hop, multi-hop, world up; temporal ~flat).
- **FAIL:** < +0.76pt (inside noise) or any category collapse >4.0pt → C1 not adopted; C2
  baselines on F48 config; finding documents why delivery gains didn't convert.
No post-hoc bands. GRAY/FAIL publish at the same resolution as PASS.

## 4. Pins & disclosure (finalized at launch; every value from a command)
- FIRST registration under the F49 §5 standing rule: pins include the FULL `_EnvFieldSpec`
  env-flag inventory diff — every flag explicitly set or listed default-with-value.
- Seven-point row per F46 §5; A/A′ pair (2×1,986); fail-close accounting; 99 corrupted-gold
  rows run as-is and disclosed (ceiling ≈95%), unchanged from F48 for comparability.
- Spend lane: codex CLI (gpt-5.6-sol answer/judge) — NOT OpenRouter; unaffected by the credit
  outage. Embeddings local fastembed.

## 5. Scheduling constraint (ops, not measurement)
Launch AFTER the V1-M smoke completes (single active heavy-writer rule; disk-I/O starvation
class documented). At the launch fork: V1-M registered run (flagship) has priority for the
machine; C1 interleaves only if measured load allows (different bottlenecks: Voyage-API vs
codex-CLI) — decided at launch time and logged, shard/arm topology is operational not measured.
Freeze-writers protocol armed during any bridge-init window.
