# SPEC v2 — Oracle pilot, then session expansion (V1 lane #150 / task #25)

**Status:** DRAFT 2026-07-29, designed jointly with the pre-freeze red-team's findings (all six of its exact
revisions adopted). **Freeze condition:** the architect's review of this v2 against the red-team's FINDINGS.md
line-by-line (done at authoring), plus the STAGE-1 GATE below — no full-scale spend before the pilot reads out.
**Evidence base:** red-team artifacts `session-notes/2026-07-29/hermes-spec-redteam/artifacts/` (reach.py:
T_TRUNC=0, S_PRESENT=92%; cost.py: turn-complete=1.41×, top-3-session≈10–13k tokens; the 300-char tier:
`_LCM_RECALL_SNIPPET_CHARS=300`, 55.7% at cap; the delivered-fraction signal worth ~10 questions).

---

## STAGE 1 — the ORACLE PILOT (run first; ~5% of full-run cost; kills or licenses the lane)

**Question:** if the missing/partial answer-turn evidence were present in delivery, would these questions
actually flip? This measures the CAUSAL ceiling directly and bypasses the selection confound in F29's gradient
(retrievability correlates with easiness — the gradient's arithmetic cannot be trusted for flip prediction).

**Design:** the pre-identified subset (n≈83: the 67 evidence-incomplete + 16 truncation-affected wave-1-control
questions — list frozen from `reach_wave1_F32.json` BEFORE any run). For each, construct the delivery as the
control run delivered it PLUS the known gold answer turns injected as properly-formed hits (with
`content_offset`/`content_returned_chars` set so the evidence-cards validator takes its offset path — verified
against `evidence-cards.ts:43-74`). Reader+judge under the F32 pin set verbatim (codex 0.144.6 PATH-pin, judge
LOW). **20-question fail-closed smoke test before the remaining 63** — any fail-close above the known #165
baseline aborts for diagnosis.

**Read:** paired flips vs the same questions' control outcomes. These are oracle numbers — an upper bound, not
a product result; no category rates (§6e.8), flip counts only.

**STAGE-1 GATE (pre-declared):** oracle flips ≥ 12 of the ~83 → proceed to Stage 2 (the ceiling is worth
chasing: ≥12 oracle flips ≈ the b≈15 Stage-2 gate becomes plausibly winnable at realistic mechanism
efficiency). Oracle flips ≤ 6 → **the lane DIES here** at ~5% of the planned spend, and F29's attribution is
revised: evidence completeness predicts but does not cause at the margin claimed. 7–11 → owner briefing with
the per-question detail; no unilateral proceed.

**Cost:** ~83 questions ≈ 0.3M tokens. Ceiling 0.5M, hard stop.

## STAGE 2 — SESSION EXPANSION (only if Stage 1 passes; the mechanism aimed at the 92%)

**Treatment (`session_expand_v1`, default OFF):** ranking untouched; for the top-K hit sessions (K=3 from the
red-team's cost curve: ≈10–13k tokens delivered), deliver an expanded window of the session around each hit
(±N turns growing to whole-session if within per-session budget), every hit carrying
`content_offset`/`content_returned_chars`. Hits 9–25 keep their current 300-char snippet form (the two-tier
structure is now understood; the expansion targets the sessions, not the snippet tier).

**Instrument:** control = F32 run (union-drop convention for fail-closed rows, both arms). Treatment = full
500 under the F32 pins. **Primary gate on the pre-declared n≈83 subset, flip counts:** b−c ≥ 8 within-subset,
McNemar p<0.05. Full-500 paired result reported as secondary (no gate — the F32/A′ null pairs showed c≈8–10
full-set noise, so the full-set floor is an ESTIMATOR, not a bar: the byte-identical-delivery subset across
arms provides the in-run noise reference, per the red-team's (f)). Mechanism check: answer-turn coverage on
the subset must reach ≥90% delivered (measured with the offset-aware method, not the 25-char substring proxy).

**Cost:** ~1.7–2.4M tokens (context ≈1.4–1.9× control per cost.py, NOT 4×). Ceiling 3M, hard stop.

## Corrections this spec carries forward

- F32 §4's "14% of budget" framing is amended (two-tier delivery: top-8 hydrated at 2,400 cap; ranks 9–25 at
  300 with 55.7% AT cap) — correction noted on F32.
- The delivered-fraction signal (accuracy 80.4% at 0.5–0.9 fraction vs 95.7% at 0.9–1.0, ≈3.6σ, ~16 wrong
  questions) is Stage-1 evidence in favour of causality but is selection-confounded like the gradient — the
  oracle exists precisely to test it.
- Provenance: all subset counts are computed on the wave-1 CONTROL run (471/67/19 & 92.6/75.0/47.4), not the
  banked #423 numbers.

## Boundaries (unchanged from v1)

Not run on V2-small (F30 §3). No reader-prompt contact (§2b). No ranking changes. Scaling lane (#167/#168,
Phase 1B) independent. Every claim names its granularity (§6e.11).
