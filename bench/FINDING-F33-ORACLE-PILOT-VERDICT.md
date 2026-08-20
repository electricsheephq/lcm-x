# F33 — Oracle pilot: evidence completeness CAUSES the V1 failures. Stage-1 gate PASSED (23 ≥ 12); session expansion is licensed.

**Date:** 2026-07-29 · **Spec:** SPEC-EVIDENCE-ORACLE-THEN-EXPANSION §Stage-1, gate pre-declared before the run.
**Cost:** 290,779 harness-reported tokens (ceiling 0.5M held) · **Pins:** the F32 set verbatim (harness 2c20cee,
answerer sol/medium, judge sol/LOW, codex-cli 0.144.6 PATH-pinned, `evidence_cards_v1`); answer-phase-only replay
(`search never re-ran`); no store writes (read-only `immutable=1`; private empty workdir).
**Artifacts:** `session-notes/2026-07-29/hermes-oracle-pilot/artifacts/` (frozen qids + sha, injection ledger,
offline render validation, smoke + full reports, all scripts per §6e.7).

---

## 1. Design integrity (why these numbers can be trusted)

- **Frozen subset before any run:** n=80 (the spec's 67+16 overlap by 3 — documented, not padded).
- **Offline render validation BEFORE any LLM call:** 80/80 rendered, 0 fail-close, all 107 injected items
  resolve via the validator's offset path, and the control prefix renders **byte-identical** to control-only —
  injection is provably additive. The #165 trap that killed spec v1 was defused by construction, then verified.
- **Smoke gate honoured:** 20 questions, 0 fail-closed vs baseline 0 → proceed.
- **Truthful injection:** gold turns injected with real store metadata (offsets, ids, dates from the store and
  sidecars); 83 turns already delivered byte-identically were NOT duplicated; nothing invented.

## 2. The result

| n=80 paired | control | oracle |
|---|---|---|
| correct | 45 | **66** |

**b = 23 flips-to-right · c = 2 breaks · net +21 · exact McNemar p = 1.94×10⁻⁵.**

The two breaks are genuine reader responses to added evidence, not artifacts (one is arguably a dataset quirk:
the injected turn exposes a 5th event where gold says 4). Fail-closed rows: 0.

**GATE: PASSED** (pre-declared: ≥12 proceed / ≤6 lane dies / 7–11 owner). **Stage 2 — session expansion — is
licensed.**

## 3. What this establishes and what it does not

**Establishes:** the F29 completeness→accuracy gradient is CAUSAL at the margin, not selection-confounded —
with complete evidence delivered, **23 of the 35 currently-wrong subset questions flip** (66% of the hard
ceiling). This is the strongest capability signal the V1 program has produced: the reader is not the binding
constraint on these questions; *our delivery is*, and fixing delivery fixes answers.

**Does not establish:** a product result. This is an oracle upper bound — the gold turns were injected by an
all-knowing hand. A real mechanism (session expansion, Stage 2) reaches some fraction of it; Stage 2's gate
(b−c ≥ 8 on this same frozen subset) requires ≈40% oracle-efficiency, which the 92%-of-pool reachability
analysis (red-team `reach.py`: the missing turns' sessions are already delivered) makes plausible but not
promised. Ceiling arithmetic for expectations: +21 net on the subset ≈ up to ~+4 points/500 at perfect
mechanism efficiency; expect less.

## 4. Two instrument notes

1. **`answer_turns.json` carries 5/190 off-by-one `session_idx` entries** (each resolved by unique
   exact-content match in the store). Anyone reusing the audit's extraction should content-match, not trust
   the index — noted on the artifact.
2. **Token budgets must NAME THEIR UNIT (§6e.14 candidate):** the harness's `report.totalTokens` (290,779
   here; the unit all our ceilings have used) vs codex-CLI wire usage (2,159,056 — includes transport
   overhead, both roles, input+output) differ by ~7×. All existing ceilings and comparisons remain
   harness-unit and consistent; future specs state the unit explicitly.

## 5. Next (per spec, no new decisions needed)

Stage-2 implementation (`session_expand_v1`) branches **after PR #169 merges** (same files; avoid conflict) —
implementation dispatches to the cross-model lane, review returns to Claude, run executes only when the machine
is clear of the Phase 1B latency re-run. The Stage-2 run gate and the frozen subset are already fixed; nothing
about this result relaxes them.
