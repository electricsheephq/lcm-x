# SPEC — Session expansion, scale-regime gate (pre-registered BEFORE any run; supersedes the V1-small Stage-2 gate per F38)

**Status: FROZEN on commit.** Tightenings allowed before results, logged; never relaxed at one-short.

## Question
Does `session_expand_v1` (merged dormant, #173) improve **delivered answer-turn evidence completeness**
in the scale regime — corpora where answer evidence is genuinely scattered and the ranked tier cannot
carry whole sessions — at acceptable token cost?

## Instrument
The F31/F34 389× single-store family (zero/cheap-LLM): rungs S=500, 2k, 8k, 19,829 sessions; the same
frozen probe-question set and gold answer-turn joins used by F34 (content-based joins only, §6e.9).
Measured per rung, expansion OFF vs ON (flag flip is the ONLY delta; same store copies, sha-verified via
bench/tools/storefreeze; pins via bench/tools/pinverify; fail-close via bench/tools/failclose):
- **Primary: answer-turn completeness of the DELIVERED payload** (all gold answer turns present in what
  the reader would receive), per question, paired OFF→ON per rung.
- Secondary: delivered tokens per question (the cost side); expansion telemetry (windows, containment
  drops, strict rejections).

## Pre-registered gate (paired, per §3 standing rules)
- **PASS:** at the 8k and 19,829 rungs, expansion ON increases complete-delivery question count with
  paired net (b−c) ≥ 8 per rung on the probe set, AND median delivered tokens/question inflate ≤ 1.6×.
- **KILL:** net ≤ 2 at both top rungs, or token inflation > 2.2× at any passing rung — the mechanism dies
  for the scale thesis too, documented like every NO-GO.
- **GRAY (3–7 net at top rungs):** owner decision with the power memo attached.
- Power memo (instrument-1 discipline) computed from the probe set's still-incomplete counts BEFORE the
  run; if the b-pool at the top rungs is < 20, STOP and re-derive the probe set instead of running (the
  F38 lesson: verify the denominator, never the gate).

## Explicitly out of scope
V1-small accuracy (saturated, F38 §1); any LLM-scored accuracy claim (this gate is a delivery-mechanism
gate; an accuracy experiment would be a separate registration on the tier where it matters — V1-medium).

## Cost ceiling
Zero LLM spend for the primary (delivery analysis is offline); ≤ 50k harness-unit tokens total if any
smoke verification needs the reader; hard stop beyond that.
