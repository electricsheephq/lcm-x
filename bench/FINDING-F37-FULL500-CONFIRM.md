# F37 — Full-500 confirm: the release V1 number is 455/500 (91.0%), zero fail-closes. The gain is directionally consistent everywhere and individually short of 0.05 — we ship the number with its p-value.

**Date:** 2026-07-29 · **Run:** `full500-confirm-20260729T023012Z`, product `2edb8fc` (the full R2 train:
#168 sanitization + #167 batched scan + #164a store_id + #174 citable delivery), F32 pins verbatim (all
verified pre AND post: harness 2c20cee, codex-cli 0.144.6 PATH-pinned, dataset + qid shas, answerer
gpt-5.6-sol medium / judge LOW). Stores: 1000/1000 frozen-source files sha-unchanged; writes confined to the
private copy. Tokens 1,753,630 of the 2.2M ceiling.
**Artifacts:** `session-notes/2026-07-29/hermes-full500-confirm/artifacts/` (MANIFEST.json, analysis JSON,
scripts, logs). Adjudicated by the orchestrator against the checklist's gate-4 trigger clause.

## 1. Headline

**455/500 raw = adjusted (91.0%). Fail-closed: 0** — the #164-class instrument loss (8 rows / 1.6% in F32's
wave-1 run) is eliminated at full scale, not just on the enriched slice. F32's 8 fail-closed qids all
scored: **7 right / 1 wrong** (`67e0d0f2`, multi-session, banked also 0). Failure buckets both shrank vs the
banked arm: false abstentions 16 → 11, wrong-or-incomplete 40 → 34 (identical matcher rule every arm).

## 2. The paired read — and exactly what we claim

| pair | n | b/c | net | p (exact McNemar) |
|---|---|---|---|---|
| **vs banked A=444** (the release comparison) | 500 | 20/9 | **+11** | **0.0614** |
| vs same-code repeat A′=442 | 500 | 26/13 | +13 | 0.0533 |
| vs F32 wave-1 436 (union-drop, 492) | 492 | 24/12 | +12 | 0.0653 |
| A′ vs A placebo (noise reference) | 500 | 8/10 | −2 | 0.8145 |

The claim discipline (§3 standing rules + candour framing): the direction is positive against all three
baselines, the placebo is flat, and the failure-enriched slice measured the same effect at p=4.0e-05 (F36) —
but **no single full-500 pairing clears 0.05, so R2 does not claim a confirmed accuracy gain.** The notes
carry "455/500, +11 vs our previous base, p=0.061" as measured. The three pairings share the E arm and are
not independent; nobody gets to multiply them.

Important accounting point: **the banked arm had zero fail-closes, so the +11 is genuine answer flips**, not
fail-close recovery bookkeeping (that recovery only appears in the F32-wave-1 pairing, which union-drops it
away anyway). Per-category vs banked: single-session-assistant **56/56 (perfect, +5)**, multi-session +4,
knowledge-update +3, temporal −2, others ≈flat — broad-based, consistent with F36's slice attribution
(#168 waking the provider's internal retrieval arms).

## 3. Delivery-profile deltas — mechanism observations, not gates

- **Hit-mix shift:** delivered hits are now `chunk` 12,499 / `chunk,fts` 1, vs the banked run's ≈even split
  (6,645 / 5,855). Present already on the pre-#174 base, so it entered with the #169 train. Consistent with
  #168 changing which internal path serves raw queries and with #172 (FTS arm cheaply dead on NL queries).
  Outcome-positive here, but it means **the FTS arm is no longer contributing delivered evidence on V1** —
  tracked under #172, not blocking.
- **Telemetry completeness:** 2edb8fc emits `content_returned_chars` on all 25 hits (banked: only the 8
  answer-ready). Apples-to-apples (answer_ready==True only): median 353 vs 340, 12.1% vs 10.9% at the 2400
  cap — delivery volume essentially unchanged. avgContextTokens 3201 vs 3102 (+3.2%).
- Summary-arm hits delivered: 0 (banked 0; F32 wave-1's 13 uncitable ones were exactly its 8 fail-closes).

## 4. Run-integrity disclosure

The original launch was killed externally (background-task stop) during the **evaluate** phase at 356/500,
after search 500/500 and answer 500/500 had completed and persisted. The resume script judged the remaining
144 (harness: "No questions pending search/answering"); **no search or answer was re-run** — every scored row
comes from the single original search+answer pass; the 4 in-flight rows were re-judged from persisted
hypotheses. Script deviation, disclosed: the fresh-launch "no wal/shm in store copy" assert (inapplicable
mid-run) was replaced by a "search+answer 500/500 complete" assert. All other pins unchanged and re-verified
post-run.

## 5. What this licenses

Gate-4's confirm clause is satisfied; the release V1 number is **455/500** and the release-kit slots are
filled with it (notes, README table, #436 body), including retiring the now-false "tie by construction /
retrieval byte-identical" limitation text — that described e99f342, not the final base. Remaining before
upstream push: gate 5 (mono-PR #175 bot rounds, round 1 in progress), gates 6–10 per the checklist. The 95%
statement will cite: F34 (scaling), F36+F37 (fail-close elimination + the number), F32 (base lineage), F33
(causality), and the open known issues (#165 harness-side, #171 ANN, #172 FTS arm).
