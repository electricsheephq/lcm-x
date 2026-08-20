# SPEC — Multi-session synthesis presentation (V1 lane #150, primary)

> ## ⛔ WITHDRAWN 2026-07-25, BEFORE DISPATCH — see `bench/FINDING-F28-PRESENTATION-IS-NOT-THE-CONSTRAINT.md`
>
> **The premise in §2 is false.** Verified against the pinned harness source and the banked artifact within hours
> of authoring this spec:
> 1. `renderEvidenceCards` **already groups evidence by session** with a date on every item — it is not
>    "interleaved with distractors and stripped of cross-session structure" as §2 claims. The only real
>    difference this spec proposed was group *ordering* (hit-rank vs chronological), a permutation of the same
>    bytes.
> 2. Gold evidence is **not under-represented**: multi-session FAILURES receive *more* gold hits than passes
>    (11.5 vs 11.2 of 25), with identical top-session share (0.20) and distinct-session count (8.9).
>
> With retrieval already excluded by F27 (98.5% all-golds on multi-session), all three presentation mechanisms are
> ruled out with data. The reader gets complete, well-represented, session-grouped, dated evidence and still
> answers wrong — a reasoning failure we do not own on this benchmark.
>
> **This spec was never run and no tokens were spent on it.** It is kept, unedited below, as the record of a
> mechanism hypothesis that died on contact with its own artifact, and because its gate design (paired, full
> category, pre-declared floor, one-primary) remains the correct template for the next spec that has a live
> premise.

---

**Status:** ~~FROZEN on authoring, 2026-07-25~~ → **WITHDRAWN, not dispatched.** Gate declared before any run, per §2b.
**Supersedes** #150's preference-application framing (F27 §0 corrected the premise).
**Base:** parameterised — whichever provider wins the pending full-500 wave-1 comparison (`bench/w3b-on-wave1`
vs the banked #423 base). The spec is written against *the winning base*, and the base must be pinned by commit
in the run's provenance block before the control arm executes.

---

## 1. Target, stated as questions not points

F27 measured, on the banked 444 at full 500:

- multi-session: **107/133 = 80.5%**, our worst category
- its 26 failures: **25 had EVERY gold session retrieved**, 1 partial, 0 with nothing
- overall, **52 of 56 failures** had complete gold evidence in the prompt

So the addressable set is **25 questions where the evidence arrived complete and was not correctly combined.**
This is a synthesis defect, not a retrieval defect, and no retrieval change can be credited against it.

## 2. Mechanism under test — and the line it must not cross

**Not touched:** the reader/answer prompt, the judge, the question set, the stores, the retrieval ranking.
Editing the reader prompt is forbidden (§2b) and would also invalidate every banked comparison.

**Under test:** `answerPresentationMode` — how our own hits are rendered into the prompt. Today
`evidence_cards_v1` emits per-hit excerpts ordered by hit rank. A question needing 2–5 sessions therefore
receives its evidence **interleaved with distractors and stripped of the cross-session structure** that makes
composition possible. Mean distinct sessions among the 25 logged hits is **9.3**, so a 3-gold question arrives as
3 relevant sessions scattered among ~6 irrelevant ones with no signal distinguishing them.

**Treatment (`evidence_cards_v2_grouped`, default OFF):** group hits by source session, order groups
chronologically rather than by score, and label each group with its session date and a one-line span descriptor.
No new content, no summarisation, no LLM call in the presentation path — pure regrouping of bytes we already send.

**Why this and not summarisation.** A synthesised cross-session summary would insert *our* interpretation into
the evidence path, which (a) makes the memory layer answer the question rather than serve it, and (b) is
unfalsifiable against the judge — we could not tell a retrieval win from a summarisation win. Regrouping is the
weakest intervention that could produce the effect, which is what makes a positive result attributable.

## 3. Instrument

- **Question set: all 133 multi-session questions.** Not an enriched failure-only slice — F27 §0 is the reason.
  A failure-enriched slice cannot produce a category rate and would hide breakage among the 107 passes, which is
  exactly the risk this mechanism carries (reordering evidence can break questions that currently work).
- **Paired, same stores, same base commit, same answerer/judge/effort.** M9 parity diff of `run_args.json`
  between control and treatment before either result is read. Any diff outside `answerPresentationMode` voids
  the pair.
- **Control arm must be re-run, not reused from the banked 444.** The banked run's judge effort is `low` and its
  harness is `wt-v1l1 @ 2c20cee`; a fresh treatment against a stale control is the asymmetric-instrument error
  from M15 §7. Both arms in the same session, same pins.
- Cost estimate: 2 × 133 questions ≈ 0.9M tokens total, extrapolating from the banked 500-question run's 1.70M.

## 4. GATE — declared now, never relaxed after results

Let **b** = control-wrong → treatment-right, **c** = control-right → treatment-wrong, over the 133 paired
multi-session questions.

**PASS requires all three:**

1. **b − c ≥ 6** (the net must clear F24's noise characterisation; two identical V1 runs differed on 18
   questions with a net of 2, so a net under 6 on 133 questions is not distinguishable from re-running the same
   code)
2. **McNemar exact p < 0.05** on (b, c), two-sided
3. **Floor: c ≤ 4.** More than 4 currently-correct multi-session questions broken is a NO-GO **regardless of b** —
   a mechanism that trades 5 working questions for 8 new ones is not a memory improvement, it is a reshuffle, and
   it will not hold on a different question set.

**Also reported, not gated:** the same b/c/p on the 367 non-multi-session questions as a **collateral check**. If
that set shows c − b ≥ 4, the mechanism is degrading the categories it was not aimed at and the result is
referred to the owner rather than adopted, even on a passing primary.

**One-primary law:** the primary is the multi-session paired comparison. Nothing else in this run may be promoted
to a headline, including any overall /500 number.

## 5. What each outcome means (written before seeing any)

- **PASS** → the defect was presentational and the fix is ours. Promote `evidence_cards_v2_grouped`, then
  re-measure the full 500 for the release number. Expect roughly +6 → 450, not more; do not extrapolate the
  multi-session rate to other categories.
- **NO-GO with b − c < 6 and c ≤ 4 (null)** → grouping is not the missing signal. The evidence is complete and
  legibly ordered and the reader still fails to combine it. That relocates the defect to the reader's reasoning,
  which we do not own on this benchmark — and it makes the honest V1 conclusion "our memory layer is saturated;
  remaining V1 headroom belongs to the model". Report it that way; do not iterate presentation variants hunting
  for a win.
- **NO-GO on the floor (c > 4)** → hit-rank ordering is load-bearing in a way we did not model. Do not retry with
  a tweak; first explain *which* questions broke and why, because that is a finding about our ranking.
- **PASS on primary but collateral c − b ≥ 4** → owner decision, not an adoption. A category-local win that costs
  other categories is a product trade-off, not a benchmark result.

## 6. Secondary lane (separate spec, separate run — NOT bundled here)

False abstention (14 questions, F27 §2b) is the secondary target and must not be combined with this run. Bundling
two mechanisms into one arm is the M13 blended-arms error: when the result moves you cannot attribute it, and
when it does not you have spent twice to learn nothing. Preference (5 questions, 4 of them false abstentions with
gold at hit rank 0–6) is the development vehicle for that lane because it is the cleanest instance — its spec
declares its own gate on the preference set plus a non-preference control, with the floor that no
currently-passing preference question breaks.

## 7. Pre-registration record

Authored 2026-07-25 by the program architect, before any treatment run existed, against F27's measured
decomposition. Gate values (b−c ≥ 6, p < 0.05, c ≤ 4) chosen from F24's noise floor and from the size of the
addressable set (25), not from any observed treatment result. Tightenings before results are permitted and must
be logged as such; relaxations are not permitted at any point.
