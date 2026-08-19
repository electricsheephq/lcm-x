# F28 — V1 multi-session failure is NOT a presentation problem: three mechanisms ruled out, spec withdrawn

> ## ⚠ AUDITED 2026-07-29 — core conclusion SURVIVES (reinforced by placebo cross-check); two parts struck. See F29.
> Struck: §2's "failures receive MORE gold evidence" (unnormalized-rate artifact — reverses per gold session)
> and §3's "golds not retrieved: excluded" row (inherits F27's session-vs-evidence conflation). Survives,
> reinforced: no reliable grouping/ordering/crowding lever exists — both audit refutation attempts failed
> verification (the claimed "fix" run was a different harness build scoring 442, and its flipped questions also
> flip on F26's placebo). NOT tested by this finding and now the live lane: **within-session evidence
> completeness** (session expansion; answer-turn coverage) — F29 §6.

**Date:** 2026-07-25 · **Spend:** zero — re-analysis of the banked 444 plus a read of the pinned harness source.
**Consequence:** `bench/specs/SPEC-MULTISESSION-SYNTHESIS.md` is **WITHDRAWN before dispatch.** Its premise was
false and I falsified it myself, hours after writing it, by checking the premise instead of the conclusion.
**Artifacts:** `session-notes/2026-07-25/hermes-preference-gap/artifacts/hits_per_gold.py`, `hits_per_gold.json`.

---

## 0. What the withdrawn spec claimed

F27 established that multi-session is our largest V1 loss (25 of 26 failures with **every** gold session
retrieved). I inferred a presentation cause and wrote a spec around it:

> "A question needing 2–5 sessions receives its evidence **interleaved with distractors and stripped of the
> cross-session structure** that makes composition possible." → treatment: group hits by session, order groups
> chronologically, label each group.

Both halves of that sentence are wrong.

## 1. The harness ALREADY groups by session (read from the pinned source)

`renderEvidenceCards` in `src/prompts/evidence-cards.ts` (harness pinned at `wt-v1l1 @ 2c20cee`, the commit the
banked 444 ran on) builds a `Map<sessionHandle, CardItem[]>` and emits one section per session:

```
SESSION <handle>
SOURCE DATE <date>
[<exactRef> | <role>]
<content>
```

So evidence is **not** interleaved and **not** stripped of session structure — it arrives grouped by session with
a date on every item. The only real difference my proposal contained was **group ORDER**: the `Map` is populated
in hit order, so groups appear by best-hit rank rather than chronologically. That is a permutation of the same
bytes, and it is the entire remaining mechanism — far smaller than the spec implied.

(Worth noting for §2b of the standing rules: group order *is* controllable provider-side, because `Map` insertion
order follows the order we return hits. So the narrow version would have stayed on our side of the line. The
problem is not permission, it is that there is no evidence it would matter.)

## 2. Gold evidence is not under-represented either — measured

Second hypothesis, formed after the first failed: perhaps golds are *present* (F27: 97.2% all-golds) but
**thin** — a gold session contributing 1 hit while a distractor contributes 8, so each gold arrives as a fragment
too small to compose. Measured over the 25-hit budget, multi-session questions:

| | n | gold hits | non-gold hits | min hits on any gold | top-session share | distinct sessions |
|---|---|---|---|---|---|---|
| **PASS** | 107 | 11.2 | 13.8 | 3.99 | 0.20 | 8.9 |
| **FAIL** | 26 | **11.5** | 13.5 | 3.73 | 0.20 | 8.9 |

**Failures receive slightly MORE gold evidence than passes.** Every distributional statistic is effectively
identical. The `min hits on any gold` distributions overlap heavily (FAIL: 4×1, 4×2, 2×3, 1×4, 15×5 · PASS: 8×1,
12×2, 16×3, 8×4, 63×5), and the modal value in both is 5 — which also suggests a per-session hit cap is already
enforcing diversity.

Across all 500 the same pattern holds (PASS gold_hits 8.2 vs FAIL 8.8). **There is no crowding-out effect to fix.**

## 3. What is therefore excluded

For V1 multi-session, all three presentation-layer mechanisms are now ruled out **with data, not argument**:

| hypothesis | status | evidence |
|---|---|---|
| golds not retrieved | **excluded** | 98.5% all-golds recall on multi-session; 25 of 26 failures complete (F27 §4) |
| golds retrieved but under-represented in the hit budget | **excluded** | failures get *more* gold hits than passes; identical distribution (§2) |
| evidence not grouped / structure lost in rendering | **excluded** | the renderer already groups by session with per-item dates (§1) |

## 4. The conclusion this forces

The reader receives **complete, well-represented, session-grouped, date-labelled evidence** and still produces a
wrong answer on 25 multi-session questions. That is a **reasoning failure in the answer model**, not a memory-layer
failure. We do not own it on this benchmark, and we cannot fix it by presenting the same bytes differently.

Combined with F27, the honest V1 position is:

- retrieval: **saturated** (100% any-gold, 97.2% all-golds)
- evidence delivery: **adequate** (no crowding, already grouped)
- the residual 11.2 points: **the model's synthesis and abstention calibration**

**So V1 has very little headroom that belongs to us.** The remaining identified exception is abstention
calibration — 14 false abstentions against 20 correct ones — where a memory-layer *signal* (not a reordering)
could plausibly move the reader.

> **⚠ CORRECTION (same day, before this finding was acted on).** An earlier version of this paragraph said the
> M7/M7b lane "already probed" this with a NO-GO and that its failure "now reads differently". **That was wrong,
> and I had asserted the link from memory instead of checking F20.** M7/M7b are a **V2** experiment targeting
> **under**-abstention: their 128-question primary is questions whose gold answer *is* an abstention, where the
> control shows the reader **asserting** 115/128 = 89.8% of the time (F20 §; SPEC-M7B-CONDITIONAL §"targeted, NOT
> over-abstention"). F27's 14 questions are the **opposite** failure on the **opposite** benchmark: V1 answerable
> questions where the reader wrongly abstains.
>
> Two consequences. (1) M7b's NO-GO is **not evidence** about the V1 over-abstention target and must not be cited
> as such. (2) The V1 over-abstention target is therefore **entirely unprobed** — which is better news than the
> incorrect version implied: it is a small (14-question), untried target rather than a re-run of a failed lane.
> Any spec for it must build its own instrument; M7b's enriched slice is the wrong population.

## 5. Consequences for the program

- **SPEC-MULTISESSION-SYNTHESIS: WITHDRAWN.** Not run. No tokens spent on it.
- **#150's ≥450 target is now doubtful on its merits, not just its bar.** It would require winning 6 of 52
  complete-evidence failures with no identified mechanism. Recommend re-scoping #150 from "beat 444" to
  "document the V1 ceiling and its attribution", and moving the lane's effort to the tracks where the constraint
  is still ours: the scaling regime (#159/#161) and the V2 trajectory subsystem.
- **#154's survey question sharpens again:** the leaders' V1 advantage cannot be a retrieval or a grouping
  technique, because ours are already adequate. If OMEGA reaches 30/30 on preference and beats us overall, ask
  specifically what *signal* they put in front of the reader — not what they retrieve or how they order it.
- **This is the second time in one day that a favourable-looking framing of mine died on contact with its own
  artifact** (F27 §0 was the first) — and §4's corrected paragraph makes three, that one an unchecked cross-lane
  claim in my own prose rather than in a spec. All three were caught the same way — by checking the premise rather
  than the conclusion — and the cheapest two were caught before anything was spent. Hence the standing habit:
  **verify a stated cause in the source or the artifact before dispatching a spec, and before asserting a link
  between lanes.** Added to PROGRAM-ARCHITECTURE §6e.10, which the third instance shows applies to prose as well
  as to specs.

## 6. What this does not claim

It does not claim the model is unimprovable, that no presentation change could ever help, or that these findings
transfer to V2 or to the scaling regime — V2's trajectory subsystem and the 389× single-store test are different
instruments with different constraints, and F27/F28 speak only to **V1-small**. It also does not touch any banked
number: 444/500 and every per-category figure stand.
