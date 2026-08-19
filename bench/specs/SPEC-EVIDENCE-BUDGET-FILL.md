# SPEC v1 — Evidence budget-fill (SUPERSEDED — red-team verdict: REVISE, 92% confidence, five voiding defects)

> ## ⛔ v1 KILLED BY THE PRE-FREEZE RED-TEAM, 2026-07-29 — see SPEC-EVIDENCE-ORACLE-THEN-EXPANSION.md (v2)
> The adversarial pass this spec required found, with source/data reproduction
> (`session-notes/2026-07-29/hermes-spec-redteam/artifacts/`):
> 1. **The mechanism reaches 0% of its pool** — no missing answer turn is truncation-caused; 92% are whole
>    turns absent from delivery while their session IS delivered. The lever is session EXPANSION (v1's
>    demoted "variant B"), not turn completion.
> 2. **v1's own mechanism check was a guaranteed false negative** (the coverage metric cannot move under the
>    v1 treatment, by construction of `covered()`).
> 3. **The c≤5 floor sits under measured noise** (null pairs show c=8/c=10) — 81% chance of a false
>    "more context hurts" conclusion.
> 4. **The treatment would fail-close ~68% of hits** via the evidence-cards `chunk_span` validation (#165's
>    mechanism at full blast) unless `content_offset`/`content_returned_chars` are emitted on every hit.
> 5. **The "14% of budget" premise was the enriched-slice-rate error again** (4th documented instance): the
>    2,400 cap covers only the top-8 hydrated hits; ranks 9–25 are capped at `_LCM_RECALL_SNIPPET_CHARS=300`
>    with 55.7% at cap. Plus cost-model self-contradiction (1.41× real vs 4× claimed) and a near-unwinnable
>    gate (needs b≥15 at c=5 against a flippable ceiling of ~22–35).
> **Kept unedited below as the record. v2 adopts the red-team's design: oracle pilot first.**

## (v1 original) SPEC — Evidence budget-fill with turn-complete inclusion (V1 lane #150 / task #25)

**Status:** DRAFT-COMPLETE 2026-07-29, authored with all premises verified. **Freezes only after one
adversarial red-team pass** (standing audit cadence; two of the last three specs died of unverified premises —
this one's premises are each tagged with their source below). No run before freeze.

---

## 1. Premises — each verified, none from memory

| premise | source |
|---|---|
| Accuracy is monotone in answer-turn evidence completeness: 92.4% / 74.0% / 52.6% (complete/partial/none) | F29 §1, reproduced personally from `has_answer` flags |
| 69 of 479 flagged questions are missing ≥1 answer turn in the delivered hits; 19 have none | F29 §1 (410/50/19 split) |
| Delivered content runs at **14% of the existing per-hit budget** (median 340 chars vs 2,400 cap; 10.9% of hits at cap; avgContextTokens ≈3,100) | F32 §4, measured on both arms of the release run |
| The provider owns hit count, per-hit caps, and slicing (`LIMIT`, `_LCM_RECALL_ANSWER_READY_CONTENT_CHARS`, `_slice_loaded_content`) | verified in `wt-w3b-on-wave1/tools.py` (task #25 metadata, §6e.10 check) |
| The verified leaders fill token budgets or deliver whole compressed logs; none win by ordering/formatting | #154 survey (Hindsight TEMPR verified; Mastra OM verified-code) |
| Grouping/ordering/crowding levers are dead; within-session completeness was never tested | F28 (survived audit, reinforced), F29 §4 |
| This lever is NOT expected to move V2-small (co-location saturated there; reader-bound dominant) | F30 §3 — boundary pre-registered |
| Base and control both exist: wave-1 `e99f342` under the F32 pins, control = the F32 run itself | F32 §2 (base settled; ship-as-measured) |

## 2. Mechanism under test

**Treatment (`budget_fill_v1`, default OFF, provider-side only):** replace the fixed 25-hit / rank-sliced
delivery with a **token-budget fill**: rank candidates exactly as today; walk the ranking; for each selected
hit, deliver the **complete turn** (never a mid-turn slice); continue until a **total evidence budget** is
reached. Budget set to ≈12k context tokens (≈4× today's ~3.1k; still ~2.5× below Mastra OM's 30k; far inside
the reader's window). No new content sources, no summarisation, no LLM in the delivery path, no ranking change,
no reader-prompt contact (§2b). The harness renders whatever we deliver, grouped by session as today (F28).

**Why this exact shape:** it is the smallest change that directly attacks the measured defect (answer turns
truncated or absent from delivery) while keeping ranking — the part F25/F26/F32 proved identical and adequate —
untouched. Attribution stays clean: same hits selected first, more of them delivered whole.

## 3. Instrument

- **Arms:** control = the F32 run (already banked, same base, same pins — no re-spend). Treatment = one
  full-500 run of `e99f342 + budget_fill_v1 ON`, under the F32 pin set verbatim (harness 2c20cee, voyage store
  private copy + sha 1000/1000 before/after, judge LOW, codex-cli **0.144.6 PATH-pinned** per §6e.13).
- **Question set: all 500.** No enriched slice (§6e.8).
- **M9 parity diff** before reading: the only permitted delta is the budget_fill flag.
- **Primary metric (capability): correct/500 paired vs control**, with the fail-closed convention: report raw
  AND adjusted (drop fail-closed rows from BOTH arms; expect ~8 per arm from #165's measured rate).
- **Mechanism check (diagnostic, not gated): answer-turn completeness** recomputed on the treatment run
  (`extract_answer_turns.py` method). The treatment should move completeness 85.6% → ≥95%; if completeness does
  NOT move, the mechanism never engaged and the accuracy result is void either way (the F32 structural-null
  lesson — check the lever engaged before reading the outcome).
- **Cost:** ~1.7M tokens (one arm; control is free). Context tokens rise ~4× per question; answer-phase cost
  scales with input — budget ceiling for the run: **2.5M tokens, hard stop.**

## 4. GATE — declared before any run; never relaxed

Let b = control-wrong→treatment-right, c = control-right→treatment-wrong, on the adjusted (both-arm
fail-closed-dropped) pairing.

**PASS requires all three:**
1. **b − c ≥ 8** — must clear the measured full-500 reference spread (A/A′: 18 flips, net −2, itself an upper
   bound per F32 §1's caveat). Net +8 on n≈492 is outside anything two same-code runs have shown.
2. **McNemar exact p < 0.05.**
3. **Floor: c ≤ 5.** More context can distract as well as inform; breaking more than 5 currently-correct
   questions is a NO-GO regardless of b.

**Also reported, ungated:** per-category b/c; completeness delta (mechanism check); token/latency cost per
question (the LAFS price of the lever — a +8 that costs 4× tokens is a *product configuration*, not a default).

## 5. Outcomes, written before any result

- **PASS** → the first measured accuracy win attributable to the memory layer on V1. Promote behind a config
  default-ON for benchmark/product profiles that can afford the tokens; re-baseline; R2 note or R3 headline
  (owner's call on timing).
- **NULL with completeness moved** (≥95% delivered, score flat) → the strongest possible confirmation that
  V1-small's residual gap is the reader's, achieved by *closing* the delivery gap rather than asserting it.
  F29's attribution then stands on measurement with the lever exhausted. Lane closes with a real answer.
- **NULL with completeness NOT moved** → implementation defect or wrong mechanism locus; diagnose before any
  variant; do not iterate blind.
- **FLOOR BREACH (c > 5)** → more evidence measurably hurts this reader at this budget; that is a finding about
  context dilution worth its own writeup; do not retry with a bigger budget.

## 6. Boundaries

Not run on V2-small (F30 §3). Not a cap raise (the cap is not the binding constraint — utilisation is). No
claim about production Voyage recall (different embedder; this is an answer-delivery experiment, not a
retrieval experiment — ranking is held fixed by design). Scaling behaviour (#167/#168, Phase 1B) is a separate
lane and is not gated on this.
