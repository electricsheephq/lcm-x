# F32 — The release number: wave-1 ties the banked 444 within the noise floor. The release base is SETTLED: wave-1.

**Date:** 2026-07-29 · **Run:** full-500 LongMemEval-V1, wave-1 (`bench/w3b-on-wave1` @ e99f342) under the exact
banked-parity pins (harness 2c20cee, voyage stores sha-verified 1000/1000 before AND after, judge effort LOW,
`evidence_cards_v1`, 1.67M tokens — under both the banked run's 1.70M and the 2M ceiling).
**Executed under protocol §6 early-release:** u-arm latency void; u-arm recall CONTENDED; this run's own latency
columns are CONTENDED and are not cited. Accuracy is the primary.
**Artifacts:** `session-notes/2026-07-26/hermes-v1-full500-wave1/artifacts/` (analysis JSON, manifest, RAW run,
retrieval-identity, store-integrity). Adjudicated by the architect; the executing agent reported numbers only.

---

## 1. Headline and the paired read (the only read that counts)

| | correct | accuracy |
|---|---|---|
| wave-1 raw (8 harness fail-closed rows scored 0) | 436/500 | 87.2% |
| **wave-1 adjusted (fail-closed dropped, both-arm convention stated)** | **436/492** | **88.6%** |
| banked #423 (A) | 444/500 | 88.8% |
| 07-24 rerun (A′ — see caveat) | 442/500 | 88.4% |

| McNemar | b | c | discordant | net | p |
|---|---|---|---|---|---|
| wave-1 vs banked, adjusted n=492 | 6 | 8 | **14** | −2 | **0.791** |
| A′ vs A reference pair | 8 | 10 | 18 | −2 | 0.815 |

**Verdict: a TIE.** The adjusted comparison shows *fewer* discordants (14) than the A-vs-A′ reference pair (18)
with the identical net (−2). **Caveat carried (audit, F29 §4):** A′ is not a pure placebo — it was a different
harness build (reader-contract rewrite + deterministic-ops path), so 18-flips/net−2 is a *conservative upper*
reference, noise plus a build delta. wave-1 sitting below even that contaminated bound makes the tie claim
stronger, not weaker.

**Why it ties, mechanically:** retrieval identity extends F25/F26 to the full 500 — **481/500 questions returned
byte-identical retrieved content** (455 identical in order; Jaccard 0.996). On V1's message-store path the two
codebases are the same retriever. The V1 score was never going to move; now that is measured, not extrapolated.

## 2. The release-base decision (mine to make; the data is now sufficient)

**wave-1 (`e99f342`) is the R2 release base.** It ties V1 within the noise floor (this run), wins V2 outright
(298/451 vs #423's poor V2 showing), carries the instrument-integrity fixes, and is the unified codebase.
Corollary, stated plainly: **#423's six answer-layer commits add no measurable V1 value** — wave-1 without them
ties the run that had them. Porting them onto wave-1 is unification hygiene at best; it leaves the capability
roadmap entirely (it was already deprioritised; this retires it).

**Ship-as-measured rule (AMENDED 2026-07-29 by the owner's Phase-1B-first decision + publish-authority
grant):** the released artifact is now the CONSOLIDATED base (e99f342 + the #169 ceiling fixes + the #434
cherry-pick + the #164a store_id fix), and "as measured" transfers to it: Phase 1B re-run + a paired V1
sanity slice validate the consolidated base itself before upstream push (release-kit/CONSOLIDATION-CHECKLIST).
The rule's intent — released code == validated code — is preserved; the validation set moved with the train.
Original text: R2 releases commit `e99f342` exactly as measured. The #164(a) `store_id` product fix
(which would prevent the 8 fail-closed rows in future runs) lands in the NEXT train alongside the Phase 1B
fixes — releasing measured-code-plus-one-fix would make the released artifact differ from the measured one, and
we just spent a week learning what silent instrument/code deltas cost.

## 3. The fail-closed rows quantify #165

**8/500 questions (1.6%)** lost to the renderer fail-close (`evidence-card item has no validated exact source
reference`), all with the identical signature, zero other failures. The harness was not patched mid-run (parity
held). This is the measured blast radius for #165's argument: the instrument silently converts a provider's hit
*shape* into a capability zero at ~1.6% incidence on this provider. Updated on the issue.

## 4. The excerpt-size addendum returned a STRUCTURAL NULL — and corrects my premise

Both arms carry the **same 2,400-char per-hit cap**; the delivered distribution is *identical to the decimal*
(median 340, mean 760, p90 2,400, 10.9% of hits at cap; avgContextTokens 3,095 vs 3,102). My pre-run premise
("banked ≈300-char excerpts vs wave-1's 2,400 cap") was wrong — ~300 was the *median hit length under the same
cap* in both arms. **The run could not test the excerpt-size lever because the lever was never engaged.** This
is a null by construction, not evidence against #25 — and the useful numbers survive — with a structural correction from the
spec red-team (same day): delivery is **two-tier**. Only the top-8 hydrated hits carry the 2,400 cap
(10.9% of THOSE at cap); ranks 9–25 are capped at `_LCM_RECALL_SNIPPET_CHARS = 300`, and **55.7% of them sit
exactly at that cap**. "Median 340 = 14% of budget" conflated the tiers — the enriched-slice-rate error, 4th
instance. The #25 lane proceeds per SPEC v2 (oracle pilot → session expansion), not utilisation-fill.

## 5. Two parity catches by the executing agent (credit where due; both now standing practice)

1. **The machine's `codex` CLI had upgraded 0.144.6 → 0.145.0 after the banked run.** The agent pinned the
   banked version by PATH (sha-matched binary) rather than running the new one — the same instrument-change
   class as the judge-effort catch. **Rule: the transport version is part of the parity pin set from now on**
   (added to the M9 checklist).
2. Store integrity verified after as well as before (frozen source 1000/1000 unchanged; wave-1 writes during
   search, so the private-copy discipline is load-bearing, and it held).

## 6. What R2 still waits on

Nothing but the owner's framing choice (R2 draft §3). All three data inputs are in: the release number (this),
the scaling curve (F31), and the audit-corrected evidence trail (F27–F31). The abstention-split definitional
gap the agent flagged (16/40 vs 14/42 — boundary answers that say "I don't know" then volunteer information) is
noted in the analysis JSON and does not touch any claim R2 makes.
