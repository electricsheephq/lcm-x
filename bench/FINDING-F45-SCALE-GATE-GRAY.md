# F45 — Session-expansion scale gate: GRAY by net, cost clause catastrophically exceeded (2026-07-30)

**Spec:** SPEC-SESSION-EXPANSION-SCALE-GATE.md (frozen; power memo attached per its own terms).
**Run:** execution 4 of the pre-registered run (three prior executions fail-closed correctly on
integrity: trigger-migration drift ×2 [declared per F44 §3 precedent], FTS integrity-check
timestamp writes [root-caused by the orchestrator; instrument now pins
`LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS=-1`, docs cce92c1]). Product source = main @ 9bbd0d5 +
the #173 patch (synthetic tree, declared; the port PR #188 makes it real). Stores sha-verified
v3 pins pre/post every arm; 0 fail-close rows; 0 LLM calls. Arithmetic inputs:
`session-notes/2026-07-30/hermes-scale-gate-run/artifacts/GATE-INPUTS.txt` (sha 65d97ef4…71d59b).

## 1. The measured table (132 scored questions per rung, paired OFF→ON)

| rung | b (incomplete→complete) | c (complete→incomplete) | net | median tok OFF | median tok ON | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 18 | 0 | +18 | 1,751.5 | 10,006.5 | 5.71× |
| 2,000 | 12 | 0 | +12 | 1,741.5 | 9,856.5 | 5.66× |
| 8,000 | 3 | 0 | +3 | 1,741.0 | 10,699.5 | 6.15× |
| 19,829 | 6 | 0 | +6 | 1,717.5 | 10,513.5 | 6.12× |

Engagement was total: ~3.5–3.9k windows added per arm; **444–450 of 450 selected sessions were
expanded at every rung** — the mechanism expands unconditionally.

## 2. Verdict, read strictly against the frozen bars

- PASS requires net ≥ 8 at BOTH top rungs AND token ratio ≤ 1.6×: **not met** (nets 3 and 6;
  ratios ~6×).
- KILL requires net ≤ 2 at both top rungs: **not met** (3, 6). The >2.2× token KILL clause is
  written against "any passing rung"; no rung passed, so it does not literally fire.
- **GRAY (net 3–7 at top rungs): MET. Per the spec, GRAY = owner decision with the power memo
  attached.** No bar is relaxed, no re-run without a fresh registration.

## 3. Architect's reading for the owner (recommendation, not a verdict)

**Recommend treating this as KILL-AS-CONFIGURED for the scale thesis.** The net lands in the
GRAY band, but the cost side is 3–4× beyond even the KILL line at every rung: ~8.3k extra
delivered tokens per question buys 3–6 completions per 132 at scale. Two facts make the
diagnosis precise rather than fatal:
1. **The mechanism is SAFE but UNTARGETED.** c = 0 everywhere — expansion never displaced a
   complete delivery (containment logic holds; the registered c-risk did not materialize). It
   simply pays the full expansion cost on every question, gold-bearing or not.
2. **The b-pool it can reach shrinks with scale.** Net falls 18 → 12 → 3 → 6 because at 389×
   the still-missing turns increasingly live in sessions the ranked tier never delivers at all
   — F33's "92% of missing turns are in delivered sessions" was measured at V1-small and DOES
   NOT TRANSFER to the scale regime. Session expansion amplifies what retrieval already found;
   at scale, completeness loss is increasingly a RETRIEVAL-REACH problem, not an amplification
   problem.

**Salvage path (would need its own registration):** a TARGETED variant — expand only when a
completeness heuristic fires (e.g., a delivered session shows answer-turn-adjacent truncation),
with a per-question token budget. The +18/500 and +12/2000 nets show the mechanism works where
its premise holds; the cost model, not the mechanism, is what failed. Whether that lane is worth
its spend against the R3 thesis (metric-standard + scale curve) is the owner's call.

## 4. Method notes
Three integrity fail-closes preceded the scored run, each caught by the tools the program built
(storefreeze/pinverify/fail-close) — the discipline paid for itself twice over tonight (§6e.16
lived; §6e.18 was written from the sibling PR the same night). Deviation declared by the lane:
the shared docs checkout advanced cce92c1→0089707 (PROGRAM-ARCHITECTURE.md only) during the
final OFF arm; instrument/tools/specs/stores byte-identical throughout.

## 5. DISPOSITION (2026-07-30 morning — decided under expanded product-owner authority)

The owner granted standing decision authority ("run decisions through the vision; ≥95% confidence →
proceed; backtrack later if needed", 2026-07-30). Applying it: **the GRAY is dispositioned as
KILL-AS-CONFIGURED.** Vision test: the program's positioning is the agent ENHANCER — fast, cheap memory
access at scales where alternatives fail (VISION-AND-ATTRIBUTION §1c: our two measured advantages are
delivery completeness and latency/cost at scale). An unconditional 5.7–6.1× delivered-token multiplier
purchasing 3–6 completions per 132 questions at the top rungs contradicts that positioning under any
reading; the cost clause landed ~4× beyond even the KILL line. Confidence: well above the 95% bar.

Consequences:
1. `session_expand_v1` stays MERGED-DORMANT (default off) — it is the measured artifact and the
   substrate for any successor; the port to main (#188) proceeds so main equals the measured tree.
2. The TARGETED variant (expand only on completeness-heuristic fire, per-question budget) is filed as a
   next-train issue with its own future registration — explicitly NOT funded for R3.1.
3. The sharper strategic read stands: at 389×, completeness loss is increasingly retrieval-REACH, not
   amplification. The next completeness lever at scale is retrieval reach — which is where the R3
   metric-standard + scale-curve thesis already points. No roadmap change required.
4. Reversibility: this is a configuration/funding decision, not a code deletion — reversal costs one
   registration and one run.
