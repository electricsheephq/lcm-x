# F46 — LoCoMo arm-A (47%) decomposed: instrument bugs + a config-class retrieval gap; capability not indicted (2026-07-30)

**Provenance:** arm A of the pre-registered A/A′ noise-floor pair (1,986 questions, full pin discipline,
MemScore 47%, per-category 49.2/43.8/29.1/52.9/45.5). Two-stage analysis: a 12-agent read-only
decomposition workflow (wf_8ec9895c; report `hermes-locomo-deepdive/artifacts/LOCOMO-ARM-A-DEEPDIVE.md`)
plus a zero-LLM retrieval-replay root-cause run (`RETRIEVAL-GAP-ROOTCAUSE.md`, sha 7051208b…). Owner
hypothesis going in: ~80% instrument/format bug. Verdict: partly — and the biggest driver is CONFIG.

## 1. Confirmed instrument bugs (fixed in memorybench PR #3, both-arms by construction)
1. **blip_caption never ingested** — 55/1,986 gold answers exist only in image captions; 42 scored wrong.
2. **Adapter truncation lying** — 5.7% of delivered results hard-cut to 300 chars mid-sentence with
   `content_truncated=false`; cap lifted to the product's 2,400-char answer-ready bound, metadata truthful.
3. **Abstention judge rubric** — zero-credited correct false-premise corrections (bounded, ~2–20 rows).
4. **groundTruth="undefined"** on adversarial rows (scoring-inert; hygiene). Plus stale retry errors.
Quantified ceiling from these alone: ~+3 points. The owner's "long answers penalized" hypothesis was
checked and is NOT supported (0/24 sampled judge explanations cite length).

## 2. The dominant driver: candidate-generation recall, CONFIG-CLASS (retrieval replay, 25-row fixed sample)
- 22/25 (88%) `ingested_never_ranked`: gold verbatim in SQLite, absent from the diagnostic UNCAPPED
  top-50. The raw natural-language FTS arm finds **0/41** missing gold turns in its top-200; the
  configured fastembed (bge-small) chunk arm finds 7/41. Failure is BEFORE fusion, before any reader.
- 2/25 ranked just below the top-25 cap (ranks 26, 36); 1/25 truncated-out (the 300-char bug).
- 0 ingestion losses, 0 query-rewrite losses; replays reproduced arm A's ordered results exactly.
**Reading:** the FTS zero is the known conjunctive-AND death on conversational prose queries — the
precise class #183 (fts_prose_mode) was built to fix — and arm A ran the R2-era product build
(543e9ea, pre-#183) with prose mode nonexistent, a small embedder, and a top-25 cap. This is the same
lesson LongMemEval taught (F31 §3); LoCoMo's 47% vs V1's 91% divergence is pipeline configuration,
not a capability contradiction.

## 3. What IS genuine (not config): adversarial attribution
243/446 adversarial rows wrong with the correct fact retrieved 78.6% of the time — the model
misattributes the speaker instead of flagging the false premise. Real answer-layer weakness; product
work, not instrument work.

## 4. Disposition (product-owner authority)
1. Arm A′ CANCELLED mid-ingest (documented in the run dir) — the noise-floor pair re-runs on the fixed
   harness with a DECLARED run config; continuing on the buggy harness spent paid phases on a
   measurement that no longer fed a decision.
2. The scored re-run config is an ENGINEERING declaration, disclosed in full lineage (this finding):
   current product main (includes #183), fts_prose_mode enabled for the conversational adapter,
   embedder choice re-evaluated, cap per the product's answer-ready contract. Config chosen after
   diagnosis is legitimate system-building; the disclosure is what keeps it honest. No score
   projections until measured (§6e.8 discipline — the 47–50% band from the deep dive assumed the gap
   might be capability; F46 §2 reclassifies it config-class, ceiling unknown until run).
3. LongMemEval interference: NONE — all fixes are LoCoMo-lane harness/adapter surfaces; zero product
   code changed; our LongMemEval numbers flow through a different harness entirely.
4. Portfolio question (LoCoMo-Plus + the community critique of LoCoMo/LongMemEval flaws) under
   research (wf_8d68b003); the scored-run decision follows that + the fixed-harness A/A′.

## 5. Portfolio disposition (research workflow wf_8d68b003, decided 2026-07-30 under product-owner authority)
- **LoCoMo-Plus: REJECTED** on the program's instrument bar — its contribution (defeating string-match
  bias) is inseparable from closed vendor judges (validated only against Gemini-2.5-Flash/GPT-4o, no
  self-hostable fallback preserving its 0.81 human agreement); no license; zero adoption; repo stale.
  τ²-Bench keeps the R4 slot (one-instrument-at-a-time). Reopen only on open license + self-hostable
  judge + an open slot.
- **LoCoMo sidecar: CONTINUES on the fixed harness** (memorybench PR #3) with the SEVEN-POINT DISCLOSURE
  required on any published number: fix-state/commit; judge identity + full prompt (the community audit
  measured the STOCK judge accepting 62.81% of intentionally-wrong-vague answers — ours errs strict);
  corrupted-gold exposure (99/1,540 documented golden-answer errors → ~93.6% ceiling; cross-reference
  vs our rows IN FLIGHT); retrieval/embedder config named; per-category breakdown; run count/variance;
  the A/A′ noise floor alongside the point estimate.
- **Citation discipline:** cite primary evidence only (snap-research/locomo#27, EverMind-AI/EverOS#73,
  mem0ai/mem0#3944) — the Reddit thread carries a vendor COI and a confirmed citation error. Our own
  audit artifacts PRE-DATE the critique fetch (dated); the external material corroborates, it is not
  the trigger. The critique is ammo FOR the metric-standard thesis, never an excuse for the score:
  the two big drivers (config-class retrieval gap, genuine adversarial attribution) are real signal.

## 6. Corrupted-gold cross-reference (measured, CORRUPTED-GOLD-XREF.md)
All 99 audit-documented corrupt questions ran in arm A (unique join by conversation + exact question
text). We were scored WRONG on 80 of them; in the deterministic 15-row failed sample, **6/15 of our
answers match the audit-CORRECTED answer** — i.e., right against bad gold. Extrapolated (sampling
caveats in the artifact): ~1.5–2 points of the 47% headline is corrupted-gold penalty. Known-corruption
ceiling for our 1,986-row run: 95.02% (the audit does not cover the 446 adversarial rows). These
denominators are now part of the seven-point disclosure (§5).

## 7. §2's mechanism hypothesis FALSIFIED by measurement (CONFIG-FIX-REPLAY.md, 2026-07-30)
The fixed config (product main w/ #183 prose mode ON + harness hydration fix, same embedder) recovers
**0/22** candidate-generation rows at top-25 AND top-50 (1/38 missing turns at top-50). The "#183-class
conjunctive-FTS death" reading of §2 was wrong for this population. The measured residual decomposes the
gap further: **24/38 missing turns ARE generated by an arm but dropped by RRF fusion from the fused
top-50** (fusion/ranking-class), and **13/38 are never generated by either arm within top-200**
(embedder-class, bge-small). The truncation-loss row DID recover (harness fix works). Next levers, in
evidence order: fusion analysis on the 24 (pre-fusion ranks exist in the replay data), stronger-embedder
replay on the 13. Still no capability indictment — the reader never saw these turns under any tested config.

## 8. Fusion + embedder diagnosis (FUSION-EMBEDDER-DIAGNOSIS.md, 2026-07-30)
400 fusion policies simulated offline on the recorded candidate lists; 144 preserved all control-row
gold. Selected safe policy (global quota FTS:chunk = 1:2): **recovers 15/24 buried turns at top-25**
(best RRF variant 14-15). Embedder upgrade (bge-large, 1.2GB, authorized download): **0/13 recovered —
because all 13 never-generated turns have ZERO ELIGIBLE CHUNKS** in the chunk index. New mechanism:
INGESTION CHUNK-ELIGIBILITY (those turns never entered the semantic arm at all — threshold/role/format
exclusion suspected, not embedding quality). Final ledger of the 38: 15 fusion-recoverable ·
13 chunk-eligibility class · ~10 neither-under-safe-fusion. Track B levers: (1) the 1:2 quota as
declared run config, (2) chunk-eligibility diagnosis → possible product/config fix, before the
declared-config re-run.

## 9. FINAL MECHANISM LEDGER (CHUNK-ELIGIBILITY-DIAGNOSIS.md closes the decomposition, 2026-07-30)
The 13 zero-eligible-chunk turns are ALL excluded by the default <40-token conversational threshold
(chunking.py) — 12 are substantive gold turns (coverage DEFECT for short-chat content shapes), 1 a
generic acknowledgment (by-design). Complete ledger of the 38 originally-missing gold turns:
**15 fusion-quota recoverable (FTS:chunk 1:2, control-safe) · 12 chunk-threshold defects · 1 by-design ·
~10 residual (measure in the re-run)**. DECLARED RE-RUN CONFIG (Track B): fixed harness (memorybench
PR #3) + fusion quota 1:2 + lowered/configurable chunk min-token threshold for conversational stores +
frontier answering model (Tier-F) — every knob named here with its measured justification. LoCoMo has
now done exactly what the owner asked of it: found real faults (threshold, fusion weighting, truncation,
captions, judge rubric) that our own harness never surfaced.

## CORRECTION (2026-08-02): the §intro per-category tuple's labels were scrambled
"49.2/43.8/29.1/52.9/45.5" was published as single/multi/temporal/world/adversarial; ground-truth
category mapping gives **multi-hop 49.2, temporal 43.8, single-hop 29.1** (world/adversarial were
correct). Propagated into F48 §3 before being caught by a premise-checking decomposition agent
(see F48 §3-CORRECTION). All F46 mechanism analyses operated on qid-level data and are unaffected;
only this summary tuple's labeling was wrong.
