# F29 — Independent audit REFUTES the V1 closeout: session-level recall is not evidence completeness. V1 reopens as a memory-layer lane.

**Date:** 2026-07-29 · **Instrument:** 13-agent adversarial audit (owner-directed fresh-eyes review) — strategy
red-team, two independent refuters on F27/F28, Phase 1A instrument review, dispatch parity, completeness critic;
material findings re-verified by independent agents (two of the audit's own refutations were killed as false
positives, which is why the survivors are credible). 1.39M subagent tokens.
**Personally reproduced by the architect before acting** (audit_evidence_in_prompt.py, dataset-native
`has_answer` flags — LongMemEval's own per-turn annotations, present in the local corpus).
**Audit artifacts:** `/Volumes/LEXAR/Codex/session-notes/2026-07-28/hermes-program-audit/artifacts/`

---

## 1. THE CENTRAL REFUTATION — my granularity error

F27 measured whether a gold **session id** appeared among the 25 logged hits and I published it as *"52 of the
56 failures had the COMPLETE gold evidence in the prompt."* That inference was wrong. Every logged hit is a
~300-char `message_excerpt`; LongMemEval flags the actual answer-bearing **turns**. Joining at that level
(479 questions have ≥1 flagged answer turn):

| metric | session level (F27) | **answer-turn level (correct)** |
|---|---|---|
| all gold retrieved | 465/479 = 97.1% | **410/479 = 85.6%** |
| any gold retrieved | ~100% | 460/479 = 96.0% |
| questions with ZERO answer-turn evidence | 0 | **19** |

| evidence in the 25 hits | n | accuracy |
|---|---|---|
| all answer turns | 410 | **92.4%** |
| partial | 50 | **74.0%** |
| none | 19 | **52.6%** |

Accuracy falls **monotonically with evidence completeness** — the relationship my session-level analysis was
structurally blind to. The failure decomposition rewrites:

- "52 of 56 failures complete" → **31 complete / 13 partial / 9 none** (of 53 failures with flagged turns)
- multi-session "25 of 26 had everything" → **14 complete / 10 partial-or-none** — ~40% of our worst category's
  failures were missing answer-bearing evidence, not failing to reason over it
- "retrieval accounts for ≤4 questions (0.8 pts)" → counterfactual at the evidence level ≈ **16.8 questions ≈
  3.4 pts** of the 11.2-pt gap (50×(.924−.740) + 19×(.924−.526))

**Retrieval is NOT saturated on V1. It is saturated at session granularity and ~86% at evidence granularity,
and the gap between those two numbers is memory-layer work.**

## 2. The second refutation — my published bound contradicted my own leaderboard doc

VISION §1b claimed *"no memory system can win those 38 questions by retrieving or presenting better; a large
competitor margin implies a stronger reader."* F22 — written a day later and never reconciled — records **OMEGA
at 466/500 and Mastra ≈474/500 on the same benchmark with GPT-4.1 readers**, older/weaker than our
`gpt-5.6-sol`. Systems 22–30 points above us on weaker readers are the direct counterexample: the margin is
evidence delivery, not reader strength. The bound is withdrawn. (This also re-legitimises #154's survey.)

## 3. Number-level caveats (audit findings, accepted)

- The **97.2% all-golds** figure carries minute-resolution collision contamination in the timestamp join (the
  3-gold row's 83.7% is a pure artifact of it). Session-level saturation stands, but cite it as "~97% session
  level, exact-to-join-resolution", and the per-gold-count table is retired.
- "any-gold 500/500 = 100.00%" → defensible range **498–500** (the script applied opposite collision
  conventions to its two headline numbers).
- The 14/42 abstention split is a **floor**: the regex only catches leading "I don't know" forms; multi-session
  has ≥3 abstentions, not 1.
- The F27 reconciliation "check" (486×89.3% + 14×71.4% = 444) was an arithmetic identity of the partition, not
  an independent check. It validated coverage, nothing else.
- F28 §2's "failures receive MORE gold evidence than passes" is an **unnormalized-rate artifact** — divide by
  gold-session count and the direction reverses. F28 §3's "golds not retrieved: excluded" row inherits the
  granularity error and is struck.

## 4. What SURVIVES the audit — verified, some reinforced

- **F28's core conclusion survives, reinforced.** Both audit attempts to refute it died on verification: the
  "sibling run fix" was an uncontrolled cross-commit comparison (contract rewrite + a new deterministic-ops
  path; net **442/500, a regression**), and its "6 fixed" questions **also flip on F26's pure placebo re-run**.
  There is still no reliable, ownable presentation lever at the grouping/ordering/crowding level. What F28
  never tested — **within-session completeness** — is exactly where §1 says the real lever is.
- **F27 §0** (the enriched-slice correction) — untouched.
- **The latency headline** (−56.3 s/q, p<0.0001) — untouched (noted: single-scale, 60q; Phase 1A extends it).
- **The scaling pivot** — every audit lane, including the red-team, says keep Phase 1A as priority.
- **Dispatch parity for the release run** — clean; the launcher encodes every pin correctly.
- **F24's noise floor** — clean (its pair was the same-code repeat arm, not the 07-24 build; verified).

## 5. Phase 1A instrument amendments (accepted BEFORE adjudication — see protocol §3g–§3i)

1. **B×S0 is non-functional** (scope dirs never materialised; all 150 queries return 0 hits, `rg` exit 2).
   Excluded from adjudication. The B curve keeps its four ladder rungs.
2. **Cross-arm cap unfairness (verified):** A-arms apply LIMIT=25 to raw sub-session hits *before* session
   dedup (~9 distinct sessions scored), while B is scored on session-level results — session-recall comparisons
   are biased **in B's favour**. Adjudication reports the bias direction with every A-vs-B recall comparison,
   or recomputes on a matched distinct-session basis.
3. **Persona-collapse confound (design caveat):** merging 500 personas into one store makes persona-keyed
   first-person questions partially ill-posed for ANY memory system; recall degradation at scale is therefore
   over-determined for those question types. Latency claims are unaffected. Quantify rival-persona collisions
   before attributing any recall drop to the index.

## 6. Program changes

- **#150 re-scope recommendation RETRACTED** (posted to the tracker). The lane is neither "beat 444 via
  preference" nor "document the ceiling" — it is **answer-turn evidence completeness**: ~3.4 pts addressable,
  mechanism candidates = session-expansion rendering (small-to-big; avg delivered context is only 3,102 tokens
  against an effectively unlimited budget) and answer-turn-coverage-aware selection. Needs a spec with a
  pre-declared gate; the withdrawn spec's gate design (paired, full category, floor, one-primary) is the
  template. **Answer-turn recall replaces session recall as the standard V1 retrieval metric.**
- **V2 parity task:** the 298/451 headline has 74 answerable-wrong questions (48% of its losses) that have
  never been decomposed. Zero-spend, message-level method now exists. Queued.
- **Reader-config falsifiers** (re-run the 56 failures at high effort / independent judge, ~112 calls): queued
  behind the release run; cheap; either result sharpens the attribution.
- **New rules:** §6e.11 — *a claim's granularity must match its metric's granularity* (session presence ≠
  evidence completeness ≠ answer-turn coverage; name the level in the sentence). §6e.12 — *before publishing a
  strategic claim, reconcile it against the full findings index* (the F22 contradiction sat in-tree for three
  days).

## 7. Honest accounting

The audit found in ~15 minutes of wall-clock what I missed across three days of self-review, including three
self-corrections that all stopped one level short of the real error. The author-≠-reviewer principle is now
measured fact in this program, not policy. Cost: 1.39M subagent tokens; yield: two load-bearing refutations,
two false-positive kills that *strengthened* a finding, three instrument bugs caught before the data they
would have corrupted was read, and a revived V1 lane worth ~3.4 points.
