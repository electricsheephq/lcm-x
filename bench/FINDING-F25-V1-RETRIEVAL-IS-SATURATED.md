# F25 — V1 retrieval is SATURATED (99.7% gold recall). The gap is the ANSWER LAYER, not retrieval.

> **Producer-script note (§6e.7, added 2026-07-29):** this finding's original inline script was never persisted —
> the number had to be re-derived from scratch by F27 (method: `recall_full500.py` in
> `session-notes/2026-07-25/hermes-preference-gap/artifacts/`, the surviving reference implementation).
> **Granularity caveat (F29):** all figures here are SESSION-level recall. Session-level saturation stands;
> it must not be read as evidence completeness — answer-turn recall is 85.6% (F29 §1).

**Date:** 2026-07-26 · **Source:** the cross-test's `retrieval-diff.json` — an artifact the executing agent
produced that I did not request. It answers the cross-test before the scores arrive.

---

## 1. The measurement

Both arms (#423 code and wave-1 code) over the same 100 V1 questions, same Voyage stores:

| | arm #423 | arm wave-1 |
|---|---|---|
| questions where ALL gold sessions retrieved | 98/100 | **98/100** |
| partial gold | 2 | 2 |
| **no** gold | **0** | **0** |
| mean gold-session recall | **0.9967** | **0.9967** |

Agreement between the two arms:

| | value |
|---|---|
| session-level Jaccard | mean **0.999**, median 1.000, min 0.917 |
| content-level Jaccard | mean **0.997**, median 1.000, min 0.852 |
| questions retrieving **identical** content | **97/100** |

## 2. Two conclusions

**(a) wave-1 does not change V1 retrieval.** 260 commits and ~27k lines of core code produce byte-identical
retrieved content on **97 of 100** questions. **Therefore any score difference the cross-test reports is
answerer noise, not capability** — and F24 already measured that noise at ±6 per 100. The cross-test is
effectively pre-answered: wave-1 cannot meaningfully move V1 accuracy, because its changes are not engaged on
this data path.

**(b) ★ V1 retrieval is SATURATED. The losses are answer-layer failures.**
Gold-session recall is **99.7%** and *zero* questions fail to retrieve any gold. Yet the banked score is
**88.8%**. So **~11% of questions have the correct evidence in hand and are still answered wrong.**

**The 22-question gap to OMEGA is not a retrieval gap.** We find the evidence. We fail to reason over it.

## 3. Why this explains everything we could not account for
- **Why #423 beats wave-1 on V1 but loses on V2:** #423's 6 commits are *answer-layer* work
  (`grounded answer-ready recall`, `lcm_compute`, finite-set coverage, fail-closed on untrusted claims).
  On V1 — where retrieval is solved — the answer layer is the only lever. On V2, where retrieval over
  trajectories is the hard part, that work does nothing and wave-1's retrieval machinery wins.
- **Why F23's losses concentrate in preference application and single-session recall.** Both are cases where
  the evidence is present and simple; failing them is a *reasoning/application* failure, not a search failure.
  Single-session especially: the evidence is in ONE session, so retrieval cannot be the problem.
- **Why the W3B static mechanism loop looked flat (M13).** Those were all *retrieval* mechanisms
  (diversity-cap, adjacency expansion, state-semantic quota, title-boost). On a saturated retrieval path there
  is nothing for them to win.

## 4. Consequences for the roadmap
1. **Stop investing in V1 retrieval.** It is at 99.7%. There is no headroom, and any mechanism that improves
   V1 retrieval improves nothing.
2. **The V1 lane (#150, re-aimed at ≥466) is an ANSWER-LAYER project.** The work is: given correct evidence,
   why does the answerer get it wrong 11% of the time? That is where all 22 questions live.
3. **This is the question to ask of OMEGA (#154).** They score 95.4% with `bge-small-en-v1.5` — a *weaker*
   embedding model than our Voyage 1024-dim. **They cannot be out-retrieving us.** Whatever they do is in the
   answer path: how evidence is presented, what is asserted, how preferences are applied. That is now a
   precisely-aimed question rather than a general survey.
4. **It does NOT generalise to V2 or to scale.** V1's stores are ~51 sessions / 514 messages — tiny. Retrieval
   is saturated *because the corpus is small* (M18/M19). At 389x (Phase 1A) or on V2 trajectories, retrieval is
   very likely the binding constraint again. **Saturation is a property of this corpus size, not of our
   retriever in general.**

## 5. Process note
This came from an artifact the subagent produced unprompted (`retrieval-diff.json`), like F24's repeat arm.
**Fourth time in two days that a subagent's initiative corrected or pre-empted the orchestrator's design.**
My dispatch asked for scores and a paired McNemar; the agent measured whether the arms even retrieve differently,
which turned out to be the question that mattered. **Dispatch packets should ask "and check whether the
mechanism is engaged at all" as a standing item — a score comparison between two arms that behave identically
is a waste of an experiment.**
