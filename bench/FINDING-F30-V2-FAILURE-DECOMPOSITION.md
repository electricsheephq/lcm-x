# F30 — V2's 74 answerable-wrong decomposed: the completeness→accuracy gradient GENERALIZES; V2's own leverage is elsewhere

**Date:** 2026-07-29 · **Spend:** zero LLM — pure re-analysis of the banked H6-P4 run artifacts (298/451, tag
`bench-H6-P4-298`), executed by a commissioned deep-analysis agent under the F29 granularity rules; validated by
independent coverage checks (not partition identities) and a from-scratch reimplementation of the official
scorer reproducing the recorded score on 226/226 phraseset questions.
**Artifacts:** `/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-v2-decomposition/artifacts/` (scripts
01–09 + JSON outputs; the 74-question ledger is `08_final_74_ledger.json`).

---

## 1. V2 carries NO gold-evidence annotation — the granularity ceiling is declared up front (§6e.11)

`questions.jsonl` is question-level only; trajectory states have no `has_answer`/gold flag; haystacks are
candidate sets, not gold. **F29's dataset-native answer-turn method has no V2 counterpart.** The finest
supportable level is **gold CONTENT-PHRASE presence** (the official scorer's own atomic phrases) inside the
**delivered `memory_context`** — i.e. "was the answer text delivered", not "was the right state delivered".
Coverage is honest, not padded: **171/323 answerable questions measurable at Tier A** (literal phrasesets),
+55 at Tier B (MC option text — a weak proxy), **97 unmeasurable** (derived counts/booleans, gotcha prose).
Two probe-discovered corrections: computed golds ("six", "true") can never be literal — including them fakes
non-monotonicity; and prompt-echoed phrases are reported as a sensitivity arm.

## 2. ★ The result that matters: the gradient reproduces

| Tier A (n=171) | evidence complete | partial | none |
|---|---|---|---|
| accuracy | **85.8%** | **50.0%** | **0.0%** |

V1 (F29): 92.4 / 74.0 / 52.6. **Same monotone shape, different benchmark, different reader (qwen3.5-9b vs
gpt-5.6-sol), different memory subsystem (trajectory vs message store), steeper under the weak reader.**
Evidence completeness at delivery is now a cross-instrument predictor of accuracy — the closest thing the
program has to a law. It is also exactly what the #154 survey found the verified leaders optimizing (token-budget
fill; no-gate compressed delivery).

## 3. The 74, as a mutually-exclusive ledger

| bucket | n | notes |
|---|---|---|
| unmeasurable at any tier | 28 | derived/gotcha classes — reasoning-heavy by construction |
| **reader-bound** (every gold phrase delivered, still wrong) | **19** | 9 wrong value, 6 dropped set-items, rest format |
| evidence-addressable, Tier B only (weak proxy) | 10 | |
| **evidence-addressable, Tier A** | **8** | |
| instrument: empty reader output (HTTP 524, scored 0) | 6 | see §5 |
| instrument: scorer number-form miss | 3 | "two" vs "2" — shared by all leaderboard systems |

**Counterfactual, stated with its scope:** Tier A measured = **~1.5 pts**; extrapolated to all answerables
2.8–4.2 pts, with the explicit caveat that the 97 unmeasurables skew reasoning-heavy so the truth likely sits
at or below the low end. **On V2, reader-bound outweighs evidence-addressable ~19:8 at the credible tier** —
the mirror image of V1's post-F29 picture, consistent with V2's fixed weak reader being the binding constraint
(which is the tier's design intent).

**Negative finding banked:** co-location is SATURATED on V2 — all 71 multi-fact "complete" questions already
have every gold phrase inside ONE delivered block. V1's token-budget/turn-completeness lever (#25) should not
be expected to move V2-small materially. Do not port it there on vibes; the gate for any V2 variant must be
pre-declared separately.

## 4. Abstention: the V1 pathology does NOT reproduce

V2's fixed prompt mandates `\boxed{UNKNOWN}`, and a tolerant matcher finds **delta = 0** vs the harness's exact
detector across all 451. Only 2 of the 74 are declines; **72 are confident wrong answers**. (Of the 79
abstention-question losses, 73 are the reader *answering* a false-premise question — reader-bound, not ours.)

## 5. Instrument findings — reported separately from capability, per standing rule

1. **7/451 questions (1.55 pts; 6 inside the 74) were never scored on a real reader response.** Provider chain
   returned HTTP 524; `evaluation/harness.py:967-985` catches, sets `response_raw=""`, scores 0, **no retry**.
   Proof: `prompt_tokens=0`, no response_id, literal 524 lines in the run logs; NOT context-length (the 7 span
   1.2k–99.6k ctx tokens). Honest reporting: headline stays **298/451** (one-primary law; no retroactive
   rescoring); the instrument-clean denominator is **298/444 = 67.1%**. QIDs in the ledger.
   → harness fix (retry-then-fail-loud) filed on the tracker; any upstream report to the V2 maintainers is
   OWNER-GATED (outward-facing).
2. **3 scorer number-form misses** (digit vs number-word; 0.67 pts) — official-scorer behaviour shared by every
   leaderboard system; bounds answer-form loss, licenses nothing.
3. 14/451 responses lack `\boxed{}`; the extractor silently treats the whole response as the answer.

## 6. What F30 changes

- **The program's central claim gets stronger and sharper:** delivery completeness predicts accuracy across
  both benchmarks. The #25 lane (token-budget fill, V1) now has a cross-instrument motivation — and a
  pre-registered boundary: **V2-small is NOT its target** (co-location saturated; reader-bound dominant there).
- **V2-small headroom decomposes as:** ~1.5–4 pts evidence-addressable (modest) · ~1.5 pts instrument (the 524
  bug — recoverable by a harness fix + one re-run of 7 questions) · the bulk reader-bound, which on this tier
  is the benchmark's point, not our failure.
- **The cheapest V2 upgrade identified, if ever needed:** a one-off zero-LLM scan of `trajectories.jsonl`
  locating literal gold phrases per state (~1.2 GB single pass) would upgrade Tier A from "answer text
  delivered" to "right state delivered". Parked — not on the critical path.
