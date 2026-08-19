# F35 — Gate-4 sanity slice: RED. The query fix unmasked an uncitable summary arm (16% fail-close); the scored questions improved far beyond noise — held unverified until the re-run.

**Date:** 2026-07-29 · **Run:** 100-qid slice, consolidated base 543e9ea, F32 pins verbatim (all verified pre/post,
stores 200/200 unchanged, codex 0.144.6 PATH-pinned). 291k harness-unit tokens.
**Artifacts:** `session-notes/2026-07-29/hermes-sanity-slice/artifacts/`.

## 1. The regression the slice exists to catch — mechanism proven, not guessed

**Fail-closes: 2 (F32, same qids) → 16.** All #164-signature. Root cause chain, confirmed by direct artifact
counts: the #168 sanitization fix made the product's internal **summary-arm FTS queries work for the first time**
on V1 — rows with ≥1 summary hit went **2/100 → 16/100** (5 → 29 summary hits). Every one of the 29 is
**uncitable** (`store_id` null; the #164a fix populated **zero**), and the fail-closed rows are **exactly** the
rows with a summary hit.
> **Mechanism correction (round-2 fix agent, verified against the stores):** the summary nodes are NOT
> nested/derived as first written — all 45 are depth-0, `source_type='messages'`, but with **`source_ids='[]'`**:
> the ingest path never recorded lineage. #164a filled zero for lack of lineage data, not nesting. The
> citability conclusion is unchanged; the repair path differs (there is nothing to walk on this corpus — the
> summary arm contributes leads, not evidence, until lineage is recorded at ingest).

One fix unmasked the other's incompleteness: an interaction bug that no per-fix test could see — the #164a
regression test rendered the OLD failure population (F32's 8 qids) clean while the NEW population
(lineage-empty summaries, woken by #168) is 100% uncitable.

**The dropped rows are not a weak tail** (baselines scored 8–10/16 on them) — this costs real points and blocks
the release. **Gate 4: RED.**

## 2. The fix (product, R2 train, dispatched)

Defect: the product delivers evidence hits it cannot cite. Invariant for the fix (mechanism = implementer's
choice): **in reference-strict delivery, no hit lacking a validated source reference is delivered; omitted
summary hits are backfilled by the next-ranked citable hit and counted in provenance meta.** Regression test =
the 16 qids' delivery re-rendered through the pinned validator with zero fail-closes AND hit-count still 25.
Message-sourced summaries may instead carry the full offset-path reference (the F33 oracle's proven shape).
Nested summaries have no truthful single-row citation — they are ranking signal, not citable evidence, and must
not be presented as such. (Harness-side drop-not-fail remains #165/upstream — not touched this train.)

## 3. The other surprise — held UNVERIFIED, promising, dangerous to believe early

On the 84 scored questions: **+17 net vs the banked arm (p=8e-05), +13 vs the same-code repeat (p=0.0044)** —
far outside the ±6 same-code band, and the raw 53/100 beats both baselines *despite* forfeiting 16 rows.
Plausible mechanism exists (the summary arm + sanitized FTS genuinely change V1-small retrieval — F32's
retrieval-identity result compared different codebases, not this delta), and the dropped-row interaction
muddies the pairing. **No claim is made.** The post-fix slice re-run gives the clean read; if the gain holds,
a full-500 confirm run is warranted before the release notes state any V1 number other than the banked 444.

## 4. Sequencing

Fix → cross-model review → merge to the train → **slice re-run** (same pins; gate: fail-close ≤ F32's 2 AND
paired flips within the ±6 band *or* a stable measured gain) → only then the mono-PR review rounds and upstream
push. The 95% bar does not move; the checklist's gate 4 stays red until the re-run is clean.
