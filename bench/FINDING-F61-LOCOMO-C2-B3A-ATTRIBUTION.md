# FINDING F61 — LoCoMo C2: sender attribution at ingest (B3-A) pairs at +13 points; adversarial robustness doubles its band

Date: 2026-08-26. Registration: `bench/specs/RUN-SHEET-LOCOMO-C2-B3A.md` (#316).
Configuration under test: F48 declared base + `HERMES_MB_SPEAKER_PREFIX=1` (the single B3-A
delta: `"<Speaker>: "` prefixes on user rows at ingest — the 45-line bridge change).
Product tree: wt-locomo-product (pinned; unaffected by main-side boundaries per the
BASELINE-LEDGER rule). Run root: session-notes 2026-08-21/hermes-locomo-c2/artifacts/
paid-aa-20260820T211813Z. Recompute record: 2026-08-26/interim-review/artifacts/F61-recompute.txt.

## 1. The paired row (1,986 questions × 2 arms, recomputed from raw checkpoint verdicts)

| metric | arm A | arm A′ | F48 declared | band |
|---|---|---|---|---|
| aggregate | **67.42%** (1339/1986) | **67.72%** (1345/1986) | 54.6% | — |
| adversarial | **62.33%** | **62.11%** | 32.7% | PASS ≥36.7 |
| multi-hop | 71.65% | 74.77% | 64.8% | no-loss |
| temporal | 46.88% | 44.79% | 41.7% | no-loss |
| world-knowledge | 78.24% | 77.65% | 69.7% | no-loss |
| single-hop | 45.39% | 46.81% | 37.2% | no-loss |

**VERDICT: PASS, both arms** — every category above F48 declared; adversarial +29.6/+29.4 over
baseline (band was ≥36.7 = +4; measured ≈ double the band). A/A′: aggregate spread **0.30pt**,
82/1986 questions discordant (**4.13%**), fail-closed accounting **0 incomplete** both arms.
Aggregates recomputed from per-question `checkpoint.questions[*].phases.evaluate.label` — never
the run's own summary (which matches: 67.42/67.72).

## 2. Reading

Attribution is the lever, robustness is the prize. Prefixing the SPEAKER onto ingested user rows
attacks the F59 §9 failure class directly (entity+recency-matched confusions; cross-epoch
mis-binds): the adversarial (unanswerable) category — where the pre-B3-A system hallucinated
answers from misattributed facts — nearly doubles. Every other category also rises, consistent
with attribution improving retrieval binding generally, not just abstention.

## 3. Provenance + disclosures (append-only)

- Attribution receipt: 211/211 user rows `"Caroline: "`-prefixed in the live store during early
  ingest (fail-closed sensor, armed pre-spend).
- Arm A completed 2026-08-21 and was postrun-signed then. **A→A′ gap ≈ 110h** (weaker A/A′
  pairing than C1's same-day pair — disclosed; the 0.30pt spread and 4.13% discordance are the
  measured pairing quality).
- A′ interruption chain, all checkpoint-recovered with 0 lost questions: (1) 2026-08-26 ~02:30
  session teardown killed the runner at search-phase start (ingest+indexing complete, 1.19M
  episodes); relaunched 08:36 via the pinned resume script (checkpoint skipped completed
  phases); (2) the resume script died after "Run complete!" before its postrun steps (bridge
  child processes orphaned — #236 class); postrun executed manually same day: store+dataset
  storefreeze+verify clean, failclose 0, paired failclose clean (union-drop: 0 dropped qids).
- PINS-POSTRUN: **PASS** — with the disclosure that the verifying shell's env was reconstructed
  from the resume script's own exports (the script died before running postrun itself); sha-class
  pins verified against disk; env pins verify the reconstruction, not the original process.
- Left-only correct 38 / right-only correct 44 (paired failclose record) — the discordance is
  two-sided, not drift in one direction.

## 4. What this does NOT prove

Claim class: benchmark result on the pinned product tree with the registered harness. It is not
a claim about current main (which has since shipped v0.23.x privacy changes — see the
BASELINE-LEDGER privacy-trio boundary), nor a production-integration claim: B3-A lives in the
benchmark bridge; the product ingest-path analog (#324 sender/timestamp provenance + #317
untrusted-evidence boundary) is the productization this result unlocks, and IT needs its own
confirmation run.

## 5. Next

Productize B3-A via #324/#317 (v0.24.0 flagship candidate); scoreboard row rides this finding's
PR; the E+R− pool publishes with the row per the registration.
