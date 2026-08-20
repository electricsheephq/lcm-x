# FINDING F58 — the "fusion artifacts" are mostly the production FTS arm; the reference fts arm's darkness hid it

Date: 2026-08-20. Zero-spend continuation of F55/F57. Artifacts:
`session-notes/2026-08-20/v1m-dump-forensics/` (fusion-artifact-decomposition.txt +
artifacts/probe/e01b8e2f.payload.json).

## 1. Decomposition of the 31 "lcm_recall-specific" rank-1 artifacts (from F55 §2)
Against the six reference arms' dumps: 24/31 have gold-above-wrong in ≥2 arms with
wrong-above in none; **12/31 have the wrong rank-1 session in ZERO reference arms' top-10s**.
The obvious hypothesis for the 12 — the scope/recency prior — was tested and **falsified**:
0/12 wrong sessions sit in the most-recent 20% of their haystacks (normalized positions
0.17–0.52).

## 2. Single-question probe: the mechanism, named
A probe of q `e01b8e2f` through the REAL production path (env-gated payload dump on a
scratch tree, reverted after capture; the instrument's own fail-closed count check rejected
the filtered run as designed) shows rank-1 is `kind: message_excerpt` — a **production FTS
arm hit** — a literal-but-wrong match ("stayed at the Hyatt Regency hotel", wrong session)
beating the gold (rank 3, same phrasing class absent) by adjacent RRF reciprocals
(0.01639 vs 0.01587).

**Why forensics couldn't see it:** the production `lcm_recall` runs a LIVE internal FTS arm,
while the instrument's REFERENCE `fts` arm is dark in the declared config (the F49
prose-flag class; 0.7% r@1 in F53) — so production-FTS-sourced winners show zero presence
across every reference arm. The 12 zero-presence artifacts are, by this mechanism,
**solo-FTS literal matches outranking semantic golds** at the fusion.

## 3. Corrections/caveats to the standing record (append-only)
- F55 §2 / F57: the "lcm_recall-specific (pure fusion artifact)" blame class CONFLATES
  fusion arbitration effects with production-vs-reference arm divergence — at least the 12
  zero-presence cases are production-FTS sourced, not fusion-order bugs. The 43 "systemic" /
  23 "shared" classifications are unaffected (they matched on reference-arm content).
- General instrument lesson: reference arms that are dark in the declared config create
  blind spots in EVERY dump-based blame analysis. The instrument-level fix worth
  considering: dump the production path's per-hit `kind` alongside session ids (the
  candidates sidecar currently keeps ids only).

## 4. Lever direction (design next; registration before any product change)
FTS-vs-vector arbitration at fusion: a solo-FTS rank-1 (no vector-arm corroboration in the
window) yields to the best vector-corroborated candidate. Bounded, zero-provider, directly
targets the 12 (+ plausibly part of the 24). This intersects the parked V1-M FTS-ON
variant decision (post-C1): the same mechanism that makes the reference fts arm dark also
governs what the production FTS arm matches — the two decisions should be designed together.

## 5. Disclosures
(1) Zero spend (probe was cache-served; one question). (2) The probe used two uncommitted
env-gated patches on a scratch worktree, reverted after capture; production code untouched.
(3) One question probed; the 12-case generalization rests on the kind/score mechanism plus
the zero-presence + non-recency evidence of §1 — per-case payload confirmation would ride
the instrument change in §3 if pursued.
