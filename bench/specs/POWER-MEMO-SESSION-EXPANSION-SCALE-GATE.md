# POWER MEMO — session-expansion scale gate (pre-run artifact, instrument-1 discipline)

Companion to SPEC-SESSION-EXPANSION-SCALE-GATE.md (frozen). Computed BEFORE any gate run,
per the spec's own requirement and red-team amendment 3. Sources are named per §6e.7.

## 1. The denominator, verified (the F38 lesson)

Baseline arm = A3 (expansion OFF), gate171-v6 run artifacts (`query-A3-<rung>.jsonl`),
head 2289589 — the code that MERGED as PR #184 (9bbd0d5). Per rung, of 150 probe rows:
18 carry no labeled answer turns (`answer_turn_delivered_complete = null`) and are excluded;
**132 scored questions** form every denominator below.

| Rung (sessions) | complete | still-incomplete (b-pool) | completeness |
|---:|---:|---:|---:|
| 500 | 72 | 60 | 0.545 |
| 2,000 | 51 | 81 | 0.386 |
| 8,000 | 27 | 105 | 0.205 |
| 19,829 | 18 | 114 | 0.136 |

These reproduce F39 §4's published baseline exactly (72/132 = 0.545 …), as expected:
recall parity across the #171 saga was net 0 at every rung, so delivered sets — and
therefore completeness — are unchanged on merged main.

## 2. Floor check (spec: STOP if b-pool < 20 at top rungs)

b-pool at 8k = **105**, at 19,829 = **114** — both >5× the ≥20 floor. The probe set is
valid as pinned; no re-derivation needed. **The run may proceed.**

## 3. What PASS requires, in mechanism terms

PASS = paired net (b−c) ≥ 8 per top rung. Against these pools that is a net conversion of
**≥ 7.6% (8/105) at 8k and ≥ 7.0% (8/114) at 19,829** of currently-incomplete deliveries.
Mechanism coverage context (F33, measured): 92% of missing gold answer turns live in
sessions the delivery already touches — exactly the population `session_expand_v1`'s
windows target. The V1-small oracle arithmetic required ~40% conversion to matter; the
scale gate's bar is 5× lower. Power is therefore NOT the binding risk for b.

## 4. The c-side and cost risks (what could still kill honestly)

- **c (complete→incomplete regressions):** expansion only ADDS payload, so c > 0 arises
  only through budget displacement (an added window pushing a previously-delivered gold
  turn out). The spec's median-token cap (PASS requires ≤1.6×; KILL >2.2×) bounds the
  displacement pressure; expansion telemetry (containment drops, strict rejections) is
  captured per the spec to attribute any c > 0.
- **Cost clause:** the primary is offline delivery analysis (zero LLM). The gate can
  therefore fail ONLY on its own merits, not on spend.

## 5. Run preconditions (all satisfied as of 2026-07-30)

1. Instrument turn-join fallback fixed properly (adb9cec — union.jsonl-derived, loud-fail,
   F39-shape regression test; the F39 run's private workaround is retired).
2. Stores: use the POST-MIGRATION re-pinned sha manifests (F44 §3 — the F43 triggers'
   one-time `lcm_resident_*` migration is the declared store state).
3. Treatment arm: `session_expand_v1` flag through the REAL bridge (#173 merged dormant;
   env-flag reachability proven during the Stage-2 fix round).
4. Tools discipline: storefreeze manifests pre/post, pinverify on probe lists (immutable,
   drift fails closed), failclose signatures on every metric row.

## 6. Declaration

No gate bar in this memo modifies the frozen spec. If the run lands GRAY (net 3–7 at top
rungs), this memo travels with the owner decision per the spec.
