# R3.0 architecture decisions — the three headline upstream-#436 findings (owner-granted authority; decided 2026-07-29)

*Triage provenance: the validated 24-comment triage (14 confirmed-real, 9 needing architecture decisions)
is archived at session-notes/2026-07-29/hermes-r3-0/artifacts/r3-panel-journal.jsonl. The remaining six
decisions are taken lazily per fix-batch (R3 plan amendment 8). Each decision below binds its fix's spec.*

## D-ARCH-1 · Event-dedupe semantics (requirements_compiler.py:1669 — the certified-wrong-answer defect)

**Defect.** `_finite_event_key` = unit+names; dedupe ignores each candidate's resolved date → two
same-entity events on different dates collapse to one, and `finite_coverage=true` certifies the wrong
count. The reviewer's fix (always include date) over-counts repeated *mentions* of one event.

**Decision: count what is date-distinguishable; certify only that.**
- Dedupe key = `(unit, names, resolved_date)` **when the candidate carries a resolved date**.
- Candidates WITHOUT a resolvable date keep the current `(unit, names)` collapse (conservative: undated
  mentions merge rather than inflate).
- `finite_coverage=true` is emitted ONLY when every counted event carries a resolved date; any undated
  contributor downgrades the response to uncertified coverage (count still returned, honesty preserved).
**Why.** The worst state is a wrong count carrying a certification; second-worst is an inflated count.
Certifying exactly the distinguishable set matches the program's citability philosophy: never certify
what the system cannot verify. Regression: two same-entity dated events → count 2 certified; dated+undated
mix → correct count, uncertified; repeated undated mentions → count 1.

## D-ARCH-2 · Adjacency-reserve composition (trajectory_store.py:3491 — 5 of 16 hits silently dropped)

**Defect.** `adjacent_reserve = min(6, limit//3)` is carved out of the nucleus, and when adjacency finds
nothing the reserve is never backfilled — `limit=16` returns 11 ranked hits. Default-on delivery path (V2).

**Decision: backfill unused reserve from the ranked candidate pool — full limit utilization always.**
- The reserve is a *priority carve-out*, not a quota: adjacency takes up to its reserve; whatever it
  leaves returns to ranked candidates in rank order.
- **Sequencing is part of the decision:** this changes V2 delivered composition → it does NOT land ad-hoc.
  It ships inside the single V2 RE-BASELINE paired batch (with the other delivery-path upstream fixes),
  measured before/after in one run; #166 (HTTP-524 retry) applies to BOTH arms of that batch (instrument
  change, never inside the treatment delta — plan amendment 5). Surface is trajectory-store (V2-only);
  a V1 sanity slice still runs at the batch boundary (amendment 4).

## D-ARCH-3 · Relative-date anchor trust boundary (reasoning.py:763 — self-consistent, unfalsifiable validation)

**Defect.** `lcm_compute` takes `session_date` from the caller's `raw_occurrence` and validates derived
dates against values derived from that same caller-supplied anchor — the check can never fail against
ground truth.

**Decision: the engine-owned occurrence sidecar is the sole trust root for anchors.**
- `resolve_occurrence_time` anchors from `engine._session_occurrence_dates` (the pattern the recall path
  already uses — tools.py:5035).
- A caller-supplied `session_date` is accepted only when it AGREES with the engine sidecar; disagreement →
  the sidecar wins and the response notes the override.
- Sidecar absent for that session → resolution proceeds but the result is marked low-trust (no certified
  temporal claim), mirroring D-ARCH-1's certify-only-what's-verifiable posture.
**Why.** Trust boundaries follow data ownership: the engine ingested the sessions; the caller is a
consumer. Self-anchored validation is the exact class the program retired in its own instruments (§6e.10:
verify a stated cause from the artifact, not from the claimant).

## Shared posture
All three decisions express one principle the R2 cycle validated seven rounds in a row: **fail honest —
deliver what is verifiable, certify nothing else, and never let a convenience default (collapse-all,
carve-and-drop, trust-the-caller) silently shape a certified answer.**
