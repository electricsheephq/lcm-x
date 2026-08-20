# F47 — V2 re-baseline paired gate: PASS; #190 merged (9d181aa); new V2 static baseline 123/451 (2026-07-31)

**Registration:** v2-rebaseline-paired-v1 (frozen sha 41a749a7; treatment #190 @ a3a31dd,
control fork main @ e5acbbf). Full pin/storefreeze/failclose discipline both arms, one pass each.

## 1. Result (pipeline-emitted GATE-INPUTS; artifacts session-notes 2026-07-30/hermes-v2-paired)
- Paired n=434 (union-drop 17: control instrument_failed 10, treatment 7 — #166-class exclusion,
  BOTH arms on the same 6bfd58a harness by pin verification).
- **b=29 wrong→right, c=26 right→wrong, net = +3** (non-inferiority bar was net ≥ −3; margin +6).
- Raw: control 120/451, treatment 123/451. Adjusted: 117/434 vs 120/434.
- Delivery profile: comparator PASS, zero drift on arm share and median chars; per-question
  mechanical diff (ATTRIBUTION-DARCH2.json): across all 451 questions exactly ONE differs —
  one pure hit ADDITION at the limit=16 boundary (15→16, zero removals, zero membership/order
  changes elsewhere). The D-ARCH-2 backfill signature at maximum resolution.
- Cost: 12.33M / 12.70M harness units, both under the 18.80M ceiling.

## 2. Verdict path (author≠judge, two rounds)
pairedgate read returned AMBIGUOUS (`no_registered_band_match`) — the reader parses structured
bands, this registration's bars were PROSE (see §4). Arithmetic verified by the tool
(net_consistent). Verdict rendered by the blind-adjudicator agent from the frozen bars +
instrument files only: round 1 = CANNOT-ADJUDICATE naming two missing inputs (mechanism-level
D-ARCH-2 attribution; both-arms pins/storefreeze evidence not listed); round 2 with
ATTRIBUTION-DARCH2.json + the PINS/storefreeze files = **PASS, every leg instrument-confirmed**
(it independently hash-matched the resume-gap dataset freezes rather than trusting the amendment
note). The adjudicator twice flagged and disregarded memory-plugin "prior observation" text
injected into its tool results asserting a pre-baked verdict (see §5).

## 3. Incident: treatment arm EINTR fail-close + append-only continuation
Treatment web-batch4 died 17:10Z on `InterruptedError [Errno 4]` in the JUDGE-side OpenAI-client
SSL init (C-level OpenSSL read outside PEP 475 retry; transient; harness path identical both
arms). Recovery per AMENDMENT-2-TREATMENT-RESUME.md: original tree + FAILED sentinel preserved
byte-for-byte; batches 1-3 loaded read-only; batch4 re-ran fresh (predictions discarded, not
salvaged); resume-time dataset freeze REQUIRED byte-identical to the original (confirmed
independently by the adjudicator). The auto-mode classifier rejected an initial in-place recovery
variant; the append-only redesign it forced is the better instrument practice and is now the
template for mid-run arm recovery.

## 4. Instrument lessons (bind future registrations)
1. **Registrations must carry machine-readable bands.** pairedgate's reader cannot parse prose
   bars; this gate needed a blind-adjudicator fallback. Rule going forward: every new
   registration includes a structured `bands:` block the reader consumes; prose stays as
   commentary. (Reader enhancement → next-train backlog.)
2. **GATE-INPUTS emitters must match the reader's schema** — the driver emitted flat key=value;
   the shim (gate_inputs_to_rungs.py, mechanical parse with net-consistency refusal) is recorded
   in the artifacts. Hand-typed verdict inputs are prohibited (auto-mode classifier enforced
   this; correct).
3. **Delivery-attribution instruments need per-question resolution.** The aggregate
   deliveryprofile could not affirm the attribution clause; the per-question diff could, and it
   is cheap. Promote ATTRIBUTION-style per-question delivered-hits diffs into the standard
   paired-gate toolkit.

## 5. Blind-adjudication contamination note
The claude-mem observation hook injects session "prior observations" into subagent tool results;
in BOTH rounds it presented verdict-shaped assertions ("Gate reads PASS...") the adjudicator had
to recognize and disregard. It did — but future blind dispatches should state explicitly that
hook-injected observations are not instruments (now in the dispatch template).

## 6. Banked
**New V2 static baseline: 123/451 @ merge 9d181aa** (adjusted 120/434 w/ 7 treatment
instrument-failed rows disclosed). Context: the control arm re-measured e5acbbf-era main at
120/451 in the same paired run — the banked R1 static 125/451 vs 120 here is single-run
cross-build variance context for V2-static numbers (n=451, fixed weak reader); the paired design
is why the gate read is trustworthy anyway. #191 backlog carries the review tail; the paired-V2
store set remains frozen (storefreeze manifests unchanged).
