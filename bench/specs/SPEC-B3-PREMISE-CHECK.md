# SPEC B3-PC — premise-check for the LoCoMo adversarial lever (zero-spend first pass)

Status: REGISTERED before execution (gates pre-declared below). Owner steer of record: B3 is
the roadmapped product fix for the adversarial category (F48 §3/§5; declared config 45.5→32.7,
measured-honest under the canonical abstention gold + premise-rejection-only rubric).

## 1. The premise being checked
B3's product hypothesis (from F46/F48 fault analysis): on adversarial questions the system
RETRIEVES the relevant fact but the question's false premise (typically a speaker/entity
misattribution) is ACCEPTED by the reader — i.e. the failure is reader-side credulity with the
contradicting evidence already in context, not retrieval-side absence. The roadmapped fix
(contradiction surfacing in memory_context, extending the M7b Evidence Assessment seam) is only
correct if this premise holds. This spec measures it before any product code is written.

## 2. Data (all on disk; mechanical derivation only)
- Run: F48 declared-config arm (A) per-question artifacts + delivered-hits accounting
  (paths cited in FINDING-F48 §4.5 and the singlehop-decomp session-notes dir).
- Universe: all adversarial-category questions judged WRONG in the declared-config arm.
  n_adv and the miss list are derived at execution time from locomo10.json categories +
  per-question results (never hand-typed; the F46→F48 label scramble is the standing warning).
- Gold contradicting evidence: LoCoMo adversarial rows carry the true binding (correct
  speaker/fact). Map gold evidence turn-ids → delivered-hits per question.

## 3. Classification (first pass = zero LLM spend)
For each adversarial miss, classify by turn-id set intersection:
- **E+R−** (evidence delivered, premise accepted): ≥1 gold-evidence turn id present in the
  delivered context, answer still accepts the false premise.
- **E−** (retrieval absence): no gold-evidence turn id delivered.
- **GOLD?**: rows among the 99 documented corrupted-gold rows → excluded, counted separately.
- **UNCLEAR**: turn-id matching inconclusive (e.g. evidence spread across paraphrased turns).
  If UNCLEAR > 20% of misses, a bounded second pass (LLM-assisted, ≤n_adv calls, flagged as
  spend, judge-family ≠ answer-family) may be run — but only after the first-pass numbers are
  banked and logged.

## 4. Pre-declared gates (decided NOW, before any counting)
- **≥60% of classifiable misses are E+R−** → premise HOLDS: proceed to design the reader-side
  lever (contradiction/conflict status rendered in the memory_context Evidence Assessment block,
  M7b seam; presence-with-conflict is a NEW status class — M7b renders absence statuses only).
  The lever gets its own registration + paired gate before any scored claim.
- **≥60% E−** → premise FAILS: the lever is retrieval-side (entity-binding query expansion /
  a premise-verification retrieval pass), and the contradiction-rendering design is SHELVED
  with the premise-failure documented.
- **Neither ≥60% (mixed)** → both levers sized by their measured share; the larger share leads
  the next train, the smaller is banked as a follow-on. No third option gets invented after
  seeing the split.
- Fixed-denominator rule: percentages are over classifiable misses (excluding GOLD? rows);
  the exclusion count is disclosed alongside.

## 5. Deliverables
1. FINDING-B3-PC doc (per-row classification table in artifacts, counts + gate verdict in the
   doc), session-notes artifacts dir + citation from #191.
2. RUN-LOG entry (zero-spend analysis runs are logged like scored runs — instrument, inputs,
   outputs, verdict).
3. If premise HOLDS: a design sketch for the conflict-status rendering (seam, status taxonomy,
   token budget impact) as an appendix — design only, no implementation in this unit.

## 6. Execution routing
Zero-spend pass: fast-worker (Sonnet) with this spec + the F48 artifact paths; the FTS-inertness
investigation (running in parallel) shares the same delivered-hits accounting artifacts — reuse
its extraction if it lands first. Verdict adjudication (gate application) is mechanical
(threshold arithmetic) but the FINDING doc is orchestrator-authored.
