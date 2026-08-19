# SPEC M7 — Negative-evidence disclosure via the harness's own dormant evidence gate

**Issue:** #157 · **Author:** orchestrator (Fable 5) · **Date:** 2026-07-25 · **Status:** frozen for pilot
**Depends on:** M7 (abstention mass) · M10 (search flailing) · M11 (low effort = operating point)

---

## 1. The discovery that reshapes this spec

M7 proposed inventing a negative-evidence channel. **We do not need to invent one — the harness already
has it, and it is switched off and untaught.** `memory_modules/codex.py` defines:

```python
VALID_EVIDENCE_STATUSES = {"directly_supported", "contradicts_premise", "near_match_only", "insufficient"}
ANSWER_POLICY_BY_EVIDENCE_STATUS = {
    "directly_supported": "answer_normally",
    "contradicts_premise": "state_premise_false",       # <- exactly the M7 false-premise class
    "near_match_only":     "say_exact_target_not_found",
    "insufficient":        "abstain_unknown",
}
```

`hermes_lcm_agentic.py` already threads `require_evidence_gate` through config → validation → metadata
(lines 294, 347, 368, 631, 849). Three facts establish that the mechanism is **dormant, not missing**:

1. `require_evidence_gate` is **absent from every run config we have shipped** → defaults to `False`
   (verified in L1/L2/L3 `memory_config.json`, and P4 inherits the same shape).
2. Neither backend's instruction text ever mentions `evidence_status`, so the agent never emits it
   voluntarily and `has_evidence_gate` is never true.
3. **Critically:** even when populated, `evidence_status` is written to per-question *metadata* only.
   `_build_memory_context_from_output` renders `memory_markdown`, span lines, and linked evidence —
   **never the evidence status.** The verdict is computed and dropped before the reader sees it.

Consequence: **enabling the gate alone changes accuracy by exactly zero.** It buys instrumentation.
The accuracy comes from rendering the verdict into `memory_context`. This spec does both.

## 2. Why this is §2b-compliant, and the line this spec will not cross

§2b forbids editing the fixed reader/answer prompt. This spec touches only (a) the curation contract
our own module hands its own sub-agent and (b) `memory_context` CONTENT — both squarely ours.

**Design decision (orchestrator, binding on the pilot): the rendered section is DECLARATIVE EVIDENCE,
never an IMPERATIVE to the reader.** We render *"the store contains no field matching X; searched
<terms>; the duration control appears in 4 states and is set by risk level alone"* — a factual absence
report. We do **not** render *"you must state the premise is false"*. The imperative form would be
reader instructions smuggled through the memory channel: self-inflating, and in my judgement
leaderboard-invalid in spirit even though it evades the letter of §2b. If the declarative arm
underdelivers, the imperative variant is **not** a fallback the program may take unilaterally — it goes
to the owner as an explicit question. Recorded so no later agent quietly crosses this line.

## 3. The change (three parts, one arm)

### 3a. Teach the gate in the curation contract
`hermes_lcm_agentic.QUESTION_INSTRUCTIONS` — extend the required output schema with the three gate
fields and explain the vocabulary. Also **repair the three lines that M7 identified as causing the
failure**:
- `"It may mention the likely answer when strongly supported"` — invites assertion. Must be paired with
  an explicit duty to report a contradicted premise.
- `"Put the most important evidence first and avoid redundant spans"` — invites dropping null results.
  Must except absence findings from the redundancy pressure.
- the only absence path today is total failure (`"If no useful evidence exists"`), which does not cover
  *"I searched for this specific field and it is not there"* — the actual M7 case.

New contract requirement: when the question presumes an entity/field/relation, the agent **must** run at
least one targeted search for it and record the outcome, including the queries tried and hit counts.

### 3b. Enable the gate
`require_evidence_gate: true` in `codex_params`. Validation is already implemented and tested
(`tests/test_hermes_lcm_agentic.py`) — this flips existing, exercised machinery on.

### 3c. Render the verdict into `memory_context` (**the part that moves accuracy**)
In `_build_memory_context_from_output`, prepend an `## Evidence Assessment` section when the gate fields
are present. Declarative form, e.g.:

```
## Evidence Assessment
The question presumes a second field controlling duration. Searched "duration control",
"second field", "risk level" across 12 states of 3 trajectories: the duration control appears in
4 states and is determined by risk level alone. No second field is present in the store.
```

Ordering: **first**, before Support Analysis. A weak 9B reader weights early context heavily (M1/M4),
and the failure being fixed is over-confidence formed from a tidy pack.

## 4. Failure-mode guard (the mirror-image risk — this is the real danger)

M7's own text says the static lane spent all night fighting **spurious unknowns** (reader abstains when
it should answer, 29% of answerable). This change pushes the agentic reader toward absence language and
can therefore **import the static lane's disease into the lane that does not have it.** Agentic
answerable accuracy is currently **77.1%** — higher than AgentRunbook-C's 74.9% overall. That number is
the program's best asset and this change is the most plausible way to destroy it.

Hence the gate below is **joint and two-sided**, and the answerable side is a hard floor, not a
tiebreak.

## 5. PREDECLARED GATE (frozen before the run — not revisable at one-short)

Pilot: the frozen 60q dev manifest (32 web / 28 enterprise), **low effort** per M11 §2h, decoding pinned
0.6/0.95/20, M9 parity-diffed against L3. L3 is the paired control (same manifest, same config, gate off).

| axis | measure | bar |
|---|---|---|
| **PRIMARY** | abstention-subset accuracy | **strictly up** vs L3 on the same subset |
| **FLOOR (hard)** | answerable-subset accuracy | **not down by more than 2.0 points** vs L3 |
| **SECONDARY** | overall accuracy | reported, not gating |
| **LEADING INDICATOR (M10)** | `searches_per_question` | instrumented and reported; expected flat-or-down |
| **INSTRUMENT** | provider-error rows (M11 §5) | counted on both arms before any comparison |

**One-primary law:** the primary is the abstention subset. Overall accuracy is *reported* and must not
be substituted for the primary if the primary disappoints.

**Achievability check (required at freeze, per the blind-R2 lesson):** confirm on L3's raw rows that the
abstention subset is non-empty and that its accuracy is below ceiling, so "strictly up" is attainable.
Record the L3 subset counts in the pilot report **before** the arm runs.

**GATE STRENGTHENING (recorded 2026-07-25, BEFORE the pilot returned any numbers — this is a
tightening, not a relaxation, and is logged as such per the never-relax-at-one-short rule):** the M11
power check shows a 60q slice cannot distinguish a 2-question move from noise (abstention subset n=17,
Fisher p=0.688 for a +2 swing). **The 60q pilot is therefore a SCREEN, not a promotion.** A pass
authorises a full-451 confirmation run; it does NOT itself bank the mechanism, and no result from it may
be published, submitted, or used to justify a mechanism story. A fail on the primary is still a fail.

Outcome rule: PASS both → promote to a full 451 run. Primary up, floor breached → **NO-GO**, and the
finding is that the mechanism trades answerable for abstention (still publishable, and it would be a
genuine result about read-time absence signalling). Primary flat → the channel is not the binding
constraint; do not fund a second variant without new evidence.

## 6. Instrumentation to add regardless of outcome
- `searches_per_question` as a first-class per-question metric (M10 asked for this; the sweep had to
  infer it). Source: agent trajectory/tool-call count, recorded in per-question metadata.
- gate-field emission rate (how often the agent actually populates the fields when asked).
- distribution over the four statuses, split by answerable vs abstention gold class.

## 7. Upstream note (do not act without owner sign-off)
That the harness computes `evidence_status` and then drops it before the reader is arguably an upstream
defect: the gate's whole purpose is to shape answering behaviour, and no backend renders it. This is a
credible, well-evidenced contribution to the benchmark repo **and it also affects the published Codex
baseline**, which runs with the same dormant gate. Do not open that PR during the autonomous window —
it is outward-facing. Park for owner review with the pilot numbers attached.
