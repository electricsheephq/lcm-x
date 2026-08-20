# FINDING F52 — AMA miss-pool decomposition: two concrete memory levers (zero-spend)

Date: 2026-08-03. Feeds the ≥72.26% AMA roadmap (F51 gap −24.9pt). Method: mechanistic
classification of the two biggest miss pools; per-question retrieved context IS retained
verbatim in the run artifacts (`answers_*.jsonl.reasoning_trace` = the literal
`memory_retrieve()` output — traced to `memory_interface.py:155,188`), so evidence-presence
classification is direct, not inferred. Disclosed limits: `results_*.json` retains only scalar
scores (no judge rationale), so rubric-fit (M2) can only be inferred from answer-vs-gold diffs.

## 1. alfworld (284 misses, FULL count; 60-sample cross-check agreed exactly)
| bucket | n | % |
|---|---|---|
| M1 retrieval-absence | 204 | **71.8%** |
| M3 reader-reasoning | 80 | 28.2% |
| M2 rubric-fit | 0 | 0% (agent self-corrected an initial 6 — the "similar text" rows differ on exactly the fact under test; ALFWorld observations are ~90% shared boilerplate, so textual similarity ≠ judge nitpick) |

**Mechanism (L1):** ALFWorld trajectories are dense with near-duplicate templated text ("You
arrive at X… available actions…"). Semantic/keyword top-k cannot disambiguate the specific
turning-point steps that aggregation questions require (gold cites steps 5/6/17; retrieval
surfaces topically-similar visits from other steps). Generic recall metrics look fine while
this fails — the lever's gate must measure retrieval recall on aggregation/multi-step queries
over templated trajectories specifically.

## 2. swebench (243 misses, full count) — the OPPOSITE profile
M1 = 91 (37.4%), M3 = 152 (62.6%). Within the M-buckets a distinct mechanism, **verified by
the orchestrator against raw rows**: gold answers cite raw trajectory indices ("Step 77",
"Step 5"); predictions answer in a self-generated scheme ("Step 6.1 — Fix implementation",
"Phase 7.1 (verification)") — and the CONTENT of those answers describes the correct events.
**The memory/summarization layer re-chunks time and destroys original step indices** even when
the underlying fact survives. Lever (L2): step-index provenance must ride through memory
construction into rendered evidence; its gate is a step-alignment check between gold citations
and answer step-references, independent of content correctness.

## 3. Levers → next-train queue (each gets own registration + paired gate when funded)
- **L1 — step-anchored retrieval for templated trajectories** (alfworld-class; ceiling if
  perfected: +204 questions ≈ +8.2pt on the AMA row).
- **L2 — provenance-preserving step indices** (swebench-class; the Phase-vs-Step subset is a
  measurable slice of 152 M3 misses; ceiling estimate requires the alignment count — first
  L2 sub-task is that zero-spend count).
- Both are PRODUCT levers in hermes-lcm proper, benchmark-agnostic (templated-log retrieval
  and temporal provenance matter for real agent memory, not just AMA).

## 4. Artifacts
session-notes 2026-08-03 `ama-funnel-decomp/artifacts/` — full per-row classifications
(alfworld CORRECTED json is authoritative), extraction/classification scripts, 60-sample
readable dumps. Orchestrator verification: episode-141 rows read raw (Step-77→"Step 6.1",
Step-5→"Phase 7.1" confirmed verbatim).

## 5. CORRECTION (2026-08-21, append-only) — L2's "measurable slice of 152 M3" was a classifier-construction artifact; L2 PARKED

§2-§3 framed the swebench step-index phenomenon as "a measurable slice of 152 M3 misses."
That framing was wrong, and the error was in this finding's own classifier construction,
not in the underlying rows:

- F52's recall-grep check tested whether the *retrieved context* contained the gold
  answer's step index. The retrieved context for these episodes is a PHASE OUTLINE
  ("Step 6.1", "Phase 7.1") that can never lexically match a raw step index ("Step 77") —
  so the grep failed on ALL 77 step-indexed misses and the classifier auto-routed every
  one of them to M1 (retrieval-absence), leaving zero in M3 by construction. The "slice
  of M3" never existed as measured.
- Full 243-row re-classification from raw (session-notes 2026-08-21 `l2-subtask0/artifacts/
  alignment-count.json`, architect-verified against the recovered 2026-08-03 payload):
  **0 of 152 M3 golds cite a step index.** The flagship episode-141 example is confirmed
  verbatim but lives in M1 territory.
- Honest L2 ceilings against the AMA row's 1.54pt A/A′ spread: strict 1q (0.04%), broad
  18q (0.73%), absolute cap 77q (3.13% — including 2 rows where the model cited the RIGHT
  step and still scored 0). Under/near the noise floor → unmeasurable, unfundable.

**Disposition:** L2 (step-index provenance through summarization) is **PARKED** as an
independent lever. The step-index phenomenon folds into L1's retrieval-lever territory
(one lever family, not two): if step-anchored retrieval (L1) delivers the right
trajectory steps, the provenance question collapses into presentation. §3's L1 lever and
its queue position are unaffected.

Lesson (program-level, repeated from F48 §3-CORRECTION class): a classifier whose
construction can silently route an entire phenomenon into a different bucket must be
validated against a hand-read sample of the SPECIFIC phenomenon it claims to measure
before its bucket counts enter a finding.
