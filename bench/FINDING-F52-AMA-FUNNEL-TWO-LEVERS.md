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
