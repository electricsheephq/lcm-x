# GOALS & ROADMAP — the program's funnel (v1, 2026-08-03; owner-directed)

## G0 — THE GOAL
**Demonstrably best-in-world agent memory for frontier-model agents** — proven by disclosed,
reproducible numbers across the field's benchmarks on a public disclosure-first scoreboard,
with the technology adoption-ready (the Hermes → Claude Memory migration path). "Best" is
claimed per-instrument against honest ceilings, never against undisclosed-config numbers.

Everything below funnels into G0 through exactly one of four channels:
**coverage row · performance lever · fault burn-down · standard artifact.**

## G1 — Coverage (every cell banked + disclosed)
| Instrument | Status | Next |
|---|---|---|
| LongMemEval V1-S | ✅ 455/500 (91.0%) | done — at measured ceiling (~91.8 oracle) |
| LongMemEval V1-M | 🔄 smoke running | #198 merge → shard plan → REGISTERED RUN (flagship) |
| LongMemEval V2-S static | ✅ 123/451 re-baseline | maintained via paired gates only |
| LongMemEval V2-S agentic | ✅ 298/451 (66.1%) | re-run on the new train post-#436-settle |
| LongMemEval V2-M | ⬜ never run by anyone | go/no-go after V1-M lands (first-mover) |
| LoCoMo | ✅ 54.6% strict-judge, noise floor 3.47% | levers: FTS-inertness fix, B3, Voyage config |
| AMA-Bench | adapter ✅, pilot staged | timing pilot → full 208 (budget from pilot) |
| AMB suite (LoCoMo10/LME-S/PersonaMem/BEAM×4/LifeBench) | adapter ✅ | rows post-AMA, Claude judge, pinned configs |
| MemoryArena | deferred (written reasons) | revisit after AMA answers the cell question |
| (No "L" tier exists on V1/V2; our 389× instrument is the de-facto L) | ✅ F44 curve | extend at V2-M time |

## G2 — Performance targets (honest ceilings, not vanity parity)
- **V1-M ≥ 90%** with a frontier reader — THE flagship claim; both constraints (scale retrieval,
  reader) are ours to win. F44 says retrieval barely degrades at 389×; prove it on the official tier.
- **V2-agentic ≥ 75–80%** (leads every published number); 90 is a long climb on a benchmark
  curated to defeat memory systems — F30's funnel attribution is the roadmap.
- **LoCoMo ≥ 70% under OUR strict judge** + a one-time dual-judge bridge number so parity vs
  lenient-judge 90s is a stated equivalence, not a config trick. Corrupt-gold ceiling ≈95% disclosed.
- **V2-M: any number** — first mover defines the reference.
- **AMA ≥ 72.26%** (their GPT-5.2 baseline) on first real run.

## G3 — Product-fault burn-down (each closes with a measured before/after)
1. FTS delivery inertness (3/49,650 — F48 CORRECTION finding) — investigation open.
2. B3 adversarial speaker-attribution (fact retrieved on 78.6% of misses) — biggest single LoCoMo lever.
3. #191 tail (validated backlog incl. evidence-pack diversity-vs-citations).
4. Voyage-context-3 embeddings as the frontier declared config (owner steer 2026-08-02) —
   own A/A′ per instrument, expected to lift the embedder-bound classes.

## G4 — The standard ships (R3)
Disclosure standard + honesty narrative + two-tier doctrine (+ the F46→F48 correction story) +
public scoreboard (owner naming pending — the only external gate) + upstream #436 merged clean.

## G5 — Ops invariants (already in force)
RUN-LOG.md append-only per run · pins/pinverify per lane · A/A′ per new instrument-config ·
paired gates for delivery changes · author≠judge on load-bearing verdicts · same-day corrections.
Superseded plan-of-record eras are recorded in `PROGRAM-HISTORY.md` (append-only; one curated entry
per era, verbatim archive hash-anchored).

## Sequencing (next ~72h, machine-bound critical path)
1. #436 settle: sanity slice verdict on fa00ec9 (RUNNING) → batch-4 re-pass → maintainer merge window.
2. #198 merge → V1-M shard plan (from smoke report) → registered V1-M run (~1-2 days sharded, Voyage).
3. AMA timing pilot (parallel, API-bound) → full-run budget decision.
4. V2-agentic re-run on the settled train; LoCoMo FTS investigation (zero-spend start).
5. R3 assembly completes when 1+2 land; scoreboard publicization on owner naming.
