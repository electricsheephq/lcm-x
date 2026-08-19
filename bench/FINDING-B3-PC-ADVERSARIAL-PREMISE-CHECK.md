# FINDING B3-PC — adversarial premise-check: the B3 hypothesis HOLDS (85.3% reader-side)

Date: 2026-08-02. Spec: `specs/SPEC-B3-PREMISE-CHECK.md` (gates pre-declared 2026-08-02 BEFORE
counting; pushed at 09adff5). Zero LLM spend. Classification by fast-worker agent; headline
counts recomputed by the orchestrator from the raw per-row CSV before banking.

## 1. Gate verdict (pre-declared §4, adjudicated mechanically)
**E+R− = 256/300 = 85.3% ≥ 60% → the B3 premise HOLDS.** On the declared-config run's
adversarial misses, the gold evidence was already DELIVERED in the reader's context in 85.3% of
cases — the reader accepted the question's false premise anyway. The lever is READER-SIDE
(conflict-status rendering, M7b Evidence Assessment seam), not retrieval. Per the pre-declared
gate, the retrieval-side design is not pursued; the E− minority (14.7%) is banked as follow-on.

## 2. Counts (fixed denominator = 300 classifiable misses)
| class | n | % |
|---|---|---|
| E+R− evidence delivered, premise accepted | 256 | 85.3% |
| E− gold evidence absent from delivery | 44 | 14.7% |
| GOLD? (corrupted-gold overlap) | 0 | 0% — STRUCTURAL: the 99-row community audit scoped categories 1–4 only; all 446 adversarial rows are outside audit scope by construction |
| UNCLEAR | 0 | 0% (vs 20% second-pass threshold — no LLM pass warranted) |

Universe: n_adv = 446 (locomo10 category-5 ∩ run questionType, 0 unmatched); misses = 300;
sanity anchor: 146/446 = 32.74% reproduces F48's banked 32.7% adversarial accuracy exactly.

## 3. Method disclosure (the honest caveat)
Evidence presence = normalized-text substring match between locomo10 gold-evidence turn text and
delivered `searchResults[].content` (the run's searchResults carry no dia_id metadata — same
proxy as the F48 single-hop decomposition). Spot check found no short-evidence false positives
(0 E+R− rows with <20-char evidence text). This is a design-direction gate, not a published
score; the proxy is disclosed, and the lever itself will be judged by a registered paired gate.

## 4. Design sketch (spec §5.3 — design only, no implementation in this unit)
Extend the M7b Evidence Assessment block (which today renders ABSENCE statuses only) with a
**presence-with-conflict** status class: at render time, when a delivered evidence item's
speaker/entity binding contradicts the binding presupposed by the query (Caroline-said-X asked,
evidence attributes X to Melanie), emit one explicit conflict line naming both bindings and the
evidence turn. Reader-facing, purely additive, one line per conflict (capped), no retrieval
change, no new model calls at answer time beyond the existing render path. Detection candidates
(to be decided in the product spec): store speaker metadata vs query-parse binding; fallback
lexical heuristic. Risks to gate: false-conflict noise on legitimate questions (must show no
regression on non-adversarial categories in the paired gate). Next step: product spec + own
registration; targets the GOALS LoCoMo ≥70% strict-judge line alongside the F49 FTS-ON config.

## 5. Artifacts
Session-notes 2026-08-02 `b3-premise-check/`: `classify_adversarial_misses.py`,
`adversarial_miss_classification.csv` (all 300 rows), `b3_pc_summary.json`, `run_output.txt`.
Orchestrator recompute: Counter over the CSV → {E+R−: 256, E−: 44} (match).
