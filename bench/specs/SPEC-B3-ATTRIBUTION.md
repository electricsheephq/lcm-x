# SPEC B3-A/B3-B — attribution-preserving ingestion, then conflict-status rendering

Status: PRODUCT SPEC (design-of-record for task: B3 conflict-status lever). Supersedes the
FINDING-B3-PC §4 sketch in one load-bearing way discovered on code+store inspection.

## 1. Design finding that reshapes the lever (verified 2026-08-02)
**The store carries NO speaker attribution.** Retained-store inspection: `messages.content` is
raw turn text ("Hey Mel! Good to see you!…") — no speaker prefix; schema has only
`role ∈ {user, assistant, tool}` (both LoCoMo speakers land in the same two roles).
Bridge cause: `hermes_lcm_bridge.py ingest()` maps each harness message to
`{role, content}` only — any speaker field the harness supplies is dropped.
Consequence: FINDING-B3-PC's "reader accepts the false premise with evidence in context"
(85.3%) is not pure reader credulity — **the attribution half of the evidence was never
stored.** No reader can verify "did Caroline or Melanie say X" from unattributed text; vocative
cues ("that's cool, Caroline!") are incidental and often inverted (a vocative names the OTHER
speaker). A render-time conflict check (the B3-PC §4 sketch / M7b extension) is INFEASIBLE
until attribution exists. The lever therefore has two stages:

## 2. B3-A (primary): attribution-preserving ingestion
- Bridge `ingest()` prepends a canonical speaker prefix to each turn's stored content
  (`"<Speaker>: <text>"`), sourced from the harness session payload (extend the provider
  payload if the harness currently omits speaker — locomo10 carries per-turn speakers).
- Delivered hits then carry attribution end-to-end (FTS text, chunk text, answer-ready
  content, evidence cards) with zero new render machinery.
- **This changes stored text → FTS tokens + embeddings + chunking = measured surface** →
  fresh stores, own registration, own A/A′. No silent combination with other changes.

## 3. B3-B (secondary, only after B3-A banks): presence-with-conflict status
The M7b/preanswer_evidence extension from FINDING-B3-PC §4 — an explicit conflict line when
the query's presupposed binding contradicts delivered attribution. Feasible only once B3-A
exists. Deferred until B3-A's measured effect is known (B3-A alone may capture most of the
85.3% pool; B3-B is the residual lever).

## 4. Registration sequencing (causal attribution discipline)
- **C1 = FTS-ON config** (F49; query-side only, SAME stores) — registers and runs first
  (prereq: fusion-resim in flight). LoCoMo answer/judge run via codex CLI (gpt-5.6-sol per
  run-locomo-aa.sh) — NOT blocked by the OpenRouter outage.
- **C2 = C1 + B3-A attribution ingestion** (fresh stores) — registers after C1 banks, so C2−C1
  isolates the attribution effect. Bars set at C2 registration against C1's banked numbers
  (structure pre-declared now: adversarial must gain materially; no other category may lose
  beyond the C1 noise floor; exact numbers fixed before C2 spend, after C1 banks).
- Zero-spend pre-check before C2: re-ingest ONE conversation with prefixes; replay its
  questions' retrieval; delivered-set overlap vs unprefixed stores must show no material
  retrieval regression (prefix tokens must not displace evidence from delivery). Bar: ≥90%
  same-gold-delivery on that conversation's question set, checked before any paid C2 run.

## 5. Non-goals
No store schema migration (prefix-in-content, not a new column) for the benchmark lane; a
first-class speaker column is a product-roadmap item, not a benchmark prerequisite. No changes
to role semantics. No judge/rubric changes (the F48 narrowed rubric stays the instrument).

## ADDENDUM 2026-08-02 — §4 zero-spend pre-check EXECUTED: PASS (105.2% vs ≥90% bar)
conv-26, 199 questions, C1 config (prose FTS + 1:1 quota + 25 slots), control-reproduction
verified (419/419 messages, all-row content equality). Gold-turn delivery: control 172 →
prefixed 181 (**105.23%**, orchestrator-recomputed from per_question.csv, exact match).
Prefixing IMPROVES delivery net +9 (13 questions gained, 5 lost — losses are single-turn
rank reshuffles). Seam confirmed: speaker present upstream (`locomo/index.ts:191-195`),
dropped by bridge ingest (`hermes_lcm_bridge.py:512-517`) — B3-A implementation is a small
bridge change. Zero LLM calls; offline socket guard clean. Artifacts: session-notes
2026-08-02 `b3a-precheck/artifacts/`. C2 is now fully pre-staged pending C1 banking.
