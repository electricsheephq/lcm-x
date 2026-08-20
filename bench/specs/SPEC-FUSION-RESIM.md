# SPEC — category-balanced fusion-ratio re-derivation (F49 §4 req 3; zero-spend)

Registered before execution. Selection rule fixed A PRIORI (§4) — chosen before any counting,
per the gate-proxy-calibration rule. Fixes the F46-era sim's two defects: category imbalance
(25 rows, not balanced) and simulating against a conjunctive (near-empty) FTS arm.

## 1. Question
With FTS prose mode genuinely ON (F49 §7: candidate volume median 155), which
`HERMES_MB_FUSION=quota:fts=X,chunk=Y` ratio should the FTS-ON declared config register?
The current 1:2 was tuned when the FTS arm returned ~nothing — it encodes no information about
a live arm.

## 2. Sample (fixed before execution)
- 100 questions: 20 per category × 5 categories, uniform within category, seed 20260802,
  drawn from questions whose conversation stores are retained in the declared-run A arm.
- Balance is STRUCTURAL (exactly 20/20/20/20/20), not expected-value.

## 3. Method (zero LLM spend; local fastembed + SQLite only)
For each question, on a COPY of its retained store:
1. FTS arm: prose-mode query via the product's real classifier + builder (replay machinery,
   session-notes fts-prose-replay/replay.py), ranked top-200.
2. Chunk arm: the product's chunk-vector search with fastembed bge-small locally (same model
   as the declared run), ranked top-200.
3. Fused delivery sim: replay `quota_merge`/`pull()` (bridge hermes_lcm_bridge.py:134-198 logic,
   imported or faithfully replicated) to the declared delivered-slot count for each ratio in
   the FIXED grid: fts:chunk ∈ {0:1 (FTS-off control), 1:3, 1:2 (current), 1:1, 2:1, 1:0}.
4. Score each ratio per question: gold-evidence turn delivered (same normalized-substring
   match as F49 §7, disclosed) — plus displaced-gold accounting (a gold CHUNK hit pushed out
   of the delivered set by FTS quota).

## 4. Selection rule (A PRIORI — no post-hoc alternatives)
Registered ratio = argmax of AGGREGATE gold-delivery rate over the 100-question sample, subject
to: no single category loses >2.0 points of gold-delivery vs the 1:2 current baseline. If the
argmax violates the constraint, take the best ratio that satisfies it. Ties → the ratio
displacing fewer gold chunk hits; still tied → the ratio closer to 1:2 (smaller config delta).
The 0:1 and 1:0 grid ends are CONTROLS, not candidates (they cannot be selected; they bound the
arms' individual contributions). Whatever wins is disclosed with the full per-category table —
including if the answer is "keep 1:2."

## 5. Deliverables
- Per-question × per-ratio CSV + per-category summary table + the selected ratio, to
  session-notes 2026-08-02 fusion-resim/ artifacts.
- Result section appended to F49 (or a short F50 if the result is surprising).
- The selected ratio becomes part of the FTS-ON declared-config registration (with F49 §5's
  full env-flag inventory diff). Registration itself is a separate step; this sim only fixes
  the ratio input.
