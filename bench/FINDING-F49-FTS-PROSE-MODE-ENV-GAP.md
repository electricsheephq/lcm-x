# FINDING F49 — LoCoMo FTS inertness solved: prose-mode flag never set (env-wiring gap)

Date: 2026-08-02. Status: mechanism CONFIRMED (zero-spend; reproduced in-main). Follow-on
measurement (replay effect estimate) in flight. Supersedes the open question in F48 §3-CORRECTION
finding 1 ("why does the prose-mode arm lose every fused slot").

## 1. The finding
The declared-config LoCoMo run (F48, 54.6%) executed with **FTS effectively dark: 3/49,650
delivered hits** — because `LCM_FTS_PROSE_MODE` (the opt-in flag gating #183's prose-mode fix,
`config.py:388`, default False) **was never set in the run environment**. The prose-mode CODE was
present in the measured build (product 9d181aa); the FEATURE was off. Every FTS query ran in the
old conjunctive-AND form, which near-never matches multi-word conversational questions.

**"Code shipped ≠ feature on."** The pins captured the git sha faithfully; nothing in the
registration process compared the run env against the product's feature-flag inventory.

## 2. Evidence (verified in-main, not agent-trusted)
- `run-locomo-aa.sh:15-27` (the actual runtime env source): 8 exports, no `LCM_FTS_PROSE_MODE`.
  `data/pins-locomo.yaml` env block likewise omits it. `config.py:388` `_EnvFieldSpec("fts_prose_mode",
  "LCM_FTS_PROSE_MODE", bool)` → False when unset; `store.py:1147` gates disjunctive routing on it;
  `tools.py:2513-2544` only pass it downstream when set.
- One-question reproduction on a retained store (conv-26 q55, copied read-only): as-run conjunctive
  query → **0 rows**; the product's own classifier `should_use_fts_prose_mode(query)` → True; the
  disjunctive form it would have built → **52 rows including the literal gold-evidence turn**.
- Not a one-off query shape: 29/30 uniform-random locomo10 questions classify prose-mode-eligible.
- Alternative mechanisms REFUTED: index fully populated (419/419 rows 1:1, H1); fusion favors the
  FTS arm in every pull round — there was simply nothing to pull (H3; matches F46 §2's 0/41 raw
  top-200 measurement); delivered-hits accounting is a faithful literal count (H4).

## 3. Implications
1. **Headroom, honestly labeled:** the banked 54.6% was earned with one retrieval arm off. An
   FTS-prose-ON config is a NEW declared config — own registration, pre-declared bars, its own
   A/A′ noise floor. No silent retune, no retroactive edits to the F48 row.
2. **`HERMES_MB_FUSION=quota:fts=1,chunk=2` was tuned against a near-empty FTS arm.** The ratio
   must be re-derived once FTS returns real candidates (part of the new config's registration).
3. The F46 §2 "0/41 gold turns in raw FTS top-200" measurement is now explained, not mysterious.

## 4. Pre-registration requirements for the FTS-ON declared config (before any paid run)
1. **Replay effect estimate (zero-spend, in flight):** replay the F46 41-miss list + a 100-question
   random sample through both query forms on retained stores; bank gold-recovery delta + candidate
   volumes. Proceed to a paid run only if the replay shows material recovery.
2. **Precision guard:** confirm classifier routing leaves compact-keyword/quoted queries unchanged.
3. **Fusion-ratio re-derivation** from replay candidate volumes, fixed before registration.

## 5. New standing rule (program-wide, effective immediately)
**Feature-flag inventory diff at registration:** a declared config's pins must include the diff of
the run env against the product's full env-flag inventory (`_EnvFieldSpec` table or equivalent) —
every flag either explicitly set or explicitly listed as default-with-value. A shipped,
default-off feature is OFF, and the registration must show it was known to be off. (Family:
wire-as-you-go / gate-every-caller; the pins pinned the code, nobody pinned the switches.)

## 6. Artifacts
- Decomposition + reproduction: session-notes 2026-08-02 `fts-inertness/` (H1–H4 verdicts,
  copied sample store); replay (pending): `fts-prose-replay/`.
- Code refs: wt-locomo-product `config.py:388,688`, `store.py:1147`, `tools.py:2387,2513-2544,
  4702-4737`, `search_query.py:158-200`; bridge `hermes_lcm_bridge.py:134-198`.

## 7. Replay results (2026-08-02, zero-spend — §4 reqs 1+2 SATISFIED)
Real code path (`should_use_fts_prose_mode` + `build_fts5_match_query` + `MessageStore.search()`
imported from the product), read-only store copies, zero LLM calls. Artifacts:
session-notes 2026-08-02 `fts-prose-replay/` (scripts, per-row CSVs, SUMMARY.json).
- **Gold recovery on the F46 41-miss list (raw FTS arm, top-200): 0/41 as-run → 22/41 (54%)
  with prose mode on.** By category: single-hop 0→10/25, multi-hop 0→7/8, world 0→5/8.
- **Candidate volume (100q uniform sample, seed 20260802): conjunctive median 0 / p90 0
  (1/100 questions returned any hit); prose median 155 / p90 200 (cap). Prose-eligible: 99/100.**
- **Precision guard: clean** — the classifier-negative sample question produced identical
  query + identical results under both modes; 0 violations.
- Nuance for exact-match tooling: new-run ingestion appends `[shared image: …]` captions, so
  gold text is a prefix of stored content on 12/41 rows (normalized-substring match used).
- Scope note: F46 §7's 0/22 FUSED recovery measured the fused output of a run where prose mode
  was (per this finding) never active; this replay measures the raw FTS arm with the feature
  genuinely on. Different stages, different config states — not in conflict.
- **Remaining prereq before registration: §4 req 3 (fusion-ratio re-derivation)** — the sim must
  be CATEGORY-BALANCED this time (the F46-era sim's imbalance is a documented
  gate-proxy-calibration echo) and uses the part2 candidate-volume data as its input.

## 8. Fusion-ratio re-derivation result (2026-08-02 — §4 req 3 SATISFIED; all C1 prereqs closed)
SPEC-FUSION-RESIM executed: 100q category-balanced (20×5, seed 20260802), real bridge
`quota_merge` imported verbatim (3-case self-test passed), chunk arm replayed offline from
retained embeddings + cached fastembed model, delivered limit 25 confirmed from source
(search.ts:60). **Selected by the a-priori rule: `fts:chunk = 1:1`** — aggregate gold-turn
delivery 53.66% vs 50.24% (prose@1:2); tie with 2:1 broken on fewer displaced gold-chunk hits
(5 vs 8) + smaller config delta. 1:3 rejected on the category-loss constraint. Headline
recomputed from the raw per-question CSV by the orchestrator (110/205, exact match).
Context: the AS-RUN banked config had a dark FTS arm ≈ the chunk-only control (40.00%), so C1's
predicted delivery delta vs the banked run is ≈ +13.7pt. Disclosed deviation: 4/2,815 dataset
evidence entries pack multiple dia_ids in one string; split via pattern, documented in resim.py.
→ Registration: `specs/RUN-SHEET-LOCOMO-C1-FTS-ON.md` (bands pre-declared; launch gated on the
machine slot per its §5).
