# FINDING F62 — V1-M re-bank of the F53 flagship row PARKED at the pre-spend gate: the shipped privacy posture does not admit the LongMemEval-M corpus

Date: 2026-09-05. Registration: `bench/specs/RUN-SHEET-V1M-REBANK.md` (#413, merged 22c12b21; Amendment 1 rides this PR).
Instrument: `benchmarking/longmemeval.py` blob ab76738a · `scripts/lcm_longmemeval.py` fd477971 · `ingest_protection.py` 864098d9 ·
`config.py` 6819944a — all at 22c12b21 (launch pins `pins-rebank-launch-20260905T024456Z.txt`, every value from a command; worktree
`wt-v1m-rebank`, 0 dirty files). Corpus: prepared-m, manifest sha256 300cf936… (identical to F53's), shard manifests 0–5 pinned
(shard-0 4bac8d3c… matches the F53 checkpoint header). Embedding cache: the F53 cache, 505,695 rows, 4,442,599,424 bytes, mtime Aug 19.
Evidence: `session-notes/2026-09-05/v1m-rebank/artifacts/` (`gate-run.log`, `prewarm-gate/`, pins). Kit: `bench/tools/v1m-rebank/`.

## 0. Verdict (one line)

**PARKED at the gate under §7 (one block is a park), 2026-09-05T02:48Z — no shard launched, no §4.3 outcome named, F53 stays the
row of record, and the #366/#374/#384–#391 ledger boundary is measured LIVE on this corpus (not inert).** The dry run stopped at
document 227,951 with `status: blocked` — "cloud embedding privacy residual detector blocked pattern names: private_key" — after 61
transformed units. The whole-corpus replay (§3) shows the shipped posture re-shapes 90 unique units (599 occurrences, 248 of 500
questions) and refuses 3 unique units (18 occurrences, 17 questions — one question carries two refused units); the §4.2 transform-change count (documents for which the transform returned `changed=True` OR raised) is
therefore 617 occurrences = 93 unique units in 258 of 500 questions. Every refusal is a false positive of one marker-independent
backstop on text with no PEM marker (§8a). The run cannot be completed under posture (a) without a posture or product change; the
options are the owner's (§10).

## 1. Declared posture (what changed vs F53, by construction)

- F53 (2026-08-19, lcm-x v0.22.0): the harness embedded raw text — no provider-copy transform existed.
- This row: the production path — durable store lossless (`sensitive_patterns_enabled=False`), provider-bound copies protected
  (`embedding_privacy_enabled=None` → auto-on for the cloud provider), revision `privacy:v3:3f350902…` (resolved by the harness's
  production-mirroring `_embedding_privacy_context`, #370, at 22c12b21). Rerank boundary not exercised (as F53). Chunk-embedding mode pinned `flat`
  (F53 semantics; the #352 contextual grouping is a separate registration — #413 r4 architect decision).

## 2. Headline metrics

Not produced. No shard ran; the gate parks before the first spend-bearing step. F53's declared values (r@1 0.4999 · r@10 0.9559 · ndcg
0.8610 · 470 scored + 30 abstention-excluded) remain the row of record. Fail-closed accounting: no rows exist to account for.

## 3. Transform-change count (§4.2 — the ledger-discharge measurement)

| step | documents walked | transformed | refused | scope |
|---|---|---|---|---|
| determinism probe (`--sample-size 20 --seed 0`) | 20 of 20 | 0 | 0 | `sample` (probe.json) |
| `prewarm-cache --dry-run` | 227,951 (stopped at the first block) | 61 (`changed-manifest.jsonl`: 61 rows / 25 questions) | 1 | not recorded — the blocked report carries only `status`, `privacy`, `error`; the gate parks on the block before its coverage check (§8) |
| whole-corpus local replay (`corpus_privacy_inventory.py`, 796 s, no provider calls; selected 500 = prepared 500, i.e. the shard union is the corpus) | **2,513,035 occurrences = 505,695 unique units** | **599 occurrences = 90 unique units, in 248 of 500 questions** | **18 occurrences = 3 unique units, in 17 of 500 questions** | all 500 questions (corpus-privacy-inventory.json) |
| the 500 query texts (same transform + validator) | 500 | 0 | 0 | — |

Placeholder occurrences among the 61 manifest units (`changed_units_classes.py`): `password_assignment` 44, `api_key` 38, `private_key` 2 —
synthetic chat credentials, the transform doing what it ships to do. Question-level footprint: 248 questions carry a re-shaped unit, 17 carry a
refused unit, 7 carry both, 258 carry either (the per-question map in the inventory). The dry run's stderr progress counter
(`prewarm processed=178900` at the stop) counts prewarm request units, not `privacy.documents`, so the two figures are not comparable.
Interpretation under §4.2: the transform-change count (returned `changed=True` OR raised) is 617 occurrences = 93 unique units in 258 of
500 questions — 599 / 90 / 248 re-shaped + 18 / 3 / 17 refused, 7 questions in both — the count is > 0, so the boundary is
**live for this corpus**; whether any metric delta would be attributable to it was never reached (§4.3 needs a completed run).

Exact spend the real prewarm would have needed had nothing blocked (`cache_membership_check.py`): the 90 unique transformed units map to
90 unique protected digests, none present in the cache → `would_populate` = 90 ≈ $0.007 at the sheet's cap-basis price ($40 / 505,695).

## 4. Corpus identity vs F53 (§4.4)

- **Unit identity, verified by membership, not by count** (`cache_membership_check.py`, 122 s): the cache holds exactly one
  (provider, model) group, `voyage` / `voyage-context-3`, 505,695 rows; **505,695 of 505,695 unique raw units of prepared-m are present
  (key = sha256 of the exact unit text, the harness's `content_sha256`); 0 raw units missing; 0 cache rows outside the corpus.** The F53
  cache is the corpus's raw unit set, exactly. This is the §4.4(a) bar established directly, because the dry run stops at the first
  block before it reports `already_cached`.
- Per-shard cache pair (§4.4(b), sum of per-question `embed_cache` rows vs F53's shard files 398,139 / 397,857 / 390,572 / 389,980 /
  394,388 / 390,617): **not produced** — no shard ran.
- Forward baseline (`corpus_counts`, `embed_cache` per question): **not produced**.

## 5. Reproducibility verdict (§4.3)

**None.** REPRODUCED / REPRODUCED-TRANSFORM-INERT / MOVED-EXPLAINED / MOVED-UNEXPLAINED all presuppose a completed run under the
registered posture; the gate parked before the run. The kit's `identity_all.sh` / `result_identity.py` were self-tested on F53's own
outputs before the run (500/500 shard self-pairs, 100/100 A′, 0 discordant A/A′ rows under the projection) and were not otherwise used.

## 6. A/A′ (prepared-m-aprime100, seed 20260802)

Not run.

## 7. Cost and time

Gate: `record_pins.sh launch` 02:44:56Z → PARK 02:48Z (probe + dry run). Voyage spend: the determinism probe's 2 × 20 documents;
the dry run embeds nothing; no shard, no A′. OpenRouter: not used. Local analysis after the park: inventory 796 s, attribution and
class probes seconds, membership check 122 s — all offline. Sheet cap ($40, prewarm-scoped) never approached.

## 8a. Root cause, attribution, production behaviour

- **Every refused unit fires exactly one sub-detector**, `_has_orphan_full_width_base64_run` (the #384 round-6 marker-independent
  backstop) (`blocked_units_attribution.py` → `blocked-units-attribution.json`; per-detector isolation over the transform's residual
  checks). None of the three texts carries a `-----BEGIN`/`-----END` marker or the words "private key"; none mentions a password or a
  key word (shape flags in `corpus-privacy-inventory.json`). Line-model record (`refused_line_model.py` → `refused-line-model.json`:
  per model segment its kind, 1-based physical line, width = `content_end − redact_start`, token count, longest token, and whether the
  backstop counts it; raw text is never written):
  - text `021513ef…` (9 occurrences / 9 questions): 14 lines, a numbered list of ever-longer strings (lines 3–11, two tokens each);
    the backstop counts **lines 10 and 11** — `PREFIXED_B64`, widths 42 and 81, longest tokens 39 and 78 (line 9, width 23, is the
    same class under the width floor). Run of 2 → refused.
  - text `eefcadc7…` (5 / 5): 58 lines of prose, longest token anywhere 20; the backstop counts **lines 5 and 6** — `PREFIXED_B64`,
    widths 60 and 42, 7 and 6 ordinary words ending in a 17- and a 16-character word (lines 7, 8 and 13 are the same class under the
    floor). Plain prose classified as PEM body lines; run of 2 → refused.
  - text `401a2dc6…` (4 / 4): 30 lines; the backstop counts **lines 11–27** — 17 contiguous `PREFIXED_B64` segments of width 62–63,
    two tokens each (`N. <60–61-character token>`), a numbered list of long strings.
  Every counted segment has `prefix_chars` 0: `redact_start` sits at the line start, so the width compared with 40 is the whole line,
  leading tokens included.
- **Mechanism** (`ingest_protection.py` at 22c12b21): `_pem_line_model` (:586) classifies a line as `PREFIXED_B64` (kind 9) when a few
  leading tokens are followed by a base64-charset tail; `_has_orphan_full_width_base64_run` (:1797–1824) counts a segment when its kind
  is `STRICT_B64` or `PREFIXED_B64` and `content_end − redact_start >= 40` (:1811–1814), resets the run on a placeholder or any other
  line, and returns True at a run of 2 (:1820). For these lines the width is the **whole line, prefix plus tail**, so two consecutive
  prose lines of 6–7 words whose last word is 16–17 characters long satisfy it. The docstring premise (:1805, "ordinary prose/config
  never produces them") is refuted by `eefcadc7…`. Rate on this corpus: 3 units per 505,695 (0.0006%); through shared haystack
  sessions 17 of 500 questions (3.4%); the harness fails a question loud on a single block, so the run cannot complete.
- **Production behaviour (flagged, not built):** `command.py`'s cloud backfill `break`s with `stop_reason = "privacy_refused"` on the
  first such document — one benign chat halts cloud embedding for the store until the operator opts out. Recorded on #394 (owner-
  reclassified redaction backlog; changes to the scanner are owner-gated since 2026-08-27).

## 8. Disclosures

- Instrument lineage: #413 r1 audit FAIL 74/3 → r2 (golden re-banked; 9 protect sites; six-key counters; blocked report) → PASS 96/0 →
  r3 (dry run; blocks always count; per-question `embed_cache`) → PASS 97/0 → r4 (chunk mode pinned flat + fail-loud cache guard;
  `--changed-manifest`) → PASS 96/0 → r5–r18 (16 real findings from the review bots and four independent-confirm FAILs, all fixed and
  disclosed) → r19–r21 (per-row cache posture; probe chunk identity; completed-resume aggregate follows the rows) → r22 (sheet
  precision) → r23 (`privacy_scope: corpus` only for an exact partition of the prepared manifest; sidecar guard) → final confirm at the
  merge head b49c53d8 **PASS 97/0** (`review-413-r23-opus-deep-reasoner.md`); merged 22c12b21 through the two-lane receipt gate.
- The dry run's blocked report carries `status`, `privacy` (six counters) and `error` (pattern names only); no run-time receipt was seen
  because no run started. The one all-zero-receipt class the sheet describes (initialization refusal) did not occur.
- Known-blocker verdicts (#235 warmup, #236 bun orchestrator): DO-NOT-APPLY, unchanged (no shard ran).
- Correction of record: the #380 park comment (03:09Z) stated "= the F53 cache exactly → corpus identity holds" on the strength of
  count equality (505,695 = 505,695). Membership was verified afterwards (§4) and the statement is true; at the time it was posted it
  was a count, not an identity. The follow-up comment on #380 says so.
- Kit: `prewarm_gate.sh`, `run_shard.sh`, `run_aprime.sh`, `result_identity.py` and `refused_line_model.py` are byte-identical to
  the as-run copies; `corpus_privacy_inventory.py`, `changed_units_classes.py` and `blocked_units_attribution.py` were lint-normalized
  (statement splits, one loop-variable rename); the review of PR #416 then fixed seven committed copies (per-file list in the kit
  README). None alters a number in this record: the three probes whose logic changed (`corpus_privacy_inventory.py`,
  `changed_units_classes.py`, `cache_membership_check.py`) were re-run with the committed copies and reproduced the as-run artifacts;
  three (`cache_pair_check.py`, `identity_all.sh`, `launch_all.sh`) never ran in the parked execution; the seventh, `record_pins.sh`,
  ran once at launch (`pins-rebank-launch-20260905T024456Z.txt`) and its as-run regex inventory listed 89 of the 96 declared
  `_EnvFieldSpec` fields under a printed `inventory_count=96`. The seven it dropped (`LCM_ADAPTIVE_RETRIEVAL_ENABLED`,
  `LCM_ASSERTION_EXTRACTION_ENABLED`, `LCM_ASSERTION_EXTRACTION_MAX_SOURCES_PER_PASS`, `LCM_ASSERTION_EXTRACTION_TIMEOUT_SECONDS`,
  `LCM_EMBEDDING_API_KEY_ENV`, `LCM_EMBEDDING_BASE_URL`, `LCM_FRESH_TAIL_PRESSURE_YIELD_MIN_OBSERVATIONS`) were all unset at launch per
  the same file's independent `LCM_/HERMES_/REBANK_` environment listing, which completes the §3 pin inventory for this execution; the
  committed copy enumerates all 96 structurally and refuses on a count mismatch. Both sha256 manifests are committed beside the kit. Four further kit findings from the review's third pass (gate-marker binding
  in `launch_all.sh`, failure propagation in the `record_pins.sh` pin block, credential redaction of the captured run environment and a
  product-sha resume guard in `run_shard.sh` / `run_aprime.sh`) touch no artifact of this execution and are tracked on #415 as
  prerequisites for the next execution of the sheet.
- What this finding does NOT prove: nothing about F53's numbers under the raw path (no raw-path re-run exists at this head); nothing
  about retrieval quality under the shipped posture (no shard ran); the false-positive rate is for this corpus only.

## 9. Ledger

Nothing discharges. `bench/BASELINE-LEDGER.md` rows #366, #374 and #384+#391 have their status cells updated in place to the measured
result (§4.2 transform-change count 617 occurrences = 93 unique units: 599 / 90 re-shaped + 18 / 3 refused) — rows never edited, per the ledger rule. The #203/#177 rows
(cache-pair identity + forward baseline) are untouched: §4.4(b) needs a completed run. #380 and #367 stay open. The F53 finding and the
F53 scoreboard row carry a dated append-only line pointing here (the freeze stands; no number changes).

## 10. What the next row can be (owner decision, #380)

1. **Nothing further now** — this record stands; F53 remains the pre-privacy-trio flagship of record with the note that the production
   arm is not executable over LongMemEval-M at 22c12b21. Zero cost. (Architect's default until the owner decides.)
2. **Precision fix of the backstop** (measure the ≥ 40 width on the base64 tail rather than prefix+tail, or require PEM body geometry),
   then re-run this registration unchanged. Redaction work → owner-gated (2026-08-27 ruling); expect a review cycle of the #366 kind;
   the re-run itself ≈ 90 embeddings + ~45 min.
3. **Opt-out-posture variant** — a NEW registration adding an instrument knob for `privacy:off` (the raw path) as the engine-drift check
   against F53. Measures the engine, not the privacy posture; small instrument change through the gate.

Architect recommendation: 1 now; 3 as the next registration; 2 only on explicit owner ask.
