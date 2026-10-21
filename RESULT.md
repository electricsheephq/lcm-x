# #1586 SQL-cost result

Claim class: advisory measurement on this Mac; product PR with local proof. Phase 1 PASS; final product PASS.

Run: `lcmx-1586-sqlcost-v1`. Receipt card: RESULT-MANIFEST.json.
Phase 1 source: `33e0b5afd6d3d4d88a0a8cc61139a822554fdea2`. Product base: `7582e302127d7d36ecbcb54e816309c484d7b7dc` (contains #1745). Product head: `4b8d5ce6f47e4e2ce9384ae1b768ab87a481c3a0`.

Q1: `What turn was acknowledged`; Q2: `Recall the filler from a turn`; Q3: `Which turn contains a value`; Q4: `VALUE_S1586_SHARD000007_0000`.

## Q2 phase profile

All phases use the captured product query `Recall OR filler OR turn`, LIMIT 1000 OFFSET 0. P1 is MATCH count; P2 is rank-only without join; P3 adds join and full sort; P4 adds all message columns; P5 adds snippet. Five interleaved idle runs per phase and index, without a progress handler.

| Index | Phase | Median [min, max], s | n |
|---|---|---:|---:|
| stem | P1 | 0.008 [0.008, 0.012] | 5 |
| stem | P2 | 0.166 [0.149, 0.189] | 5 |
| stem | P3 | 0.175 [0.164, 0.177] | 5 |
| stem | P4 | 0.279 [0.271, 0.286] | 5 |
| stem | P5 | 1.666 [1.527, 1.735] | 5 |
| plain | P1 | 0.008 [0.008, 0.008] | 5 |
| plain | P2 | 0.162 [0.154, 0.170] | 5 |
| plain | P3 | 0.175 [0.168, 0.181] | 5 |
| plain | P4 | 0.278 [0.269, 0.281] | 5 |
| plain | P5 | 1.513 [1.399, 1.806] | 5 |

Both indexes match 250,000 rows. The instrumented P5 evaluates both content and snippet 125,504 times for a 1,000-row window. The instrument preserves output column names. EXPLAIN QUERY PLAN scans the FTS index, looks up each message by primary key and uses a temporary ORDER BY B-tree.

H1: refuted as literally phrased (not every match); confirmed pre-LIMIT amplification: 125,504 evaluations rather than 1,000, consistent with top-N sorter admissions. H2: refuted: one traced MATCH page, LIMIT 1000 OFFSET 0, zero stem-to-plain retries. E is therefore inapplicable. H3: rank/full-fanout work costs about 0.17 s here; it is a residual floor, but does not explain the 1.67 s P5 cost or prevent the one-second gate.

The Q2 call traced three top-level statements (375,570 including internal FTS trace entries), with one ranked page. Pages and fallbacks were counted from execution, not inferred from code.

## Phase 1

Each cell: median [min, max] seconds, n=5, 8 s caller budget, exactly one busy in-process Python thread or idle. Candidates interleaved within each repeat. A and B implementations are unchanged from the preceding convoy harness (function/class AST equality recorded). D materializes the exact ranked key window before fetching content and the same snippet MATCH by rowid. Candidate limits and ranking inputs are unchanged.

| Query | Candidate | Idle s | One busy thread s | Busy timeouts / 5 | SQL rows idle / busy |
|---|---|---:|---:|---:|---|
| Q1 | D | 0.223 [0.213, 0.240] | 8.012 [8.009, 8.030] | 5 | [1000] / [0] |
| Q1 | D+B | 0.218 [0.212, 0.233] | 1.062 [0.950, 1.187] | 0 | [1000] / [1000] |
| Q1 | A+B+D | 0.222 [0.213, 0.240] | 0.412 [0.407, 0.435] | 0 | [1] / [1] |
| Q2 | D | 0.238 [0.231, 0.241] | 8.017 [8.012, 8.027] | 5 | [1000] / [0] |
| Q2 | D+B | 0.236 [0.225, 0.239] | 1.008 [0.866, 1.334] | 0 | [1000] / [1000] |
| Q2 | A+B+D | 0.241 [0.229, 0.242] | 0.433 [0.411, 0.439] | 0 | [1] / [1] |
| Q3 | D | 0.199 [0.192, 0.205] | 8.012 [8.010, 8.019] | 5 | [1000] / [0] |
| Q3 | D+B | 0.199 [0.188, 0.202] | 1.062 [0.867, 1.151] | 0 | [1000] / [1000] |
| Q3 | A+B+D | 0.201 [0.192, 0.208] | 0.406 [0.383, 0.434] | 0 | [1] / [1] |
| Q4 | D | 0.192 [0.188, 0.201] | 8.020 [8.017, 8.043] | 5 | [1000] / [0] |
| Q4 | D+B | 0.193 [0.189, 0.199] | 1.043 [0.697, 1.314] | 0 | [1000] / [1000] |
| Q4 | A+B+D | 0.196 [0.192, 0.202] | 0.412 [0.392, 0.427] | 0 | [1] / [1] |

Only A+B+D qualifies. D retains the callback convoy; D+B retains costly Python row fetches. No E was implemented. The sole qualifying combination is also the simplest measured passing combination.

## Final product

Actual committed product methods, with no prototype installed. Same corpus, queries, n and busy-thread protocol. Base includes #1745: Q4 now takes its accepted literal-first path and returns one hit; Phase 1 used the prior base and broad OR path. Final identity compares the same 51 inputs against the frozen stock statement on the current base, preserving that identifier routing. Q1-Q3 retain the broad search path.

| Query | Product idle s | Product one busy thread s | Busy timeouts / 5 | SQL rows idle / busy |
|---|---:|---:|---:|---|
| Q1 | 0.226 [0.218, 5.705] | 0.420 [0.412, 0.425] | 0 | [1] / [1] |
| Q2 | 0.245 [0.240, 0.449] | 0.431 [0.407, 0.449] | 0 | [1] / [1] |
| Q3 | 0.202 [0.201, 0.206] | 0.402 [0.369, 0.418] | 0 | [1] / [1] |
| Q4 | 0.001 [0.001, 0.011] | 0.197 [0.166, 0.214] | 0 | [1] / [1] |

| Candidate | Ordered IDs | Full payload hashes | Forced drain max, s | Slots restored |
|---|---:|---:|---:|---|
| D | 51/51 | 51/51 | 0.145 | 4/4 |
| D+B | 51/51 | 51/51 | 0.083 | 4/4 |
| A+B+D | 51/51 | 51/51 | 0.104 | 4/4 |
| Final product | 51/51 | 51/51 | 0.090 | 4/4 |

Every recorded observation restored all four worker slots. Five forced-deadline trials use a 0.25 s caller budget, with the burner active through drain. Phase 1 uses broad Q4; final product uses Q2 because current-base Q4 is selective. All five final trials timed out as intended. The idle Q1 range retains a 5.705 s outlier; its cause was not proven. Final maximum drain: 0.089693 s. No full-text worker or joined watchdog remains when each observation ends.

## Design and invariants

The exact WHERE/ORDER BY/LIMIT/OFFSET applies to a MATERIALIZED keys-only window; only its rows load all message fields and the same snippet() call over the same MATCH. Explicit ordinals retain window order. One JSON aggregate crosses into Python, with tagged round-trip REAL values preserving binary64 ranking/timestamp bits. Python directness adjustment, page growth, candidate cap, and final sorting are unchanged.

The caller's worker.join ceiling remains. One daemon watchdog repeatedly interrupts only the helper's private message/DAG connections after expiry, stopping in finally and serializing interrupt with close; closed-connection ProgrammingError is tolerated. SQLITE_INTERRUPT is terminal before stem/plain/LIKE retries. Non-interrupt fallbacks, exclusions, immutable RetrievalContext binding, query_only, disclosure vocabulary, recall arm, LIKE path and excerpt flag remain unchanged.

## Local proof

- Exact-base regression RED: 7 failed in 0.83s. Candidate GREEN: 7 passed in 0.84s. Deterministic counters cover bounded materialization, one-row Python fetch, terminal interrupt, no hot progress callback, and frozen-stock full-payload identity with ties, directness adjustment, multiple offsets, filters and exclusions.
- Existing focused recall/context suite: 403 passed in 4.79s.
- Initial full suite: 6 failed, 15846 passed, 137 skipped, 64 xfailed, 23 warnings in 233.03s. Five failures were SQL golden expectations. Regeneration changed exactly three SQL digest events, from 810 to 3552 characters; returned-state hashes and every other event are identical. The remaining embedding-maintenance drain timeout passed in isolation on both base and candidate; its initial cause was not proven.
- Full suite after fixture correction: 15852 passed, 137 skipped, 64 xfailed, 23 warnings in 355.63s. Neither #1429 nor #1744 occurred; no rerun was attributed to those issues.
- Ruff 0.15.13: All checks passed! git diff --check: exit 0, empty.
- Product: 71 added + 22 deleted = 93 changed lines. Tests/reference fixture: 332 added lines. Generated golden: 3 added + 3 deleted lines. No 200-product-line stop was crossed.

Not proven: independent review, exact-head remote CI, two-CPU Linux floor/ON semantic coverage, the full #1586 acceptance, merge, release, runtime safety or customer readiness.

T1/T2: unpublished. Public PR evidence is a bounded receipt, not private bench-evidence T2 publication.

Exact re-score command (from this run root): `python SCORER.txt`.
