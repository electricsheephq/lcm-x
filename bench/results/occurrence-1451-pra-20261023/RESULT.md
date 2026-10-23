# #1451 PR-A synthetic acceptance

Run `lcmx-1451-pra-20261023`. Claim class: advisory synthetic source evidence.
Product measured at `0eb95e50f4bae6ef0bcc305efde49c83f3d41fca`; compatibility and cost at `e6e4f0e5586245e0cf87bb0fec52745c82f17fd0` (only rollback-test setup differs). Product: 118 changed lines. No DDL or carry-rebind changes.

| Gate | BASE | Candidate | Receipt |
|---|---|---|---|
| New shape/write-boundary regressions | 14 failed in 2.16s | 19 passed in 1.92s (adds legacy cases) | `tests/test_issue_1451_atomic_snapshot.py` |
| Matrix cells | 1,296 | 1,296 | SUMMARY.json |
| Error cells | 172 | 0 | SUMMARY.json |
| Cells with any checkpoint loss | 622 | 594 | SUMMARY.json |
| Duplicate cells | 482 | 482 | SUMMARY.json |
| Restart-growth cells | 298 | 298 | SUMMARY.json |
| New/increased loss checkpoints | — | 0 | SUMMARY.json |
| Changed cells | — | 176; all listed | CHANGED-CELLS.md |
| Checkpoint loss improvements | — | 122 cells | CHANGED-CELLS.md |
| Duplicate increases / positive-growth increases | — | 0 / 0 | CHANGED-CELLS.md |
| v0.28.1 rollback reader | — | 1 passed in 1.52s | `tests/compat/test_occurrence_snapshot_rollback.py` |
| Proof-commit cost, median ms | 1.5960415 | 1.4105205 | cost.json |
| Paired cost delta, median ms (n=200) | — | -0.200084, limit +0.15960415; PASS | cost-samples.json |
| ruff 0.15.13 | — | All checks passed! | CI lint + focused local run |
| Diff whitespace | — | git diff --check: exit 0 | candidate diff |

The supplied matrix baseline is `5df5a3c2736c01b46acf26906a510e41e367e73e`; requested implementation BASE is `10407e12e33c1d7b09fe0e05141153555aec5d62`. The intervening target-file delta is an unrelated ingest-ignore optimization; proof publication/reading is unchanged. This report compares the supplied baseline at every checkpoint and does not claim a fresh full matrix on the newer BASE. Remaining inherited loss, duplicates and growth belong to PR-B; they are not relabeled as passes.

The inherited runner/scorer/oracle are reused. Changes are output/scratch roots, stricter disk checks and an assertion identifying the imported engine file. The PR-B candidate flag remains off because it requires excluded carry rebinding. Raw snapshots were preserved before scoring; SUMMARY.json records their digests. Independent scoring recomputes all row multisets and compares loss, duplicates and ingest errors at every checkpoint plus positive restart growth.

The rollback test uses existing `tests/compat/helpers.py::run_old` and `tests/compat/conftest.py::old_tree`, pinned to v0.28.1 `77b2b4f795fb6a4f84c55abf6d0441da4469971e`. It opens a candidate-written store, verifies snapshot attachment, ingests a new user row, replays without growth, preserves the original prefix, and checks SQLite integrity. An initial fixture admission failure occurred before the old reader ran; deterministic summarization and leaf size corrected it.

Cost: 20 warmup pairs, then 200 pairs in alternating AB/BA order; separate processes import the two trees and execute one sample at a time. Only `_record_compress_commit_proof` is timed, including its durable writes, on the same deterministic compaction fixture. One execution; no cross-machine or live-turn claim.

Testing improved: malformed-shape and per-write-boundary fault regressions, a v0.28.1 rollback-reader check, and a per-cell matrix comparison against BASE.

Four ingest golden scenarios (compaction, host-merge, rotation, rewrite) change only the fingerprints containing side-snapshot `output_length`. Removing just that field restores every prior event, including logs, SQL, row state and outputs. Historical native fixtures keep their proof-only writer. A failed publication now clears memory proof, so its existing test checks durable prefix retention and a single fresh row while allowing bounded replay duplicates.

Re-score: the preserved full result set runs `python compare.py`; it reads `inputs/red-base.jsonl` and `candidate.jsonl` through the inherited scorer. This public directory is a summary; complete raw inputs and attribution payloads remain in the retained run packet. T2 tag reserved: `lcm-x/lcmx-1451-pra-20261023`; T2 is unpublished. No results were disposed.

Not proven: live-host incidence, fleet impact, release, runtime/customer readiness, PR-B correctness, or absence of inherited occurrence defects. Remote full-suite CI and independent review are tracked by the PR separately.
