# B8 recognition extraction

This refactor moves decomposition and occurrence matching into
`recognition_matching.py`, and emission descriptor finalization and projection
into `emission_projection.py`. Executable function bodies stay unchanged;
annotations describe the existing heterogeneous row and proof dictionaries.

This is one extraction, not two. Identity recognition reads emission projection
through the `reconcile._project_emitted_occurrences` seam, so the two modules move
as one unit, and `tests/test_recognition_extraction_contract.py` pins that joint
seam. Both new modules are turn-path files: they are listed in `TURN_PATH_FILES`
(differential workflow and `scripts/maintainer_gate.py`) and in the I1, I2 and
I10 rows of `docs/invariants.md`.

The original import paths in `identity_anchor` and `reconcile` remain available.
In particular, `reconcile._project_emitted_occurrences` remains the lookup seam
for reconciliation and the lazy import in identity recognition. The existing
consumer-switch test patches that seam. Matching warnings retain the
`hermes_lcm.identity_anchor` logger name. Budget monkeypatches now target
`recognition_matching`, where the algorithm reads the constants.

## Characterization and invariants

`tests/test_recognition_extraction_contract.py` pins ordered decomposition,
matching augmentation, NULL reservation, descriptor bytes, duplicate occurrence
handling, slicing, immutable retained-source metadata, and rejection of unproven
emissions. The existing matching brute-force tests and emission descriptor and
consumer-switch tests remain the integration checks.

These cover the moved portions of I1 (occurrence identity), I2 (durability before
removal), and I10 (projection identity). They do not establish the full invariant
matrix or a release gate.

## Rebase and landing notes

- Draft only until the post-GA refactor window, after #1230 → #1243 and #1239.
- Product-first ordering still applies to #1114, #1117, #1146 and #1204;
  preserve the original import paths used by their consumers.
- #1257 changes the adjacent identity-anchor import seam; keep both changes.
- #1280 rewrites `tests/test_issue_436_matching.py`, including the work-budget
  monkeypatch region. When rebasing, retain its test improvements and target
  `_MATCH_WORK_PER_ITEM` and `_MATCH_WORK_FLOOR` on `recognition_matching`.
- Recheck overlap and exact-head tests before landing. C4 frontier ownership,
  runtime changes, releases and any behavior fixes are outside this extraction.
