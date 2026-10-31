# Fix scope: what a fix is compared with, and how far it must reach

Six v0.27.0 release candidates were respun. Per-PR review caught none of the causes, and four were cases the previous GA handled
and the candidate did not. Four of those were side effects of an earlier fix. These rules address that pattern.

## Baselines
- **Regression baseline = the previous GA, never the previous rc.** A fix is judged by whether the candidate handles every input
  the previous GA handled. A comparison against the previous rc only shows the fix's own delta, and that is how a narrowed fix
  shipped in v0.27.0-rc6 and needed rc7.
- **Refactor baseline = the immediate pre-refactor commit, byte-identical.** See [refactor-window.md](refactor-window.md).
- **Absolute regression cases are kept across GA rotations.** When the GA baseline moves, archive the deltas that were accepted
  against the old GA. Keep absolute cases such as the #1225 residual, so a known gap never becomes "parity with the GA" by rotation.
- The tool is `tests/differential` (`LCM_DIFFERENTIAL=1`). A fix PR on the turn path cites its differential row.
- `differential` is a required status check on `main`. It reports on every PR. When a PR touches a turn-path file
  (`scripts/maintainer_gate.py` `TURN_PATH_FILES`), `tests/differential/` or `.github/workflows/differential.yml`, the full
  differential runs and any unexplained red fails the check. Otherwise the job passes after the relevance step.

## Escaped regressions become permanent checks
- A fix for a regression that escaped to an rc, a GA or a user, on the turn path (assembly, ingest, compaction, reconcile), adds a
  permanent check in the same PR, in addition to its unit test: a `tests/differential` case (a pinned shape or a generated family) or an
  invariant under `tests/invariants`. The check is red on the code before the fix and green after, and the PR body cites both runs.
- A fix confined to a pure helper, with no turn-path behaviour, keeps the unit test only.
- Why: of the 222 fixes merged in the 60 days to 2026-10-17, 168 are guarded today only by a unit test, and the differential
  and invariant suites would catch 3 and 0 of them. A unit test pins the reported shape; the differential and invariants also
  cover its neighbours, which is where the v0.27.0 respins came from.

## Sibling sweep
After any Phase B finding, sweep the whole sibling shape family before the next rc, not only the reported shape: role × tool
topology (parallel batch, sequential pairs, mixed) × row size (around 256 tokens and the 1,200/2,400-character projection
edges) × window (16K to 1M) × summarizer (healthy, outage) × configuration (defaults, fleet overlay). #1220 (tool rows, parallel)
and #1227 (assistant rows, sequential) were one family found one respin apart.

## Spec amendments that narrow a fix
A spec may name the existing tests it changes and justify each change. An amendment that narrows a fix so that old tests "pass
unchanged" must show the same reachability evidence as the fix itself, for every shape it drops. The rc6 → rc7 respin came from
such an amendment.

## Default flips
- v0.28.0: no product-default flips. The planned local-embeddings flip (#1186) moved to v0.29.0.
- v0.29: one planned flip, local embeddings on by default (#1186), under the same per-flip rules as below.
- From v0.30: at most 2 default flips per release. Each one carries a differential row and a pinned row in
  `tests/test_issue_1013_product_defaults.py`, and lands in its own PR.
- A test that pins an old default needs a twin at the new default only when a PR touches that test file. There is no sweep of
  untouched files.
- Rationale: v0.27.0 changed 11 defaults across 5 PRs, and the fallout (#1139, #1221) appeared only at shipped defaults on
  particular windows.

## Checklist for a fix PR
- [ ] The previous-GA differential row for the fixed shape (red before, green after).
- [ ] The sibling family swept, with the cells listed.
- [ ] The invariant IDs touched ([docs/invariants.md](../invariants.md)).
- [ ] Every changed existing test named and justified in the PR body.
