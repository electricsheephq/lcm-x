# e1b-s0-ci313-fixture-supplement

## Headline table

| benchmark | metric | value | n | baseline | receipt |
| --- | --- | --- | --- | --- | --- |
| Schema positive + T7 contention | passed | 3 | 3 | Python 3.13.14 | resume2-ci313-focused.txt |
| Corrected 62-case set on BASE | expected RED | 62 | 62 | BASE 5df5a3c | resume2-ci313-red-base.txt |
| Corrected 62-case candidate | passed | 62 | 62 | same 62 nodeids | resume2-ci313-green.txt |
| Corrected 62-case product reverse | expected RED | 62 | 62 | tests retained | resume2-ci313-red-product-revert.txt |
| Restored product / pinned fixtures | exact hashes | 8 / 7 | 8 / 7 | original candidate / BASE | resume2-ci313-restoration-verified.json |
| Ruff 0.15.13 | passed | — | — | corrected test + unchanged product | resume2-ci313-ruff.txt |

This supplement measures HEAD `cd7b57d041aad4d4d598c4c967e3efd4367b0f96`
plus one uncommitted insertion in `tests/test_sender_columns_schema.py`.
The exact file hashes are in `resume2-ci313-freeze.json`.
The correction joins the existing background workers after derived setup and
before baseline snapshots. Every strict expected-1, total-changes and later
expected-0 assertion remains unchanged. Production and benchmark bytes remain
unchanged; the earlier cost card remains the cost evidence.

## Run command

```sh
python resume2-ci313-run.py <label> <named-test-selectors>
```

## Re-score command

```sh
python score.py .
```

## Not proven

Not proven: the exact swallowed cause of the original Python 3.13 CI return-0,
final-head full-matrix CI, merge, release, live-host sender correctness, admitted
turn semantics, runtime or customer readiness. This is not a proven-flakiness
claim. The prior local full suite used the prior fixture; no new full local
suite or cost run was performed. Root owns the next commit, push and final CI.

## Links

Supplemental public receipt publication is root-owned and pending. All older
published cards and failed-CI evidence remain preserved. No result or scratch
deletion occurred. Candidate `e1b-s0-1`, source corrections used: 3.

