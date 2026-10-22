# e1b-s0-resume2-acceptance

## Headline table

| benchmark | metric | value | n | baseline | receipt |
| --- | --- | --- | --- | --- | --- |
| E1b inventory | passed | 4 | 4 | repaired contracts | resume2-four-guards.txt |
| E1b true BASE | expected RED | 62 | 62 | 5df5a3c2736c01b46acf26906a510e41e367e73e | resume2-red-base.txt |
| E1b candidate | passed | 62 | 62 | same cases as BASE | resume2-green-acceptance.txt |
| E1b product-only reverse | expected RED | 62 | 62 | same cases, tests retained | resume2-red-product-revert.txt |
| Consumers/PIN/DECLARE | passed | 932 | 932 | unchanged pinned fixtures | resume2-consumers-pins.txt |
| Full suite | passed, zero failures/errors | 19,940 | 20,147 selected | serial CI-style local exception | resume2-full-suite.txt |
| Ruff 0.15.13 | passed | — | — | frozen candidate | resume2-ruff.txt |
| actionlint 1.7.12 | passed | — | — | frozen candidate | resume2-actionlint.txt |

Measured source: HEAD `04627819e6599935ac38a62e6ea0d8e2fa213eb0` plus dirty
working candidate SHA256 `554c626a2a7ce760577118343a5fe71ca4999241b84ee6d25080e8eaccb7daf3`.
`resume2-candidate-frozen.json` names all 33 BASE-to-candidate files and hashes.
The working candidate was not committed when measured.

Full-suite actual terminal statistics: 19,940 passed, 141 skipped, 66 xfailed,
38 deselected, 29 warnings; exit 0. The structured pytest terminal summary is
preserved exactly; the human-readable duration line was outside the runner's
allowlist because it included deselections, so no elapsed-time number is claimed.
The inherited isolated compat gate reports 254 passed and 27 xfailed; its prior
receipt was not rerun in resume2 (`compat-pin-inherited.txt`).

## Run command

```sh
python -m pytest --assert=plain --tb=no --disable-warnings -q -rA -p resume2_receipts -m "not latency" -o tmp_path_retention_policy=failed --basetemp="$SCRATCH/pt-resume2-full-suite"
```

## Re-score command

```sh
python score-acceptance.py .
```

## Not proven

Not proven: live Hermes sender correctness, native inbound verification,
semantics of admitted turns, remote CI, independent semantic review, merge,
release, runtime or customer readiness. Unit/integration/compat evidence only.

## Links

T1/T2 publication is root-owned and pending. No results or scratch were deleted.

