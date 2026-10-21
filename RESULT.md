# Media preview acceptance: 20261021-media-preview

| Gate | Result | Receipt |
| --- | --- | --- |
| New issue regressions on base | 19 expected failures; four prose controls pass | red.log, base.log |
| New candidate regressions and controls | 24 passed | final-focused.log |
| Existing #1718 corpus | 128 passed | shapes-green.log |
| Single 32 MiB bytes repr | peak 234,906,080 -> 34,496 bytes; caption kept | memory.json |
| Two-part 32 MiB aggregate repr | peak 218,129,202 -> 44,594 bytes; both captions kept | memory.json |
| Differential | 367 common shapes: 150 unchanged; 196 repr/byte removals; 14 recovered missing previews; seven metadata removals | differential.json |

The two-part recovery measurement uses two 16 MiB literals to stay below the unchanged 64 MiB source ceiling. A separate control verifies two 32 MiB literals still produce an empty preview above that ceiling.

Re-score: read memory.json and differential.json. Reproduce with `python -m pytest tests/invariants/test_issue_1733_media_preview.py -q -s -p no:cacheprovider` at the recorded candidate SHA. Baseline is fe209618fabc513e71b1a7f42b03fdcfe2adeaaf.

Not proven: merge, release, runtime or customer readiness; no payload storage changes. T1/T2 publication is outside this preview-fix specification; raw local evidence is preserved.

Candidate source SHA: bc8c0c51f3ea43c65b889adf279d24b8c4fdb8bd.

Full suite: `15930 passed, 137 skipped, 64 xfailed, 23 warnings in 264.66s (0:04:24)`; exit 0. Neither known local-only failure occurred. Pinned Ruff 0.15.13: `All checks passed!`; diff check exit 0.
