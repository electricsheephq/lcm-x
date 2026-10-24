# S1-C.5 dependency qualification — COMPLETE

Claim class: advisory. Four fresh exact-pin hosted arms at PR head `c8de1a45ec8bdea027a50c8713eee85528de7165` and execution merge `91bab9a07fdc4d3883fee78020e0840012db446f` (parents `796b755e94e0cd45b2e349d04ba77f81be02c453` and PR head). Host pins PM `6f7a7991bb069db07ae74a479823ce8310f8c7e0`, reference `f97608f178d1ffeca59860195ab7da295f7c8e5f`.

| Receipt | Result |
| --- | --- |
| [arm dependency-1186-ref-0.21.5-installer](arms/dependency-1186-ref-0.21.5-installer/cells.json) | 13 passed in 88.96s (0:01:28) |
| [arm dependency-1186-ref-0.21.5-native](arms/dependency-1186-ref-0.21.5-native/cells.json) | 13 passed in 53.83s |
| [arm dependency-1186-upstream-main-installer](arms/dependency-1186-upstream-main-installer/cells.json) | 13 passed in 239.90s (0:03:59) |
| [arm dependency-1186-upstream-main-native](arms/dependency-1186-upstream-main-native/cells.json) | 13 passed in 330.76s (0:05:30) |

Permanent cells a-f retain their route/legacy N/A scope. PM installer additionally proves actual named-profile install, root config unchanged, core/carrier loaded/imported and scanner ON/safe. Both PM routes require the actual host-owned hint; deletion and Rich-wrapping regressions are checked in the test file.

| Companion range edit | Wall seconds | Allocated generation bytes | Net scratch delta bytes | Held old leases retained |
| --- | ---: | ---: | ---: | ---: |
| upper0.99 | 8.225521084999968 | 418062336 | 117006336 | 1 |
| upper0.98 | 8.752104301000031 | 418062336 | 117006336 | 2 |

Source: [measurement rows](arms/dependency-1186-upstream-main-native/rebuild-measurements.json). N=2, warm package cache, full exact host graph; baseline/comparator/judge/confidence interval N/A. Each edit creates one new generation and retained old kernel-held leases are verified. Core version-only update preserves full selected PM state/stamp. Scanner-ON core refusal remains EXPECTED_NEGATIVE; exceptions scoped only to synthetic hosted core operations, restored before companions.

Re-score: `python3 rescore.py`. Raw outputs were downloaded into this run before scoring. No model download. Not proven: cold-cache/network rebuild SLA, additive fastembed size, default-on/model readiness, default full-core scanner/catalog admission, merge/release/deployed-runtime/customer readiness.

T1: incomplete, no release-score row in this slice. T2: incomplete, lane forbids tags/releases. Results and re-score inputs remain preserved; no cleanup of result files. Hosted run: https://github.com/electricsheephq/lcm-x/actions/runs/40096462964
