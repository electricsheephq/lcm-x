# S1-C.6 dependency qualification — COMPLETE

Claim class: advisory. All four fresh exact-pin hosted arms pass at PR head `9f2e8134608ae3bf8d4149be77fe3451635c7418`; execution merge `161b237acbd26b66d8ed8e2f456b6fd538b8fae5`, parents `1771678475aa0d3ef4426ce56fc3b6f333406c5c` and that PR head. Host pins PM `6f7a7991bb069db07ae74a479823ce8310f8c7e0`, reference `f97608f178d1ffeca59860195ab7da295f7c8e5f`.

| Receipt | Result |
| --- | --- |
| [dependency-1186-ref-0.21.5-installer](arms/dependency-1186-ref-0.21.5-installer/cells.json) | 27 passed in 68.69s (0:01:08) |
| [dependency-1186-ref-0.21.5-native](arms/dependency-1186-ref-0.21.5-native/cells.json) | 27 passed in 98.65s (0:01:38) |
| [dependency-1186-upstream-main-installer](arms/dependency-1186-upstream-main-installer/cells.json) | 27 passed in 233.08s (0:03:53) |
| [dependency-1186-upstream-main-native](arms/dependency-1186-upstream-main-native/cells.json) | 27 passed in 407.99s (0:06:47) |

Every arm also passes 14 installer stub cases executing the real pinned root/profile resolution: profile-shaped home with unset, matching or conflicting profile, root/sticky/custom/default routes, with and without an existing companion. Profile validation/liveness and native verbs in those cells are synthetic; real host admission is covered separately in a-f. Conflict refuses before links. PM f executes the entire printed HERMES_HOME-prefixed command through bash, then proves companion enabled/importable. Both actual host-owned hint phrases remain required.

Permanent cells a-f keep their route/legacy N/A scope. Scanner-ON core refusal remains EXPECTED_NEGATIVE; approved core-only exceptions stay synthetic/hosted and restore scanning before every companion. Companion scans are ON/safe. Core version-only update preserves selected PM state/stamp. Actual PM named-profile installer preserves root config. Only approved hash-locked tools are used; no model download.

| Companion edit | Wall seconds | Allocated generation bytes | Net scratch delta bytes | Held old leases |
| --- | ---: | ---: | ---: | ---: |
| upper0.99 | 9.332705676999979 | 418062336 | 117006336 | 1 |
| upper0.98 | 9.819055683000045 | 418062336 | 117006336 | 2 |

Source: [measurement rows](arms/dependency-1186-upstream-main-native/rebuild-measurements.json). N=2, full exact host graph/warm cache; baseline/comparator/judge/confidence interval N/A. Each edit creates one new generation and retains kernel-held old leases. No result was packed, moved or deleted.

Re-score: `python3 rescore.py`. Raw outputs were snapshotted into this run before scoring. Not proven: cold-cache/network SLA, additive fastembed size, default scanner/catalog admission, model/default-on readiness, merge, release, deployed runtime or customer readiness.

T1 incomplete: no release-score row in this slice. T2 incomplete: tags/releases are outside lane authority. All results and re-score inputs retained. Hosted run: https://github.com/electricsheephq/lcm-x/actions/runs/40099535101
