# S1-C.3 exact-host dependency qualification
Claim class: advisory; PARTIAL, with the PM unattended-update cell blocked by default core scanner.

PR head `4be7e931d4e1a3450114fda4c57ad0ff39e9e761`; executed merge tree `af7994c5d631f0d95e80789069aa3481dd920b2a`.
Run: https://github.com/electricsheephq/lcm-x/actions/runs/40089321305

| Receipt | State |
| --- | --- |
| Reference native a/b; c/d/e/f N/A legacy | PASS |
| Reference installer a | PASS |
| PM installer a/d/f | PASS |
| PM native a/c/d | PASS |
| PM native e dependency rebuild, two range edits | PASS |
| PM native b version-only unattended update | BLOCKED by core development-checkout scanner; no exception applied to update |
| Every companion default scanner and two approved archive hashes | PASS |

## Rebuild measurement
Two sequential warm-package-cache shared-generation rebuilds, companion range upper bound 0.99 then 0.98. Exact PM pin `6f7a7991bb069db07ae74a479823ce8310f8c7e0`, Linux hosted CI. This is full selected host graph, not an additive fastembed size, cold-network SLA or per-profile copy.

| Trial | Wall seconds | Allocated new generation bytes | Net scratch delta bytes | Held old leases |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8.039389928 | 418062336 | 117006336 | 1 |
| 2 | 8.308031679 | 418062336 | 117006336 | 2 |

One new generation per edit. Retained leased generations remained present and kernel leases were observed held; processes released at teardown. Mean 8.1737108035 s, range 8.039389928–8.308031679 s (N=2). Each generation allocated 0.38935 GiB; both below 600 s / 3 GiB. The separate PM core-version-bump/no-rebuild observation remains UNPROVEN because update was refused at scanning.

Published summary inputs: `arms/*/cells.json`, core file/rule metadata, and `arms/dependency-1186-upstream-main-native/rebuild-measurements.json`. Full per-command stdout/stderr/metadata remain in the hosted run artifacts and the retained operator evidence packet; they are not part of this public summary snapshot.
Re-score command: `python3 rescore.py`.
Not proven: PM core update/no-rebuild, all-green hosted qualification, default core scanner/catalog admission, cold-cache/network rebuild cost, model readiness, release, runtime or customer readiness.
T1: not published; this is a partial acceptance snapshot, not a released benchmark registry row.
T2: incomplete; this lane forbids tags/releases. No result, re-score input or bulk was packed, moved or deleted.
