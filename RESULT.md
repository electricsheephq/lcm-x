# S1-C.4 exact-host dependency qualification
Claim class: advisory for these synthetic acceptance receipts; PR readiness additionally requires exact-head full hosted CI.

PR head `4536f171bb4ffaf95d3ff186eb6f11e4706b54e5`; executed merge tree `0f86de4520abb672295882e430022ecfe4b769d9`, parents base `849da257b52fc9833955928816dd1585184988b1` and PR head.
Run: https://github.com/electricsheephq/lcm-x/actions/runs/40092657328
PM host `6f7a7991bb069db07ae74a479823ce8310f8c7e0`; reference `f97608f178d1ffeca59860195ab7da295f7c8e5f`. Exact-built venvs, no foreign seed.

| Receipt | State |
| --- | --- |
| Reference native a/b | PASS; cells.json and pytest.stdout |
| Reference installer a | PASS; cells.json and pytest.stdout |
| Reference c/d/e/f | N/A: accepted legacy semantics, no PM shared generations/consent |
| PM installer a/d/f | PASS; cells.json and pytest.stdout |
| PM native a/c/d | PASS; cells.json and pytest.stdout |
| PM native b scanning-ON refusal | EXPECTED_NEGATIVE; 051.stdout and cells.json, installed version/state unchanged |
| PM native b unattended core version update | PASS; 054.stdout; version999.1186.1 published, actual interactive=False call |
| PM native e core-version no-rebuild | PASS; b and e-core-version-no-rebuild identical before/after stamp/environment |
| Scanner restore then companion scan | PASS; 055.json restore in finally, 056.stdout scanner_enabled=true and verdict=safe |
| PM native e dependency rebuild N>=2 | PASS; one new generation per range edit; rebuild-measurements.json |
| Every companion route scanner | PASS ON/safe; includes declined/noninteractive installs and post-update |
| Both approved SHA256-locked tool archives | PASS; both PM arms' locked-tools cells |

Both scanner exceptions apply only to synthetic CI core operations: native core install (S1-C.3), and PM core version-only unattended update b (S1-C.4), each restored in finally before companion work. Companion scanning and host dependency consent remain intact. The default full-core scanner limitation remains separately tracked in [#1946](https://github.com/electricsheephq/lcm-x/issues/1946); no default core scanner/catalog admission claim.

PM unchanged stamp: `f2cc435d7772e5303c63ac4d4fc2c800e0ee2ddfa56411e5ec4a58f370b2453f`. Selected environment and Python, committed flag and state directory also compare equal.

Two sequential warm-package-cache shared-generation rebuilds, upper range0.99 then0.98. Full selected host graph, not additive fastembed size, per-profile copies or cold-network SLA. Baseline/CI/judge: N/A.

| Trial | Wall seconds | Allocated generation bytes | Net scratch delta bytes | Retained held old leases |
| --- | ---: | ---: | ---: | ---: |
| 1 | 9.256448911999996 | 418062336 | 117006336 | 1 |
| 2 | 9.898763099000007 | 418062336 | 117006336 | 2 |

Mean 9.577606005500002 seconds, N=2; each generation0.38935GiB, below600s/3GiB. Retained old generation directories existed with held kernel leases; released by processes at teardown, no generation deleted by this lane.

Raw inputs: `arms/*/cells.json`, command stdout/stderr/metadata, `arms/dependency-1186-upstream-main-native/rebuild-measurements.json`.
Re-score command: `python3 rescore.py`.
Not proven: full current-head CI from this receipt alone; default core scanner/catalog admission, cold-cache/network rebuild cost, default-on semantics, model readiness, release, deployed runtime or customer readiness.
T1: not published; this is slice qualification, no release score row.
T2: incomplete; tags/releases are forbidden by this lane. All raw results and re-score inputs remain preserved; nothing packed, moved or deleted.
