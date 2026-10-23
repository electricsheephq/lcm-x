# Occurrence revert proof

| gate | observed result | receipt |
|---|---|---|
| full reachable matrix | 1,296 cells completed; zero cells worse than main on loss | three-way-comparison.json |
| named regression cells | all 25 equal GA on loss at every checkpoint | comparison.md |
| duplicates versus main | 190 cells increase; every increase equals GA | comparison.md |
| restart growth versus main | 4 cells increase from 0 to 2 at restart 1, matching GA | comparison.md |
| product regression on BASE | 1 failed (expected negative: c/d tool results absent) | red-regression.log |
| product regression and retained tests on revert | 5 passed in 1.01s | green-regression.log |

Matrix product source: 07dd79d185b1e4968f5e3d94267e5417a4f2c26c. BASE reproduction: cc063de8599f0367c504be9045f3d6b152df8e77. Preserved main comparison: 5df5a3c2736c01b46acf26906a510e41e367e73e. GA: v0.28.1, 77b2b4f795fb6a4f84c55abf6d0441da4469971e.

The matrix scorer independently rechecked raw host-ledger snapshots using the existing identity projection and multiset oracle. This public packet contains checkpoint counts and test output; full raw synthetic ledgers and scorer inputs are preserved separately. T2 raw publication is incomplete; no result data was removed.

All increases over main in duplicates or restart growth match GA. Inherited cells occ-0296–0301 and occ-0304–0309 remain worse than GA on duplicates; the revert does not introduce those increases over main.

Not proven: losslessness of every synthetic history, live-host incidence, fleet impact, merge, release, runtime or customer readiness. The steady no-UID cell still loses two call-envelope rows, as GA does; this revert restores its two tool-result rows.
