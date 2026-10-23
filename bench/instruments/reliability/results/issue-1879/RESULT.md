# Issue #1879 reliability instrument acceptance

Claim class: advisory. Four conservative instrument regressions only; no product changes.

Measured source: `4ec4d25d8ef08fcd681d0641ba3ffab3b6cbc26d`; r2 baseline: `03e53281065bb560a170a6cb642b8aef8ada5586`.

| Cell | B1 base/candidate | B2 deficits base/candidate | Surplus base/candidate | Class counts base/candidate (PRODUCT_LOSS / INSTRUMENT_DEFECT / HOST_TRANSFORM / UNRESOLVED) | Result |
|---|---:|---:|---:|---|---|
| runtime host 1: publication-failure/rotation-child-persistent | 18/18 | 0/0 | 28/28 | 0/17/0/0 unchanged | FAIL |
| runtime host 2: publication-failure/rotation-child-persistent | 18/18 | 0/0 | 28/28 | 0/17/0/0 unchanged | FAIL |
| runtime host 3: publication-failure/rotation-child-persistent | 18/18 | 0/0 | 28/28 | 0/17/0/0 unchanged | FAIL |
| reference host in-place: continuity/survival-fit/in-place | 0/0 | 0/0 | 0/0 | 0/2/0/0 unchanged | PASS |
| reference host rotation: continuity/survival-fit/rotation | 0/0 | 0/0 | 0/0 | 0/2/0/0 unchanged | PASS |

No move toward PRODUCT_LOSS or UNRESOLVED; no new credit relative to r2 on the same snapshots. Classification scope is the 55 distinct previously diagnosed instrument discrepancies, excluding residual surplus. All 54 B1 entries (18 per runtime host), with key, kind, row hashes, store IDs, host-tag counts/IDs and merge-event witnesses, are in the content-free summary. Every mismatch is excess user-tag row multiplicity involving composite rows; none is a deficit.

Proposed declared issue bars: #1797 `["B2"]`; #1799 retains `["B2"]`. Keep current cell bars and non-gating status pending the release-manager decision. No declared targets or cell bars changed.

Largest cell: 220 stored rows, 160 expected keys, 865 events, 326634 maximum user bytes, three witnesses per key. Assignment has 25 candidates total, at most eight per key, eight units and eight expanded states. Worst-case search at these candidate sizes: 20000 states times eight options times at most 64 overlap checks, or 10240000 interval comparisons, plus option sorting and list copies; stack depth at most 64. General search is O(20000*C*(64+log C)), C=max options per key; memory O(total candidates+64^2). Candidate enumeration is O(R*W*(20000*64*L + P*64*L)) using the existing bounded boundary expansion, R=stored user rows, W=witnesses per row, P=missing keys, L=max bytes; no unbounded search is introduced. Exhaustion retains only explicitly witnessed non-overlapping assignments.

RED at BASE: 32 failed, 9 passed (four requested defects plus fault variants). Candidate: 222 focused/affected passed; broader reliability 487 passed plus 17 environment failures (missing CI agent package), then all 45 P8 tests passed with a scratch-only empty agent package. Six prior negative families and future-witness controls remain permanent unchanged tests. Ruff 0.15.13 passed; diff check clean.

Testing improved: four RED-to-GREEN instrument regressions with paired witness-removed negatives, plus a five-cell base-vs-candidate rescore.

Run command: `python bench/instruments/reliability/run_matrix.py --plugin-ref 4ec4d25d8ef08fcd681d0641ba3ffab3b6cbc26d --jobs 1 --timeout 180 --keep-dbs all --hosts-file <pinned-hosts.json> --hosts <pinned-host> --cells <cell> --out <new-evidence> --scratch-root <new-scratch>`; reference host adds `--transport acp-process --turn-timeout 60`.

Re-score command: `python -m bench.instruments.reliability.scorers.cli --cell-dir <preserved-cell>` under each pinned scorer tree; compare all five preserved receipts. Exact original argv and raw snapshots remain retained with the full local card.

T1: registry row added in this PR. T2 incomplete: raw publication is outside this ready-PR lane; full raw results remain retained, with no cleanup.

Not proven: live-host behavior, historical product preservation, release, runtime/customer readiness, or resolution of the remaining B1 and surplus failures.
