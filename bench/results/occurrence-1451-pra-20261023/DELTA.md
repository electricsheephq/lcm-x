# Legacy-system correction after initial acceptance

Correction source: `9538c8387bba578db127729697f428b38a9d5551`. Total product delta against requested BASE: 120 lines. Only the `output_length`-absent legacy-validation branch changes; writer, transaction, wire, digest twins and lifecycle timing are unchanged.

A valid old snapshot can contain a descriptor-less leading system anchor carrying the LCM annotation. Its null-identity positions are `[0, 1]`, while its emission positions are `[1]`. The initial candidate incorrectly required equality. Validation now requires all known emission positions and permits only the optional leading null position 0. All field types, effective digest sequence, exact length for new snapshots, and rejection of extra trailing null rows remain checked.

| Focused check | Before correction | After correction |
|---|---|---|
| `test_legacy_snapshot_with_leading_system_anchor_attaches` | `1 failed in 2.86s` | `1 passed in 1.55s` |
| `pytest tests/test_issue_1451_atomic_snapshot.py -q -k legacy` | — | `6 passed, 14 deselected in 0.85s` |
| ruff 0.15.13 | — | `All checks passed!` |
| `git diff --check` | — | exit 0 |

Matrix coverage remains measured at `0eb95e50f4bae6ef0bcc305efde49c83f3d41fca`: scanning every preserved cell finds 5,184 non-null snapshot records, all with `output_length`. Missing snapshots skip validation; malformed metadata returns before the changed branch. Therefore this legacy-only delta does not change any of the 1,296 measured paths. The full matrix was not rerun. The cost path/writer and candidate-written rollback record are also unchanged; their original measurement identities remain explicit.

The initial independent acceptance review passed at `ba6bee3530e24a0459236cec359781715acf7dbf`. An independent adversarial review identified this supported legacy case; its mandatory preservation failure is fixed by the delta above. Targeted independent delta review and new-head remote CI are tracked by the PR. Prior acceptance coverage is retained for unchanged atomicity/timing code.

Not proven: a fresh full matrix/cost run on this delta, live-host incidence, fleet impact, merge, release or runtime/customer readiness. No new loss cell or rollback-reader failure was demonstrated; the correction fixes valid-evidence rejection before downstream ingest.
