# W1b r5c mutation and contract receipt - PR #1926

Claim class: advisory synthetic default-off entity-bridge evidence. This is not independent merge, release, runtime or customer acceptance.

The entry product is `2df9ef1192dd32b18048513679c4ebc31b4f5b6b`; the tested candidate adds only the current-relation read guard (five product lines).
R5-F1 is RED on that entry product and GREEN with the guard. Under R5-Q, F2 is a next-rebuild liveness contract and is already GREEN on entry.
The ambiguity-removal test requires immediate None, resolution after a healthy-sender rebuild, and None after a dropped-sender rebuild.
A further contract test prevents an explicitly scoped override from classifying an ambiguous alias slot as admitted.

## Tested inputs

- `entity_ops.py` SHA256: `d25f5842d6bc28a04f82e2516b8e3cdfb92c7398c9a4bc49cd46d5ebe29dd5f2`.
- `entity_store.py` SHA256: `ca769c5bd61663ba7c7109e04958077c40a788de4d1faf857d3baae14bcd9613`.
- `tests/test_entity_ops_stateful.py` SHA256: `fe709beb57fdbe54e7aa12d99b68e9b5be78a518b2f9df90f863f936c8f15a81`.

Profiles: ci max_examples=25 / stateful_step_count=20; nightly max_examples=500 / stateful_step_count=30. Both retain deterministic generation, no example DB and no explain phase.
Both copied-store and empty-derived-index references remain. Observed non-None answers must equal each reference everywhere.
No-override liveness applies to slots that resolved unambiguously at the last rebuild; previously ambiguous slots may remain None until another rebuild.
The test reuses observed answers across the two references and completed checks only for identical fixture writer counters, sources, sender gates and liveness state. Every changed state runs both references.
Recent post-setup undo, the G1 macro, supplied probes, global citation, row-retention and default-off checks remain.

## Six-cell matrix

| Candidate | ci | nightly | Generated machine |
| --- | --- | --- | --- |
| head | `13 passed in 12.97s` | `13 passed in 2923.25s (0:48:43)` | GREEN |
| mA | `11 failed, 2 passed in 9.04s` | `11 failed, 2 passed in 19.78s` | RED on both |
| mB | `11 failed, 2 passed in 11.48s` | `11 failed, 2 passed in 6.01s` | RED on both |

mA reverses every `entity_ops.py` hunk from `f3478b82996d21a635f05c4823c3201f2078ecbe`, retaining the r5c relation-support guard.
mB replaces undo's derivation assignment with `derivation = []`, retaining every other fix.
Both mutations execute in the imported module's memory; production files are never changed by the runner. Each RED has assertion failures, zero setup errors, and includes `test_entity_bridge_stateful` itself.
- mA product SHA256: `0041356e238d7986861bd8af1cdf88074ef2924bf013da4e9deb2ff24bffe131`.
- mB product SHA256: `2abc86b542405fa1a2eeba87ab5be3ee2ce3d83262bfc17b942fb3f4a5baac19`.

## Commands and other gates

`$W1B_PYTHON` denotes the lane interpreter; `$W1B_SCRATCH` and `$W1B_EVIDENCE` denote synthetic temporary stores and retained outputs. Expanded commands and XML are preserved with the result card.

```sh
LCM_HYPOTHESIS_PROFILE=ci $W1B_PYTHON -m pytest -q -p no:cacheprovider tests/test_entity_ops_stateful.py --basetemp=$W1B_SCRATCH/pt-head-ci -o tmp_path_retention_policy=failed --junitxml=$W1B_EVIDENCE/head-ci.xml
LCM_HYPOTHESIS_PROFILE=nightly $W1B_PYTHON -m pytest -q -p no:cacheprovider tests/test_entity_ops_stateful.py --basetemp=$W1B_SCRATCH/pt-head-nightly -o tmp_path_retention_policy=failed --junitxml=$W1B_EVIDENCE/head-nightly.xml
```

For mA/mB, the same pytest arguments run after `exec(compile(mutant_source, module.__file__, "exec"), module.__dict__)` in `hermes_lcm.entity_ops`; pytest is then invoked in that process.
- Entry contract control (F1 RED; F2 and scoped-ambiguity contracts GREEN): `1 failed, 2 passed in 1.16s`.
- All r4 entity, compat, supplied probes and inventory census targets: `1397 passed in 137.60s (0:02:17)`.
- Fourteen-file invariant set: `321 passed, 6 skipped, 2 xfailed, 3 warnings in 55.32s`.
- Ruff 0.15.13: `All checks passed!`; `git diff --check`: exit 0.

The first r5c oracle tracker incorrectly counted scoped-override resolution as alias admission. Its interrupted nightly run is unscored and preserved; this matrix uses the corrected tracker. The correction is an oracle error under R5-Q, not a third product class. No third product counterexample appeared in the completed matrix.
Not proven: independent semantic acceptance, complete invariant-catalogue coverage, agent-facing behavior, recall gains, merge/release/runtime/customer readiness. No quality/performance score is claimed.
