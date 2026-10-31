# Invariant seed receipts

These receipts bind the [catalogue](../invariants.md) to particular historical failures.
`proven` means that the named seed is detected, **not** that every clause, input, or
concurrent schedule in the invariant is covered. Open residuals remain visible.
All runs below used the same rc8 archive, Python 3.13, pytest 9.1, one worker,
no network or model calls, and the CI `ContextEngine` stub. The recorded date is 2026-10-16.
The working branch's older product source was never changed.

## Reproduce a pair

Run from a checkout containing this document, with Python 3.13's `pytest` and `numpy`
installed in an isolated environment. Set `PY` to that environment's absolute Python path.
No xdist was installed for these receipts; if using xdist, append `-n 4` to both runs.
Every command below targets a disposable archive, never a live store or working checkout.
Keep the nested `trees/rc8` layout so the test bootstrap adds the isolated seed root to
`sys.path`, rather than the shared temporary directory.

```sh
REPO=$PWD
BASE=4e603ed65f137a909606afccfc735c691e363d25
RECEIPTS="$REPO/docs/invariants"
SEED_ROOT=$(mktemp -d)
SEED_ROOT=$(cd "$SEED_ROOT" && pwd -P)
SEED_TREE="$SEED_ROOT/trees/rc8"
mkdir -p "$SEED_TREE" "$SEED_ROOT/host" "$SEED_ROOT/cache"
git archive "$BASE" | tar -x -C "$SEED_TREE"
export BASE SEED_ROOT
# Extract the exact CI host stub from the pinned source, without running the workflow.
"$PY" - <<'PYCODE'
import os
from pathlib import Path
import subprocess
import textwrap
workflow = subprocess.check_output(
    ["git", "show", os.environ["BASE"] + ":.github/workflows/ci.yml"], text=True)
stub = workflow.split("cat > agent/context_engine.py <<'STUB'\n", 1)[1].split("\n          STUB", 1)[0]
host = Path(os.environ["SEED_ROOT"]) / "host" / "agent"
host.mkdir()
(host / "__init__.py").write_text("")
(host / "context_engine.py").write_text(textwrap.dedent(stub))
PYCODE
export PYTHONPATH="$SEED_TREE:$SEED_ROOT/host"
export PYTHONDONTWRITEBYTECODE=1 HERMES_DISABLE_LAZY_INSTALLS=1
export FASTEMBED_CACHE_PATH="$SEED_ROOT/cache" LCM_TEMPORAL_ROLLUPS_ENABLED=false
seed() (
    label=$1
    shift
    cd "$SEED_TREE" || exit 2
    pwd
    printf 'archive HEAD: %s\n' "$BASE"
    patch="$RECEIPTS/reverts/$label.patch"
    git apply --check "$patch" || exit 2
    git apply "$patch" || exit 2
    if nice -n 19 "$PY" -m pytest "$@" -q --tb=short --rootdir=. --confcutdir=. -p no:cacheprovider > "$SEED_ROOT/$label-red.log" 2>&1; then
        red=0
    else
        red=$?
    fi
    git apply -R "$patch" || exit 2
    if nice -n 19 "$PY" -m pytest "$@" -q --tb=short --rootdir=. --confcutdir=. -p no:cacheprovider > "$SEED_ROOT/$label-green.log" 2>&1; then
        green=0
    else
        green=$?
    fi
    printf '%s: reverted exit=%s restored exit=%s\n' "$label" "$red" "$green"
    test "$red" -eq 1 && test "$green" -eq 0
)
```

The patches are **forward-applying inverses** of the identified fix behavior. Apply
one at a time and restore before starting another. They retain all tests and later
unrelated behavior. Historical `git show FIX -- FILE | git apply -R --check` conflicts
on rc8 for most fixes; each receipt identifies the exact equivalent inverse used.
For a merge-commit fix (#1000's `bba59649`, #5's `9c8dbc9f`) `git show` prints an empty
combined diff; read the fix with `git diff FIX^1 FIX -- FILE` instead.
For #959 and #671 the full file inverse still applies, but the supplied patch selects
only the binding fix hunk, leaving other fixed behavior intact. No helper was removed
to manufacture an import or fixture failure. Reverted exit 1 below is an assertion
failure for the seed, not collection failure. Restored exit 0 means no skips or xfails.

## Receipt index

| Catalogue row | Receipt | Reverted | Restored |
|---|---|---|---|
| I1 | [#633](#i1-633) | 2 failed | 2 passed |
| I1 | [#1000](#i1-1000) | 8 failed | 8 passed |
| I2 | [#582](#i2-582) | 1 failed | 1 passed |
| I3 | [#1220](#i3-1220) | 2 failed | 2 passed |
| I3 | [#1227](#i3-1227) | 1 failed | 1 passed |
| I4 (row stays gap) | [#1221](#i4-1221) | 1 failed | 1 passed |
| I7 | [#628](#i7-628) | 1 failed | 1 passed |
| I8 | [#1048](#i8-1048) | 1 failed | 1 passed |
| I9 | [#559](#i9-559) | 1 failed | 1 passed |
| I10 | [#916](#i10-916) | 1 failed | 1 passed |
| I10 | [#959](#i10-959) | 1 failed | 1 passed |
| I13 | [#671](#i13-671) | 1 failed | 1 passed |
| I14 | [#5](#i14-5) | 1 failed | 1 passed |

## I1 633

Seed [#633](https://github.com/electricsheephq/lcm-x/issues/633); fix [`285bad2ea9c8a1c2110654e00967ceb0d8b1a111`](https://github.com/electricsheephq/lcm-x/commit/285bad2ea9c8a1c2110654e00967ceb0d8b1a111).
Inverse: [I1-633.patch](reverts/I1-633.patch). Remove the deferred insertion occurrence reservation in `identity_anchor.py`, preserving later host-UID bookkeeping. It reverts the fix's production change and runs the existing deterministic lead test; the fix's test-deletion hunk is excluded.

```sh
seed I1-633 tests/test_issue_633_steer_row_before_cursor.py::test_t6_a_repeated_steer_is_stored_again
```

Reverted: **2 failed** (exit 1). Restored: **2 passed** (exit 0).
`assert len(_steer_rows(engine)) == 2` → `assert 1 == 2` for identical and edge-whitespace insertions.

## I1 1000

Seed [#1000](https://github.com/electricsheephq/lcm-x/issues/1000); fix [`bba596498b615a036f278a54947b4dd096e47946`](https://github.com/electricsheephq/lcm-x/commit/bba596498b615a036f278a54947b4dd096e47946).
Inverse: [I1-1000.patch](reverts/I1-1000.patch). Remove only the stamped and unstamped `emitted_carrier` claims added by merged PR [#1001](https://github.com/electricsheephq/lcm-x/pull/1001). Keep descriptor generation and all other identity checks. The issue is open for release closure despite the merged fix.

```sh
seed I1-1000 tests/test_issue_650_survival_fit_keeps_summary.py::test_1000_held_fit_then_inplace_boundary_stores_no_carrier
```

Reverted: **8 failed** (exit 1). Restored: **8 passed** (exit 0).
`assert not any("Summary (d" in c[:40] for c in added)` fails: the carrier is stored again in all eight held/exception, stamped/unstamped, and turn-end-rewrite combinations.

## I2 582

Seed [#582](https://github.com/electricsheephq/lcm-x/issues/582); fix [`a3b9a04fe69257e402d1c8ba3fcda71ab02c820e`](https://github.com/electricsheephq/lcm-x/commit/a3b9a04fe69257e402d1c8ba3fcda71ab02c820e).
Inverse: [I2-582.patch](reverts/I2-582.patch). Restore phrase-based `_survival_generated` durability in `_survival_cut`, instead of requiring a store mapping or DAG-verified scaffold. This is the #582 loss-prevention hunk of the #581/#582 fix chain.

```sh
seed I2-582 tests/test_issue_581_582_store_complete_survival.py::test_r3a_an_unstored_summary_looking_row_is_never_dropped
```

Reverted: **1 failed** (exit 1). Restored: **1 passed** (exit 0).
`assert any(m is reply for m in kept)` → `assert False`: an unstored summary-looking reply disappears after the injected store lock.

## I3 1220

Seed [#1220](https://github.com/electricsheephq/lcm-x/issues/1220); fix [`f92a32c1d433cddfeaba52ce14454c07d835adb7`](https://github.com/electricsheephq/lcm-x/commit/f92a32c1d433cddfeaba52ce14454c07d835adb7).
Inverse: [I3-1220.patch](reverts/I3-1220.patch). Remove only `tool` from the short-row mark pass. Preserve the later #1227 assistant pass and rc8 last-resort user pass; this is the minimal equivalent inverse after those overlapping changes.

```sh
seed I3-1220 tests/test_issue_1220_survival_short_rows.py::test_parallel_short_results_fit_and_remain_stored
```

Reverted: **2 failed** (exit 1). Restored: **2 passed** (exit 0).
`assert engine._survival_measure(fitted) <= budget` → `assert 40122 <= 27852` in both compress and re-ingest cases.

## I3 1227

Seed [#1227](https://github.com/electricsheephq/lcm-x/issues/1227); fix [`dc5121ccc66ae92eba0a0bf5f26daae77957f7e1`](https://github.com/electricsheephq/lcm-x/commit/dc5121ccc66ae92eba0a0bf5f26daae77957f7e1).
Inverse: [I3-1227.patch](reverts/I3-1227.patch). Remove only `assistant` from the short-row mark pass, retaining tool handling and rc8 last-resort user handling.

```sh
seed I3-1227 tests/test_issue_1220_survival_short_rows.py::test_sequential_short_assistants_fit_during_summary_outage
```

Reverted: **1 failed** (exit 1). Restored: **1 passed** (exit 0).
`assert instance._survival_measure(fitted) <= budget` → `assert 32894 <= 27852` during the scripted summary outage.

## I4 1221

Seed [#1221](https://github.com/electricsheephq/lcm-x/issues/1221); fix [`f92a32c1d433cddfeaba52ce14454c07d835adb7`](https://github.com/electricsheephq/lcm-x/commit/f92a32c1d433cddfeaba52ce14454c07d835adb7).
Inverse: [I4-1221.patch](reverts/I4-1221.patch). Remove the refined `anchor_exempt` branch from #1221 so an over-default-bound objective competes with nodes again. Keep the existing bound and other selection logic. The refinement is [9f4639af](https://github.com/electricsheephq/lcm-x/commit/9f4639af); #1225 remains open, so I4 stays a gap.

```sh
seed I4-1221 tests/test_issue_1221_objective_default_bound.py::test_large_objective_survives_default_bound_and_fits
```

Reverted: **1 failed** (exit 1). Restored: **1 passed** (exit 0).
`assert sum(PREFIX in row.get("content", "") for row in fitted) == 1` → `assert 0 == 1`.

## I7 628

Seed [#628](https://github.com/electricsheephq/lcm-x/issues/628); fix [`93e46301f91a6809efa16ba026f17ec34a9a6a2e`](https://github.com/electricsheephq/lcm-x/commit/93e46301f91a6809efa16ba026f17ec34a9a6a2e).
Inverse: [I7-628.patch](reverts/I7-628.patch). Route content rejection back to `record_failure`, reverting only the #628 failure-versus-rejection distinction at the current route-key call site. No circuit API or fixture is removed.

```sh
seed I7-628 tests/test_issue_628_summary_circuit.py::test_t1_not_shorter_results_give_level_3_without_opening_the_circuit
```

Reverted: **1 failed** (exit 1). Restored: **1 passed** (exit 0).
`assert engine._summary_circuit_breaker.allows("")` → `assert False`: two rejected results incorrectly open the provider-failure circuit. This does not close the separate open #1120 deadline seed.

## I8 1048

Seed [#1048](https://github.com/electricsheephq/lcm-x/issues/1048); fix [`cb7e7bcad90c35f5786eb802022f7ccac1c136d6`](https://github.com/electricsheephq/lcm-x/commit/cb7e7bcad90c35f5786eb802022f7ccac1c136d6).
Inverse: [I8-1048.patch](reverts/I8-1048.patch). Remove only the missing-result stub token reservation from the ordinary assembly tail-selection path. Retain occurrence counting, pair repair, and orphan refunds.

```sh
seed I8-1048 tests/test_overflow_reserves_sanitizer_stubs.py::test_objective_survives_when_calls_lack_results
```

Reverted: **1 failed** (exit 1). Restored: **1 passed** (exit 0).
`assert OBJECTIVE in view(out)` fails; the sanitizer adds 60 tokens beyond the cap and the objective is dropped. This binds the named reservation defect; it does not prove every tool-pair shape.

## I9 559

Seed [#559](https://github.com/electricsheephq/lcm-x/issues/559); fix [`34bfa4a855613174c552e862cd762124cab05742`](https://github.com/electricsheephq/lcm-x/commit/34bfa4a855613174c552e862cd762124cab05742).
Inverse: [I9-559.patch](reverts/I9-559.patch). Restore the leaf-rescue fallback `current_chunk[:-1]` in place of the group-safe prefix end; preserve the group helper and ordinary leaf selection.

```sh
seed I9-559 tests/test_issue_559_tool_group_boundary.py::test_leaf_rescue_fallback_does_not_split_a_group
```

Reverted: **1 failed** (exit 1). Restored: **1 passed** (exit 0).
`assert engine._next_leaf_rescue_chunk(chunk, 50) == chunk[:1]` fails: two additional rows remain, splitting the parallel tool group.

## I10 916

Seed [#916](https://github.com/electricsheephq/lcm-x/issues/916); fix [`90dad17538ce71e69e110726f62d19c0809d5daf`](https://github.com/electricsheephq/lcm-x/commit/90dad17538ce71e69e110726f62d19c0809d5daf).
Inverse: [I10-916.patch](reverts/I10-916.patch). Restore mark-only emission for nonempty rows up to twice the head size, rather than whole/head-plus-mark emission. Keep long-row encoding and subsequent recognition logic.

```sh
seed I10-916 tests/test_issue_916_projection_keeps_head.py::test_newest_user_keeps_opening_under_survival_budget
```

Reverted: **1 failed** (exit 1). Restored: **1 passed** (exit 0).
`assert newest["content"].startswith(view[-1]["content"][:1200])` → `assert False`; the newest user opening is replaced by a mark.

## I10 959

Seed [#959](https://github.com/electricsheephq/lcm-x/issues/959); fix [`f3495f8cf13b7618d1c5ba19c9923b89370480e7`](https://github.com/electricsheephq/lcm-x/commit/f3495f8cf13b7618d1c5ba19c9923b89370480e7).
Inverse: [I10-959.patch](reverts/I10-959.patch). Remove only the `fields["tool_calls"] and marked` re-projection condition; retain the first-projection storage-only call mark and bounded lookup/parser changes.

```sh
seed I10-959 tests/test_issue_959_projection_marks.py::test_second_projection_of_a_storage_only_projection_keeps_a_mark
```

Reverted: **1 failed** (exit 1). Restored: **1 passed** (exit 0).
`assert source is not None and source["store_id"] == row["store_id"]` → `assert (None is not None)` after the second projection loses its mark.

## I13 671

Seed [#671](https://github.com/electricsheephq/lcm-x/issues/671); fix [`13669e4e3bff44f3d62c11bddc4babee9c6d6dd8`](https://github.com/electricsheephq/lcm-x/commit/13669e4e3bff44f3d62c11bddc4babee9c6d6dd8).
Inverse: [I13-671.patch](reverts/I13-671.patch). Remove only the early `persist=False` return before lineage/snapshot/emission publication in `_assemble_context`. Keep trial assembly and payload/recall guards.

```sh
seed I13-671 tests/test_issue_671_stub_first_exit.py::test_a_trial_that_misses_leaves_no_emission_candidates_behind
```

Reverted: **1 failed** (exit 1). Restored: **1 passed** (exit 0).
`assert engine._pending_emission_candidates == ["before the trial"]` → `assert [] == ["before the trial"]`.

## I14 5

Seed [#5](https://github.com/electricsheephq/lcm-x/issues/5); fix [`9c8dbc9fd0cdac42d2854e1b3ac0785d4c604666`](https://github.com/electricsheephq/lcm-x/commit/9c8dbc9fd0cdac42d2854e1b3ac0785d4c604666).
Inverse: [I14-5.patch](reverts/I14-5.patch). Remove only the expected-versus-actual frontier generation guard from `stage_compaction_publication`. Keep the DAG transaction, provenance checks, and frontier staging.

```sh
seed I14-5 tests/test_issue_5_contiguous_frontier.py::test_stale_expected_frontier_rolls_back_node_publication
```

Reverted: **1 failed** (exit 1). Restored: **1 passed** (exit 0).
`assert result == active` fails: a summary is published after the test advances the frontier, instead of rolling back the node. This seed binds stale-generation rejection, not every transactional failure mode.

## Gaps retained

- **I4:** #1221 has the pair above, but objective retention still has open residual [#1225](https://github.com/electricsheephq/lcm-x/issues/1225).
- **I5:** [#601](https://github.com/electricsheephq/lcm-x/issues/601) shipped operator recovery guidance for unsupported readers, not a fix to those readers. Removing only its older-reader backup advice with [I5-601.patch](reverts/I5-601.patch), while retaining later rollback-floor wording, leaves all three listed lead files green: **31 passed reverted, 31 passed restored**, no skips. The fix's own guidance tests do go red under the same inverse (the doctor survival-fit guidance tests in `tests/test_lcm_command.py` and `tests/test_issue_620_rollback_advice.py`: **5 failed reverted, 5 passed restored**), but they bind the operator advice, not the rollback readers I5 describes, so no seed proof is claimed for I5. The older-reader tests used the three pinned reader commits from CI and `GIT_DIR` pointing to the source checkout's object database (read-only `cat-file`/`archive` access).
- **I6:** no named historical seed has been selected. Schema and migration-compatibility tests are leads, not a named fix-reversion pair; the row stays `gap`.
- **I11:** [#1139](https://github.com/electricsheephq/lcm-x/issues/1139) fix [`c4b18f9b`](https://github.com/electricsheephq/lcm-x/commit/c4b18f9b) is merged. Removing just the default quarter-threshold cap with [I11-1139.patch](reverts/I11-1139.patch) leaves the exact small-window (32K) progress comparison against the previous GA green: **1 passed reverted, 1 passed restored**. The previous GA (v0.26.2, managed configuration) and both candidate variants each run **49 compactions, 3 summary calls**. That case has 40 user turns (122.5 compactions per 100 turns); it does not establish the absolute 25/100 bound or the 1M window. The fix's own unit test, `tests/test_p0_leaf_chunk_default.py`, does go red under the same inverse (**2 failed reverted, 6 passed restored**), but it pins the sweep target, not the progress bound, so no seed proof is claimed for I11.
- **I12:** [#1157](https://github.com/electricsheephq/lcm-x/issues/1157) and [#1222](https://github.com/electricsheephq/lcm-x/issues/1222) are open. The unchanged child-isolation behaviour file and the shared test harness were run on a scratch copy of rc8: **2 passed, 6 strict xfailed**. No merged fix exists to revert.

The differential backtest's historical comparisons and the characterization/compatibility
runs did not themselves hold a fix-reverted/fix-restored pair. They were not copied
into proven receipts. Where rc8 already had a deterministic regression, it supplied
the binding test above. No runtime, release, or universal invariant enforcement is claimed.
