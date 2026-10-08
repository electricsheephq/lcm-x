# Track S run kit

Bench-only instruments for the v0.27.0 Track S gate: the S2 engine-direct replay driver,
the offline scorer, paired analysis, and the reported S1/S4 external-arm drivers.
Related work: [#898](https://github.com/electricsheephq/lcm-x/issues/898),
[#660](https://github.com/electricsheephq/lcm-x/issues/660), and
[#659](https://github.com/electricsheephq/lcm-x/issues/659).
No product behavior changes. Keep this work draft until the v0.26.0 GA.

## Inputs and outputs

Use Python 3.11+ (S4 uses Python 3.12+), bash 3.2+, and a clean product checkout.
The driver requires `S2_PRODUCT_WORKTREE`; set `S2_PRODUCT_SHA` to that checkout's exact
commit (the retained historical default is v0.24.8 GA). The seam refuses a different
commit or tracked modifications. `TRACK_S_OUT` is required for driver output;
`TRACK_S_MATERIAL` points to generated material. Both should be outside the repo.
`GLM_API_KEY` comes only from the caller's environment. No key files are read by GLM.
Runs, stores, material, logs, credentials, and receipts are never source artifacts.

## Generate material

From the repository root, generate the existing `track-s-v3` material; no material is
bundled with the kit. Use identical generator settings and material digests for paired runs.

```bash
KIT=bench/instruments/compaction_probe/track_s
export TRACK_S_OUT="$RUN_OUTPUT_ROOT"
export TRACK_S_MATERIAL="$MATERIAL_ROOT"
export S2_PRODUCT_WORKTREE="$PRODUCT_CHECKOUT"
export S2_PRODUCT_SHA="$PRODUCT_COMMIT"
for seed in 1 2 3; do
  python3 bench/instruments/compaction_probe/gen_material.py \
    --seed "$seed" --placements --classes12 \
    --out-dir "$TRACK_S_MATERIAL/seed-$seed"
done
```

For wiring-only smoke material add `--smoke` and use `smoke-seed-1` as the output name.
The generator's own manifest records the version, settings, tokenizer and digests.

## S2 replay (model calls)

Supply `GLM_API_KEY` through the environment before running a replay. Replays use a fresh
store and refuse nonempty run directories. The fleet-v2 arm changes only
`LCM_SUMMARY_PROMPT_VERSION=2` from the fleet arm. Existing arms and controls remain.

```bash
python3 -B "$KIT/s2/run_s_lcmx.py" --arm LCMX-fleet-v2 \
  --seed 1 --run d1-r1 --lane glm --reader glm --checkpoints 176,304
```

Run both arms for seeds 1/2/3 and labels `d<seed>-r1` / `d<seed>-r2`. Store completion
wall receipts in a caller-owned `decision/logs` directory with names
`s2-<arm>-d<seed>-r<rep>.log.wall`, containing `start <unix_seconds>` and
`exit 0 end <unix_seconds>` lines. Scoring excludes missing/failed receipts.
For config inspection only, use `--arm ALL --seed smoke-1 --slice 10 --dry-run`.
See [S2](s2/README.md) for replay, snapshots and probe isolation.

## Offline scoring and paired analysis (no model calls)

`EXISTING_RUN_ROOT` is an existing `lcmx-runs` directory. `EXISTING_LOG_ROOT` contains
its completion receipts. These commands read existing material/runs and write scores,
loss classifications, reports and analysis into the chosen output directory only.

```bash
python3 -B "$KIT/decision/score_decision.py" \
  --run-root "$EXISTING_RUN_ROOT" --logs "$EXISTING_LOG_ROOT" \
  --material "$TRACK_S_MATERIAL" --out "$TRACK_S_OUT/decision" \
  --arms LCMX-fleet LCMX-fleet-v2 --seeds 1 2 3 --checkpoints 176 304
python3 -B "$KIT/decision/analyze_paired.py" \
  --run-root "$EXISTING_RUN_ROOT" --logs "$EXISTING_LOG_ROOT" \
  --material "$TRACK_S_MATERIAL" --decision-root "$TRACK_S_OUT/decision" \
  --arms LCMX-fleet LCMX-fleet-v2 --seeds 1 2 3 --checkpoints 176 304 \
  > "$TRACK_S_OUT/paired.json"
```

Pass `--load-log` to include timestamped host-load observations. Pairing is by seed and
run index. Each axis reports exact two-sided McNemar on paired items and an exact
per-run sign test (wins favor the second arm; ties excluded from the p-value).
`v1`/`v2` and `b_v1_only`/`c_v2_only` always denote first/second CLI arms, irrespective
of their names. Continuity compares items only at event rows shared by both arms and
excludes the identical system-slot host instruction. Latency reports completed
compactions and positive-wall noncompacting sweeps separately, with sample counts.
Level-3 provenance and r1/r2 spread remain separately reported.
Missing requested pairs mark the analysis `INCOMPLETE`; isolation or reader-pin failures, failed forced S4 checkpoints, auth refreshes, and a reused S4 home with `config.toml` fail the run.
Incremental scoring maintains `decision/manifest.json` with receipt hashes and scoring times; reports ignore and list unmanifested checkpoint JSONs, while legacy directories without a manifest retain their existing behavior and report `manifest: absent`.

### Release over release (Phase D, D2)

D2 pairs the same arm on two product trees: the previous GA and the candidate. Replay each
tree into its own output root (`PREV_ROOT`, `CAND_ROOT`), score each root once, then pair them.

```bash
# Previous GA tree, then the candidate tree: same arm, seeds 1/2/3, runs d<seed>-r1/-r2 each, one wall receipt per run.
replay() {  # <root> <checkout> <commit>
  mkdir -p "$1/decision/logs"
  for seed in 1 2 3; do for rep in r1 r2; do
    wall="$1/decision/logs/s2-LCMX-fleet-d$seed-$rep.log.wall"
    echo "start $(date +%s)" > "$wall"
    TRACK_S_OUT="$1" S2_PRODUCT_WORKTREE="$2" S2_PRODUCT_SHA="$3" \
      python3 -B "$KIT/s2/run_s_lcmx.py" --arm LCMX-fleet \
      --seed "$seed" --run "d$seed-$rep" --lane glm --reader glm --checkpoints 176,304 \
      && echo "exit 0 end $(date +%s)" >> "$wall"
  done; done
}
replay "$PREV_ROOT" "$PREV_CHECKOUT" "$PREV_COMMIT"
replay "$CAND_ROOT" "$CAND_CHECKOUT" "$CAND_COMMIT"
for root in "$PREV_ROOT" "$CAND_ROOT"; do
  python3 -B "$KIT/decision/score_decision.py" \
    --run-root "$root/lcmx-runs" --logs "$root/decision/logs" \
    --material "$TRACK_S_MATERIAL" --out "$root/decision" \
    --arms LCMX-fleet --seeds 1 2 3 --checkpoints 176 304
done
python3 -B "$KIT/decision/analyze_paired.py" --material "$TRACK_S_MATERIAL" \
  --arms LCMX-fleet LCMX-fleet --labels prev cand --seeds 1 2 3 --checkpoints 176 304 \
  --run-root "$PREV_ROOT/lcmx-runs" --decision-root "$PREV_ROOT/decision" --logs "$PREV_ROOT/decision/logs" \
  --second-tree "$CAND_ROOT/lcmx-runs" "$CAND_ROOT/decision" "$CAND_ROOT/decision/logs" \
  --expect-shas "$PREV_COMMIT" "$CAND_COMMIT" \
  > "$CAND_ROOT/d2-paired.json"
```

`--second-tree RUN_ROOT DECISION_ROOT LOGS` reads the second arm from the candidate root;
`--labels` (distinct; default the two arm names) key `per_arm`, `loss_classes` and `trees`.
Every `p` is rounded for display and has an unrounded `p_exact`. The gate reads `d2.verdict`
(`INCOMPLETE`, `REPEAT_SEEDS`, `BLOCK` or `PASS`) and, per checkpoint, `d2.cp-<n>.p_exact` and
`effect_pts_exact` (candidate minus previous). D2's unit is the run, stratified by seed: the facts of a
run share its compactions, so they are correlated (the run-level variance was 3–25× the
independent-fact variance at cp-304 on real runs). A run's kept share is kept / scored facts; the effect is
the mean over seeds of the candidate's mean run share minus the previous tree's, and `p_exact` is the
exact stratified run-level permutation test (`test`): every within-seed split of a seed's 4 runs into
2 + 2, 216 splits for 3 seeds, so the minimum reachable p is 2/216 ≈ 0.0093. Every seed needs both runs
on both sides; otherwise `p_exact` is null and the verdict is `INCOMPLETE`. `d2.cp-<n>.run_shares` and
`<cp>.run_level` show the runs. The per-fact statistics are diagnostics only (`d2.cp-<n>.facts_per_fact`,
`diagnostic_only`): the exact sign-flip test on per-fact share differences (`fact_sign_flip_p_exact`),
the sign test (`sign_test_p_exact`) and the per-occurrence McNemar (`occurrence_mcnemar_p_exact`, also
in `axes.facts_all`) all assume independent facts. `d2.spread_over_0.10` lists, per label,
each `cp-<n>/seed-<n>` whose r1/r2 facts-kept spread exceeds 0.10 (`REPEAT_SEEDS`); repeating those
seeds, and the inconclusive outcome, is the operator's step. `--expect-shas` (required with
`--second-tree`) binds each arm to its product commit. The verdict is also `INCOMPLETE` when a
checkpoint/seed spread is unmeasured (`d2.missing_spread`, e.g. a missing receipt), a counted run
records another commit (`d2.sha_mismatch`), an admitted score's recorded receipt hash differs from
the selected tree's run receipt (`d2.receipt_mismatch`, e.g. a decision root from another tree), or,
with `--second-tree`, the configuration the arm applies differs between the trees (`d2.config_diff_keys`;
single-tree runs compare different arms on purpose; the tokenizer is part of the configuration), a counted run
replayed other material than the analysis reads, or a material file no longer matches the hash its manifest lists or
is named outside its seed directory (`d2.material_mismatch`; the runner records `material_sha256`; file mismatches are
reported before any material is parsed), or the analysis omits a D2 seed or checkpoint
(`d2.missing_required`; D2 is seeds 1/2/3 at checkpoints 176 and 304). Repeated `--seeds` or `--checkpoints` values
stop the analysis with an error. With `--second-tree`, each tree's decision root and logs must sit in its run root's
tree, or the analysis stops with an error. `d2.effective_config_diff_keys` names the keys of the runs' recorded
`config.json` that differ; it is a diagnostic, not a verdict input, because a product default that changes between
releases is the change under test. It is `null` when any run lacks a readable config, and
`d2.effective_config_unavailable` names those runs. Host load is also a diagnostic, not a verdict input:
with `--second-tree`, pass one `--load-log` that covers both trees' run windows.

The historical decision runner accepts `glm|s4 <seed> [lossless-checkout] [CLI-auth-file]`;
`run_seeds_2_3.sh` accepts those last two inputs and retains its existing scheduling.
The stub2k gate retains its historical rc3 SHA requirement and writes below
`TRACK_S_OUT/stub2k-gate`. `render_decision.py --report-notes <directory>` reads the
caller's frozen `report-unit-notes.md/json` inputs. Scoring external runs uses
`score_decision.py --external-root <root-containing-lc-runs-and-codex-runs>`.

## External arms and offline checks

[S1](s1/README.md) runs the actual pinned lossless-claw checkout supplied by the caller;
[S4](s4/README.md) uses the installed CLI with caller-supplied authentication.
External arms are reported, never used as the per-release gate. The optional harness
Claude adapter uses a caller-selected/PATH CLI and the existing macOS Keychain login;
its configuration and scratch directories live under `TRACK_S_OUT`.

```bash
python3 -m pytest "$KIT/scorer/tests" -q -p no:xdist
ruff check .
```

The scorer uses only the standard library; offline fixtures require pytest and numpy.
Scratch follows `TMPDIR`. Score reproduction proves kit portability and measurement
consistency; it establishes no product-quality, release or runtime claim.
