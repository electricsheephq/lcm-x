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
