# S1: lossless-claw external reference

This reported external arm bundles the actual caller-supplied lossless-claw checkout
at `988dee85592b9066ffe1c859542e8e18c23f2345`. `run.sh` checks that commit;
esbuild resolves `track-s-lossless` imports to that checkout. No third-party source,
node_modules or authentication files are bundled in this kit.
Requires Node 22+, the checkout's existing esbuild/dependencies, the shared output/material
environment inputs, and `GLM_API_KEY` in the environment. See the [kit recipe](../README.md).

```bash
KIT=bench/instruments/compaction_probe/track_s
bash "$KIT/s1/run.sh" "$LOSSLESS_CHECKOUT" smoke-seed-1 dry --slice 10 --dry-run
bash "$KIT/s1/run.sh" "$LOSSLESS_CHECKOUT" seed-1 d1-r1-default \
  --arm lossless-claw --checkpoints 176,304
```

Dry-run inspects effective config without model calls. Replay calls the real engine and
GLM summariser/reader. Default and tuned arms, open probes, cadence, prefix60k and
checkpoint snapshot behavior are preserved. The runner scrubs inherited `LCM_*` inputs,
then sets only its log file. The tuned arm changes `freshTailMaxTokens` to 64000.

Build output lives under `TRACK_S_OUT/s1/build`; stores, files, logs, probe clones and
isolated HOME live under `TRACK_S_OUT/lc-runs`. Material is read from `TRACK_S_MATERIAL`.
`kit_positions.py <run-dir> <material-seed-dir>` adds the kit-token axis to the chosen
run summary offline, using this repository's existing generator/token counter.

Each turn uses genuine engine ingest/assemble/afterTurn behavior on the declared cadence.
Publication and pending-summary telemetry, admission, ancestry, externalization and
continuity receipts accompany cloned probes. Level markers have the external engine's
own proof limits. External comparisons are reported and never per-release gates.
