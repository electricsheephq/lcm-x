# S4: native CLI external reference

This reported arm reuses the repository's existing `drive_codex.py` and `parse_rollout.py`
by relative import. It requires Python 3.12+, the installed CLI, `TRACK_S_OUT`,
`TRACK_S_MATERIAL`, and an explicit `--auth-file` supplied by the caller.
Authentication is copied only to the isolated run home below `TRACK_S_OUT/s4/home`.
See the [kit recipe](../README.md); never place auth or run data in the repository.

```bash
KIT=bench/instruments/compaction_probe/track_s
python3 -B "$KIT/s4/run_s_codex.py" --auth-file "$CLI_AUTH_FILE" \
  --seed smoke-1 --slice 10 --force-event --dry-run
python3 -B "$KIT/s4/run_s_codex.py" --auth-file "$CLI_AUTH_FILE" \
  --seed 1 --run d1-r1 --dictation file --stop-row 304
python3 -B "$KIT/s4/estimate.py" 1
```

Dry-run checks the existing login and token-expiry gate and materializes planned inputs;
it makes no model calls. Replay uses the retained model/effort pins and reads them back
from the rollout. `--readmit` re-extracts existing admission/survival offline.
`estimate.py` fits the existing smoke data and writes `TRACK_S_OUT/estimate-seed-<n>.json`.

The system row becomes workspace instructions; tool rows become files the agent reads.
Assistant notes use the retained user-dictation or file mapping. Native compaction is
read from rollout telemetry; smoke's forced event remains wiring-only. Every probe
batch uses a fresh context-only fork, with tools disabled and unexpected tool items
rejected. Checkpoint digests, per-role admission, survival and reader readback are recorded.

Run artifacts live under `TRACK_S_OUT/codex-runs/<seed>/<run>`; dry-run plans use
`TRACK_S_OUT/s4/dry-run`. Native server-side encrypted summaries limit observable
survival evidence. External comparisons are reported and never per-release gates.
