# S2: engine-direct replay

See the [kit recipe](../README.md) for required environment inputs and offline scoring.
The clean product checkout comes from `S2_PRODUCT_WORKTREE`; its exact commit is checked
against `S2_PRODUCT_SHA`. The driver registers it as `hermes_lcm`, supplies the host
auxiliary-client seam, and routes model calls through the repo-local harness lanes.
No installed host code or product source is edited.

```bash
KIT=bench/instruments/compaction_probe/track_s
python3 -B "$KIT/s2/run_s_lcmx.py" --arm ALL --seed smoke-1 --slice 10 --dry-run
python3 -B "$KIT/s2/run_s_lcmx.py" --arm LCMX-fleet-v2 \
  --seed 1 --run d1-r1 --lane glm --reader glm --checkpoints 176,304
```

Config inspection makes no model calls; replay uses the supplied lane and reader.
Output is below `TRACK_S_OUT/lcmx-runs/<arm>/<seed>/<run>`; config inspection uses
`TRACK_S_OUT/s2/dry-run`. Material is read from `TRACK_S_MATERIAL/<seed>`.

Arms are the retained `s2lib/arms.py` config inputs. Fleet-v2 inherits the fleet env
and adds only `LCM_SUMMARY_PROMPT_VERSION=2`; `<arm>-open` enables public-tool probes.
C0 supplies the full admitted transcript (unavailable when it exceeds the reader window);
C1 supplies only the fresh tail. Historical unsupported-arm labels are preserved.

Each run creates a fresh store, feeds genuine-role rows, ingests at host model-call
boundaries and invokes compaction only at the engine's own preflight gate. The system
row stays in the system slot. Every sweep records before/after tokens, node provenance,
escalation levels, summariser calls and continuity. `is_compaction=false` means no
summary publication, even if the sweep took time.

At each `--checkpoints` row, the store and payloads are snapshotted. Each probe batch
uses a fresh clone, with before/after digests. Plain arms answer from the frozen context;
open arms may call the public tools on the clone. `results.jsonl`, `summary.json`,
`config.json`, assembled contexts, answers, continuity and engine logs remain run outputs.

`prefix60k.py <material-seed-dir>` freezes the first eligible 60k-token workload row
once and refuses to overwrite it. `--prefix60k` replays that frozen workload without
probes. Its timing population stays separate from full-stream and wiring-only smoke data.
