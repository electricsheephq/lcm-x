# Embedding consumer inventory: benchmark producers

Refs #1959 and #1186. This inventory covers producer configuration after the
local-recall default flip. It records selection and output identity, not model
readiness, retrieval quality or release acceptance. Product readers are outside
this producer-only inventory.

## Producers with omitted embedding settings

These are the default-following producers and the deterministic probes that
previously omitted embedding settings. An omitted setting can enable LOCAL and
background preparation on a new tree; it must not masquerade as an OFF run.

| Producer / consumer | Selection | Effective record or explicit OFF pin |
| --- | --- | --- |
| `benchmarking/replay.py::_config_from_policy` (including `scripts/lcm_benchmark.py`) | policy@v1 deterministic compaction | Pins `embeddings_enabled=False`; background model work is outside this policy. |
| `benchmarking/steady_state.py::_build_engine` | Ingest/preflight timing | Pins OFF to isolate the measured hot path. |
| `benchmarking/stress.py::StressRun.make_engine` | Deterministic stress cases | Defaults explicitly OFF; a named embedding case can explicitly override it. |
| `benchmarking/stress.py::_case_lifecycle_soak_and_profile_rebinds` | Lifecycle probe | Pins OFF. |
| `benchmarking/stress.py::_profile_rebind_probe` | Profile rebind probe | Pins OFF. |
| `scripts/benchmark_threshold_full_sweep.py::_run_policy` | Deterministic compaction-policy comparison | Pins OFF. |
| `bench/instruments/occurrence_fuzz/driver.py::run` | Occurrence/lifecycle sequence probe | Pins OFF. |
| `bench/instruments/compaction_probe/track_s/s2/s2lib/arms.py::FLEET` and `run_s_lcmx.py::new_engine` | Fleet arms omit embeddings; env-file arms can supply them | Follows measured-tree `from_env()` defaults. `config_dict` resolves the model and persists enabled/provider/model in `config.json`; D2/D7 rows must retain that file. |
| `benchmarks/qa-harness/src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py::Bridge` | Both inputs omitted: measured-tree defaults; model-only: historical fastembed | Initialize response and `effective-config.json` record enabled/provider/resolved model, source and explicitness. Disabled records have empty provider/model. Feature-detects older config fields and helpers. |
| `benchmarking/longmemeval.py::run_harness` / `scripts/lcm_longmemeval.py --shipped-default` | Explicit tree-default measurement | Report and checkpoint/dump headers record enabled/provider/resolved model and default provenance. Explicit provider without model is rejected; abbreviations are rejected. Ordinary CLI default remains the offline stub. |
| `scripts/eval_retrieval_recall.py --live-provider` | Explicit live-provider gate, then measured-tree `from_env()` | JSON and Markdown include `effective_config` from the resolved provider. Without the live flag the deterministic mock remains offline. |

## Explicit arms and generic readers

| Consumer | Configuration boundary |
| --- | --- |
| FMP `bench/fmp/stores.py::Stores` | OFF/LOCAL/VOYAGE remain explicit. Build identity includes effective configuration; OFF has no provider/model. Identity-only callers without a measured tree record null. |
| `bench/fmp/dev.py` | Recall override allowlist; no independent embedding default or model choice. Store identity comes from `Stores`. |
| Local catch-up `common.py`, `measurement.py`, `calibration.py` | Explicit LOCAL or OFF. Existing configuration serialization records the resolved model; calibration has no independent model default. |
| Bgprep `replay.py` | Explicit OFF; existing configuration JSON resolves the model for enabled configurations. |
| Scale389 `phase1a.py` | Explicit OFF for the ingest probe; provider/cache bindings are explicit. |
| Reliability dependency qualification | Explicit LOCAL provider/model, separate readiness checks. |
| Reliability `scorers/externalized.py` | Reads default externalization policy only; does not start an embedding engine. |
| `scripts/gen_config_reference.py` | Reads declared defaults to generate reference text; does not run a model. |
| `scripts/maintainer_gate.py` | Generic risk classification; does not select or execute an embedding arm. |

Release-gauntlet instruments, product doctor/readiness behavior and historical
result artifacts are unchanged by this producer correction. A configuration
record alone does not certify LOCAL index coverage.
