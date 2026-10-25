# Embedding consumer inventory: benchmark producers

Refs #1959 and #1186. This inventory covers producer configuration after the
local-recall default flip. It records selection and output identity, not model
readiness, retrieval quality or release acceptance. Product tools and defaults-only readers are listed as exemptions below.
The constructor sweep covers `bench/`, `benchmarking/`, `benchmarks/` and
`scripts/`: eight deterministic constructor sites pin OFF, and 15 retained default-reading
sites are accounted for (including four Track S reads and three backfill reads).

## Producers with omitted embedding settings

These are the default-following producers and the deterministic probes that
previously omitted embedding settings. An omitted setting can enable LOCAL and
background preparation on a new tree; it must not masquerade as an OFF run.

| Producer / consumer | Selection | Effective record or explicit OFF pin |
| --- | --- | --- |
| `benchmarks/benchmark_active_tool_stubbing.py::_run_mode` | Deterministic assembly latency and token savings | Pins OFF; background model preparation is outside the timed workload. |
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
| Track S `run_s_lcmx.py::dry_run` and local catch-up `tests/test_fix_round.py::test_5_product_defaults` | Defaults-only `LCMConfig()` comparisons; Track S serializes the effective defaults with `config_dict`, and the catch-up test compares policy fields without starting a default-config engine. |
| `scripts/store_wedge_inspect.py::_ReadEngine` | Product inspection tool: `LCMConfig(**config)` supplies replay-policy fields to a read-only facade; no `LCMEngine` or embedding execution. |
| `scripts/backfill_externalized_tool_outputs.py` (three `from_env()` sites) | Product backfill tool: preserves runtime configuration for protection/externalization policy; does not select a benchmark arm. |
| `scripts/import_lossless_claw.py` (`from_env()`) | Product import tool: reads protection policy for imported data; does not select a benchmark arm. |
| `scripts/maintainer_gate.py` | Generic risk classification; does not select or execute an embedding arm. |

Release-gauntlet instruments, product doctor/readiness behavior and historical
result artifacts are unchanged by this producer correction. A configuration
record alone does not certify LOCAL index coverage.
