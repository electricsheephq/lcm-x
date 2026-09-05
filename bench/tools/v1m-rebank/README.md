# V1-M re-bank kit (RUN-SHEET-V1M-REBANK, first execution 2026-09-05)

Run-record copies of the scripts the 2026-09-05 execution of `bench/specs/RUN-SHEET-V1M-REBANK.md` used, committed under
Amendment 1 of that sheet. They are a record first and a tool second: the shell scripts carry this host's paths
(`/Users/m1/hermes-work/...`, the session-notes evidence directory) and read the Voyage key from the environment (never from a file);
re-running them elsewhere means re-parametrising those paths. Nothing here is imported by the product or by `benchmarking/`.

| file | role | as-run vs committed |
|---|---|---|
| `prewarm_gate.sh` | the §5/§7/§8-step-4 pre-spend gate: determinism probe → `prewarm-cache --dry-run` (+ changed-unit manifest) → coverage/scope park → cap park → block park → real prewarm → consistency checks; exit 3 = PARK | byte-identical |
| `record_pins.sh` | launch / postrun pins: repo head + blob shas, dataset + manifest sha256s, cache size and row count, `_EnvFieldSpec` inventory | byte-identical |
| `launch_all.sh`, `run_shard.sh`, `run_aprime.sh` | the six shards (5-minute stagger, `--dump-candidates` ON, `--resume` derived from the checkpoint on disk) and the A/A′ subset | byte-identical |
| `identity_all.sh`, `result_identity.py` | the §4.3 identity projection over the eight pairs (six F53↔re-bank shards, F53-A′↔re-bank-A′, A/A′ discordance); self-test over F53's own outputs = 500/500, 100/100, 0 discordant | byte-identical |
| `cache_pair_check.py` | §4.4 per-shard cache pair = sum of per-question `embed_cache` rows vs F53's shard files; forward-baseline export | byte-identical |
| `corpus_privacy_inventory.py` | whole-corpus local replay of `protect_embedding_text` + `validate_embedding_privacy_dispatch` per unit (no provider calls); records digests, lengths and redacted shape statistics only — never raw text | lint-normalized |
| `changed_units_classes.py`, `blocked_units_attribution.py` | placeholder classes of the changed units; per-sub-detector attribution of the refused units with redacted line shapes | lint-normalized |
| `cache_membership_check.py` | §4.4 unit identity against the F53 cache keys (`sha256` of the exact unit text under provider/model), plus the exact `would_populate` the dry run would have reported had no unit blocked | byte-identical |

`KIT-MANIFEST-AS-RUN.sha256` holds the sha256 of every file as it ran (from the session-notes kit directory);
`KIT-MANIFEST-COMMITTED.sha256` holds the sha256 of the committed copies. The four rows marked lint-normalized differ only by
statement splits (`a; b` → two lines, `if x: y` → a block) and one loop-variable rename (`l` → `line_`) required by the repository's
ruff rules; every other row is byte-identical.

What the first execution produced and why it parked: `bench/FINDING-F62-V1M-REBANK-PARKED.md`.
