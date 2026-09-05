# V1-M re-bank kit (RUN-SHEET-V1M-REBANK, first execution 2026-09-05)

Run-record copies of the scripts the 2026-09-05 execution of `bench/specs/RUN-SHEET-V1M-REBANK.md` used, committed under
Amendment 1 of that sheet. They are a record first and a tool second: the shell scripts carry this host's paths
(`/Users/m1/hermes-work/...`, the session-notes evidence directory) and read the Voyage key from the environment (never from a file);
re-running them elsewhere means re-parametrising those paths. Nothing here is imported by the product or by `benchmarking/`.

| file | role | committed copy vs the as-run copy |
|---|---|---|
| `prewarm_gate.sh` | the §5/§7/§8-step-4 pre-spend gate: determinism probe → `prewarm-cache --dry-run` (+ changed-unit manifest) → coverage/scope park → cap park → block park → real prewarm → consistency checks; exit 3 = PARK | byte-identical |
| `record_pins.sh` | launch / postrun pins: repo head + blob shas, dataset + manifest sha256s, cache size and row count, `_EnvFieldSpec` inventory | fixed after review: the env inventory is built from `ENV_FIELD_SPECS` structurally (the as-run regex dropped multi-line declarations) with emitted == declared asserted; credential-shaped variable values (`KEY\|TOKEN\|SECRET\|PASS\|COOKIE\|AUTH\|CREDENTIAL`) are redacted in both listings; `REBANK_PY` default applied consistently |
| `launch_all.sh` | the six shards (5-minute stagger) | fixed after review: refuses to launch unless `prewarm_gate.sh` has written `gate-passed-at.txt` (it does so only on PASS) |
| `run_shard.sh`, `run_aprime.sh` | one shard / the A/A′ subset (`--dump-candidates` ON, `--resume` derived from the checkpoint on disk) | byte-identical |
| `identity_all.sh` | the §4.3 identity projection over the eight pairs (six F53↔re-bank shards, F53-A′↔re-bank-A′, A/A′ discordance); self-test over F53's own outputs = 500/500, 100/100, 0 discordant | fixed after review: a missing A′ run is exit 2 (missing mandatory input), never 0; `result_identity.py`'s exit 2 is preserved instead of collapsing into 1 (= a delta) |
| `result_identity.py` | the per-question projection comparator | byte-identical |
| `cache_pair_check.py` | §4.4 per-shard cache pair = sum of per-question `embed_cache` rows vs F53's shard files; forward-baseline export | fixed after review: misses must be zero on both sides (equal non-zero misses are not parity); total misses reported |
| `corpus_privacy_inventory.py` | whole-corpus local replay of `protect_embedding_text` + `validate_embedding_privacy_dispatch` per unit (no provider calls); records digests, lengths and redacted shape statistics only — never raw text | lint-normalized; fixed after review: records `repo_head` + `repo_dirty_files` and refuses a checkout that does not match an optional expected sha |
| `changed_units_classes.py` | placeholder classes of the changed units | lint-normalized; fixed after review: every manifest unit must exist in the prepared dataset and still transform, else the run fails loud |
| `blocked_units_attribution.py` | per-sub-detector attribution of the refused units with redacted line shapes | lint-normalized |
| `cache_membership_check.py` | §4.4 unit identity against the F53 cache keys (`sha256` of the exact unit text under provider/model), plus the exact `would_populate` the dry run would have reported had no unit blocked | fixed after review: selects the run's cache identity (`voyage` / `voyage-context-3`) explicitly and fails if that group is absent, instead of the largest group |
| `refused_line_model.py` | per-segment line-model record of the refused units (kind, physical line, width, token count, longest token, counted-by-backstop) — the F62 §8a evidence; no raw text | byte-identical |

`KIT-MANIFEST-AS-RUN.sha256` holds the sha256 of every file as it ran on 2026-09-05 (from the session-notes kit directory);
`KIT-MANIFEST-COMMITTED.sha256` holds the sha256 of the committed copies. Three files (`corpus_privacy_inventory.py`,
`changed_units_classes.py`, `blocked_units_attribution.py`) were lint-normalized for the repository's ruff rules — statement splits
(`a; b` → two lines, `if x: y` → a block) and one loop-variable rename (`l` → `line_`). Seven files then received the post-review
fixes listed in the table (PR #416 review threads). The parked execution's evidence was produced by the as-run copies; the three
probes whose logic changed were re-run against the same inputs with the committed copies and reproduced their as-run artifacts
(`cache-membership.committed-tool.json`, `changed-units-classes.committed-tool.json`, `corpus-privacy-inventory.committed-tool.json`
beside the originals in the evidence directory). The tools that never ran in the parked execution (`launch_all.sh`, `cache_pair_check.py`,
the post-A′ branch of `identity_all.sh`, `record_pins.sh postrun`) carry their fixes for the next execution.

What the first execution produced and why it parked: `bench/FINDING-F62-V1M-REBANK-PARKED.md`.

Four further review findings (third pass of PR #416) on tools that did not run in the parked execution — gate-marker binding in
`launch_all.sh`, failure propagation in the `record_pins.sh` pin block, credential redaction of the captured run environment and a
product-sha resume guard in `run_shard.sh` / `run_aprime.sh` — are tracked on #415 and must be fixed before the next execution of the sheet.
