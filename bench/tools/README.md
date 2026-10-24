# Benchmark run instruments

These are Python 3.11+ program-side tools. They use only the standard library.

## Fail-close accounting

`python bench/tools/failclose.py RUN_DIR [--compare OTHER_RUN_DIR]` emits raw and adjusted scores, fail-closed qids, and each row's signature as one JSON object. Paired comparisons use the named **union-drop** convention: a qid that fail-closes in either arm is dropped from both. This prevents the F35/#165 fail-close classes from silently becoming capability zeros ([F35](../FINDING-F35-SANITY-SLICE-RED.md), [F32 §3](../FINDING-F32-RELEASE-NUMBER-VERDICT.md)).

## Pin verification

`python bench/tools/pinverify.py pre-run pins.yaml` and `post-run` verify named worktree commits and cleanliness, PATH binary hashes, dataset/qid hashes, and environment knobs. Each writes a want/got `PINS-PRERUN.txt` or signed-off `PINS-POSTRUN.txt` beside the pins file and exits nonzero on drift. This prevents F32 §5.1's silent transport upgrade ([F32 §5](../FINDING-F32-RELEASE-NUMBER-VERDICT.md)).

Because no YAML package is allowed, `pins.yaml` uses YAML's JSON-compatible form:

```json
{
  "version": 1,
  "worktrees": {"name": {"path": "/worktree", "git_sha": "abc123", "clean": true}},
  "binaries": {"codex": {"name": "codex", "sha256": "64 hex characters"}},
  "files": {"dataset": {"path": "/data/questions.json", "sha256": "64 hex characters"}},
  "env": {"HERMES_MB_CODEX_EFFORT": "medium"}
}
```

See [`pins-f32-example.yaml`](pins-f32-example.yaml) for the confirm run's documented F32 pin set. Values whose full hashes lived only in the unavailable private run artifacts are marked for verbatim substitution from `PINS-PRERUN.txt`; they are not fabricated here.

## Store freeze

`python bench/tools/storefreeze.py freeze DIR`, `verify DIR MANIFEST`, and `copy-verified SRC DST MANIFEST` create, check, and privately copy exact-name store snapshots. The canonical manifest is one JSON line with every file's sha256 and size plus a hash of the unsigned manifest. This prevents the spec's STRATEGY §5 store-extension near-miss (the finding is under Phase 2 in the current [STRATEGY](../STRATEGY-2026-07-25.md)) that almost shifted corpus IDF.

## D4 recall-health replacement

`python bench/tools/d4_health_rerun.py --report "$HARNESS/data/runs/X/report.json" --checkpoint "$HARNESS/data/runs/X/checkpoint.json" --output MERGED/report.json --receipt MERGED/receipt.json --instruments . --config OFF -- bun run "$HARNESS/src/index.ts"` applies the D4 one-replacement rule. Set `HARNESS` to the absolute benchmark-harness root containing the original run under `data/runs/X/`. Keep that layout: the tool derives the bridge working directory as the harness root from the report location, and the replacement lands beside `X`. A shallow `RUN/report.json` or a relocated report directory is unsupported. Run the driver from the instrument tree; `--instruments` must name that same tree. Use `--config LOCAL` for local semantic recall and `--workers 1` through `4` to bound the replacement. Only unhealthy completed recalls are selected; abstention questions are exempt, and failures outside recall are not retried. The fresh replacement supplies both score and health. Inputs stay byte-unchanged, existing outputs or replacement runs stop execution, and no second replacement is allowed. The JSON receipt records input hashes and the exact command. Exit codes are `0` for COMPLETE, `3` for INCOMPLETE or INCONCLUSIVE (more than 1% still unhealthy), and `2` for STOP. This produces a merged report for scoring and health accounting; it does not certify a release row.

Replacement inputs (the `-health-rerun1` run-ID suffix or a `healthReplacement` marker in either input) stop before command construction. Recorded answerer/judge execution provenance and dataset/scorer pins must agree across attempts; a field absent from both is not compared. The original report/checkpoint pair is validated before health selection, including healthy-only runs; validated checkpoint-only pins are retained in the merged report. Call/retry counts and the selected-population `questionSetSha256`, `normalizedInputsSha256`, and `loaderOrderSha256` may differ, while each attempt's report/checkpoint subset pins must agree when present. Category correctness, latency and retrieval, plus overall retrieval, are rebuilt from merged evaluations using the harness formulas. Overall latency, token, MemScore and CLI-ledger summaries are removed and listed in `aggregatesRemoved`: evaluation rows lack the full phase/token/call inputs needed to rebuild those summaries together.
