# Benchmark run instruments

These are Python 3.11+ program-side tools. They use only the standard library.

## Run a candidate the way CI does

Prepare a Linux host or container as a non-root user with `bash scripts/dev/cloud_setup.sh`.
Make a patch with `git -c diff.noprefix=false diff --binary BASE..HEAD > candidate.patch`, then run:

```bash
bash bench/tools/run_candidate.sh --base SHA40 --patch candidate.patch --label example \
  --focused "tests/test_example.py" --red "tests/test_example.py" --full
```

Each run makes fresh stamped directories under `--root` (default `candidate-runs/` in the
checkout, which git ignores): a run directory, a green worktree with the whole patch, and, with
`--red`, a red worktree at the base with only the patch's `tests/` hunks. The phases are ruff on
the changed `.py` files, `green-focused`, `red-on-base`, and, with `--full`, CI's main run
without `tests/test_embedding_privacy.py` (`green-full`) followed by that file alone
(`green-privacy`), as CI runs them. Each pytest phase gets an empty HOME in the run directory.
The runner prints each phase's exit and last lines and writes `summary.json` to the run
directory; its commands use paths relative to that directory. Test failures are recorded: exit 0
means every requested phase ran, and exit 2 reports a setup failure with its reason, such as
`NO_ENV`, `NO_RUFF`, `APPLY_FAILED` or `NO_TEST_HUNKS`. Nothing is deleted; the runner prints
the `git worktree remove` commands for the worktrees it made.
Red-on-base applies only `tests/` hunks, so a test that needs a new non-test file errors on base.
For a remote host, copy the patch and run the same command over your own SSH using
`<remote>` and `<checkout>` placeholders. A green run is advisory; GitHub CI gates.
`--env-file PATH` and `--no-stub` exist for the runner's own fixture test; normal runs load the
setup environment and run CI's stub step. Defaults: 4 focused/red workers, 6 full workers,
nice 10; override with `--workers-focused N`, `--workers-full N`, `--nice N`.
Use `--no-fetch` only when the base is already available locally.
The runner sets no time limit; on an unattended host, wrap the command in `timeout`.
Test targets are whitespace-separated, checkout-relative paths or node IDs.

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

Replacement inputs (the `-health-rerun1` suffix on `runId` or `dataSourceRunId` or a `healthReplacement` marker in either input) stop before command construction. Recorded answerer/judge execution provenance and dataset/scorer pins must agree across attempts; a field absent from both is not compared. The original report/checkpoint pair is validated before health selection, including healthy-only runs; validated checkpoint-only pins are retained in the merged report. The full-dataset `normalizedInputsSha256` and `loaderOrderSha256` pins must match across attempts, including presence on only one attempt. Only the selected-population `questionSetSha256` may differ, while each attempt's report/checkpoint selection pins must agree when present. Run-ledger `callCount`, `retryCount`, `executionIdentityCount`, and `callLedgerComplete` may differ; `mixedExecutionIdentity` remains compared. Null or non-object provenance blocks STOP with a field name. Each JSON receipt lists `provenance.comparedFields` and `provenance.absentOnBoth`, with a scope distinguishing cross-attempt comparison, original report/checkpoint validation, or validation not reached. Absent fields are listed from the writer-defined identity/provenance inventory; unknown recorded provenance keys are compared too. Category correctness, latency and retrieval, plus overall retrieval, are rebuilt from merged evaluations using the harness formulas. Overall latency, token, MemScore and CLI-ledger summaries are removed and listed in `aggregatesRemoved`: evaluation rows lack the full phase/token/call inputs needed to rebuild those summaries together.
