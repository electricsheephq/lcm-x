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
