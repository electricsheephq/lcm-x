# Scale389 benchmark instrument

This is the promoted F34 Phase-1B scale probe. It is bench-side only and uses
Python's standard library (the product's existing embedding provider may use
NumPy during a real query run).

`phase1a.py` preserves `all_gold` as a compatibility alias and emits the
explicit session-granularity name `session_gold_all`. It also persists the
delivered hit content and emits `answer_turn_delivered_complete`, computed by
matching LongMemEval `has_answer=true` turn content against that payload. The
join verifies every hit through the store's `.dates.json` sidecar and never
uses ingest position.

The u-arm population is configurable with `--uncensored-n` and defaults to 50,
the full primary set. Each arm emits `probe-questions-<arm>.txt`, a sha256 pin
file, and a passing pin-verification report before queries begin. An existing
probe list is immutable: a changed selection fails closed.

Build an enriched corpus with `python bench/instruments/scale389/build_corpus.py`.
Run the sequential chain with `run_scale.py ingest` or `run_scale.py queries`;
optional pins and store manifests are checked through `bench/tools/pinverify.py`
and `bench/tools/storefreeze.py`. Query summaries use
`bench/tools/failclose.py` signatures and refuse fail-closed metric rows.

Offline acceptance:

```sh
python bench/instruments/scale389/archive_regression.py \
  --output /Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-r3-1/artifacts/laneINSTPREP-logs/archive-regression.json
pytest -q bench/instruments/scale389/tests
```

The regression recomputes F34's A3/B published session table from archived
per-question JSONL, checks all archived summary fields exactly, snapshots the
archive before and after, and exercises the new turn join on a real F34 primary
question plus its archived sidecar with controlled incomplete/complete payloads.
The controlled payload check is not claimed as a reconstructed F34 turn metric:
F34 did not persist delivered content.
