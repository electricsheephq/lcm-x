# Dispatch packet — scoreboard generator

Objective: `bench/scoreboard/generate.py` (python3.11+, stdlib only) reading
`bench/scoreboard/results.jsonl` plus new `bench/scoreboard/results.d/<id>.json`
(schema and combined digest: RESULTS-SCHEMA.md) and
emitting `bench/scoreboard/SCOREBOARD.md`.

Requirements:
1. FAIL CLOSED: any row missing a required schema field, or any malformed JSON
   line → exit 2 naming the line number and field; generate NOTHING partial.
2. Output structure:
   - Header: title + one-paragraph standard statement ("every number ships with
     its full run config, variance, fail-close accounting, and known dataset
     defects — rows that cannot meet the standard do not render") + generated-from
     line (combined source SHA256, row count) — NO wall-clock timestamp (byte-stable
     output for a given input).
   - Summary table: Benchmark | Metric | Result | Tier | Date | Details anchor.
     Sort: benchmark asc, then date desc. Superseded rows struck through
     (~~...~~) with link to successor.
   - Per-row disclosure sections (anchor = row id): every schema field rendered
     under bold labels; `evidence`, `caveats`, and optional `supersedes` as lists.
3. `--check`: require page == render(all sources), no write; exit 0/2.
   `--pending`: list source rows absent from the committed page as `pending: <id>` (exit 0),
   supersession-only changes from an unrendered successor as `pending: <id> (superseded by <successor>)` (exit 0),
   removed or otherwise mismatched rendered rows as `stale: <id>` (exit 2). Compare disclosures and summary values, ignoring the header.
   Supersession mismatches remain fatal if the successor is already rendered or removing supersession alone does not restore an exact match.
   Count IDs only in summary Details cells and disclosure anchors; each rendered ID must occur exactly once in each. Duplicates are stale. Reject supersession cycles before rendering, naming the cycle.
4. Tests in `bench/scoreboard/test_generate.py` (pytest): golden render of a
   2-row fixture; missing-field refusal; malformed-line refusal; supersession
   strike-through; byte-stable (two runs identical output).
5. Keep legacy results.jsonl unchanged; new runs use results.d/<id>.json.
   Validate filename/id equality and reject duplicate IDs across both sources.
   Existing row sections stay byte-identical; new rows link their own file.
6. Row PRs add only their `results.d/<id>.json` and their own evidence files; do NOT regenerate `SCOREBOARD.md`. Validate with `--pending`.
   The release manager regenerates the page in a separate batched PR at a release/showcase step or any time; that PR must pass strict `--check`.

Acceptance (from the repository root):
```
uv run --with pytest python3 -m pytest bench/scoreboard/test_generate.py -q -p no:xdist
python -m bench.scoreboard.generate --pending
# Separate batched aggregation PR only:
python -m bench.scoreboard.generate
python -m bench.scoreboard.generate --check
```
Bounded report: files + LOC, tests green, one rendered-row excerpt.
Scratch → local run `2026-07-30/hermes-scoreboard` (raw data kept locally; not published).
