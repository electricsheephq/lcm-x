# Dispatch packet — scoreboard generator

Objective: `bench/scoreboard/generate.py` (python3.11+, stdlib only) reading
`bench/scoreboard/results.jsonl` (schema: RESULTS-SCHEMA.md, seeds present) and
emitting `bench/scoreboard/SCOREBOARD.md`.

Requirements:
1. FAIL CLOSED: any row missing a required schema field, or any malformed JSON
   line → exit 2 naming the line number and field; generate NOTHING partial.
2. Output structure:
   - Header: title + one-paragraph standard statement ("every number ships with
     its full run config, variance, fail-close accounting, and known dataset
     defects — rows that cannot meet the standard do not render") + generated-from
     line (results.jsonl sha256, row count) — NO wall-clock timestamp (byte-stable
     output for a given input).
   - Summary table: Benchmark | Metric | Result | Tier | Date | Details anchor.
     Sort: benchmark asc, then date desc. Superseded rows struck through
     (~~...~~) with link to successor.
   - Per-row disclosure sections (anchor = row id): every schema field rendered
     under bold labels; `evidence` as a list; `caveats` as a list.
3. `--check` flag: validate only, no write; exit 0/2.
4. Tests in `bench/scoreboard/test_generate.py` (pytest): golden render of a
   2-row fixture; missing-field refusal; malformed-line refusal; supersession
   strike-through; byte-stable (two runs identical output).
5. Do NOT modify results.jsonl / RESULTS-SCHEMA.md. No git commands. Write only
   generate.py, test_generate.py, and the generated SCOREBOARD.md.

Acceptance (from repo root /Volumes/LEXAR/hermes-work/wt-ci-fix):
```
uv run --with pytest python3 -m pytest bench/scoreboard/test_generate.py -q -p no:xdist
python3 bench/scoreboard/generate.py --check && python3 bench/scoreboard/generate.py
```
Bounded report: files + LOC, tests green, one rendered-row excerpt.
Scratch → /Volumes/LEXAR/Codex/session-notes/2026-07-30/hermes-scoreboard/artifacts/ (create), never /tmp.
