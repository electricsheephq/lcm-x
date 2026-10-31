# Scoreboard results schema (Track F — disclosure-first public scores page)

New BANKED runs go in `results.d/<id>.json`, one JSON object per file;
the filename stem must equal the row's `id`. The frozen legacy `results.jsonl`
remains readable, including appends from PRs already in flight. The page generator
renders a summary table plus a per-row disclosure block. **A row without its
disclosure fields does not render** — the schema IS the seven-point standard
(F46 §5); numbers never appear bare.

## Required fields
| Field | Meaning |
|---|---|
| `id` | stable row id (`<benchmark>-<variant>-<date>`) |
| `benchmark` | public benchmark name + variant (e.g. `LongMemEval-V1 (S, 500q)`) |
| `metric` | what the number measures (`accuracy`, `latency_s_per_q`, `p50_ms`) |
| `value` | the measured value (number) |
| `display` | human form (e.g. `455/500 (91.0%)`) |
| `tier` | `P` (published-claims discipline) or `F` (fault-finding, frontier-first) |
| `date` | run completion date (ISO); release-backed backfills may use publication day if completion day is absent, disclosed in `caveats` |
| `system_commit` | product commit(s) measured, with any post-run delta named |
| `harness_commit` | harness/adapter commit |
| `judge` | judge identity + where its FULL prompt lives (repo path) |
| `reader` | answering model + effort/config |
| `retrieval_config` | delivery/fusion/embedder/cap config, every knob named |
| `dataset_exposure` | known dataset defects exposure (corrupted-gold ceiling etc.) or `none documented` |
| `breakdown` | per-category numbers or artifact path holding them. This field is a string; put structured data in the evidence file and link it here. |
| `variance` | run count + variance/noise-floor statement (A/A′ discordants, p-values) |
| `failclose` | fail-close accounting (rows excluded/scored-zero by instrument fault) |
| `evidence` | non-empty list of durable URLs backing every claim above (except the frozen legacy IDs below): GitHub release-download asset URLs or blob URLs pinned to a full 40-hex commit SHA |
| `caveats` | anything a skeptical reader needs (or `[]`) |

## Optional fields
- `local_raw`: list of relative machine-local raw artifact paths/references, for locating raw
  material only; these are not durable citations. Repository files belong in
  `evidence` as SHA-pinned blob URLs. Absolute paths (POSIX or Windows) and
  home-relative paths starting with `~` are rejected for every row.
- `superseded_by`: id of the successor row.
- `supersedes`: list of old row IDs on a new `results.d` row. The loader applies `superseded_by: <new id>` in memory; legacy JSONL stays frozen and the old row strikes through as before.
  Rendered as a list in the successor's disclosure. Unknown IDs, self-reference,
  supersession cycles, two different successors for one row, or disagreement with an existing `superseded_by` are errors.

## Rules
- Only the frozen 18 IDs in `generate.py`'s `LEGACY_ROW_IDS` (the registry at
  commit `764ed74c6142141a6d2102dd0c99eb84806f7f66`) are grandfathered for
  evidence validation. Every other row, including backfills with an older date,
  needs a non-empty `evidence` list whose entries all match
  `https://github.com/OWNER/REPO/releases/download/TAG/ASSET` or
  `https://github.com/OWNER/REPO/blob/FULL_COMMIT_SHA/PATH` (optionally followed
  by a line anchor such as `#L12` or `#L12-L20`). Branch-name blobs, local paths,
  and dot segments (`.` or `..`, including percent-encoded forms) are rejected.
  Legacy repository references are converted to pinned blob URLs; only local raw
  artifact references move to `local_raw`. Measured values stay unchanged.
- Rows are APPEND-ONLY; a superseding run gets a new row with `supersedes`,
  preserving the old row and its history without editing legacy JSONL.
- Row PRs add only their row file and their own evidence; do NOT regenerate `SCOREBOARD.md`.
  Run `python -m bench.scoreboard.generate --pending`: missing source rows and supersession-only changes from unrendered successors are pending (exit 0); the old row is listed with its successor ID.
  Removed rows, other rendered-value changes, supersession mismatches with already-rendered successors, and mismatches not explained by supersession alone are stale (exit 2).
  Each rendered ID must occur exactly once in the summary's Details cells and exactly once in the disclosures; duplicates are stale. Link-like text in other summary cells is not an ID reference.
  The release manager regenerates in a separate batched PR at a release/showcase step or any time, then requires strict `--check` equality.
- Tier-F rows carry their fault-finding yield in `caveats`/`evidence` — a low
  number with its decomposition is a feature of this page, not an embarrassment.
- No projections, no adjusted-pending-measurement numbers, ever (§6e.8).

## Source order and digest
The loader reads legacy non-empty lines in line order, then `results.d/*.json`
in filename order. Duplicate IDs anywhere are hard errors; supersession can
cross sources. Summary order stays benchmark ascending, date descending, id
ascending; disclosures stay id ascending. Existing evidence links, including
`results.jsonl#L<n>`, stay unchanged; file-based disclosures link their row file.

The header's combined SHA256 is SHA256 of UTF-8 compact JSON (`ensure_ascii=True`,
separators `(',', ':')`) containing `[relative_filename, SHA256(raw_file_bytes)]`
pairs: `results.jsonl` first, then `results.d/*.json` sorted by filename. File
names and exact bytes participate; a missing or empty `results.d` adds no pairs.
