# Scoreboard results schema (Track F — disclosure-first public scores page)

One JSON object per line in `results.jsonl` = one BANKED run. The page generator
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
| `date` | run completion date (ISO) |
| `system_commit` | product commit(s) measured, with any post-run delta named |
| `harness_commit` | harness/adapter commit |
| `judge` | judge identity + where its FULL prompt lives (repo path) |
| `reader` | answering model + effort/config |
| `retrieval_config` | delivery/fusion/embedder/cap config, every knob named |
| `dataset_exposure` | known dataset defects exposure (corrupted-gold ceiling etc.) or `none documented` |
| `breakdown` | per-category numbers or artifact path holding them |
| `variance` | run count + variance/noise-floor statement (A/A′ discordants, p-values) |
| `failclose` | fail-close accounting (rows excluded/scored-zero by instrument fault) |
| `evidence` | artifact paths (fork docs branch) backing every claim above |
| `caveats` | anything a skeptical reader needs (or `[]`) |

## Rules
- Rows are APPEND-ONLY; a superseding run gets a new row and the old row gains
  `superseded_by: <id>` (history stays visible — that is the point).
- Tier-F rows carry their fault-finding yield in `caveats`/`evidence` — a low
  number with its decomposition is a feature of this page, not an embarrassment.
- No projections, no adjusted-pending-measurement numbers, ever (§6e.8).
