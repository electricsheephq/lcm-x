# Returning-work reach in Hermes, gpt-6.1-sol

Run `returning-work-reach-hermes-kit7-gpt-6.1-sol-20261113`. Claim class: advisory. Run date 2026-11-13; card written 2026-11-14T11:47:23Z.
Measured: LCM-X 0.30.0 runtime payload from `1fc6112e38a11e99080ccc0a9288b93aea396cd7`, in Hermes v0.23.1 (`3b421578588243f920886e6fe79a6faa87ea026e`). Model gpt-6.1-sol through Hermes's Codex subscription route, effort medium and tier priority as sent. Study kit build 7, manifest sha256 `3e44bc40d295cacbaf339656731177ec1ad77ac70eeb6078737464c749678fe4`; seeded store sha256 `2d301b849af73185c554ccaa2082d3835a2dbc76b53defe542acd76cdda08696`.

Each trial is one first reply to a returning puzzle whose id the agent worked on in earlier sessions held in the store. Designs: the unchanged plugin; one added instruction to search the exact id; one added instruction to recall the id; a card the kit prepared and put in view; the same card with a second wording of its "nothing stored" line. Each design ran on the real id and, where listed, on a made-up id of the same form.

Headline: with the plugin unchanged, the agent reached its earlier work with the id on 20 of 25 returning ids.

| Design | Id | n | Reached | Reached with the id | Right at once | Wrong at once | Asked for the examples | No reply | Median s |
|---|---|---|---|---|---|---|---|---|---|
| unchanged plugin | made-up id | 25 | 19 | 19 | 0 | 0 | 25 | 0 | 18.14 |
| unchanged plugin | real id | 25 | 20 | 20 | 17 | 2 | 6 | 0 | 29.79 |
| instruction, exact search | made-up id | 25 | 25 | 25 | 0 | 0 | 25 | 0 | 18.48 |
| instruction, exact search | real id | 25 | 25 | 25 | 24 | 1 | 0 | 0 | 30.38 |
| instruction, recall | made-up id | 25 | 25 | 25 | 0 | 0 | 25 | 0 | 18.49 |
| instruction, recall | real id | 25 | 25 | 25 | 25 | 0 | 0 | 0 | 31.28 |
| card | made-up id | 25 | 0 | 0 | 0 | 0 | 25 | 0 | 16.03 |
| card | real id | 25 | 2 | 2 | 24 | 1 | 0 | 0 | 24.01 |
| card with the second miss-line wording | made-up id | 25 | 0 | 0 | 0 | 0 | 25 | 0 | 16.12 |

| Bar (declared before the run) | Verdict | Receipt |
|---|---|---|
| reach under an instruction (better of the two instruction designs, real ids) | pass at this size | SUMMARY.json |
| card against the better instruction design | pass | SUMMARY.json |
| card against the better instruction design: right at once | pass | SUMMARY.json |
| card against the better instruction design: the user's wait | pass, clear at this size | SUMMARY.json |
| wrong first answers after a made-up id (instructions and card) | pass at this size | SUMMARY.json |
| wrong first answers after a made-up id (card, second miss-line wording) | pass at this size | SUMMARY.json |
| self-hits on exact searches that pass the exclusion flag | pass | SUMMARY.json |
| self-hits on exact searches that pass the exclusion flag, real ids only | pass | SUMMARY.json |
| miss line against the unchanged plugin on made-up ids (control, no verdict) | no difference seen at this size | SUMMARY.json |
| second miss-line wording against the unchanged plugin on made-up ids (control, no verdict) | no difference seen at this size | SUMMARY.json |

Rows: 225 read, 225 final, 0 infrastructure failures, 0 left out, 0 with no reply within the limit (counted, never right or wrong).

Re-score: the retained raw rows run through the study's frozen analysis script (sha256 `6e66c9bf4be4130905f32e6e419304eaef1cd3744a0a41c5cf2be30206416531`): `python3 -I -B analyze_kit7.py --trials trials.jsonl --report-json report.json --items items.json --bars BARS-kit7.md --model gpt-6.1-sol --luna-json analysis-hermes-kit7-gpt-6-luna.json --out-dir OUT`. It reproduces the retained analysis outputs byte for byte. This directory is a public summary; the trial rows hold raw model replies and are kept privately, cited here only by digest (SUMMARY.json). T2 is unpublished. Discussion: https://github.com/electricsheephq/lcm-x/issues/2654.

Not proven: one run of 25 ids per design on one model; the card was prepared by the kit, not built by the product; cells ran in a fixed order, not interleaved; puzzles of one public benchmark, not customer work; token figures are not included here; no whole-job, default-change or release claim.
