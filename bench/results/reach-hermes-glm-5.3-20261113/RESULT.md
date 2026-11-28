# Returning-work reach in Hermes, glm-5.3

Run `returning-job-reach-hermes-first-cut-20261113`. Claim class: advisory. Run date 2026-11-13; card written 2026-11-14T11:47:23Z.
Measured: LCM-X 0.30.0 runtime payload from `1fc6112e38a11e99080ccc0a9288b93aea396cd7`, in Hermes v0.23.1 (`3b421578588243f920886e6fe79a6faa87ea026e`). Model glm-5.3 through the provider's coding-plan API, reasoning at the Hermes default. Study kit build 6, manifest sha256 `7f570609365bd89c297c58ac1f3ca1465d9c40700c49a738f095c2bc6c23a59a`; seeded store sha256 `2d301b849af73185c554ccaa2082d3835a2dbc76b53defe542acd76cdda08696`.

Each trial is one first reply to a returning puzzle whose id the agent worked on in earlier sessions held in the store. Designs: the unchanged plugin; one added instruction to search the exact id; one added instruction to recall the id; a card the kit prepared and put in view. Each design ran on the real id and, where listed, on a made-up id of the same form.

Headline: with the plugin unchanged, the agent reached its earlier work with the id on 6 of 25 returning ids.

| Design | Id | n | Reached | Reached with the id | Right at once | Wrong at once | Asked for the examples | No reply | Median s |
|---|---|---|---|---|---|---|---|---|---|
| unchanged plugin | real id | 25 | 8 | 6 | 4 | 3 | 18 | 0 | 99.04 |
| instruction, exact search | made-up id | 25 | 9 | 9 | 0 | 5 | 19 | 0 | 91.66 |
| instruction, exact search | real id | 25 | 10 | 10 | 8 | 2 | 15 | 0 | 169.31 |
| instruction, recall | made-up id | 25 | 7 | 7 | 0 | 1 | 24 | 0 | 73.44 |
| instruction, recall | real id | 25 | 12 | 12 | 8 | 4 | 12 | 1 | 150.61 |
| card | made-up id | 25 | 3 | 3 | 0 | 0 | 25 | 0 | 63.21 |
| card | real id | 25 | 3 | 3 | 19 | 4 | 1 | 1 | 166.25 |

| Bar (declared before the run) | Verdict | Receipt |
|---|---|---|
| reach under an instruction (better of the two instruction designs, real ids) | fail, clear at this size | SUMMARY.json |
| card against the better instruction design | not decidable | SUMMARY.json |
| wrong first answers after a made-up id (instructions and card) | fail, clear at this size | SUMMARY.json |
| self-hits on exact searches that pass the exclusion flag | pass | SUMMARY.json |

Rows: 175 read, 175 final, 0 infrastructure failures, 0 left out, 2 with no reply within the limit (counted, never right or wrong).

Re-score: the retained raw rows run through the study's frozen analysis script (sha256 `5b8e5acfdcb321b4deb1723233f98f58453381519638d4f6f0c77bdcf3316538`): `python3 -I -B analyze_first_cut.py --trials trials.jsonl --report-json report.json --items items.json --out-dir OUT`. It reproduces the retained analysis outputs byte for byte. This directory is a public summary; the trial rows hold raw model replies and are kept privately, cited here only by digest (SUMMARY.json). T2 is unpublished. Discussion: https://github.com/electricsheephq/lcm-x/issues/2654.

Not proven: one run of 25 ids per design on one model; the card was prepared by the kit, not built by the product; cells ran in a fixed order, not interleaved; puzzles of one public benchmark, not customer work; token figures are not included here; no whole-job, default-change or release claim.
