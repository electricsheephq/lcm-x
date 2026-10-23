# Session-end attribution: current-main cost

Baseline:440976dc4ca05caa8e82ccc8e7adcb167fa70bd9; candidate:2819b6f89861c56b6fa896821a25b27f24d84ac7.

Run:https://github.com/electricsheephq/lcm-x/actions/runs/39921293650

| Scope | Level | Base median ms | Candidate median ms | Paired delta ms | Allowed ms | Result | Receipt |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| preflight | INFO | 0.413261 | 0.414472 | 0.002483 | 0.050000 | PASS | [summary](summary.json),[pairs](pairs.jsonl) |
| post_llm_call_ingest | INFO | 0.410961 | 0.413345 | 0.003079 | 0.050000 | PASS | [summary](summary.json),[pairs](pairs.jsonl) |
| preflight | DEBUG | 0.433686 | 0.446084 | 0.012564 | 0.050000 | PASS | [summary](summary.json),[pairs](pairs.jsonl) |
| post_llm_call_ingest | DEBUG | 0.435243 | 0.445598 | 0.010075 | 0.050000 | PASS | [summary](summary.json),[pairs](pairs.jsonl) |

Each cell:200 alternating AB/BA pairs after10warmups. Synthetic warm24-row store; embeddings and temporal rollups disabled in both arms. Threshold:max(10% of baseline,0.05ms), applied to marginal and paired median deltas. Preflight covers full warm preflight; post-turn covers ingest span only. Hook dispatch, turn-complete notification, leaf preparation and host/model work excluded. DEBUG includes stream-handler formatting.

Re-score:python score.py. Measurement source:measurement-source.py.

Not proven:whole-hook latency, live slow-commit attribution, speedup, release, runtime or customer behavior.

T1:historical dispatch comparison remains separately registered; no refreshed scoreboard row. T2:unpublished; no result disposal.
