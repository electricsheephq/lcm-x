# Session-end attribution: ordinary-turn cost

Synthetic warm 24-row store; 200 alternating baseline/candidate pairs per cell after 10 warmups. Baseline ca98ba6aa217f5db31ec3e4c32a9eff84cf3d2d2; measured candidate ec11b30a3a96ac27fe2f81021a3a6254a2a6ef72.

Run: https://github.com/electricsheephq/lcm-x/actions/runs/39918175350/job/120296672627

| Scope | Log level | Base median ms | Candidate median ms | Paired median delta ms | Allowed ms | Result | Receipt |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| preflight | INFO | 0.775940 | 0.779201 | 0.004043 | 0.077594 | PASS | [summary](summary.json), [pairs](pairs.jsonl) |
| post_llm_call_ingest | INFO | 0.763808 | 0.766803 | 0.003702 | 0.076381 | PASS | [summary](summary.json), [pairs](pairs.jsonl) |
| preflight | DEBUG | 0.812618 | 0.835006 | 0.024706 | 0.081262 | PASS | [summary](summary.json), [pairs](pairs.jsonl) |
| post_llm_call_ingest | DEBUG | 0.817553 | 0.837551 | 0.021259 | 0.081755 | PASS | [summary](summary.json), [pairs](pairs.jsonl) |

The preflight scope is the full warm preflight. The post-turn scope is the ingest span, excluding hook dispatch, turn-complete notification, leaf preparation, and host/model work. DEBUG includes formatting to a stream handler. INFO ships the session-end record and disables per-turn span clocks. Both marginal median delta and paired median delta must meet max(10% of baseline, 0.05 ms).

Re-score: `python score.py`. Measurement source: [measurement-source.py](measurement-source.py).

The later shallow-copy correction restores the base copy contract; it does not touch either measured path. No measured speedup is claimed.

Not proven: whole-hook/whole-host latency, live slow-commit attribution, release, runtime or customer behavior.

T1: pending scoreboard registration. T2: unpublished; no result disposal authorized. All raw pairs and re-score inputs are retained.
