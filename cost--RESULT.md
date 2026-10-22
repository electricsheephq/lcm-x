# e1b-s0-resume2-cost-paired-200

## Headline table

| benchmark | metric | value | n | baseline | receipt |
| --- | --- | --- | --- | --- | --- |
| 200-message ingest | baseline median ms | 7.2074585 | 200 | BASE 5df5a3c | raw-timings.jsonl |
| 200-message ingest | candidate median ms | 7.236625 | 200 | 7.2074585 ms | raw-timings.jsonl |
| Interleaved ingest pairs | paired delta median ms | +0.078103 | 200 | same fresh session shape | summary.json |
| P creation plus one UPDATE | candidate median ms | 0.203125 | 200 | reported, not gated | raw-timings.jsonl |

Gate PASS: candidate-minus-baseline medians = +0.0291665 ms (+0.405%), below
the allowed max(+10%, +0.05 ms) = +0.72074585 ms. One serial interleaved run;
each arm has 200 fresh 200-message sessions. IQRs: baseline 0.52371875 ms,
candidate 0.54056225 ms; hook 0.0359685 ms. No confidence interval is claimed.

Measured source: BASE `5df5a3c2736c01b46acf26906a510e41e367e73e` versus HEAD
`04627819e6599935ac38a62e6ea0d8e2fa213eb0` plus dirty working candidate SHA256
`554c626a2a7ce760577118343a5fe71ca4999241b84ee6d25080e8eaccb7daf3`.
The measured working candidate was not committed. Both arms use the same
fallback token counter; embeddings and temporal rollups are off.

## Run command

```sh
python resume2-paired-cost.py
```

## Re-score command

```sh
python resume2-score-cost.py .
```

## Not proven

Not proven: live-host latency, admission semantics, sender correctness, capture
rate, production throughput, release or runtime readiness. Local fixture cost only.

## Links

T1 and public-safe receipt publication are root-owned and pending. No T2 raw
publication, result deletion or scratch cleanup occurred in this lane.

