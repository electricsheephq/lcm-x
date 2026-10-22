# 20261022-occurrence-time-fix-r2-public

Claim class: advisory — local fixture proof for review only.
Code: `907043cb7ba9dbe5c5158b801a61acca0f9898c4`. Baseline: `9eacedb9a667e517e40b9ed5c1725aa4f44df1cf`; default-mode predecessor: `e06cc78879cc345c153a9e6afdc5eb3cde638c2e`.

| Proof | Result | Receipt |
| --- | --- | --- |
| Full non-date base identity | PASS: 7656 / 7656 | [cells](nondate-cell-hashes.json), [summary](nondate-summary.json) |
| Date-dependent predecessor identity | PASS: 14552 / 14552; no new answers/day changes vs base | [cells](date-cell-hashes.json), [summary](date-summary.json) |
| Both resolver modes | PASS: 292632 comparisons each; zero differences | [counts/digests](resolver-identity.json) |
| Pinned hashes / recall goldens | unchanged | [inventory](golden-pinned-inventory.json) |
| Matched suite | not green; all 8 failures reproduced at base | [disposition](matched-disposition.json) |
| Full suite | see exact result and disposition | [disposition](full-disposition.json) |

| Operation / input refs | Warm base ms | Warm head ms | Delta ms | Bar ms | n/arm | Gate | Receipt |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| date-2 | 0.104125 | 0.106688 | +0.002563 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| date-6 | 0.173104 | 0.179812 | +0.006708 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| date-10 | 0.229500 | 0.237958 | +0.008458 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| date-12 | 0.325375 | 0.341104 | +0.015729 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| date-24 | 0.316667 | 0.323958 | +0.007292 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| order-2 | 0.175063 | 0.177500 | +0.002437 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| order-6 | 0.281438 | 0.271562 | -0.009875 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| order-10 | 0.421791 | 0.409562 | -0.012229 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| order-12 | 0.509083 | 0.493563 | -0.015520 | 0.050908 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| order-24 | 0.437001 | 0.422104 | -0.014897 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| interval-2 | 0.184500 | 0.189688 | +0.005188 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| interval-6 | 0.499104 | 0.520416 | +0.021312 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| interval-10 | 0.745812 | 0.780312 | +0.034500 | 0.074581 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| interval-12 | 0.521271 | 0.533937 | +0.012666 | 0.052127 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| interval-24 | 0.578896 | 0.606834 | +0.027938 | 0.057890 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| finite-2 | 0.474187 | 0.483791 | +0.009604 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| finite-6 | 0.976042 | 0.964000 | -0.012042 | 0.097604 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| finite-10 | 1.360979 | 1.366271 | +0.005292 | 0.136098 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| finite-12 | 1.695375 | 1.618354 | -0.077021 | 0.169538 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| finite-24 | 2.156687 | 2.301833 | +0.145146 | 0.215669 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| nondate-2 | 0.209146 | 0.210687 | +0.001542 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| nondate-6 | 0.355750 | 0.362854 | +0.007104 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| nondate-10 | 0.581458 | 0.585542 | +0.004083 | 0.058146 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| nondate-12 | 0.323271 | 0.327750 | +0.004479 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |
| nondate-24 | 0.342625 | 0.350875 | +0.008250 | 0.050000 | 200 | PASS | [samples](cost-ruling-b-raw.json) |

One interleaved AB/BA run. Cold values are report-only in the cost summary and proof receipt. Point estimates; confidence intervals N/A. Inputs/clock match; ordinary hydration cap stays 12; no measured store writes. All source digests match the final commit.

## Re-score

```sh
python3 rescore.txt
```

Re-score reads retained outputs, not a product/runtime rerun. Corpus byte-identity reruns require the original authorized datasets at the recorded digests; corpus text and identifiers are excluded from the packet.

Not proven: answer-quality improvement, production workload latency, independent semantic review, hosted exact-head CI, merge, release, installed runtime or customer behavior.

T1: not published; registry publication is outside this source lane.
T2: not published; archive publication is outside this source lane. No cleanup or packing performed.

[Proof receipt](proof-receipt.md)
