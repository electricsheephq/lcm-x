# #1586 paired offline gap-discovery probe

Claim class: advisory mechanism proof.

Baseline: `e1dc2a164c66d2419d8d580852d69eb851f95eed`. Candidate: `eb6a8aee77beba0130b78596ae30b986850a36b0`.
Candidate `command.py` SHA256: `2e3939d7feaad65290004c25ba5e49d60d1a361fcf98ae02b14a1d523e7267a4`.
The raw candidate run recorded the base SHA before commit; its source hash matches the committed candidate exactly.

| cell | ordinary gap chunkings | retry gap chunkings | productive slice seconds: mean (range) | n | receipt |
| --- | --- | --- | --- | --- | --- |
| empty / stock | [0, 0, 0] | [0, 0, 0] | 0.020320 (0.019867–0.020774) | 3 count samples; 2 timing samples | [raw](stock-probe.json) |
| empty / fixed | [0, 0, 0] | [0, 0, 0] | 0.019424 (0.018703–0.020145) | 3 count samples; 2 timing samples | [raw](fixed-probe.json) |
| upgrade / stock | [4000, 4000, 4000] | [2000, 2000, 2000] | 0.984998 (0.975386–0.994611) | 3 count samples; 2 timing samples | [raw](stock-probe.json) |
| upgrade / fixed | [22, 23, 22] | [2, 2, 2] | 0.034052 (0.033391–0.034712) | 3 count samples; 2 timing samples | [raw](fixed-probe.json) |

Same machine, stock then candidate, each with fresh deterministic 6,000-row mixed stores and fake offline embeddings. Socket access was denied. The first productive slice was profiled and excluded from timing. Initial selected-document/meta hashes match between stock and candidate for both store shapes. Timings have no confidence interval; this small mechanism check does not measure real embedding or full-store catch-up performance.

## Commands

Use the existing repository test environment and host stub. `PROBE_REPO`, `PROBE_HOST`, `PROBE_SCRATCH`, `PROBE_OUT`, and `PROBE_SHA` select each isolated checkout/store/output.

```sh
python paired-probe-source.txt
python rescore.txt
```

`rescore.txt` recomputes `paired-summary.json` solely from the two retained raw JSON files, including bound and selection-equality assertions.

Not proven: 72-hour acceptance, production throughput, recall latency, runtime/customer readiness, default flip, merge or release readiness.

T1/T2: not published by this bounded source lane. All raw/results are retained; no cleanup authorized or performed. A SHA-pinned PR evidence branch supplies the public card/receipts.
