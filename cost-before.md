# 20261022-occurrence-time-cost

Claim class: advisory; synthetic read-path latency only. Candidate e329f7e82ce0269d048870d326a94676eb36face; base 9eacedb9a667e517e40b9ed5c1725aa4f44df1cf.

| Surface | Base p50 ms | Head p50 ms | Delta ms | Allowed delta ms | n/arm | Gate | Receipt |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| resolver | 0.003917 | 0.014208 | 0.010291 | 0.05 | 5000 | PASS | resolver-raw.json |
| ordinary compiler | 0.1222085 | 0.215292 | 0.0930835 | 0.05 | 200 | FAIL | compiler-raw.json |

Resolver: fixed 1,000 quotes, five interleaved rounds, alternating base/head within every quote. Quotes are synthetic, with equal single-ISO, named-month, relative-only, mixed, and no-date groups. Each arm receives the same session date and observed_at. Per-round p50s are in summary.json.

Compiler: one paired run, ten alternating AB/BA blocks, twenty calls per block/arm. Existing 1801 plant builder, ordinary baseline refs only; an unrelated carried control remains outside those refs. Same synthetic texts, same two ordinary baseline refs, same question/date, retrieval disabled. Separate base/head fixture stores; setup and ten warmups are excluded. Token counter pinned to the shared fallback. No store writes during measured calls. Point estimates only; no confidence interval or production timing claim.

Run: python paired_proof.py (original script retained; do not rerun into this settled folder).
Re-score: `python3 rescore.py` from this run directory. Inputs: resolver-raw.json, compiler-raw.json. All measurement and scorer files are preserved.

Not proven: answer-quality effect, remote CI, merge readiness, release, runtime/customer behavior, or timing outside this synthetic fixture. The compiler cost gate failed.

T1: unpublished; no repository write.
T2: unpublished; results stay in place and are not eligible for cleanup.
