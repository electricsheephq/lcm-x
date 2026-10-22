# 20261022-occurrence-time-cost-amendment3

Claim class: advisory. Head c993816bec8e78c0c46c0b1a5b67f03afd00893e; base 9eacedb9a667e517e40b9ed5c1725aa4f44df1cf.

| Surface | Base p50 ms | Head p50 ms | Delta ms | Bar ms | n/arm | Gate | Receipt |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| resolver | 0.003042000 | 0.009334000 | 0.006292000 | 0.050000000 | 5000 | PASS | resolver-raw.json |
| compiler | 0.112667000 | 0.142854000 | 0.030187000 | 0.050000000 | 200 | PASS | compiler-raw.json |

One authorized performance correction: bounded pure quote parsing cache. Same harness and synthetic inputs as the failed prior run, same method, one new paired run. Resolver: 1000 quotes in five equal groups (ISO, month-name, relative, mixed, no date), five rounds interleaved per quote. Compiler: ten alternating AB/BA blocks, twenty calls/arm each, ten excluded warmups, existing 1801 ordinary fixture, setup/RPC excluded, no store writes. Point estimates, no confidence bounds. Paired membership outputs are retained separately; this card covers cost.

Run: python paired_proof.py from the proof artifact directory, pinned code worktree as cwd. Re-score: `python3 rescore.py` from this run folder, reads resolver-raw.json and compiler-raw.json. Synthetic quote input hash matches prior measurement.

Not proven: answer quality, cold-cache production workloads, independent review, hosted CI, merge, release, runtime or customer behavior.

T1: unpublished (no registry change in authorized source scope).
T2: unpublished; all result and re-score files stay in place.
