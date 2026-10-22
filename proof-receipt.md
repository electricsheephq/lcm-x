# PR #1826 — Fix round 3 stopped

Claim class: advisory — local fixture and timing evidence only. Result: VERIFIED_FAILURE; candidate TERMINAL under the exact final-round SPEC.

The final optimization attempt passes all 15 reviewer long-row order gates but fails one of the original 25 warm gates. `nondate-10`: base 0.586479 ms, R3 0.6541665 ms, delta +0.0676875 ms, allowed +0.0586479 ms. The SPEC says any warm failure after one attempt stops this round and parks the PR as a draft. No timing rerun, further source attempt, successor, or remaining acceptance run was started.

Input/public PR head remains `907043cb7ba9dbe5c5158b801a61acca0f9898c4`. Local tests-only commits: `0a7431c23701a1dc655b651035933cef5e82ab39`, `e67889835227e49bb939c9bb7f74512b3a690873`, `1ca8b9322d2f51c5a43f37e77c9b5bdecbccc738`. The three-file source candidate remains uncommitted and unpushed; `candidate-patch.txt` and `candidate-binding.json` preserve its exact bytes. Production delta: +27/-12, 39 changed lines, inside the 80-line signal. No merge/rebase of main.

## Implemented candidate and caller modes

1. The date-dependent source pre-loop records ambiguous-relative, ambiguous-explicit and mixed-date resolver withholds as `uncertain_source_observation_time` exclusions. The resolver and its mode decisions are unchanged.
2. Long-row order names use a digest-keyed map capped at 4,096 entries, each value None or at most 256 characters. Large stripped quotes are never retained there. The existing strip/normalization functions remain unchanged. The final name is computed with the same expression/filter as the predecessor.
3. `lcm_compute` supplies `withhold_mixed = plan.temporal_window is not None or plan.operation in {date_filter, order, date_interval}` to grounding.

| Caller | Candidate mode |
| --- | --- |
| Compiler normalization / date candidates / carried helpers / availability / latest selection | Ruling B contract date dependence; non-date uses base semantics (unchanged from round 2) |
| Compiler compute / finite grounding | Contract date dependence (unchanged from round 2) |
| `lcm_compute` grounding | Plan date dependence, same rule; new in round 3 |
| `reasoning` wrapper and `_ground_one` | Forward the supplied mode; default True (unchanged) |
| `evidence_pack` legacy proposal path | Default True, separate from active AnswerContract compiler (unchanged) |
| `recall_delivery` / `recall_session` | Default True under accepted Amendment 1 (unchanged) |

## Tests and unfinished proof

Tests were committed before production changes. The first two invocations had collection/setup errors (missing host PYTHONPATH, then missing basetemp parent); the next run collected and returned `11 failed in 6.59s`. That RED run included genuine refusal/compute failures and an absent cache helper, plus test errors: a question without a date compiled as operation none, the compute result is under trace and formatted as `$175`, and `We` is excluded from names. Test-only correction commits fixed these expectations. No production revision was made after the single candidate patch.

The focused command was:

```sh
python -m pytest tests/test_issue_1811_round3.py tests/test_issue_1811_contract_mode.py tests/test_issue_1811_occurrence_time_batch.py tests/test_issue_1801_twin_parity.py tests/test_issue_1801_carried_followups.py tests/test_issue_1779_carried_time.py tests/test_occurrence_time.py -q -n 4 --tb=short -p no:cacheprovider --basetemp="$LCMX_SCRATCH/pt-focused" -o tmp_path_retention_policy=failed
```

```text
2309 passed, 9 warnings in 17.33s
```

This includes four named Louvre/Prado/Uffizi order/date-filter cells, all six named non-date compute sum cases, the bounded/reused name test, the committed 1,500-cell non-date guard and prior pinned/twin controls. The 7,656-cell full capture, 9,633-cell reviewer capture, both 292,632-comparison resolver proofs, date matrices, 30,000-label comparison, realistic report, matched suite, full CI-style suite and Ruff were NOT RUN after the hard stop. Recall goldens and pinned fixtures were not edited. Their full requested runtime proof is incomplete. `git diff --check` returned exit 0 before the cost run; no claim of final lint or readiness.

## Warm long-row gates

250 scored samples per arm/cell; original reviewer driver/worker copied before adaptation; three arms interleaved AB/BA, setup outside timings. Includes long and longiso (~1.7 KB) plus long8k (~8.5 KB); all output digests equal across base, 907043cb and R3. Point estimates only, no confidence-bound claim.

| Fixture | Refs | Base ms | 907043cb ms | R3 ms | R3 delta ms | Bar ms | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| long | 2 | 1.2268 | 1.7757 | 1.0154 | -0.2113 | 0.1227 | PASS |
| long | 6 | 4.8352 | 13.4496 | 5.2474 | +0.4121 | 0.4835 | PASS |
| long | 10 | 6.6359 | 8.6787 | 4.7161 | -1.9198 | 0.6636 | PASS |
| long | 12 | 6.3528 | 11.0037 | 6.1388 | -0.2140 | 0.6353 | PASS |
| long | 24 | 8.5963 | 22.8498 | 8.2186 | -0.3776 | 0.8596 | PASS |
| longiso | 2 | 0.8062 | 1.2816 | 0.8421 | +0.0359 | 0.0806 | PASS |
| longiso | 6 | 4.3978 | 12.9831 | 2.7002 | -1.6975 | 0.4398 | PASS |
| longiso | 10 | 8.4102 | 17.2548 | 5.5994 | -2.8108 | 0.8410 | PASS |
| longiso | 12 | 6.6968 | 11.4012 | 4.1340 | -2.5629 | 0.6697 | PASS |
| longiso | 24 | 5.9538 | 11.4521 | 3.9131 | -2.0406 | 0.5954 | PASS |
| long8k | 2 | 4.3020 | 7.4656 | 2.9445 | -1.3575 | 0.4302 | PASS |
| long8k | 6 | 20.7087 | 26.3231 | 16.3022 | -4.4065 | 2.0709 | PASS |
| long8k | 10 | 32.8340 | 44.6751 | 31.7637 | -1.0703 | 3.2834 | PASS |
| long8k | 12 | 33.6403 | 50.1446 | 20.6030 | -13.0373 | 3.3640 | PASS |
| long8k | 24 | 22.7684 | 33.5036 | 16.0594 | -6.7090 | 2.2768 | PASS |

## Original 25 gates

| Operation | Refs | Base warm ms | R3 warm ms | Delta ms | Bar ms | Gate | Base cold ms | R3 cold ms |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| date | 2 | 0.116208 | 0.120480 | +0.004271 | 0.050000 | PASS | 0.118646 | 0.169459 |
| date | 6 | 0.208896 | 0.218333 | +0.009437 | 0.050000 | PASS | 0.170875 | 0.278958 |
| date | 10 | 0.233875 | 0.245479 | +0.011604 | 0.050000 | PASS | 0.263646 | 0.453500 |
| date | 12 | 0.259833 | 0.271167 | +0.011334 | 0.050000 | PASS | 0.278979 | 0.466750 |
| date | 24 | 0.292166 | 0.297521 | +0.005354 | 0.050000 | PASS | 0.277708 | 0.481604 |
| order | 2 | 0.170688 | 0.173563 | +0.002875 | 0.050000 | PASS | 0.177667 | 0.220021 |
| order | 6 | 0.283104 | 0.279958 | -0.003146 | 0.050000 | PASS | 0.273458 | 0.367833 |
| order | 10 | 0.480312 | 0.468521 | -0.011791 | 0.050000 | PASS | 0.394396 | 0.536583 |
| order | 12 | 0.466166 | 0.451854 | -0.014313 | 0.050000 | PASS | 0.412854 | 0.579979 |
| order | 24 | 0.452896 | 0.445791 | -0.007105 | 0.050000 | PASS | 0.428312 | 0.598500 |
| interval | 2 | 0.191417 | 0.197667 | +0.006250 | 0.050000 | PASS | 0.188459 | 0.242834 |
| interval | 6 | 0.496437 | 0.489395 | -0.007042 | 0.050000 | PASS | 0.507104 | 0.709750 |
| interval | 10 | 0.721271 | 0.767416 | +0.046145 | 0.072127 | PASS | 0.716584 | 0.951937 |
| interval | 12 | 0.494917 | 0.511583 | +0.016666 | 0.050000 | PASS | 0.514250 | 0.796771 |
| interval | 24 | 0.574625 | 0.588251 | +0.013625 | 0.057463 | PASS | 0.542917 | 0.793083 |
| finite | 2 | 0.476125 | 0.480542 | +0.004417 | 0.050000 | PASS | 0.402500 | 0.452292 |
| finite | 6 | 0.964062 | 0.960625 | -0.003437 | 0.096406 | PASS | 0.922666 | 1.048145 |
| finite | 10 | 1.368708 | 1.456687 | +0.087979 | 0.136871 | PASS | 1.304229 | 1.505500 |
| finite | 12 | 1.529479 | 1.537563 | +0.008084 | 0.152948 | PASS | 1.460730 | 1.664896 |
| finite | 24 | 2.157376 | 2.135855 | -0.021521 | 0.215738 | PASS | 2.200042 | 2.485750 |
| nondate | 2 | 0.173417 | 0.178979 | +0.005562 | 0.050000 | PASS | 0.180667 | 0.179104 |
| nondate | 6 | 0.363792 | 0.357791 | -0.006001 | 0.050000 | PASS | 0.423208 | 0.417791 |
| nondate | 10 | 0.586479 | 0.654166 | +0.067688 | 0.058648 | FAIL | 0.620021 | 0.626938 |
| nondate | 12 | 0.299000 | 0.299355 | +0.000355 | 0.050000 | PASS | 0.310229 | 0.307896 |
| nondate | 24 | 0.340729 | 0.343292 | +0.002563 | 0.050000 | PASS | 0.316875 | 0.317396 |


200 scored samples per arm/cell, interleaved twenty ten-call blocks after 30 warmups. Cold results report only. The exact round-2 harness was copied with only the scratch destination changed. The candidate did not pass all warm gates. No causality or noise ruling is inferred from the failed point estimate.

## Realistic first-seen report

NOT RUN: the warm-gate stop fired first. No base / 907043cb / R3 realistic claim is available.

## Recommended next steps

Keep #1826 draft. The release manager should use this failed card and preserved patch to decide a separately scoped successor, preserving the non-date parity obligation and the successful long-row design. First inspect this exact failed cell and proposed scope; then issue a new written SPEC with candidate identity, allowed mechanism and validation budget before another attempt. This lane has no authority for that successor. Resume only after that new instruction; do not rerun until green or publish the current candidate as validated.

Testing improved: multi-relative refusal cells, long-row order cost gates, and the lcm_compute non-date parity cells.

Not proven: the complete identity matrices, resolver/label runtime byte proof, realistic first-seen cost, full/remote CI, independent semantic review, answer quality, merge, release, runtime or customer behavior. No comments, replies, resolutions, review requests, enqueue or merge were performed. T1/T2 registry/archive publication is outside this source lane; public-safe summaries are published in the same evidence-branch format as round 2.
