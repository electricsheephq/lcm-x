Refs #1811 #1814 #1817 #1818 #1801 #1804

Mixed quotes could date one event with another event's day. Share the date-text grammar and reviewed past-relative mention list between the resolver and compiler, and withhold uncertain quotes on ordinary and carried read paths. Normalize ordinal/weekday order labels and return `store_unavailable` for the auto-mode SQLite fallback. Stored rows stay unchanged.

A bounded `lru_cache(maxsize=4096)` reuses the pure quote-date parse across compiler candidates. No dependency, schema, ingest, engine or recall-source change.

## Rulings and scope

- Withhold only: no quote gains a day. A base-unknown resolver object stays byte-identical; each changed day object becomes a strictly smaller unknown object. One month-name date alone stays unsupported on ordinary rows; relative-only supported quotes retain their day.
- Twin parity: carried and ordinary rows share one date grammar and one mention list in `occurrence_time.py`. The resolvable `_RELATIVE` grammar and its resolution branch remain byte-identical. Yearless month-name dates remain unsupported.
- Amendment 1: annotations can affect delivery byte budgets. Recall never sorts or filters by resolved occurrence day. Hits admission is prefix-greedy; session admission skips oversized groups and continues, so membership displacement is accepted. No recall-budget change in this lane.
- Amendment 2 accepts this candidate's census: LongMemEval-S loses 10/106 explicit days (nine mixed absolute/relative, one distinct-absolute quote), although that exceeds the original 5% surprise guard. Every change is day to unknown and strictly smaller; no unknown object changes. The tradeoff is less dated evidence with reader fallback for mixed quotes.
- Amendment 3 authorizes exactly one performance correction. Cache the unchanged pure parse; one new paired cost run passes. The five combined-run recall failures reproduce at base in the identical file order and are pre-existing order dependence; the release manager owns that follow-up. Session cap-3340's non-superset result is also accepted; the release manager owns the existing final-refit gap, which drops admitted groups without retrying earlier skipped groups.

answer-quality effect not measured; the next D8/LME rows will show it.

`POLICY_VERSION` remains `occurrence-time-v1` to preserve Amendment 1's byte identity. The Python consumer grep found no occurrence-policy persistence/equality consumer; other policy domains are distinct. Reasoning compares source/day using the same resolver on both sides. Pack creation and grounding execute successfully at head with zero writes.

## Evidence

Claim class: advisory — read-path correctness with local proof and a counts-only corpus census. Ready for the requested review; no merge-readiness claim.

[Base](https://github.com/electricsheephq/lcm-x/commit/9eacedb9a667e517e40b9ed5c1725aa4f44df1cf) · [tests-first commit](https://github.com/electricsheephq/lcm-x/commit/1d804aabd) · [original proven head](https://github.com/electricsheephq/lcm-x/commit/e329f7e82ce0269d048870d326a94676eb36face) · [candidate](https://github.com/electricsheephq/lcm-x/commit/c993816bec8e78c0c46c0b1a5b67f03afd00893e).

### RED/GREEN and negative probes

| Issue / control | RED / characterization | GREEN |
| --- | --- | --- |
| #1811 ordinary absolute + relative | Original strict xfail becomes XPASS(strict) under the fix | Marker removed; characterization passes |
| #1814 ISO + named conflict | Base answers the wrong day; two-ISO control supplies parity | Carried/ordinary conflict withholding and exclusion pass |
| #1817 reviewed mention groups | New phrase/spelling/path cells fail at base | Shared mention group × ISO/long/abbreviated/no-comma/ordinal × carried/ordinary × date_filter/count/order controls pass |
| #1818 labels and SQLite reason | Base ordinal/weekday labels and reason fail | Label twins and all eight injected error paths pass |
| Tests-first combined new regressions | `339 failed, 40 passed in 27.67s` | Included in 643 focused passes |
| Resolver-only in-memory revert at original proven head | `5 failed in 0.93s` (expected negative for #1811/#1814) | Genuine assertion failures, no import/collection error |
| Mention-list-only in-memory revert at original proven head | `2 failed, 2 passed in 0.93s` (expected negative for #1817) | Existing recognized-phrase controls remain |

Exact-head focused command (the isolated scratch root is supplied through `LCMX_SCRATCH`):

```sh
python -m pytest tests/test_issue_1811_occurrence_time_batch.py tests/test_issue_1801_twin_parity.py tests/test_issue_1801_carried_followups.py tests/test_issue_1779_carried_time.py tests/test_occurrence_time.py -q --tb=short -p no:cacheprovider --basetemp="$LCMX_SCRATCH/pt-focused" -o tmp_path_retention_policy=failed
```

```text
643 passed, 9 warnings in 27.89s
```

The 542-object snapshot covers table/new-test quotes, base-unknown byte identity and strict shrink for every changed result. The 150-cell prior carried table passes; all twelve pinned ordinary serialization hashes stay unchanged. **Changed pinned hashes: none.** Three recall golden JSON files are byte-identical to base; **golden diffs: none**.

The identical matched file list and order on baseline and candidate:

```sh
python -m pytest bench/fmp/tests/test_session_delivery.py tests/behaviour/test_recall_delivery_contract.py tests/behaviour/test_recall_pipeline_matrix.py tests/test_evidence_compiler.py tests/test_evidence_pack.py tests/test_issue_1507_carried_time.py tests/test_issue_1776_round1.py tests/test_issue_1779_carried_time.py tests/test_issue_1779_fix_round2.py tests/test_issue_1801_distinct_dates.py tests/test_issue_1801_twin_parity.py tests/test_issue_1811_occurrence_time_batch.py tests/test_lcm_recall.py tests/test_occurrence_time.py tests/test_preanswer_evidence.py tests/test_reasoning.py -q --tb=short -p no:cacheprovider --basetemp="$LCMX_SCRATCH/pt-matched-ARM" -o tmp_path_retention_policy=failed
```

```text
base: 329 failed, 840 passed, 1 xfailed, 3 warnings in 34.68s
head: 5 failed, 1165 passed, 3 warnings in 31.82s
```

At base, only the two newly introduced test inputs are temporarily overlaid so the identical file list can collect; baseline source and existing tracked tests remain unchanged. New regression failures are expected there. The same five recall failures occur at original head, base and candidate: all four provider-alias cases of `test_provider_alias_uses_existing_vector_corpus_for_fts_fairness`, and `test_preflight_scan_expiry_fails_open_to_full_fts_budget`; each observes coverage `none` versus `full`. These five pass in isolation on base and original head. This combined command is **not green**; Amendment 3 accepts the demonstrated pre-existing order dependence.

The single full CI-style suite already ran at `e329f7e8`, with four workers and a 900-second ceiling; Amendment 3 requires only focused and matched reruns after the cache correction:

```sh
ulimit -n 1024
python -m pytest tests/ -q -m 'not latency' -n 4 --ignore=tests/test_embedding_privacy.py -p no:cacheprovider --basetemp="$LCMX_SCRATCH/pt-full-ci" -o tmp_path_retention_policy=failed
python -m pytest tests/test_embedding_privacy.py -q -m 'not latency' -p no:cacheprovider --basetemp="$LCMX_SCRATCH/pt-privacy" -o tmp_path_retention_policy=failed
```

```text
17632 passed, 141 skipped, 63 xfailed, 32 warnings in 619.24s (0:10:19)
77 passed in 2.54s
```

At candidate:

```sh
uv tool run --from ruff==0.15.13 ruff check . --no-cache
git diff --check
git diff 9eacedb9a667e517e40b9ed5c1725aa4f44df1cf HEAD --check
```

```text
All checks passed!
git diff --check: exit 0, no output
base-to-head diff --check: exit 0, no output
```

### Counts-only census and accepted threshold

Census card: `20261022-occurrence-time-census`. The resolver is byte-identical between the original accepted census head and candidate; both input digests still match. No corpus rescan or corpus-scope change for the performance correction. Counts include message occurrences, including repeated LongMemEval sessions; no corpus text or raw identifiers are published.

| Corpus | Messages | Base explicit | Explicit to unknown | Share | Changed objects | Max changed JSON-size delta |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| LongMemEval-S | 246750 | 106 | 10 | 9.433962%, accepted by Amendment 2 | 137 | -38 bytes |
| LoCoMo | 5882 | 0 | 0 | N/A | 1 | -48 bytes |

Unknown-identity violations, newly dated rows, changed-to-day objects and non-shrink changes: **zero in both corpora**.

LongMemEval-S transitions:

| Base source / reason | Head source / reason | Count |
| --- | --- | ---: |
| unknown / no_supported_occurrence_time | unknown / no_supported_occurrence_time | 240706 |
| relative_to_session | relative_to_session | 5064 |
| unknown / ambiguous_relative_expression | unknown / ambiguous_relative_expression | 616 |
| unknown / ambiguous_multiple_explicit_dates | unknown / ambiguous_multiple_explicit_dates | 131 |
| relative_to_session | unknown / ambiguous_relative_expression | 107 |
| explicit | explicit | 96 |
| relative_to_session | unknown / mixed_absolute_and_relative_dates | 17 |
| explicit | unknown / mixed_absolute_and_relative_dates | 9 |
| relative_to_session | unknown / ambiguous_multiple_explicit_dates | 3 |
| explicit | unknown / ambiguous_multiple_explicit_dates | 1 |

LoCoMo transitions:

| Base source / reason | Head source / reason | Count |
| --- | --- | ---: |
| unknown / no_supported_occurrence_time | unknown / no_supported_occurrence_time | 5744 |
| relative_to_session | relative_to_session | 136 |
| unknown / ambiguous_relative_expression | unknown / ambiguous_relative_expression | 1 |
| relative_to_session | unknown / ambiguous_relative_expression | 1 |

### Cost before and after the one correction

Bar per paired run: head minus base p50 ≤ max(10% of base, 0.05 ms). Both measurements are sub-1 ms. Point estimates only; confidence interval N/A. Same harness and identical synthetic quote inputs; each row compares its own contemporaneous base arm.

| Run / card | Surface | Base p50 ms | Head p50 ms | Delta ms | Bar ms | n/arm | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| before: 20261022-occurrence-time-cost | resolver | 0.0039170 | 0.0142080 | +0.0102910 | 0.0500000 | 5000 | PASS |
| before: 20261022-occurrence-time-cost | compiler | 0.1222085 | 0.2152920 | +0.0930835 | 0.0500000 | 200 | FAIL |
| after: 20261022-occurrence-time-cost-amendment3 | resolver | 0.0030420 | 0.0093340 | +0.0062920 | 0.0500000 | 5000 | PASS |
| after: 20261022-occurrence-time-cost-amendment3 | compiler | 0.1126670 | 0.1428540 | +0.0301870 | 0.0500000 | 200 | PASS |

Resolver: fixed 1,000 synthetic quotes, equal single-ISO/month-name/relative-only/mixed/no-date groups; five interleaved rounds, alternating arm per quote. Compiler: one paired run per candidate, ten AB/BA blocks of twenty calls/arm, ten excluded warmups, existing 1801 ordinary fixture, common character counter, fixed question/date, retrieval disabled. Setup/RPC excluded; no measured store writes. Pure parsing cache bounded at 4096 quotes; no second optimization or unchanged timing repeat. Checked result cards retain raw samples and re-score inputs.

### Recall delivery membership and both cap scans

Semantic isolation: [ranking precedes dispatch](https://github.com/electricsheephq/lcm-x/blob/c993816bec8e78c0c46c0b1a5b67f03afd00893e/tools.py#L4390), [hits occurrence annotations](https://github.com/electricsheephq/lcm-x/blob/c993816bec8e78c0c46c0b1a5b67f03afd00893e/recall_delivery.py#L235), [session annotations](https://github.com/electricsheephq/lcm-x/blob/c993816bec8e78c0c46c0b1a5b67f03afd00893e/recall_session.py#L204). No sort/filter reads a resolved day. [Hits admission breaks at the first non-fitting hit](https://github.com/electricsheephq/lcm-x/blob/c993816bec8e78c0c46c0b1a5b67f03afd00893e/recall_delivery.py#L291), giving the prefix/superset property for shrink-only annotations. [Session admission skips and continues](https://github.com/electricsheephq/lcm-x/blob/c993816bec8e78c0c46c0b1a5b67f03afd00893e/recall_session.py#L287); its final refit can remove admitted groups without retrying skipped ones. No recall source changed.

One fixed synthetic three-session fixture, public `lcm_recall`, annotations enabled, zero recall store writes. Listed IDs are synthetic positions. All fixed-cap and cap-scan outputs at candidate equal the original proven head. At hits cap 4760, `[]` becomes `[1]`; at session cap 3340, `[3]` becomes `[]`. The latter non-superset is the accepted existing refit behavior from Amendment 3.

Fixed-cap delivered IDs:

| Delivery | Cap | Base IDs | Head IDs |
| --- | ---: | --- | --- |
| hits | 4000 | [] | [] |
| hits | 5000 | [1] | [1] |
| hits | 6000 | [1] | [1] |
| hits | 8000 | [1, 2, 3] | [1, 2, 3] |
| hits | 64000 | [1, 2, 3] | [1, 2, 3] |
| session | 4000 | [2] | [2] |
| session | 5000 | [1] | [1] |
| session | 6000 | [1, 3] | [1, 3] |
| session | 8000 | [1, 2, 3] | [1, 2, 3] |
| session | 64000 | [1, 2, 3] | [1, 2, 3] |

hits cap scan (character cap, step 20; stops at first differing membership):

| Cap | Base IDs | Head IDs |
| ---: | --- | --- |
| 3000 | [] | [] |
| 3020 | [] | [] |
| 3040 | [] | [] |
| 3060 | [] | [] |
| 3080 | [] | [] |
| 3100 | [] | [] |
| 3120 | [] | [] |
| 3140 | [] | [] |
| 3160 | [] | [] |
| 3180 | [] | [] |
| 3200 | [] | [] |
| 3220 | [] | [] |
| 3240 | [] | [] |
| 3260 | [] | [] |
| 3280 | [] | [] |
| 3300 | [] | [] |
| 3320 | [] | [] |
| 3340 | [] | [] |
| 3360 | [] | [] |
| 3380 | [] | [] |
| 3400 | [] | [] |
| 3420 | [] | [] |
| 3440 | [] | [] |
| 3460 | [] | [] |
| 3480 | [] | [] |
| 3500 | [] | [] |
| 3520 | [] | [] |
| 3540 | [] | [] |
| 3560 | [] | [] |
| 3580 | [] | [] |
| 3600 | [] | [] |
| 3620 | [] | [] |
| 3640 | [] | [] |
| 3660 | [] | [] |
| 3680 | [] | [] |
| 3700 | [] | [] |
| 3720 | [] | [] |
| 3740 | [] | [] |
| 3760 | [] | [] |
| 3780 | [] | [] |
| 3800 | [] | [] |
| 3820 | [] | [] |
| 3840 | [] | [] |
| 3860 | [] | [] |
| 3880 | [] | [] |
| 3900 | [] | [] |
| 3920 | [] | [] |
| 3940 | [] | [] |
| 3960 | [] | [] |
| 3980 | [] | [] |
| 4000 | [] | [] |
| 4020 | [] | [] |
| 4040 | [] | [] |
| 4060 | [] | [] |
| 4080 | [] | [] |
| 4100 | [] | [] |
| 4120 | [] | [] |
| 4140 | [] | [] |
| 4160 | [] | [] |
| 4180 | [] | [] |
| 4200 | [] | [] |
| 4220 | [] | [] |
| 4240 | [] | [] |
| 4260 | [] | [] |
| 4280 | [] | [] |
| 4300 | [] | [] |
| 4320 | [] | [] |
| 4340 | [] | [] |
| 4360 | [] | [] |
| 4380 | [] | [] |
| 4400 | [] | [] |
| 4420 | [] | [] |
| 4440 | [] | [] |
| 4460 | [] | [] |
| 4480 | [] | [] |
| 4500 | [] | [] |
| 4520 | [] | [] |
| 4540 | [] | [] |
| 4560 | [] | [] |
| 4580 | [] | [] |
| 4600 | [] | [] |
| 4620 | [] | [] |
| 4640 | [] | [] |
| 4660 | [] | [] |
| 4680 | [] | [] |
| 4700 | [] | [] |
| 4720 | [] | [] |
| 4740 | [] | [] |
| 4760 | [] | [1] |

session cap scan (character cap, step 20; stops at first differing membership):

| Cap | Base IDs | Head IDs |
| ---: | --- | --- |
| 3000 | [3] | [3] |
| 3020 | [3] | [3] |
| 3040 | [3] | [3] |
| 3060 | [3] | [3] |
| 3080 | [3] | [3] |
| 3100 | [3] | [3] |
| 3120 | [3] | [3] |
| 3140 | [3] | [3] |
| 3160 | [3] | [3] |
| 3180 | [3] | [3] |
| 3200 | [3] | [3] |
| 3220 | [3] | [3] |
| 3240 | [3] | [3] |
| 3260 | [3] | [3] |
| 3280 | [3] | [3] |
| 3300 | [3] | [3] |
| 3320 | [3] | [3] |
| 3340 | [3] | [] |

Testing-improved: `test_shared_grammar_mixed_withheld_and_recorded`, `test_order_label_shared_grammar_parity`, the base-unknown/strict-shrink snapshot, and counts-only `census.py` expose mixed-spelling and past-relative leakage.

Production delta: 130 changed lines (+87/-43), within the 180-line envelope; seven tracked paths, all source or tests authorized by the batch. Full diff +210/-47, including the one-line generated snapshot. No stored-row change.

Not proven: answer-quality improvement, cold-cache or production-workload latency, exact-candidate full suite, independent semantic review, exact-head hosted CI, merge, release, installed runtime or real customer behavior. Raw result bundles remain preserved; registry/archive publication is outside this source lane. Requested next gate: one Codex review of this ready PR, then return to the release manager.
