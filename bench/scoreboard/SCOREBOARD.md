# Scoreboard

Every number ships with its full run config, variance, fail-close accounting, and known dataset defects — rows that cannot meet the standard do not render.

Generated from `results.jsonl` (sha256: `1c208c8506850f0838dde19ec8c28bd56a328699c88a7dd039a928606a17561a`, rows: 11)

## Summary

| Benchmark | Metric | Result | Tier | Date | Details |
|---|---|---|---|---|---|
| AMA-Bench (open-ended, 208 episodes) | strict judged accuracy, fail-closed denominator | 1180/2496 = 47.3% (scored-only 1180/2460 = 48.0%) | F | 2026-08-02 | [details](#ama208-full-2026-08-02) |
| Internal scale instrument (389×, 19,829 sessions / 199,641 messages) | p50_ms | 263.6 ms @ 19,829 sessions (24× vs pre-fix; 125 ms @ 8k) | P | 2026-07-30 | [details](#scale-curve-fastscan-2026-07-30) |
| LoCoMo (locomo10, 1,986q) | accuracy | 67.4% (1,339/1,986 MemScore) | F | 2026-08-26 | [details](#locomo10-1986-b3a-2026-08-26) |
| LoCoMo (locomo10, 1,986q) | accuracy | 54.6% (1,085/1,986 MemScore) | F | 2026-08-02 | [details](#locomo10-1986-declared-2026-08-02) |
| ~~LoCoMo (locomo10, 1,986q)~~ | ~~accuracy~~ | ~~47.0% (MemScore)~~ | ~~F~~ | ~~2026-07-30~~ | ~~[details](#locomo10-1986-arm-a-2026-07-30)~~ → [successor](#locomo10-1986-declared-2026-08-02) |
| LongMemEval-V1 (S, 500q) | accuracy | 455/500 (91.0%) | P | 2026-07-29 | [details](#longmemeval-v1-s500-accuracy-2026-07-29) |
| LongMemEval-V1 (S, 500q) | latency_delta_s_per_q | −56.3 s/question vs vanilla (22% faster) | P | 2026-07-29 | [details](#longmemeval-v1-s500-latency-2026-07-29) |
| LongMemEval-V1 MEDIUM (500q, retrieval row) | lcm_recall (primary arm): recall@10 / ndcg@10 / recall@1, fail-closed | recall@10 95.6% \| ndcg@10 0.861 \| recall@1 50.0% (470 scored + 30 abstention-excluded; 0 instrument failures) | P | 2026-08-19 | [details](#longmemeval-v1-m500-retrieval-2026-08-19) |
| LongMemEval-V2 (451q, agentic) | accuracy | 298/451 (66.1%) | P | 2026-07-27 | [details](#longmemeval-v2-451-agentic-2026-07-27) |
| LongMemEval-V2 (451q, static) | accuracy | 123/451 (27.3%) — official static protocol, fixed weak reader | P | 2026-07-31 | [details](#longmemeval-v2-451-static-2026-07-31) |
| LongMemEval-V2 static (451q) | judged accuracy, full set | 143/451 = 31.7% (web 97/240 = 40.4%, enterprise 46/211 = 21.8%) | F | 2026-08-03 | [details](#v2-static-451-sota-luna-2026-08-03) |

## Row disclosures

### <a id="ama208-full-2026-08-02"></a>ama208-full-2026-08-02

**id:**
ama208-full-2026-08-02

**benchmark:**
AMA-Bench (open-ended, 208 episodes)

**metric:**
strict judged accuracy, fail-closed denominator

**value:**
0.4728

**display:**
1180/2496 = 47.3% (scored-only 1180/2460 = 48.0%)

**tier:**
F

**date:**
2026-08-02

**system_commit:**
hermes-lcm product via wt-ama-adapter c06d6a5; clone ddfd319e + 2 disclosed patches

**harness_commit:**
AMA-bench ddfd319e0be33424288c13806f1eafc63e625b59

**judge:**
anthropic/claude-sonnet-5 (cross-family; judge-fail-closed patch applied, 0 unparsed)

**reader:**
openai/gpt-5.6-luna (default effort; harness exposes no effort surface — disclosed)

**retrieval_config:**
hermes_lcm method overlay, per-episode ingest, fastembed local

**dataset_exposure:**
dataset sha256 45c36052e...; no training/tuning on it

**breakdown:**
tiers: small<100k 1026/2172=47.2%, mid 134/252=53.2%, large>800k 20/36=55.6% \| worst tasks: alfworld 18.4%, swebench 43.8%, webarena 47.3% \| best: candy_crush 75.0%, crafter 66.7%, gaia_level3 65.0%

**variance:**
A/A' 30-episode fixed-seed subset (seed 20260802) EXECUTED: 27/30 pairs compared (3 skipped where either arm had a failed episode, disclosed), 324 questions, 31/324 discordant (9.6%), aggregate spread 1.54pt (A-subset 45.4% vs A' 46.9%) — question-level churn ~10%, aggregate stable

**failclose:**
3/208 episodes failed (luna empty-content completions x3 retries; AttributeError in harness model_client.py:109) — 36 questions counted in denominator, never dropped

**evidence:**
- bench/specs/RUN-SHEET-AMA-208.md
- bench/FINDING-F51-AMA-208-FIRST-FULL.md
- session-notes 2026-08-02 hermes-ama-full208 (per-episode results, timing, telemetry)

**caveats:**
- First full-208 disclosed number; GOALS target >=72.26% NOT met (gap -24.9pt); funnel decomposition queued
- Accuracy RISES with episode size (large>800k tier 55.6% vs small 47.2%) — no context-scale degradation
- Reader effort = model default (AMA custom-provider path exposes no effort surface)

### <a id="locomo10-1986-arm-a-2026-07-30"></a>locomo10-1986-arm-a-2026-07-30

**id:**
locomo10-1986-arm-a-2026-07-30

**benchmark:**
LoCoMo (locomo10, 1,986q)

**metric:**
accuracy

**value:**
0.47

**display:**
47.0% (MemScore)

**tier:**
F

**date:**
2026-07-30

**system_commit:**
R2-era product build 543e9ea (pre-#183 prose mode, pre-#197 chunk threshold)

**harness_commit:**
pre-fix memorybench adapter (bugs enumerated in F46 §1, fixed in PR #3 @19d9c0b)

**judge:**
sol·low, full prompts in src/prompts/defaults.ts (pinned sha in data/pins-locomo.yaml); STRICTER than the stock LoCoMo judge (community audit measured the stock judge accepting 62.81% of intentionally-wrong-vague answers)

**reader:**
sol·medium (pre two-tier doctrine; declared re-run uses a frontier answerer)

**retrieval_config:**
fastembed bge-small, top-25 cap, RRF fusion (pre-quota); every defect class measured in F46 §2/§7-9

**dataset_exposure:**
99/1,540 documented corrupted-gold questions all ran; we scored wrong on 80; 6/15 sampled failures match the audit-CORRECTED answer; known-corruption ceiling 95.02% for this slice

**breakdown:**
per-category (ground-truth labels, corrected 2026-08-02): single-hop 29.1 / multi-hop 49.2 / temporal 43.8 / world 52.9 / adversarial 45.5 — an earlier published tuple scrambled the first three labels (F46 CORRECTION)

**variance:**
arm A of a pre-registered A/A' pair; A' cancelled when harness bugs were found mid-pair (documented in run dir) — noise floor re-runs on the fixed harness

**failclose:**
pin discipline PASS; truncation bug meant 5.7% of delivered results were silently cut (fixed PR #3)

**evidence:**
- bench/FINDING-F46-LOCOMO-ARM-A-DECOMPOSITION (full §1-9 ledger)
- session-notes 2026-07-30 hermes-locomo-deepdive artifacts

**caveats:**
- published AS THE FAULT-FINDING RESULT IT IS: decomposition attributes the gap to instrument bugs (~+3pts ceiling), corrupted gold (~1.5-2pts), a config-class retrieval gap (fusion quota + chunk threshold, both since landed), and one genuine answer-layer weakness (adversarial speaker attribution — fact retrieved on 78.6% of wrong rows); declared-config re-run pending
- this row is the disclosure standard demonstrated on our own worst number

**superseded_by:**
locomo10-1986-declared-2026-08-02

### <a id="locomo10-1986-b3a-2026-08-26"></a>locomo10-1986-b3a-2026-08-26

**id:**
locomo10-1986-b3a-2026-08-26

**benchmark:**
LoCoMo (locomo10, 1,986q)

**metric:**
accuracy

**value:**
0.6742

**display:**
67.4% (1,339/1,986 MemScore)

**tier:**
F

**date:**
2026-08-26

**system_commit:**
hermes-lcm fork main @ 9d181aa (identical product tree to the F48 declared row — B3-A is a bridge-side ingest delta)

**harness_commit:**
memorybench wt-locomo-prep @ d305d590 (C1-era prep 2c36f98 + B3-A attribution cherry-pick 6233661; HERMES_MB_SPEAKER_PREFIX=1 is the single delta vs F48)

**judge:**
gpt-5.6-sol @ low; full prompts pinned (defaults.ts 7662f6…) — same judge and rubric as F48 (C1 pins amendments 1-4 inherited as the baseline environment)

**reader:**
gpt-5.6-sol @ medium (frontier answerer per two-tier doctrine, unchanged from F48)

**retrieval_config:**
F48 declared config verbatim (fastembed bge-small; fusion quota fts:chunk=1:2; conversational chunk threshold 10; answer-ready 2,400 chars) + HERMES_MB_SPEAKER_PREFIX=1: '<Speaker>: ' prefixes on user rows at ingest (SPEC-B3-ATTRIBUTION.md; 45-line bridge change)

**dataset_exposure:**
99 documented corrupted-gold rows all ran, scored as-is; known-corruption ceiling ≈95% (unchanged from F46 §6)

**breakdown:**
single-hop 45.4 (+8.2 vs F48) / multi-hop 71.7 (+6.9) / temporal 46.9 (+5.2) / world 78.2 (+8.5) / adversarial 62.3 (+29.6) — every category above F48 declared; band was adversarial ≥+4.0

**variance:**
pre-registered A/A′ pair on fresh stores: 82/1,986 discordant (4.13%), aggregate spread 0.30pt (A 67.42 / A′ 67.72); arm A is the scored read per the run sheet; A→A′ gap ≈110h (weaker pairing than C1's same-day pair — disclosed)

**failclose:**
0/1,986 incomplete both arms; paired failclose union-drop 0; PINS-POSTRUN PASS both arms (A′ postrun executed manually after the resume script exited pre-postrun — env pins verify a reconstruction from the script's own exports; sha-class pins verified against disk)

**evidence:**
- bench/FINDING-F61-LOCOMO-C2-B3A-ATTRIBUTION.md
- bench/specs/RUN-SHEET-LOCOMO-C2-B3A.md (#316)
- session-notes 2026-08-21 hermes-locomo-c2 artifacts (paid-aa-20260820T211813Z)
- session-notes 2026-08-26 interim-review artifacts F61-recompute.txt

**caveats:**
- attribution is the lever, robustness is the prize: the adversarial (unanswerable) category nearly doubles (32.7 → 62.3) — the F59 §9 failure class (entity+recency-matched confusions from misattributed facts) attacked directly; every other category also rises, consistent with attribution improving retrieval binding generally
- benchmark claim on the pinned trees only: B3-A lives in the benchmark bridge; the product ingest-path analog (#324 sender/timestamp provenance + #317 untrusted-evidence boundary) is the productization this unlocks and needs its own confirmation run (#379 epic)
- not a claim about current main: v0.23.x privacy changes landed after these pins (BASELINE-LEDGER privacy-trio boundary); the F53 re-bank (#380) governs the retrieval row, not this one
- A′ interruption chain disclosed in the finding §3: session teardown at search-phase start + resume-script death after 'Run complete!' — all checkpoint-recovered, 0 lost questions

### <a id="locomo10-1986-declared-2026-08-02"></a>locomo10-1986-declared-2026-08-02

**id:**
locomo10-1986-declared-2026-08-02

**benchmark:**
LoCoMo (locomo10, 1,986q)

**metric:**
accuracy

**value:**
0.5463

**display:**
54.6% (1,085/1,986 MemScore)

**tier:**
F

**date:**
2026-08-02

**system_commit:**
hermes-lcm fork main @ 9d181aa (post-#190/#197)

**harness_commit:**
memorybench feat/locomo-hermes-prep @ f55eba3 (+2c36f98 run-prep); all F46 §1 instrument fixes live

**judge:**
gpt-5.6-sol @ low; full prompts pinned (defaults.ts 7662f6…); narrowed abstention rubric (credits verifiable premise-rejection only) — stricter than the stock LoCoMo judge

**reader:**
gpt-5.6-sol @ medium (frontier answerer per two-tier doctrine)

**retrieval_config:**
fastembed bge-small; fusion quota fts:chunk=1:2 (measured selection, FUSION-EMBEDDER-DIAGNOSIS); conversational chunk threshold 10 (measured, CHUNK-ELIGIBILITY); answer-ready 2,400 chars; full pins in run artifacts

**dataset_exposure:**
99 documented corrupted-gold rows all ran, scored as-is; known-corruption ceiling ≈95% (unchanged from F46 §6)

**breakdown:**
single-hop 37.2 (+8.1) / multi-hop 64.8 (+15.6) / temporal 41.7 (−2.1) / world 69.7 (+16.8) / adversarial 32.7 (−12.8) — corrected table in F48 §3-CORRECTION

**variance:**
pre-registered A/A′ pair on fresh stores: 69/1,986 discordant (3.47%), aggregate spread 0.76 pts; arm A is the scored read per the run sheet

**failclose:**
0/1,986 both arms; union-drop 0; pins signed pre+post both arms

**evidence:**
- bench/FINDING-F48-LOCOMO-DECLARED-CONFIG.md
- bench/specs/RUN-SHEET-LOCOMO-DECLARED-CONFIG.md
- session-notes 2026-07-31 hermes-locomo-declared artifacts

**caveats:**
- +7.6 vs the superseded row is NOT pure retrieval: the judge rubric and adversarial gold changed between configs (both disclosed); corrupted gold unchanged (ceiling ≈95%)
- initial publication claimed a single-hop regression from a label-scramble — corrected within hours (F48 §3-CORRECTION); the REAL new finding: the FTS arm is near-inert in delivery on this dataset in BOTH configs (3/49,650 vs 2/49,650) despite prose mode — under investigation
- adversarial 32.7 is the measured-honest number against canonical abstention gold with the known B3 product weakness unfixed; Tier-F config; Voyage variant queues separately

### <a id="longmemeval-v1-m500-retrieval-2026-08-19"></a>longmemeval-v1-m500-retrieval-2026-08-19

**id:**
longmemeval-v1-m500-retrieval-2026-08-19

**benchmark:**
LongMemEval-V1 MEDIUM (500q, retrieval row)

**metric:**
lcm_recall (primary arm): recall@10 / ndcg@10 / recall@1, fail-closed

**value:**
0.9559

**display:**
recall@10 95.6% \| ndcg@10 0.861 \| recall@1 50.0% (470 scored + 30 abstention-excluded; 0 instrument failures)

**tier:**
P

**date:**
2026-08-19

**system_commit:**
lcm-x v0.22.0 (92fd69b7) — includes ported #198 instrument + #199 checkpoint/embed-cache

**harness_commit:**
in-repo instrument (benchmarking/longmemeval.py @ v0.22.0)

**judge:**
deterministic retrieval metrics (no LLM)

**reader:**
NONE — pure retrieval row (the QA/reader row is a separate future registration)

**retrieval_config:**
voyage-context-3 embeddings, 6 shards, warm content-hash cache (505,695 entries; determinism probe 20/20 bitwise); FTS prose flag unset (disclosed, F49-class)

**dataset_exposure:**
longmemeval_m sha fb5413e3, HF revision 2ec2a557, prepared manifest 300cf936 (LEXAR-recovered, re-verified); no tuning on it

**breakdown:**
arms: lcm_recall r@1 .500/r@10 .956/ndcg .861 \| hybrid_rerank .365/.779/.668 \| summary_vectors .363/.779/.667 \| chunk_vectors .301/.424/.427 \| hybrid_rrf .143/.697/.456 \| fts .007/.046/.026

**variance:**
A/A' 100q fixed-seed subset (seed 20260802): 95/95 scored questions per-question IDENTICAL, aggregate spread 0.00pt — fully deterministic instrument

**failclose:**
0/500 instrument failures; 30 abstention questions excluded by instrument design and counted

**evidence:**
- bench/FINDING-F53-V1M-FLAGSHIP-FIRST-ROW.md
- bench/specs/RUN-SHEET-V1M-REGISTERED.md
- session-notes 2026-08-19 v1m-launch2 artifacts + lme-runs/m-full2-shard-*

**caveats:**
- The 25q smoke's r@1 0.96 was a first-N sampling artifact (full-500 r@1 = 0.50); first-N smokes retired, seeded-random only
- Delivery (r@10 95.6%) clears the 90% line; top-slot ranking (r@1 50%) is the declared next lever
- FTS arm dark (prose flag unset) — disclosed; V1-M FTS-ON variant decided after LoCoMo C1
- FROZEN vs current main (caveat appended 2026-08-26 per PR #368 review): row measured pre-v0.23.1 privacy trio (#332/#333/#338); the production lcm_recall arm may not be compared against post-v0.23.1 production-path rows until the registered re-bank (BASELINE-LEDGER privacy-trio boundary; FINDING-F53 reproducibility note). Reproducibility on current main is UNVERIFIED, not proven broken.
- Re-bank status (caveat appended 2026-09-05 per FINDING-F62): the registered re-bank on the shipped posture (RUN-SHEET-V1M-REBANK, 22c12b21) PARKED at the pre-spend gate — the production provider-copy transform re-shapes 599 unit occurrences (248/500 questions) of this corpus and the residual validator refuses 3 distinct units (17 questions; §4.2 transform-change count 617 occurrences = 93 unique units, 258 questions), so the production lcm_recall arm is NOT EXECUTABLE over LongMemEval-M at 22c12b21; the raw path this row was measured on has not been re-run; the freeze above stands; next step is an owner decision (#380).

### <a id="longmemeval-v1-s500-accuracy-2026-07-29"></a>longmemeval-v1-s500-accuracy-2026-07-29

**id:**
longmemeval-v1-s500-accuracy-2026-07-29

**benchmark:**
LongMemEval-V1 (S, 500q)

**metric:**
accuracy

**value:**
0.91

**display:**
455/500 (91.0%)

**tier:**
P

**date:**
2026-07-29

**system_commit:**
2edb8fc (measured); review-round delta 2edb8fc→7d025e8 verified non-delivery-affecting (D27)

**harness_commit:**
fork train through 93a3ade; upstream PR #436 @ ccc76ee

**judge:**
LongMemEval official protocol judge; config in bench/release-kit (fork docs branch)

**reader:**
per official V1 protocol (fixed reader); full config in F37/release kit

**retrieval_config:**
wave-1 delivery (answer_ready, evidence_cards_v1); pins per F32 (transport version pinned, 6e.13)

**dataset_exposure:**
none documented

**breakdown:**
bench/FINDING-F37 §per-type (fork docs branch)

**variance:**
+11 vs banked 444 (b20/c9, p=0.061 REPORTED AS MEASURED, no significance claim); direction consistent across all three baselines; placebo flat; A/A' reference pair = 18 discordants

**failclose:**
0 fail-closes (F32's 8 destroyed qids all scored; 7 correct)

**evidence:**
- bench/FINDING-F37
- bench/release-kit/RELEASE-NOTES-R2
- upstream PR #436

**caveats:**
- single scored confirm run; paired vs banked baseline, not a fresh A/B

### <a id="longmemeval-v1-s500-latency-2026-07-29"></a>longmemeval-v1-s500-latency-2026-07-29

**id:**
longmemeval-v1-s500-latency-2026-07-29

**benchmark:**
LongMemEval-V1 (S, 500q)

**metric:**
latency_delta_s_per_q

**value:**
-56.3

**display:**
−56.3 s/question vs vanilla (22% faster)

**tier:**
P

**date:**
2026-07-29

**system_commit:**
R2 train (see accuracy row)

**harness_commit:**
same as accuracy row

**judge:**
n/a (latency)

**reader:**
same agent both arms

**retrieval_config:**
same as accuracy row; CONTENDED concurrency condition disclosed

**dataset_exposure:**
n/a

**breakdown:**
faster on 48/60 questions (M12)

**variance:**
p < 1e-4 paired

**failclose:**
n/a

**evidence:**
- bench/FINDING-M12 lineage in PROGRAM-ARCHITECTURE
- bench/release-kit/RELEASE-NOTES-R2

**caveats:**
- accuracy A/B on the same instrument was NULL (M12) — the established claim is latency, not accuracy; we publish both facts together

### <a id="longmemeval-v2-451-agentic-2026-07-27"></a>longmemeval-v2-451-agentic-2026-07-27

**id:**
longmemeval-v2-451-agentic-2026-07-27

**benchmark:**
LongMemEval-V2 (451q, agentic)

**metric:**
accuracy

**value:**
0.661

**display:**
298/451 (66.1%)

**tier:**
P

**date:**
2026-07-27

**system_commit:**
H6-P4 tree; tag bench-H6-P4-298

**harness_commit:**
lme-v2-official (fixed weak reader per official protocol); #166 instrument fix landed after this run

**judge:**
official V2 protocol judge (gpt-5.2 per benchmark default)

**reader:**
official fixed reader (weak by design — see caveats)

**retrieval_config:**
H6-P4 delivery; full config in the tagged tree

**dataset_exposure:**
none documented

**breakdown:**
per-ability table in F30/H6-P4 bench docs

**variance:**
single scored run; instrument-clean recount 298/444 = 67.1% excluding #166's 7 transport-zeroed rows

**failclose:**
7/451 rows scored 0 on EMPTY reader responses (HTTP 524, no retry) — instrument defect #166, fixed upstream-report #7; both numbers shown

**evidence:**
- bench/FINDING-F30
- upstream harness reports #6/#7
- tag bench-H6-P4-298

**caveats:**
- below the published 69.9 baseline for this protocol; V2 medium tier (7.4× store) never run by anyone — parked R3.2
- paired V2 re-baseline in flight (batch/v2-rebaseline gate)

### <a id="longmemeval-v2-451-static-2026-07-31"></a>longmemeval-v2-451-static-2026-07-31

**id:**
longmemeval-v2-451-static-2026-07-31

**benchmark:**
LongMemEval-V2 (451q, static)

**metric:**
accuracy

**value:**
0.2727

**display:**
123/451 (27.3%) — official static protocol, fixed weak reader

**tier:**
P

**date:**
2026-07-31

**system_commit:**
fork main @ 9d181aa (PR #190 V2 re-baseline batch: upstream delivery ports + D-ARCH-1/2/3)

**harness_commit:**
lme-v2-official @ 6bfd58a (#166 bounded-retry + instrument_failed exclusion, both arms)

**judge:**
official V2 protocol judge

**reader:**
official fixed reader (weak by design; static protocol measures retrieval+delivery, not frontier answering)

**retrieval_config:**
answer_ready delivery, full pin set (pins-treatment.yaml in run artifacts)

**dataset_exposure:**
none documented

**breakdown:**
per-question rows in run artifacts (session-notes 2026-07-30/hermes-v2-paired)

**variance:**
paired vs control (prior main @ e5acbbf) in the same run: net +3 (b=29/c=26, n=434) under a pre-registered non-inferiority gate, verdict PASS via blind adjudication; control re-measured 120/451 vs the banked R1 125/451 — single-run cross-build variance context for V2-static numbers

**failclose:**
17/451 union-dropped as instrument_failed (control 10, treatment 7), disclosed; adjusted 120/434

**evidence:**
- bench/FINDING-F47-V2-REBASELINE-GATE-PASS.md
- bench/specs/GATE-V2-REBASELINE.yaml (frozen 41a749a7)
- run artifacts incl. ATTRIBUTION-DARCH2.json

**caveats:**
- static-protocol numbers on V2 are low-absolute by design (fixed weak reader); the agentic protocol row (298/451) is the capability surface
- treatment arm completed via a documented append-only continuation after a transient EINTR fail-close (AMENDMENT-2; dataset freeze byte-identical across the gap, independently verified)

### <a id="scale-curve-fastscan-2026-07-30"></a>scale-curve-fastscan-2026-07-30

**id:**
scale-curve-fastscan-2026-07-30

**benchmark:**
Internal scale instrument (389×, 19,829 sessions / 199,641 messages)

**metric:**
p50_ms

**value:**
263.6

**display:**
263.6 ms @ 19,829 sessions (24× vs pre-fix; 125 ms @ 8k)

**tier:**
P

**date:**
2026-07-30

**system_commit:**
F44 gate head (fork main post-#184)

**harness_commit:**
bench/instruments/scale389 (F34 cells reproduce exactly)

**judge:**
n/a (retrieval instrument)

**reader:**
n/a

**retrieval_config:**
fast-scan residency + int8 quantize-on-load; recall parity net 0 vs all three baselines; small rungs 1.365/1.560× inside accepted envelope (F41 GRAY acceptance)

**dataset_exposure:**
n/a

**breakdown:**
full rung table in FINDING-F44

**variance:**
six gate executions, five root causes, mandatory confirmation-run protocol (F42 lesson); B-arm reproduced within 5.5% across runs

**failclose:**
3 integrity fail-closes across the saga, all correct, all documented

**evidence:**
- bench/FINDING-F44
- bench/specs/SPEC-171-FAST-SCAN-GATE v2

**caveats:**
- uncontended latency is reported separately (P1/L4); this is the published cost-curve instrument

### <a id="v2-static-451-sota-luna-2026-08-03"></a>v2-static-451-sota-luna-2026-08-03

**id:**
v2-static-451-sota-luna-2026-08-03

**benchmark:**
LongMemEval-V2 static (451q)

**metric:**
judged accuracy, full set

**value:**
0.3171

**display:**
143/451 = 31.7% (web 97/240 = 40.4%, enterprise 46/211 = 21.8%)

**tier:**
F

**date:**
2026-08-03

**system_commit:**
product wt-upstream-w1 @ 40f81bf (endorsed #436 head, F50 delivery-neutral verdict)

**harness_commit:**
lme-v2-official 6bfd58a; variant runner official_unit_runner_sota.py (models + batch-count relaxation only, documented)

**judge:**
anthropic/claude-sonnet-5 (cross-family; same pinned evaluator prompts as the official-config rows)

**reader:**
openai/gpt-5.6-luna (SOTA-models policy row)

**retrieval_config:**
identical to F47 instrument (memory config unchanged; ONLY reader+judge models differ)

**dataset_exposure:**
same frozen static-451 set as F47; no tuning on it

**breakdown:**
web 97/240=40.4%, enterprise 46/211=21.8%

**variance:**
A/A' 100q fixed-seed subset (seed 20260802) EXECUTED: 11/100 discordant (11.0%), aggregate spread 1.0pt (A-subset 32.0% vs A' 33.0%), 0 instrument failures — question-level churn high (sampling), aggregate stable; the +4.4pt cross-config delta is ~4x the subset spread

**failclose:**
0/451 instrument failures both domains

**evidence:**
- bench/specs/RUN-SHEET-V2-SOTA-LUNA.md (incl. Amendment 1)
- session-notes 2026-08-02 v2-sota-luna artifacts (full-web, full-enterprise, per-question)

**caveats:**
- Cross-config context: official-config (qwen3.5-9b reader) row is 123/451=27.3%; the +4.4pt is READER-dominated by design — this row anchors the SOTA-reader funnel, it is NOT a memory-delta claim
- Official-config rows remain visible; both configs disclosed side by side
