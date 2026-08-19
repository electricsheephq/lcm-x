# F44 — #171 gate CLOSED: PASS under the standing F41 GRAY acceptance (execution #6). 263.6ms @19,829 (24× vs F34) with the full correctness batch intact. Six executions, five root causes, zero unexplained numbers.

**Run:** gate171-v6 on 56fd7b4, spec v2 unchanged (sixth execution), pins/stores/probes verified (one
DECLARED deviation, §3), zero LLM. Artifacts: hermes-r3-1/artifacts/gate171-v6-run/.

## 1. Reading and adjudication
PASS-1 (recall net 0 vs F34, v3 AND v5 — 100% concordant), PASS-2, **PASS-3: 263.55ms ≤ 350** (norm
251.0), PASS-4 (0 ≥8s anywhere), PASS-6 (disclosures exact, 80/80 parity cells). PASS-5: 500=1.365×,
2000=1.560× — GRAY band, **within the envelope the owner accepted in F41** (1.39/1.54 at the same
clause for the same trade) → the standing acceptance applies; no re-litigation. **GATE CLOSED: PASS.**
Residency 1/99/100 everywhere eligible; hit path zero-SQL at scale; temp tables gone at scale.
Residual vs v3 (+41ms top rung): the resident-budget accounting runs O(3N) `sys.getsizeof` per hit
(~32ms @185k) — optimization follow-up noted on fork #185; NOT another gate cycle (in band, diminishing
returns, six executions).

## 2. The saga, banked as method (F39→F44)
Six executions of ONE frozen spec found and fixed five distinct root causes — dtype-gated eligibility,
instance-lifetime cache death, a review batch that broke both mechanisms while fixing 20 findings
(3.17× worse than pre-fix, caught only by the mandatory confirmation), a 334ms per-hit staleness
recheck, and the final accounting residual — with recall parity held at net 0 through every build.
**The rule, now standing (→ PROGRAM-ARCHITECTURE §6e next docs pass): a fix batch on a measured surface
is an unmeasured build until its gate re-runs; performance mechanisms carry their engagement telemetry
as permanent tests (builds/hits counts, zero-scan statement counts).**

## 3. Frozen-store protocol note (declared deviation)
F43 installs invalidation triggers on first store open (one-time idempotent schema migration; content
digests of all data tables proven identical, identity_hash/data_version unchanged — trap #3 held). The
frozen-store protocol gains a rule: gate runs against pinned stores DECLARE the migration pass and
re-pin post-migration shas; content-digest comparison is the arbiter when byte-shas move for
schema-metadata reasons.

## 4. Claim (ships with both sides, per the candour pattern)
Retrieval p50 at 19,829 sessions / 185k vectors: **6,311ms (R2 published) → 263.6ms (24×)**, 8k:
1,790→125ms (14×), with recall parity proven at net 0 across 340 paired questions and small-store cost
published (+7ms @500, +25ms @2k). #184 merges after a final bot pass on 56fd7b4.
