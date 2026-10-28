# Exact recall P5 — advisory

Measured product: `6ef9b824c6b49fef5f52c08e3881c6eef88d94ee`; pinned base: `849da257b52fc9833955928816dd1585184988b1`.

| Gate / diagnostic | Base or option1 | Selected candidate | Result / receipt |
|---|---:|---:|---|
| Arbitrary-byte fixture, UTF-8 / UTF-16le / UTF-16be | r4 failed UTF-16 | all encodings pass | focused file: `10 passed in 1.07s` at measured product head |
| No-deadline ordered top-k | 400 paired queries | 400 identical, max score delta0 | PASS; SCORE.json, N-A1/N-B1/N-A2/N-B2 |
| Peak RSS |935690240 B|882573312 B|PASS; below base +256MiB; SCORE.json |
| One prototype diagnostic, class a p50 |1.880805 s|1.010063 s|kept; n8/class/arm, load-qualified ABAB, V blocks |
| One prototype diagnostic, class b p50 |2.135665 s|1.289552 s|kept; identical ordered IDs/scores on all16 pairs, V blocks |
| C1 quiet idle |not admitted|not admitted|PENDING quiet admission; raised-deadline timings are not C1 |
| C2 one competing Python thread |not admitted|not admitted|PENDING quiet admission; no C2 performance claim |

The prototype avoids hex for UTF-8 stores and keeps the corrected coalesced hex transfer for UTF-16 stores. Every candidate and tie key remains; float32 scoring and default deadlines are unchanged. Both store audits passed with250000chunks/500leaves, integrityOK, raw data and vectors unchanged. Untimed per-site allocation capture follows each arm after acceptance RSS is saved. Warm summary caches need not reload during that diagnostic; absence of a summary-allocation event is not a measured zero.

Same-policy RCA capacity projection for this raw transfer: `RSS(D)=882573312 +64*1000*4*(D-384)` bytes. At384d:882573312 B; at768d:980877312 B. This holds fixed overhead/model residency and the assumed64-entry allocator policy constant. It is not measured768d or16GB-host RSS. UTF-16 retains hex, so this raw-transfer projection is not a UTF-16 estimate.

Width fixture PASS:2026-10-24T15:20:15.735858Z; quiet-admission cutoff:2026-10-24T16:20:15.735858Z. The current card records unadmitted latency gates and does not extend or prematurely terminate the approved window.

The three assembly scenarios in assembly-delta.json retain all non-SQL events. Commit6ef9b824 adds `PRAGMA encoding` and changes the transfer SQL, replacing each757-character aggregate event with the encoding probe and a766-character aggregate event. The intentional test-only fixture refresh makes those observations explicit; it changes no measured product bytes.

Re-score from this directory: `python3 rescore-r5.py`. Inputs are the saved synthetic per-query outputs and per-process summaries; no live store is read. Source identities and store digests are in SOURCE.json. Warmups are excluded from per-query rows.

Not proven: quiet C1/C2/C3 acceptance, the2-vCPU floor cell, saturation compaction, actual768d/16GB RSS, a fresh GA-readiness proof, independent final review, merge, release, runtime or customer readiness.

T1: pending registry commit in this PR. T2: unpublished; release manager owns archival publication under the lane's tag/release prohibition. All original raw outputs and inputs are preserved.
