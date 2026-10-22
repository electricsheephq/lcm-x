# #1822 proof receipt

Candidate: 158191079b4ec61e676f9d03ea3f34069543e8fb, branch fix/1822-rewind-mask. Base:238d1a9e876dea55b2e2f7a68110afd21c4e779e. Claim class: advisory.

## Functional and preservation evidence

| Gate | Observed result | Identity |
|---|---|---|
| Focused1628/1540/property | 352 passed in10.73s | final head |
| Golden suite | 221 passed in21.29s | final head |
| Committed property | 64 seeds + merge-head regression; entire affected352-cell suite under20s | final head |
| Random differential | 2000 histories, selected decisions0 differences, complete stored rows0 differences, repeat ingest stable | final head |
| Existing1628 row snapshots | all11 original cells match base dictionary-for-dictionary | final head; only per-run synthetic payload-root normalization |
| Existing trace entries | all201 byte-identical, only anchor-version-ignore added | fixture unchanged after initial final generation |
| New trace row tables | base equal in state-after, next-state and cold-state | final head |
| Full suite once | 19961 passed,141 skipped,63 xfailed,29 warnings in1863.60s (0:31:03), exit0 | b0821b6810868180a6a26bac258f167faaf381fb |
| Ruff0.15.13 | All checks passed! | final head |
| Committed diff whitespace | exit0; no output | final head vs base |

Sweep SHA256:23cf0ad81eb861bc47d96d2701957a9507b777a477b498aaef0ede752588bf07. Every stored dictionary field is compared, including timestamps. Existing deterministic trace clock is installed; no normalization removes fields. All raw synthetic histories preserved in the T2 result bundle (tag listed in RESULT.md).

## Mutation sensitivity and RED-first sequence

RED commit:c9e258cf97af6471aa775825df3ac461e5f6abe9 precedes first production fix b0821b6810868180a6a26bac258f167faaf381fb.
- Base RED:2failed11passed. Rewind mask missing and ledger writes3 rather than1. Exact ignore-content test is base-green by design.
- Same-candidate stale-flag mutant at15819107 changes exactly the pair splice to replay-only. Tightened exact-role/text and exactly-one-drop assertion:1failed1passed0.66s, expected negative. Candidate passes.
- New golden candidate455 events vs mutant385; first divergence59 is the candidate's single ignore-drop log versus mutant storage append. Mutant also adds a stored row.
- Metadata ledger ignored_active_replay_placeholder_hashes writes3→1. Removed redundant anchored digest931b1926ba4dc20e and repeated fresh digest9c4bdcd54bc33fe2. Initial fresh write remains.
- Initial optimization's merge-head defect reproduced with spaced A / merged A+T / separately stamped T: stale map produced5 rows and0drops, fresh map4rows1drop. Regression RED at b0821b68:1failed1.16s. Corrected by snapshot comparison in fb1b6ac5.

## Independent review and bounded corrections

One independent Astra reviewer, read-only, with no tests, skills, GitHub writes or descendants:
1. Initial b0821b68 BLOCKED on demonstrated merge-head override invalidation.
2. fb1b6ac5 targeted delta PASS: override snapshot invalidates stale map; original finding fixed now.
3. e6ec29ec targeted delta PASS: current-session carry is inert; ancestor/unknown carries continue invalidation.
4. 15819107 targeted delta PASS: no-pattern plain-list path and cache-access short circuit preserve decisions; pattern state unchanged.

Review scope is source semantics; author-owned execution and measurement remain separate. No new general review was started. Canonical Governor checkpoint after third targeted pass froze further hardening and redirected to final proof/publication.

## Cost candidate history

- fb1b6ac5:2pattern rewind cells failed (39–42% above pre1628); six other cells passed. Full store map rebuilt because current-session carry was treated as an ancestor mutation.
- e6ec29ec:7/8cells passed; 200-row no-pattern rewind missed strict limit by0.25%. Removed unused no-pattern subclass allocation and map guard traversal in next delta.
- Final cost card:RESULT.md, raw4800 samples:cost-200.jsonl, scorer:score.py, complete arm medians/IQRs/limits:cost-summary.json.

The failed runs' raw samples, summaries and cards are preserved in the final T2 bundle. They were not silently discarded or rescored against changed limits. Same workloads/three arms and median200 per cell throughout.

## Probe boundaries

Masked builds leave unselected flags False and replay values at input defaults. The only production masked caller consumes exactly the supplied rewound set, so those defaults never replace anchored rows or the original tail. Full-list occurrence mapping never receives a truncated list. The original first full-build call remains unchanged.

This is a synthetic correctness/cost proof; no new host contract, persistence, schema or configuration. No comments/replies/thread resolution, merge/enqueue, product release, customer or installed-runtime actions were performed. Full-suite evidence predates the bounded corrections; exact-head remote CI is separate.

T1 registry edit omitted under explicit SPEC owned-path boundary. Durable public proof is this evidence branch; private receipt store is identified only by its T2 tag, without naming it here.

Not proven: final-head full local suite, merge/release/runtime/customer readiness.

## Previous-GA CI

[Exact-head run39854613143](https://github.com/electricsheephq/lcm-x/actions/runs/39854613143) passed: `97 passed, 2 xfailed in 183.61s (0:03:03)`. The18 summary reports comprise1076 cases (60generated,1000projection,16single shapes including one older managed baseline). All oracle violation counts{}, unexplained events0, accepted deltas0. See GA-summary.md for complete shapes and baselines; raw outputs are preserved in T2. Existing #1225/#1247 xfails remain.

T2 verified2026-10-22T21:15:53Z: all4asset names, sizes and SHA256digests read back; complete bundle includes raw/scorer inputs, previous-GA outputs and superseded measurements. Tag:lcm-x/1822-r2-15819107-cost.
