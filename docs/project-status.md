# LCM-X project state

This page separates the current LCM-X baseline from active candidate work.
GitHub issues, pull requests, tags, and exact commit heads are the live source
of truth; this snapshot was reconciled on 2026-08-16.

## Naming and compatibility

The project is **LCM-X — Lossless Context Memory eXtension**. Existing runtime
identifiers remain stable for compatibility:

| Surface | Identifier |
| --- | --- |
| Repository and project | `electricsheephq/lcm-x` / LCM-X |
| Plugin manifest, install directory, and skill | `hermes-lcm` |
| Runtime context engine | `lcm` |
| Current version tag | `v0.21.0-rc2` |

Changing `hermes-lcm` or `lcm` would require a separately designed migration
covering configuration, installed paths, skill discovery, databases, scripts,
and operator rollback. No such rename is part of the documentation change.

### Remaining naming follow-ups

- `docs/banner.png` still contains the historical HERMES-LCM wordmark. The
  README no longer renders it; a new LCM-X banner/social image needs a separate
  visual-design pass.
- A few runtime/model-facing source strings still say “Lossless Context
  Management” or “Hermes-LCM.” Updating them can affect prompts, snapshots, and
  compatibility tests, so it should be a small source change rather than a
  silent docs rewrite.
- Historical release notes and upstream evidence retain the name used when they
  were created. Rewriting those records would erase provenance.

## Current baseline

The code baseline reviewed for this snapshot was `main` at
`223a4a47c1ab6caefdeec8033d19fbcfff28f7ff`. The RC2 tag is `v0.21.0-rc2` at
`10cbb78347ec86f3004153b24767324ded9e37b4`; it is contained in that baseline.

RC2 includes the 15-tool LCM surface, opt-in assertion/evidence/query-view and
adaptive-retrieval features, and the committed memory evaluation harnesses.
Those optional features remain default-off. The repository has the RC2 tag but
does not currently publish an LCM-X GitHub Release object or packaged release
artifact, so the supported install remains a clone or symlink checkout.

## Memory evaluation state

LCM-X has two distinct evaluation layers. They must not be collapsed into one
quality claim.

### Retrieval quality: landed

The in-tree LongMemEval harness measures session- and turn-level recall@k and
NDCG@10 against pinned labeled evidence. It creates a fresh temporary LCM store
for each question, uses deterministic summaries, and includes an arm that calls
the production `lcm_recall` tool directly.

The committed 500-question FastEmbed result has 470 scoreable questions. In
that exact configuration, `chunk_vectors` reports session R@5 0.96 / R@10 0.98,
and the 50-question production-`lcm_recall` subset reports R@5 0.98 / R@10
1.00. These are retrieval measurements for the recorded dataset, model, and
configuration—not general answer-accuracy or product-readiness scores.

- [Methodology](../benchmarks/METHODOLOGY.md)
- [500-question v3 FastEmbed result](../benchmarks/results/longmemeval-v3-500q-fastembed.md)
- [Machine-readable metrics](../benchmarks/results/longmemeval-v3-500q-fastembed-metrics.json)

### Judged QA accuracy: harness landed, full result pending

The vendored MemoryBench adapter exercises ingest, production `lcm_recall`,
answer generation, LongMemEval judging, and reporting in fresh per-question
containers. The methodology requires exact answerer and judge model disclosure
and treats non-`gpt-4o` judge runs as directional. The results index still marks
the full 500-question judged-QA run as pending. The recommended
`voyage-context-3` retrieval run is also pending.

- [QA replication guide](../benchmarks/qa-harness/REPLICATION.md)
- [Results index](../benchmarks/METHODOLOGY.md#results-index)

The current evidence proves deterministic harness behavior and
configuration-specific retrieval performance. It does not prove autonomous
employee performance, live learning, merge readiness, release readiness,
runtime safety, or customer readiness.

## Active candidate work

The following pull requests preserve exact upstream commit history but remain
unmerged LCM-X candidates:

| Candidate | Exact head | Current boundary |
| --- | --- | --- |
| [LCM Teams v1 #200](https://github.com/electricsheephq/lcm-x/pull/200) | `eb7b22a4afadc5130bb7e5c13b55122a82ee8df3` | Draft and conflicting. Adds access-context, policy, owner-scoped storage, catalog/connector, and isolation-test surfaces, but its policy source still labels itself a draft and permits legacy unstamped rows. It is not in RC2 or `main`. |
| [RC2 reconciliation #214](https://github.com/electricsheephq/lcm-x/pull/214) | `496e1021b7c39859f731c892a04cd79f56faff93` | Draft, conflicting, and explicitly CI-only. It carries post-RC2 runtime fixes but authorizes no tag, release, or activation. |
| [Codex continuity and whitepaper control flow #215](https://github.com/electricsheephq/lcm-x/pull/215) | `02428dde78f70a9bc101b9d3e70b2d7c5e6c4e02` | Draft and conflicting. It proposes provider-native Codex continuity, explicit compaction paths, concurrent storage recovery, and persistent map operators. Its original local validation is preserved in the PR, but the changes are not current LCM-X behavior. |

Do not copy feature lists from these branches into current-user instructions
until the relevant candidate is reconciled, reviewed, and merged.

## What is still needed

### LCM Teams

The canonical [Teams roadmap #75](https://github.com/electricsheephq/lcm-x/issues/75)
remains open. Before LCM-X can document Teams as supported, the project needs:

1. a conflict-free candidate based on current LCM-X `main`;
2. a completed, reviewed fail-closed policy for stamped and legacy data;
3. the authenticated Hermes host carrier and dependency reconciliation;
4. the provider-neutral management connector and local recovery path;
5. witnessed enable, disable, migration, rollback, delegation, revocation, and
   two-principal isolation acceptance;
6. post-merge operator/configuration docs and an explicit release decision.

The Teams candidate is default-off, but default-off is not a substitute for the
security and operations acceptance above.

### RC2 and release reconciliation

1. Decide whether the fixes in #214 belong in a new candidate release or should
   be superseded by a current-main reconciliation.
2. Run the repository's required Python matrix and semantic review on the exact
   selected head; the imported PR's current destination checks do not by
   themselves prove the runtime suite.
3. If a release is authorized, create immutable release evidence and update the
   changelog and install guidance. The existing tag alone is not a packaged or
   customer-proven release.

### Memory evaluation

1. Complete and publish the 500-question judged-QA run with exact answerer,
   judge, dataset revision, harness head, and configuration.
2. Complete the pending Voyage retrieval run and keep it separate from the
   FastEmbed floor.
3. Re-run the affected evaluation layers on any candidate whose retrieval,
   storage, scoping, compaction, or answer-context behavior changes.
4. Report harness defects separately from retrieval defects, answerer defects,
   and product/runtime defects.

## Documentation tracker

The naming and status reconciliation is tracked in
[#216](https://github.com/electricsheephq/lcm-x/issues/216).
