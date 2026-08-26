# LCM-X Product Roadmap

Status: living document, reconciled 2026-08-24. GitHub issue #323 is the canonical work graph; this page explains the durable tracks without duplicating issue lifecycle state.

## What LCM-X is

LCM-X is Lossless Context Memory eXtension: a Hermes-compatible context engine that preserves raw source, builds a recoverable summary DAG, and exposes exact and semantic retrieval with explicit provenance and failure states. The tested host contract includes Hermes Agent v0.16 through its context-engine schema path; no newer unreleased host capability is assumed.

## Current product baseline

Latest stable is `v0.23.1@81d8d41197dddc4c09b57097f4955ebae32366a9`. The source snapshot for this roadmap is `main@3d4fbb4c979dc09aef0b831bb50d928e0e18d68f`.

Stable and main are separate identities. Stable is the product-under-test and installed-runtime baseline; main is the continuing development line. #342 owns main's deferred version-metadata policy.

Eva has accepted v0.23.1 with hosted `voyage-4-large`, 1024-dimensional float32 summary
vectors under that release's fail-closed privacy identity (v0.23.1 coupled cloud embedding
to durable redaction; main has since split the two flags — see docs/features-overview.md).
The proof ceiling is Eva only.

## Track A — Exact-stable retrieval provenance

Milestone **v0.23.1 Retrieval Provenance Audit** and #341 are the active finite program.

The default-off benchmark instrument traces public LongMemEval evidence through FTS, summary-vector, and chunk-vector candidate generation, shipped `lcm_recall` fusion, and content-free reference validation. Product and instrument hashes remain separate. #252 owns preregistration, score-sensitive disposition, and the post-merge ledger.

The audit is answer-blind. It may decide `KEEP CURRENT` or establish oracle headroom sufficient to earn a separate fusion-design issue. It does not authorize a product retrieval change.

## Track B — Provider trust and hierarchical context quality

Roadmap ordering remains:

1. #317 and #298 define untrusted retrieved/history presentation; #89 defines current task-state precedence.
2. #324 supplies bounded timestamp/role/sender provenance.
3. #318 defines depth-specific differential summary semantics and consumes #89/#324.
4. #319 defines bounded durable tool-result evidence.
5. #240 retains local auxiliary summary-envelope compatibility.

Every score-sensitive prompt, summary, selection, or retrieval change requires #252 disposition and a comparable baseline.

## Track C — Bounded active assembly

PR #206 retains the cap/reserve candidate. #320 owns deterministic recency/query-aware frontier selection after the cap contract, with #90/PR #297 newest-user authority preserved.

No node may become unreachable. Prompt-sensitive ranking remains default-off until a separate accepted design proves value against exact-stable provenance and cache/noise gates.

## Track D — Compaction, recovery, and diagnostics

Issue `#247` owns the reproduced multi-session `publication_invariant_conflict` defect. Issue `#314` is a current-main `needs-repro` semantics issue and must not drive threshold tuning until publication failure is separated from attempt frequency.

Issues `#36` and `#74` retain background-compaction and persistence/recovery architecture. Issue `#265` owns degraded semantic/FTS behavior, and issue `#321` owns bounded fast doctor plus explicit deep integrity checks.

## Track E — Deferred privacy and scale work

Issues `#334`-`#337` are accepted deferred privacy/identity/locality contracts. They do not reopen v0.23.1 or enable trajectory embeddings, binary prescreen, cloud reranking, or remote Ollama on Eva.

Issue `#328` is a P4 deterministic telemetry-test follow-up. Issue `#342` is the next-release version-policy follow-up.

## Track F — Teams and host integration

Teams remains separate from the current product/evaluation program. Dormant code, pilot enablement, host identity, connector behavior, and customer acceptance keep their own issues and milestones. Default-off code is not proof of safe enablement.

## Release discipline

Release work requires:

- an accepted issue and bounded change;
- exact-head CI and independent semantic review for the changed risk lane;
- required non-author code-owner approval and zero unresolved blocker threads;
- merge commits through normal protection;
- detached release validation and exact tag/release readback;
- rc-first delivery for any release touching product code: a `vX.Y.Z-rcN` prerelease that
  passes the live gauntlet in `bench/specs/RELEASE-READINESS-V1.md` (Phase A all-tools
  matrix on a fresh clone across privacy postures, Phase B multi-agent P0/P1 sweep of the
  full release diff, Phase C 30+ turn `hermes acp` soak) before the GA tag, with the GA
  commit differing from the passing rc tree by exactly the added release notes;
- explicit proof boundaries and rollback ownership.

Never restamp stable from main casually, move an existing tag, bypass the ruleset, or treat a benchmark result as runtime/customer proof.

## Near-term sequence

1. Finish canonical documentation and the default-off instrument through separate protected PRs.
2. Bind the exact merged instrument to stable v0.23.1 without changing product bytes.
3. Run seeded smoke, registered 95-question cached A/A-prime, then the full public 500-question audit within the privacy and cost caps.
4. Record `KEEP CURRENT` or `FUSION DESIGN EARNED`, run the two blind final reviews, and close only the finite audit milestone.
5. Resume later quality work through the accepted issue/dependency graph rather than an all-features campaign.

Benchmark discipline remains registration before spend, seeded sampling instead of first-N, deterministic A/A-prime noise floors, fail-closed accounting, and append-only corrections.
