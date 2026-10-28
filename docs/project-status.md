# LCM-X project state

This page separates released product identity, current development source, evaluation evidence, and later roadmap work. GitHub issues, pull requests, tags, releases, and exact commit heads are the live source of truth. This snapshot was reconciled on 2026-09-05 and is updated at every release; the latest update is 2026-10-28, for v0.29.0 GA.

## Naming and compatibility

The project is **LCM-X — Lossless Context Memory eXtension**. From v0.24.0 (#471) the plugin and engine identifiers are:

| Surface | Identifier |
| --- | --- |
| Repository and project | `electricsheephq/lcm-x` / LCM-X |
| Plugin manifest and install directory | `hermes-lcm-x` (legacy, v0.23.x and earlier: `hermes-lcm`) |
| Runtime context engine | `lcm-x` (legacy alias `lcm` accepted with a deprecation warning) |
| Bundled skill | `hermes-lcm` (unchanged) |
| Latest stable | `v0.29.0@c13dea3783bdbd1aab52d83dff5d009bfade0bd2` (ships `hermes-lcm-x` / `lcm-x`; GA date 2026-10-28) |

The rename is a breaking change with a documented migration path; see the operator guide's migration section. Historical notes and upstream evidence retain the names and identities used when they were created.

## Released product and development source

The latest stable release is [v0.29.0](https://github.com/electricsheephq/lcm-x/releases/tag/v0.29.0), released 2026-10-28 at `c13dea3783bdbd1aab52d83dff5d009bfade0bd2`. The previous stable release is `v0.28.1` at `77b2b4f795fb6a4f84c55abf6d0441da4469971e`.

The [published GA notes](https://github.com/electricsheephq/lcm-x/releases/tag/v0.29.0) describe local semantic recall on by default, fuller recall from long messages and safer retries:

- **Upgrade:** Hermes 0.21.3 or newer remains required (0.21.5 or newer recommended); the schema stays at version 5, with two empty sender columns on the messages table and three chunk embedding triggers when local embeddings are on. Local semantic recall turns on at upgrade unless `LCM_EMBEDDINGS_ENABLED=false`, and rollback targets v0.28.1 with a SQLite-safe backup (see the operator guide).
- **Data integrity and session continuity:** Retried tool turns are stored once, re-sent messages after a restart are kept, failed reads retry, stopped turns are saved right away, and compaction writes its coverage record in a single database write. Some retry and restart shapes remain open for v0.30.0 (#1836, #1837, #1838, #1451).
- **Recall and operator behavior:** Local semantic recall is on by default once the companion `hermes-lcm-x-local-embeddings` is installed; until it is ready, and on hosts with 2 or fewer effective CPUs, recall answers from full text and reports `degraded=true`. Recall returns up to its full 25 matches deep in long messages, counts exact duplicate copies once and flags uncertain times, and gateway messages carry an unverified sender label.
- **Repeated compaction and slow summarisers:** A compaction pass that changes nothing is recorded so a restarted engine skips it, summaries ending with a doubled closer are kept, and a read-only store-wedge inspector reports why compaction is stuck. The pause after a no-op pass still ends when any new message is stored (#1880, v0.29.1), and slow summaries can shrink the live view while the store keeps every row (#2004).
- **Bench and instruments:** Real-host CI adds an advisory upstream-Hermes cell, the message-loss reliability checker's false failures are fixed, and the D4, B4 and Phase A gauntlet instruments are tightened; these are instrument changes, not product changes.
- **Qualification:** Phase A passes, Phase B is CLEAR (0 confirmed P0/P1 defects), Phase C shows 0 loss and is lossless on all 7 hosts, D4 recall passes in both configurations, and D7 passes. Seven rows are not green under the written gate, and the release ships by disclosed release-manager decisions recorded in the release spec (PR #2006): D1 LOCAL (not run; #1971); D2/D3 (its placebo run failed, so the row gave no valid reading; #2004); D8 (its placebo is invalid on every row); the reference-agent window (raw verdict FAIL; compaction-activity bar not met); the Phase C doctor bar; the D5-B doctor and traceback bars; and the Phase B host suites and reliability matrix. Neither the Phase C nor the D5-B gap shows a lost, duplicated or reordered stored row, or a failed doctor check; the reliability-matrix failures include missing-row and duplicate-row shapes that v0.28.1 also shows, plus the known intermittent #861. Strict #1501 acceptance fails on the two pre-existing duplicate shapes, as on v0.28.1.

The source snapshot used for this reconciliation is `main@c10ff1b8c3ae8ef50cbb1be2e16e09457eb692dc`. Stable and main are different proof planes: stable is the released product baseline, while main contains later development and documentation work. Do not describe a main checkout as the installed stable release merely because it contains stable commits.

Main continues toward the v0.29.1 patch (see the [roadmap](../ROADMAP.md)). Releases with product code are rc-first under `bench/specs/RELEASE-READINESS-V1.md`; the release identity test keeps `plugin.yaml`, README, the operator guide, CHANGELOG and the bug-report template synchronized.

## Reference-agent acceptance state

The reference agent is a long-running Hermes agent the project operates itself. From v0.25.0, each minor's release candidate runs on it before GA: 24 hours through v0.26.x, and for v0.27.0 a 12-hour gated window with capture continuing to 24 hours after GA (the 12-hour gate was declared before the window, in place of the release spec's 24 hours; the rc8 window ended at 11.9 hours, when the agent's Hermes host was updated outside the release, so the 12-hour duration bar was not met; the release manager accepted the shorter window as a disclosed exception on the owner's delegation, every other gate held over the window, and #1305 adds both rules to the release spec). For v0.26.0, host-message-identity shadow mode agreed with LCM-X's content matching on 363 of 363 counted rows over 24.0 h, with no disagreement. The 200-row minimum was met by a disclosed synthetic session on the same agent and candidate; 61 of the counted rows came from organic traffic. The release notes and the shadow-window receipt give the details. No compaction ran in the agent's own traffic during that window, so there is no latency sample; reference-agent soak latency is recorded, not gated for v0.26.0; gauntlet latency gates still apply. For v0.27.0, latency gated on an engine-direct proxy over a copy of the reference agent's store, and the 11.9-hour window on the agent itself gated real use (the duration exception above): every other gate passed, with 0 LCM-X tracebacks, every LCM-owned stop within 120 s (host-owned time is reported, not gated) (slowest 116.3 s), and 4 answered organic turns plus 1 sent by the release manager. The release notes and the soak receipt give the details. For v0.28.0, the reference-agent window qualified integrity only (3 organic turns over 8.7 h, all answered, 0 lost or duplicate rows, 0 LCM-X tracebacks), and latency gated on controlled surfaces.

The full acceptance packet dates from v0.23.1. It covers the reference agent on exact stable v0.23.1 with hosted `voyage-4-large`, 1024-dimensional float32 summary vectors under the privacy-bound vector identity. The packet covers:

- complete eligible summary vectors (4,559/4,559 at acceptance);
- fail-closed cloud summary/query privacy handling;
- SQLite and FTS parity;
- exact and semantic recall;
- all 15 LCM tools with supported or explicit disabled/degraded outcomes;
- provider accounting, continuity, controlled restart without replay amplification, and copied-state rollback;
- two blind final reviews at 98 and 97.

That result establishes `runtime_safe` for the reference agent's hosted privacy-safe configuration only. It does not prove fleet, customer, Teams, local-model production, or universal benchmark readiness.

At v0.23.1 acceptance, reranking, binary prescreen/int8, proactive recall, V4 assertion/adaptive/pre-answer features, raw-chunk embeddings, and local-model switching were off for the reference agent.

## Current program

Issue #658 is the tracker, and [ROADMAP.md](../ROADMAP.md) lays the work out by horizon and milestone. The leadership target is long-session continuity: nothing lost, operative facts and constraints kept exactly across compactions, and a small visible wait. The target is measured on generated sessions run through 600+ checkpoints and nested summaries; current Codex, Claude Code and Hermes built-in will be measured under the same instrument, with receipts published when the runs complete (#1595, #1587). This is a target, not a claim that it has been met. The summary prompt default stays v1 (#660).

v0.29.0 shipped on 2026-10-28 (see above); this was its pre-release program, kept as history:<ul><li>local embeddings become the default and release-gate arm (#1186) only after #1586 passes, with embeddings off kept beside it;</li><li>delivery and abstention share one paired gate (#1591): more evidence than declared behaviour ships with a sufficiency signal or after its own unanswerable-question check passes; the declared-delivery defect fix in PR #1679 is judged on total utility with that guard disclosed;</li><li>sender provenance (#995), duplicate-hit collapse at delivery (#1681), and the continuity instrument and head-to-head (#1595, #1587);</li><li>measurement infrastructure: noise floor, second judge and nightly factorial runs;</li><li>offline reconstruction (#1598) and the user's current request kept verbatim (#1592).</li></ul>The ≤30-minute development loop is a kill/futility screen for large effects; a nightly factorial on ≥400 clustered questions promotes, and an untouched cohort confirms. Each screen gets a power check; no bar the baseline cannot meet. The noise floor and second judge are measured before ±2-point bars are used.

v0.30.0 brings evidence packs / session expansion if the two-arm LoCoMo oracle supports them, depth and breadth re-screened with sufficiency and speaker names (the current #1188 and rerank #1187 issues remain in v0.29.0), replay identity repair (#1674), write-time extraction (#1191), supersession (#1590), time order (#1589), and a second harness. v0.31.0 holds invisible compaction and the cache-forked compaction study (#1228), plus the pre-answer evidence gate (#1111) and user carry packet decision (#659).

The recall QA target is ≥90% on LongMemEval-S and LoCoMo (non-adversarial) under the matched public protocol; a rank is claimed only on matched rows that hold under a second judge. Two gate-adapter corrections are in flight: retaining conversation speaker names and carrying summary source lineage. Each lands with paired old-vs-new adapter rows on a fixed product release. Earlier gate rows remain as recorded and are compared only with rows from the same adapter.

From v0.26.0, every release candidate also carries a recall scorecard (#898). Each candidate gates on the configuration its own tree ships by default, chosen per candidate. When local semantic recall is the default (#1186), embeddings off is recorded beside it. The local row must first meet its own A/A′ prerequisite: 0 discordant rows for recall, and D4's placebo rule for QA. The release spec sets the rc1 cadence and carry-forward rules.

Each release's gauntlet carries answer-accuracy regression rows (in the candidate's shipped-default configuration, embeddings off through v0.28.x; the same question ids and scoring path as the last release that ran them; for v0.27.0 the reader ran on codex-cli 0.162.1, against 0.161.0 for the v0.26.1 rows). v0.26.2, the previous GA, was a one-fix patch that ran no answer-accuracy rows, so the v0.27.0 rows compare with v0.26.1. For v0.27.0 against v0.26.1: LongMemEval-S 381 of 500 correct (v0.26.1: 363) and LoCoMo 1,014 of 1,986 (987), no loss on either. For v0.28.0 against v0.27.0: LongMemEval-S 383 vs 380 of 500 and LoCoMo 1,009 vs 1,007 of 1,986, no significant difference. These are regression rows under an internal reader and judge, not leaderboard rows; [ROADMAP.md](../ROADMAP.md) (H2) gives where LCM-X stands on public benchmarks and the frontier-model rows planned for v0.29.0.

## History

v0.23.2 (2026-08-27) made durable redaction and cloud-embedding privacy independent flags (`LCM_SENSITIVE_PATTERNS_ENABLED` vs `LCM_EMBEDDING_PRIVACY_ENABLED`, #374), made privacy-policy errors on the recall path fail loud (#370), and made releases with product code rc-first under `bench/specs/RELEASE-READINESS-V1.md` (#373).

The v0.23.1 Retrieval Provenance Audit (#341, closed) and the post-v0.23.1 roadmap (#323, closed) are history. #252 still holds the score-sensitive dossier and the conditional `LAND` verdict; historical F53-F58 rows used older product and provider identities and remain context only. Any retrieval behavior change still requires a separate accepted issue and a fresh baseline.

Teams remains a separate dormant/pilot program with its own milestones and host acceptance. No current LCM-X evidence should be presented as Teams enablement.

## Proof boundaries

Keep these claims separate:

- source and tag identity;
- PR/CI/review/merge state;
- benchmark candidate and delivery metrics;
- answer accuracy;
- released product;
- one-profile runtime safety;
- fleet/customer readiness.

A passing clone, benchmark, or reference-agent acceptance packet proves only its named plane. Current details and ownership live in #658 and the GitHub milestones; #252 holds the score-sensitive eval queue.
