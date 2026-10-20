# LCM-X project state

This page separates released product identity, current development source, evaluation evidence, and later roadmap work. GitHub issues, pull requests, tags, releases, and exact commit heads are the live source of truth. This snapshot was reconciled on 2026-09-05 and is updated at every release; the latest update is 2026-10-20, for v0.28.1 GA.

## Naming and compatibility

The project is **LCM-X — Lossless Context Memory eXtension**. From v0.24.0 (#471) the plugin and engine identifiers are:

| Surface | Identifier |
| --- | --- |
| Repository and project | `electricsheephq/lcm-x` / LCM-X |
| Plugin manifest and install directory | `hermes-lcm-x` (legacy, v0.23.x and earlier: `hermes-lcm`) |
| Runtime context engine | `lcm-x` (legacy alias `lcm` accepted with a deprecation warning) |
| Bundled skill | `hermes-lcm` (unchanged) |
| Latest stable | `v0.28.1@77b2b4f795fb6a4f84c55abf6d0441da4469971e` (ships `hermes-lcm-x` / `lcm-x`; GA date 2026-10-20) |

The rename is a breaking change with a documented migration path; see the operator guide's migration section. Historical notes and upstream evidence retain the names and identities used when they were created.

## Released product and development source

The latest stable release is [v0.28.1](https://github.com/electricsheephq/lcm-x/releases/tag/v0.28.1), released 2026-10-20 at `77b2b4f795fb6a4f84c55abf6d0441da4469971e`. The previous stable release is `v0.28.0` at `709099b3a397a7893a7abae8802ebe7bf605f79e`.

The [published GA notes](https://github.com/electricsheephq/lcm-x/releases/tag/v0.28.1) describe session continuity, replay matching and profile settings:

- **Upgrade:** Hermes 0.21.3 or newer remains required, with 0.21.5 or newer recommended; there is no SQL schema migration, but chunk gap-fill adds coverage and scan metadata. Rollback targets v0.28.0 with a SQLite-safe backup and ordered-message and summary-DAG checks; derived vectors can lag, and the inherited low-side gap after a second downgrade remains open (#1543).
- **Data integrity and session continuity:** CLI `/new` binds the next unfiltered session after an ignored or stateless session, and middle-window replay matches literal and proven aliased placeholders without copying the durable tool window. Warm proof preserves the newest two literal results for fresh retries; the older-result sibling remains open (#1540).
- **Recall and operator behavior:** The doctor checks integrity on a fresh connection and avoids duplicate threshold warnings, while chunk coverage is revalidated after rollback. Routed profiles use their own settings and provider keys, including in background jobs; the containment root stays process-wide.
- **Repeated compaction and slow summarisers:** Empty leaf input distinguishes missing source rows from a contiguity cut, and retained-occurrence stops identify the blocking row. This is diagnostics, not wedge recovery.
- **Bench and instruments:** Real-host CI covers CLI reset and filtered-session `/new`, and previous-GA differentials and rollback readers pin v0.28.0. Written rules cover inconclusive D2 results, reference-agent windows, D4 scheduling and resume, and injected-crash duplicates; these instruments change no runtime behavior.
- **Qualification:** The notes disclose release-manager decisions departing from the written gate: Phase A passes, Phase B reports no release-blocking defects, and Phase C and the integrity-only reference-agent window are accepted with dispositions. Raw host suites and the final reference-agent read remain failed, D2 remains inconclusive, D5-B passes with dispositions, and strict #1501 acceptance fails; no aggregate Phase D pass is established, and the reference-agent compaction-activity requirement is waived.
- **Known follow-ups:** Replay duplicates (#1674, planned for v0.28.2), host review-fork detach refusals (#1675), crash and interrupted-turn persistence shapes, coverage costs, wedge recovery and other listed continuity work remain open. Local recall and background leaf preparation remain optional and off by default, with no default change or new dependency.
- **Benchmark boundary:** The gauntlet ran at v0.28.1-rc3; v0.28.0 provides comparison evidence rather than carry-forward qualification. Controlled latency and regression rows do not establish an aggregate gauntlet pass, reference-agent latency, strict #1501 acceptance, or runtime or customer readiness; no improvement is claimed.

The source snapshot used for this reconciliation is `main@ab608663`. Stable and main are different proof planes: stable is the released product baseline, while main contains later development and documentation work. Do not describe a main checkout as the installed stable release merely because it contains stable commits. Releases with product code are rc-first under `bench/specs/RELEASE-READINESS-V1.md`.

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

Issue #658 is the tracker, and [ROADMAP.md](../ROADMAP.md) lays the work out by horizon and milestone. H1 compaction continues in v0.29.0 (reconstruction #1598 and the user's current request kept verbatim, #1592) and v0.31.0 (invisible compaction, plus pre-answer evidence #1111 once it passes the default-on test and the user carry packet decision #659). The summary prompt default stays v1 (#660). H2 recall work runs in v0.29.0 and v0.30.0. v0.29.0 is the evidence release:<ul><li>delivery depth step 1 passed on LongMemEval-S dev 100 plus LoCoMo conversations 1–2 (304 questions), with 397 evidence-bearing questions and 7 excluded; delivery coverage is not answer accuracy (#1607);</li><li>depth step 2 cannot pass as declared: the ≥50% BEAM event-coverage bar fails at 25, 50 and 100 hits on conversations 3–20 (360 questions, 36 event-ordering questions), a re-used parameter-selection holdout; no depth default was selected (#1607);</li><li>the time-order slice remains conditional on coverage headroom (#1589);</li><li>no frozen-pool rerank arm passed its bar; rerank moves behind breadth, and open PR #1200 remains off-by-default plumbing (#1187);</li><li>within-session depth and breadth is the next screen (#1188);</li><li>sender and role step 1 is declared, not yet run (#995);</li><li>the LOCAL default gate is declared on #1586; open PR #1639 is the default-off precursor, and the flip (#1186) waits for the declared measurements;</li><li>agentic memory.</li></ul>v0.30.0 brings source-linked derived memory (#1191), full timeline assembly, source-linked superseded facts (#1590, with #1191), abstention, and a second harness. Step 0 rejected #1590's delivery-only slice: the newer update was never delivered in 36 of 67 HaluMem Dynamic Update misses, and only 4 had both states delivered.

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
