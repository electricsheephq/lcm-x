# LCM-X project state

This page separates released product identity, current development source, evaluation evidence, and later roadmap work. GitHub issues, pull requests, tags, releases, and exact commit heads are the live source of truth. This snapshot was reconciled on 2026-09-05 and is updated at every release; the latest update is 2026-11-03, for v0.29.1 GA.

## Naming and compatibility

The project is **LCM-X — Lossless Context Memory eXtension**. From v0.24.0 (#471) the plugin and engine identifiers are:

| Surface | Identifier |
| --- | --- |
| Repository and project | `electricsheephq/lcm-x` / LCM-X |
| Plugin manifest and install directory | `hermes-lcm-x` (legacy, v0.23.x and earlier: `hermes-lcm`) |
| Runtime context engine | `lcm-x` (legacy alias `lcm` accepted with a deprecation warning) |
| Bundled skill | `hermes-lcm` (unchanged) |
| Latest stable | `v0.29.1@3a4a08ff501347a79669fab51444c471b09b551d` (ships `hermes-lcm-x` / `lcm-x`; GA date 2026-11-03) |

The rename is a breaking change with a documented migration path; see the operator guide's migration section. Historical notes and upstream evidence retain the names and identities used when they were created.

## Released product and development source

The latest stable release is [v0.29.1](https://github.com/electricsheephq/lcm-x/releases/tag/v0.29.1), released 2026-11-03 at `3a4a08ff501347a79669fab51444c471b09b551d`. The previous stable release is `v0.29.0` at `c13dea3783bdbd1aab52d83dff5d009bfade0bd2`.

The [published GA notes](https://github.com/electricsheephq/lcm-x/releases/tag/v0.29.1) describe a patch on v0.29.0 that fixes compaction and restart problems found in v0.29.0's release testing. There are no new settings and the schema version stays 5:

- **Compaction:** Compaction now gets past a row the host still holds by summarising the blocked rows from the stored copy, so passes make progress again (#1162, #1880). The 10-minute pause after a pass that changes nothing holds the full 600 s again, and summaries read a leaf's rows in store order after a host reorder (#2007).
- **Restarts and retries:** A restart after a cleanup pass no longer stores the conversation again; this problem never lost a message. Two retry shapes after a compaction now store exactly once (#1446), and a re-stamped tool turn after a cold restart is stored (#1967).
- **Doctor and install:** `lcm_doctor` reports local recall as ready, preparing or unavailable, with clearer messages (#1959), and documentation links in the install payload use the release's version tag (#1960).
- **Testing improved:** 54 new restart tests, 35 of which fail on the second release candidate, plus fixes to the pinned-host suites, the reliability matrix, the benchmark instruments and the release gates; patch releases now run the gauntlet rows their diff selects (#1995).
- **Known issues with a fix planned for v0.29.2:** on hosts that reuse tool-call IDs, a restart can skip a repeated tool call and result while the host still shows the earlier identical pair (#2172); and the bundled skill cannot be opened with `skill_view` (#571, #2131).
- **Known issues planned for v0.30.0:** the same #2172 case when that earlier pair was compacted out of the host's view, and a tool result stored again without its call after a restart (#2163); both need the ingest record that v0.30.0 adds. Also a compaction pass that makes no progress can still make the host rebuild its prompt cache (#1880), because the fix tried for v0.29.2 did not pass review. On one shape a compaction pass can spend about 80 to 130 s before a cold turn's first reply (#2173). A restart of a long conversation can store old rows again, with nothing lost (#1674). The notes list the other known issues.
- **Qualification:** As a patch, v0.29.1 ran the 17 gauntlet rows its diff selects: Phase A passes, Phase B is CLEAR, the full suite passes at the rc4 SHA and the reliability matrix shows 0 new failing cells against v0.29.0, while D1 LOCAL and D4/D8 were not run before this release (raw verdict UNRUN; not green) by new release-manager decisions, with results in the v0.29.2 notes, the occurrence grader was selected but not run, and the D6 recorded rows and the optional D5-A row were not run. The reference agent ran rc4 in four windows, whose three reads before window 4 are raw FAILs with no decision (so D5 is not green) and whose window 4 was interrupted by a host update and three gateway restarts and cannot decide, so the release rests on the D5-B store-clone cells, the seven replayed copy events and a row-by-row check that found no stored row lost or changed in identity from the install to window 4's start; the Phase C doctor and caught host-trace bars, the D5-B doctor and LCM-X-frame traceback bars and the lossless bars on three local host cells are not green, the strict #1501 verdict is FAIL with no product regression, and the notes give each disclosed decision.

The source snapshot used for this reconciliation is `main@c10ff1b8c3ae8ef50cbb1be2e16e09457eb692dc`. Stable and main are different proof planes: stable is the released product baseline, while main contains later development and documentation work. Do not describe a main checkout as the installed stable release merely because it contains stable commits.

v0.29.1 was cut from the `release/v0.29.x` maintenance branch, not from main; v0.29.x patch releases are cut from that branch. That branch carries the forward identity `0.29.2` (`hermes-lcm-x`), first tagged as `v0.29.2-rc1` (the first release train); `v0.29.1` remains the latest stable release, and the candidate is not qualified yet. Main keeps the identity `0.29.1` (`hermes-lcm-x`) in `plugin.yaml` until its own next release. The current roadmap, including LCM-X for Claude Code as its own release unit and the cloud steps, is on main ([ROADMAP.md](https://github.com/electricsheephq/lcm-x/blob/main/ROADMAP.md)). Releases with product code are rc-first under `bench/specs/RELEASE-READINESS-V1.md`; the release identity test keeps `plugin.yaml`, README, the operator guide, CHANGELOG and the bug-report template synchronized.

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
