# LCM-X project state

This page separates released product identity, current development source, evaluation evidence, and later roadmap work. GitHub issues, pull requests, tags, releases, and exact commit heads are the live source of truth. This snapshot was reconciled on 2026-09-05 and is updated at every release; the latest update is 2026-11-09, for v0.29.2 GA.

## Naming and compatibility

The project is **LCM-X — Lossless Context Memory eXtension**. From v0.24.0 (#471) the plugin and engine identifiers are:

| Surface | Identifier |
| --- | --- |
| Repository and project | `electricsheephq/lcm-x` / LCM-X |
| Plugin manifest and install directory | `hermes-lcm-x` (legacy, v0.23.x and earlier: `hermes-lcm`) |
| Runtime context engine | `lcm-x` (legacy alias `lcm` accepted with a deprecation warning) |
| Bundled skill | `hermes-lcm` (unchanged) |
| Latest stable | `v0.29.2@261d41ea928eec3dc5b8db13c70f044392adf124` (ships `hermes-lcm-x` / `lcm-x`; GA date 2026-11-09) |

The rename is a breaking change with a documented migration path; see the operator guide's migration section. Historical notes and upstream evidence retain the names and identities used when they were created.

## Released product and development source

The latest stable release is [v0.29.2](https://github.com/electricsheephq/lcm-x/releases/tag/v0.29.2), released 2026-11-09 at `261d41ea928eec3dc5b8db13c70f044392adf124`. The previous stable release is `v0.29.1` at `3a4a08ff501347a79669fab51444c471b09b551d`.

The [published GA notes](https://github.com/electricsheephq/lcm-x/releases/tag/v0.29.2) describe a patch on v0.29.1 and the first release train: it carries every fix that was ready when the train was cut. It adds one setting, `LCM_SHUTDOWN_COMPACTION_WAIT_S`, and the schema version stays 5. Local semantic recall stays the shipped default, as from v0.29.0, with embeddings off recorded beside it in the release rows:

- **Compaction:** a contiguity stop at the scan limit no longer repeats on every pass: compaction retries once on the part of the chunk before the blocking row, so the long tool-heavy sessions it stalled can form summaries again; some cases stay stuck, such as a retry that would take a claim from a row it cut, which is refused, and they stay open in #1998. An exit fit keeps its headroom when the list is also over the model's window: it cuts whole oldest turns down to the exit budget, so a successful cut keeps that headroom, though text the token count leaves out, such as reasoning, can still use it up; when no whole-turn cut is deep enough, the fit behaves as before and the next request can still exceed the threshold (#2039).
- **Shutdown:** shutdown waits for a running compaction, up to `LCM_SHUTDOWN_COMPACTION_WAIT_S` (default 10 s) (#2101).
- **Restarts and storage:** on hosts that reuse tool-call IDs, a repeated tool call and result are stored after a restart while the host still shows the earlier identical pair (#2172), and a profile's storage opens on first real use, so another profile's gateway no longer opens and holds it (#2152).
- **Skill:** the bundled skill opens with `skill_view` (#571, #2131).
- **Testing improved:** regression tests for each fix, and CI on pushes and pull requests to `release/v*.x` maintenance branches (#2171).
- **Known issues:** the #2172 case when the earlier pair was compacted out of the host's view, and, on the same hosts, after a restart whose resume point lands past an old stored copy of the earlier step, a tool step stored twice or an old step stored in place of a new one, where v0.29.1 stores it correctly; so far it has been reached only with an injected restart point (#2197). Both are planned for v0.30.x. The notes list the other known issues, including an unreproduced loss lead (#2330).
- **Qualification:** as the first release train, v0.29.2 ran a smoke stage first and then the rows its diff selects; both hubs changed, so every memory row was selected and the D6 recorded rows were not. The smoke stage, Phase A, Phase B, the probe matrix, the host suites and D7 pass, and the local host cells pass on integrity. Not green, each by a disclosed release-manager decision: three Phase C bars and two D5-B bars ("v0.29.2 only" rows in the release spec), D2 and D3 (not run, because the prospective A/A was inconclusive), the host cells' raw lossless and zero-lost-rows bars (the scorers' fragment caps, #2214) and D5 (not run; its clone-run runner is #2199). D1 LOCAL: pass, with 0 recall flips over 500 questions. D4 and D8: D4: pass on both datasets, with no significant change against v0.29.1. D8: pass on LoCoMo; not green on LongMemEval-S (inconclusive, because its noise check failed in both pairs), by a disclosed release-manager decision.

The source snapshot used for this reconciliation is `main@a9d30d693f126224c30c7385a55aed7cf88fa96c`. Stable and main are different proof planes: stable is the released product baseline, while main contains later development and documentation work. Do not describe a main checkout as the installed stable release merely because it contains stable commits.

v0.30.0 is its sixth candidate's tree plus the release notes. Its candidates were cut from main up to rc3 and from rc4 on the `release/v0.30.x` maintenance branch, and v0.30.x patch releases are cut from that branch. That branch carries the forward identity `0.30.1` (`hermes-lcm-x`), first tagged as `v0.30.1-rc1` (the first v0.30.x release train); `v0.30.0` remains the latest stable release, and the candidate is not qualified yet. Main keeps the identity `0.30.0` (`hermes-lcm-x`) in `plugin.yaml` until its own next release. The current roadmap, including LCM-X for Claude Code as its own release unit and the cloud steps, is on main ([ROADMAP.md](https://github.com/electricsheephq/lcm-x/blob/main/ROADMAP.md)). Releases with product code are rc-first under `bench/specs/RELEASE-READINESS-V1.md`; the release identity test keeps `plugin.yaml`, README, the operator guide, CHANGELOG and the bug-report template synchronized.

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

The order after v0.29.2 is in the [roadmap](../ROADMAP.md#the-order-roadmap-v5) (roadmap v5, reconciled 2026-11-09): LCM-X for Claude Code as its own release unit, whose preview shipped on 2026-11-07 ([claude-code-v0.1.0](https://github.com/electricsheephq/lcm-x/releases/tag/claude-code-v0.1.0)), with GA and Grok Build capture by 2026-11-30; nothing lost (v0.30.0, by 2026-11-26, with its phase 2 in a v0.30.x train); cloud capture in two steps (by 2026-12-02 and 2026-12-09); facts that update (by 2026-12-16); more harnesses, Codex first (by 2026-12-23); and compaction you don't notice (by 2026-12-30). Each date means no later than, and each release ships as soon as its gate passes.

The recall QA target is ≥90% on LongMemEval-S and LoCoMo (non-adversarial) under the matched public protocol; a rank is claimed only on matched rows that hold under a second judge. Two gate-adapter corrections are in flight: retaining conversation speaker names and carrying summary source lineage. Each lands with paired old-vs-new adapter rows on a fixed product release. Earlier gate rows remain as recorded and are compared only with rows from the same adapter.

From v0.26.0, every release candidate also carries a recall scorecard (#898). Each candidate gates on the configuration its own tree ships by default, chosen per candidate. Local semantic recall has been the default since v0.29.0 (#1186), so it is the gated row, and embeddings off is recorded beside it. The local row must first meet its own A/A′ prerequisite: 0 discordant rows for recall, and D4's placebo rule for QA. The release spec sets the rc1 cadence and carry-forward rules.

Each release's gauntlet carries answer-accuracy regression rows (in the candidate's shipped-default configuration, embeddings off through v0.28.x; the same question ids and scoring path as the last release that ran them; for v0.27.0 the reader ran on codex-cli 0.162.1, against 0.161.0 for the v0.26.1 rows). v0.26.2, then the previous GA, was a one-fix patch that ran no answer-accuracy rows, so the v0.27.0 rows compare with v0.26.1. For v0.27.0 against v0.26.1: LongMemEval-S 381 of 500 correct (v0.26.1: 363) and LoCoMo 1,014 of 1,986 (987), no loss on either. For v0.28.0 against v0.27.0: LongMemEval-S 383 vs 380 of 500 and LoCoMo 1,009 vs 1,007 of 1,986, no significant difference. For v0.29.0 against v0.28.1, with embeddings off: LongMemEval-S 408 vs 379 of 500 and LoCoMo 1,165 vs 1,012 of 1,986; with local semantic recall, the new default: LongMemEval-S 426 vs 425 and LoCoMo 1,089 vs 1,083, neutral release over release. v0.29.1 has no answer-accuracy row of its own: its bench run was void, because the bench host's reader lacked a helper and every answer was rejected (both versions run fresh in v0.29.2's D4, which gives the v0.29.1 numbers that follow). For v0.29.2 against v0.29.1, in the local default: LongMemEval-S 421 vs 427 of 500 and LoCoMo 1,083 vs 1,088 of 1,986, no significant difference (exact McNemar p 0.18 and 0.64). These are regression rows under an internal reader and judge, not leaderboard rows; [ROADMAP.md](../ROADMAP.md) (H2) gives where LCM-X stands on public benchmarks and the frontier-model rows planned for v0.29.0.

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
