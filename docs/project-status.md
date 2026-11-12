# LCM-X project state

This page separates released product identity, current development source, evaluation evidence, and later roadmap work. GitHub issues, pull requests, tags, releases, and exact commit heads are the live source of truth. This snapshot was reconciled on 2026-09-05 and is updated at every release; the latest update is 2026-11-12, for v0.30.0 GA.

## Naming and compatibility

The project is **LCM-X — Lossless Context Memory eXtension**. From v0.24.0 (#471) the plugin and engine identifiers are:

| Surface | Identifier |
| --- | --- |
| Repository and project | `electricsheephq/lcm-x` / LCM-X |
| Plugin manifest and install directory | `hermes-lcm-x` (legacy, v0.23.x and earlier: `hermes-lcm`) |
| Runtime context engine | `lcm-x` (legacy alias `lcm` accepted with a deprecation warning) |
| Bundled skill | `hermes-lcm` (unchanged) |
| Latest stable | `v0.30.0@6260c31fbce11e4cd542878f1f6a37410d678ae8` (ships `hermes-lcm-x` / `lcm-x`; GA date 2026-11-12) |

The rename is a breaking change with a documented migration path; see the operator guide's migration section. Historical notes and upstream evidence retain the names and identities used when they were created.

## Released product and development source

The latest stable release is [v0.30.0](https://github.com/electricsheephq/lcm-x/releases/tag/v0.30.0), released 2026-11-12 at `6260c31fbce11e4cd542878f1f6a37410d678ae8`. The previous stable release is `v0.29.2` at `261d41ea928eec3dc5b8db13c70f044392adf124`.

The [published GA notes](https://github.com/electricsheephq/lcm-x/releases/tag/v0.30.0) describe a minor release that works toward "nothing lost": storing each message once across retries, replays and restarts. The schema version stays 5, with no new columns in `messages`; the ingest ledger adds its own tables. Rolling back to v0.29.2 loses no rows; one case stores a copy (#2552). No answer-quality gain is claimed. Local semantic recall stays the shipped default, as from v0.29.0:

- **Ingest:** each batch of incoming messages commits or rolls back as a unit, and a shadow ledger records message bindings, observed timestamps and the conversation view (#2055; `LCM_INGEST_LEDGER` defaults to `shadow`). With the ledger on, a cold start no longer stores again the rows it can prove it already holds.
- **Helper agents:** delegated helper agents' sessions are stored (#1931), and an interrupted helper keeps its rows up to its last save, which runs before each of its model calls (#2482).
- **Saved tool results:** a tool result that Hermes saved to a file is stored in full and recognised when Hermes sends its marker again (#2539).
- **Recall and tools:** search and recall include the current session's delegated children; full-text search is retried once when it timed out and nothing else was found; the read-only `lcm-x mcp` reader (`scripts/lcm_x_mcp.py`, from a source checkout) serves recall to Claude Code over MCP (#1192). New recall presentation options stay off by default.
- **Removed and deprecated:** cache-friendly follow-on condensation and transcript garbage collection are removed (their settings are parsed and ignored); deferred maintenance is deprecated, with removal planned for v0.31.0.
- **Known issues:** the notes list every known exception, one line each. Among them: the busy-session check failed (#2293; its bar gates v0.30.1); a reply in flight at a host crash right after a rotation is in the host's database but not yet in the store (#2624); and the limits of saved-result matching and what a replay can still store again (#2544, #2604, #2605).
- **Qualification:** on the final candidate, v0.30.0-rc6, which has the same code, with every row run again on it. Pass: the certifying driven run (0 lost rows; 1,166 rows expected, 1,174 stored, with 11 visible extra copies, and 3 stored rows each hold two expected rows; delegated helper agents 12 of 12), the required CI checks, the smoke stage (the reliability matrix on its second run; the fault-cell pairs with no regression, raw FAILs kept as known issues), Phase A, D1 with embeddings off (0 changes over 500 questions), the review sweep, D4/D8 as a paired sample of 100 questions (87 against 86 on v0.29.2), D7 (mean wait per turn 5.02 s), same-store recall latency and the host cells. Not green, each by a disclosed decision: the busy-session check (FAIL), the Phase B probe matrix (raw FAIL; the differential reading finds 0 worse cells) and host-suite comparison, two bars each of Phase C and D5-B, D2 and D3 (not run), D5 (not run), two test-kit rows (removed), and the D6 rows other than its `OFF` recall row, which run after the release with D1 with local embeddings and the full D4 and D8 rows (#2649).

The source snapshot used for this reconciliation is `main@1fc6112e38a11e99080ccc0a9288b93aea396cd7`. Stable and main are different proof planes: stable is the released product baseline, while main contains later development and documentation work. Do not describe a main checkout as the installed stable release merely because it contains stable commits.

v0.30.0 is its sixth candidate's tree plus the release notes. Its candidates were cut from main up to rc3 and from rc4 on the `release/v0.30.x` branch, so main stayed open during the gate. Later v0.30.x patch trains, starting with v0.30.1, are cut from that branch. Main still carries the identity `0.30.0` (`hermes-lcm-x`). The order after v0.30.0, including LCM-X for Claude Code as its own release unit and the cloud steps, is in the [roadmap](../ROADMAP.md#the-order-roadmap-v6). Releases with product code are rc-first under `bench/specs/RELEASE-READINESS-V1.md`; the release identity test keeps `plugin.yaml`, README, the operator guide, CHANGELOG and the bug-report template synchronized.

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

The old tracker, #658, was closed on 2026-11-06, and [ROADMAP.md](../ROADMAP.md) lays the work out by horizon and milestone. The leadership target is long-session continuity: nothing lost, operative facts and constraints kept exactly across compactions, and a small visible wait. The target is measured on generated sessions run through 600+ checkpoints and nested summaries; current Codex, Claude Code and Hermes built-in will be measured under the same instrument, with receipts published when the runs complete (#1595, #1587). This is a target, not a claim that it has been met. The summary prompt default stays v1 (#660).

v0.29.0 shipped on 2026-10-28 (see above); this was its pre-release program, kept as history:<ul><li>local embeddings become the default and release-gate arm (#1186) only after #1586 passes, with embeddings off kept beside it;</li><li>delivery and abstention share one paired gate (#1591): more evidence than declared behaviour ships with a sufficiency signal or after its own unanswerable-question check passes; the declared-delivery defect fix in PR #1679 is judged on total utility with that guard disclosed;</li><li>sender provenance (#995), duplicate-hit collapse at delivery (#1681), and the continuity instrument and head-to-head (#1595, #1587);</li><li>measurement infrastructure: noise floor, second judge and nightly factorial runs;</li><li>offline reconstruction (#1598) and the user's current request kept verbatim (#1592).</li></ul>The ≤30-minute development loop is a kill/futility screen for large effects; a nightly factorial on ≥400 clustered questions promotes, and an untouched cohort confirms. Each screen gets a power check; no bar the baseline cannot meet. The noise floor and second judge are measured before ±2-point bars are used.

The order after v0.30.0 is in the [roadmap](../ROADMAP.md#the-order-roadmap-v6) (roadmap v6, reconciled 2026-11-12): LCM-X for Claude Code as its own release unit, whose preview shipped on 2026-11-07 ([claude-code-v0.1.0](https://github.com/electricsheephq/lcm-x/releases/tag/claude-code-v0.1.0)), with GA, Grok Build capture and Codex install and recall by 2026-11-30 (Claude Code GA ships without Grok capture if its proof is not in by 2026-11-24); nothing lost (v0.30.0 shipped on 2026-11-12, followed by v0.30.1 by 2026-11-28 and phase 2 in v0.30.2 by 2026-12-12); facts that update (by 2026-12-16, re-dated at a 2026-12-02 checkpoint); compaction part 1 (by 2026-12-19); cloud capture in two steps (a step 1 design record by 2026-11-21 sets its ship date, no earlier than 2026-12-16; one shared store no earlier than 2027-01-06); more harnesses' capture (by 2026-12-23, each on its own design and proof); and compaction part 2 (by 2026-12-30). Dates mean no later than unless stated as no earlier than or subject to a checkpoint, and each release ships as soon as its gate passes.

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

A passing clone, benchmark, or reference-agent acceptance packet proves only its named plane. Current details and ownership live in the GitHub milestones (the old tracker, #658, was closed on 2026-11-06); #252 holds the score-sensitive eval queue.
