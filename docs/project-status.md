# LCM-X project state

This page separates released product identity, current development source, evaluation evidence, and later roadmap work. GitHub issues, pull requests, tags, releases, and exact commit heads are the live source of truth. This snapshot was reconciled on 2026-09-05 and is updated at every release; the latest update is 2026-10-19, for v0.28.1-rc1 preparation.

## Naming and compatibility

The project is **LCM-X — Lossless Context Memory eXtension**. From v0.24.0 (#471) the plugin and engine identifiers are:

| Surface | Identifier |
| --- | --- |
| Repository and project | `electricsheephq/lcm-x` / LCM-X |
| Plugin manifest and install directory | `hermes-lcm-x` (legacy, v0.23.x and earlier: `hermes-lcm`) |
| Runtime context engine | `lcm-x` (legacy alias `lcm` accepted with a deprecation warning) |
| Bundled skill | `hermes-lcm` (unchanged) |
| Latest stable | `v0.28.0@709099b3a397a7893a7abae8802ebe7bf605f79e` (ships `hermes-lcm-x` / `lcm-x`; a minor release on `v0.27.0`; its tree is the `v0.28.0-rc1` tag plus the GA notes file) |

The rename is a breaking change with a documented migration path; see the operator guide's migration section. Historical notes and upstream evidence retain the names and identities used when they were created.

## Released product and development source

The latest stable release is `v0.28.0` at `709099b3a397a7893a7abae8802ebe7bf605f79e`, a minor release on `v0.27.0` with the theme repeated compaction and slow summarisers. It changes no default and adds no dependency and no schema change; rollback to `v0.27.0` is plugin-only. Background leaf preparation slice 1 ships off (#787). A hung summary route opens its circuit (#1137), and extraction leaves the summary call its time (#1122). Short user and assistant turns now receive embeddings (#1197). Local semantic recall stays optional, with FastEmbed installed separately; it is built in and turned on by default in v0.29.0 (#1186). It requires Hermes 0.21.3 or newer, and 0.21.5 or newer is recommended (#1080). Its gauntlet ran at the `v0.28.0-rc1` tag and passed, with four release-manager decisions disclosed in the release notes (#1554). Because GitHub reports the tag as mutable, operators and evidence packets must verify the exact SHA rather than trust the tag name alone. The previous stable release is `v0.27.0` at `c67d7633d963a2ffd24cf90995189ee58e2f88c7`.

The source snapshot used for this reconciliation is `main@03443712e81d492f41e08b424adc80bd5b33f8f6`, which contains the `v0.28.0` tag commit and later merges. Stable and main are different proof planes: stable is the released product baseline, while main contains later development and documentation work. Do not describe a main checkout as the installed stable release merely because it contains stable commits.

Main carries the forward identity `0.28.1` (`hermes-lcm-x`), first tagged as `v0.28.1-rc1`; `v0.28.0` remains the latest stable release. The candidate is not qualified yet. The patch covers session continuity, replay matching, doctor integrity checks and chunk coverage after rollback. Releases with product code are rc-first under `bench/specs/RELEASE-READINESS-V1.md`; the release identity test keeps `plugin.yaml`, README, the operator guide, CHANGELOG and the bug-report template synchronized.

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

Issue #658 is the tracker, and [ROADMAP.md](../ROADMAP.md) lays the work out by horizon and milestone. H1 compaction continues in v0.28.1 (patch-class follow-ups, including wedge diagnostics for #1162; reconstruction #1598 is in v0.29.0), v0.29.0 (the user's current request kept verbatim, #1592) and v0.31.0 (invisible compaction, plus pre-answer evidence #1111 once it passes the default-on test and the user carry packet decision #659). The summary prompt default stays v1 (#660). H2 recall work runs in v0.29.0 and v0.30.0. v0.29.0 is the evidence release:<ul><li>delivery depth step 1, done and passed (#1607);</li><li>the time-order slice (#1589);</li><li>delivery depth step 2, held-out answers (#1607);</li><li>rerank on the chosen pool (#1187);</li><li>within-session depth and breadth (#1188);</li><li>sender and role (#995);</li><li>local semantic recall on by default #1186;</li><li>agentic memory.</li></ul>v0.30.0 brings source-linked derived memory (#1191), full timeline assembly, source-linked superseded facts (#1590, with #1191), abstention, and a second harness. Step 0 rejected #1590's delivery-only slice: the newer update was never delivered in 36 of 67 HaluMem Dynamic Update misses, and only 4 had both states delivered.

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
