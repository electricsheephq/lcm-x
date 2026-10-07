# LCM-X roadmap

Status: reconciled 2026-10-07 for the v0.26.0 GA. The [vision](VISION.md) says what we are building and why. Issue [#658](https://github.com/electricsheephq/lcm-x/issues/658) is the tracker; GitHub milestones hold the issues for each release. The current stable release is recorded in [docs/project-status.md](docs/project-status.md).

## How releases work

- **One themed minor about every two weeks.** Each minor has one theme and a written "GA accepts when" line. Scope freezes at the first release candidate. After the third candidate, a minor ships only with known, non-blocking items; otherwise the failing change is taken out.
- **Patches only for release-blocking defects reachable in real deployments:** data loss, duplicate rows, a wedged or reset session, a request over the context window, or a security defect such as a secret exposure or an access-control bypass. At most one patch a week unless an incident is live.
- **Behaviour and default changes ship only in minors.** v0.24.9 was scoped before this policy was written; it is the last release under the old rule.
- **Every release candidate runs the gauntlet** in [bench/specs/RELEASE-READINESS-V1.md](bench/specs/RELEASE-READINESS-V1.md). A candidate that fails is fixed and respun; only a candidate that passes can become GA, and the GA tree differs from it only by its release notes.
- **Every release also carries a recall scorecard** ([#898](https://github.com/electricsheephq/lcm-x/issues/898)), first applied at v0.26.0:
  - **Every release candidate:** the default configuration (embeddings off) is scored on LongMemEval-M retrieval against the previous GA. It passes when no question's `lcm_recall` result changes. Otherwise each change must be traced to a named commit and judged intended. Net R@10 may not fall more than 0.005, and no category may fall more than 0.02.
  - **Once per minor:** two embeddings-on configurations are recorded, a cached Voyage tracker and local fastembed. Each starts gating once a repeat run in the same session agrees with itself on every question.
  - **Once per minor, at the first release candidate:** facts kept after compaction (Track S) are compared with the previous GA's row by a paired per-fact test on identical material. Answer accuracy (LongMemEval-S, LoCoMo) uses its own paired rule.
  - Comparisons with other compaction systems are reported as H1 evidence, not per-release gates.

## H1 — compaction you don't notice

Every compaction LCM-X starts runs on the turn thread, so the user waits for it. v0.24.9 bounded that wait. v0.25.0 met its bar: on the gauntlet's live soak, compaction ran at p90 55.5 s with the longest at 72.1 s. Each compaction also costs one prompt-cache break. H1 removes the wait in steps.

| Release | Theme | GA accepts when |
|---|---|---|
| **v0.24.9** | Drain: bounded foreground compaction, capped hidden leaves, exit fit | Shipped 2026-10-01 |
| **v0.25.0** | Long sessions: drain speed and correctness (#597, #750, removal of native recovery) | Shipped 2026-10-04 |
| **v0.26.0** | Host message identity: record the host's stable message id beside LCM-X's own row matching (shadow mode; `on` mode is deferred, #643) | Shipped 2026-10-07 |
| **v0.27.0** | Summary quality; release candidate planned for mid-October. Scope: <ul><li>the summariser sees the middle of long messages (#611: each leaf's messages share the leaf token budget; on synthetic material, middle-placed facts reaching leaf summaries rise from 0/72 to 42/72; this also closes #899);</li><li>the LCM note and the retained-user anchor reach the model on Hermes (#900);</li><li>fixes for long single-turn tool loops (#921, #922, #923), compaction that grows the context on heartbeat sessions (#904, #930, #931), survival fits (#798, #916) and pre-leaf condensation (#909);</li><li>instruction-continuity probes (#659), recorded in this release.</li></ul>The summary prompt default stays v1 (#660). v2 has not passed the flip rule: at least as good as v1 on every measured axis, better on one, and within the latency bars. | <ul><li>No system-level axis below the v0.26.0 row: paired per-fact test on identical material, α 0.05, minimum effect 5 points.</li><li>The theme gain is present: middle-placed facts reach summaries, and constraint continuity is at or above the floor declared from the v0.26.0 baseline.</li><li>Compaction p90 ≤ 60 s and longest ≤ 120 s on the ≥ 24 h soak; gauntlet cells ≤ 145 s.</li><li>No release-blocking defects.</li></ul> |
| **v0.28.0** | Invisible compaction: summaries prepared in the background and published at the threshold (#787, with #610, #626, #749, #775, #788, #926) | <ul><li>Visible-wait p90 ≤ 5 s on the reference-agent soak.</li><li>No compaction slower than v0.24.9.</li><li>The continuity suite (#659) passes its per-class bars, declared from the v0.27.0 run (100% for host instructions).</li></ul> |

LCM-X is also compared at system level with Codex native compaction and lossless-claw ([#664](https://github.com/electricsheephq/lcm-x/issues/664)). That comparison is H1 exit evidence, not a release gate. The last run was on v0.24.8 ([#658](https://github.com/electricsheephq/lcm-x/issues/658)). The next is reported at the v0.27.0 release candidate.

Plan, todo and goal survival across a compaction is host behaviour. Fixes are proposed upstream in Hermes and tracked in [#901](https://github.com/electricsheephq/lcm-x/issues/901). LCM-X's own part is [#900](https://github.com/electricsheephq/lcm-x/issues/900).

## H2 — memory that measurably wins (in parallel)

1. Re-baseline recall on v0.26.0 in three configurations. LongMemEval-M retrieval is done for two of them on the release candidate, whose tree the GA ships unchanged:
   - the default (embeddings off, full-text recall): R@10 0.847, with no question changed against v0.25.1;
   - a cached Voyage tracker: 0.950 on the 456 questions it scores (0.847 without embeddings on the same questions). This is retrieval only, and not the production contextual Voyage path.

   Next: local fastembed; then answer accuracy on LongMemEval-S and LoCoMo, the first such rows on a current release; then recall latency at scale. Turning embeddings on by default for managed deployments is decided on this data.
2. B3-A: treat retrieved context as untrusted evidence (#317), then keep timestamp, role and sender provenance in summariser inputs (#324), then re-measure.
3. Publish the results in the repository scoreboard with tokens per query and cost. The v0.25.1 and v0.26.0 retrieval rows are added after the v0.26.0 GA. Then add BEAM.

Milestone: **H2 — memory that measurably wins**.

## H3 — later

Teams and multi-agent use, dormant features (assertions, query views, rollups), and memory-provider integration. Revisited after v0.28.0. Milestone: **H3 — later**.

## Backlog

Verified items outside the current themes sit in the **Backlog (unscheduled)** milestone. They are pulled into a themed minor by priority (agent-experience impact × exposure × fit ÷ effort), not merged ad hoc.
