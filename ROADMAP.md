# LCM-X roadmap

Status: reconciled ⟦GA_DATE⟧ for the v0.26.0 GA. The [vision](VISION.md) says what we are building and why. Issue [#658](https://github.com/electricsheephq/lcm-x/issues/658) is the tracker; GitHub milestones hold the issues for each release. The current stable release is recorded in [docs/project-status.md](docs/project-status.md).

## How releases work

- **One themed minor about every two weeks.** Each minor has one theme and a written "GA accepts when" line. Scope freezes at the first release candidate. After the third candidate, a minor ships only with known, non-blocking items; otherwise the failing change is taken out.
- **Patches only for release-blocking defects reachable in real deployments:** data loss, duplicate rows, a wedged or reset session, a request over the context window, or a security defect such as a secret exposure or an access-control bypass. At most one patch a week unless an incident is live.
- **Behaviour and default changes ship only in minors.** v0.24.9 was scoped before this policy was written; it is the last release under the old rule.
- **Every release candidate runs the gauntlet** in [bench/specs/RELEASE-READINESS-V1.md](bench/specs/RELEASE-READINESS-V1.md). A candidate that fails is fixed and respun; only a candidate that passes can become GA, and the GA tree differs from it only by its release notes.
- **Every release candidate also runs a recall scorecard** ([#898](https://github.com/electricsheephq/lcm-x/issues/898)), first applied at v0.26.0:
  - The default configuration (embeddings off) is scored on LongMemEval retrieval against the previous GA. A question whose `lcm_recall` result changes must be traced to a named commit and judged intended. The net R@10 may not fall more than 0.005 and no category more than 0.02.
  - Two embeddings-on configurations are recorded once per minor: a cached Voyage tracker and local fastembed. Each starts gating once a repeat run in the same session agrees with itself on every question.
  - Comparisons with other compaction systems are reported as H1 evidence, not per-release gates.

## H1 — compaction you don't notice

Every compaction LCM-X starts runs on the turn thread, so the user waits for it. v0.24.9 bounded that wait. v0.25.0 met its bar: on the gauntlet's live soak, compaction ran at p90 55.5 s with the longest at 72.1 s. Each compaction also costs one prompt-cache break. H1 removes the wait in steps.

| Release | Theme | GA accepts when |
|---|---|---|
| **v0.24.9** | Drain: bounded foreground compaction, capped hidden leaves, exit fit | Shipped 2026-10-01 |
| **v0.25.0** | Long sessions: drain speed and correctness (#597, #750, removal of native recovery) | Shipped 2026-10-04 |
| **v0.26.0** | Host message identity: record the host's stable message id beside LCM-X's own row matching (shadow mode) | Shipped ⟦GA_DATE⟧ |
| **v0.27.0** | Summary quality. Scope: <ul><li>the summariser sees the middle of long messages (#611: in a synthetic test, middle-placed facts reaching leaf summaries rise from 0/72 to 42/72; this also closes #899);</li><li>the summary prompt v2 default decision (#660: v2 must be at least as good as v1 on every measured axis and better on one). The first paired run at v0.26.0-rc1 found v2 keeps far more facts but stalls: its depth-1 condensations overrun their output cap and compaction is held (#646), so v0.27.0 fixes condensation stability first and then re-runs the comparison;</li><li>#798, #900 and #906.</li></ul> | <ul><li>No system-level axis below the v0.26.0 row: paired per-fact test on identical material, α 0.05, minimum effect 5 points.</li><li>The theme gain is present.</li><li>Compaction p90 ≤ 60 s and longest ≤ 120 s on the ≥ 24 h soak.</li><li>No release-blocking defects.</li></ul> |
| **v0.28.0** | Invisible compaction: summaries prepared in the background and published at the threshold (#787, with #610, #626, #775, #788) | <ul><li>Visible-wait p90 ≤ 5 s.</li><li>No compaction slower than v0.24.9.</li><li>The continuity suite (#659) passes its per-class bars, which are declared from the v0.27.0 run.</li></ul> |

Plan, todo and goal survival across a compaction is host behaviour. It is being fixed in Hermes and tracked in [#901](https://github.com/electricsheephq/lcm-x/issues/901). LCM-X's own part is [#900](https://github.com/electricsheephq/lcm-x/issues/900).

## H2 — memory that measurably wins (in parallel)

1. Re-baseline recall on v0.26.0 in three configurations. LongMemEval retrieval is done for two of them on the release candidate, whose tree the GA ships unchanged:
   - the default (full-text recall): R@10 0.847, with no question changed against v0.25.1;
   - a cached Voyage tracker: 0.950 on the 456 questions it scores (0.847 without embeddings on the same questions).
   Next: local fastembed, then LongMemEval-S QA and LoCoMo. Turning embeddings on by default for managed deployments is decided on this data.
2. B3-A: treat retrieved context as untrusted evidence (#317), then keep timestamp, role and sender provenance in summariser inputs (#324), then re-measure.
3. Publish the results in the repository scoreboard with tokens per query and cost; add BEAM.

Milestone: **H2 — memory that measurably wins**.

## H3 — later

Teams and multi-agent use, dormant features (assertions, query views, rollups), and memory-provider integration. Revisited after v0.28.0. Milestone: **H3 — later**.

## Backlog

Verified items outside the current themes sit in the **Backlog (unscheduled)** milestone. They are pulled into a themed minor by priority (agent-experience impact × exposure × fit ÷ effort), not merged ad hoc.
