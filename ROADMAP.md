# LCM-X roadmap

Status: reconciled 2026-10-01. The [vision](VISION.md) says what we are building and why. Issue [#658](https://github.com/electricsheephq/lcm-x/issues/658) is the tracker; GitHub milestones hold the issues for each release. The current stable release is recorded in [docs/project-status.md](docs/project-status.md).

## How releases work

- **One themed minor about every two weeks.** Each minor has one theme and a written "GA accepts when" line. Scope freezes at the first release candidate. After the third candidate, a minor ships only with known, non-blocking items; otherwise the failing change is taken out.
- **Patches only for release-blocking defects reachable in real deployments:** data loss, duplicate rows, a wedged or reset session, a request over the context window, or a security defect such as a secret exposure or an access-control bypass. At most one patch a week unless an incident is live.
- **Behaviour and default changes ship only in minors.** v0.24.9 was scoped before this policy was written; it is the last release under the old rule.
- **Every release candidate runs the gauntlet** in [bench/specs/RELEASE-READINESS-V1.md](bench/specs/RELEASE-READINESS-V1.md). A candidate that fails is fixed and respun; only a candidate that passes can become GA, and the GA tree differs from it only by its release notes.

## H1 — compaction you don't notice

Today every compaction LCM-X starts runs on the turn thread, so the user waits for it. v0.24.9 bounds that wait: it aims to finish within 60 s, and it limits summary-call admission and request timeouts to a 120 s budget. It does not interrupt a call already running, so a compaction can still end past 120 s. Making 120 s a true maximum is part of the v0.25.0 exit below. Each compaction also costs one prompt-cache break. H1 removes the wait in steps.

| Release | Theme | GA accepts when |
|---|---|---|
| **v0.24.9** | Drain: bounded foreground compaction, capped hidden leaves, exit fit | The release gauntlet passes |
| **v0.25.0** | Long sessions: drain speed and correctness. The first background step: leaves prepared below the threshold (#610, minimal). Also #597, #750, #626, #605, #625, #84, #7, the cache-break measurement (#788), and removal of native recovery | Compaction p90 ≤ 60 s and never above 120 s on the field-window counter and the gauntlet; no release-blocking defects |
| **v0.26.0** | Host message identity: use the host's stable message id (shadow mode first), then simplify LCM-X's own row matching | Shadow mode agrees with content matching on ≥ 99% of rows over 24 h, and every mismatch is explained; the full reliability matrix passes |
| **v0.27.0** | Summary quality: prompt v3 by default (#660), summariser input cue lines and, if latency permits, a wider per-message clip (#611), instruction continuity (#659) | Facts kept at least as high as the best compaction we measure against, within the three-seed band; continuity checks pass |
| **v0.28.0** | Invisible compaction: summaries prepared in the background and published at the threshold (#787) | Visible-wait p90 ≤ 5 s, and no compaction slower than v0.24.9 |

## H2 — memory that measurably wins (in parallel)

1. Re-baseline recall on a pinned release: LongMemEval retrieval and QA, and LoCoMo, for the default configuration (full-text recall) and, as a separate row, with embeddings on.
2. B3-A: treat retrieved context as untrusted evidence (#317), then keep timestamp, role and sender provenance in summariser inputs (#324), then re-measure.
3. Publish the results in the repository scoreboard with tokens per query and cost; add BEAM.

Milestone: **H2 — memory that measurably wins**.

## H3 — later

Teams and multi-agent use, dormant features (assertions, query views, rollups), and memory-provider integration. Revisited after v0.28.0. Milestone: **H3 — later**.

## Backlog

Verified items outside the current themes sit in the **Backlog (unscheduled)** milestone. They are pulled into a themed minor by priority (agent-experience impact × exposure × fit ÷ effort), not merged ad hoc.
