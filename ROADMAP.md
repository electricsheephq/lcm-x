# LCM-X roadmap

Status: reconciled 2026-10-10 (v0.26.2 is the current stable release; v0.27.0 is in progress). The [vision](VISION.md) says what we are building and why. Issue [#658](https://github.com/electricsheephq/lcm-x/issues/658) is the tracker; GitHub milestones hold the issues for each release. The current stable release is recorded in [docs/project-status.md](docs/project-status.md).

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

Every compaction LCM-X starts runs on the turn thread, so the user waits for it. v0.24.9 bounded that wait. v0.25.0 met its bar: on the gauntlet's live soak, compaction ran at p90 55.5 s with the longest at 72.1 s. Compaction breaks the prompt cache; additional breaks from stub swaps are tracked in #788 and have not been isolated. H1 removes the wait in steps.

### Compaction architecture: four layers

The leading systems no longer rely on one lossy summary. They stack four layers, and this is LCM-X's compaction architecture (decided 2026-10-10, tracked in [#658](https://github.com/electricsheephq/lcm-x/issues/658)):

| Layer | What it does | LCM-X | Work |
|---|---|---|---|
| 1. Clear old tool output first | Replace large or old tool results with durable references before summarising | Large-output externalization and active-replay stubbing (first sight above 10k tokens; aged tier above 2k tokens outside the fresh tail) | On by default in v0.27.0 (#1013; the default flip is #1070) |
| 2. Summary | Summarise what is compacted | The summary DAG; the summariser sees the middle of long messages (#611) | v0.27.0; the prompt-v2 flip rule in #660 |
| 3. Deterministic re-insertion | Put back what must not be lost: the user's own words, instructions, plan | The objective survives assembly overflow (#1048); the LCM note reaches the model (#900); the user carry packet (#659); an identifier index (#1066). The host rebuilds the system prompt and re-inserts the todo list. | #659 (v0.27.0 if Eval-2 keeps it before the rc1 freeze, else v0.27.1); #1066 |
| 4. Lossless, searchable history | Let the agent recover anything exactly | `lcm_recall`, `lcm_grep`, `lcm_expand` and `lcm_describe` over the lossless store; stemmed full-text recall (#1035); local semantic recall for managed deployments (#1014, #1063) | v0.27.0 |

**Why layer 3 comes first.** On Track S (checkpoint 304, LCM-X v0.24.8 against Codex native compaction), Codex kept 132/132 facts stated by the user against LCM-X's 75/132, and 27/72 facts from the middle of long tool outputs against 0/72. LCM-X was ahead on assistant facts (+20) and on the head and tail of tool outputs (+21). The Codex arm was read by a different reader model in that run.
- The tool-middle gap was the summariser input clip, fixed by #611: 42/72 middle facts now reach leaf summaries.
- The user-fact gap is the missing verbatim user carry; #611 does not change it. In an offline screen with the same reader, the carry packet raised user-row facts by 50.8 points with no losses (#659). Eval-2 decides ([#1064](https://github.com/electricsheephq/lcm-x/issues/1064)).

**Never below Hermes built-in.**
- Hermes's built-in compressor keeps user messages verbatim, an identifier index, a recovery footer and stubs for old tool results.
- When LCM-X is the context engine it replaces that compressor. The floor is measured, not assumed: LCM-X must score at least as well as Hermes built-in on every fact class, read by the same reader in Eval-2's Hermes built-in arm (#1064).
- v0.27.0 records that comparison; v0.28.0 gates on it. Layer 1 (#1070) and the objective and LCM-note parts of layer 3 (#1048, #900) ship in v0.27.0. The user carry packet (#659) ships when Eval-2 keeps it. The identifier index (#1066) ships only if Eval-2 measures a gain.

**Deferred, with revisit triggers.**
- **Summary-free compaction**, where the model writes notes and a fresh window starts. Revisit if a provider enables it for a generally available model; LCM-X's layer 4 already provides the history tool that design depends on.
- **Delegating the active-window summary to provider-native compaction.** Revisit after coexistence with Hermes native compaction is defined (#1068) and an Eval-2 arm shows a gain.
- **Turn-relevant evidence injection.** Revisit when the pre-answer arm passes its pre-declared bar.

| Release | Theme | GA accepts when |
|---|---|---|
| **v0.24.9** | Drain: bounded foreground compaction, capped hidden leaves, exit fit | Shipped 2026-10-01 |
| **v0.25.0** | Long sessions: drain speed and correctness (#597, #750, removal of native recovery) | Shipped 2026-10-04 |
| **v0.26.0** | Host message identity: record the host's stable message id beside LCM-X's own row matching (shadow mode; `on` mode is deferred, #643) | Shipped 2026-10-07 |
| **v0.27.0** | Summary quality; release candidate planned for mid-October. Scope: <ul><li>the summariser sees the middle of long messages (#611: each leaf's messages share the leaf token budget; on synthetic material, middle-placed facts reaching leaf summaries rise from 0/72 to 42/72; this also closes #899);</li><li>the LCM note and the retained-user anchor reach the model on Hermes (#900);</li><li>fixes for long single-turn tool loops (#921, #922, #923), compaction that grows the context on heartbeat sessions (#904, #930, #931), survival fits (#798, #916) and pre-leaf condensation (#909);</li><li>instruction-continuity probes (#659), recorded in this release;</li><li>product defaults set to the measured best configuration (#1013): compaction at 75%, fresh tail 24 messages / 24k tokens, leaf chunks of 8k tokens, full sweep, rollups, and large-output externalization and replay stubbing on (the last two flip in #1070);</li><li>the objective survives assembly overflow (#1048); the inline floor scales with the context window (#1055);</li><li>recall: a stemmed full-text index (#1035), working `lcm_grep` OR/NOT/prefix operators (#1015), incremental embedding (#1014) and local-profile auto-registration (#1063), so managed deployments can turn on local semantic recall;</li><li>dormant tools are not advertised while their features are off (#1017);</li><li>the user carry packet (#659) if Eval-2 keeps it before the rc1 freeze.</li></ul>The summary prompt default stays v1 (#660). v2 has not passed the flip rule: at least as good as v1 on every paired axis, better on one, within the latency bars, and with a level-3 rate within band of zero; changing the default remains an owner decision. | <ul><li>No system-level axis below the v0.26.0 row: paired per-fact exact McNemar on identical v3 material, α 0.05, minimum effect 5 points; add runs when run1/run2 spread exceeds 0.10. QA rows use their own paired rule.</li><li>The theme gain is present: middle-placed facts reach summaries, and constraint continuity is at or above the floor declared from the v0.26.0 baseline.</li><li>Compaction p90 ≤ 60 s and longest ≤ 120 s on the ≥ 24 h soak; gauntlet cells ≤ 145 s.</li><li>Recall (#898): zero embeddings-off per-question flips versus v0.26.0, or each flip attributed and classified intended within the net −0.005 / category −0.02 floors.</li><li>Eval-2 recorded per fact class against the Hermes built-in arm with the same reader (#1064).</li><li>No release-blocking defects.</li></ul> |
| **v0.28.0** | Invisible compaction: summaries prepared in the background and published at the threshold (#787, with #610, #626, #749, #775, #788, #926) | <ul><li>Visible-wait p90 ≤ 5 s on the reference-agent soak.</li><li>No compaction slower than v0.24.9.</li><li>The continuity suite (#659) passes its per-class bars, declared from the v0.27.0 run (100% for host instructions).</li><li>No system-level axis below v0.27.0 under the same paired rule as v0.27.0.</li><li>Recall (#898): zero embeddings-off per-question flips versus v0.27.0, or each flip attributed and classified intended within the net −0.005 / category −0.02 floors.</li><li>At or above the Hermes built-in arm on every Eval-2 fact class (#1064).</li><li>No release-blocking defects.</li></ul> |

LCM-X is also compared at system level with Codex native compaction and lossless-claw ([#664](https://github.com/electricsheephq/lcm-x/issues/664)). That comparison is H1 exit evidence, not a release gate. The last run was on v0.24.8 ([#658](https://github.com/electricsheephq/lcm-x/issues/658)). The next is reported at the v0.27.0 release candidate.

Plan, todo and goal survival across a compaction is host behaviour. Fixes are proposed upstream in Hermes and tracked in [#901](https://github.com/electricsheephq/lcm-x/issues/901). LCM-X's own part is [#900](https://github.com/electricsheephq/lcm-x/issues/900).

## H2 — memory that measurably wins (in parallel)

1. Re-baseline recall on v0.26.0 in three configurations. LongMemEval-M retrieval is done for two of them on the release candidate, whose product code is unchanged in the GA-notes merge (the only added file is the GA release notes):
   - the default (embeddings off, full-text recall): R@10 0.847, with no question changed against v0.25.1;
   - a cached Voyage tracker: 0.950 on the 456 questions it scores (0.847 without embeddings on the same questions). This is retrieval only, and not the production contextual Voyage path.

   - local fastembed: R@10 +3.05 points over the default on 470 paired questions (49 wins, 20 losses). Managed deployments turn it on with v0.27.0, one profile first.

   Measured offline after stemming and rejected against pre-declared bars: session-level scoring (#1039), a session-aware hit cap (#1040), time ranges read from the query (#1025), hit diversity (#1023), session score (#1022), entity and identifier postings, and returning neighbours with each hit. Offline cost to find is near its floor (about 1.2 calls per question). The remaining gap is in the agent loop and is measured on real-host runs.

   Next: answer accuracy on LongMemEval-S and LoCoMo as regression rows, with a judge from a different model family than the generator; then recall latency at scale.
2. B3-A: treat retrieved context as untrusted evidence (#317), then keep timestamp, role and sender provenance in summariser inputs (#324), then re-measure.
3. Publish the results in the repository scoreboard with tokens per query and cost. The v0.25.1 and v0.26.0 retrieval rows are added after the v0.26.0 GA.
4. Adopt forward-leaning benchmarks for long-horizon memory and compaction, after verifying each harness and its cost ([#1067](https://github.com/electricsheephq/lcm-x/issues/1067)).

Milestone: **H2 — memory that measurably wins**.

## H3 — later

Teams and multi-agent use, dormant features (the fact layer is blocked on #694, #987 and #1065; query views on #694; proactive recall on #962, #969, #1008 and #865), and memory-provider integration. Rollups moved to v0.27.0 as a default. Revisited after v0.28.0. Milestone: **H3 — later**.

## Backlog

Verified items outside the current themes sit in the **Backlog (unscheduled)** milestone. They are pulled into a themed minor by priority (agent-experience impact × exposure × fit ÷ effort), not merged ad hoc.
