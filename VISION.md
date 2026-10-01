# LCM-X vision

**Bounded context, unbounded memory. Nothing is ever lost.**

LCM-X is a lossless context engine for Hermes Agent. It is built on Lossless Context Management (the [LCM paper](https://papers.voltropy.com/LCM) by Ehrlich & Blackman, Voltropy PBC, 2026): every message is kept in a durable store, older turns are folded into a summary DAG, and the agent can always walk back from a summary to the exact source. This page states what we are building and how we will know it works. The [roadmap](ROADMAP.md) says when; issue [#658](https://github.com/electricsheephq/lcm-x/issues/658) tracks the work.

## The promise

1. **Long sessions keep working.** A session can run for weeks. The model's context stays inside its window, and the user does not wait on compaction.
2. **Nothing is lost.** Raw rows are stored once and kept. Every summary links back to the rows it covers. The one rewrite is an optional cleanup, off by default, that moves a large tool output into a file and leaves a reference the agent can expand.
3. **The agent can get the detail back.** Recall and expand tools let the agent search the store and open the source behind any summary or stub.
4. **We measure it.** Claims about speed, continuity and recall come with a pinned commit, a configuration and a published number.

## What LCM-X is

- A context engine plugin: it decides what the model sees each turn and what is folded into summaries.
- A durable store (SQLite) with a summary DAG, externalized large outputs, and full-text and optional semantic search.
- A set of agent tools (`lcm_*`) and operator commands (`/lcm`, the doctor).

## What LCM-X is not

- Not a separate memory provider. Hermes memory providers keep facts about a user; LCM-X keeps the conversation itself, losslessly. They can run side by side.
- Not a vector database product. Semantic search is optional; full-text recall works without it.
- Not a fork of the host. LCM-X runs inside Hermes through the public plugin contract.

## Principles

- **Lossless before clever.** A change that can lose or duplicate a stored row does not ship, however much it improves anything else.
- **The user's wait is a product metric.** Compaction time and prompt-cache breaks count as much as summary quality.
- **Default configuration is the product.** The numbers we publish are for the configuration users actually run.
- **Release in themes.** One themed minor about every two weeks, each with a written exit criterion. Patches only for defects that lose data, duplicate rows, wedge or reset a session, send an over-window request, or break security.

## Three horizons

| Horizon | Goal | Done when |
|---|---|---|
| **H1 — compaction you don't notice** (now → v0.28.0) | Fast, correct compaction, then compaction prepared in the background and published at the threshold | Compaction p90 ≤ 60 s and never above 120 s, then visible wait p90 ≤ 5 s; continuity within band of the best compaction we measure against; no loss or duplicate rows across two consecutive minors |
| **H2 — memory that measurably wins** (in parallel) | Re-baseline recall on the shipped default, then improve it (B3-A provenance) | LongMemEval and LoCoMo rows measured on a pinned release, published with tokens per query and cost |
| **H3 — later** | Teams and multi-agent use, dormant features, memory-provider integration | Revisited after v0.28.0 |

## How we measure

- **Compaction parity:** facts kept, instruction continuity and compaction wall time against other compaction systems, on the same material, three seeds.
- **Lifecycle reliability:** every release candidate passes a real-host gauntlet (install, upgrade, rollback, restart, rotation, long sessions) before GA.
- **Recall:** LongMemEval (retrieval and QA) and LoCoMo, on the default configuration and with embeddings on, as separate rows. See [benchmark methodology](benchmarks/METHODOLOGY.md) and [benchmark vision and attribution](bench/VISION-AND-ATTRIBUTION.md).
