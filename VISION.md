# LCM-X vision

**The memory layer agents think with. Nothing is lost by default.**

LCM-X keeps a lossless record of everything an agent sees and does. Today it runs the context window from that record and lets the agent recall the exact source. It is becoming the layer that derives active memory from the record, puts the right evidence in front of the model unasked, and serves any harness.

It is built on Lossless Context Management (the [LCM paper](https://papers.voltropy.com/LCM) by Ehrlich & Blackman, Voltropy PBC, 2026):
- every message is kept in a durable store;
- older turns are folded into a summary DAG;
- the agent can always walk back from a summary or a memory to the exact source.

This page states what we are building and how we will know it works. The [roadmap](ROADMAP.md) says when; issue [#658](https://github.com/electricsheephq/lcm-x/issues/658) tracks the work.

## The promise

1. **Nothing is lost by default.**
   - Raw rows are stored once and kept, and every summary and derived memory links back to the rows it came from.
   - The one rewrite is a cleanup (on by default from v0.27.0) that moves a large tool output into a file and leaves a reference the agent can expand.
   - Two opt-ins discard content by design: durable sensitive-pattern redaction (`LCM_SENSITIVE_PATTERNS_ENABLED`) and message ignore patterns (`LCM_IGNORE_MESSAGE_PATTERNS`).
2. **The right evidence at the right moment.**
   - Recall finds what matters across months of history. Full-text search is built in today; local semantic search is built in and on by default from v0.29.0 if it passes its measurement, and a built-in reranker follows in v0.29.0 once it passes its measurement (today reranking needs the Voyage cloud provider).
   - Next (v0.31.0): evidence the agent needs is put in front of it before it answers, not only when it thinks to search.
   - The agent can always open the exact source behind what it was shown.
3. **Memory that grows.**
   - Next (v0.30.0): from the record, LCM-X will derive what an agent should know about its user and its work: facts, preferences, entities, events and their times.
   - Each derived memory links to the turns it came from. Derived memory is a view of the record, never a replacement for it.
4. **Long sessions keep working.**
   - The same record runs the context window: a session can run for weeks and the model stays inside its window. Today the user waits for compaction (p90 about 55 s); v0.31.0 removes that wait.
   - Anything compacted can be recovered exactly.
5. **One memory, any harness.**
   - Next (from v0.30.0): the memory an agent builds in Hermes becomes available in other harnesses, Claude Code first, then Codex, OpenClaw and any MCP client.
   - Each event keeps a stable identity across hosts, so the record stays lossless when several harnesses share it.
6. **We measure it.** Claims about recall, continuity and speed come with a pinned commit, a configuration and a published number. From v0.29.0 our rows are read and judged by current frontier models; earlier rows name the judge they used.

## What LCM-X is

- **The memory layer that also runs the context window:** one lossless store behind context assembly, compaction, recall and derived memory.
- **A durable store** (SQLite) with:
  - a summary DAG;
  - externalized large outputs;
  - full-text search, with optional semantic search and cloud reranking today; local semantic search (built in and on by default from v0.29.0 if it passes its measurement), a built-in reranker (v0.29.0) and source-linked derived memory (v0.30.0) next.
- **Agent tools** (`lcm_*`), **operator commands** (`/lcm`, the doctor), and, from v0.30.0, a service endpoint so other harnesses can read the same memory (writes from other harnesses in v0.32.0).

## What LCM-X is not

- **Not a filing cabinet.** Keeping everything is the floor; the product is the right evidence at the moment of need.
- **Not a lossy fact store.** A derived memory never replaces the record it came from; every one links back to its source.
- **Not a fork of any host.** LCM-X integrates through each host's public contract: the Hermes plugin today, hooks and MCP for other hosts from v0.30.0.

## Principles

- **Lossless before clever.** A change that can lose or duplicate a stored row does not ship, however much it improves anything else.
- **Proven quality ships on.**
  - A feature becomes the default when it passes one pre-declared, paired measurement on held-out questions, read by a frontier model.
  - The measurement must show a significant gain in evidence or answer quality.
  - It ships on even when it uses more tokens; we publish tokens per query beside every number, and every default can be turned off. Token cost is not a gate; losslessness and the user's wait are.
- **Built in, not bolted on.** From v0.29.0 the default install includes local semantic search, on by default if it passes its measurement, and a local reranker, if it passes its measurement. Cloud providers are optional upgrades.
- **Local first, offline capable.** The memory layer needs no cloud: one SQLite database, with a folder beside it for large externalized payloads, and local embeddings, built in from v0.29.0. With a local summariser model it runs fully offline. So agents on laptops today, and on-device agents and robots later (research), keep their whole memory on the device.
- **The user's wait is a product metric.** Compaction time and prompt-cache breaks count as much as summary quality.
- **Default configuration is the product.** The numbers we publish are for the configuration users actually run, beside our best configuration and labelled as such.
- **Frontier-first measurement.** Our own benchmark rows use current frontier models as readers and judges, and we publish the protocol and the reasons. Where a shared-reader board (every system run with the same reader and judge by someone other than the vendor) fixes its own models, we enter it on its terms.
- **Release in themes, by gate.** Each themed minor has a written exit criterion, and a feature ships in the first release after it passes its own gate. Patches only for defects that:
  - lose data or duplicate rows;
  - wedge or reset a session;
  - send an over-window request;
  - break security.

## Horizons

| Horizon | Goal | Done when |
|---|---|---|
| **H1 — compaction you don't notice** | Fast, correct compaction, prepared in the background and published at the threshold | Exit targets, not release gates (the release spec holds the gates): LCM-owned compaction time p90 ≤ 60 s and never above 120 s; visible wait p90 ≤ 5 s; no loss or duplicate rows across two consecutive minors |
| **H2 — memory that wins** | Built-in semantic search and reranking, agentic recall, sender and time provenance, and source-linked derived memory | LCM-X places first on the shared-reader boards it enters (long-horizon, agentic and in-host), with frontier-model rows published for the shipped default |
| **H3 — one memory, any harness** | A memory service with an MCP endpoint and adapters, so one record serves several harnesses | Hermes and a second harness share one memory through restarts with stable identities and no duplicate rows; then more adapters and portable export |
| **H4 — memory for agents in the world** | Research: events with time, place and modality, for agents and robots that act over days | One measured use before any schema commitment; the first step is an offline cold start on the smallest supported host that passes the recall and facts-kept bars |

## How we measure

- **Our frontier-model protocol.**
  - Answers are read by current frontier models, and two frontier judges from different families score the same blinded answers.
  - We publish prompts, question ids, configurations and harness commits.
  - Each table carries a same-reader full-context baseline where one fits.
  - Agentic rows (the reader drives LCM-X's tools) and single-retrieval rows are reported separately.
- **Shared-reader boards.** We enter boards that run every system with the same models. Until local semantic recall becomes the default (v0.29.0), we enter the configuration we ship to the managed fleet, labelled as such; after that, the shipped default. We report where we place.
- **Compaction parity.** Facts kept, instruction continuity and compaction wall time against other compaction systems, on the same material, three seeds.
- **Lifecycle reliability.** Every release candidate runs a real-host gauntlet (install, upgrade, rollback, restart, rotation, long sessions). Only a candidate that passes the [release spec](bench/specs/RELEASE-READINESS-V1.md) becomes GA. v0.27.0 shipped with two disclosed exceptions; #1305 proposes a narrow spec clause for those two cases only.

See the [benchmark methodology](benchmarks/METHODOLOGY.md) and [benchmark vision and attribution](bench/VISION-AND-ATTRIBUTION.md).
