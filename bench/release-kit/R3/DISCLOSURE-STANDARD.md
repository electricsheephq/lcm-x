# The Agent-Memory Benchmark Disclosure Standard (proposed practice, R3 draft 1)

## Why this exists
Agent-memory benchmark numbers are published into a market that has already moved past the
conditions they were measured under. Leaderboards mix readers ranging from two-year-old fixed
models to current frontier stacks; judges differ per vendor and are rarely named; datasets float
on `main` branches; run configs are undisclosed; documented dataset errors go unmentioned; and
single runs are presented without any statement of run-to-run noise. The numbers are not wrong —
they are **unanchored**: a reader cannot tell what world a number measured, so numbers cannot be
compared, reproduced, or even dated.

We propose the minimum disclosure that makes a published agent-memory number mean something.
We apply it to every number we publish — including our worst ones. Our own LoCoMo 47% is on our
scoreboard with its complete failure decomposition attached, because a low number with its
mechanism ledger teaches more than a high number with nothing.

## The seven points
Every published number carries, in the same artifact as the number itself:

1. **Fix-state.** The exact system commit(s) measured, and any post-run delta named and classified.
2. **Judge identity + the full judge prompt.** Not "an LLM judge" — the model, its config, and the
   verbatim prompt text (or a pinned link to it). Cross-family judge/answerer disclosed as such.
3. **Known dataset-defect exposure.** If the dataset has documented gold errors, state the exposure
   and the implied score ceiling (e.g., LoCoMo's 99 documented corrupted-gold rows → a 95.02%
   ceiling for our slice). "None documented" is itself a disclosure.
4. **Retrieval/delivery config, every knob named.** Embedder, caps, fusion policy, thresholds —
   with the measured justification when a knob was chosen after diagnosis. Config chosen after
   diagnosis is legitimate engineering; disclosure is what keeps it honest.
5. **Per-category breakdown** (or a pinned artifact holding it). Aggregates hide mechanisms.
6. **Run count + variance.** Single-run numbers say so. Paired claims carry discordant counts and
   the pre-registered gate they were read against, not just a delta.
7. **Fail-close accounting.** How many rows were excluded or zero-scored by instrument fault, in
   which arm, under which convention. A number whose denominator is silent is not a number.

## The stronger claims this enables
- **Non-inferiority and gains become auditable**: our V2 re-baseline shipped on a pre-registered
  paired gate (net +3, b=29/c=26, n=434) adjudicated blind from instrument outputs — the verdict
  record is published beside the number.
- **Fault-finding results become publishable**: Tier-F numbers (frontier-stack, fault-hunting
  configs) sit on the same board as Tier-P numbers (published-claims discipline), distinguished
  by a tier column instead of hidden.
- **Comparisons across systems become honest**: where another board's config differs, the caveat
  is one precise line instead of an unstated assumption.

## Reference implementation
Our scoreboard (`bench/scoreboard/`) enforces this schema mechanically: a row missing any of the
seven fields does not render. Rows are append-only; superseding runs link back rather than erase.
The generator, schema, and every row's evidence trail are in this repository.
