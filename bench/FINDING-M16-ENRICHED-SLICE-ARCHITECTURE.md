# M16 — The dev-loop bottleneck was slice COMPOSITION, not speed. New two-tier measurement architecture.

**Date:** 2026-07-25 · **Status:** architecture decision (architect, under owner delegation)
**Supersedes:** the "60q stratified slice" dev-loop convention in §2 · **Answers:** owner question on fast iteration

---

## 1. The diagnosis

Today we ran four arms (L1–L3, M7) in ~23 minutes each and could not learn from any of them. The instinct
is to blame speed. **The data says the problem is composition.**

The full benchmark is 451 questions: **128 abstention / 323 answerable**. A *random* 60-question dev slice
therefore contains **~17 abstention questions** (we got exactly 17). M7 is a mechanism that acts almost
entirely on the abstention class. So we tested an abstention mechanism against a 17-question sample.

Power to detect the effect M7 actually produced (29.4% → 52.9%, +23.5 pts), holding the measured discordant
ratio (b=5, c=1) and scaling with slice size:

| abstention n | b / c | McNemar p | detectable? |
|---|---|---|---|
| **17 (current random 60q)** | 5 / 1 | **0.219** | **no** |
| 40 | 12 / 2 | 0.013 | yes |
| 64 | 19 / 4 | 0.003 | yes |
| **128 (all)** | 38 / 8 | **<0.0001** | yes |

And for a smaller, more typical effect (+8 pts, ratio b=3/c=1): 17 → p=0.625, 64 → p=0.119,
**128 → p=0.011**. Only the full abstention set can see a realistic mechanism.

**This is the root cause of today's entire audit cycle.** M11's non-significant effort deltas, M13's flat
static loop, M15's unreadable primary — all four are the same underdesigned instrument. We were not
unlucky; we were under-powered by construction.

## 2. Why NOT the obvious alternative (a faster/stronger reader, e.g. the Sol protocol)

The owner proposed iterating on the Sol internal protocol for speed, then validating officially. The
instinct — cheap loop, expensive gate — is exactly right. **Sol is the wrong way to buy it.**

Sol measures a **frontier** reader. M1/M3 established that this benchmark's difficulty *is* delivering
compact evidence into a **weak** fixed consumer (Qwen3.5-9B), and that curation quality into that weak
consumer is the entire static→agentic gap. A frontier reader tolerates bulk context and therefore **masks
the exact failure mode we are optimising.** We have already paid for this lesson once: Sol scored 205/451
where the official protocol scored 125/451 — a 80-question illusion.

**Rule: the dev loop and the gate must share the same consumer.** Buy iteration speed from *question
selection*, never from *reader strength*.

## 2b. ★ OWNER CORRECTION — Sol is the PRODUCT, not a deprecated instrument (owner, mid-turn 07-25)

§2 above is correct about the leaderboard and **wrong as product strategy**, and the distinction matters
more than the measurement point. Owner: *"Sol won't provide official benchmark numbers, but it is the
product we use in real life. It's our default product for us and our customers... they are the results we
actually rely on."*

So the program has **two objectives, and they must not be collapsed**:

| objective | consumer | why it matters |
|---|---|---|
| **Leaderboard score** (LME-V2) | fixed weak Qwen3.5-9B | public credibility, first-mover slot |
| **Product quality** | Sol / frontier readers | what customers actually experience — revenue reality |

**The risk this creates, named explicitly: optimising curation for a weak reader can DEGRADE the product.**
Padding evidence packs with absence disclosures, hand-holding scaffolding, or compactness hacks that a 9B
model needs may be noise-or-worse to a frontier consumer. A leaderboard-only mechanism shipped default-on
is a product regression we would not have measured.

**Reframe of today's results under the product objective:** the M12 latency finding — **-22% per question,
p<0.0001** — is not merely a corrected claim, it is **the most product-relevant result of the day.** It is
consumer-independent: a faster memory is faster for Sol users too. Conversely M7's abstention mechanism is
currently a *leaderboard* hypothesis whose product value is untested.

**DECISION — dual-consumer evaluation, and it is nearly free.** The expensive stage (agent curation) is
shared; only the reader differs. So we re-read the **same stored `memory_context` packs** with a frontier
reader instead of re-running the agent — a reader-only second pass at roughly **1.2x cost, not 2x**.
Every mechanism from now on reports:
- **PRIMARY (leaderboard):** official fixed 9B reader, enriched slice, paired.
- **PRODUCT CHECK (Sol):** same packs, frontier reader. Does the mechanism help, no-op, or HARM the
  consumer we actually ship?

**Ship rule:** helps both → ship default-on. Helps leaderboard, harms product → **config-gated, off by
default, labelled benchmark-only** (never silently default-on). Helps product, no leaderboard effect →
ship it anyway; that is the real business. This also un-parks H6-P5 (#156), which was exactly the
frontier-consumer measurement, and promotes it from "owner-gated curiosity" to a standing gate component.

## 3. DECISION — two-tier measurement architecture

**Tier 1 — mechanism loop (fast, cheap, POWERED on the axis under test).**
A **purpose-built enriched slice**, composed for the mechanism, not sampled at random:
- for abstention mechanisms (M7 family): **all 128 abstention questions + 64 random answerable** = 192q.
  Primary axis fully powered; answerable side serves as the harm floor.
- for answerable/delivery mechanisms (wave-3.5 family): invert it — enrich the answerable class.
- **Both arms run on the identical frozen manifest, always paired** (M12: paired McNemar, never headline
  percentages). A control arm at matched config is part of the cost, not an optional extra.

**Tier 2 — confirmation gate.** Full 451, official protocol, for anything to be banked, published, or
submitted. Unchanged. **Plus the Tier-1 product check must be green or explicitly config-gated (§2b) before
any mechanism ships default-on in the product.**

**The 60-question random slice is RETIRED as a mechanism instrument.** It survives only as a smoke test for
"does the pipeline run", never for "does the mechanism work."

## 4. Cost — this is cheap, which is why the old convention was a false economy

The agentic lane ran 60q in ~23 min/arm concurrently. 192q ≈ **~75 min/arm**; two arms (control + variant)
≈ **~2.5h** and a few dollars. That is roughly one M7-scale afternoon for a result we can actually read,
versus four arms across a day that told us nothing measurable. **The old 60q loop was not faster. It was
cheaper per run and worthless per conclusion.**

## 5. What this changes about how we report
Every mechanism result from now on carries: the slice composition (n per class), the paired McNemar with
discordant counts, the provider-error count on both arms, and the run-to-run spread if repeated arms exist.
A result without its slice composition is not interpretable — that is the lesson of M11–M15 in one line.

## 6. Consequence for the roadmap
M7b runs on the enriched slice, with its own matched control. If the +23.5-point abstention lift is real,
the enriched slice will show it at p<0.001 instead of p=0.22 — and if it is not real, we will know that too,
which the 60q slice could never have told us either way.
