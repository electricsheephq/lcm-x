# R3 release notes — OUTLINE v2 (2026-08-02; assemble when S1 slice verdict + V1-M row land)

Headline: **the metric-standard release** — we publish how agent memory must be measured, with
our own numbers (good and bad) as the reference implementation.

1. The disclosure standard (DISCLOSURE-STANDARD.md) — the proposed practice.
2. The scoreboard — disclosure-first, append-only, fail-closed generator; live rows incl. V1
   455/500 + latency, V2 re-baseline 123/451 (F47 gate PASS, blind-adjudicated), the F44 scale
   curve, LoCoMo 47% (F46 decomposition) → superseded-but-visible under the declared-config
   54.6% row (F48, A/A′ noise floor 3.47%).
3. The honesty narrative (HONESTY-NARRATIVE.md, four episodes): F45 (a pre-registered gate
   KILLED our own feature) · F42→F44 (six gate executions, five root causes, confirmation-run
   rule) · F47 (blind adjudication + EINTR append-only continuation) · **F46→F48 label-scramble
   correction (published wrong, instrument-caught in hours, corrected append-only on every
   surface — and the recomputation surfaced the REAL finding nobody was looking for).**
4. The two-tier doctrine: Tier-F frontier fault-finding vs Tier-P published claims — why both,
   and the "stale-world measurement" argument (market moved; undated configs are unanchored).
5. LoCoMo arc as the worked example, now with the full mechanism story:
   47% → instrument bugs + corrupted gold + config-class gap → declared config 54.6% honest →
   **F49: a shipped retrieval feature was OFF at runtime (env-wiring gap; "code shipped ≠
   feature on"; new standing rule: pins must diff the run env against the product's full
   feature-flag inventory)** → zero-spend replay chain (22/41 gold recovery; fusion ratio
   re-derived 1:1 by a-priori rule) → **C1 FTS-ON registered with pre-declared bands** →
   **B3-PC: 85.3% of adversarial misses had the evidence delivered — and the store carries no
   speaker attribution at all (B3-A: attribution-preserving ingestion, C2)**. The arc
   demonstrates the method: every step zero-spend-first, gates before spend, causal deltas
   isolated (C1 vs C2).
6. V1-MEDIUM first-mover row: instrument PR #198 merged (11 review-bot passes, 45 threads
   fixed-or-refuted on the record — the review loop itself is disclosure-grade evidence);
   registered 6-shard run from measured smoke timings; retrieval row first, reader row as a
   separate registration.
7. Upstream record: #436 (R2 train, review record, maintainer engagement, owner-ordered paired
   sanity slice on the maintainer's remediation — symmetric measurement standard, no author
   exemption), harness reports #6/#7, memorybench instrument PRs #3/#5, adapters #4/#6.

PENDING INPUTS: S1 slice verdict (frozen mid-run on the credit outage; resumes on top-up) ·
V1-M registered-run row (launches on smoke completion) · AMA pilot timings (parked, resumable)
· owner branding for the public scoreboard surface (OWNER-ONLY).
