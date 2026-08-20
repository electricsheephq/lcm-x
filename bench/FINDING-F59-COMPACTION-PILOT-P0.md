# FINDING F59 — Compaction-quality pilot P0: the 5.4-replacement regime measurement

Registered: RUN-SHEET-COMPACTION-PILOT + Amendments 9–12. Pins: specs/PINS-COMPACTION-PILOT-FINAL.md.
Question (owner, 2026-08-20): after gpt-5.4 leaves codex-auth (08-31), which regime should carry long
coding-agent runs on gpt-5.6 — native codex compaction, hermes+LCM, or long context?

## 1. Scoreboard (30 canaries, 5 classes × 3 epochs; mechanical scoring, frozen scorer 77b62dde)
| arm | regime | retention | traps | A/A′ band | tokens moved |
|---|---|---|---|---|---|
| R1-lc | codex exec long-context (0 compactions) | 100% | 5/5 | unmeasured (§5 cut) | 29.5M (98% cached) |
| R2s-A/A′ | LCM single-session (ACP; 1 compaction) | 100% / 100% | 5/5, 4/5* | **0.0pt** (0/30 discordant) | ~8.9M |
| R3 | LCM handoff (compacted store → FRESH session probes) | 100% | 5/5 | single run | — |
| R2r-A/A′ | LCM per-turn restart bridge (one-shot) | 86.7% / 96.7% | 5/5 | **10.0pt** (3/35, all C1) | ~3.0–3.6M |
| R2b | compressor per-turn restart (control) | 86.7% | 4/5 | single run | ~3M |
*disclosed frozen-regex miss ("can't determine"), Amendment 9 freeze honored.

## 2. Gates (pre-declared §5)
G1 PASS (0 unparseable across 7 valid arms; R1 markers parsed; 0 epoch-ambiguous canaries — no
compaction boundaries crossed in R2r/R2b, one clean early boundary in R2s/R3). G2 PASS (bands
published above; both < 25pt). **G3 PASS — the pilot is NOT underpowered**: equal-window contrasts
R2s-vs-R2r (+13.3pt on the A pair vs the 10pt band) and R2s-vs-R2b (+13.3pt) separate beyond the
band. R1-lc is EXCLUDED from contrasts by the §4 window-parity gate (served 554K vs 272K) and stands
as a reference row. G4 declared per §4 + Amendment 11.

## 3. Mechanism findings (each verified against raw artifacts)
1. **Headless codex does not compact.** codex exec+resume (0.148.0, OAuth) advertised
   model_context_window=258,400 yet served 553,914 tokens in-context with ZERO compaction markers;
   rollout labels the mechanism `history_mode: "paginated"`. E0 10/10 falsifies silent truncation.
   Compaction is TRANSPORT-DEPENDENT (live/interactive sessions trigger at 86–93% of window; exec
   chains do not) → the "native compaction quality" question is unanswerable on the exec transport
   because the machinery never engages; it burns long-context instead.
2. **LCM single-session = long-context retention at ~1/3 the tokens moved.** R2s matched R1-lc's
   100% (0.0pt A/A′) with one 136K-policy compaction + active memory-tool recall, moving ~8.9M
   tokens vs 29.5M. R1-lc's run also exhausted the account's OAuth quota window in 9 minutes —
   the long-context regime's practical cost is quota, not just latency.
3. **The handoff regime works at full marks.** R3 (fresh session over the compacted store) scored
   100% incl. the confusable class — the capability native compaction structurally lacks (session
   death = context death). This plus (2) is the product story.
4. **The restart-bridge is cheapest and lossy in ONE specific way**: all R2r/R2b misses concentrate
   in the confusable-naming class, and every substitution was noun+epoch-MATCHED but
   attribute-type-confused (C1 "what did we name X" answered with C5 "what prefix do X-names use").
   Store assembly binds entity and recency reliably; decision-type binding is the weak link, and it
   is stochastic (A′ recovered 3 of A's 4 misses). Lever candidate for P1: attribute-type
   disambiguation at assembly.
5. **Host-level memory ≈ LCM on aggregate for the restart-bridge topology** (R2b 86.7 = R2r-A 86.7,
   overlapping miss sets), per the Amendment-9 one-sided rule: no mechanism attribution claimed;
   LCM's differentiation shows in the single-session/handoff regimes, trap discipline, and the
   C1-recovery band, not in restart-bridge aggregate.

## 4. Recommendation (applies the pre-declared decision rule)
For long coding-agent runs after 5.4: **hermes+LCM single-session (R2s regime) on gpt-5.6-sol** —
retention parity with long-context at ~1/3 the token movement and no quota cliff, with the handoff
regime (R3) covering session restarts, which native compaction cannot. Native codex exec remains
fine for SHORT runs; at 600K-token depth it silently converts to long-context and consumed a full
OAuth quota window in one run. Long-context-only (R4-class) is the ceiling reference, not a regime
recommendation. Publication of any per-model comparison = P1, owner content sign-off required.

## 5. Disclosures (fail-closed record)
Instrument round-trips before the first valid run: pty truncation (2 attempts, abandoned), one-shot
text-vs-index defect, probe-collision fix (#274), scorer schema binding + abstain-regex expansion
BEFORE official scoring then FROZEN (both R2r-A trap readings on record: 1/5 original set, 5/5
expanded), concrete-answer guard, drive_codex stdin stall (17 min, 0 turns consumed) + missing turn
timeout (#291), 3rd wire-contract catch (kind-less codex rows scored 0/30 pre-#293). Contamination:
agentic filesystem tools are a live vector (Amendment 10 audit: ZERO tainted-correct rows in banked
arms; trap rows in 4 turn-sessions flagged design-read-informed); a #275 test fixture leaked one real
value into the repo (purged #289); ACP arms could not be tool-restricted mechanically (product gap
#301) — R3 att.1 invalidated (9 values into a threaded probe session), att.2 killed on an engine-pin
drift caught by console receipts (all six banked arms verified tree=91d5706f), att.3 passed the
Amendment-12 audit gate (memory-tools only) and is the counting run. R4-ref pending the Aug 26–31
OAuth window. m2 (continuation-task correctness) not measured (§5 scope cut). Trap metric =
secondary diagnostic throughout (Amendment 9).

## 6. m4 (owner addendum: "study what native compaction kept")
For the exec transport there is nothing to study — native compaction never produced a summary; the
observable is paginated long-context. The reverse-engineering opportunity moves to interactive-
transport captures (P1), where the encrypted server-side summary is behaviorally probeable via the
same canary battery. LCM-side: the single compaction event in R2s/R3 (30.6K tokens → 236-token
summary) plus 100% downstream retention shows the memory-tool recall path, not the summary alone,
carries retention — a design validation for compact-aggressively + recall-on-demand.
