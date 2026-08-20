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

## 7. APPENDIX (2026-08-20 UTC, append-only) — R4-ref disposition: unmeasurable on exec for gpt-5.4; quota-window note void

**Account/quota correction:** the "Aug 26–31 window" constraint in §5 is VOID — the exhausted
bucket was simply a low account the owner swapped same-day; gpt-5.4 was live immediately.

**R4-ref: three attempts, zero valid runs — DECLARED UNMEASURABLE as a context-retention
reference on the exec transport:**
1. Attempt 1 VOID: `exec resume` carried no `-m`; 69/70 turns silently served the default
   gpt-5.6-sol (fixed with a fail-loud per-turn served-model assertion, PRs #308/#311 — the
   guard then correctly caught its own author's parser bug on attempt 2's first turn).
2. Attempt 3 ran clean mechanically (70 turns, model receipt 285× gpt-5.4, zero compactions —
   the §3b long-context expectation held) but **tripped the Amendment-10 detection gate**:
   16 exec tool calls during probe turns, `rg`-ing codex's own memories directory and the
   session's OWN LIVE ROLLOUT FILE for the answers, with literal canary values in the search
   patterns. Answers derived from disk are not context retention; the pre-declared rule
   invalidates the arm.
3. **Behavioral observation banked in lieu of the row:** on identical material and probes,
   gpt-5.6-sol (R1-lc) made ZERO tool calls and answered from context; gpt-5.4 reached for
   filesystem tools on every attempt. Since codex exec offers no tool off-switch and the
   model's own home (memories, rollouts) cannot be hidden from it, a tools-clean gpt-5.4
   exec reference is not obtainable — and the model retires from codex-auth 2026-08-31.
   The pilot's contrasts never depended on R4 (§3b excluded it from all contrasts).

Extension arms (sol control, luna pair, multi-compaction stress) are registered as run-sheet
Amendment 13.

## 8. APPENDIX 2 (2026-08-20 UTC, append-only) — Amendment-13 extension arms: sol control, luna pair, multi-compaction stress

| arm | model | retention | traps | notes |
|---|---|---|---|---|
| R2s-S-ctrl | sol | **100%** (30/30) | 4/5[^ctrl-trap] | same-day control: NO serving drift; sol single-session now 3/3 at 100% |
| R2s-L-A | luna | 90.0% (27/30) | 5/5 | 3 misses ALL honest ABSTAIN, all E0 (deepest epoch) |
| R2s-L-A′ | luna | **100%** (30/30) | 5/5 | all three E0 flips recovered — luna band 90–100, 3/35 discordant |
| R2s-MC | sol, threshold 0.12 | 93.3% (28/30) | 5/5 | **UNDEREXPOSED** per the §13 validity gate (0 compaction events, see below); 2 misses honest ABSTAIN, 0 hallucinations at a 32.6K live window |

[^ctrl-trap]: R2s-S-ctrl's 4/5 is a regex-missed abstention (frozen set honored, disclosed).

**Luna verdict (pre-declared read, band-disciplined):** the sol-vs-luna retention difference
(0 or 10 points depending on the luna arm) never exceeds luna's OWN repeated-run spread
(10 points; discordance 3/30 canaries = 10%, traps excluded per §5), so **no model
recommendation is licensed by these data** — sol's 3×100% vs luna's one 3-miss run in two is
SUGGESTIVE of a depth-of-retrieval difference and nothing more; separating it needs more
repetitions (P1). What IS established: luna's failure MODE — every miss was an honest
E0 abstention, zero hallucinations, traps 5/5 both arms — so the cost tier fails loud, not
wrong. Served-model receipts from the host record both arms.

**Multi-compaction stress — the question was ill-posed for this architecture (measured):**
lowering compression.threshold 0.5→0.12 (verified effective pre-spend: 32,640 tokens) produced
ZERO compactions, not more. Re-examining the banked arms with timestamps: R2s-A's single
compaction fired ~80 SECONDS into the run (source = material turns 1-2, ~30.6K tokens) — **on
this path LCM compaction never recurred and never scaled in ANY tested configuration —
once-early at threshold 0.5 (×3 runs), zero events at 0.12 — so compaction count is not
controllable via compression.threshold, and the trigger semantics are unexplained (#314).
Everything after the early window is store retrieval.** §13's regime-under-pressure caveat was therefore the right instinct but
understated: compaction COUNT is not a controllable variable on the ACP path via
compression.threshold, and "retention after a few compactions" reduces, for LCM, to "retention
under retrieval at depth" — which the MC record measures anyway (93.3% at a 32.6K live window,
honest failures only). Native-side multi-compaction remains the P1 interactive-transport
question. The threshold-vs-compaction-count semantics are filed as a product question (see
issue reference in the PR); the R2s-MC arm is excluded from any count claim per the
pre-declared gate.

**Session-hygiene note:** the MC pre-spend gate's warm/status one-shots added 2 extra host
sessions to the home before the ACP run (benign — no canary content; disclosed).


## 9. CORRECTION (2026-08-20 UTC, append-only — adversarial-review round; run-sheet Amendment 14 carries the receipts)

1. **Mechanism attribution corrected.** §3.2/§6 attributed retention to "LCM's store-level
   cross-session assembly / memory-tool recall path". The registered lcm_* diagnostic fired
   0/350 across all arms; recall was carried by the HOST memory surface (session_search,
   memory) operating over the run's own history, with the LCM engine as context assembler —
   an attribution Amendment 9 explicitly forbids collapsing. The REGIME-level results and the
   recommendation are unchanged (arms measured end-to-end systems; R2s 100% ×3 vs R2b 86.7
   still separates the setups); the internal mechanism story as previously written was wrong,
   and the R2s-vs-R2b gap is attributable to the LCM ENGINE (assembly/limits), not to lcm_*
   retrieval tools, which never engaged.
2. **Luna failure-mode claim downgraded to prompt-conditioned.** The luna arms (and the sol
   control) carried a system-prompt abstain instruction the banked sol pair lacked (Amendment
   14 §1). "The cost tier fails loud, not wrong" is therefore unestablished as a MODEL
   property; the banked statement becomes: luna's misses were abstentions UNDER A PROMPT THAT
   INSTRUCTED ABSTENTION, and miss-mode attribution across the pilot is UNATTRIBUTABLE
   (prompt, regime, and date are collinear). Retention numbers unchanged. Trap metrics
   unaffected by the soul split (Fisher ≈ 1.0).
3. **Restart-bridge miss pattern corrected.** "ALL R2r/R2b misses concentrate in the
   confusable-naming class" and "every substitution was noun+epoch-matched" are both false at
   the row level: R2r's 5 misses all fit that pattern; R2b's do not — 3/4 fit, one is a
   NON-CANARY SPLICE (saffron-zephyr-b20c = fragments of two canaries across class AND epoch),
   and one is an intra-class cross-epoch entity mis-bind (C1-E0-1 answered C1-E2-1's value).
   "Store assembly binds entity and recency reliably" is thereby weakened to a tendency with
   two counter-examples; the P1 lever scope widens from attribute-type disambiguation to
   binding fidelity generally.
4. **Engine-failure disclosure**: every ACP arm ran with 59 (R3: 24; MC: 78) failed summary
   publications (publication_invariant_conflict — plausibly lcm-x#247 live); the "(ACP; 1
   compaction)" characterization was incomplete. Rows stand as measurements of the shipped
   system; the product defect is now on the record and joins #314's investigation.
5. **Numeric relabels**: R1 last-request pin → 554,094 (553,914 was request 64/70; parity-gate
   conclusion unchanged). R2s-MC "32.6K live window" → 32,640 is the TRIGGER; the protected
   tail reached ~123K, same as the 0.5 arms — threshold does not shrink the live window.
