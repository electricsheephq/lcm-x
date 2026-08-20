# RUN SHEET — Compaction-Quality Pilot (P0): native Codex vs LCM vs compressor vs handoff

Status: DRAFT-FOR-REGISTRATION — pins finalized at launch from smoke outputs (S0–S3);
this sheet lands on main BEFORE the six-run spend. Owner mandate 2026-08-20 (ultracode):
the gpt-5.4→5.6 migration recommendation must rest on measurement; native codex
compaction is a black box; every run doubles as reverse-engineering material.

## 1. Question + regimes
Which regime should carry LONG coding-agent runs on gpt-5.6 after the 08-31 gpt-5.4
codex-auth retirement?
- **R1** Codex CLI native compaction (server-side remote_compaction_v2; black box —
  behavioral probes only; structural observables: window_number counter,
  replacement_history composition, ~99s measured stall per compaction, token edges).
- **R2** Hermes host + LCM context engine (`context.engine: lcm`), provider openai-codex.
- **R2b** Hermes host + built-in compressor (`context.engine: compressor`) — the
  intra-host control; ONE config flip from R2.
- **R3** R2 material phase → session exit → FRESH hermes session on the same LCM home →
  probes only (the cross-session handoff native compaction cannot do).
Model: gpt-5.6-sol only in P0 (luna → P1). R4 (long-context >272K) deferred to P1.

## 2. Material + probes (D1/D2 of the approved design)
- One FIXED synthetic coding-session script: 35 turns × ~17K tokens of varied,
  non-repetitive filler + a one-line bounded task per turn; identical across all arms.
  Sized to cross the native trigger band (86–93% of 258,400) ≥2 times.
- 30 canaries (5 info classes: decisions / file-paths / constraints / tool-output facts /
  user preferences; ×2 per class per epoch; 3 epochs by token depth) + 5 hallucination
  traps. Values salted per run (contamination alarm). Probes never contain the value.
- Post-hoc epoch re-binning against MEASURED compaction boundaries; canaries within ±1
  turn of a boundary excluded + disclosed.
- Scoring: mechanical 3-way per probe — CORRECT (normalized substring) / ABSTAIN
  (declared regex list) / HALLUCINATE (any other concrete answer; unparseable lands here,
  disclosed). Negative probes score abstention as correct. No LLM judge in P0.

## 3. Runs + order (D4)
S0 cost/auth smoke → S1 resume-continuity smoke → S2 pty 16K-turn smoke → S3 denominator
gate → then: **R2-A, R2-A′, R1, R2b, R3, R1-on-gpt-5.4 (perishable, before 08-31)**.
Fresh HERMES_HOME/LCM store per run; per-turn append-only results JSONL (the V1M
Amendment-2 lesson); ~2h wall per run, staggered.

## 4. Denominator gate (fail-closed, pre-spend)
R1 asserts model_context_window == 258,400 from token_count telemetry. R2/R2b capture the
effective window via `lcm_status` mid-session and reconcile against **lcm-x#263** (the
codex_routing 372K-vs-272K substring-match bug, filed with receipts): arms proceed only
after parity is verified or the inequality is DECLARED here with rationale. The #263 fix
itself is a separate product PR, never bundled into the pilot.

## 5. Pre-declared success criteria (the pilot proves MEASUREMENT, not a winner)
- **G1 integrity**: all three R1 markers parse on ≥2 boundaries; ≥90% of probes
  mechanically classifiable; ≤20% of canaries epoch-ambiguous.
- **G2 noise**: R2 A/A′ retention discordance published as THE band; >25pt overall →
  instrument too noisy, redesign before P1.
- **G3 sensitivity**: ≥1 pre-registered contrast (R2 vs R2b on overall retention, or
  E0-epoch retention R1 vs R2) separates beyond the G2 band; else verdict "underpowered"
  (valid P0 outcome; blocks P1).
- **G4 denominators**: verified or declared per arm before main-run spend.
P0 SCOPE CUTS (explicit): m2 continuation-task correctness not measured (trivial per-turn
task); R1's own noise floor unmeasured — R1 contrasts read against the R2 band + caveat.

## 6. Spend + accounting
Expected marginal $0 (ChatGPT OAuth; S0 verifies, not assumes). Real budget = OAuth quota
+ wall-clock. OpenRouter untouched. Abort criteria: any arm failing its config assertion
(engine/model/sha) aborts THAT run, reported with partial data, never silently rerun.

## 7. Landing + ownership (D5)
Code: bench/instruments/compaction_probe/{gen_material,drive_codex,drive_hermes,
parse_rollout,score_probes,report_pilot}.py — codex-dispatched against this sheet;
architect writes canary/probe content, scoring rules, bands. Zero product code. Verdict:
next-free FINDING number at write time. Publication of any per-model comparison = P1,
owner content sign-off required.

## 8. Pins (every value from a command at launch; placeholders until smokes complete)
- codex binary: release path + sha256 (S0 output) · cli_version asserted from session_meta
- hermes version + config.yaml sha256 per arm · lcm-x sha (engine under test)
- material/probe script shas + run seed · effective windows per arm (S3 output)
