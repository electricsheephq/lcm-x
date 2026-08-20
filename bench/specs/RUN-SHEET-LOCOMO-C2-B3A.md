# RUN SHEET — LoCoMo C2: B3-A attribution ingestion (registered A/A′)

Status: REGISTERED (pins finalize at launch from commands). Baseline of record: **F48's declared
config** per FINDING-F60 (C1 FTS-ON = FAIL-inside-noise; C2 does NOT stack on C1). Premise:
FINDING-B3-PC — in 85.3% of F48's adversarial misses the gold evidence was DELIVERED but the
attribution half was never ingested; B3-A ingests speaker attribution (HERMES_MB_SPEAKER_PREFIX=1;
SPEC-B3-ATTRIBUTION.md; premise-check sanity anchor reproduced F48's 32.7% adversarial exactly).

## 1. Config under test
F48 declared config + B3-A attribution ingestion ONLY (one delta; env HERMES_MB_SPEAKER_PREFIX=1
recorded in the env inventory per the F49 §5 standing rule). Same corpus, same 99 corrupted-gold
disclosure (ceiling ≈95%), same paid-aa machinery as C1 (fresh stores, sequential arms A then A′,
frozen codex transport for answer/judge, HF_HUB_OFFLINE=1, INITIALIZE_TIMEOUT_MS=1200000 — C1
pins amendments 1-4 inherited as the baseline environment). Spend lane: the established LoCoMo
paid A/A′ category (C1-class magnitude).

## 2. Pre-declared verdict bands (aggregate + category; no post-hoc bands)
- **PASS (adopt B3-A):** adversarial ≥ +4.0pt over F48 §3-CORRECTION's 28.7 (i.e. ≥32.7 — wait:
  28.7 is the FLOOR value; the F48 banked adversarial accuracy is 32.7) — PRECISELY: adversarial
  ≥ 36.7 (= 32.7 + 4.0, clearing the C1-era churn class with margin) AND aggregate ≥ 54.6 − 0.76
  (no net loss beyond noise) AND no other category −4.0pt below its F48 §3-CORRECTION value.
- **GRAY:** adversarial +0.76..+4.0pt → disposition in writing; default ADOPT only if the E+R−
  pool (the 85.3% mechanism rows) shows the gains concentrated there (mechanism-consistent).
- **FAIL:** adversarial inside noise (<+0.76) or any category collapse >4.0pt.
- A/A′ agreement reported per F48 practice (spread + per-question discordance over 1,986×2).
Verdict statistic recomputed from per-question raw, never the run's own aggregate.

## 3. Diagnostics (recorded, not scored) — Amendment-14 lessons applied at registration
- Attribution-ingestion receipt: the ingest log/store must show speaker prefixes present for >0
  rows BEFORE arm A's search phase (a registered diagnostic that never fires = a finding; assert
  it fired, fail-closed pre-spend).
- E+R− pool tracking: per-row deltas on the 85.3% mechanism pool published regardless of verdict.
- System-prompt/protocol pin: the run inherits C1's environment INCLUDING any SOUL/system-prompt
  state; all such state is pinned by sha at launch (Amendment-14 rule — protocol members join
  the pin set, no "identical protocol" claims without receipts).

## 4. Landing
Verdict = next-free F-number; scoreboard row only on PASS; GRAY/FAIL publish at full resolution.
Bench PR flow with the scripted merge barrier. Post-verdict: B3-B (residual lever) decision.
