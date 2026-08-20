# FINDING F60 — LoCoMo C1 (FTS-ON) verdict: FAIL-inside-noise; declared config remains F48's

Registered: RUN-SHEET-LOCOMO-C1-FTS-ON (§3 bands, pre-declared; no post-hoc bands). Run root:
session-notes/2026-08-19/hermes-locomo-c1/artifacts/paid-aa-20260819T155735Z/ (arms a, a-prime;
pins-locomo-c1.yaml with amendments 1–4). Recomputed from per-question raw (1,986 rows/arm), not
the runs' own aggregates.

## 1. Verdict
**FAIL (inside noise).** A = 54.03% (1073/1986), A′ = 54.38% (1080/1986) vs the F48 baseline
54.6%: −0.57pt / −0.22pt — below the +0.76pt noise line, nowhere near the ≥56.1% PASS bar.
No category collapse (§2). Per the registered band: **C1 is not adopted; the LoCoMo declared
config remains F48's; C2 baselines on the F48 config.** FTS-ON delivery gains did not convert
to answer-quality gains at this configuration.

## 2. Full resolution (A / A′ vs F48 §3-CORRECTION; floor = −4.0pt)
| category | F48 | C1-A | C1-A′ | Δ(A) | floor check |
|---|---|---|---|---|---|
| adversarial | 28.7 | 30.3 | 30.5 | +1.6 | ✓ |
| multi-hop | 64.8 | 65.7 | 66.7 | +0.9 | ✓ |
| single-hop | 37.2 | 35.8 | 35.8 | −1.4 | ✓ |
| temporal | 41.7 | 42.7 | 43.8 | +1.0 | ✓ |
| world-knowledge | 69.7 | 69.6 | 69.8 | −0.1 | ✓ |
A/A′: spread 0.35pt aggregate; per-question discordance 71/1986 (3.58%) — consistent with the
F48-era churn class (3.47%). Provenance both arms: frozen codex-cli 0.148.0, gpt-5.6-sol
answer(medium)/judge(low), modelExplicit, isolated; 1,986/1,986 rows each, zero missing.

## 3. Why delivery gains didn't convert (registered-hypothesis disposition)
The fusion-sim predicted single-hop/multi-hop/world gains; measured deltas are ≤ +1.6pt with
single-hop mildly NEGATIVE — the retrieval-delivery improvements FTS-ON provides on paper are
already answerable from the vector arm's context at this corpus/config, and the fts arm's
literal-match wins land on rows the answerer already gets right (or wrong for reasons upstream
of retrieval). This matches the F58 mechanism finding (solo-FTS literal matches vs semantic
golds) and folds the FTS-ON question into the **joint FTS program** (#173 scan-budget cap +
F58 fusion-arbitration lever + this null) as ONE design decision — separately registered when
taken up.

## 4. Disclosures (fail-closed record)
Run integrity incidents, all diagnosed + amended (pins amendments 1–4), none touching scoring:
(1) arm-A search-phase warmup network hang → HF_HUB_OFFLINE=1 amendment (#235); (2) 13-hour
inter-arm stall — bun orchestrator lingered after arm-A success (5 leaked bridge children;
#236 success-path variant); (3) A′ search-phase initialize timeout — 3-way concurrent COLD
fastembed model load ~7 min vs the 300s client timeout sized from a warm single-process probe
(INITIALIZE_TIMEOUT_MS 300s→1200s; #235 third variant); relaunches were themselves gated twice
by design (paid-run guard; pinverify sha mismatch after the timeout patch — both guards
worked, both passes recorded). Transport frozen through all of it. The 99 corrupted-gold rows
run as-is per F48 comparability (ceiling ≈95%).
