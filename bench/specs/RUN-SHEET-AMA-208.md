# RUN SHEET — AMA-Bench full 208-episode run (registered; GO from pilot arithmetic)

Date: 2026-08-02. G0 channel: coverage row (Tier-F benchmark, first full AMA number).
GOALS target line: ≥72.26%. Budget decision (S3): **GO** — pilot-measured projection ≈
12.85M in / 3.02M out answerer + 1.27M/8k judge ⇒ ≈ $5.7 at policy prices; pilot itself cost
~$0.60; credits $19.33 ≥ $8 guard. ~10.9 serial hours projected.

## 1. Config (SOTA policy; pilot-proven plumbing 72/72 + 72/72)
- Answerer: `openai/gpt-5.6-luna` via OpenRouter, harness-default sampling. DISCLOSED pilot
  finding: AMA's custom-provider path exposes NO reasoning-effort surface — effort is the
  model default, recorded as such (not "medium").
- Judge: `anthropic/claude-sonnet-5` via OpenRouter (cross-family). Judge-fail-closed patch
  applied (unparseable verdict raises; 72/72 parsed in pilot).
- Pins: clone ddfd319e + our two patches (apply-checked, applied); dataset
  open_end_qa_set.jsonl sha256 45c36052e…; overlay wt-ama-adapter (README subclass shim);
  fastembed cache reused read-only from the pilot.
- Episode 89 (1.03M tokens) fits luna's 1.05M context untruncated (pilot-verified); the
  truncation-retry patch stays applied as a guard but is expected unexercised.

## 2. Execution
- Fresh output dir (hermes-ama-full208/) — pilot scores are timing-pilot data, NOT rows; the
  registered number comes only from this clean full pass over all 208 episodes (serial).
- Per-episode exit codes + telemetry + timing captured as in the pilot; a failed episode is
  recorded and the run continues (fail-closed accounting: failures count in the denominator).
- Machine-slot note: V1-M shards launch during this run. AMA is API-dominant; during the six
  V1-M shard-init windows the AMA runner is SIGSTOP'd, then SIGCONT'd (freeze-writers
  protocol) to keep bridge inits contention-free.

## 3. Pre-declared reporting (seven-point row)
- Headline: judged accuracy over all 208 (fail-closed: unparseable/errored episodes disclosed
  in the denominator, never dropped).
- Noise floor: A/A′ on a 30-episode fixed-seed subset (seed 20260802) AFTER the full pass,
  same config; discordance + spread disclosed like F48 before any delta narrative.
- Per-tier breakdown (median <100k / 100k–800k / >800k source tokens: 184/21/3).
- Negative result ships at the same resolution as positive.

## 4. Abort/park
- OpenRouter 402 → runner parks resumable (per-episode append-only; no mid-run contamination
  class — each episode scores independently).
- >5 consecutive episode failures of the same cause → park, root-cause first.
