# RUN SHEET (registered; launch gated on AMA-pilot plumbing green) — V2 static 451 SOTA row

Date: 2026-08-02. G0 channel: coverage row (first V2 row under the SOTA-models policy).
This is a NEW declared config — it does not supersede the official-config rows (123/451 F47
et al.), which stay visible; the scoreboard discloses both configs side by side.

## 1. Config (SOTA policy of record, strategy addendum 2026-08-02)
- Reader: `openai/gpt-5.6-luna` @ medium reasoning effort, via OpenRouter. Sampling: harness
  defaults for the luna path, RECORDED in pins (the official config's qwen sampling knobs are
  qwen-specific; whatever the harness applies to luna is captured, not assumed).
- Evaluator/judge: `anthropic/claude-sonnet-5` via OpenRouter (cross-family: Claude judges a
  GPT answerer). Judge prompts: the SAME pinned evaluator prompts as the official config —
  ONLY the models change; rubric drift would confound the config comparison.
- Memory system: fork main @ 9d181aa product train (the settled #436-endorsed lineage);
  harness = official_unit_runner + evaluation.harness, same as F47/slice.
- Everything else identical to the F47 instrument (static 451, same batches, fail-close
  accounting, FROZEN-PROTOCOL per run).

## 2. Why now
- fa00ec9 endorsed delivery-neutral (F50) → the train is settled.
- Cost collapsed under the policy: luna at $0.10/M in ≈ single-digit dollars for 451 questions
  (vs the reader's prior cost class). Judge legs at sonnet-5 $2/$10 similar order.
- The AMA pilot (running) exercises the exact luna+sonnet-5 OpenRouter stack first — launch
  here only after the pilot proves the plumbing (judge parse fidelity, effort passthrough).

## 3. Pre-declared reporting (seven-point row)
- Headline: scored read over 451, fail-closed accounting; b/c vs the official-config 123/451
  reported as CONTEXT (different reader — not a paired claim; the comparison row is labeled
  cross-config, reader-dominated).
- A/A′: a 100-question fixed-seed subset pair (seed 20260802) for the luna-config noise floor
  BEFORE any narrative about deltas; disclosed on the row like F48.
- Expectation, stated honestly: most of any gain over 123/451 will be READER capability, not
  memory. The row's purpose is the SOTA-anchored funnel (what frontier agents actually get
  from our memory), per the owner's stale-world thesis — not a memory-delta claim. Memory
  deltas remain paired-instrument territory (same reader both arms).

## 4. Budget guard
Pre-launch: check OpenRouter remaining ≥ $12 (rough ceiling: 451q × ~60k in avg × $0.10/M ≈
$2.7 reader-in + outputs + judge legs + A′ subset ≈ $6-10). If remaining < $12 after the AMA
pilot, park and surface — never start a scored run that can 402 mid-flight (the slice-freeze
lesson).

## AMENDMENT 1 (2026-08-02, pre-launch — probe-measured budget guard)
3-question probe EXECUTED (probe-run-4): 3/3 scored, 0 instrument failures, sonnet-5 evaluator
parsed via chat-completions transport, luna reader clean. Variant runner
(official_unit_runner_sota.py: models swapped + the frozen 60-batch count assertions relaxed —
both patches documented in-file; the frozen original untouched). Probe-measured cost projects
the full 451 at ≈$5–8. §4's guard is amended BEFORE launch from the pre-probe "remaining ≥ $12"
to **remaining ≥ measured-projection + 30% ≈ $10.40**. Same intent (no mid-run 402), better
arithmetic. Launch decision applies the amended guard at the post-AMA credits reading.
