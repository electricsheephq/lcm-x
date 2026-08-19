# FINDING F50 — #436 maintainer-remediation sanity slice: DELIVERY-NEUTRAL (endorsed)

Date: 2026-08-02. Owner-ordered (symmetric-measurement standard — the maintainer's 435-line
remediation fa00ec9 touching the certified-answer path gets the same paired instrument our own
fix batches get; no author exemption).

## 1. Design
Paired 60-question official web-batch1 slice on the frozen V2 instrument (official_unit_runner,
qwen3.5-9b reader / gpt-5.2 evaluator, FROZEN-PROTOCOL captured per arm): **pre = a2f519a**
(upstream-wave-1 before the maintainer's push) vs **post = fa00ec9** (the five-finding
remediation). Identical runtime inputs (questions/haystack/memory_config byte-shared).

## 2. Verdict — the strongest clean available
- **memory_context is BIT-IDENTICAL on 60/60 questions across arms** (0 content diffs, 0
  token-count diffs): the remediation did not change retrieval or delivered context AT ALL on
  this slice. Delivery-neutral, proven bitwise, not inferred from score.
- Scores: pre 19/60 (31.7%) vs post 22/60 (36.7%), b=6 / c=3, net **+3** (bar: net ≥ −3), 9/60
  discordant (15%) — and **all 9 flips have identical memory_context**, i.e. the entire score
  delta is reader sampling variance (temp 0.6), consistent with this instrument's known
  answer-variance band. No improvement claim is made from the +3.
- Fail-close: 0 instrument failures, both arms. No arm-death.

## 3. Ops record (disclosed)
- Post-arm was frozen mid-generation (SIGSTOP, ~46/60) when the OpenRouter account hit zero —
  deliberate, to keep the gpt-5.2 evaluator phase from 402-contaminating the paired verdict.
  Thawed after top-up; resumed cleanly; freeze/thaw left no artifacts (0 fail-closes).
- Wrapper defect (ours, disclosed): run_slice.py's final step invoked failclose.py --compare,
  which expects the memorybench report layout (`report.json`); the V2 official harness writes
  `aggregated_metrics.json` + `per_question.jsonl`. The verdict was computed directly from both
  arms' per_question.jsonl (paired b/c + context-identity checks); slice-paired.json carries
  the machine-readable result. failclose.py gains a V2-layout mode only if a future slice needs
  it — not retrofitted now.

## 4. Disposition
**fa00ec9 ENDORSED with instrument receipts** (upgrading the earlier source-review-only
validation the owner correctly flagged as insufficient). #436 proceeds to the maintainer's
merge window. The symmetric-measurement rule worked exactly as intended: the endorsement now
rests on a paired measurement, and the measurement happened to be maximally clean.

## 5. Artifacts
session-notes 2026-08-02 `436-sanity-slice/`: slice-paired.json (verdict), pre/run + post/run
(frozen; per_question.jsonl, aggregated_metrics.json, FROZEN-PROTOCOL.json each), run_slice.py.
