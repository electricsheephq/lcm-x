# Public results

OmniMemEval: An open evaluation harness that requests the same data, prompts, answer model, judge model and metric logic for every backend within each benchmark (the disclosure below states how the LCM-X rows requested the models).
The LCM-X maintainers ran these rows on the unchanged upstream pipeline and prompts.
Claim class: **contributor-run, accepted upstream**.
Placements count only listed backends with a score in the pinned upstream summary; higher scores rank first.
The upstream page does not claim an optimal configuration for every backend.

| Benchmark | Metric | LCM-X value | Placement | Links |
| --- | --- | ---: | --- | --- |
| LoCoMo | LLM-as-a-judge accuracy | 78.83 | 5th of 15 | [Source](https://github.com/MemTensor/OmniMemEval/blob/f7476c55/docs/user_memory/results.md#result-summary) · [Evidence](https://github.com/electricsheephq/OmniMemEval/releases/tag/lcmx-v0.28.0-results-20261018) |
| LongMemEval | LLM-as-a-judge accuracy | 87.00 | 2nd of 13 | [Source](https://github.com/MemTensor/OmniMemEval/blob/f7476c55/docs/user_memory/results.md#result-summary) · [Evidence](https://github.com/electricsheephq/OmniMemEval/releases/tag/lcmx-v0.28.0-results-20261018) |
| BEAM 100K | nugget score | 71.84 | 1st of 12 | [Source](https://github.com/MemTensor/OmniMemEval/blob/f7476c55/docs/user_memory/results.md#result-summary) · [Evidence](https://github.com/electricsheephq/OmniMemEval/releases/tag/lcmx-v0.28.0-results-20261018) |
| PersonaMem v2 | rule-matching accuracy | 33.96 | 8th of 13 | [Source](https://github.com/MemTensor/OmniMemEval/blob/f7476c55/docs/user_memory/results.md#result-summary) · [Evidence](https://github.com/electricsheephq/OmniMemEval/releases/tag/lcmx-v0.28.0-results-20261018) |
| HaluMem | LLM-as-a-judge accuracy | 81.68 | 5th of 13 | [Source](https://github.com/MemTensor/OmniMemEval/blob/f7476c55/docs/user_memory/results.md#result-summary) · [Evidence](https://github.com/electricsheephq/OmniMemEval/releases/tag/lcmx-v0.28.0-results-20261018) |

Run date: 2026-10-18. LCM-X v0.28.0-rc1; executable code identical to v0.28.0.
Configuration: local fastembed BAAI/bge-small-en-v1.5 embeddings; top-k 20.
Harness: [`162640314863db64b5af62138b5b81776516d1ff`](https://github.com/electricsheephq/OmniMemEval/tree/162640314863db64b5af62138b5b81776516d1ff).

## Disclosures

The template requests direct OpenAI dated models; these runs requested those models via OpenRouter, whose answer receipts resolve to `openai/gpt-4.1-mini`. The receipts establish the reported model identifier and token usage, but do not establish the exact dated revision, equivalence to a direct OpenAI run, or that the model differed.

Not yet run: BEAM 10M.
In review: our Agent Memory Leaderboard Cycle 2 entry; the board publishes its own results.
