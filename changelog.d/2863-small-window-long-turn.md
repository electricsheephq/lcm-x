- Fix: on a small context window, compaction now keeps making progress inside one long tool turn for the reported
  shape (long hex tool results): LCM reports to the host when a pass made real progress and clears the host's
  wait-for-usage flags on every response, also when the provider omits usage, so the host's per-turn compaction
  budget re-arms instead of going silent after three passes. For turns of prose tool output this is a measured
  improvement, not a cure (#2879). Tool results too large for the window can arrive as pointers that name the tool
  and its size; the model reads the original with `lcm_expand`. Installs without a tokenizer now size long hex
  payloads (runs of 65 or more hex characters) at 2 characters per token instead of 4; the tokenizer path is
  unchanged (#2863, #2872).
