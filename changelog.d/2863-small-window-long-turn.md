- Fix: on a small context window, compaction now keeps making progress inside one long tool turn: LCM reports to
  the host when a pass made real progress and clears the host's wait-for-usage flags once a real response arrives,
  so the host's per-turn compaction budget re-arms instead of going silent after three passes. A tool result larger
  than the window can hold arrives as a pointer that names the tool and its size; the model reads the original with
  `lcm_expand`. Installs without a tokenizer now size long hex payloads (runs of 65 or more hex characters) at
  2 characters per token instead of 4; the tokenizer path is unchanged (#2863, #2872).
