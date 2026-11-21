- Fix: on a small context window, compaction now keeps making progress inside one long tool turn for the reported
  shape (long hex tool results): LCM reports progress to the host when a pass changes the list and makes it smaller,
  and clears the host's wait-for-usage flags on every main-path response, also when the provider omits usage, so the
  host's post-tool check runs again. Clearing the wait does not re-arm the host's per-turn compaction budget: the host
  re-arms it only on a positive prompt count below the threshold, after a committed pass that reported progress, so a
  usage-less response does not restore the three attempts. Responses the engine classifies by thread as auxiliary do
  not clear the foreground's flags. A pass that lands just under the trigger can re-arm that budget, as the host's own
  compressor does (#2904). For turns of prose tool output the results were mixed, measured on a modelled host: one
  shape (4,000-character steps) went from 7 to 10 requests over 64K, and the others improved (#2879). Tool results too
  large for the window can arrive as pointers that name the tool and its size; the model reads the original with
  `lcm_expand`. Installs without a tokenizer now size long hex payloads at 2 characters per token instead of 4: runs of
  at least 65 hex characters that contain both a digit and an a-f letter, in ASCII text only; the tokenizer path is
  unchanged (#2863, #2872).
