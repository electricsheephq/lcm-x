- Fix: a user row carried behind the summary head no longer leaves the list uncounted. When a compaction pass
  summarised only rows the list does not carry, the assembly wrapped the incoming carrier (the summary with that
  row behind it) inside the new one, so the head held two summaries; a survival-fit cut of that head then took
  the row for generated text and dropped it without counting or naming it. The assembly now carries only the
  real row behind the new summary, which keeps its single proven host uid and its stored-row mapping, and the
  survival fit splits a nested carrier down to its real row so the row is counted and named if it is cut (#2895).
  Assembly budgets the recovered row before selecting the summary and is byte-identical to main off the carrier path.
