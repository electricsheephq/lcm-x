- Fix: the tool-pair repair keeps every real tool result of a multi-call assistant row, in call order: a result
  that arrives before an earlier call's result, or after a call with no result, is no longer replaced by a stub (#2894).
