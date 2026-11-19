- Fix: overlapping managed compaction returns the caller's message list unchanged while the active pass
  finishes. The host sees a temporary cooldown, and turn telemetry skips a busy store lock.
