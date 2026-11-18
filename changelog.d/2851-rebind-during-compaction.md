- Fix: a session switch on the same engine while a compaction pass waits for its summary no longer makes the
  new session skip rows. The pass now notices the switch at each of its exits, drops its stale result, returns the
  caller's own list unchanged, and leaves the session the engine now serves flagged to reconcile its ingest cursor
  at its next ingest, so a write of the old pass that slipped in is repaired there (#2851; what remains: #2865).
