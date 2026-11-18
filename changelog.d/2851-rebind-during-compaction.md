- Fix: a session switch on the same engine while a compaction pass waits for its summary no longer makes the
  new session skip rows. The pass now notices the switch at each of its exits, drops its stale result, returns the
  caller's own list unchanged, and flags the session the engine now serves to reconcile its ingest cursor at its
  next ingest, which repairs a cursor the old pass moved. That repairs the cursor only: a condensed summary node
  the old pass wrote under the new session can remain, and is tracked in #2865. (#2851)
