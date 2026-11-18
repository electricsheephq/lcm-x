- Fix: a session switch on the same engine while a compaction pass waits for its summary no longer makes the
  new session skip rows. The pass now notices the switch at each of its exits, writes none of the old session's
  state into the new one (the ingest cursor among it), and returns the caller's own list unchanged (#2851).
