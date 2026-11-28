- Changed: a stored row's replay identity is now remembered under the same conditions as a live row's: no
  summary, projection, pruned, out-of-band, persisted-output, externalized or ingest-payload marker in its
  content or in any string of its tool calls, dict keys included. Every identity equals the one computed
  before. The key holds the row's identity inputs by value, so a content update, a delete and re-insert or a
  tool-call rewrite computes afresh. A compaction pass now computes each such stored identity about once,
  where it used to compute it many times; no timing claim.
