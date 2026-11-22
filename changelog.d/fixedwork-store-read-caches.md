- Changed: the token-count cache now holds a whole walk of a long list (65,536 entries and 32 MB of keys;
  strings over 1,024 characters are keyed by a 128-bit digest and never held), and a compaction pass pages
  in the store-id map's candidate rows once per store write generation instead of on every rebuild. The
  rows are remembered only outside a transaction and the ingest prelock, handed out as copies, and dropped
  by any write on this or another connection. Every count and map equals the one computed before; no
  timing claim.
