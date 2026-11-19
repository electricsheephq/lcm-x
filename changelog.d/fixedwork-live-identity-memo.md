- Changed: a live row's replay identity is now computed once and remembered while the row reads no stored,
  on-disk or DAG state (no summary, projection, pruned, out-of-band, persisted-output or externalized marker
  in its content or tool calls). Every identity equals the one computed before; the memo is dropped when the
  config, session, conversation or Hermes home changes and is bounded to 65,536 entries and 32 MB. An
  ordinary turn now recomputes identities only for its new rows; no timing claim. Stored rows are not
  remembered yet. `scripts/identity_row_classes.py` counts which memo class a list of messages falls in
  (counts only).
