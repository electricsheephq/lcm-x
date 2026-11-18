- Fix: A message holding a lone UTF-16 surrogate no longer makes ingest fail.
  Each surrogate code point is stored as U+FFFD, the form the host itself uses,
  so the raw message, the host's rewritten copy and a row stored by an earlier
  version compare equal and are stored exactly once. Colliding dictionary keys
  get a deterministic suffix and keep every value.
