- Fix: A message holding a lone UTF-16 surrogate no longer makes ingest fail.
  Each surrogate code point is stored as U+FFFD, the form the host itself uses,
  so the raw message, the host's rewritten copy and a row stored by an earlier
  version compare equal and are stored exactly once. When legacy tool-call JSON
  decodes an escaped high/low surrogate pair to one astral character, the row is
  stored once more on its first replay after upgrade. Colliding dictionary keys
  get a deterministic suffix and keep every value.
