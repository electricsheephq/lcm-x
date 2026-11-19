- When ingest exhausts the foreground budget on an automatic pass, compaction keeps the committed messages and returns the
  original host context with status `deadline_yield`. The budget reserves time before the effective Hermes idle limit (#2924).
