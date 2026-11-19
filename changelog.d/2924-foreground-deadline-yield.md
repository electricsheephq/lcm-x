- When ingest exhausts the foreground budget, compaction keeps the committed messages and returns the original
  host context with status `deadline_yield`. The budget reserves time before a configured Hermes idle limit (#2924).
