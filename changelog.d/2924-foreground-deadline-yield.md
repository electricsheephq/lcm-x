- Eligible automatic compaction passes return the original host context with status `deadline_yield` when ingest exhausts
  the foreground budget. Ingest stays committed, and automatic passes at the survival ceiling still fit. The budget reserves
  time before the effective Hermes idle limit (#2924).
