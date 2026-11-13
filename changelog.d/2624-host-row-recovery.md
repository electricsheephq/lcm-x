- Fix: on Hermes v0.21.6 to v0.23.1 with rotating compaction, a turn interrupted by a kill or restart no longer loses
  the rows the host kept in the abandoned compaction child (#2624). At the next start of that conversation a
  background worker reads that child with the host's own reader and stores the rows the store lacks, only when the
  child's durable compaction proof covers everything before them. Those turns do not return to the model's view
  (#2626). `LCM_HOST_ROW_RECOVERY_ENABLED=false` turns it off. Named limits: `docs/exactly-once-ingest.md` M10.
