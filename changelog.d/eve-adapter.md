- New: `adapters/eve`, LCM-X for Eve agents (`eve@1.0.0-rc.6`): the unchanged LCM-X engine runs beside each Eve
  agent and is driven as the Hermes plugin drives it (exactly-once ingest, summary DAG on the agent's own model,
  the 13 default `lcm_*` tools as `lcmx__*`). Capture is spooled durably and never fails the Eve turn. Eve keeps its
  own context window and compaction (Eve has no hook to replace them).
