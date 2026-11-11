- New: `adapters/eve`, LCM-X for Eve agents (`eve@1.0.0-rc.6`): the unchanged LCM-X engine runs beside each Eve agent
  and is driven as the Hermes plugin drives it (exactly-once ingest, summary DAG on the agent's own model, the 13
  default `lcm_*` tools as `lcmx__*`). Capture is spooled durably and never fails the Eve turn. Eve keeps its own
  context window and compaction (Eve has no hook to replace them). At turn start, Eve's memory-slot recall adds one
  message with the best store hits for the turn's user text (bounded, never throws; `recall: false` turns it off).
  `lcmx-eve import` copies the text conversation of sessions from before the install into the store (one session
  `import-<id>` each; imported messages carry the import time).
