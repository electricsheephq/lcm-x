- Fix: on a Hermes gateway restart, full-session replay can match a tool result whose name is missing on either side (#2886).
  The result’s unique owning call supplies the missing name for comparison without changing stored rows.
