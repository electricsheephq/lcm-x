- Fix: a delegated child (sub-agent) that is interrupted before it finishes no longer loses its whole transcript
  (#2482). Before, its rows were stored only when its turn ended, at its session end or before its own compaction,
  so a host restart, kill or exit while it ran left it with no rows in LCM-X. Its rows are now stored each time the
  host hands over its list: once at the start of each turn and before every model call. Rows a child stores this way
  are not stored again when it finishes. Known issue (v0.30.0): a child's rows after its last completed store are lost
  if the process ends first: while its tools run (that round's tool-call row and finished results stay only in the
  host), or while a store waits more than 5 s on a busy database at shutdown, which logs a warning (#2483).
