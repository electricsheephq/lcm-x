- Fix: leaf selection charges a glued summary carrier (a host list with no leading system row) at the row the
  summariser reads, not at its full size, so a carrier-led leaf no longer shrinks as the summary area grows and
  compaction keeps making progress instead of stalling into the survival fit (#2901).
