- Fix: leaf selection charges a glued summary carrier (a host list with no leading system row) at the row the
  summariser reads, not at its full size, so a carrier-led leaf no longer shrinks as the summary area grows, and leaf
  selection no longer stalls on the glued carrier (#2901). Survival fits with no pointer can still occur on such lists
  through throughput (#2896).
