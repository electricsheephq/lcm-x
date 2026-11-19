- Fix: above 480,000-token windows the default summary-prefix target of a threshold sweep follows the window (5% of
  it), clamped to 80 live summary-frontier nodes of their mean stored size, the unit the sweep compares with its
  target. At sweep start at most 80 nodes are live unless today's default binds, so a level stays under the 100
  uncondensed summaries the view shows. The target never falls below today's default. Smaller windows, forced and
  overflow passes, explicit settings and background leaf preparation keep today's target (#2869).
