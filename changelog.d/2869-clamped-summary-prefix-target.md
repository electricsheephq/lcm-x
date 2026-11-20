- Fix: above 480,000-token windows an automatic threshold sweep keeps condensing its summary frontier only while the
  frontier is over today's summary-prefix target AND either over 5% of the window (capped at a quarter of the
  threshold) or holding more than 80 live summaries. The rule is checked at every condensation step.
  - It never condenses where today's target would not.
  - A normal stop leaves at most 80 live summaries, or a frontier at or under today's target.
  - Because one sweep adds at most 12 leaves, no level grows past 92 of the 100 uncondensed summaries the view shows.
  - Smaller windows, forced, overflow and recovery passes, explicit targets and background leaf preparation are
    unchanged (#2869).
