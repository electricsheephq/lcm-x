- Fix: above 480,000-token windows the default summary-prefix target of a threshold sweep follows the window (5% of
  it), clamped to 80 of the session's mean depth-0 summaries so a level stays under the 100 uncondensed summaries the
  view shows, and never below today's default. Smaller windows, forced and overflow passes and explicit settings are
  unchanged (#2869).
