- Fix: automatic compaction on windows above 480,000 tokens with neither foreground setting configured lets a
  pass that has already stored a summary keep summarising while the whole prompt is above the low-water mark: the
  depth target when one is set, else the trigger less the larger of two leaf chunks and a tenth of the window. A
  call the 60 s soft target or the 120 s bound would refuse may start if it is expected to end within 210 seconds
  of the pass start, less the finalize reserve (at least 20 s). Such a call's timeout stays under 0.85 of the
  configured summary timeout, so a slow call is cut as a budget stop, never classified as a hang, and its time is
  not added to the call estimates. A provider failure on such a call counts nothing against any route, tries no
  fallback route, and ends the catch-up for the rest of that pass. The first call of a pass, explicit settings, forced and overflow passes, and
  windows up to 480,000 tokens are unchanged.
