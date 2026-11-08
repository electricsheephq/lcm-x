- Changed: v0.30.0's named limits now also cover these shapes, each pinned by a strict-xfail test where it
  reproduces engine-direct. Loss: #2259, a fresh byte-identical tool step and the user row after it, skipped after
  a cold start. #2191, half of a re-stamped tool pair stored without its partner. #2253, a busy session-end tail
  missing after a gateway stop until resume. #2042, Desktop-source sessions with no rows, cause unknown. Surplus:
  #2118, a re-formed summary carrier stored after a restart. #2488, a todo snapshot merged with the next prompt
  after a resume. #2134, a hidden-only or recovery pass after a restart, not reproduced. #2197's duplicate shape.
  #2490, a carried turn after a failed predecessor-snapshot read. #2431 item 4, a retained prompt shown with a
  leading space. #2436, a cursor-0 tool-pair replay on a store with blank conversation ids.
  See `docs/exactly-once-ingest.md` §5.
