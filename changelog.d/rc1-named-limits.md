- Changed: v0.30.0's named limits now also cover these shapes, each pinned where it reproduces engine-direct (a
  strict-xfail test, or a passing test of today's behaviour where §5 says so). Loss: #2259, a fresh byte-identical tool step and the user row after it, skipped after
  a cold start. #2191, half of a re-stamped tool pair stored without its partner. #2253, a busy session-end tail
  missing after a gateway stop until resume. #2042, Desktop-source sessions with no rows, cause unknown. Surplus:
  #2118, a re-formed summary carrier stored after a restart. #2488, a todo snapshot merged with the next prompt
  after a resume. #2134, a hidden-only or recovery pass, then a host commit, then a restart, not reproduced. #2197's duplicate shape.
  #2490, a carried turn after a failed predecessor-snapshot read. #2431 item 4, a retained prompt shown with a
  leading space. #2436, a cursor-0 tool-pair replay on a store with blank conversation ids. #2491, a compression
  child's reply after a host insert before its carried tail.
  See `docs/exactly-once-ingest.md` §5.
