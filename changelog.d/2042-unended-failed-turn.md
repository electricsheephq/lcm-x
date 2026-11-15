- Fix (opt-in, off by default): on Hermes v0.23.1, a Desktop session whose turn failed or was stopped before its reply
  finished, and that never got a later completed turn or a session end, no longer leaves its rows out of LCM-X
  (#2042). At a later session start on the same profile, a background job stores such a session from the host's own
  transcript when its last row is the host's failed-turn notice, it is 10 minutes to 7 days old, and LCM-X holds none
  of its rows. Forward only: only failures after the profile's first catch-up job are stored; failures before it,
  including those after the upgrade while the setting is off, are not (#2652). A first turn stopped during a tool
  round is #2653. Off by default because of known issues in #2624 and #2042; `LCM_HOST_ROW_RECOVERY_ENABLED=true`
  enables recovery, catch-up and the ingest guard together. Named limits: `docs/exactly-once-ingest.md`.
