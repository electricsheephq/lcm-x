- Changed: Known limit, not a regression (#588): on a host that sends no message timestamps, a session that already
  holds a later exact copy of some of its rows stores its replayed history once more on each restart. Every extra copy
  is linked to its original as a possible replay, and nothing is lost. The fix is planned for v0.30.x.
