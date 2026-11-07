- Changed: Known limit, not a regression (#588): on a host that sends no message timestamps, a session that already
  holds a later exact copy of some of its rows stores its replayed history once more on each restart. Each extra copy
  is linked as a possible replay of an earlier stored copy (a copy whose text is only reasoning is stored unlinked), and
  nothing is lost. The fix is planned for v0.30.x.
