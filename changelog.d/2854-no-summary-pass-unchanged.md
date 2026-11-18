- Fix: a compaction pass that publishes no summary and does no other real work now returns the caller's list
  unchanged, so the host has nothing to commit. That removes one cause of a full prompt-cache rewrite on the next
  turn; the effect on the provider's cache is measured separately, not by this change's tests. Two changes: an
  assistant row with empty content that carries API-side content is kept (it was dropped on every exit that
  made no leaf), and a pass with no leaf keeps the head the host replayed when the store has nothing newer to
  show (it was re-rendered when a new user row had entered the recent tail) (#2854).
