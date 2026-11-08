- Fix: Recognise Hermes assistant folds anchored to stored parts and unstamped todo snapshots already
  held in the same session as replays, including cursor-zero persisted batches. Preserve original
  stored bytes and stamped user snapshots; ambiguous and over-budget shapes remain new.
  Require stamped-prompt evidence before treating an unstamped todo block as scaffold. Preserve
  later in-place changes of replayed host objects and proven rewinds when optional reads fail.
