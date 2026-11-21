- Fix: the ingest ledger's fallback for a packed run of ordinals searched windows at a cost that grew with the
  cube of the run's width (minutes on a 25,000-row session). Each search is now bounded to windows of at most 256
  labels; past that the lineage is re-spaced once at the normal step where space permits. Packing can form again
  after many inserts at one spot and is handled the same bounded way. Applying a re-spacing uses set-based
  statements (135 instead of 40,021 for 20,010 labels) (#2840).
