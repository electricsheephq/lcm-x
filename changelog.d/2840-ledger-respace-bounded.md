- Fix: placing a row between existing rows of a long session no longer stalls ingest. The ingest ledger's
  fallback for a packed run of ordinals searched windows at a cost that grew with the cube of the run's width
  (minutes on a 25,000-row session, past the host's watchdog). The search is now bounded, and past the bound the
  whole lineage is re-spaced once at the normal step, which also stops the packing from recurring. Applying a
  re-spacing uses set-based statements (135 instead of 40,021 for 20,010 labels) (#2840).
