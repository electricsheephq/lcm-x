With `LCM_DEPTH_TARGET_RATIO` active, automatic compaction can continue toward
the depth target after its first stored leaf or condensed node, within a
bounded catch-up budget. Explicit foreground settings of 60 s soft and 120 s
hard count as defaults. Depth sweeps also credit summary-block savings before
the first leaf and after each post-leaf condensation group.

Refs #2878, refs #2848
