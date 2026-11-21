- Fix: automatic compaction on windows above 480,000 tokens lets the default summary prefix grow to 5% of the
  window, capped at a quarter of the compaction threshold, when that raises the existing target. Explicit
  targets still take precedence. (#2869)
