- Fix: A replayed copy whose content is only reasoning blocks is now linked to its earlier stored copy when the
  bytes match exactly, so no extra copy from #588's unstamped restart shape is stored without a link.
