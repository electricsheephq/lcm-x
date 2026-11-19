- Perf: looking up an externalized tool result by its call id now reads a small index of the payload
  folder; an unchanged folder costs file stats only, where every lookup used to parse every file (#2856).
