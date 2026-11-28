- Fix: Enforce the 64,000-character cap on the whole snippets recall response,
  including the envelope, provenance and optional markers, with windows off or on,
  skipped reads and extras. Cut an oversized no-hit envelope's query echo to its
  first 4,096 characters and report `query_truncated`; drop full hits from the
  end in rank order with `response_cap_truncated`. Responses that already fit
  and stored rows are unchanged.
- Fix: Use the default integer argument value when conversion overflows,
  including non-finite per-call recall limits.
