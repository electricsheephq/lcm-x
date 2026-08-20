Closing in favour of the consolidated wave-1 PR #436. On inspection the fix's exact fields
(`embedding_query_spend_max_calls` / `_window_seconds` / `_backoff_seconds`, generous defaults, env-mapped)
turned out to be **already present in the wave-1 base** (verified at e99f342 and after — patch-identical),
so every wave-1 benchmark result published in `bench/` was measured WITH this fix in place. PR #170 carries
provenance-marker commits acknowledging the original authorship. Thanks — the fix was right, it shipped
earlier than any of us tracked, and #436 is where it reaches upstream.
