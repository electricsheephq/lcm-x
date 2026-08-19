# Dispatch packet — consolidation branch (fires after PR #169 merges)

**Branch:** `release/r2-consolidated` from `bench/w3b-on-wave1` post-#169-merge.
**Three items, one PR to bench/w3b-on-wave1:**

1. **Cherry-pick #434 verbatim** (upstream stephenschoettler/hermes-lcm PR #434, 3 files: config.py,
   embedding_provider.py, tests/test_embedding_provider.py — fields embedding_query_spend_max_calls=600 /
   _window_seconds=60.0 / _backoff_seconds=60.0 + env specs). Its fields are verified ABSENT from wave-1.
   Resolve conflicts minimally; keep the original tests.
2. **#164(a) store_id fix:** wave-1 retrieval can emit a `kind:"summary"` hit with no `store_id` (~1/2,500
   hits; 8/500 questions fail-closed in F32). Locate the summary-hit construction in tools.py (grep
   `"summary"` kind paths), populate store_id from the summary's source row. Regression test: a summary hit
   from a store whose summary rows previously lacked the field carries store_id; render through the pinned
   evidence-cards validator (the F33 oracle's validate_render.ts pattern) → no fail-close.
3. **Version/CHANGELOG touch** per repo convention for the R2 train.

Acceptance: targeted tests green (single-process); the F32-run's 8 fail-closed qids re-rendered through the
validator with a synthetic summary-hit fixture → 0 fail-close; no other behavior change (`git diff` review).
Then: fork mono-PR (this + #169 content) opens for the multi-bot rounds; Phase 1B re-run points its venv at
THIS branch's checkout (note: phase1a.py's ProbeBridge imports hermes-lcm from /Volumes/LEXAR/hermes-work/
hermes-lcm/.venv-fastembed — the re-run needs that venv reinstalled from release/r2-consolidated, or an
equivalent venv; record which in the F34 provenance).
