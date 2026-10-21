# #1586 validation receipt

- Claim class: advisory source/test and offline mechanism proof.
- T1 base archive: `assert 4000 <= 64` failed; ordinary/retry counts were 4000/2000. Archive contains the base and only the new regression file.
- Final focused command: `python -m pytest tests/test_issue_1586_gap_discovery.py tests/test_issue_1536_chunk_revision.py tests/behaviour/test_backfill_trace.py 'tests/test_issue_1127_pre_leaf_diagnostic.py::test_a_held_write_lock_is_attributed_to_the_slow_statement[True]' -q -p no:cacheprovider`.
- Observed: `166 passed in 23.23s`.
- Full CI-style command, with a lane-local scratch basetemp: `ulimit -n 1024; python -m pytest tests/ -q -m "not latency" -n 4 --ignore=tests/test_embedding_privacy.py -p no:cacheprovider`.
- Observed: `15846 passed, 137 skipped, 64 xfailed, 23 warnings in 324.90s (0:05:24)`.
- Initial full run had two incremental trace mismatches and one lock-timing failure. Only the two incremental golden cases were updated to match the approved lazy SQL/checkpoint protocol; the other 78 stayed identical. The lock-timing test passed one narrow rerun and the final full run.
- Ten isolated source mutants fail the named regression assertions: step-over budget reduction, pending order reversal, pending replay truncation, pre-yield ceiling suppression, conservative remaining suppression, status plus removal, zero-prefix progress suppression, boundary omission, empty-pending retirement suppression, and ceiling extension across an unread interval. No mutation was committed.
- An additional instrument check passed 629 tests but exposed a missing host-path environment variable and one cold-prepare instrumentation failure. The environment-specific test passed with its required variable; the remaining cold-prepare failure reproduces on the untouched base. No out-of-scope product change was made.
- Pinned installed `ruff 0.15.13`: `ruff check .` → `All checks passed!`; `git diff --check` → exit 0.
- Product delta: command.py +90/-54 = 144 lines. New regression file: 502 lines. Trace assertion +9/-2; generated fixture +11/-12, two keys only.

Not proven: exact-head remote CI, independent review, release throughput, runtime/customer safety, default flip or release readiness.
