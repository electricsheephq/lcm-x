# Issue #1828 final-refit acceptance receipt

Run: `issue-1828-refit-20261023-r2`. Claim class: advisory, delivery correctness and local proof.
Measured code: `07510c16558eb3acca0dec0b274a82fa20b99f8d`.
Base: `e5a239ee1d02acf2068150583b7eff2576635d09`.

| Gate | Receipt |
| --- | --- |
| Two-group refit | cap 3240: base [] -> head [2] |
| Three-session refit | caps 3370 and 3380: base [] -> head [2, 3] |
| RED -> GREEN | Both new parametrized cases fail at base; 49 affected tests pass at head |
| CAP-SCAN | 1502 cells: equal 1496; head_superset 6; head_subset 0; incomparable 0 |
| Maximality | 0 skipped-group additions fit the final head envelope |
| COST, no skips | 300 runs per arm: base median 4.7885 ms; head 4.7802085 ms; threshold 5.31735 ms: PASS |
| COST, skips | 300 runs per arm: base median 0.224083 ms; head 0.2966875 ms; 1 skipped group, 1 retry check |
| Ruff | ruff 0.15.13; `ruff check .`: `All checks passed!` |

The cost fixture has 25 ranked sessions with 300-852 character source messages,
real in-memory store/DAG readers, whole JSON/delta accounting, fallback tokens,
and a 64000-character cap. Arms alternate base/head and head/base after warmup;
the timer covers `deliver_session` only. No reader, model or semantic judge runs.
Raw timing samples and every cap's membership are in `summary.json`.
Single-machine synthetic measurements have no cross-machine performance claim.

The cap-scan test independently adds each undelivered admission-budget skip to
the final envelope, updates session and delta metadata, and checks JSON and
token fixed points, removing optional hints as production does. Groups removed
by refit are excluded. Seen refs and deadline stops remain excluded from retry.
Existing session goldens were unchanged; only the six listed cap cells gain groups.

Commands:
```sh
python -m pytest tests/test_issue_1828_refit.py -q
python -m pytest tests/test_issue_1828_refit.py tests/test_recall_session_delivery.py -q
ruff --version
ruff check .
git diff --check
```

Observed final lines:
```text
2 failed in 0.88s
49 passed, 12 warnings in 10.48s
ruff 0.15.13
All checks passed!
```

Re-score: the counts derive by comparing each `base` and `head` set in the
two fixtures' `rows` in `summary.json`; cost is the median of each arm's
`samples_ms`, with no-skips threshold `base * 1.10 + 0.05` ms.

Not proven: board scores, retrieval/answer quality, production runtime,
release or customer behavior. Full suite and supported-version matrix are
delegated to the PR's CI workflow; this local receipt alone does not prove them.
