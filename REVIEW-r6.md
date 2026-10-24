# Independent r6 delta review — PASS

Reviewer: astra_reviewer, gpt-6-astra/high; independent, no authorship/advice conflict. Reviewed delta: `c8de1a45ec8bdea027a50c8713eee85528de7165..9f2e8134608ae3bf8d4149be77fe3451635c7418`. Exact head unchanged, worktree clean.

- `scripts/install.sh:9–18`: profile-shaped home with unset or matching HERMES_PROFILE selects root + work. Conflicting p1 exits before mutation, naming both inputs and both valid spellings.
- `scripts/install.sh:26–44,108–152`: normalized selection feeds links, legacy/config checks, companion preservation guard, native calls and printed commands. Canonical-name refusal, root/sticky/custom behavior, separate legacy preservation and optional-failure handling remain intact.
- Both supplied resolver snapshots implement the same relevant root/profile resolution; their SHA-256 hashes match source-identity.json.
- `tests/test_issue_1186_dependency_companion.py:182–210` covers all three new spellings, existing-companion preservation and command/target agreement. `tests/test_issue_1186_host_dependencies.py:117–134` applies extracted pinned resolution to those and the retained root/profile cases.
- `dependency_qualification.py:271–277` extracts and executes the complete printed command through bash -c, retaining its HERMES_HOME assignment and -p; the regression assertion at `test_issue_1186_host_dependencies.py:89–90` checks exact execution and recording.
- All five named documentation sections describe profile-shaped homes and conflict refusal.

Verified findings: none. No verified blocker within the inspected S1-C.6 delta.

Proof boundary: read-only semantic review; no tests executed. Supplied logs show eight targeted RED failures and each supplied GREEN run reports 49 passed, 15 skipped. Resolver source is real; validation/liveness and native operations remain synthetic. This does not establish CI, release, runtime or customer readiness. Unchanged PR surfaces were not re-reviewed; prior independent coverage retained.
