## Summary

-

## Linked work and change class

- Accepted issue: Closes/Refs # (required for bugs, features, design changes, and concrete follow-ups; use N/A only for other change classes)
- Combined scope: authorized by the linked issue / N/A
- Change class: bug fix / feature / design / docs / test / maintenance
- Base SHA:
- Head SHA (`headRefOid`):

## Why

-

## Root cause and best-fix assessment

- Root cause:
- Mandatory invariant and owning component:
- Caller, callee, and sibling paths checked:
- Alternative fix location considered:
- Assessment: best fix / acceptable mitigation / wrong layer

## Hermes compatibility and lossless-data impact

- Hermes host contract changed: none / plugin API / ContextEngine / lifecycle / profile-session identity / tools / config / model routing
- Minimum Hermes capability:
- Backward-compatible default and fallback:
- Profile/session ownership and isolation:
- Durable context, identity, chronology, provenance, source coverage, tool grouping, real user turns, and fresh-tail impact:
- Documentation and configuration consequences:
- Upstream links and contributor attribution:

## Real-behavior proof

- Claim being proved:
- Hermes/LCM surface and supported scenario or fixture:
- Exact command and environment:
- Observed result:
- Artifact or redacted trace:
- Proof boundary—what this does not prove:

## Validation

- [ ] Focused validation: `<command>` -> `<result>`
- [ ] Bug regression fails on the recorded base and passes on this head, or the violated mandatory invariant is named.
- [ ] Feature/design issue is maintainer-accepted and its acceptance criteria are covered.
- [ ] Default validation:
  - [ ] `ruff check .`
  - [ ] `pytest tests/test_lcm_core.py tests/test_lcm_engine.py tests/test_packaging_install.py -q`
  - [ ] `pytest -q`
  - [ ] `bash -lc 'ulimit -n 1024 && python -m pytest tests/ -q'`
  - [ ] `python -m compileall -q .`
  - [ ] `python -m py_compile scripts/import_lossless_claw.py`
  - [ ] `bash -n scripts/install.sh scripts/update.sh`
  - [ ] `git diff --check`
- [ ] Workflow validation, if workflows changed: `actionlint`

## Release-note impact

- User-facing release note required: yes / no
- Suggested curated note or reason none is needed:

## Review and merge gate

- [ ] Current `headRefOid` is recorded.
- [ ] Required exact-head CI is green.
- [ ] One non-author code owner approved the current head.
- [ ] Data-integrity/security/migration/compaction/persistence/profile-session identity/lifecycle/Hermes-contract risk has independent semantic review when applicable.
- [ ] All actionable review threads have terminal dispositions and are resolved.
- [ ] The accepted issue, PR head, checks, reviews, threads, and authorization were deterministically re-fetched immediately before any authorized GitHub write.
- [ ] Merge method is merge commit; do not squash or rebase.

## Non-goals and notes

-
