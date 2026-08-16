# LCM-X Repository Instructions

These instructions govern maintainers, contributor agents, and review bots working in
`electricsheephq/lcm-x`. LCM-X is the independent Lossless Context Memory extension for
Hermes. Upstream `stephenschoettler/hermes-lcm` is an evidence and attribution source, not
an authority for writes or automatic merges.

## Non-Negotiable Invariants

- Lossless means lossless. Do not delete, truncate, overwrite, or make durable user context
  unreachable during automatic maintenance, migration, reconciliation, compaction, cleanup,
  or recovery.
- Prefer degraded coverage, a safe no-op, explicit warnings, or a tracked repair over
  destructive recovery when continuity is uncertain.
- Preserve message identity, chronology, conversation/session ownership, summary provenance,
  source coverage, tool-call/result grouping, real user turns, and the active fresh tail.
- Preserve contributor authorship, upstream links, license attribution, and original commits.
- Never commit credentials, tokens, cookies, customer data, live Hermes configuration, or
  local authentication state.
- Do not change customer runtimes, publish a release, write upstream, or announce LCM-X as
  canonical unless the user explicitly authorizes that separate gate.

## Canonical Work Graph

- Use a GitHub issue for accepted bugs, features, design changes, and concrete follow-ups.
- Keep one observable acceptance gate per issue and one issue per PR unless the issue explicitly
  approves a combined change.
- Link upstream reports and PRs as evidence while retaining contributor credit.
- Treat labels, CI, reviewer severity, and unmerged PR claims as evidence inputs—not proof by
  themselves.
- Give every verified review finding one terminal disposition: fixed now, accepted follow-up
  linked to an issue, accepted tradeoff/won't-fix, false/not-applicable, or escalated.

## Bugs And Regressions

- Reproduce a bug on the exact current `main` SHA or identify the mandatory invariant it
  violates before implementation.
- Record the supported path, impact, failing proof, expected behavior, and affected SHA.
- Add the smallest regression that fails on the base and passes on the candidate.
- Do not implement a `needs-repro` report merely because an upstream PR or issue claims a fix.
- Prioritize verified supported-path impact:
  - `P0`: active catastrophic data loss, disclosure, or unusable core path.
  - `P1`: severe integrity, security, crash, lockout, or release-blocking failure.
  - `P2`: significant bounded correctness or resource failure.
  - `P3`: limited bug, feature gap, or moderate maintainability/performance issue.
  - `P4`: documentation, tooling, portability, testing, or cosmetic work.

## Features And Design Changes

- Require an accepted feature/design issue before its PR becomes ready for review or can merge.
- The issue must state the problem, user/operator impact, proposed behavior, alternatives,
  acceptance criteria, non-goals, backward-compatible defaults, and documentation/config
  consequences.
- Identify the Hermes host contract involved: plugin API, `ContextEngine` behavior, lifecycle,
  profile/session identity, tool schema, config, model routing, or no host-contract change.
- Keep existing behavior as the default unless the accepted issue explicitly approves a
  breaking change and its migration path.
- Treat best-practice work as `P3`/`P4` unless current evidence proves a violated invariant.
- Do not add a dependency, schema, background service, persistence surface, or public contract
  when the existing path can satisfy the accepted behavior.

## Hermes Compatibility

- Test against the repository's supported Python versions and the Hermes `ContextEngine`
  import/contract exercised by CI.
- Keep profile and session state isolated; never infer one Hermes profile from another.
- When changing configuration, update the runtime loader, user documentation, bundled
  `skills/hermes-lcm/references/configuration.md`, examples, and regression tests together.
- When changing plugin tools or lifecycle behavior, update the bundled Hermes skill/reference
  material and test host-facing failure and fallback paths.
- State any minimum Hermes capability explicitly. Do not silently depend on an unreleased or
  unverified Hermes host behavior.

## Validation

Before publication, run the narrowest focused regression and the checks relevant to the changed
surface. GitHub Actions is authoritative for the full supported matrix.

Default repository checks:

```bash
pytest tests/test_lcm_core.py tests/test_lcm_engine.py tests/test_packaging_install.py -q
pytest -q
bash -lc 'ulimit -n 1024 && python -m pytest tests/ -q'
ruff check .
python -m compileall -q .
python -m py_compile scripts/import_lossless_claw.py
bash -n scripts/install.sh scripts/update.sh
git diff --check
```

Run `actionlint` when workflows change. Record exact commands and results in the PR.

## Automation Boundary

- Treat AI and bot output as proposal and evidence by default.
- Deterministic tooling must re-fetch the live issue, PR head, checks, reviews, threads, and
  authorization immediately before an authorized write.
- Model output alone cannot close, label, assign, push, approve, or merge.
- Automated repair is opt-in and limited to the exact accepted issue and current gate.
- Security and data-integrity code changes and public disclosure retain non-author human
  code-owner approval. Classification alone does not elevate routine reversible issue metadata
  to that stronger gate.
- Use `.agents/skills/triage-backlog/SKILL.md` read-only unless a maintainer explicitly
  authorizes one exact mutation; never use it for an automatic backlog sweep.
- Invoking a skill never creates write authority. Routine reversible issue metadata needs one
  exact maintainer authorization and live read-back; public sensitive disclosure and terminal
  lifecycle actions retain the stronger owner and lifecycle gates in `triage-backlog`.
- Preserve open executable upstream work as an active continuation. Do not silently turn it into
  an archive-only record, supersede it, or close it while its continuation state is ambiguous.

## Review And Merge

- Use `.agents/skills/land-pr/SKILL.md` when deciding readiness or landing a PR.
- Pin the PR `headRefOid`; checks and semantic review must cover that head or an explicitly
  bounded delta.
- Require one independent semantic review for changes involving data integrity, security,
  migrations, compaction, lifecycle/session identity, persistence, or a Hermes host contract.
- Do not merge with failing/pending required checks, unresolved actionable threads, a changed
  head, missing issue acceptance, or unowned product/security decisions.
- Never push directly to `main`, bypass the ruleset, use auto-merge, or force-push/delete
  `main`.
- Use merge commits for PRs so contributor and upstream commits remain intact. Do not squash
  or rebase-merge into `main`.
- Merge deterministically with the pinned head:

```bash
gh pr merge <PR> --merge --match-head-commit <HEAD_SHA>
```

- After merging, verify the merge commit, linked issue state, final `main` SHA, and required
  post-merge checks. Thank external contributors and preserve uncertain issue reports rather
  than closing them speculatively.

## Maintainer And Bot Roles

- `@100yenadmin` and `@Tosko4` are code owners. The PR author cannot satisfy their own required
  approval.
- Bots and agents may triage, reproduce, implement, test, review, and prepare merge evidence.
- Bots and agents may not bypass checks, approvals, code-owner review, review-thread resolution,
  issue acceptance, or a product/security owner decision.
- Maintainers own feature acceptance, priority, compatibility decisions, terminal dispositions,
  and releases.
